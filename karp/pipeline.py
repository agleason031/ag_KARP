"""
The reduction of one object (one OBJNAME with one instrument setup): per-frame
calibration, tracing, extraction and wavelength calibration, then combining the
frames, and the optional continuum normalization and flux calibration.

Output layout, in <output_dir>/<object>/:
    <obj>_<n>_extracted.fits        each frame's spectrum (electrons)
    <obj>_combined.fits / .png      the combined spectrum (electrons)
    <obj>_normalized.fits / .png    continuum-normalized (optional)
    <obj>_fluxed.fits / .png        flux-calibrated (optional, needs a standard)
    KARP_log.txt, KARP_run_info.txt
    calib/                          master bias, final flat, sensitivity function
    diagnostics/                    plots and tables for checking the reduction
    diagnostics/frame_<n>/          per-frame diagnostics
"""
import logging
import os
from dataclasses import dataclass, field

import ccdproc
import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.stats import mad_std
from astropy.table import Table

from . import detector, fluxcal, wavecal
from .calibration import load_arc, load_frame
from .combine import Combined, combine_frames, snr_in
from .extract import extract, variance_image
from .normalize import fit_continuum
from .observation import exposure_info, heliocentric_correction
from .regions import mask_regions
from .trace import fit_trace
from .wavecal import C_KMS

log = logging.getLogger("karp")


class OutputPaths:
    """Where every output file of one object goes (see the module docstring)."""

    def __init__(self, output_dir, out_name, objid):
        self.root = os.path.join(output_dir, out_name)
        self.objid = objid
        self.calib = os.path.join(self.root, "calib")
        self.diagnostics = os.path.join(self.root, "diagnostics")
        self.log = os.path.join(self.root, "KARP_log.txt")
        self.run_info = os.path.join(self.root, "KARP_run_info.txt")
        self.master_bias = os.path.join(self.calib, f"{objid}_master_bias.fits")
        self.final_flat = os.path.join(self.calib, f"{objid}_final_flat.fits")
        self.sensfunc = os.path.join(self.calib, f"{objid}_sensfunc.fits")

    def make_dirs(self):
        for d in (self.root, self.calib, self.diagnostics):
            os.makedirs(d, exist_ok=True)

    def product(self, name):
        """A final product: <obj>_<name> in the object's folder (e.g. product("combined.fits"))."""
        return os.path.join(self.root, f"{self.objid}_{name}")

    def extracted(self, frame):
        return self.product(f"{frame}_extracted.fits")

    def diagnostic(self, name):
        return os.path.join(self.diagnostics, name)

    def frame_dir(self, frame):
        return os.path.join(self.diagnostics, f"frame_{frame}")

    def frame_file(self, frame, name):
        return os.path.join(self.frame_dir(frame), name)


@dataclass
class FrameSpectrum:
    """One science frame's extracted, wavelength-calibrated spectrum and how it was made."""
    frame: int
    wave: np.ndarray # A, air, heliocentric
    flux: np.ndarray # electrons per row
    sigma: np.ndarray
    sky: np.ndarray # sky electrons under the aperture
    rows: np.ndarray # detector row of each pixel
    v_helio: float # km/s, applied to wave
    solution: wavecal.WaveSolution
    trace_columns: np.ndarray # trace, whole columns, every row
    flexure: dict = field(default_factory=dict) # {sky line: offset in A}
    flexure_offset: float = 0.0 # A subtracted from the arc solution


@dataclass
class Normalized:
    wave: np.ndarray
    flux: np.ndarray
    sigma: np.ndarray
    mask: np.ndarray
    continuum: np.ndarray
    used: np.ndarray # pixels used in the continuum fit


class ObjectReduction:
    def __init__(self, cfg, group, out, plots):
        """
        cfg: full config (config.load_config); group: discover.Group with this object's
        frames; out: OutputPaths; plots: plots.Plotter.
        """
        self.cfg = cfg
        self.group = group
        self.objid = group.objid
        self.out = out
        self.plots = plots
        self.verbose = cfg["output"]["verbose"]
        self.det = detector.settings(cfg["detector"]) # layout itself is read from each frame's header
        self.target = cfg["target"]

        # Sizes in the config are physical (arcsec across the slit, unbinned rows along the
        # dispersion); convert them to pixels of this group's binning
        self.binning = detector.setup_binning(group.setup)
        binx, biny = self.binning
        ext = cfg["extraction"]
        arcsec_per_col = cfg["detector"]["pixel_scale"] * binx
        # Aperture: 2*half+1 columns (a width w in pixels uses (w-1)/2 per side, rounded down)
        self.aperture_half = int(np.round(ext["aperture_arcsec"] / arcsec_per_col - 1) / 2)
        self.buffer_cols = int(round(ext["buffer_arcsec"] / arcsec_per_col)) # per side
        self.background_cols = int(round(ext["background_arcsec"] / arcsec_per_col)) # per side
        self.profile_window = detector.to_binned(ext["profile_window"], biny, odd=True, minimum=3)
        trace_col = cfg["trace"]["col"] # unbinned CCD column
        self.trace_start = None if trace_col is None else int(trace_col // binx)

        with fits.open(group.science[min(group.science)]) as hdul:
            log.info(detector.describe(detector.frame_geometry(hdul[0].header, self.det)))
        log.info(f"Binning {binx}x{biny} ({arcsec_per_col:.3f}\"/column): aperture {2 * self.aperture_half + 1} "
                 f"columns, buffer {self.buffer_cols}, background {self.background_cols} per side; profile window "
                 f"{self.profile_window} rows")

        # Wavelength calibration: arc image, lamps, reference solution; [wavecal] row
        # sizes (unbinned) converted to this binning
        self.arc_image, header_lamps = load_arc(group.arcs, self.det)
        wc = cfg["wavecal"]
        self.wcfg = dict(wc, search=wc["search"] / biny, match_tol=wc["match_tol"] / biny)
        self.arc_lamps = wc["lamps"] or header_lamps
        if not self.arc_lamps:
            raise ValueError(f"No lamps are on in the arc headers of {group.arcs}; set [wavecal] lamps")
        self.wave_reference = wavecal.load_reference(group.setup[0], wc["reference"])
        self.sky_lines = wc["sky_lines"]

        self.frames = {} # frame number -> FrameSpectrum
        self.combined = None # combine.Combined
        self.snr = None # S/N at 5500 A of the combined spectrum
        self.snr_bands = {}
        self.normalized = None # Normalized
        self.fluxed = None # (flux, sigma)

    # ------------------------------------------------------------------
    # per frame
    # ------------------------------------------------------------------

    def reduce_frame(self, frame, bias, flat):
        """Calibrate, trace, extract and wavelength-calibrate one science frame; writes its extracted spectrum."""
        path = self.group.science[frame]
        log.info(f"KARP is reducing science image {frame} ({os.path.basename(path)})")
        os.makedirs(self.out.frame_dir(frame), exist_ok=True)

        # Overscan, trim, electrons; bias structure; flat field; cosmic rays
        raw = load_frame(path, self.det)
        v_helio = heliocentric_correction(raw.meta, self.target["site_height"], self.target["ra"], self.target["dec"])
        flat_fielded = np.asarray(ccdproc.subtract_bias(raw, bias).data) / flat
        image = flat_fielded
        det = self.cfg["detector"]
        if self.cfg["cosmic_rays"]["enabled"]:
            # Data are in electrons and still include sky, as L.A.Cosmic's noise model expects
            image, cr_mask = ccdproc.cosmicray_lacosmic(
                image, sigclip=self.cfg["cosmic_rays"]["sigclip"], gain=1.0, readnoise=det["readnoise"],
                satlevel=det["full_well"], verbose=False)
            log.info(f"Cosmic rays: {int(cr_mask.sum())} pixels cleaned in image {frame}")
        if self.verbose:
            CCDData(flat_fielded, unit="electron").write(self.out.frame_file(frame, "image.fits"), overwrite=True)
        variance = variance_image(image, flat, det["readnoise"]) # e-^2 per pixel

        # Trace
        columns, trace, trace_fits, trace_used = fit_trace(
            image, self.aperture_half, self.buffer_cols, self.background_cols, self.cfg["trace"]["order"],
            self.binning, self.trace_start)
        self.plots.trace(trace_fits, trace, trace_used, frame)
        if self.verbose:
            self.plots.frame_image(image, trace, self.aperture_half, self.buffer_cols, self.background_cols, frame)

        # Extraction
        method = self.cfg["extraction"]["method"]
        log.debug(f"Extracting flux ({method})")
        n_rows = image.shape[0]
        res = extract(image, variance, flat, trace, self.aperture_half, self.buffer_cols, self.background_cols,
                      det["readnoise"], method=method, sky_order=self.cfg["extraction"]["sky_order"],
                      profile_window=self.profile_window, reject_sigma=self.cfg["extraction"]["reject_sigma"],
                      diag_rows=range(0, n_rows, 200) if self.verbose else ())
        flux, sky, var = res["flux"], res["sky"], res["var"] # e- per row; var includes the sky-fit uncertainty
        log.info(f"Extraction ({method}): median S/N per row {np.nanmedian(flux / np.sqrt(var)):.1f}, "
                 f"median sky under aperture {np.nanmedian(sky):.0f} e-, "
                 f"{int(np.isnan(flux).sum())} rows not extracted, {int(res['n_rejected'].sum())} pixels rejected")
        if self.verbose:
            self.plots.extraction_rows(res["diag"], frame)

        # Keep rows where the extraction worked, with their detector row so the
        # wavelength solution is evaluated at the right place. Zero or negative flux is
        # a valid measurement of a faint source and is kept.
        flux = np.array(flux)
        keep = np.isfinite(flux) & np.isfinite(var) & (var > 0)
        rows = np.arange(n_rows)[keep]
        flux, sky, sigma = flux[keep], sky[keep], np.sqrt(var[keep])

        # Wavelengths: arc solution, flexure offset (if corrected), heliocentric correction
        sol = self.wavelength_solution(frame, columns)
        offsets, offset = self.flexure_check(image, columns, sol.poly)
        wave = (sol.poly(np.asarray(rows, dtype=float)) - offset) * (1 + v_helio / C_KMS)
        log.debug(f"Heliocentric correction for image {frame}: {v_helio:+.3f} km/s")

        self.plots.frame_spectrum(wave, flux, sigma, sky, frame)
        if self.verbose:
            ends = (sol.poly(np.array([0.0, n_rows - 1.0])) - offset) * (1 + v_helio / C_KMS)
            log.debug(f"Wavelength range {ends.min():.2f}-{ends.max():.2f} A, "
                      f"mean dispersion {abs(ends[0] - ends[1]) / n_rows:.4f} A/pix")

        t = Table([wave, flux, sigma, sky, rows], names=("Lam", "Flux", "Sigma", "Sky", "Row"))
        t["Flux"].unit = t["Sigma"].unit = t["Sky"].unit = "electron"
        t["Lam"].unit = "Angstrom"
        t.meta.update({"OBJECT": self.objid, "FRAME": frame, "EXTRACT": method,
                       "VHELIO": (v_helio, "km/s, applied to Lam"), "WAVERMS": (sol.rms_kms, "km/s arc fit RMS"),
                       "FLEXURE": (offset, "A subtracted from the arc solution"),
                       "ARCFWHM": (sol.line_fwhm, "A, median FWHM of the arc lines")})
        t.write(self.out.extracted(frame), format="fits", overwrite=True)
        self.frames[frame] = FrameSpectrum(frame, wave, flux, sigma, sky, rows, v_helio, sol, columns.copy(),
                                           offsets, offset)

    def wavelength_solution(self, frame, columns):
        """
        Wavelength solution for one frame: the arc is extracted along that frame's trace
        and its lines identified automatically (see wavecal.py). Writes the matched lines
        to diagnostics/frame_<n>/wavecal_lines.txt.
        """
        log.debug("Wavelength calibration")
        arc_spec = wavecal.extract_arc(self.arc_image, columns, self.aperture_half,
                                       baseline_window=detector.to_binned(101, self.binning[1], odd=True, minimum=3))
        sol = wavecal.solve(arc_spec, self.arc_lamps, self.wave_reference, self.binning[1], self.wcfg)
        log.info(sol.summary())
        for note in sol.notes:
            log.debug("  " + note)
        self.plots.wavecal(sol, arc_spec, frame)
        with open(self.out.frame_file(frame, "wavecal_lines.txt"), "w") as f:
            f.write(f"# {sol.summary()}\n")
            f.write(f"# arcs {', '.join(os.path.basename(a) for a in self.group.arcs)}, "
                    f"lamps {', '.join(self.arc_lamps)}, reference {sol.reference}\n")
            for note in sol.notes:
                f.write(f"# {note}\n")
            f.write(sol.table())
        return sol

    def flexure_check(self, image, columns, poly):
        """
        Night-sky emission lines in the frame's background compared with the arc
        solution ([wavecal] flexure). Returns ({line: offset A}, offset to subtract).
        """
        mode = self.wcfg["flexure"]
        if mode == "off":
            return {}, 0.0
        sky_spec = wavecal.extract_sky(image, columns, self.aperture_half, self.buffer_cols, self.background_cols)
        offs = wavecal.sky_offsets(sky_spec, poly, self.sky_lines)
        offset = 0.0
        if offs:
            # Median, so one discrepant line can't drive a correction
            med_off = float(np.median(list(offs.values())))
            med_kms = float(np.median([o / l * C_KMS for l, o in offs.items()]))
            spread = max(offs.values()) - min(offs.values())
            details = ", ".join(f"{l:.1f} A: {o:+.3f} A" for l, o in offs.items())
            msg = f"Flexure check (sky line minus arc solution): {details}; median {med_off:+.3f} A = {med_kms:+.1f} km/s"
            if spread > 0.2:
                msg += f" (WARNING: sky lines disagree by {spread:.2f} A)"
            if mode == "correct":
                offset = med_off
                msg += " (corrected)"
        else:
            msg = "Flexure check: no night-sky lines detected"
        log.info(msg)
        return offs, offset

    # ------------------------------------------------------------------
    # combined spectrum
    # ------------------------------------------------------------------

    def combine(self):
        """
        Combine the extracted spectra of all science frames (read from disk, so this also
        works on an earlier run's output) and write <obj>_combined.fits. Returns the S/N at
        5500 A: the median of Flux/Sigma over 5450-5550 A.
        """
        log.info("KARP is combining spectra")
        nums = sorted(self.group.science)
        paths = {n: self.out.extracted(n) for n in nums}
        missing = [p for p in paths.values() if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"Extracted spectra not found (run with reduce = true first): {missing}")
        frames = {}
        for n, path in paths.items():
            t = Table.read(path)
            frames[n] = tuple(np.asarray(t[c], dtype=float) for c in ("Lam", "Flux", "Sigma"))
        log.debug("Using these files: " + ", ".join(paths.values()))

        ccfg = self.cfg["combine"]
        comb = combine_frames(frames, ccfg["scale"], ccfg["scale_order"], ccfg["method"],
                              float(ccfg["reject_sigma"]), ccfg["jackknife"])

        # S/N per pixel from the propagated errors, around several wavelengths
        snr5500 = snr_in(comb.wave, comb.flux, comb.sigma, comb.mask, 5450, 5550)
        bands = {lam: snr_in(comb.wave, comb.flux, comb.sigma, comb.mask, lam - 50, lam + 50)
                 for lam in range(4000, 10000, 500)}
        bands = {lam: v for lam, v in bands.items() if np.isfinite(v)}
        log.info(f"S/N per pixel at 5500 A (median Flux/Sigma, 5450-5550 A): {snr5500:.1f}")
        log.info("S/N per pixel by wavelength: " + ", ".join(f"{lam} A: {v:.0f}" for lam, v in bands.items()))

        path = self.out.product("combined.fits")
        t = Table([comb.wave, comb.flux, comb.sigma, comb.mask], names=("Lam", "Flux", "Sigma", "Mask"))
        t["Lam"].unit = "Angstrom"
        t["Flux"].unit = t["Sigma"].unit = "electron"
        t.meta["SCINUMS"] = ",".join(str(n) for n in nums)
        t.meta["REFFRAME"] = (comb.ref_frame, "frames scaled to this one")
        t.meta["SCALE"] = (ccfg["scale"], "frame scaling method")
        t.meta["COMBINE"] = (ccfg["method"], "frame combination method")
        if ccfg["method"] == "weighted":
            t.meta["REJSIG"] = (float(ccfg["reject_sigma"]), "rejection threshold, sigma")
            t.meta["NREJECT"] = (int(sum(comb.n_rejected.values())), "frame pixels rejected")
        t.meta["SNR5500"] = (snr5500, "median Flux/Sigma, 5450-5550 A")
        t.write(path, format="fits", overwrite=True)
        log.info(f"Combined spectrum written to {path}")
        self.plots.combined(comb, snr5500)
        self.plots.frame_comparison(comb)

        self.combined, self.snr, self.snr_bands = comb, snr5500, bands
        return snr5500

    def load_combined(self):
        """The combined spectrum from this run, or <obj>_combined.fits from an earlier one."""
        if self.combined is not None:
            return self.combined
        path = self.out.product("combined.fits")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Combined spectrum not found (run with combine = true first): {path}")
        t = Table.read(path)
        self.combined = Combined(*(np.asarray(t[c], dtype=float) for c in ("Lam", "Flux", "Sigma")),
                                 np.asarray(t["Mask"]), int(t.meta["REFFRAME"]),
                                 [int(n) for n in str(t.meta.get("SCINUMS", "")).split(",") if n])
        return self.combined

    def reference_exposure(self):
        """(exptime s, mid-exposure airmass, mid-exposure MJD) of the frame the combined spectrum is scaled to."""
        header = CCDData.read(self.group.science[self.load_combined().ref_frame], unit="adu").meta
        return exposure_info(header, self.target["site_height"], self.target["ra"], self.target["dec"])

    # ------------------------------------------------------------------
    # optional: continuum normalization
    # ------------------------------------------------------------------

    def normalize(self):
        """
        Continuum-normalize the combined spectrum and write <obj>_normalized.fits with
        Flux and Sigma divided by the continuum, the Continuum itself, Mask, and ContFit
        (1 = pixel used in the continuum fit).
        """
        comb = self.load_combined()
        ncfg = self.cfg["normalization"]
        lam, flux, sig, mask = comb.wave, comb.flux, comb.sigma, comb.mask.copy()

        log.debug("Fitting the continuum")
        regions = mask_regions(ncfg["mask"], ncfg["extra_mask"], "stellar.txt")
        cont, used, notes = fit_continuum(lam, flux, sig, (mask & 2) != 0, regions, # bit 4 is informational
                                          knot_spacing=ncfg["knot_spacing"], low_reject=ncfg["low_reject"],
                                          high_reject=ncfg["high_reject"])
        norm_flux, norm_sig = flux / cont, sig / cont

        # Optional flags on normalized flux (they don't change any values)
        with np.errstate(invalid="ignore"):
            if ncfg["clip"] is not None:
                mask[norm_flux > ncfg["clip"]] |= 1
            for lam_min, lam_max, threshold in ncfg["clip_regions"]:
                mask[(lam >= lam_min) & (lam <= lam_max) & (norm_flux > threshold)] |= 1
        msg = "Continuum: " + "; ".join(notes)
        if ncfg["clip"] is not None or len(ncfg["clip_regions"]):
            msg += f"; {int(((mask & 1) != 0).sum())} pixels flagged above the clip thresholds"
        log.info(msg)

        t = Table([lam, norm_flux, norm_sig, cont, mask, used.astype(np.int16)],
                  names=("Lam", "Flux", "Sigma", "Continuum", "Mask", "ContFit"))
        t["Lam"].unit = "Angstrom"
        t["Continuum"].unit = "electron"
        t.meta["KNOTSPAC"] = (ncfg["knot_spacing"], "continuum spline knot spacing, A")
        t.meta["LOWREJ"] = (ncfg["low_reject"], "continuum clipping below, sigma")
        t.meta["HIGHREJ"] = (ncfg["high_reject"], "continuum clipping above, sigma")
        t.write(self.out.product("normalized.fits"), format="fits", overwrite=True)
        self.plots.continuum(lam, flux, cont, used, regions)
        self.plots.normalized(lam, norm_flux, norm_sig)
        self.normalized = Normalized(lam, norm_flux, norm_sig, mask, cont, used)

    # ------------------------------------------------------------------
    # optional: flux calibration
    # ------------------------------------------------------------------

    def instrument_resolution(self):
        """
        KOSMOS resolution (FWHM, A) for comparing a standard with its tabulated flux:
        [fluxcal] resolution if set, else the median arc line FWHM of this object's
        frames (an upper limit for a star smaller than the slit), else 3 pixels.
        Returns (fwhm, where it came from).
        """
        fcfg = self.cfg["fluxcal"]
        if fcfg["resolution"] > 0:
            return float(fcfg["resolution"]), "[fluxcal] resolution"
        values = []
        for n in self.group.science:
            if os.path.exists(self.out.extracted(n)):
                v = Table.read(self.out.extracted(n)).meta.get("ARCFWHM")
                if v is not None and np.isfinite(v):
                    values.append(float(v))
        if values:
            return float(np.median(values)), "arc line width"
        return 3 * float(np.median(fluxcal.pixel_width(self.load_combined().wave))), "3 pixels"

    def sensitivity_function(self, std_name):
        """
        Sensitivity function from this object's combined spectrum, this object being the
        standard `std_name` (library name, or [fluxcal] reference for the tabulated flux).
        Writes calib/<obj>_sensfunc.fits and a plot. Returns the fluxcal.sensitivity dict
        plus the standard's name and the reference exposure's MJD.
        """
        fcfg = self.cfg["fluxcal"]
        comb = self.load_combined()
        exptime, airmass, mjd = self.reference_exposure()
        ref_lam, ref_flux, ref_fwhm, ref_label = fluxcal.load_standard(fcfg["reference"] or std_name)
        ext = fluxcal.load_extinction(fcfg["extinction"])
        inst_fwhm, source = self.instrument_resolution()
        sens = fluxcal.sensitivity(comb.wave, comb.flux, comb.sigma, (comb.mask & 2) != 0,
                                   exptime, airmass, ext, (ref_lam, ref_flux, ref_fwhm), inst_fwhm, fcfg)
        sens.update(standard=self.objid, reference=ref_label, mjd=mjd, setup=self.group.setup)
        in_range = (ref_lam >= comb.wave.min()) & (ref_lam <= comb.wave.max())
        log.info(f"Sensitivity function from {self.objid} ({ref_label}; frame {comb.ref_frame}, "
                 f"{exptime:g} s, airmass {airmass:.3f}): " + "; ".join(sens["notes"])
                 + f"; valid {sens['range'][0]:.0f}-{sens['range'][1]:.0f} A")
        log.info(f"  Compared at matched resolution: KOSMOS {inst_fwhm:.2f} A FWHM ({source}), tabulated "
                 f"{ref_fwhm[in_range].min():.1f}-{ref_fwhm[in_range].max():.1f} A; the standard's lines are "
                 f"offset by {sens['offset']:+.2f} A = {sens['offset'] / 5500 * C_KMS:+.0f} km/s at 5500 A from the "
                 "tabulated spectrum (wavelength calibration check)")
        t = Table([sens["lam"], sens["sens"], sens["fit"], sens["used"].astype(np.int16), sens["ref_flux"]],
                  names=("Lam", "Sens", "SensFit", "Used", "RefFlux"))
        t["Lam"].unit = "Angstrom"
        t["Sens"].unit = t["SensFit"].unit = "mag" # 2.5 log10(e-/s/A per erg/s/cm2/A)
        t["RefFlux"].unit = "erg / (s cm2 Angstrom)"
        t.meta.update({"STANDARD": self.objid, "REFFILE": ref_label, "REFFRAME": comb.ref_frame,
                       "INSTFWHM": (inst_fwhm, "A, KOSMOS resolution assumed"),
                       "STDOFF": (sens["offset"], "A, standard's lines minus tabulated"),
                       "EXPTIME": exptime, "AIRMASS": (airmass, "mid-exposure"),
                       "EXTINCT": os.path.basename(str(fcfg["extinction"])),
                       "SENSRMS": (sens["rms"], "mag, RMS of the fit"),
                       "LAMMIN": sens["range"][0], "LAMMAX": sens["range"][1]})
        t.write(self.out.sensfunc, format="fits", overwrite=True)
        self.plots.sensitivity(sens)
        return sens

    def flux_calibrate(self, sens):
        """
        Apply a sensitivity function (from sensitivity_function, possibly another
        object's) to the combined spectrum. Writes <obj>_fluxed.fits with Lam, Flux,
        Sigma (erg/s/cm^2/A) and Mask (bit 2 also outside the calibrated range, bit 8
        where the sensitivity was extrapolated).
        """
        fcfg = self.cfg["fluxcal"]
        comb = self.load_combined()
        exptime, airmass, _ = self.reference_exposure()
        ext = fluxcal.load_extinction(fcfg["extinction"])
        flux, sig, valid, extrap = fluxcal.calibrate(comb.wave, comb.flux, comb.sigma, exptime, airmass, ext,
                                                     sens["lam"], sens["fit"], sens["range"], fcfg["max_extrapolate"])
        mask = comb.mask.copy()
        mask[~valid] |= 2 # 2 = no data (outside the calibrated range)
        mask[extrap] |= 8 # 8 = sensitivity extrapolated (linearly) beyond the fitted range
        cal = comb.wave[valid]
        msg = (f"Flux calibrated with the sensitivity function of {sens['standard']} "
               f"(frame {comb.ref_frame}: {exptime:g} s, airmass {airmass:.3f}); calibrated "
               f"{cal.min():.0f}-{cal.max():.0f} A, {int(extrap.sum())} pixels extrapolated")
        is_standard = sens["standard"] == self.objid
        if is_standard:
            # Closure test: the standard calibrated with its own sensitivity function
            ratio = flux / sens["ref_flux"]
            ok = valid & ~extrap & sens["used"] & np.isfinite(ratio)
            msg += (f"; closure test: flux / tabulated flux = {np.median(ratio[ok]):.4f} median, "
                    f"{mad_std(ratio[ok]) * 100:.2f}% scatter over the fitted pixels")
        log.info(msg)
        t = Table([comb.wave, flux, sig, mask], names=("Lam", "Flux", "Sigma", "Mask"))
        t["Lam"].unit = "Angstrom"
        t["Flux"].unit = t["Sigma"].unit = "erg / (s cm2 Angstrom)"
        t.meta.update({"BUNIT": "erg/s/cm2/A", "STANDARD": sens["standard"], "REFFILE": sens["reference"],
                       "REFFRAME": comb.ref_frame, "EXPTIME": exptime, "AIRMASS": (airmass, "mid-exposure"),
                       "SENSRMS": (sens["rms"], "mag, RMS of the sensitivity fit"),
                       "FLUXNOTE": "shape reliable; absolute level depends on slit losses and transparency"})
        t.write(self.out.product("fluxed.fits"), format="fits", overwrite=True)
        self.plots.fluxed(comb.wave, flux, sig, valid, extrap, sens["ref_flux"] if is_standard else None)
        self.fluxed = (flux, sig)
        return flux, sig
