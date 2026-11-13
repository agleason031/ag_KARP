#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 15:52:33 2025

@author: agleason
"""
from tqdm import tqdm
from scipy.interpolate import interp1d
from astropy.nddata import CCDData
from astropy.table import Table
from astropy.stats import mad_std
from numpy.polynomial import Polynomial
from functions import fit_cent_gaussian, build_fit_args
from extract import extract, variance_image
import wavecal
from wavecal import C_KMS
from normalize import fit_continuum, load_mask
from sci_tools import heliocentric_correction, detector_geometry, read_reduced, read_arcs, exposure_info
import fluxcal
import detector
from astropy.io import fits
from joblib import Parallel, delayed
import ccdproc
import logging
import os
import warnings
import numpy as np

#-------------------------------------
#helper functions

# Error of the median of n values relative to the error of their mean, for Gaussian
# noise and similar errors in each frame (Monte Carlo, 400k trials; large-n limit sqrt(pi/2))
_MEDIAN_ERROR_FACTOR = np.array([1.0, 1.0, 1.0, 1.159, 1.094, 1.196, 1.135, 1.213, 1.161,
                                 1.222, 1.174, 1.232, 1.185, 1.234, 1.194, 1.235])

def median_error_factor(n):
    """Factor for each n (array); n = 0 gives 1 (those pixels have no data anyway)."""
    n = np.asarray(n)
    return np.where(n < len(_MEDIAN_ERROR_FACTOR),
                    _MEDIAN_ERROR_FACTOR[np.minimum(n, len(_MEDIAN_ERROR_FACTOR) - 1)],
                    np.sqrt(np.pi / 2))

log = logging.getLogger("karp")

#-------------------------

class sci_modules:
    def __init__ (self, cfg, group, target_dir, grapher):
        """
        cfg: full config (config.load_config); group: discover.Group with this
        object's frames; target_dir: output directory for this object (ends in "/").
        """
        self.group = group
        self.objid = group.objid # Object ID
        self.target_dir = target_dir
        ext = cfg["extraction"]
        # Sizes in the config are physical (arcsec across the slit, unbinned rows along the
        # dispersion); convert them to pixels of this group's binning
        self.binx, self.biny = detector.setup_binning(group.setup)
        arcsec_per_col = cfg["detector"]["pixel_scale"] * self.binx
        # Aperture: 2*appw+1 columns (a width w in pixels uses (w-1)/2 per side, rounded down)
        self.appw = int(np.round(ext["aperture_arcsec"] / arcsec_per_col - 1) / 2)
        self.buffw = int(round(ext["buffer_arcsec"] / arcsec_per_col)) # gap between aperture and background, per side
        self.bckw = int(round(ext["background_arcsec"] / arcsec_per_col)) # background width, per side
        self.extract_method = ext["method"] # "optimal" or "boxcar"
        self.sky_order = ext["sky_order"] # polynomial order of the sky fit across the slit
        # rows over which the spatial profile is smoothed
        self.profile_window = detector.to_binned(ext["profile_window"], self.biny, odd=True, minimum=3)
        log.info(f"Binning {self.binx}x{self.biny} ({arcsec_per_col:.3f}\"/column): aperture {2 * self.appw + 1} "
                 f"columns, buffer {self.buffw}, background {self.bckw} per side; profile window "
                 f"{self.profile_window} rows")
        self.reject_sigma = ext["reject_sigma"] # optimal extraction outlier threshold
        self.trace_order = cfg["trace"]["order"] # Polynomial order of the trace fit
        # Optional starting column for the trace (config: unbinned CCD column); found automatically if None
        self.trace_col = None if cfg["trace"]["col"] is None else int(cfg["trace"]["col"] // self.binx)
        norm = cfg["normalization"]
        self.knot_spacing = norm["knot_spacing"] # A between continuum spline knots
        self.low_reject = norm["low_reject"] # continuum clipping below the fit, sigma
        self.high_reject = norm["high_reject"] # continuum clipping above the fit, sigma
        self.norm_mask = norm["mask"] # "default", "none", or a mask file
        self.norm_extra_mask = norm["extra_mask"] # extra [lam_min, lam_max] regions left out of the fit
        # Cosmic-ray cleaning of the 2D science frames (L.A.Cosmic)
        self.cr_clean = cfg["cosmic_rays"]["enabled"]
        self.cr_sigclip = cfg["cosmic_rays"]["sigclip"] # detection threshold in sigma
        self.readnoise = cfg["detector"]["readnoise"] # electrons
        self.full_well = cfg["detector"]["full_well"] # electrons
        # Optional flags on the normalized spectrum (None / empty = off). Flagged pixels
        # keep their value but get Mask bit 1 in the output.
        self.norm_clip = norm["clip"] # normalized flux > this
        self.norm_clip_regions = norm["clip_regions"] # [[lam_min, lam_max, threshold], ...]
        # Combining frames
        self.comb_scale = cfg["combine"]["scale"] # "smooth", "median" or "none"
        self.comb_scale_order = cfg["combine"]["scale_order"]
        self.comb_method = cfg["combine"]["method"] # "weighted", "mean" or "median"
        self.comb_reject_sigma = float(cfg["combine"]["reject_sigma"])
        # Heliocentric correction inputs: coordinates from headers unless overridden
        self.target_ra = cfg["target"]["ra"]
        self.target_dec = cfg["target"]["dec"]
        self.site_height = cfg["target"]["site_height"]
        self.comb_flux = None # combined spectrum (set by combine_spectra)
        self.continuum = None # normalized combined spectrum (set by normalize_combined)
        
        self.grapher = grapher
        
        self.v_helio_dict = {} # Heliocentric correction (km/s) per science frame
        self.flux_raw_cor1_dict = {}
        self.wavelengths_dict = {}
        self.sky_raw_dict = {}
        self.err_dict = {} # 1-sigma extraction error per kept row
        self.var_dict = {} # 2D variance image per science frame
        self.trace_dict = {} # sub-pixel trace per science frame
        self.sci_final_dict = {}
        self.cen_line_dict = {}
        self.wave_solutions = {} # wavecal.WaveSolution per science frame
        self.flexure_dict = {} # {sky line: offset in A} per science frame
        self.geom = detector_geometry(cfg["detector"]) # [detector] settings; layout read from each frame's header
        with fits.open(group.science[min(group.science)]) as hdul:
            log.info(detector.describe(detector.frame_geometry(hdul[0].header, self.geom)))
        # Arc image: overscan-subtracted, trimmed (electrons); repeats median-combined, lamps added
        self.arc_image, header_lamps = read_arcs(group.arcs, self.geom)
        # Wavelength calibration settings
        # [wavecal] with its row sizes (unbinned) converted to this binning
        self.wcfg = dict(cfg["wavecal"], search=cfg["wavecal"]["search"] / self.biny,
                         match_tol=cfg["wavecal"]["match_tol"] / self.biny)
        self.arc_lamps = self.wcfg["lamps"] or header_lamps
        if not self.arc_lamps:
            raise ValueError(f"No lamps are on in the arc headers of {group.arcs}; set [wavecal] lamps")
        self.wave_reference = wavecal.load_reference(group.setup[0], self.wcfg["reference"])
        self.binning_y = int(group.setup[2].split("x")[1]) # Binning along the dispersion (rows)
        self.flexure = self.wcfg["flexure"]
        self.sky_lines = self.wcfg["sky_lines"]
        
    #does all the image reduction through wavelength calibration
    def reduce_image(self, scinum, masterbias, final_flat, verbose):
        sci_location = self.group.science[scinum]
        
        # Make a output text file detaling relevant fit paramaters that KARP calculates
        # Initalize directory for reduction plots for this science image
        os.makedirs(self.target_dir+"ImageNumber_"+str(scinum), exist_ok=True) # Initializes a directory for E20 files            
        
        log.info(f"KARP is reducing science image {scinum} ({os.path.basename(sci_location)})")
        
        #creates sci_final_1 if not already created
        if scinum in self.sci_final_dict:
            sci_final_1 = self.sci_final_dict[scinum]
        else:
            # Overscan-subtract, trim and convert to electrons, then remove the bias structure and flat-field
            sci_raw = read_reduced(sci_location, self.geom)
            # Heliocentric correction for this exposure, from its header
            self.v_helio_dict[scinum] = heliocentric_correction(
                sci_raw.meta, self.site_height, self.target_ra, self.target_dec)
            sci_final = np.asarray(ccdproc.subtract_bias(sci_raw, masterbias).data)/final_flat
            sci_final_1 = sci_final

            if self.cr_clean:
                # Data are in electrons and still include sky, as L.A.Cosmic's noise model expects
                sci_final_1, cr_mask = ccdproc.cosmicray_lacosmic(
                    sci_final_1, sigclip=self.cr_sigclip, gain=1.0, readnoise=self.readnoise,
                    satlevel=self.full_well, verbose=False)
                log.info(f"Cosmic rays: {int(cr_mask.sum())} pixels cleaned in image {scinum}")
            self.sci_final_dict[scinum] = sci_final_1

            # Plot and write to disk only when first created; a cached image was already written
            if (verbose == True):
                self.grapher.plot_sci_final(sci_final, scinum)
                sci_final_write = CCDData(sci_final, unit="electron")
                sci_final_write.write(self.target_dir+"ImageNumber_"+str(scinum)+"/"+"sci_final_"+str(scinum)+".fits", overwrite = True)

            # Per-pixel variance (e-^2) for the extraction errors
            self.var_dict[scinum] = variance_image(sci_final_1, final_flat, self.readnoise)

    #----------------------------
    #trace fitting
        
        #trace shouldn't change between aperature changes, brightest spot is center and is always in aperature
        #so save and attempt to get from dict to save expensive calculations
        if scinum in self.cen_line_dict:
            cen_line = self.cen_line_dict[scinum]
            trace = self.trace_dict[scinum]
        else:
            cen_line, trace, trace_fits = self.fit_trace(sci_final_1, verbose) #get center trace
            self.grapher.plot_cen_line(cen_line, scinum)
            self.cen_line_dict[scinum] = cen_line.copy() #save center trace (whole pixels)
            self.trace_dict[scinum] = trace.copy() #sub-pixel trace used for extraction
            if (verbose == True):
                self.grapher.gauss_cen_plots(trace_fits, cen_line, scinum)
    #----------------------------------
    #flux extraction
        log.debug(f"Extracting flux ({self.extract_method})")
        n_rows = sci_final_1.shape[0]
        res = extract(sci_final_1, self.var_dict[scinum], final_flat, trace, self.appw, self.buffw, self.bckw,
                      self.readnoise, method=self.extract_method, sky_order=self.sky_order,
                      profile_window=self.profile_window, reject_sigma=self.reject_sigma,
                      diag_rows=range(0, n_rows, 200) if verbose else ())
        flux_raw1 = res["flux"] # electrons per row
        sky_raw1 = res["sky"] # sky electrons under the aperture per row
        var_raw1 = res["var"] # variance of flux_raw1 (e-^2), including the sky-fit uncertainty
        msg = (f"Extraction ({self.extract_method}): median S/N per row {np.nanmedian(flux_raw1 / np.sqrt(var_raw1)):.1f}, "
               f"median sky under aperture {np.nanmedian(sky_raw1):.0f} e-, "
               f"{int(np.isnan(flux_raw1).sum())} rows not extracted, {int(res['n_rejected'].sum())} pixels rejected")
        log.info(msg)
        
        if (verbose == True):
            self.grapher.plot_raw_flux(flux_raw1, scinum)
            self.grapher.plot_extraction_rows(res["diag"], scinum)
        
    #--------------------------------------
    #wavelength calibration
        
        # Drop rows where the extraction failed (no usable pixels), but remember the
        # detector row of every kept pixel so the wavelength solution is evaluated at
        # the right place. Zero or negative flux is a valid measurement of a faint
        # source and is kept (dropping it would bias faint spectra high).
        n_rows = len(flux_raw1)
        flux_raw1 = np.array(flux_raw1)
        ii = np.isfinite(flux_raw1) & np.isfinite(var_raw1) & (var_raw1 > 0)
        y_pix = np.arange(n_rows)[ii] # Detector row of each kept pixel
        flux_raw_cor1 = flux_raw1[ii]
        sky_raw1 = sky_raw1[ii]
        err_raw1 = np.sqrt(var_raw1[ii]) # 1-sigma error of each kept pixel

        # Wavelength solution from the arc, extracted along this frame's trace
        sol = self.wavelength_solution(scinum, cen_line, verbose)
        poly = sol.poly # row -> air wavelength, before flexure and heliocentric corrections

        # Flexure check: night-sky emission lines in this frame's background
        offset = 0.0 # A, subtracted from the arc solution when flexure = "correct"
        if self.flexure != "off":
            sky_spec = wavecal.extract_sky(sci_final_1, cen_line, self.appw, self.buffw, self.bckw)
            offs = wavecal.sky_offsets(sky_spec, poly, self.sky_lines)
            self.flexure_dict[scinum] = offs
            if offs:
                # Median, so one discrepant line can't drive a correction
                med_off = float(np.median(list(offs.values())))
                med_kms = float(np.median([o / l * C_KMS for l, o in offs.items()]))
                spread = max(offs.values()) - min(offs.values())
                details = ", ".join(f"{l:.1f} A: {o:+.3f} A" for l, o in offs.items())
                msg = f"Flexure check (sky line minus arc solution): {details}; median {med_off:+.3f} A = {med_kms:+.1f} km/s"
                if spread > 0.2:
                    msg += f" (WARNING: sky lines disagree by {spread:.2f} A)"
                if self.flexure == "correct":
                    offset = med_off
                    msg += " (corrected)"
            else:
                msg = "Flexure check: no night-sky lines detected"
            log.info(msg)

        # Wavelength of each kept pixel, with the flexure offset and this exposure's heliocentric correction
        def lam_at(y, fit, v_helio):
            return (fit(np.asarray(y, dtype=float)) - offset) * (1 + v_helio / C_KMS)
        v_helio = self.v_helio_dict[scinum]
        wavelengths = lam_at(y_pix, poly, v_helio)

        # Arc solution residuals (lines kept by the fit)
        res_rows = sol.rows[sol.used]
        del_lam = sol.residuals[sol.used]
        res_vel = del_lam / sol.lam[sol.used] * C_KMS
        self.grapher.lam_cal_res(res_rows, del_lam, scinum)
        self.grapher.lam_cal_res_vel(res_rows, res_vel, scinum)

        log.debug(f"Heliocentric correction for image {scinum}: {v_helio:+.3f} km/s")
        if (verbose == True):
            self.grapher.plot_sci_lamcal(lam_at, y_pix, flux_raw_cor1, poly, v_helio, scinum)
            rnErr = err_raw1 # Extraction error per pixel
            self.grapher.plot_sci_lamcal_res(lam_at, y_pix, flux_raw_cor1, rnErr, poly, v_helio, scinum)

            # Dispersion over the full detector (wavelength decreases with row number)
            lam_out_max = lam_at(0, poly, v_helio) # Wavelength at the first row
            lam_out_min = lam_at(n_rows - 1, poly, v_helio) # Wavelength at the last row
            dispersion = (lam_out_max-lam_out_min)/n_rows
            log.debug(f"Wavelength range {lam_out_min:.2f}-{lam_out_max:.2f} A, mean dispersion {dispersion:.4f} A/pix")
            
        if (verbose == True):
            # Store flux_raw_cor1
            with open(self.target_dir+"ImageNumber_"+str(scinum)+"/"+"flux_raw_cor1_"+str(scinum)+".txt", "w") as file:
                for val in flux_raw_cor1:
                    file.write(f"{val}\n")
            
            with open(self.target_dir+"ImageNumber_"+str(scinum)+"/"+"wavelengths_"+str(scinum)+".txt", "w") as file:
                for wave in wavelengths:
                    file.write(f"{wave}\n")
        
            # Store raw sky values
            with open(self.target_dir+"ImageNumber_"+str(scinum)+"/"+"skyraw_"+str(scinum)+".txt", "w") as file:
                for sky in sky_raw1:
                    file.write(f"{sky}\n")
                    
        #save data to class dictionaries for easy loading
        self.flux_raw_cor1_dict[scinum] = flux_raw_cor1
        self.sky_raw_dict[scinum] = sky_raw1
        self.err_dict[scinum] = err_raw1
        
        # Extracted (not normalized) spectrum of this frame
        t = Table([wavelengths, flux_raw_cor1, err_raw1, sky_raw1, y_pix],
                  names=("Lam", "Flux", "Sigma", "Sky", "Row"))
        t["Flux"].unit = t["Sigma"].unit = t["Sky"].unit = "electron"
        t["Lam"].unit = "Angstrom"
        t.meta.update({"OBJECT": self.objid, "FRAME": scinum, "EXTRACT": self.extract_method,
                       "VHELIO": (v_helio, "km/s, applied to Lam"), "WAVERMS": (sol.rms_kms, "km/s arc fit RMS"),
                       "FLEXURE": (offset, "A subtracted from the arc solution"),
                       "ARCFWHM": (sol.line_fwhm, "A, median FWHM of the arc lines")})
        t.write(f"{self.target_dir}OUT/{self.objid}_{scinum}_extracted.fits", format="fits", overwrite=True)
        self.wavelengths_dict[scinum] = wavelengths
    
    def wavelength_solution(self, scinum, cen_line, verbose):
        """
        Wavelength solution for one science frame: the arc is extracted along that
        frame's trace and its lines identified automatically (see wavecal.py).
        Cached per frame. Also writes OUT/<objid>_wavecal_<n>.txt with the matched lines.
        """
        if scinum in self.wave_solutions:
            return self.wave_solutions[scinum]
        log.debug("Wavelength calibration")
        arc_spec = wavecal.extract_arc(self.arc_image, cen_line, self.appw,
                                       baseline_window=detector.to_binned(101, self.biny, odd=True, minimum=3))
        sol = wavecal.solve(arc_spec, self.arc_lamps, self.wave_reference, self.binning_y, self.wcfg)
        log.info(sol.summary())
        for note in sol.notes:
            log.debug("  " + note)
        with open(f"{self.target_dir}OUT/{self.objid}_wavecal_{scinum}.txt", "w") as f:
            f.write(f"# {sol.summary()}\n")
            f.write(f"# arcs {', '.join(os.path.basename(a) for a in self.group.arcs)}, lamps {', '.join(self.arc_lamps)}, reference {sol.reference}\n")
            for note in sol.notes:
                f.write(f"# {note}\n")
            f.write(sol.table())
        self.wave_solutions[scinum] = sol
        return sol

#-----------------------------------------

    def combine_spectra(self, scinums, verbose, jackknife=False):
        """
        Combine the extracted spectra of the given science images
        (OUT/{objid}_{n}_extracted.fits) and write OUT/{objid}_combined.fits.

        Each frame is interpolated onto the wavelength grid of the frame with the
        highest S/N (the reference) and scaled to it: by a smooth function of
        wavelength (scale = "smooth", a polynomial of order scale_order fit to the
        flux ratio with clipping), a single factor ("median"), or not at all ("none").
        The scaled frames are combined per pixel by comb_method:
          "weighted": inverse-variance weighted mean, after rejecting values more than
                      reject_sigma (their own error) from the median of the frames
                      (only where >= 3 frames have data); exact error 1/sqrt(sum 1/sigma^2).
                      Mask bit 4 marks pixels where a value was rejected.
          "mean":     plain mean; error sqrt(sum sigma^2) / n.
          "median":   median; error that of the inverse-variance mean times the
                      median/mean noise factor for that many frames.
        Mask bit 2 marks pixels with no data.
        Flux is in electrons per pixel, on the reference frame's scale.
        Returns the S/N at 5500 A: the median of Flux/Sigma over 5450-5550 A.
        With jackknife=True, also report the S/N with each image left out in turn.
        """
        log.info("KARP is combining spectra")

        # Only this object's spectra for the requested images
        file_names = {n: f"{self.target_dir}OUT/{self.objid}_{n}_extracted.fits" for n in scinums}
        missing = [f for f in file_names.values() if not os.path.exists(f)]
        if missing:
            raise FileNotFoundError(f"Extracted spectra not found (run with reduce = true first): {missing}")
        frames = {}
        for n, fname in file_names.items():
            t = Table.read(fname)
            frames[n] = (np.asarray(t["Lam"], dtype=float), np.asarray(t["Flux"], dtype=float),
                         np.asarray(t["Sigma"], dtype=float))
        log.debug("Using these files: " + ", ".join(file_names.values()))

        # Reference: the frame with the highest median S/N; its wavelength grid is used
        frame_snr = {n: float(np.nanmedian(f / s)) for n, (_, f, s) in frames.items()}
        ref = max(frame_snr, key=frame_snr.get)
        gwave = frames[ref][0]
        log.info("Frame S/N per pixel (median): " + ", ".join(f"{n}: {v:.1f}" for n, v in frame_snr.items())
            + f"; reference frame {ref}")

        # Interpolate every frame onto the grid
        on_grid = {}
        for n, (lam, f, s) in frames.items():
            fi = interp1d(lam, f, bounds_error=False, fill_value=np.nan)(gwave)
            si = interp1d(lam, s, bounds_error=False, fill_value=np.nan)(gwave)
            on_grid[n] = (fi, si)

        # Scale each frame to the reference
        ref_f = on_grid[ref][0]
        x = (gwave - gwave.mean()) / (np.ptp(gwave) / 2) # scaled wavelength for a well-conditioned fit
        for n, (fi, si) in on_grid.items():
            if n == ref or self.comb_scale == "none":
                continue
            ratio = ref_f / fi
            ok = np.isfinite(ratio) & (fi > 0) & (ref_f > 0)
            if self.comb_scale == "median":
                scale = np.full(len(gwave), np.median(ratio[ok]))
            else: # smooth
                use = ok.copy()
                for _ in range(10):
                    coef = np.polyfit(x[use], ratio[use], self.comb_scale_order)
                    resid = ratio - np.polyval(coef, x)
                    new_use = ok & (np.abs(resid) < 3 * mad_std(resid[use]))
                    if np.array_equal(new_use, use):
                        break
                    use = new_use
                scale = np.polyval(coef, x)
            on_grid[n] = (fi * scale, si * scale)
            log.info(f"Frame {n} scaled to frame {ref}: factor {np.nanmedian(scale):.4f} (median), "
                f"range {np.nanmin(scale):.4f}-{np.nanmax(scale):.4f}")

        def combine(nums):
            """
            Combine frames `nums` on gwave with self.comb_method.
            Returns (flux, sigma, mask, n_rejected per frame).
            """
            gflux = np.array([on_grid[n][0] for n in nums])
            gsig = np.array([on_grid[n][1] for n in nums])
            good = np.isfinite(gflux) & np.isfinite(gsig) & (gsig > 0)
            gflux = np.where(good, gflux, np.nan)
            gsig = np.where(good, gsig, np.nan)
            n_used = good.sum(axis=0)
            n_rej = np.zeros(len(nums), dtype=int)
            mask = np.zeros(len(gwave), dtype=np.int16)
            with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
                warnings.simplefilter("ignore", RuntimeWarning) # All-NaN columns are flagged below
                med = np.nanmedian(gflux, axis=0)
                if self.comb_method == "median":
                    # Error: that of the inverse-variance mean of the frames used, times how
                    # much noisier a median of that many values is than a mean
                    inv_var = np.where(good, 1.0 / gsig**2, 0.0)
                    flux = med
                    sig = np.sqrt(1.0 / np.sum(inv_var, axis=0)) * median_error_factor(n_used)
                elif self.comb_method == "mean":
                    # Plain average of the frames used; exact error propagation
                    flux = np.nanmean(gflux, axis=0)
                    sig = np.sqrt(np.nansum(gsig**2, axis=0)) / n_used
                else: # weighted
                    # Reject values more than reject_sigma (their own error) from the median,
                    # where >= 3 frames have data (with 2, an outlier drags the median halfway
                    # and both would go). Never reject every value at a pixel.
                    use = good.copy()
                    if self.comb_reject_sigma > 0: # 0 = no rejection
                        dev = np.abs(gflux - med) / gsig
                        reject = good & (dev > self.comb_reject_sigma) & (n_used >= 3)
                        reject &= ~np.all(reject | ~good, axis=0) # keep a pixel's values if all would go
                        use &= ~reject
                        n_rej = reject.sum(axis=1)
                        mask[reject.any(axis=0)] |= 4 # 4 = a frame was rejected here (informational)
                    inv_var = np.where(use, 1.0 / gsig**2, 0.0)
                    wsum = np.sum(inv_var, axis=0)
                    flux = np.sum(np.where(use, gflux, 0.0) * inv_var, axis=0) / wsum
                    sig = np.sqrt(1.0 / wsum)
            no_data = (n_used == 0) | ~np.isfinite(flux) | ~np.isfinite(sig)
            flux = np.where(no_data, np.nan, flux)
            mask[no_data] |= 2 # 2 = no data
            return flux, sig, mask, n_rej

        def snr_in(flux, sig, mask, lam_min, lam_max):
            """Median Flux/Sigma of pixels with data in a wavelength range (NaN if none)."""
            m = (gwave >= lam_min) & (gwave <= lam_max) & ((mask & 2) == 0)
            return float(np.median(flux[m] / sig[m])) if m.any() else np.nan

        nums = list(frames)
        if jackknife == True:
            # Leave out each frame in turn and report the S/N, for comparison only
            for leave in nums:
                rest = [n for n in nums if n != leave]
                f_i, s_i, m_i, _ = combine(rest)
                log.info(f"S/N at 5500 A: {snr_in(f_i, s_i, m_i, 5450, 5550):.1f} without frame {leave}")

        # The combined spectrum always uses all frames
        comb_flux, comb_sig, comb_mask, n_rej = combine(nums)
        method_desc = {"median": "median", "mean": "plain mean",
                       "weighted": "inverse-variance weighted mean, " +
                                   (f"{self.comb_reject_sigma:g} sigma rejection" if self.comb_reject_sigma > 0
                                    else "no rejection")}
        log.info(f"Combined {len(nums)} frames: {method_desc[self.comb_method]}")
        if self.comb_method == "weighted":
            log.info("Pixels rejected per frame: " + ", ".join(f"{n}: {r}" for n, r in zip(nums, n_rej)))

        # S/N per pixel from the propagated errors, around several wavelengths
        snr5500 = snr_in(comb_flux, comb_sig, comb_mask, 5450, 5550)
        bands = {lam: snr_in(comb_flux, comb_sig, comb_mask, lam - 50, lam + 50)
                 for lam in range(4000, 10000, 500)}
        bands = {lam: v for lam, v in bands.items() if np.isfinite(v)}
        log.info(f"S/N per pixel at 5500 A (median Flux/Sigma, 5450-5550 A): {snr5500:.1f}")
        log.info("S/N per pixel by wavelength: " + ", ".join(f"{lam} A: {v:.0f}" for lam, v in bands.items()))

        # Write the combined spectrum
        out_path = f"{self.target_dir}OUT/{self.objid}_combined.fits"
        t = Table([gwave, comb_flux, comb_sig, comb_mask], names=("Lam", "Flux", "Sigma", "Mask"))
        t["Lam"].unit = "Angstrom"
        t["Flux"].unit = t["Sigma"].unit = "electron"
        t.meta["SCINUMS"] = ",".join(str(n) for n in scinums)
        t.meta["REFFRAME"] = (ref, "frames scaled to this one")
        t.meta["SCALE"] = (self.comb_scale, "frame scaling method")
        t.meta["COMBINE"] = (self.comb_method, "frame combination method")
        if self.comb_method == "weighted":
            t.meta["REJSIG"] = (self.comb_reject_sigma, "rejection threshold, sigma")
            t.meta["NREJECT"] = (int(np.sum(n_rej)), "frame pixels rejected")
        t.meta["SNR5500"] = (snr5500, "median Flux/Sigma, 5450-5550 A")
        t.write(out_path, format="fits", overwrite=True)
        log.info(f"Combined spectrum written to {out_path}")
        self.grapher.plot_combined(gwave, comb_flux, comb_sig, snr5500)

        self.comb_wave = gwave
        self.comb_flux = comb_flux
        self.comb_sigma = comb_sig
        self.comb_mask = comb_mask
        self.comb_ref = ref
        self.snr_bands = bands
        return snr5500

    def load_combined(self):
        """The combined spectrum from this run, or OUT/{objid}_combined.fits from an earlier one."""
        if getattr(self, "comb_flux", None) is not None:
            return
        path = f"{self.target_dir}OUT/{self.objid}_combined.fits"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Combined spectrum not found (run with combine = true first): {path}")
        t = Table.read(path)
        self.comb_wave = np.asarray(t["Lam"], dtype=float)
        self.comb_flux = np.asarray(t["Flux"], dtype=float)
        self.comb_sigma = np.asarray(t["Sigma"], dtype=float)
        self.comb_mask = np.asarray(t["Mask"])
        self.comb_ref = int(t.meta["REFFRAME"])

    def reference_exposure(self):
        """(exptime s, mid-exposure airmass, mid-exposure MJD) of the frame the combined spectrum is scaled to."""
        header = CCDData.read(self.group.science[self.comb_ref], unit="adu").meta
        return exposure_info(header, self.site_height, self.target_ra, self.target_dec)

    def instrument_resolution(self, fcfg):
        """
        KOSMOS resolution (FWHM, A) for comparing a standard with its tabulated flux:
        fcfg["resolution"] if set, else the median arc line FWHM of this object's frames
        (from OUT/{objid}_<n>_extracted.fits; an upper limit for a star smaller than the
        slit), else 3 pixels. Returns (fwhm, where it came from).
        """
        if fcfg["resolution"] > 0:
            return float(fcfg["resolution"]), "[fluxcal] resolution"
        values = []
        for n in self.group.science:
            path = f"{self.target_dir}OUT/{self.objid}_{n}_extracted.fits"
            if os.path.exists(path):
                v = Table.read(path).meta.get("ARCFWHM")
                if v is not None and np.isfinite(v):
                    values.append(float(v))
        if values:
            return float(np.median(values)), "arc line width"
        return 3 * float(np.median(fluxcal.pixel_width(self.comb_wave))), "3 pixels"

    def sensitivity_function(self, std_name, fcfg):
        """
        Sensitivity function from this object's combined spectrum, this object being the
        standard `std_name` (library name, or fcfg["reference"] for the tabulated flux).
        Writes OUT/{objid}_sensfunc.fits and a plot. Returns the fluxcal.sensitivity dict
        plus the standard's name and the reference exposure's MJD.
        """
        self.load_combined()
        exptime, airmass, mjd = self.reference_exposure()
        ref_lam, ref_flux, ref_fwhm, ref_label = fluxcal.load_standard(fcfg["reference"] or std_name)
        ext = fluxcal.load_extinction(fcfg["extinction"])
        inst_fwhm, source = self.instrument_resolution(fcfg)
        sens = fluxcal.sensitivity(self.comb_wave, self.comb_flux, self.comb_sigma, (self.comb_mask & 2) != 0,
                                   exptime, airmass, ext, (ref_lam, ref_flux, ref_fwhm), inst_fwhm, fcfg)
        sens.update(standard=self.objid, reference=ref_label, mjd=mjd, setup=self.group.setup)
        in_range = (ref_lam >= self.comb_wave.min()) & (ref_lam <= self.comb_wave.max())
        log.info(f"Sensitivity function from {self.objid} ({ref_label}; frame {self.comb_ref}, "
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
        t.meta.update({"STANDARD": self.objid, "REFFILE": ref_label, "REFFRAME": self.comb_ref,
                       "INSTFWHM": (inst_fwhm, "A, KOSMOS resolution assumed"),
                       "STDOFF": (sens["offset"], "A, standard's lines minus tabulated"),
                       "EXPTIME": exptime, "AIRMASS": (airmass, "mid-exposure"),
                       "EXTINCT": os.path.basename(str(fcfg["extinction"])),
                       "SENSRMS": (sens["rms"], "mag, RMS of the fit"),
                       "LAMMIN": sens["range"][0], "LAMMAX": sens["range"][1]})
        t.write(f"{self.target_dir}OUT/{self.objid}_sensfunc.fits", format="fits", overwrite=True)
        self.grapher.plot_sensitivity(sens)
        return sens

    def flux_calibrate(self, sens, fcfg):
        """
        Apply a sensitivity function (from sensitivity_function, possibly another
        object's) to this object's combined spectrum. Writes OUT/{objid}_fluxed.fits with
        Lam, Flux, Sigma (erg/s/cm^2/A) and Mask (bit 2 also outside the calibrated range).
        """
        self.load_combined()
        exptime, airmass, _ = self.reference_exposure()
        ext = fluxcal.load_extinction(fcfg["extinction"])
        flux, sig, valid, extrap = fluxcal.calibrate(
            self.comb_wave, self.comb_flux, self.comb_sigma, exptime, airmass, ext, sens["lam"], sens["fit"],
            sens["range"], fcfg["max_extrapolate"])
        mask = self.comb_mask.copy()
        mask[~valid] |= 2 # 2 = no data (outside the calibrated range)
        mask[extrap] |= 8 # 8 = sensitivity extrapolated (linearly) beyond the fitted range
        cal = self.comb_wave[valid]
        msg = (f"Flux calibrated with the sensitivity function of {sens['standard']} "
               f"(frame {self.comb_ref}: {exptime:g} s, airmass {airmass:.3f}); calibrated "
               f"{cal.min():.0f}-{cal.max():.0f} A, {int(extrap.sum())} pixels extrapolated")
        if sens["standard"] == self.objid:
            # Closure test: the standard calibrated with its own sensitivity function
            ratio = flux / sens["ref_flux"]
            ok = valid & ~extrap & sens["used"] & np.isfinite(ratio)
            msg += (f"; closure test: flux / tabulated flux = {np.median(ratio[ok]):.4f} median, "
                    f"{mad_std(ratio[ok]) * 100:.2f}% scatter over the fitted pixels")
        log.info(msg)
        t = Table([self.comb_wave, flux, sig, mask], names=("Lam", "Flux", "Sigma", "Mask"))
        t["Lam"].unit = "Angstrom"
        t["Flux"].unit = t["Sigma"].unit = "erg / (s cm2 Angstrom)"
        t.meta.update({"BUNIT": "erg/s/cm2/A", "STANDARD": sens["standard"], "REFFILE": sens["reference"],
                       "REFFRAME": self.comb_ref, "EXPTIME": exptime, "AIRMASS": (airmass, "mid-exposure"),
                       "SENSRMS": (sens["rms"], "mag, RMS of the sensitivity fit"),
                       "FLUXNOTE": "shape reliable; absolute level depends on slit losses and transparency"})
        t.write(f"{self.target_dir}OUT/{self.objid}_fluxed.fits", format="fits", overwrite=True)
        self.grapher.plot_fluxed(self.comb_wave, flux, sig, valid,
                                 sens["ref_flux"] if sens["standard"] == self.objid else None)
        self.fluxed_flux, self.fluxed_sigma = flux, sig
        return flux, sig

    def normalize_combined(self, verbose):
        """
        Continuum-normalize the combined spectrum (OUT/{objid}_combined.fits, read from
        disk if this run didn't combine) and write OUT/{objid}_normalized.fits with
        Flux and Sigma divided by the continuum, the Continuum itself, Mask, and
        ContFit (1 = pixel used in the continuum fit). Optional step.
        """
        self.load_combined()
        lam, flux, sig, mask = self.comb_wave, self.comb_flux, self.comb_sigma, self.comb_mask.copy()

        log.debug("Fitting the continuum")
        regions = load_mask(self.norm_mask)
        if len(self.norm_extra_mask):
            regions = np.vstack([regions, np.array(self.norm_extra_mask, dtype=float).reshape(-1, 2)])
        cont, used, notes = fit_continuum(lam, flux, sig, (mask & 2) != 0, regions, # bit 4 is informational
                                          knot_spacing=self.knot_spacing, low_reject=self.low_reject,
                                          high_reject=self.high_reject)
        norm_flux = flux / cont
        norm_sig = sig / cont

        # Optional flags on normalized flux (they don't change any values)
        with np.errstate(invalid="ignore"):
            if self.norm_clip is not None:
                mask[norm_flux > self.norm_clip] |= 1
            for lam_min, lam_max, threshold in self.norm_clip_regions:
                mask[(lam >= lam_min) & (lam <= lam_max) & (norm_flux > threshold)] |= 1
        msg = "Continuum: " + "; ".join(notes)
        if self.norm_clip is not None or len(self.norm_clip_regions):
            msg += f"; {int(((mask & 1) != 0).sum())} pixels flagged above the clip thresholds"
        log.info(msg)

        t = Table([lam, norm_flux, norm_sig, cont, mask, used.astype(np.int16)],
                  names=("Lam", "Flux", "Sigma", "Continuum", "Mask", "ContFit"))
        t["Lam"].unit = "Angstrom"
        t["Continuum"].unit = "electron"
        t.meta["KNOTSPAC"] = (self.knot_spacing, "continuum spline knot spacing, A")
        t.meta["LOWREJ"] = (self.low_reject, "continuum clipping below, sigma")
        t.meta["HIGHREJ"] = (self.high_reject, "continuum clipping above, sigma")
        t.write(f"{self.target_dir}OUT/{self.objid}_normalized.fits", format="fits", overwrite=True)
        self.grapher.plot_continuum(lam, flux, cont, used, regions)
        self.grapher.plot_normalized(lam, norm_flux, norm_sig)

        # Normalized combined spectrum
        self.norm_wave = lam
        self.norm_flux = norm_flux
        self.norm_sigma = norm_sig
        self.norm_mask_out = mask
        self.continuum = cont
        self.cont_used = used
    
#-----------------------------------------
    
    def find_coarse_trace(self, sci_final_1, block=100, search=20):
        """
        Rough trace position for every row, from the peak of the median spatial
        profile in blocks of `block` rows. Starts at the middle of the frame and
        walks outward, looking within +/- `search` columns of the previous block's
        peak, so it follows any tilt. Blocks with no clear peak are interpolated over.
        block and search are in unbinned pixels.
        """
        block = detector.to_binned(block, self.biny, minimum=10)
        search = detector.to_binned(search, self.binx, minimum=2)
        n_rows, n_cols = sci_final_1.shape
        starts = np.arange(0, n_rows, block)
        centers = starts + block / 2
        profiles = np.array([np.nanmedian(sci_final_1[s:s+block], axis=0) for s in starts])
        profiles -= np.nanmedian(profiles, axis=1, keepdims=True) # Remove the sky/background level

        if self.trace_col is None:
            # Brightest column in the median spatial profile of the middle half of the frame
            col0 = int(np.nanargmax(np.nanmedian(sci_final_1[n_rows//4 : 3*n_rows//4], axis=0)))
        else:
            col0 = self.trace_col

        def peak_near(k, col):
            lo, hi = max(0, col - search), min(n_cols, col + search + 1)
            seg = profiles[k, lo:hi]
            j = int(np.nanargmax(seg))
            # Require a clear peak (5x the profile's robust scatter)
            return lo + j if seg[j] > 5 * mad_std(profiles[k], ignore_nan=True) else None

        peaks = np.full(len(starts), np.nan)
        k_mid = len(starts) // 2
        for direction in (range(k_mid, len(starts)), range(k_mid - 1, -1, -1)):
            col = col0
            for k in direction:
                p = peak_near(k, col)
                if p is not None:
                    peaks[k] = p
                    col = p

        found = ~np.isnan(peaks)
        if found.sum() < 2:
            raise RuntimeError("KARP could not find the trace; set trace_col in the parameter file")
        return np.round(np.interp(np.arange(n_rows), centers[found], peaks[found])).astype(int)

    def fit_trace(self, sci_final_1, verbose):
        """
        Trace center (column) for every row: Gaussian fits per row around a coarse
        trace, then a polynomial of order self.trace_order fit to the Gaussian
        centers with iterative 3-sigma clipping. Returns (cen_line rounded to whole
        pixels, sub-pixel trace, per-row Gaussian fits [a, mu, sigma, bck]).
        """
        n_rows = sci_final_1.shape[0]
        rows = np.arange(n_rows)
        coarse = self.find_coarse_trace(sci_final_1)

        # Fit a Gaussian to each row around the coarse trace
        log.debug("Fitting the trace")
        fit_args = build_fit_args(sci_final_1, coarse, self.appw, self.buffw, self.bckw)
        cen_fit = np.array(
            Parallel(n_jobs=-1)(delayed(fit_cent_gaussian)(*args) for args in tqdm(fit_args))
            )
        a_vals, mu_vals, sig_vals = cen_fit[:, 0], cen_fit[:, 1], cen_fit[:, 2]

        # Usable fits: converged, centered within the aperture, and not too broad
        ok = (np.isfinite(a_vals) & np.isfinite(mu_vals)
              & (np.abs(mu_vals - coarse) < self.appw) & (sig_vals < self.appw))
        if ok.sum() <= self.trace_order:
            raise RuntimeError("Too few rows with a usable Gaussian fit to fit the trace")

        # Polynomial fit with iterative 3-sigma clipping (rejected rows can come back in)
        used = ok.copy()
        for _ in range(10):
            poly = Polynomial.fit(rows[used], mu_vals[used], self.trace_order)
            resid = mu_vals - poly(rows)
            scatter = mad_std(resid[used])
            new_used = ok & (np.abs(resid) < 3 * scatter)
            if np.array_equal(new_used, used):
                break
            used = new_used

        trace = poly(rows) # Sub-pixel trace center per row
        cen_line = np.round(trace).astype(int)

        msg = (f"Trace: order {self.trace_order} polynomial, {used.sum()} of {n_rows} rows used, "
               f"residual scatter {scatter:.3f} pix, columns {cen_line.min()}-{cen_line.max()}")
        log.info(msg)
        log.debug(f"Trace polynomial coefficients: {poly.convert().coef}")

        return cen_line, trace, cen_fit