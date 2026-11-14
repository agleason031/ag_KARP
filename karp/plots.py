"""
Diagnostic and summary plots. Each method writes one figure (or a set) to the path
given by the object's OutputPaths and closes it.

Always made: the trace, wavelength solution and spectrum of each frame; the combined
spectrum and the frame-to-frame comparison; the continuum and sensitivity fits and the
normalized and fluxed spectra when those steps run. With verbose = true also: 2D images
(calibration frames, each science frame with its extraction windows), cuts across the
slit, and zoomed views of the continuum fit and normalized spectrum.
"""
import os

import matplotlib
matplotlib.use("Agg") # Files only; no display needed
import matplotlib.pyplot as plt # noqa: E402
import numpy as np # noqa: E402
from astropy.visualization import ZScaleInterval # noqa: E402
from scipy.ndimage import median_filter # noqa: E402

WIDE = (11, 6) # spectra
TALL = (9, 9) # stacked panels


def _save(fig, path, dpi=150):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _zscale(image):
    """Display limits that show the noise and faint structure (zscale, as in ds9/IRAF)."""
    finite = np.asarray(image)[np.isfinite(image)]
    return ZScaleInterval().get_limits(finite) if finite.size else (0, 1)


def _sorted(wave, *arrays):
    o = np.argsort(wave)
    return (wave[o],) + tuple(a[o] for a in arrays)


class Plotter:
    def __init__(self, objid, out, verbose=False):
        """objid: object name for titles; out: pipeline.OutputPaths; verbose: also make the extra plots."""
        self.objid = objid
        self.out = out
        self.verbose = verbose

    # --- 2D images -----------------------------------------------------------

    def image(self, image, title, path, units="e-"):
        """A 2D frame (rows up, columns across), zscale stretch."""
        vmin, vmax = _zscale(image)
        fig, ax = plt.subplots(figsize=(6, 10))
        im = ax.imshow(np.asarray(image), origin="lower", aspect="auto", cmap="gray", vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, label=units)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        ax.set_title(title)
        fig.tight_layout()
        _save(fig, path)

    def frame_image(self, image, trace, half, buffer, background, frame):
        """
        A calibrated science frame with the trace, the aperture (blue) and the background
        windows (red); full frame and a zoom around the trace.
        """
        rows = np.arange(len(trace))
        edges = [(half + 0.5, "cyan"), (half + 0.5 + buffer, "red"), (half + 0.5 + buffer + background, "red")]
        lo = max(0, int(np.nanmin(trace) - edges[-1][0] - 10))
        hi = min(image.shape[1], int(np.nanmax(trace) + edges[-1][0] + 10))
        vmin, vmax = _zscale(image[:, lo:hi])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 10), gridspec_kw={"width_ratios": [1, 1.4]})
        for ax, (c0, c1) in ((ax1, (0, image.shape[1])), (ax2, (lo, hi))):
            ax.imshow(image, origin="lower", aspect="auto", cmap="gray", vmin=vmin, vmax=vmax,
                      extent=(-0.5, image.shape[1] - 0.5, -0.5, image.shape[0] - 0.5))
            ax.set_xlim(c0 - 0.5, c1 - 0.5)
            ax.set_xlabel("Column")
        for offset, color in edges:
            for sign in (-1, 1):
                ax2.plot(trace + sign * offset, rows, color=color, lw=1.0)
        ax2.plot(trace, rows, color="yellow", lw=0.6, ls="--")
        ax1.axvspan(lo, hi, color="yellow", alpha=0.15)
        ax1.set_ylabel("Row")
        ax1.set_title(f"Frame {frame}")
        ax2.set_title("trace (yellow), aperture (cyan), background (red)")
        fig.tight_layout()
        _save(fig, self.out.frame_file(frame, "image.png"))

    # --- per-frame diagnostics -----------------------------------------------

    def slit_sources(self, profile, sources, chosen, center, frame):
        """The median profile across the slit with the stars found in it, numbered as --pick lists them."""
        cols = np.arange(len(profile))
        fig, ax = plt.subplots(figsize=WIDE)
        ax.plot(cols, profile, color="black", lw=0.8)
        ax.axvline(center, color="gray", ls=":", lw=1, label=f"slit center ({center})")
        ax.axvline(chosen, color="C0", lw=1.5, alpha=0.6, label=f"trace start ({chosen})")
        for i, (c, peak) in enumerate(sources, start=1):
            ax.plot(c, peak, "v", color="C3", ms=8)
            ax.annotate(f"{i}: {c}", (c, peak), textcoords="offset points", xytext=(0, 8), ha="center",
                        color="C3")
        finite = profile[np.isfinite(profile)]
        if finite.size:
            top = max([p for _, p in sources] + [finite.max()])
            ax.set_ylim(np.percentile(finite, 1) - 0.05 * top, 1.2 * top)
        ax.set_xlabel("Column (binned)")
        ax.set_ylabel("Median across rows, background removed (e-)")
        ax.set_title(f"Stars in the slit, frame {frame}: {len(sources)} found")
        ax.legend(loc="upper right")
        fig.tight_layout()
        _save(fig, self.out.frame_file(frame, "slit.png"))

    def trace(self, fits, trace, used, frame):
        """Gaussian centers with the trace fit, the residuals (rejected rows marked), and the profile width."""
        rows = np.arange(len(trace))
        ok = np.isfinite(fits[:, 1])
        resid = fits[:, 1] - trace
        rejected = ok & ~used
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=TALL, sharex=True,
                                            gridspec_kw={"height_ratios": [2, 1, 1]})
        ax1.plot(rows[ok], fits[ok, 1], ".", ms=1, color="C0", label="Gaussian center")
        ax1.plot(rows, trace, color="black", lw=1, label="trace fit")
        ax1.set_ylabel("Column")
        ax1.legend(loc="best")
        ax2.plot(rows[used], resid[used], ".", ms=1, color="C0")
        ax2.plot(rows[rejected], resid[rejected], "x", ms=3, color="C3", label=f"rejected ({rejected.sum()})")
        ax2.axhline(0, color="black", lw=0.8)
        scatter = np.nanstd(resid[used])
        ax2.set_ylim(-8 * scatter, 8 * scatter)
        ax2.set_ylabel("Residual (columns)")
        ax2.legend(loc="upper right")
        ax3.plot(rows[ok], 2.3548 * fits[ok, 2], ".", ms=1, color="C2")
        ax3.set_ylim(0, np.nanpercentile(2.3548 * fits[ok, 2], 99) * 1.3)
        ax3.set_ylabel("Profile FWHM (columns)")
        ax3.set_xlabel("Row")
        ax1.set_title(f"Trace, frame {frame}: {used.sum()} of {len(rows)} rows used, "
                      f"residual scatter {scatter:.3f} columns")
        fig.tight_layout()
        _save(fig, self.out.frame_file(frame, "trace.png"))

    def wavecal(self, sol, arc_spec, frame):
        """
        The arc spectrum against wavelength with the matched lines (green: used, red:
        rejected by the fit), and the residuals of the matched lines in A and km/s.
        """
        lam_rows = sol.poly(np.arange(len(arc_spec), dtype=float))
        line_lam = sol.poly(sol.rows) # fitted position of each matched line
        resid = sol.residuals
        kms = resid / sol.lam * 2.99792458e5
        used = sol.used
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 10), sharex=True,
                                            gridspec_kw={"height_ratios": [2, 1, 1]})
        # Log scale, so the faint lines show next to the bright ones
        floor = max(np.nanpercentile(np.abs(arc_spec), 50), 1.0)
        ax1.plot(lam_rows, np.clip(arc_spec, floor, None), color="black", lw=0.6)
        ymax = np.nanmax(arc_spec)
        for lam, row, u in zip(sol.lam, sol.rows, used):
            color = "C2" if u else "C3"
            r = int(round(row))
            height = max(arc_spec[r], floor) if 0 <= r < len(arc_spec) else ymax
            ax1.plot([lam, lam], [height * 1.3, height * 3], color=color, lw=0.8)
            ax1.text(lam, height * 3.3, f"{lam:.1f}", rotation=90, fontsize=6, ha="center", va="bottom", color=color)
        ax1.set_yscale("log")
        ax1.set_ylim(floor, ymax * 100)
        ax1.set_ylabel("Arc (e-)")
        for ax, values, label in ((ax2, resid, "Residual (A)"), (ax3, kms, "Residual (km/s)")):
            # Scale to the used lines; rejected lines beyond the axis are drawn at its edge as triangles
            lim = 5 * np.max(np.abs(values[used])) if used.any() else np.max(np.abs(values))
            inside = np.abs(values) <= lim
            ax.plot(line_lam[used], values[used], "o", color="C2", ms=4, label="used")
            ax.plot(line_lam[~used & inside], values[~used & inside], "x", color="C3", ms=5, label="rejected")
            for sign, marker in ((1, "^"), (-1, "v")):
                off = ~used & (sign * values > lim)
                ax.plot(line_lam[off], np.full(off.sum(), sign * 0.95 * lim), marker, color="C3", ms=5)
            ax.axhline(0, color="black", lw=0.8, ls="--")
            ax.set_ylabel(label)
            ax.set_ylim(-1.05 * lim, 1.05 * lim)
        ax2.legend(loc="upper right", fontsize=8, title="triangles: beyond the axis", title_fontsize=7)
        ax3.set_xlabel("Wavelength (A)")
        ax1.set_title(f"Wavelength solution, frame {frame}: order {sol.poly.degree()}, {used.sum()} of "
                      f"{len(sol.rows)} lines used, RMS {sol.rms_ang:.3f} A = {sol.rms_kms:.1f} km/s")
        fig.tight_layout()
        _save(fig, self.out.frame_file(frame, "wavecal.png"))

    def extraction_rows(self, diag, frame):
        """
        For a sample of rows: the data across the slit, the sky fit, the aperture (blue)
        and background windows (red), and (optimal extraction) the profile x flux + sky model.
        """
        for r, d in diag.items():
            t = d["trace"]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.axvspan(t - d["half"], t + d["half"], color="C0", alpha=0.12, lw=0)
            for sign in (-1, 1):
                ax.axvspan(t + sign * d["inner"], t + sign * d["outer"], color="C3", alpha=0.12, lw=0)
            ax.step(d["x"], d["data"], where="mid", color="black", lw=0.8, label="data")
            ax.plot(d["x"], d["sky"], color="C2", label="sky fit")
            if "model" in d:
                ax.step(d["model_x"], d["model"], where="mid", color="C4", lw=0.8, label="optimal model")
            ax.axvline(t, color="black", linestyle="--", lw=0.6)
            title = f"Frame {frame}, row {r}: boxcar {d['boxcar']:.0f} e-"
            if "optimal" in d:
                title += f", optimal {d['optimal']:.0f} e-"
            ax.set_title(title)
            ax.set_xlabel("Column")
            ax.set_ylabel("e-")
            ax.legend(fontsize=8)
            fig.tight_layout()
            _save(fig, self.out.frame_file(frame, os.path.join("extraction_rows", f"row{r:04d}.png")))

    def frame_spectrum(self, wave, flux, sigma, sky, frame):
        """One frame's extracted spectrum with its error, and the sky that was subtracted."""
        wave, flux, sigma, sky = _sorted(wave, flux, sigma, sky)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=TALL, sharex=True, gridspec_kw={"height_ratios": [2, 1]})
        ax1.plot(wave, flux, color="black", lw=0.6, label="flux")
        ax1.plot(wave, sigma, color="C3", lw=0.6, label="error")
        ax1.set_ylabel("e- per pixel")
        ax1.legend(loc="best", fontsize=8)
        ax2.plot(wave, sky, color="C0", lw=0.6)
        ax2.set_ylabel("Sky under aperture (e-)")
        ax2.set_xlabel("Wavelength (A)")
        ax1.set_title(f"{self.objid} frame {frame}")
        fig.tight_layout()
        _save(fig, self.out.frame_file(frame, "spectrum.png"))

    # --- object-level results ------------------------------------------------

    def combined(self, comb, snr):
        """The combined spectrum with its error band, and the S/N per pixel."""
        wave, flux, sigma = comb.wave, comb.flux, comb.sigma
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=TALL, sharex=True, gridspec_kw={"height_ratios": [2, 1]})
        ax1.fill_between(wave, flux - sigma, flux + sigma, color="C3", alpha=0.4, lw=0, label="±1 sigma")
        ax1.plot(wave, flux, color="black", lw=0.6, label="flux")
        ax1.set_ylabel("e- per pixel")
        ax1.legend(loc="best", fontsize=8)
        with np.errstate(invalid="ignore", divide="ignore"):
            ax2.plot(wave, flux / sigma, color="C0", lw=0.6)
        ax2.axhline(snr, color="black", ls="--", lw=0.8, label=f"S/N at 5500 A: {snr:.1f}")
        ax2.set_ylim(bottom=0)
        ax2.set_ylabel("S/N per pixel")
        ax2.set_xlabel("Wavelength (A)")
        ax2.legend(loc="best", fontsize=8)
        ax1.set_title(f"Combined spectrum for {self.objid} ({len(comb.frames)} frames, "
                      f"scaled to frame {comb.ref_frame})")
        fig.tight_layout()
        _save(fig, self.out.product("combined.png"), dpi=200)

    def frame_comparison(self, comb):
        """Each frame (scaled to the reference) divided by the combined spectrum, smoothed: any frame that departs stands out."""
        if not comb.scaled:
            return
        fig, ax = plt.subplots(figsize=WIDE)
        with np.errstate(invalid="ignore", divide="ignore"):
            for n, (flux, _) in comb.scaled.items():
                ratio = flux / comb.flux
                good = np.isfinite(ratio)
                smooth = np.full_like(ratio, np.nan)
                smooth[good] = median_filter(ratio[good], size=15, mode="nearest")
                ax.plot(comb.wave, smooth, lw=0.8, label=f"frame {n}" + (" (reference)" if n == comb.ref_frame else ""))
        ax.axhline(1, color="black", lw=0.8)
        ax.set_ylim(0.9, 1.1)
        ax.set_xlabel("Wavelength (A)")
        ax.set_ylabel("Frame / combined (15-pixel median)")
        ax.set_title(f"{self.objid}: frames compared with the combined spectrum")
        ax.legend(fontsize=8)
        fig.tight_layout()
        _save(fig, self.out.diagnostic("frames.png"))

    def continuum(self, wave, flux, cont, used, regions):
        """Combined spectrum with the fitted continuum, the pixels the fit used, and masked regions."""
        fig, ax = plt.subplots(figsize=WIDE)
        ax.plot(wave, flux, color="0.6", linewidth=0.6, label="flux")
        ax.scatter(wave[used], flux[used], s=0.5, color="black", label="used in fit")
        ax.plot(wave, cont, color="C3", linewidth=1.0, label="continuum")
        for lo, hi in regions:
            if lo < np.nanmax(wave) and hi > np.nanmin(wave):
                ax.axvspan(lo, hi, color="C0", alpha=0.1, linewidth=0)
        ax.set_xlim(np.nanmin(wave), np.nanmax(wave))
        ax.set_xlabel("Wavelength (A)")
        ax.set_ylabel("e- per pixel")
        ax.set_title(f"Continuum fit for {self.objid} (shaded: masked regions)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        os.makedirs(self.out.diagnostics, exist_ok=True)
        fig.savefig(self.out.diagnostic("continuum.png"), dpi=200)
        if self.verbose:
            # Zoomed views, 500 A wide
            for start in np.arange(np.floor(np.nanmin(wave) / 100) * 100, np.nanmax(wave), 400):
                m = (wave >= start) & (wave <= start + 500)
                if m.any():
                    ax.set_xlim(start, start + 500)
                    ax.set_ylim(0, 1.15 * np.nanmax(np.r_[flux[m], cont[m]]))
                    path = self.out.diagnostic(os.path.join("continuum_zoom", f"continuum_{int(start)}.png"))
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    fig.savefig(path, dpi=200)
        plt.close(fig)

    def normalized(self, wave, flux, sigma):
        fig, ax = plt.subplots(figsize=WIDE)
        ax.plot(wave, flux, color="black", linewidth=0.6, label="flux")
        ax.plot(wave, sigma, color="C3", linewidth=0.6, label="error")
        ax.axhline(1, color="C3", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Wavelength (A)")
        ax.set_ylabel("Normalized flux")
        ax.set_ylim(-0.1, 1.5)
        ax.set_title(f"Normalized spectrum for {self.objid}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(self.out.product("normalized.png"), dpi=200)
        if self.verbose:
            for start in np.arange(np.floor(np.nanmin(wave) / 100) * 100, np.nanmax(wave), 400):
                ax.set_xlim(start, start + 500)
                path = self.out.diagnostic(os.path.join("normalized_zoom", f"normalized_{int(start)}.png"))
                os.makedirs(os.path.dirname(path), exist_ok=True)
                fig.savefig(path, dpi=200)
        plt.close(fig)

    def sensitivity(self, sens):
        """Measured sensitivity (mag), the pixels used, the masked regions, and the fit."""
        lam, s, fit, used = sens["lam"], sens["sens"], sens["fit"], sens["used"]
        fig, (ax, axr) = plt.subplots(2, 1, figsize=TALL, sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        ax.plot(lam, s, color="0.6", lw=0.7, label="measured")
        ax.plot(lam[used], s[used], ".", ms=2, color="black", label="used in fit")
        ax.plot(lam, fit, color="C3", label="fit")
        for a in (ax, axr):
            for lo, hi in sens["regions"]:
                if hi > lam.min() and lo < lam.max():
                    a.axvspan(lo, hi, color="C1", alpha=0.15, lw=0)
        axr.plot(lam[used], (s - fit)[used], ".", ms=2, color="black")
        axr.axhline(0, color="C3")
        ax.set_ylabel("2.5 log10(e-/s/A per erg/s/cm2/A)")
        ax.set_title(f"Sensitivity from {sens['standard']} ({sens['reference']}): RMS {sens['rms']:.4f} mag, "
                     f"offset {sens['offset']:+.2f} A")
        ax.legend(fontsize=8)
        axr.set_xlabel("Wavelength (A)")
        axr.set_ylabel("Residual (mag)")
        fig.tight_layout()
        _save(fig, self.out.diagnostic("sensfunc.png"), dpi=200)

    def fluxed(self, wave, flux, sigma, valid, extrapolated, ref_flux=None):
        """Flux-calibrated spectrum (shaded: sensitivity extrapolated), with the tabulated flux for a standard."""
        ref = ref_flux if ref_flux is not None else np.full(len(wave), np.nan)
        wave, flux, sigma, valid, extrapolated, ref = _sorted(wave, flux, sigma, valid, extrapolated, ref)
        fig, ax = plt.subplots(figsize=WIDE)
        mid = np.median(wave)
        for side in (extrapolated & (wave < mid), extrapolated & (wave >= mid)): # blue and red ends
            if side.any():
                ax.axvspan(wave[side].min(), wave[side].max(), color="C1", alpha=0.15, lw=0)
        ax.plot(wave[valid], flux[valid], color="black", lw=0.7, label="KARP")
        ax.plot(wave[valid], sigma[valid], color="C3", lw=0.5, label="error")
        if ref_flux is not None:
            ax.plot(wave[valid], ref[valid], color="C0", lw=0.8, label="tabulated (matched resolution)")
        ax.set_xlabel("Wavelength (A)")
        ax.set_ylabel("Flux (erg/s/cm2/A)")
        title = f"{self.objid} flux calibrated"
        if extrapolated.any():
            title += " (shaded: sensitivity extrapolated)"
        ax.set_title(title)
        ax.legend(fontsize=8)
        fig.tight_layout()
        _save(fig, self.out.product("fluxed.png"), dpi=200)
