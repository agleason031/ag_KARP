"""
Diagnostic and summary plots. Each method writes one figure (or a set) to the path
given by the object's OutputPaths and closes it.
"""
import os

import matplotlib
matplotlib.use("Agg") # Files only; no display needed
import matplotlib.pyplot as plt # noqa: E402
import numpy as np # noqa: E402

BALMER = ((6562.81, "red"), (4861.35, "cyan"), (4340.47, "blueviolet")) # H alpha, beta, gamma


def _save(fig, path, dpi=150):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


class Plotter:
    def __init__(self, objid, out, verbose=False):
        """objid: object name for titles; out: pipeline.OutputPaths; verbose: also make the extra plots."""
        self.objid = objid
        self.out = out
        self.verbose = verbose

    # --- 2D images and per-frame diagnostics --------------------------------

    def image(self, image, title, path):
        fig, ax = plt.subplots(figsize=(6, 10))
        im = ax.imshow(np.asarray(image), origin="lower", aspect="auto")
        fig.colorbar(im, ax=ax)
        ax.set_title(title)
        _save(fig, path)

    def trace(self, columns, frame):
        """Trace column (whole pixels) against row."""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(np.arange(len(columns)), columns, s=3)
        ax.set_xlabel("Row")
        ax.set_ylabel("Column")
        ax.set_title(f"Trace, frame {frame}")
        _save(fig, self.out.frame_file(frame, "trace.png"))

    def trace_fits(self, fits, columns, frame):
        """Per-row Gaussian centers with the fitted trace, and the Gaussian widths."""
        ok = ~np.isnan(fits[:, 0])
        rows = np.arange(len(fits))[ok]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
        ax1.scatter(rows, fits[ok, 1], s=1, color="blue", label="Gaussian center")
        ax1.plot(np.arange(len(columns)), columns, color="black", lw=0.8, label="trace (whole pixels)")
        ax1.set_ylabel("Column")
        ax1.legend()
        ax2.scatter(rows, fits[ok, 2], s=1, color="green")
        ax2.set_ylim(0, np.nanpercentile(fits[ok, 2], 99) * 1.2)
        ax2.set_xlabel("Row")
        ax2.set_ylabel("Gaussian sigma (columns)")
        fig.tight_layout()
        _save(fig, self.out.frame_file(frame, "trace_fits.png"))

    def arc_residuals(self, rows, resid_ang, resid_kms, frame):
        """Residuals of the arc lines kept by the wavelength solution, in A and km/s."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        ax1.scatter(rows, resid_ang, color="blue")
        ax1.set_ylabel("Residual (A)")
        ax2.scatter(rows, resid_kms, color="green")
        ax2.set_ylabel("Residual (km/s)")
        ax2.set_xlabel("Row")
        for ax in (ax1, ax2):
            ax.axhline(0, linestyle="--", color="black")
        ax1.set_title(f"Wavelength solution residuals, frame {frame}")
        fig.tight_layout()
        _save(fig, self.out.frame_file(frame, "wavecal_residuals.png"))

    def extraction_rows(self, diag, frame):
        """
        For a sample of rows: the data across the slit, the sky fit, the aperture and
        background windows, and (optimal extraction) the profile x flux + sky model.
        """
        for r, d in diag.items():
            t = d["trace"]
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.step(d["x"], d["data"], where="mid", color="black", label="data")
            ax.plot(d["x"], d["sky"], color="green", label="sky fit")
            if "model" in d:
                ax.step(d["model_x"], d["model"], where="mid", color="purple", label="optimal model")
            ax.axvline(t, color="black", linestyle="--")
            for edge, color in ((d["half"], "blue"), (d["inner"], "red"), (d["outer"], "red")):
                ax.axvline(t - edge, color=color, linewidth=0.8)
                ax.axvline(t + edge, color=color, linewidth=0.8)
            title = f"Row {r}: boxcar {d['boxcar']:.0f} e-"
            if "optimal" in d:
                title += f", optimal {d['optimal']:.0f} e-"
            ax.set_title(title)
            ax.set_xlabel("Column")
            ax.set_ylabel("Counts (e-)")
            ax.legend()
            _save(fig, self.out.frame_file(frame, os.path.join("extraction_rows", f"row{r:04d}.png")))

    def extracted_flux(self, flux, frame):
        """Extracted flux against row, before the wavelength calibration."""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(np.arange(len(flux)), flux, s=1)
        ax.set_xlabel("Row")
        ax.set_ylabel("Extracted flux (e-)")
        _save(fig, self.out.frame_file(frame, "extracted_flux.png"))

    def frame_spectrum(self, wave, flux, sigma, frame):
        """One frame's wavelength-calibrated spectrum and error, with the Balmer lines marked."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(wave, flux, s=1, label="flux")
        ax.scatter(wave, sigma, s=1, color="red", label="error")
        for lam, color in BALMER:
            ax.axvline(x=lam, color=color, lw=0.8)
        ax.set_xlabel("Wavelength (A)")
        ax.set_ylabel("Electrons")
        ax.set_title(f"{self.objid} frame {frame}")
        ax.legend()
        _save(fig, self.out.frame_file(frame, "spectrum.png"))

    # --- object-level results ------------------------------------------------

    def combined(self, wave, flux, sigma, snr):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(wave, flux, color="black", linewidth=0.6, label="flux")
        ax.plot(wave, sigma, color="red", linewidth=0.6, label="error")
        ax.set_xlabel("Wavelength (A)")
        ax.set_ylabel("Flux (e- per pixel)")
        ax.set_title(f"Combined spectrum for {self.objid}, S/N 5500 A: {snr:.1f}")
        ax.legend()
        fig.tight_layout()
        _save(fig, self.out.product("combined.png"), dpi=200)

    def continuum(self, wave, flux, cont, used, regions):
        """Combined spectrum with the fitted continuum, the pixels the fit used, and masked regions."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(wave, flux, color="0.6", linewidth=0.6, label="flux")
        ax.scatter(wave[used], flux[used], s=0.5, color="black", label="used in fit")
        ax.plot(wave, cont, color="red", linewidth=1.0, label="continuum")
        for lo, hi in regions:
            if lo < np.nanmax(wave) and hi > np.nanmin(wave):
                ax.axvspan(lo, hi, color="blue", alpha=0.1, linewidth=0)
        ax.set_xlim(np.nanmin(wave), np.nanmax(wave))
        ax.set_xlabel("Wavelength (A)")
        ax.set_ylabel("Flux (e- per pixel)")
        ax.set_title(f"Continuum fit for {self.objid} (shaded: masked regions)")
        ax.legend()
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
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(wave, flux, color="black", linewidth=0.6)
        ax.plot(wave, sigma, color="red", linewidth=0.6)
        ax.axhline(1, color="red", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Wavelength (A)")
        ax.set_ylabel("Normalized flux")
        ax.set_ylim(-0.1, 1.5)
        ax.set_title(f"Normalized spectrum for {self.objid}")
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
        fig, (ax, axr) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        ax.plot(lam, s, color="0.6", lw=0.7, label="measured")
        ax.plot(lam[used], s[used], ".", ms=2, color="black", label="used in fit")
        ax.plot(lam, fit, color="red", label="fit")
        for a in (ax, axr):
            for lo, hi in sens["regions"]:
                if hi > lam.min() and lo < lam.max():
                    a.axvspan(lo, hi, color="orange", alpha=0.15, lw=0)
        axr.plot(lam[used], (s - fit)[used], ".", ms=2, color="black")
        axr.axhline(0, color="red")
        ax.set_ylabel("2.5 log10(e-/s/A per erg/s/cm2/A)")
        ax.set_title(f"Sensitivity from {sens['standard']} ({sens['reference']}), RMS {sens['rms']:.4f} mag")
        ax.legend()
        axr.set_xlabel("Wavelength (A)")
        axr.set_ylabel("residual (mag)")
        fig.tight_layout()
        _save(fig, self.out.diagnostic("sensfunc.png"), dpi=200)

    def fluxed(self, wave, flux, sigma, valid, ref_flux=None):
        """Flux-calibrated spectrum, with the tabulated flux for a standard."""
        o = np.argsort(wave)
        wave, flux, sigma, valid = wave[o], flux[o], sigma[o], valid[o]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(wave[valid], flux[valid], color="black", lw=0.7, label="KARP")
        ax.plot(wave[valid], sigma[valid], color="red", lw=0.5, label="error")
        if ref_flux is not None:
            ax.plot(wave[valid], ref_flux[o][valid], color="cyan", lw=0.8, label="tabulated (smoothed)")
        ax.set_xlabel("Wavelength (A)")
        ax.set_ylabel("Flux (erg/s/cm2/A)")
        ax.set_title(f"{self.objid} flux calibrated")
        ax.legend()
        fig.tight_layout()
        _save(fig, self.out.product("fluxed.png"), dpi=200)
