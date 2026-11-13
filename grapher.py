'''
created by Alex Gleason

aux grapher module for KARP
'''

#import packages
import matplotlib
matplotlib.use("Agg") # Files only; no display needed
import matplotlib.pyplot as plt
import numpy as np
import os

class grapher:
    def __init__ (self, cfg, objid, target_dir):
        self.objid = objid # Object ID
        self.target_dir = target_dir # Output directory for this object (ends in "/")
        self.verbose = cfg["output"]["verbose"]
        ext = cfg["extraction"]
        self.appw = int(np.round(float(ext["aperture"])-1)/2) # So an appature width inputu of 13 is two 6 pixel sides, and if you put 14 it rounds to 13
        self.buffw = int(ext["buffer"]) # Width of the buffer in pixels (on either side of the appature, so 4 is two, 4 pix buffers su,etrically spaced around the center line)
        self.bckw = int(ext["background"]) # The width of the background in pixels, (on either side of the buffer)

    def _image(self, image, title, path):
        fig, ax = plt.subplots(figsize=(6, 10))
        im = ax.imshow(image, origin="lower", aspect="auto")
        fig.colorbar(im, ax=ax)
        ax.set_title(title)
        fig.savefig(path)
        plt.close(fig)

    def plot_image(self, image, name):
        self._image(np.asarray(image), name + " Image", self.target_dir + "OUT/" + self.objid + "_" + name + ".png")

    def plot_sci_final(self, sci_final, scinum):
        self._image(sci_final, "Final Science " + str(scinum),
                    self.target_dir + "ImageNumber_" + str(scinum) + "/" + self.objid + "_final_sci" + str(scinum) + ".png")

    def gauss_cen_plots(self, sci1_fit, cen_line, scinum):
    
        # ---- Ensure target directory exists ----
        save_dir = os.path.join(self.target_dir, f"ImageNumber_{scinum}")
        os.makedirs(save_dir, exist_ok=True)
    
        # ---- Extract valid data in one pass ----
        valid_data = [(i, mu, sigma) for i, (a, mu, sigma, bck) in enumerate(sci1_fit) if not np.isnan(a)]
    
        if valid_data:
            indices, mus, sigmas = zip(*valid_data)
            
            # ---- Plot 1: mu vs index with cen_line ----
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            ax1.scatter(indices, mus, s=1, color='blue', label='mu')
            ax1.scatter(np.arange(len(cen_line)), cen_line, s=3, color='black', label='cen_line')
            ax1.tick_params(labelsize=14)
            ax1.set_xlabel("Y Pix", size=14)
            ax1.set_ylabel("G fit mu", size=14)
            ax1.legend()
            fig1.tight_layout()
            fig1.savefig(os.path.join(save_dir, f"Sci_Gcen_y_Cenline_{scinum}.png"))
            plt.close(fig1)
    
            # ---- Plot 2: sigma vs index ----
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            ax2.scatter(indices, sigmas, color='green')
            ax2.tick_params(labelsize=14)
            ax2.set_xlabel("Y Pix", size=14)
            ax2.set_ylabel("G fit Sig", size=14)
            fig2.tight_layout()
            fig2.savefig(os.path.join(save_dir, f"Sci_Gsig_y_{scinum}.png"))
            plt.close(fig2)
    
            # ---- Plot 3: zoomed-in sigma plot ----
            fig3, ax3 = plt.subplots(figsize=(8, 6))
            ax3.scatter(indices, sigmas, color='green')
            ax3.tick_params(labelsize=14)
            ax3.set_xlabel("Y Pix", size=14)
            ax3.set_ylabel("G fit Sig", size=14)
            ax3.set_ylim(0, 2)
            fig3.tight_layout()
            fig3.savefig(os.path.join(save_dir, f"Sci_Gsig_y_ZOOM{scinum}.png"))
            plt.close(fig3)
            
    def lam_cal_res(self, pixc, del_lam, scinum):
        fig, axL = plt.subplots(1, 1, figsize=(8,6))
        
        axL.scatter(pixc,del_lam,color="blue")
        axL.axhline(0,linestyle="--",color="black")
        axL.set_xlabel("Y Pixel Value")
        axL.set_ylabel("Angstroms")
        
        plt.savefig(self.target_dir+"ImageNumber_"+str(scinum)+"/"+"LamCalRes_"+str(scinum)+".png")
        plt.close(fig)
        
    def lam_cal_res_vel(self, pixc, res_vel, scinum):
        fig, axV = plt.subplots(1, 1, figsize=(8,6))
        axV.scatter(pixc,res_vel,color="green")
        axV.axhline(0,linestyle="--",color="black")
        axV.set_xlabel("Y Pixel Value")
        axV.set_ylabel("km/s")
        
        plt.savefig(self.target_dir+"ImageNumber_"+str(scinum)+"/"+"LamCalRes_Velocity_"+str(scinum)+".png")
        plt.close(fig)
    
    def plot_cen_line(self, cen_line, scinum):
        fig, axcs = plt.subplots(1, 1, figsize=(8,6))
        axcs.scatter(np.arange(len(cen_line)), cen_line, s=3)
        axcs.set_xlabel("Y Pixel")
        axcs.set_ylabel("X Pixel")
        fig.savefig(self.target_dir+"ImageNumber_"+str(scinum)+"/"+"cen_line_"+str(scinum)+".png")
        plt.close(fig)

    def plot_extraction_rows(self, diag, scinum):
        """
        For a sample of rows: the data across the slit, the sky fit, the aperture and
        background windows, and (optimal extraction) the profile x flux + sky model.
        """
        save_dir = os.path.join(self.target_dir, f"ImageNumber_{scinum}", f"Extraction_{scinum}")
        os.makedirs(save_dir, exist_ok=True)
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
            ax.set_xlabel("X pixels")
            ax.set_ylabel("Counts (e-)")
            ax.legend()
            fig.savefig(os.path.join(save_dir, f"Extraction_row{r}.png"))
            plt.close(fig)
            
    def plot_raw_flux(self, flux_raw1, scinum):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(np.arange(len(flux_raw1)), flux_raw1, s=1)
        ax.set_xlabel("Y Pixel")
        ax.set_ylabel("Background subtracted Flux (Electrons)")
        fig.savefig(self.target_dir+"ImageNumber_"+str(scinum)+"/"+"Sci_Flux_bsub_"+str(scinum)+".png", dpi=300)
        plt.close(fig)

    def plot_sci_lamcal(self, y_lam, y_pix, flux_raw_cor1, fit, v_helio, scinum):
        fig, ax2 = plt.subplots(1, 1, figsize=(8,6))
        ax2.scatter(y_lam(y_pix, fit, v_helio), flux_raw_cor1, s=1)
        for lam, color in ((6562.81, "red"), (4861.35, "cyan"), (4340.47, "blueviolet")): # H alpha, beta, gamma
            ax2.axvline(x=lam, color=color)
        ax2.set_xlabel("Wavelength (A)")
        ax2.set_ylabel("Background subtracted Flux (Electrons)")
        ax2.set_title("Sci Image " + str(scinum))
        fig.savefig(self.target_dir+"ImageNumber_"+str(scinum)+"/"+"Sci_Flux_LamCal_"+str(scinum)+".png", dpi=300)
        plt.close(fig)

    def plot_sci_lamcal_res(self, y_lam, y_pix, flux_raw_cor1, rnErr, fit, v_helio, scinum):
        fig, axE = plt.subplots(1, 1, figsize=(8,6))
        lam = y_lam(y_pix, fit, v_helio)
        axE.scatter(lam, flux_raw_cor1, s=1, label="flux")
        axE.scatter(lam, rnErr, s=1, color="red", label="error")
        axE.set_xlabel("Wavelength (A)")
        axE.set_ylabel("Electrons")
        axE.legend()
        fig.savefig(self.target_dir+"ImageNumber_"+str(scinum)+"/"+"Sci_Flux_LamCal_Res_"+str(scinum)+".png", dpi=300)
        plt.close(fig)

    def plot_combined(self, lam, flux, sigma, snr):
        """Combined (not normalized) spectrum and its error."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(lam, flux, color="black", linewidth=0.6, label="flux")
        ax.plot(lam, sigma, color="red", linewidth=0.6, label="error")
        ax.set_xlabel("Wavelength (A)")
        ax.set_ylabel("Flux (e- per pixel)")
        ax.set_title(f"Combined spectrum for {self.objid}, S/N 5500 A: {snr:.1f}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.target_dir + "OUT/" + self.objid + "_combined.png", dpi=200)
        plt.close(fig)

    def plot_continuum(self, lam, flux, cont, used, regions):
        """Combined spectrum with the fitted continuum, the pixels the fit used, and masked regions."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(lam, flux, color="0.6", linewidth=0.6, label="flux")
        ax.scatter(lam[used], flux[used], s=0.5, color="black", label="used in fit")
        ax.plot(lam, cont, color="red", linewidth=1.0, label="continuum")
        for lo, hi in regions:
            if lo < np.nanmax(lam) and hi > np.nanmin(lam):
                ax.axvspan(lo, hi, color="blue", alpha=0.1, linewidth=0)
        ax.set_xlim(np.nanmin(lam), np.nanmax(lam))
        ax.set_xlabel("Wavelength (A)")
        ax.set_ylabel("Flux (e- per pixel)")
        ax.set_title(f"Continuum fit for {self.objid} (shaded: masked regions)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.target_dir + "OUT/" + self.objid + "_continuum.png", dpi=200)
        if self.verbose:
            # Zoomed views, 500 A wide
            os.makedirs(self.target_dir + "OUT/continuum_zoom", exist_ok=True)
            for start in np.arange(np.floor(np.nanmin(lam) / 100) * 100, np.nanmax(lam), 400):
                ax.set_xlim(start, start + 500)
                m = (lam >= start) & (lam <= start + 500)
                if m.any():
                    ax.set_ylim(0, 1.15 * np.nanmax(np.r_[flux[m], cont[m]]))
                    fig.savefig(self.target_dir + f"OUT/continuum_zoom/{self.objid}_continuum_{int(start)}.png", dpi=200)
        plt.close(fig)

    def plot_normalized(self, lam, flux, sigma):
        """Normalized combined spectrum and its error."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(lam, flux, color="black", linewidth=0.6)
        ax.plot(lam, sigma, color="red", linewidth=0.6)
        ax.axhline(1, color="red", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Wavelength (A)")
        ax.set_ylabel("Normalized flux")
        ax.set_ylim(-0.1, 1.5)
        ax.set_title(f"Normalized spectrum for {self.objid}")
        fig.tight_layout()
        fig.savefig(self.target_dir + "OUT/" + self.objid + "_normalized.png", dpi=200)
        if self.verbose:
            os.makedirs(self.target_dir + "OUT/normalized_zoom", exist_ok=True)
            for start in np.arange(np.floor(np.nanmin(lam) / 100) * 100, np.nanmax(lam), 400):
                ax.set_xlim(start, start + 500)
                fig.savefig(self.target_dir + f"OUT/normalized_zoom/{self.objid}_normalized_{int(start)}.png", dpi=200)
        plt.close(fig)
