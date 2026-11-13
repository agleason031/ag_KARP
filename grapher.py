'''
created by Alex Gleason

aux grapher module for KARP
'''

#import packages
import matplotlib.pyplot as plt
import numpy as np
import os

# Define a Gaussian
#different from functions.G bc no njit
def G(x, a, mu, sigma, bck):
	return (a * np.exp(-(x-mu)**2/(2*sigma**2))) + bck
	# A 4d Gaussian
    
class grapher:
    def __init__ (self, cfg, objid, target_dir):
        self.objid = objid # Object ID
        self.target_dir = target_dir # Output directory for this object (ends in "/")
        self.verbose = cfg["output"]["verbose"]
        ext = cfg["extraction"]
        self.appw = int(np.round(float(ext["aperture"])-1)/2) # So an appature width inputu of 13 is two 6 pixel sides, and if you put 14 it rounds to 13
        self.buffw = int(ext["buffer"]) # Width of the buffer in pixels (on either side of the appature, so 4 is two, 4 pix buffers su,etrically spaced around the center line)
        self.bckw = int(ext["background"]) # The width of the background in pixels, (on either side of the buffer)

    #fit individual plots for each metal line
    def metal_line_plt(self, metal_fits, gwave, masks, med_comb, lines, center_shifts):
        for i, popt in enumerate(metal_fits):
            plt.clf()
            fig, ax = plt.subplots(1, 1, figsize=(8,6))
            ax.scatter(gwave,med_comb, s=5,color="black")
            ax.plot(gwave,G(gwave,popt[0],popt[1],popt[2], popt[3]))
            ax.axhline(1,color="red",linestyle="--")
            ax.axvline(popt[1],color="black",linestyle="--")
            ax.axvline(lines[i],color="blue",linestyle="--")
            ax.axvline(lines[i]+center_shifts[i], color="green", linestyle="--")
            ax.set_xlim(int(lines[i]-5+center_shifts[i]),int(lines[i]+5)+center_shifts[i])
            ax.set_xlabel("Wavelength (A)")
            ax.set_ylabel("Median Normalized Flux")
            ax.set_title("Metal line fit at "+str(lines[i])+" A")
            
            #set y scale for each plot
            ydata = med_comb[masks[i]]
            try: #safeguard for data out of range
                ymin = np.nanmin(ydata)
                ymax = np.nanmax(ydata)
            except ValueError:
                ymin = 1
                ymax = 1
            ax.set_ylim(min(.9, ymin-0.05), max(ymax+0.05, 1.1))
            
            plt.tight_layout(rect=[0.06, 0.06, 1, 1])  # Leave space for global labels
            plt.savefig(self.target_dir + f"OUT/{lines[i]}_metal_line.png", dpi=300)
            ax.clear()

    def metal_line_big_plt(self, metal_fits, metal_lines, masks, gwave, med_comb, center_shifts):
        n_lines = len(metal_lines)
        ncols = 4
        nrows = int(np.ceil(n_lines / ncols))
        fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*3), sharey=False)
        axs = axs.flatten()
    
        for i, popt in enumerate(metal_fits):
            ax = axs[i]
            ax.scatter(gwave, med_comb, s=5, color="black")
            ax.plot(gwave, G(gwave, *popt), color="orange")
            ax.axhline(1, color="red", linestyle="--", linewidth=0.8)
            ax.axvline(popt[1], color="blue", linestyle="--", linewidth=0.8)
            ax.axvline(metal_lines[i],color="blue",linestyle="--")
            ax.set_xlim(int(metal_lines[i]-5+center_shifts[i]),int(metal_lines[i]+5)+center_shifts[i])
            ax.set_title(f"{metal_lines[i]:.1f} Å")
            # Remove individual axis labels for clarity
            ax.set_xlabel("")
            ax.set_ylabel("")
            #set y scale for each plot
            ydata = med_comb[masks[i]]
            try: #safeguard for data out of range
                ymin = np.nanmin(ydata)
                ymax = np.nanmax(ydata)
            except ValueError:
                ymin = 1
                ymax = 1
            ax.set_ylim(min(.9, ymin-0.05), max(ymax+0.05, 1.1))
        
        # Hide unused subplots
        for j in range(i+1, len(axs)):
            axs[j].axis('off')
    
        fig.text(0.55, 0.04, "Wavelength (A)", ha='center', fontsize=16)
        fig.text(0.04, 0.5, "Normalized Flux", va='center', rotation='vertical', fontsize=16)
    
        plt.tight_layout(rect=[0.06, 0.06, 1, 1])  # Leave space for global labels
        plt.savefig(self.target_dir + f"OUT/{self.objid}_metal_lines_all.png", dpi=300)
        plt.close(fig)
    
    def plot_vel_lines(self, vel_fit, vel_fit_linenum, vel_lines, gwave, med_comb, vel_mask):
        for i in range(len(vel_fit)):
            a, mu, sig, bck = vel_fit[i]
            print("For line:",vel_fit_linenum[i]+1,"KARP fit:",mu,"Angstroms")
            print("a:",a,"sig:",sig,"bck:",bck)
            print("(lam-lam_0)/lam_0:",((float(mu)-float(vel_lines[vel_fit_linenum[i]]))/float(vel_lines[vel_fit_linenum[i]])),"A")
            print("mu in km/s:",((float(mu)-float(vel_lines[vel_fit_linenum[i]]))/float(vel_lines[vel_fit_linenum[i]]))*3*10**5)
            
            plt.cla()
            fig, axVel = plt.subplots(1, 1, figsize=(8,6))
            axVel.scatter(gwave,med_comb, s=5,color="black")
            axVel.plot(gwave,G(gwave,a,mu,sig,bck))
            axVel.axhline(1,color="red",linestyle="--")
            axVel.axvline(mu,color="black",linestyle="--")
            axVel.axvline(vel_lines[vel_fit_linenum[i]],color="blue",linestyle="--")
            axVel.set_xlim(int(vel_lines[vel_fit_linenum[i]]-vel_mask[i]),int(vel_lines[vel_fit_linenum[i]]+vel_mask[i]))
            axVel.set_xlabel("Wavelength (A)")
            axVel.set_ylabel("Median Normalized Flux")
            del_vel_lam = np.round(float(mu)-float(vel_lines[vel_fit_linenum[i]]),decimals=4)
            axVel.set_title("lam-lam_0: "+str(del_vel_lam)+" A")
            if i >= 8:
                axVel.set_ylim(0.9,1.1)
                axVel.set_title("HeI "+str(vel_lines[vel_fit_linenum[i]])+"A line lam-lam_0: "+str(del_vel_lam)+" A radV: "+str(((float(mu)-float(vel_lines[vel_fit_linenum[i]]))/float(vel_lines[vel_fit_linenum[i]]))*3*10**5)+"km/s")
            
            plt.savefig(self.target_dir+"OUT/"+self.objid+"_Vel_"+str(i+1)+".png", dpi=300)
        
    def plot_image(self, image, name):        
        plt.clf()
        plt.imshow(image)
        plt.colorbar()
        plt.title(name + ' Image')
        plt.savefig(self.target_dir+"OUT/"+self.objid+"_"+name+".png")
        
    def plot_sci_final(self, sci_final, scinum):
        plt.clf()
        plt.imshow(sci_final)
        plt.colorbar()
        plt.title('Final Science ' + str(scinum))
        plt.savefig(self.target_dir+"ImageNumber_"+str(scinum)+"/"+self.objid+"_final_sci" + str(scinum)+".png")
        print("Master Science Number: " + str(scinum) + " Made")
        
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
    
            # ---- Plot 2: sigma vs index ----
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            ax2.scatter(indices, sigmas, color='green')
            ax2.tick_params(labelsize=14)
            ax2.set_xlabel("Y Pix", size=14)
            ax2.set_ylabel("G fit Sig", size=14)
            fig2.tight_layout()
            fig2.savefig(os.path.join(save_dir, f"Sci_Gsig_y_{scinum}.png"))
    
            # ---- Plot 3: zoomed-in sigma plot ----
            fig3, ax3 = plt.subplots(figsize=(8, 6))
            ax3.scatter(indices, sigmas, color='green')
            ax3.tick_params(labelsize=14)
            ax3.set_xlabel("Y Pix", size=14)
            ax3.set_ylabel("G fit Sig", size=14)
            ax3.set_ylim(0, 2)
            fig3.tight_layout()
            fig3.savefig(os.path.join(save_dir, f"Sci_Gsig_y_ZOOM{scinum}.png"))
            
    def lam_cal_res(self, pixc, del_lam, scinum):
        fig, axL = plt.subplots(1, 1, figsize=(8,6))
        
        axL.scatter(pixc,del_lam,color="blue")
        axL.axhline(0,linestyle="--",color="black")
        axL.set_xlabel("Y Pixel Value")
        axL.set_ylabel("Angstroms")
        
        plt.savefig(self.target_dir+"ImageNumber_"+str(scinum)+"/"+"LamCalRes_"+str(scinum)+".png")
        plt.close(fig)
        
    def lam_cal_res_vel(self, pixc, res_vel, scinum):
        plt.cla()
        fig, axV = plt.subplots(1, 1, figsize=(8,6))
        axV.scatter(pixc,res_vel,color="green")
        axV.axhline(0,linestyle="--",color="black")
        axV.set_xlabel("Y Pixel Value")
        axV.set_ylabel("km/s")
        
        plt.savefig(self.target_dir+"ImageNumber_"+str(scinum)+"/"+"LamCalRes_Velocity_"+str(scinum)+".png")
        plt.close(fig)
    
    def plot_cen_line(self, cen_line, scinum):
        fig, axcs = plt.subplots(1, 1, figsize=(8,6))
        censcat = np.arange(0,len(cen_line),1)
        axcs.scatter(censcat,cen_line, s=3)
        axcs.set_xlabel("Y Pixel")
        axcs.set_ylabel("X Pixel")
        plt.savefig(self.target_dir+"ImageNumber_"+str(scinum)+"/"+"cen_line_"+str(scinum)+".png")
        #print("len(cen_line)",len(cen_line))
        
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
        plt.cla()
        y_pix = np.arange(0,len(flux_raw1),1)
        plt.scatter(y_pix,flux_raw1, s=1)
        plt.xlabel("Y Pixel")
        plt.ylabel("Background subtracted Flux")
        plt.savefig(self.target_dir+"ImageNumber_"+str(scinum)+"/"+"Sci_Flux_bsub_"+str(scinum)+".png", dpi=300)
        
    def plot_sci_lamcal(self, y_lam, y_pix, flux_raw_cor1, fit, v_helio, scinum):
        fig, ax2 = plt.subplots(1, 1, figsize=(8,6))
        ax2.scatter(y_lam(y_pix, fit, v_helio),flux_raw_cor1, s=1)
        ax2.axvline(x=6562.81,color='red')
        ax2.axvline(x=4861.35,color='cyan')
        ax2.axvline(x=4340.47,color='blueviolet')
        plt.xlabel("Wavelength (A)")
        plt.ylabel("Background subtracted Flux (Electrons)")
        plt.ylim(0,30000)
        plt.title("Sci Image"+str(scinum))
        plt.savefig(self.target_dir+"ImageNumber_"+str(scinum)+"/"+"Sci_Flux_LamCal_"+str(scinum)+".png", dpi=300)
    
    def plot_sci_lamcal_res(self, y_lam, y_pix, flux_raw_cor1, rnErr, fit, v_helio, scinum):
        plt.cla() # Clear plt to prevent over plotting
        fig, axE = plt.subplots(1, 1, figsize=(8,6))
        axE.scatter(y_lam(y_pix, fit, v_helio), flux_raw_cor1, s=1)
        axE.scatter(y_lam(y_pix, fit, v_helio), rnErr, s=1,color='red')
        axE.set_xlabel("Wavelength (A), Error (10x)")
        axE.set_ylabel("Background subtracted Flux (Electrons)")
        plt.savefig(self.target_dir+"ImageNumber_"+str(scinum)+"/"+"Sci_Flux_LamCal_Res_"+str(scinum)+".png", dpi=300)   
        
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
