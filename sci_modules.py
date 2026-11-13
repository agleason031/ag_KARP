#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 15:52:33 2025

@author: agleason
"""
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter
from scipy.interpolate import interp1d
from astropy.nddata import CCDData
from astropy.table import Table
from astropy.stats import mad_std
from numpy.polynomial import Polynomial
from functions import fit_cent_gaussian, G, build_fit_args, gaussian_integral_vec
from functions import format_fits_filename, y_lam, con_lam, string_to_bool, optional_float
from sci_tools import heliocentric_correction, fit_all_fluxes
from joblib import Parallel, delayed
import ccdproc
import ast
import os
import warnings
import numpy as np

#-------------------------------------
#helper functions

#-------------------------

class sci_modules:
    def __init__ (self, params, grapher):
        # assign relevant values
        self.dloc = str(params["dloc"]) # Location to store output data
        self.objid = str(params["objid"]) # Object ID
        self.objRA = str(params["objRA"]) # Object RA in the form "05 02 58.72"
        self.objDEC = str(params["objDEC"]) # Object DEC in the form "-70 49 44.7"
        self.otime = str(params["obsTime"]) # The time the object was observed, listed as t = Time("2015-06-30 23:59:60.500")
        self.scistart = int(params["scistart"]) #first sci image
        self.argnum = int(params["argnum"]) # The image number of the Argon spectral calibration image
        self.appw = int(np.round(float(params["appw"])-1)/2) # So an appature width inputu of 13 is two 6 pixel sides, and if you put 14 it rounds to 13
        self.buffw = int(params["buffw"]) # Width of the buffer in pixels (on either side of the appature, so 4 is two, 4 pix buffers su,etrically spaced around the center line)
        self.bckw = int(params["bckw"]) # The width of the background in pixels, (on either side of the buffer)
        self.trace_order = int(params.get("trace_order", 3)) # Polynomial order of the trace fit
        trace_col = params.get("trace_col", "").strip() # Optional starting column for the trace; found automatically if blank
        self.trace_col = int(trace_col) if trace_col else None
        self.target_dir = str(params["target_dir"]) 
        self.norm_line_width = int(params["norm_line_width"])  # When normalizing spectra, this is the one-half width of the line KARP will try and fit at each point
        self.norm_line_boxcar = int(params["norm_line_boxcar"]) # When normalizing the spectra,
        # Cosmic-ray cleaning of the 2D science frames (L.A.Cosmic)
        # Off by default until overscan subtraction is in: without it the background is
        # negative and L.A.Cosmic's noise model flags ~4% of the frame
        self.cr_clean = string_to_bool(params.get("cr_clean", "false"))
        self.readnoise = float(params.get("readnoise", 5.7)) # electrons; 5.7 measured from the 2024-09-29 bias frames
        self.cr_sigclip = float(params.get("cr_sigclip", 4.5)) # detection threshold in sigma
        # Optional masking thresholds (blank or none = off). Masked pixels keep their
        # value but get a nonzero Mask flag in the output tables.
        self.norm_clip = optional_float(params.get("norm_clip")) # normalized flux >= this is masked, per frame
        self.comb_clip = optional_float(params.get("comb_clip")) # combined flux > this is masked
        # List of [lam_min, lam_max, threshold]: combined flux above threshold in that range is masked
        self.comb_clip_regions = ast.literal_eval(params.get("comb_clip_regions", "").strip() or "[]")
        self.gwave = []
        self.med_comb = []
        self.sig_final = []
        
        self.grapher = grapher
        
        self.v_helio = heliocentric_correction(self.objRA, self.objDEC, self.otime)
        self.flux_raw_cor1_dict = {}
        self.wavelengths_dict = {}
        self.sky_raw_dict = {}
        self.sci_final_dict = {}
        self.cen_line_dict = {}
        self.lam_fit_dict = {}
        self.lam_fit_linenum_dict = {}
        self.argon = np.asarray(CCDData.read(format_fits_filename(self.dloc, self.argnum), unit="adu").data)
        
    #does all the image reduction through wavelength calibration
    def reduce_image(self, scinum, global_args):
        sci_list, masterbias, final_flat, verbose = global_args   
    
        sci_location = sci_list[scinum - self.scistart]
        
        # Make a output text file detaling relevant fit paramaters that KARP calculates
        # Initalize directory for reduction plots for this science image
        os.makedirs(self.target_dir+"ImageNumber_"+str(scinum), exist_ok=True) # Initializes a directory for E20 files            
        
        print("KARP is reducing Science Image number:",scinum, flush=True)
        with open(self.target_dir+"ImageNumber_"+str(scinum)+"/KARP_log.txt", "a") as f:
            f.write("KARP Output Log\n")
            f.write("- - - ><(((º>\n")
            f.write("\n")
            f.write("KARP is reducing Science Image number:"+str(scinum)+"\n")
        
        #creates sci_final_1 if not already created
        if scinum in self.sci_final_dict:
            sci_final_1 = self.sci_final_dict[scinum]
        else:
            sci_final = np.asarray((ccdproc.subtract_bias(CCDData.read(sci_location,format = 'fits', unit = "adu"),masterbias).data)/final_flat)
            # Convert from ADU to electrons
            gain = 0.6 # e-/ADU
            sci_final_1 = sci_final*gain

            if self.cr_clean:
                # Data are in electrons and still include sky, as L.A.Cosmic's noise model expects
                sci_final_1, cr_mask = ccdproc.cosmicray_lacosmic(
                    sci_final_1, sigclip=self.cr_sigclip, gain=1.0, readnoise=self.readnoise,
                    satlevel=65535 * gain, verbose=False)
                msg = f"Cosmic rays: {int(cr_mask.sum())} pixels cleaned in image {scinum}"
                print(msg)
                with open(self.target_dir+"ImageNumber_"+str(scinum)+"/KARP_log.txt", "a") as f:
                    f.write(msg + "\n")
            self.sci_final_dict[scinum] = sci_final_1

            # Plot and write to disk only when first created; a cached image was already written
            if (verbose == True):
                self.grapher.plot_sci_final(sci_final, scinum)
                sci_final_write = CCDData(sci_final, unit="adu")
                sci_final_write.write(self.target_dir+"ImageNumber_"+str(scinum)+"/"+"sci_final_"+str(scinum)+".fits", overwrite = True)

    #----------------------------
    #trace fitting
        
        #trace shouldn't change between aperature changes, brightest spot is center and is always in aperature
        #so save and attempt to get from dict to save expensive calculations
        if scinum in self.cen_line_dict:
            cen_line = self.cen_line_dict[scinum]
        else:
            cen_line = self.fit_trace(sci_final_1, verbose) #get center trace
            self.grapher.plot_cen_line(cen_line, scinum)
            self.cen_line_dict[scinum] = cen_line.copy() #save center trace
    #----------------------------------
    #flux extraction
        fit_args = build_fit_args(sci_final_1, cen_line, self.appw, self.buffw, self.bckw)
        sci1_fit = np.array(
            Parallel(n_jobs=-1)(delayed(fit_cent_gaussian)(*args) for args in tqdm(fit_args))
            )
    
        print("KARP finished fitting flux Gaussians")
        
        if (verbose == True):
            print("10 Flux Fit Parameters:")
            with open(self.target_dir+"ImageNumber_"+str(scinum)+"/KARP_log.txt", "a") as f:
                f.write("10 Flux Fit Parameters:"+"\n")
            for i in range(0,4000,400):
                a, mu, sigma, bck = sci1_fit[i]
                print("Y:",i," a:",a," mu:",mu," sig:",sigma," bck:",bck)
                with open(self.target_dir+"ImageNumber_"+str(scinum)+"/KARP_log.txt", "a") as f:
                    f.write("Y:"+str(i)+" a:"+str(a)+" mu:"+str(mu)+" sig:"+str(sigma)+" bck:"+str(bck)+"\n")
            print("Plotting Gauss cen values and Gauss sig values vs Y pix")
            self.grapher.gauss_cen_plots(sci1_fit, cen_line, scinum)        
        
        # Make a plot of 20 evenly spaced pix
        # Showing the data, the apertures and buffers as lines,
        # And the over plotted G fit
        if (verbose == True):
            print("Making 20 aperture plots and aperature fit plots")
            self.grapher.make_aperature_fit_plots(sci1_fit, sci_final_1, cen_line, scinum)
            self.grapher.make_aperature_plots(sci1_fit, sci_final_1, cen_line, scinum)
            
        # Extract spectrum value from each y row
        print("Extracting flux...")
        # Now set down an aparature and get the flux in the center and another aperture to get the background level
        
        #mask out rows with nan gaussian fits
        sci1_fit = np.array(sci1_fit)
        valid_mask = ~np.isnan(sci1_fit[:, 0])  # 'a' values not NaN
        indices = np.where(valid_mask)[0]
        valid_fit = sci1_fit[valid_mask]
        valid_cen = cen_line[valid_mask]
        
        a, mu, sigma, bck = valid_fit.T
        c = valid_cen
        
        # Define fixed offset ranges from each center
        x1_ap = c - self.appw
        x2_ap = c + self.appw
        x1_br = c + self.appw + self.buffw
        x2_br = c + self.appw + self.buffw + self.bckw
        
        ap_flux = gaussian_integral_vec(a, mu, sigma, bck+1000, x1_ap, x2_ap) #adding 1000 to prevent negative bck values which cause the integral to sum the wrong part
        bkg_right = gaussian_integral_vec(a, mu, sigma, bck+1000, x1_br, x2_br)
        flux_valid = ap_flux - (((self.appw * 2) / self.bckw) * (bkg_right)) #note appw is not actual width, it is half width
        
        # Sky background is just the flat background level scaled to aperture width
        sky_valid = bck * (2 * self.appw)
        
        flux_raw1 = np.full_like(cen_line, np.nan, dtype=np.float64)
        flux_raw1[indices] = flux_valid
        sky_raw1 = np.full_like(cen_line, np.nan, dtype=np.float64)
        sky_raw1[indices] = sky_valid
        
        if (verbose == True):
            # Plot output raw flux (We can remove CRs later)
            self.grapher.plot_raw_flux(flux_raw1, scinum)
            for i in range(1, len(flux_valid), 10):
                print(f"Flux fitted value: {flux_valid[i]}")
                if (flux_valid[i] < 0):
                    print(f"Parameters: {a[i]} {mu[i]} {sigma[i]} {bck[i]}")
                    print(f"integrals: {ap_flux[i]} {bkg_right[i]}")
                    print(f"widths: {self.appw*2} {self.bckw}")
                    print(f"{c[i]}")
        
    #--------------------------------------
    #wavelength calibration
        
        pixc = [250,295,765,925,953,1275,1330,1396,1480,2870,3355,3421,3612,3780]
        lam = [6416.31,6384.72,6032.13,5912.09,5888.58,5650.70,5606.73,5558.70,5495.87,4510.73,4200.67,4158.59,4044.42,3948.97]

        # Lambda calibration
        print("Fitting lambda Gaussians...")

        # Drop rows with failed or non-positive extractions, but remember the
        # detector row of every kept pixel so the wavelength solution is
        # evaluated at the right place (NaN > 0 is False, so failed rows drop too)
        n_rows = len(flux_raw1)
        flux_raw1 = np.array(flux_raw1)
        ii = flux_raw1 > 0
        y_pix = np.arange(n_rows)[ii] # Detector row of each kept pixel
        flux_raw_cor1 = flux_raw1[ii]
        sky_raw1 = sky_raw1[ii]

        if scinum in self.lam_fit_dict:
            lam_fit = self.lam_fit_dict[scinum]
            lam_fit_linenum = self.lam_fit_linenum_dict[scinum]
        else:
            lam_fit = [] # Fit parameters [a, mu, sig, bck] of each successfully fit line
            lam_fit_linenum = [] # Lab wavelength of each successfully fit line
            for p, l in zip(pixc, lam):
                # Use the arc column under the trace at this line's row
                popt = self.fit_arc_line(cen_line[p], p, verbose)
                if popt is None:
                    print(f"Arc line {l} A near pixel {p} could not be fit, skipping it")
                    continue
                lam_fit.append(popt)
                lam_fit_linenum.append(l)

            if (verbose == True):
                print("KARP fitted ", len(lam_fit), " lines")
                print("Argon Lines Fit Parameters:")
                with open(self.target_dir+"ImageNumber_"+str(scinum)+"/KARP_log.txt", "a") as f:
                    f.write("KARP fitted"+str(len(lam_fit))+"lines\n")
                    f.write("Argon Lines Fit Parameters:\n")

                s_lsp = [] # Sigmas of the lam line fits for the line split function
                for i in range(len(lam_fit)):
                    a, mu, sigma, bck = lam_fit[i]
                    s_lsp.append(sigma)
                    print("Lam Line Number:",i," lam:",lam_fit_linenum[i]," a:",a," mu:",mu," sig:",sigma," bck:",bck)
                    with open(self.target_dir+"ImageNumber_"+str(scinum)+"/KARP_log.txt", "a") as f:
                        f.write("Lam Line Number:"+str(i)+" lam:"+str(lam_fit_linenum[i])+" a:"+str(a)+" mu:"+str(mu)+" sig:"+str(sigma)+" bck:"+str(bck)+"\n")

                print("Line Split Function:",np.mean(s_lsp))
                print("(avg of sig of line fits)")
                with open(self.target_dir+"ImageNumber_"+str(scinum)+"/KARP_log.txt", "a") as f:
                    f.write("Line Split Function:"+str(np.mean(s_lsp))+"\n")
                    f.write("(avg of sig of line fits)\n")

            self.lam_fit_dict[scinum] = lam_fit
            self.lam_fit_linenum_dict[scinum] = lam_fit_linenum

        cfits = np.array([fit[1] for fit in lam_fit]) # Fitted line centers in pixels
        lam_ok = np.array(lam_fit_linenum) # Lab wavelengths of the same lines

        # Fit a cubic polynomial: line centers (pixels, scaled by 1/2000) -> lab wavelength
        print("Fitting cubic polynomial")
        y_lam_fit = np.polyfit(cfits / 2000, lam_ok, 3)

        # Wavelength of each kept pixel, with heliocentric correction
        wavelengths = y_lam(y_pix, y_lam_fit, self.v_helio)

        if (verbose == True):
            print("v_helio:",self.v_helio,"km/s") # -4 km/s for G093440
            self.grapher.plot_sci_lamcal(y_lam, y_pix, flux_raw_cor1, y_lam_fit, self.v_helio, scinum)

        # Arc solution residuals: lab wavelength minus the fit at the fitted center.
        # Evaluated without the heliocentric correction, since the arc lines are at rest.
        del_lam = lam_ok - y_lam(cfits, y_lam_fit, 0.0)
        res_vel = (del_lam / lam_ok) * 3*10**5 # Residuals in velocity space

        if (verbose == True):
            print("RMS A:",np.sqrt(np.mean(del_lam**2)))
            with open(self.target_dir+"ImageNumber_"+str(scinum)+"/KARP_log.txt", "a") as f:
                f.write("RMS A:"+str(np.sqrt(np.mean(del_lam**2)))+"\n")
        print("Calculating RMS km/s")
        rms_kms = np.sqrt(np.mean(res_vel**2))
        print("RMS km/s:",rms_kms)
        with open(self.target_dir+"ImageNumber_"+str(scinum)+"/KARP_log.txt", "a") as f:
            f.write("RMS km/s:"+str(rms_kms)+"\n")
        # Should be ~10 km/s 0.1 A or even 0.05 A

        if (verbose == True):
            for l, dl, rv in zip(lam_ok, del_lam, res_vel):
                print("Angstrom:",l," residual A:",dl," km/s:",rv)
            print("Plotting Residuals...")

            rnErr = np.sqrt(flux_raw_cor1 + sky_raw1) # Poisson error per pixel

            #make lamcalres plots
            self.grapher.lam_cal_res(cfits, del_lam, scinum)
            self.grapher.lam_cal_res_vel(cfits, res_vel, scinum)

            print("Mean Wavelength residual (delLAM):",np.mean(del_lam))
            print("Mean Velocity residual (delVEL):",np.mean(res_vel))
            with open(self.target_dir+"ImageNumber_"+str(scinum)+"/KARP_log.txt", "a") as f:
                f.write("Mean Wavelength residual (delLAM):"+str(np.mean(del_lam))+"\n")
                f.write("Mean Velocity residual (delVEL):"+str(np.mean(res_vel))+"\n")

            self.grapher.plot_sci_lamcal_res(y_lam, y_pix, flux_raw_cor1, rnErr, y_lam_fit, self.v_helio, scinum)

            # Dispersion over the full detector (wavelength decreases with row number)
            lam_out_max = y_lam(0, y_lam_fit, self.v_helio) # Wavelength at the first row
            lam_out_min = y_lam(n_rows - 1, y_lam_fit, self.v_helio) # Wavelength at the last row

            dispersion = (lam_out_max-lam_out_min)/n_rows
            print("lam_out_max:",lam_out_max)
            print("lam_out_min:",lam_out_min)
            print("Dispersion: ",dispersion, " Angstroms/pix")

            with open(self.target_dir+"ImageNumber_"+str(scinum)+"/KARP_log.txt", "a") as f:
                f.write("lam_out_max:"+str(lam_out_max)+"\n")
                f.write("lam_out_min:"+str(lam_out_min)+"\n")
                f.write("Dispersion:"+str(dispersion)+"Angstroms/pix\n")
            
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
        self.wavelengths_dict[scinum] = wavelengths
    
    def fit_arc_line(self, col, pix, verbose):
        """
        Fit a Gaussian to one arc line in column `col` of the argon frame,
        near detector row `pix`. Tries a narrow window first, then a wider one.
        Returns [a, mu, sigma, bck], or None if both attempts fail.
        """
        a_col = self.argon[:, col]
        for width, maxfev in ((8, 1000), (11, 10000)):
            loc_vals = np.arange(pix - width, pix + width)
            flux_vals = a_col[pix - width : pix + width]
            bck0 = np.median(flux_vals) # Arc is not bias subtracted, so the floor is ~bias level
            p0 = [max(np.max(flux_vals) - bck0, 1.0), pix, 2.0, bck0]
            # Keep the center inside the window so the fit can't wander off to another row
            bounds = [(0, pix - width, 1, -100), (np.inf, pix + width, width, np.inf)] # a, mu, sig, bck
            try:
                popt, _ = curve_fit(G, loc_vals, flux_vals, p0=p0, bounds=bounds, maxfev=maxfev)
                return popt
            except RuntimeError:
                if (verbose == True):
                    print(f"Arc fit failed near pixel {pix} with half-width {width}, maxfev={maxfev}")
        return None

#-----------------------------------------

    def normalization(self, scinum, verbose, optimize):
        # Mask out deep absorption features
        # Deep features in order of ascending wavelength, in angstroms
        # We have spectra down to Lam = 3780 and up to 6600 angstroms
        # H_8,H_eta, H_zeta,Ca_K,Ca_H, H_epsilon,H_delta,H_gamma, H_beta, SodiumD2, SodiumD1, H_alpha
        features = [3796.94,3835.40,3889.06,3933.7,3969,3970.075,4101.73,4340.47,4861.35,5889.96,5895.93,6562.81]
        
        flux_raw_cor1 = list(self.flux_raw_cor1_dict[scinum])
        wavelengths = list(self.wavelengths_dict[scinum])
        sky_raw1 = list(self.sky_raw_dict[scinum])
        
        # Remove 4 pixels from either edge to ensure bad chip pixels aren't affecting our fits
        flux_raw_cor1 = flux_raw_cor1[3:-4]
        wavelengths = wavelengths[3:-4]
        sky_raw1 = sky_raw1[3:-4]
        flux_masked = np.array(flux_raw_cor1.copy()) # Create a copy so we can still use flux_raw_cor1
        flux_raw_cor1_np = np.array(flux_raw_cor1, dtype=float)
        wavelengths_np = np.array(wavelengths, dtype=float)
        sky_raw1 = np.array(sky_raw1)
        
        print(f"Estimating continuum on {scinum}")
        # Make each line have its own feature mask width to get a good mask
        # H_8,H_eta, H_zeta,Ca_K,Ca_H, H_epsilon,H_delta,H_gama, H_beta, SodiumD2, SodiumD1, H_alpha
        
        # Wide features, for bright spectra
        feature_mask = [10,15,19,10,10,35,40,60,60,10,15,30]
    
        for i in range(len(features)):
            # Find pixel index closest to the feature wavelength
            wavelengths = np.array(wavelengths)
            # use np.argmin to get the closest pixel value to our feature
            center = np.argmin(np.abs(wavelengths - features[i])) # argmin so we round to the closest indexed value, not skipping over wavelengths that we have pixels for
            # Get the maximum and minimum values (up to 40 pix away from center)
            start = max(0, center - feature_mask[i]) # Not 40 pixels, 40*1/*0.7 Ang/pix so like 57
            stop = min(len(flux_masked), center + feature_mask[i])
            flux_masked[start:stop] = np.nan
        
        # Fit a fourth degree polynomial to the continuum estimate
        flux_mask_red = []
        lam_red = []
        flux_mask_remove = []
        lam_remove = []
        removed_index = [] # 0 is keep and 1 is remove
        # reduce the flux for a better fit
        for i in range(len(flux_masked)):
            if np.isnan(flux_masked[i]) != True:
                # Pixels we want to keep
                flux_mask_red.append(flux_masked[i])
                lam_red.append(wavelengths[i])
                removed_index.append(0) # The index values of the pixels that we keep for fitting later
            else:
                # Pixels that we take out
                flux_mask_remove.append(flux_raw_cor1[i])
                removed_index.append(1) # The index values of the pixels that we remove for fitting later
                lam_remove.append(wavelengths[i])
        
        # Fit a continuum to out spectra:
        fit_params = np.polyfit(lam_red,flux_mask_red,6)    
        Fnorm_Es = flux_raw_cor1_np / con_lam(wavelengths_np, fit_params) # Divide raw flux by our continuum that we just fit
        
        # Plot our first estimate on the normalized flux
        if (verbose == True):
            con_lam_test = []
            for i in range(len(wavelengths)):
                con_lam_test.append(con_lam(wavelengths_np[i], fit_params)) 
            self.grapher.plot_sci_flux_norm_est(wavelengths, Fnorm_Es, scinum)
            self.grapher.plot_sci_flux_masked(lam_red, lam_remove, flux_mask_red, flux_mask_remove, wavelengths, con_lam_test, scinum)
        
        print("Line-fit normilization")
        removed_index = np.array(removed_index)
        
        fitted_values = []
        fitted_wavelengths = []
        fitted_sky = []
        fitted_fluxes = []
        
        flux_masked_global = flux_raw_cor1_np.copy()
        for i, feat in enumerate(features):
            center = np.argmin(np.abs(wavelengths_np - feat)) # Use argmin to get the index of the minimum value of all_wavelengths-all_features, (i.e. where the center of each feature is)
            start = max(0, center - feature_mask[i]) # From the center out to our feature mask size, start as far out from the edge as we can
            stop = min(len(flux_masked_global), center + feature_mask[i]) # From the center out to our feature mask size,  start as close to the edge as we can
            flux_masked_global[start:stop] = np.nan # Mask absorption lines with NANs
        
        lam_reda = [] # Kept flux values that do not have Fnorm_es > 1.3
        flux_mask_reda = []
        lam_removea = [] # Removed flux values that do not have Fnorm_es > 1.3
        flux_mask_removea = []
        
        rm_param = 1.3
        include = Fnorm_Es <= rm_param
        
        results = fit_all_fluxes(Fnorm_Es, wavelengths_np, flux_masked, include, removed_index,
                           flux_raw_cor1_np, flux_masked_global, sky_raw1, self.norm_line_width)
        for row in results:        
            if np.isnan(row[0]):
                continue
            else:
                val, sky, flux, wave, f_keep, f_remove, l_keep, l_remove = row
                fitted_values.append(val)
                fitted_sky.append(sky)
                fitted_fluxes.append(flux)
                fitted_wavelengths.append(wave)
                if f_keep is not None:
                    flux_mask_reda.append(f_keep)
                    lam_reda.append(l_keep)
                if f_remove is not None:
                    flux_mask_removea.append(f_remove)
                    lam_removea.append(l_remove)
            
        # Now smooth this fit with a moving boxcar
        smooth_cont = uniform_filter(np.array(fitted_values), size=self.norm_line_boxcar)
        
        if optimize == False:
            # Plot fitted flux values from the line fit to see what is going on
            self.grapher.plot_fit_over_flux(fitted_wavelengths, smooth_cont, fitted_values, fitted_fluxes, lam_removea, flux_mask_removea, lam_red, lam_remove, flux_mask_red, flux_mask_remove, lam_reda, flux_mask_reda, scinum)
                
        Fnorm = np.array(fitted_fluxes) / smooth_cont # Divide raw flux by our continuum that we just fit

        # Flag (don't overwrite) pixels above the optional threshold
        mask = np.zeros(len(Fnorm), dtype=np.int16) # 0 = good, 1 = above norm_clip
        if self.norm_clip is not None:
            flagged = Fnorm >= self.norm_clip
            mask[flagged] = 1
            print(f"Masked {int(flagged.sum())} pixels with normalized flux >= {self.norm_clip} in image {scinum}")
            if verbose:
                print("Masked values:", Fnorm[flagged])

        sf_sm = np.sqrt(np.array(fitted_fluxes) + np.array(fitted_sky)) / smooth_cont
        
        if (verbose == True):
            print("Plotting Normalized Spectra")
            self.grapher.plot_sci_normalized(fitted_wavelengths, Fnorm, sf_sm, scinum)
        
        # Write a file for the spectra of this sci image
        # In the form G123456_61_OUT.fits in the OUT folder
        t = Table([fitted_wavelengths,Fnorm,sf_sm,mask], names=('Lam', "Flux", "Sigma", "Mask"))
        t.write(self.target_dir+"OUT/"+str(self.objid)+"_"+str(scinum)+"_OUT.fits", format='fits', overwrite=True)

#---------------------------------------------
    
    def combine_spectra(self, scinums, verbose, jackknife=False):
        """
        Median-combine the normalized spectra of the given science images
        (OUT/{objid}_{n}_OUT.fits) on the wavelength grid of the first one,
        write the result to OUT/{objid}_combined.fits, and return the S/N at 5500 A.
        With jackknife=True, also report the S/N with each image left out in turn.
        """
        print("KARP is combining spectra")
        log_path = self.target_dir+"OUT/KARP_OUT_log.txt"
        with open(log_path, "w") as fo:
            fo.write("KARP is combining spectra\n")

        # Only this object's spectra for the requested images
        file_names = [f"{self.target_dir}OUT/{self.objid}_{n}_OUT.fits" for n in scinums]
        missing = [f for f in file_names if not os.path.exists(f)]
        if missing:
            raise FileNotFoundError(f"Normalized spectra not found (run with spec_norm = true first): {missing}")
        if (verbose == True):
            print("Using these file names:",file_names)
            with open(log_path, "a") as fo:
                fo.write("Using these file names:"+str(file_names)+"\n")

        # Use wavelength grid from first file
        gwave = np.asarray(Table.read(file_names[0])["Lam"])

        def combine(files):
            """
            Median-combine `files` on gwave. Returns (med_comb, comb_mask, gsig, rms).
            comb_mask bits: 1 = above a comb_clip threshold, 2 = no unmasked data.
            """
            # Interpolate onto reference grid; masked pixels are left out of the median
            gflux_interp = []
            gsig = []
            for fname in files:
                t = Table.read(fname)
                flux = np.array(t["Flux"], dtype=float)
                if "Mask" in t.colnames:
                    flux[np.asarray(t["Mask"]) != 0] = np.nan
                f = interp1d(t["Lam"], flux, bounds_error=False, fill_value=np.nan)
                s = interp1d(t["Lam"], t["Sigma"], bounds_error=False, fill_value=np.nan)
                gflux_interp.append(f(gwave))
                gsig.append(s(gwave))

            # Stack and take median, ignoring NaNs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning) # All-NaN columns are flagged below
                med_comb = np.nanmedian(np.array(gflux_interp), axis=0)

            # Flag (don't overwrite) values above the optional thresholds,
            # e.g. the zeroth order or remaining bad pixels
            comb_mask = np.zeros(len(gwave), dtype=np.int16)
            with np.errstate(invalid="ignore"):
                if self.comb_clip is not None:
                    comb_mask[med_comb > self.comb_clip] |= 1
                for lam_min, lam_max, threshold in self.comb_clip_regions:
                    comb_mask[(gwave >= lam_min) & (gwave <= lam_max) & (med_comb > threshold)] |= 1
            comb_mask[np.isnan(med_comb)] |= 2

            print("Computing RMS")
            # Compute RMS in 5450–5550 from unmasked pixels
            mask = (gwave >= 5450) & (gwave <= 5550) & (comb_mask == 0)
            rms = np.sqrt(np.mean((1 - med_comb[mask])**2))
            return med_comb, comb_mask, np.array(gsig), rms

        if jackknife == True:
            # Leave out each file in turn and report the S/N, for comparison only
            for i in range(len(file_names)):
                selected_files = [f for j, f in enumerate(file_names) if j != i]
                _, _, _, rms_i = combine(selected_files)
                print("SNR at 5500:",1/rms_i,f"with files {selected_files}")
                with open(log_path, "a") as fo:
                    fo.write("SNR at 5500:"+str(1/rms_i)+"\n")
                    fo.write(f"with files {selected_files}\n")

        # The combined spectrum always uses all files
        med_comb, comb_mask, gsig, rms = combine(file_names)

        # Calculate and print SNR at 5500A
        print("SNR at 5500:",1/rms)
        with open(log_path, "a") as fo:
            fo.write("SNR at 5500:"+str(1/rms)+"\n")

        # Calculate sig_final(lam), should be 1.2,1.3
        print("Calculating sig_final(lam)")

        # Avoid division by zero or NaNs
        inv_var_array = np.where(gsig > 0, 1.0 / gsig**2, 0)

        # Sum over spectra (axis=0 is spectra, axis=1 is wavelength, so we sum over 0)
        sum_sigw = np.sum(inv_var_array, axis=0)  # shape: (N_wavelength,)

        with np.errstate(divide="ignore"): # Wavelengths with no valid sigma give inf
            sig_w = np.sqrt(1/sum_sigw)

        # Get the closest sig_w to 5500 angstroms
        sig_w55_idx = np.argmin(np.abs(gwave - 5550))
        sig_w55 = sig_w[sig_w55_idx]

        # Compute sig_final
        sig_final = rms * sig_w/sig_w55

        if (verbose == True):
            print("(RMS/sig_w55)",(rms/sig_w55))
            with open(log_path, "a") as fo:
                fo.write("(RMS/sig_w55)"+str(rms/sig_w55)+"\n")

        # Write the combined spectrum
        out_path = f"{self.target_dir}OUT/{self.objid}_combined.fits"
        t = Table([gwave, med_comb, sig_final, comb_mask], names=("Lam", "Flux", "Sigma", "Mask"))
        t.meta["SCINUMS"] = ",".join(str(n) for n in scinums)
        t.meta["SNR5500"] = 1/rms
        t.write(out_path, format="fits", overwrite=True)
        print("Combined spectrum written to", out_path)

        self.grapher.plot_combined_norm(gwave, med_comb, sig_final, rms)

        self.gwave = gwave
        self.med_comb = med_comb
        self.sig_final = sig_final
        self.comb_mask = comb_mask

        return 1/rms
    
#-----------------------------------------
    
    def compare_cen_lines(self, scinum1, scinum2):
        if scinum1 not in self.cen_line_dict or scinum2 not in self.cen_line_dict:
            print("One or both science image centerlines not found.")
            return None
    
        cl1 = self.cen_line_dict[scinum1]
        cl2 = self.cen_line_dict[scinum2]
    
        min_len = min(len(cl1), len(cl2))
        cl1 = cl1[:min_len]
        cl2 = cl2[:min_len]
    
        delta = cl2 - cl1
        abs_diff = np.abs(delta)
        mean_diff = np.mean(abs_diff)
        max_diff = np.max(abs_diff)
    
        print(f"Comparison between image {scinum1} and {scinum2}:")
        print(f"  Mean absolute difference: {mean_diff:.3f} pixels")
        print(f"  Max absolute difference: {max_diff:.3f} pixels")
    
        return delta
    
    def find_coarse_trace(self, sci_final_1, block=100, search=10):
        """
        Rough trace position for every row, from the peak of the median spatial
        profile in blocks of `block` rows. Starts at the middle of the frame and
        walks outward, looking within +/- `search` columns of the previous block's
        peak, so it follows any tilt. Blocks with no clear peak are interpolated over.
        """
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
        centers with iterative 3-sigma clipping. Rounded to whole pixels.
        """
        n_rows = sci_final_1.shape[0]
        rows = np.arange(n_rows)
        coarse = self.find_coarse_trace(sci_final_1)

        # Fit a Gaussian to each row around the coarse trace
        print("KARP fitting centerline")
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

        cen_line = np.round(poly(rows)).astype(int)

        msg = (f"Trace: order {self.trace_order} polynomial, {used.sum()} of {n_rows} rows used, "
               f"residual scatter {scatter:.3f} pix, columns {cen_line.min()}-{cen_line.max()}")
        print(msg)
        if (verbose == True):
            print("Trace polynomial coefficients:", poly.convert().coef)

        return cen_line