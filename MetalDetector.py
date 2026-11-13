#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 14:27:34 2025

@author: chasesmith

Metal Detector to add to KARP

Model spectra at https://pollux.oreme.org/explore/AMBRE/
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import integrate
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.stats import sem  # standard error of the mean
import csv
import sys

if __name__ == "__main__":
    input_object = sys.argv[1]
    logg = str(sys.argv[2])
    Teff = int(sys.argv[3])
    input_path = f"/d/col1/agleason/carp/reductions/{input_object}/OUT/{input_object}_metal_lines.csv"
    #input_path = "/d/com1/rcilley/rcilley_KARP/G093440/RC_RED_G093440/OUT/G093440EQ_Width_Table.csv"
    #input_path = "/d/users/agleason/Downloads/G750016_kaitlyn_lines.csv"

#Read in synthetic spectra
# List of lines
lam=[3820.425,3933.660,4045.812,4063.594,4226.728,4260.474,4271.760,4307.902,4383.545,4404.750,4957.596,5167.321,5172.684,5183.604,5269.537,5328.038]
num_lam=len(lam)

Teff_model=[4750, 5000, 5250, 5500, 5750, 6000, 6250, 6500, 6750, 7000, 7250]
FeH_model=[-5.0, -4.0, -3.0,-2.5,-2.0, -1.00]

#EW_model = np.empty()

Teffname=['4750', '5000', '5250', '5500', '5750','6000','6250','6500','6750', '7000', '7250']
FeHname=['-5.00', '-4.00', '-3.00','-2.50','-2.00', '-1.00']
gname = ['3.0','3.5','4.0','4.5']

EW_model = np.arange(len(Teffname) * len(FeHname) * num_lam, dtype=np.float32).reshape(len(Teffname), len(FeHname), num_lam)

metal_mask = []
with open(input_path) as csvfile:
    print(csvfile)
    reader = csv.DictReader(csvfile, skipinitialspace=True)
    for row in reader:
        value = float(row["Width of Mask"])
        metal_mask.append(value)

for i in range(0,len(Teffname),1):
    for j in range(0,len(FeHname),1):
        try:
            file='/d/col1/agleason/carp/my_spectra/M_p'+Teffname[i]+'g'+logg+'z'+FeHname[j]+'t1.0_a0.00c0.00n0.00o0.00r0.00s0.00_VIS.spec.FITS'
            test_metal = fits.open(file)
        except FileNotFoundError:
            try:
                file='/d/col1/agleason/carp/my_spectra/M_s'+Teffname[i]+'g'+logg+'z'+FeHname[j]+'t1.0_a0.00c0.00n0.00o0.00r0.00s0.00_VIS.spec.FITS'
                test_metal = fits.open(file)
            except FileNotFoundError:
                file='/d/col1/agleason/carp/my_spectra/M_s'+Teffname[i]+'g'+logg+'z'+FeHname[j]+'t2.0_a0.00c0.00n0.00o0.00r0.00s0.00_VIS.spec.FITS'
                test_metal = fits.open(file)
        #test_metal.info()
        image_data = test_metal[1].data
        #print(type(image_data))
        #print(image_data.shape)
        wavelength = image_data['wavelength']
        flux = image_data['flux']
        norm_flux = image_data['normalized flux']
        #plt.title("T: "+str(Teffname[i])+" g: "+str(gname[0])+" Z: "+str(FeH_model[j]))
        #plt.scatter(wavelength,norm_flux,s=0.1)
        #plt.xlabel("Wavelength")
        #plt.show()
        
        for k in range(len(lam)):
            wavelength = np.array(wavelength)
            norm_flux = np.array(norm_flux)
            ind = (wavelength > (lam[k]-metal_mask[k])) & (wavelength < (lam[k]+metal_mask[k]))
            #if k == 1: # A wider width for k
                #ind = (wavelength > (lam[k]-5)) & (wavelength < (lam[k]+5))
            
            #print(lam[k],":")
            EW = integrate.simpson(1-norm_flux[ind],wavelength[ind])
            #print(EW)
            #print("For:",Teffname[i]," ", FeHname[j])
            EW_model[i][j][k]=EW
            
            #plt.title("LAM T: "+str(Teffname[i])+" g: "+str(gname[0])+" Z: "+str(FeH_model[j]))
            #plt.scatter(wavelength[ind],norm_flux[ind],s=0.1)
            #plt.xlabel("Wavelength")
            #plt.show()
            
            
            
            
        """
        image_data = test_metal[1].data
        print(type(image_data))
        print(image_data.shape)
        """
'''
#print(np.shape(EW_model))

# Measure EW from the models


# Interpolate the grid at the Teff of the object


# Max's values
EW_max = np.array([0.152, 1.392, 0.102, 0.089, 0.119, 0.035, 0.050, 0.107, 0.082, 0.043, 0.029,0.129, 0.104, 0.111, 0.057, 0.047])  # EW for G093440

FeH_fine = np.arange(101) * 0.02 - 4.0  # 101 points from -4.0 to 1.0

# EM G080960 (Known4) -2.3
#EW = np.array([0.00799,2.17118,0.22827,0.20463,0.08896,0.07922,0.04235,0.25761,0.11348,0.01848,0.00368,0.21689,0.19622,0.18076,0.14759,0.11637])


# RC G080960 (Known4) -2.3
EW = np.array([0.15640272466711900,1.5003931355901500,0.08735957862449970,0.13405644522899,0.24996536268538900,0.09705887530831600,0.14659508130625200,0.14740546705235000,0.15263927642668700,0.1101551985792020,0.09280700100292290,0.1470360050929930,0.12916816831806300,0.17319182782902300,0.1058697126549980,0.07270530853702050])


# GN G080960 (Known4) -2.3
#EW = np.array([5.26066205026247E-09,1.6966453339285100,1.7858553711964E-06,2.83331769200252E-05,0.07576740327849370,0.02587756794373440,0.012751584788323000,0.21448190147508300,0.08663309282039400,0.026668615201617700,0.012658352345568800,0.21044186756286500,0.6651146703257960,0.1503873619799220,0.07926364085341490,0.13277045004172600])


# AG G080960 (Known4) -2.3
#EW = np.array([0.15015325604840400,1.5149007175284500,0.09007341093909370,0.08431050407674120,0.24949148924138500,0.08025474929787490,0.2450182991139730,0.12190755176619400,0.14871476056511500,0.08956270177155450,0.08324614225795290,0.14746323937462000,0.15552662162906500,0.16310386854728500,0.11443651863879000,0.21087954116075400])


#AG and RC for G750016

#RC Ca only
#EW = [0.758,0.063]
'''
EW = []

with open(input_path) as csvfile:
    print(csvfile)
    reader = csv.DictReader(csvfile, skipinitialspace=True)
    for row in reader:
        value = float(row["Adopted_EW"])
        EW.append(value)

print('EWs')
print(EW)

FeH_fine = np.arange(101) * 0.04 - 5.0  # 101 points from -5.0 to 1.0

# 1. Interpolate Teff
valx = interp1d(Teff_model, np.arange(len(Teffname)), kind='linear', fill_value='extrapolate')(Teff)


# Above is the closest "ith" temprature to the input Teff
# 2. Construct fine EW grid
EW_fine = np.zeros((101, num_lam))

# 3. Loop to interpolate EW_model grid over valx, valy for each wavelength
for i in range(101):
    valy = interp1d(FeH_model, np.arange(len(FeHname)), kind='linear', fill_value='extrapolate')(FeH_fine[i])
    # Loops through the FeH_models and finds the closes value
    for j in range(num_lam):
        # Interpolating over a 2D grid: EW_model[:,:,j] shape (3,4)
        # For each EW (:, :, j) 
        interp_func = RegularGridInterpolator(
            (np.arange(len(Teffname)), np.arange(len(FeHname))), EW_model[:, :, j], method='linear', bounds_error=False, fill_value=None
        )
        EW_fine[i, j] = interp_func([[valx, valy]])[0]

# 4. Now measure [M/H] per line by interpolating the fine grid
MH_line = np.zeros(num_lam)
for i in range(num_lam):
    interp_func = interp1d(EW_fine[:, i], FeH_fine, bounds_error=False, fill_value='extrapolate')
    MH_line[i] = interp_func(EW[i])

# 5. Average Fe lines
Fe_ind = np.array([2, 3, 5, 6, 7, 8, 9, 10, 14, 15])
print(MH_line[Fe_ind])
FeH = np.mean(MH_line[Fe_ind])
FeH_err = sem(MH_line[Fe_ind])

# 6. Average alpha lines
alpha_ind = np.array([1, 4, 11, 12, 13])
alphaH = np.mean(MH_line[alpha_ind])
alphaH_err = sem(MH_line[alpha_ind])

# 7. Alpha enhancement
alphaFe = alphaH - FeH
alphaFe_err = np.sqrt(FeH_err**2 + alphaH_err**2)

# CA lines
#CA_ind = np.array([0,1])
#print(MH_line[CA_ind])
#FeH = np.mean(MH_line[CA_ind])
#FeH_err = sem(MH_line[CA_ind])

#print("RC G080960 (Known4) -2.3")
# Print results
print(f"[Fe/H]     = {FeH:.3f} ± {FeH_err:.3f}")
print(f"[alpha/H]  = {alphaH:.3f} ± {alphaH_err:.3f}")
print(f"[alpha/Fe] = {alphaFe:.3f} ± {alphaFe_err:.3f}")

