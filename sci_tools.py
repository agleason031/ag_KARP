#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 17:01:27 2025

@author: agleason
"""
from scipy.ndimage import uniform_filter
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time
from astropy import units as u
from astropy.stats import mad_std
from astropy.nddata import CCDData
from numpy.polynomial import Chebyshev
import numpy as np
import ccdproc
import hashlib
import inspect
import os

_calibration_cache = {}

def get_var_name(var):
    frame = inspect.currentframe().f_back  # Get the caller's frame
    local_vars = frame.f_locals
    for name, value in local_vars.items():
        if value is var:
            return name
    return None  # If no match is found

def inv_median(im):
    """ccdproc.combine multiplies each image by this, so return 1/median to normalize."""
    return 1 / np.median(im.data)

def heliocentric_correction(header, site_height=2788.0, ra=None, dec=None):
    """
    Heliocentric radial-velocity correction (km/s) for one exposure, to be applied
    as lam * (1 + v/c). Uses the frame's own header: RA/DEC (unless overridden),
    the observatory latitude/longitude, and the mid-exposure time (DATE-OBS is the
    start of the exposure, in the time system given by TIMESYS).
    """
    ra = ra if ra is not None else str(header["RA"])
    dec = dec if dec is not None else str(header["DEC"])
    sc = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
    site = EarthLocation.from_geodetic(lon=float(header["LONGITUD"]) * u.deg,
                                       lat=float(header["LATITUDE"]) * u.deg,
                                       height=site_height * u.m)
    scale = str(header.get("TIMESYS", "UTC")).strip().lower()
    t = Time(header["DATE-OBS"], scale=scale if scale in ("tai", "utc") else "utc")
    t = t + float(header.get("EXPTIME", 0.0)) / 2 * u.s # Middle of the exposure
    corr = sc.radial_velocity_correction("heliocentric", obstime=t, location=site)
    return float(corr.to(u.km / u.s).value)

# Bump when the way master calibrations are built changes, so old files on disk are rebuilt
CAL_VERSION = 3 # 2: overscan subtraction, trimming and per-amplifier gain; 3: flat bias/scaling order

def detector_geometry(det):
    """
    The parts of the [detector] config that define how raw frames are turned into
    trimmed, overscan-subtracted electron images (see config.py for the defaults).
    """
    return {k: det[k] for k in ("amp_data", "amp_overscan", "gain", "overscan_order")}

def overscan_trim(ccd, geom):
    """
    For each amplifier: subtract its overscan level (row-by-row median of its overscan
    strip, smoothed along the rows with a Chebyshev polynomial of order
    geom["overscan_order"] and one 5-sigma clipping pass), convert to electrons with
    its gain, and keep only the data columns. Returns CCDData in electrons.
    """
    data = np.asarray(ccd.data, dtype=float)
    rows = np.arange(data.shape[0])
    n_cols = max(c1 for _, c1 in geom["amp_data"])
    out = np.full((data.shape[0], n_cols), np.nan)
    for (c0, c1), (o0, o1), gain in zip(geom["amp_data"], geom["amp_overscan"], geom["gain"]):
        level = np.median(data[:, o0:o1], axis=1)
        model = Chebyshev.fit(rows, level, geom["overscan_order"])
        resid = level - model(rows)
        keep = np.abs(resid) < 5 * mad_std(resid)
        model = Chebyshev.fit(rows[keep], level[keep], geom["overscan_order"])
        out[:, c0:c1] = (data[:, c0:c1] - model(rows)[:, None]) * gain
    return CCDData(out, unit="electron", meta=ccd.meta)

def read_reduced(path, geom):
    """Read a raw frame, subtract overscan, trim, and convert to electrons."""
    return overscan_trim(CCDData.read(path, unit="adu"), geom)

def calibration_key(bias_files, flat_files, geom):
    """
    Short fingerprint of the bias and flat input frames (path, size, modification time),
    the detector settings, and CAL_VERSION. Saved calibrations are only reused when this
    matches, so changing the frames, the detector settings, or the processing forces the
    masters to be rebuilt.
    """
    h = hashlib.sha1()
    h.update(f"v{CAL_VERSION}|{sorted(geom.items())}\n".encode())
    for kind, files in (("bias", bias_files), ("flat", flat_files)):
        h.update(kind.encode())
        for f in files:
            st = os.stat(f)
            h.update(f"{os.path.abspath(f)}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    return h.hexdigest()[:16]

#creates and returns bias and flat images
def get_cal_images(bias_files, flat_files, geom, verbose, grapher):
    #get images from cache if already created from the same input frames and settings
    key = calibration_key(bias_files, flat_files, geom)
    bias_path = grapher.target_dir + "OUT/" + grapher.objid + "_master_bias.fits"
    flat_path = grapher.target_dir + "OUT/" + grapher.objid + "_final_flat.fits"

    if key in _calibration_cache:
        return _calibration_cache[key]

    # Load from disk if both exist and were made from these same input frames
    if os.path.exists(bias_path) and os.path.exists(flat_path):
        masterbias = CCDData.read(bias_path, unit='electron')
        final_flat = CCDData.read(flat_path, unit='adu')
        if masterbias.meta.get("KARPCAL") == key and final_flat.meta.get("KARPCAL") == key:
            print("Loading disk calibration files...")
            final_flat = np.asarray(final_flat.data) # Same type as a freshly made flat
            _calibration_cache[key] = (masterbias, final_flat)
            return masterbias, final_flat
        print("Calibration files on disk were made from different frames or settings, rebuilding them")

    # Overscan-subtract, trim and convert every frame to electrons before combining
    blist = [read_reduced(f, geom) for f in bias_files]

    # Make a master bias from the input bias image
    print("Making Master Bias")
    masterbias = ccdproc.combine(blist, method='median', unit='electron',sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std,mem_limit=350e6)
    
    # Make a master flat: subtract the master bias from each flat, normalize each by its
    # median, then median-combine. (The bias has to come off before scaling, otherwise
    # the scaled bias level stays in the flat.)
    print("Making Master Flat")
    flist = [ccdproc.subtract_bias(read_reduced(f, geom), masterbias) for f in flat_files]
    masterflatDEbias = ccdproc.combine(flist, method='median', scale=inv_median, unit='electron',sigma_clip=False,mem_limit=350e6)
    
    # Smooth master flatdebias:
    np_mfdb = np.asarray(masterflatDEbias) # Convert to a numpy array        
    smooth_mf_mb = uniform_filter(np_mfdb, size=5)
    
    # Make a Final Flat by dividing mf_mb by smooth_mf_mb
    final_flat = np_mfdb/smooth_mf_mb
    # Y top of 0th order line 2025ish
    # Y bottom of 0th order line 1995ish
    # Go ten pixels on either side
    # X=165-835
    xZo = np.arange(165,835,1)
    yZo = np.arange(1985,2035,1)
    
    # Remove zeorth orther line
    for i in range(len(xZo)):
        for k in range(len(yZo)):
            final_flat[yZo[k], xZo[i]] = 1
      
    # Remove other bad features
    xOther1 = np.arange(165,835,1)
    yOther1 = np.arange(2074,2083,1)
    
    for i in range(len(xOther1)):
        for k in range(len(yOther1)):
            final_flat[yOther1[k], xOther1[i]] = 1 # Set the flat value of these small areas to 1
            # Note that this minimally affects the final flat as these areas are relativly small
            # So we still get a representative final flat image

    if (verbose == True):
        grapher.plot_image(masterbias, get_var_name(masterbias))
        grapher.plot_image(final_flat, get_var_name(final_flat))

    # Record which input frames these were made from, so stale files aren't reused later
    masterbias.meta["KARPCAL"] = (key, "KARP fingerprint of bias+flat input frames")
    masterbias.write(bias_path, overwrite=True)
    CCDData(final_flat, unit="adu", meta={"KARPCAL": (key, "KARP fingerprint of bias+flat input frames")}).write(flat_path, overwrite=True)
    
    print("Final Flat Made")
    
    _calibration_cache[key] = (masterbias, final_flat)
    return masterbias, final_flat
