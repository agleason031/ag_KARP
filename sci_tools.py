#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 17:01:27 2025

@author: agleason
"""
from scipy.ndimage import uniform_filter, median_filter, binary_dilation, label
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time
from astropy import units as u
from astropy.stats import mad_std
from astropy.nddata import CCDData
from numpy.polynomial import Chebyshev
import numpy as np
import ccdproc
import hashlib
import logging
import os

log = logging.getLogger("karp")

_calibration_cache = {}

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
CAL_VERSION = 4 # 2: overscan subtraction, trimming and per-amplifier gain; 3: flat bias/scaling order;
                # 4: zeroth-order rows found from the flat (or set in [flat])

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

def calibration_key(bias_files, flat_files, geom, flat_cfg):
    """
    Short fingerprint of the bias and flat input frames (path, size, modification time),
    the detector and [flat] settings, and CAL_VERSION. Saved calibrations are only reused when this
    matches, so changing the frames, the detector settings, or the processing forces the
    masters to be rebuilt.
    """
    h = hashlib.sha1()
    h.update(f"v{CAL_VERSION}|{sorted(geom.items())}|{sorted(flat_cfg.items())}\n".encode())
    for kind, files in (("bias", bias_files), ("flat", flat_files)):
        h.update(kind.encode())
        for f in files:
            st = os.stat(f)
            h.update(f"{os.path.abspath(f)}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    return h.hexdigest()[:16]

def illuminated_columns(flat):
    """Columns the slit illuminates: median level above half the typical illuminated level."""
    colp = np.median(flat, axis=0)
    return colp > 0.5 * np.median(colp[colp > 0.5 * np.max(colp)])

def find_bright_rows(flat, threshold=0.02, min_width=4, pad=10, window=101):
    """
    Rows crossed by a bright band that runs along the slit (e.g. the zeroth order of
    the grating in the flat-lamp exposure). The median of each row over the illuminated
    columns is compared with a running median over `window` rows; runs of at least
    `min_width` consecutive rows more than `threshold` (fractional) above it are bright
    bands. Single bright rows are ordinary pixel response and stay in the flat.
    Returns (boolean mask over rows after padding by `pad` rows, [(first, last)] of the
    bands before padding).
    """
    rowp = np.median(flat[:, illuminated_columns(flat)], axis=1)
    rel = rowp / median_filter(rowp, window, mode="nearest") - 1
    labels, n = label(rel > threshold)
    bands = []
    for k in range(1, n + 1):
        rows = np.where(labels == k)[0]
        if len(rows) >= min_width:
            bands.append((int(rows[0]), int(rows[-1])))
    bright = np.zeros(len(rowp), dtype=bool)
    for r0, r1 in bands:
        bright[r0:r1 + 1] = True
    if pad > 0 and bright.any():
        bright = binary_dilation(bright, iterations=pad)
    return bright, bands

#creates and returns bias and flat images
def get_cal_images(bias_files, flat_files, geom, flat_cfg, grapher):
    """
    Master bias and final (pixel-to-pixel) flat, built from the raw frames or loaded
    from OUT/ when they were made from the same frames and settings. flat_cfg is the
    [flat] config section: the final flat is set to 1 in rows crossed by the zeroth
    order (found automatically, or given as row ranges), where the flat-lamp light
    doesn't represent the detector response.
    """
    #get images from cache if already created from the same input frames and settings
    key = calibration_key(bias_files, flat_files, geom, flat_cfg)
    bias_path = grapher.target_dir + "OUT/" + grapher.objid + "_master_bias.fits"
    flat_path = grapher.target_dir + "OUT/" + grapher.objid + "_final_flat.fits"

    if key in _calibration_cache:
        return _calibration_cache[key]

    # Load from disk if both exist and were made from these same input frames
    if os.path.exists(bias_path) and os.path.exists(flat_path):
        masterbias = CCDData.read(bias_path, unit='electron')
        final_flat = CCDData.read(flat_path, unit='adu')
        if masterbias.meta.get("KARPCAL") == key and final_flat.meta.get("KARPCAL") == key:
            log.info("Loading calibration files from disk")
            final_flat = np.asarray(final_flat.data) # Same type as a freshly made flat
            _calibration_cache[key] = (masterbias, final_flat)
            return masterbias, final_flat
        log.info("Calibration files on disk were made from different frames or settings, rebuilding them")

    # Overscan-subtract, trim and convert every frame to electrons before combining
    blist = [read_reduced(f, geom) for f in bias_files]

    # Make a master bias from the input bias image
    log.info(f"Making the master bias from {len(bias_files)} frames")
    masterbias = ccdproc.combine(blist, method='median', unit='electron',sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std,mem_limit=350e6)

    # Make a master flat: subtract the master bias from each flat, normalize each by its
    # median, then median-combine. (The bias has to come off before scaling, otherwise
    # the scaled bias level stays in the flat.)
    log.info(f"Making the master flat from {len(flat_files)} frames")
    flist = [ccdproc.subtract_bias(read_reduced(f, geom), masterbias) for f in flat_files]
    masterflatDEbias = ccdproc.combine(flist, method='median', scale=inv_median, unit='electron',sigma_clip=False,mem_limit=350e6)

    # Pixel-to-pixel flat: the master flat divided by a 5x5 smoothed copy of itself
    np_mfdb = np.asarray(masterflatDEbias)
    final_flat = np_mfdb / uniform_filter(np_mfdb, size=5)

    # Zeroth order: a bright band across the slit in the flat-lamp exposure. The
    # smoothing can't follow its sharp edges, so it would print into the science
    # frames; set the flat to 1 there (over the illuminated columns).
    zo = flat_cfg["zeroth_order"]
    if zo == "auto":
        rows, bands = find_bright_rows(np_mfdb, flat_cfg["zeroth_threshold"], pad=flat_cfg["zeroth_pad"])
        found = ", ".join(f"{r0}-{r1}" for r0, r1 in bands) or "none"
        log.info(f"Zeroth order in the flat: bright rows {found}; flat set to 1 in {int(rows.sum())} rows "
                 f"(padded by {flat_cfg['zeroth_pad']})")
    else: # "none" or explicit [[row_min, row_max], ...] (inclusive)
        rows = np.zeros(np_mfdb.shape[0], dtype=bool)
        for r0, r1 in ([] if zo == "none" else zo):
            rows[int(r0):int(r1) + 1] = True
        if rows.any():
            log.info(f"Flat set to 1 in rows {zo}")
    final_flat[np.ix_(rows, illuminated_columns(np_mfdb))] = 1

    if grapher.verbose:
        grapher.plot_image(masterbias, "master_bias")
        grapher.plot_image(final_flat, "final_flat")

    # Record which input frames these were made from, so stale files aren't reused later
    masterbias.meta["KARPCAL"] = (key, "KARP fingerprint of bias+flat input frames")
    masterbias.write(bias_path, overwrite=True)
    CCDData(final_flat, unit="adu", meta={"KARPCAL": (key, "KARP fingerprint of bias+flat input frames")}).write(flat_path, overwrite=True)

    log.info("Final flat made")

    _calibration_cache[key] = (masterbias, final_flat)
    return masterbias, final_flat
