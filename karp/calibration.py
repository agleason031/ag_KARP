"""
Basic CCD calibration: turning raw frames into trimmed, overscan-subtracted images in
electrons, the arc image, and the master bias and pixel-to-pixel flat.

Master calibrations are saved (calib/ in the object's output folder) with a
fingerprint of the input frames and settings, and reused only when it matches.
"""
import hashlib
import logging
import os

import ccdproc
import numpy as np
from astropy.nddata import CCDData
from astropy.stats import mad_std
from numpy.polynomial import Chebyshev
from scipy.ndimage import binary_dilation, label, median_filter, uniform_filter

from . import detector

log = logging.getLogger("karp")

# Bump when the way master calibrations are built changes, so old files on disk are rebuilt
CAL_VERSION = 5 # 2: overscan subtraction, trimming and per-amplifier gain; 3: flat bias/scaling order;
                # 4: zeroth-order rows found from the flat (or set in [flat]); 5: layout from each header

LAMPS = ("ARGON", "NEON", "KRYPTON")

_cache = {} # fingerprint -> (master bias, final flat), for this Python session


# ---------------------------------------------------------------------------
# single frames
# ---------------------------------------------------------------------------

def overscan_trim(ccd, det):
    """
    For each amplifier: subtract its overscan level (row-by-row median of its overscan
    strip, smoothed along the rows with a Chebyshev polynomial of order
    det["overscan_order"] and one 5-sigma clipping pass), convert to electrons with
    its gain, and keep only the data columns and rows. The layout comes from the frame's
    header (see detector.py); det is the [detector] config. Returns CCDData in electrons.
    """
    geom = detector.frame_geometry(ccd.meta, det)
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
    r0, r1 = geom["rows"]
    return CCDData(out[r0:r1], unit="electron", meta=ccd.meta)


def load_frame(path, det):
    """A raw frame, overscan-subtracted, trimmed and in electrons."""
    return overscan_trim(CCDData.read(path, unit="adu"), det)


def load_arc(paths, det):
    """
    Arc image from one or more arc frames: frames with the same lamps on are
    median-combined (removing cosmic rays when there are three or more), and the
    combinations for different lamps are added. Returns (image, sorted lamps that are on).
    """
    by_lamps = {}
    for p in paths:
        ccd = load_frame(p, det)
        lamps = tuple(sorted(l for l in LAMPS if str(ccd.meta.get(l, "off")).strip().lower() == "on"))
        by_lamps.setdefault(lamps, []).append(np.asarray(ccd.data))
    image = 0
    for lamps, images in by_lamps.items():
        image = image + (np.median(images, axis=0) if len(images) > 1 else images[0])
        log.debug(f"Arc: {len(images)} frame(s) with {', '.join(lamps) or 'no lamps'} on")
    return image, sorted({l for lamps in by_lamps for l in lamps})


# ---------------------------------------------------------------------------
# flat field
# ---------------------------------------------------------------------------

def _inv_median(im):
    """ccdproc.combine multiplies each image by its scale, so return 1/median to normalize."""
    return 1 / np.median(im.data)


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


def zeroth_order_rows(master_flat, flat_cfg, biny):
    """
    Rows of the flat to set to 1: the zeroth-order bands found in the flat ("auto"), the
    given unbinned row ranges, or none. Row sizes in the config are unbinned rows.
    """
    zo = flat_cfg["zeroth_order"]
    if zo == "auto":
        pad = detector.to_binned(flat_cfg["zeroth_pad"], biny, minimum=0)
        rows, bands = find_bright_rows(master_flat, flat_cfg["zeroth_threshold"],
                                       min_width=detector.to_binned(4, biny, minimum=2), pad=pad,
                                       window=detector.to_binned(101, biny, odd=True, minimum=3))
        found = ", ".join(f"{r0}-{r1}" for r0, r1 in bands) or "none"
        log.info(f"Zeroth order in the flat: bright rows {found}; flat set to 1 in {int(rows.sum())} rows "
                 f"(padded by {pad})")
        return rows
    rows = np.zeros(master_flat.shape[0], dtype=bool)
    for r0, r1 in ([] if zo == "none" else zo):
        rows[int(r0) // biny:int(r1) // biny + 1] = True
    if rows.any():
        log.info(f"Flat set to 1 in unbinned rows {zo} ({int(rows.sum())} rows at binning y = {biny})")
    return rows


# ---------------------------------------------------------------------------
# master calibrations
# ---------------------------------------------------------------------------

def calibration_key(bias_files, flat_files, det, flat_cfg):
    """
    Short fingerprint of the bias and flat input frames (path, size, modification time),
    the [detector] and [flat] settings, and CAL_VERSION. Saved calibrations are only
    reused when this matches.
    """
    h = hashlib.sha1()
    h.update(f"v{CAL_VERSION}|{sorted(det.items())}|{sorted(flat_cfg.items())}\n".encode())
    for kind, files in (("bias", bias_files), ("flat", flat_files)):
        h.update(kind.encode())
        for f in files:
            st = os.stat(f)
            h.update(f"{os.path.abspath(f)}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    return h.hexdigest()[:16]


def cached_calibrations():
    """The (master bias, final flat) pairs made or loaded in this session (for tests)."""
    return list(_cache.values())


def master_calibrations(bias_files, flat_files, det, flat_cfg, out, plots):
    """
    Master bias (CCDData, electrons) and final pixel-to-pixel flat (array). Built from
    the raw frames, or loaded from out.master_bias / out.final_flat when they were made
    from the same frames and settings. With no flats the flat is 1 everywhere.
    det: [detector] settings; flat_cfg: [flat] settings; out: pipeline.OutputPaths;
    plots: plots.Plotter (images of both when verbose).
    """
    det = detector.settings(det)
    key = calibration_key(bias_files, flat_files, det, flat_cfg)
    if key in _cache:
        return _cache[key]

    if os.path.exists(out.master_bias) and os.path.exists(out.final_flat):
        bias = CCDData.read(out.master_bias, unit="electron")
        flat = CCDData.read(out.final_flat, unit="adu")
        if bias.meta.get("KARPCAL") == key and flat.meta.get("KARPCAL") == key:
            log.info("Loading calibration files from disk")
            _cache[key] = (bias, np.asarray(flat.data))
            return _cache[key]
        log.info("Calibration files on disk were made from different frames or settings, rebuilding them")

    log.info(f"Making the master bias from {len(bias_files)} frames")
    bias = ccdproc.combine([load_frame(f, det) for f in bias_files], method="median", unit="electron",
                           sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
                           sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std, mem_limit=350e6)

    if not flat_files:
        log.warning("No flats: no flat-field correction (the flat is 1 everywhere)")
        _cache[key] = (bias, np.ones(bias.shape))
        return _cache[key]

    # Master flat: the master bias comes off each flat before it is scaled by its median
    # (otherwise the scaled bias level would stay in the flat), then median-combine
    log.info(f"Making the master flat from {len(flat_files)} frames")
    flats = [ccdproc.subtract_bias(load_frame(f, det), bias) for f in flat_files]
    master_flat = np.asarray(ccdproc.combine(flats, method="median", scale=_inv_median, unit="electron",
                                             sigma_clip=False, mem_limit=350e6))

    # Pixel-to-pixel flat: the master flat divided by a 5x5 smoothed copy of itself
    flat = master_flat / uniform_filter(master_flat, size=5)

    # Zeroth order: a bright band across the slit in the flat-lamp exposure. The
    # smoothing can't follow its sharp edges, so it would print into the science
    # frames; set the flat to 1 there (over the illuminated columns).
    _, biny = detector.binning(flats[0].meta)
    rows = zeroth_order_rows(master_flat, flat_cfg, biny)
    flat[np.ix_(rows, illuminated_columns(master_flat))] = 1

    if plots.verbose:
        plots.image(bias, "master bias", out.diagnostic("master_bias.png"))
        plots.image(flat, "final flat", out.diagnostic("final_flat.png"))

    # Record which input frames these were made from, so stale files aren't reused later
    stamp = (key, "KARP fingerprint of bias+flat input frames")
    bias.meta["KARPCAL"] = stamp
    bias.write(out.master_bias, overwrite=True)
    CCDData(flat, unit="adu", meta={"KARPCAL": stamp}).write(out.final_flat, overwrite=True)
    log.info("Final flat made")

    _cache[key] = (bias, flat)
    return bias, flat
