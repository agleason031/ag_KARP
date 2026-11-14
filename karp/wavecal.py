"""
Wavelength calibration for KARP.

Steps, for each science frame:
  1. Extract the arc spectrum along the science trace, through the same aperture,
     with the local background removed (extract_arc).
  2. Find the arc emission lines and centroid them with Gaussian fits, leaving out
     features much wider or narrower than a typical line (ghosts, blends) and lines on
     the wing of a much brighter one (find_arc_lines).
  3. Start from a rough solution for the grating (a reference file in
     karp/data/wavecal/), and find the shift and stretch that line up the most
     catalog lines with the observed ones (initial_guess).
  4. Match observed lines to the catalog, fit a polynomial row -> wavelength with
     iterative sigma clipping, re-match with the improved solution, and repeat
     until the set of matched lines stops changing (identify_lines).
  5. Optionally measure night-sky emission lines in the science frame's background
     to check for flexure between the arc and the science exposure (sky_offsets).

Line lists are in karp/data/linelists/ (air wavelengths). Reference solutions are
JSON files with a list of [row, wavelength] pairs; make one for a new grating with
    python scripts/make_wavecal_reference.py <wavecal table> --disperser NAME --binning-y N -o FILE

Wavelengths are in air, in Angstroms. Rows are detector rows of the trimmed frame.
"""
import json
import warnings
import os
from dataclasses import dataclass, field

import numpy as np
from astropy.stats import mad_std
from numpy.polynomial import Polynomial
from scipy.optimize import curve_fit
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

from .paths import data_file
LAMP_FILES = {"ARGON": "ArI.txt", "NEON": "NeI.txt", "KRYPTON": "KrI.txt"}
C_KMS = 299792.458


def _gauss(x, a, mu, sigma, bck):
    return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + bck


# ---------------------------------------------------------------------------
# catalog and reference
# ---------------------------------------------------------------------------

def load_linelist(lamps):
    """Sorted, unique catalog wavelengths for the given lamps (e.g. ["ARGON"])."""
    waves = []
    for lamp in lamps:
        name = LAMP_FILES.get(lamp.upper())
        if name is None:
            raise ValueError(f"No line list for lamp {lamp!r}; known: {sorted(LAMP_FILES)}")
        with open(data_file("linelists", name)) as f:
            for line in f:
                line = line.split("#", 1)[0].split()
                if line:
                    waves.append(float(line[0]))
    if not waves:
        raise ValueError(f"Empty line list for lamps {lamps}")
    return np.unique(waves)


def remove_blends(catalog, min_sep):
    """Drop catalog lines that have another catalog line within min_sep (A): blends shift centroids."""
    gaps = np.diff(catalog)
    blended = np.zeros(len(catalog), dtype=bool)
    blended[:-1] |= gaps < min_sep
    blended[1:] |= gaps < min_sep
    return catalog[~blended]


def load_reference(disperser, path=None):
    """
    Reference solution: from `path` if given, else karp/data/wavecal/kosmos_<disperser>.json.
    Returns the parsed JSON (keys: disperser, binning_y, order, lines = [[row, lam], ...]).
    """
    if path is None:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in disperser.lower())
        path = data_file("wavecal", f"kosmos_{safe}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No reference wavelength solution for disperser {disperser!r} ({path}). "
            "Make one from a table of identified arc lines with scripts/make_wavecal_reference.py "
            "(see the README, 'Wavelength calibration data'), or set [wavecal] reference.")
    with open(path) as f:
        ref = json.load(f)
    ref["path"] = path
    return ref


def reference_polynomial(ref, binning_y):
    """Row -> wavelength polynomial from a reference, converted to this frame's row binning."""
    rows, lam = np.array(ref["lines"], dtype=float).T
    b_ref = ref.get("binning_y", 1)
    # Reference rows -> unbinned -> this binning (pixel centers)
    rows = ((rows + 0.5) * b_ref) / binning_y - 0.5
    return Polynomial.fit(rows, lam, min(ref.get("order", 3), len(rows) - 1))


# ---------------------------------------------------------------------------
# arc extraction and line finding
# ---------------------------------------------------------------------------

def extract_arc(arc, columns, aperture_half, baseline_window=101):
    """
    Arc spectrum along the trace: sum over the science aperture (columns +/- aperture_half),
    minus a running median along the rows as the baseline. (The lamp lights the whole
    slit, so the science background windows contain the same lines and can't be used
    as background.)
    """
    n_rows, n_cols = arc.shape
    spec = np.full(n_rows, np.nan)
    for r in range(n_rows):
        c = int(columns[r])
        lo, hi = c - aperture_half, c + aperture_half + 1
        if lo >= 0 and hi <= n_cols:
            spec[r] = np.sum(arc[r, lo:hi])
    good = np.isfinite(spec)
    filled = np.where(good, spec, np.nanmedian(spec))
    baseline = median_filter(filled, size=baseline_window, mode="nearest")
    return np.where(good, spec - baseline, np.nan)


def _half_max_width(y, p):
    """Full width at half maximum (rows) of the peak at index p, from the half-maximum crossings."""
    half = y[p] / 2
    lo = p
    while lo > 0 and y[lo] > half:
        lo -= 1
    hi = p
    while hi < len(y) - 1 and y[hi] > half:
        hi += 1
    return hi - lo


def find_arc_lines(spec, nsigma=5.0, min_sep=2, max_sigma=8.0, width_range=(0.7, 1.3), min_rel_prominence=0.2):
    """
    Emission lines in a 1D arc spectrum.
      1. Peaks more prominent than nsigma times the noise and than min_rel_prominence
         times their own height. With a wide slit the lines are flat-topped (the image
         of the slit), and noise on the top makes extra local maxima; those rise only a
         little above the dip that separates them from the line's highest point, so
         they fail the relative test, while separate lines pass it.
      2. The typical line width W (FWHM, rows) from the 20 most prominent peaks.
      3. Each line is centroided with a Gaussian fit (the baseline is already removed):
         first in a window of +/- max(5, 1.5 W) rows that stops halfway to the
         neighbouring peaks (so a fit can't slide onto a brighter neighbour), then (three
         times) over the full window with the neighbours' fits subtracted (so a close neighbour
         doesn't truncate or pull it). Fits that land within W/2 of each other are merged.
      4. Dropped: fits much wider or narrower than the median (outside width_range
         times the median FWHM: ghosts and scattered light are broad, blends are broad
         or lopsided), and lines within 2 W of one more than three times as bright (on its
         wing). They would bias the solution if matched to a catalog line.
    Returns (rows, amplitudes, sigmas, info), info = {"fwhm": median FWHM in rows,
    "found": lines before these cuts, "too_wide", "too_narrow", "crowded": numbers dropped}.
    """
    empty = (np.array([]), np.array([]), np.array([]),
             {"fwhm": np.nan, "found": 0, "too_wide": 0, "too_narrow": 0, "crowded": 0})
    good = np.isfinite(spec)
    filled = np.where(good, spec, np.nanmedian(spec))
    noise = mad_std(np.diff(filled[good])) / np.sqrt(2)
    peaks, props = find_peaks(filled, prominence=nsigma * noise, distance=min_sep)
    height = filled[peaks] - np.median(filled)
    distinct = props["prominences"] >= min_rel_prominence * np.maximum(height, 0)
    peaks, prom = peaks[distinct], props["prominences"][distinct]
    if len(peaks) == 0:
        return empty
    strongest = peaks[np.argsort(prom)[::-1][:20]]
    width = float(np.median([_half_max_width(filled - np.median(filled), p) for p in strongest]))
    half_window = max(5, int(np.ceil(1.5 * width)))

    x_all = np.arange(len(spec))

    def fit_one(x, y, p0):
        """Gaussian (background fixed at zero: the baseline is already removed); None if the fit fails."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Unbounded Levenberg-Marquardt is much faster than a bounded fit; bad fits are rejected below
                popt, _ = curve_fit(lambda x, a, mu, sigma: _gauss(x, a, mu, sigma, 0.0), x, y, p0=p0, maxfev=400)
        except RuntimeError:
            return None
        a, mu, sigma = popt[0], popt[1], abs(popt[2])
        if a <= 0 or not (x[0] <= mu <= x[-1]) or not (0.3 <= sigma <= max_sigma):
            return None
        return a, mu, sigma

    # Pass 1: each line in a window that stops halfway to its neighbours
    first = []
    for k, p in enumerate(peaks):
        left = (p + peaks[k - 1]) // 2 + 1 if k > 0 else 0
        right = (p + peaks[k + 1]) // 2 if k < len(peaks) - 1 else len(spec) - 1
        lo, hi = max(0, p - half_window, left), min(len(spec), p + half_window + 1, right + 1)
        x, y = x_all[lo:hi], filled[lo:hi]
        w = np.clip(y, 0, None)
        if hi - lo < 4 or w.sum() <= 0:
            first.append(None)
            continue
        first.append(fit_one(x, y, [y.max(), np.sum(x * w) / w.sum(), max(width / 2.3548, 0.5)]))

    # Pass 2 (repeated): refit each line over its full window with its neighbours'
    # latest fits subtracted, so a nearby line neither pulls the fit nor truncates the window
    current = list(first)
    for _ in range(3):
        updated = []
        for k, p in enumerate(peaks):
            if current[k] is None:
                updated.append(None)
                continue
            lo, hi = max(0, p - half_window), min(len(spec), p + half_window + 1)
            x = x_all[lo:hi]
            y = filled[lo:hi].copy()
            for j, f in enumerate(current):
                if j != k and f is not None and abs(f[1] - p) < half_window + 4 * f[2]:
                    y -= _gauss(x, f[0], f[1], f[2], 0.0)
            refit = fit_one(x, y, list(current[k]))
            updated.append(refit if refit is not None else current[k])
        current = updated
    fits = [(f[1], f[0], f[2]) for f in current if f is not None]
    if not fits:
        return empty

    # Merge fits that converged on the same line (keep the brighter)
    fits.sort(key=lambda f: f[1], reverse=True)
    kept = []
    for f in fits:
        if all(abs(f[0] - g[0]) > width / 2 for g in kept):
            kept.append(f)
    kept.sort()
    rows, amps, sigs = (np.array(v) for v in zip(*kept))

    ratio = sigs / np.median(sigs)
    too_wide, too_narrow = ratio > width_range[1], ratio < width_range[0]
    # A line within 2 W of one more than three times as bright sits on that line's wing; the
    # line shape isn't exactly Gaussian, so its centroid can't be trusted
    crowded = np.array([np.any((np.abs(rows - r) < 2 * width) & (amps > 3 * a)) for r, a in zip(rows, amps)])
    keep = ~too_wide & ~too_narrow & ~crowded
    info = {"fwhm": float(2.3548 * np.median(sigs)), "found": len(rows), "too_wide": int(too_wide.sum()),
            "too_narrow": int(too_narrow.sum()), "crowded": int((crowded & ~too_wide & ~too_narrow).sum())}
    return rows[keep], amps[keep], sigs[keep], info


# ---------------------------------------------------------------------------
# identification and fitting
# ---------------------------------------------------------------------------

def initial_guess(peak_rows, peak_amps, catalog, poly0, n_rows, search, tol_pix,
                  stretches=np.linspace(0.98, 1.02, 21)):
    """
    Shift and stretch of the reference solution that best lines up catalog lines with
    observed peaks. The guess is lam(row) = poly0(c + (row + shift - c) * stretch),
    c = middle row. Score: sum of log-amplitude of observed peaks with a catalog line
    predicted within tol_pix. Returns (guess function, shift, stretch, score).
    """
    c = (n_rows - 1) / 2
    grid = np.linspace(-search - n_rows * 0.05, n_rows * 1.05 + search, 20 * n_rows)
    weights = np.log1p(np.maximum(peak_amps, 0))
    shifts = np.arange(-search, search + 1)
    best = (-1.0, 0, 1.0)
    for k in stretches:
        lam_grid = poly0(c + (grid - c) * k)
        order = np.argsort(lam_grid)
        # Rows (in the stretched reference frame) where each catalog line would fall
        model_rows = np.interp(catalog, lam_grid[order], grid[order], left=np.nan, right=np.nan)
        model_rows = model_rows[np.isfinite(model_rows)]
        if model_rows.size == 0:
            continue
        # Observed row = model row - shift
        d = np.abs(peak_rows[:, None, None] - (model_rows[None, :, None] - shifts[None, None, :]))
        hit = (d.min(axis=1) <= tol_pix) # (n_peaks, n_shifts)
        scores = (hit * weights[:, None]).sum(axis=0)
        i = int(np.argmax(scores))
        if scores[i] > best[0] or (scores[i] == best[0] and abs(shifts[i]) < abs(best[1])):
            best = (float(scores[i]), int(shifts[i]), float(k))
    score, shift, k = best
    guess = lambda rows: poly0(c + (np.asarray(rows, dtype=float) + shift - c) * k)
    return guess, shift, k, score


def match_lines(peak_rows, lam_pred, catalog, tol_ang):
    """One-to-one matches between observed lines and catalog lines within tol_ang. Returns index arrays."""
    j = np.clip(np.searchsorted(catalog, lam_pred), 1, len(catalog) - 1)
    nearest = np.where(np.abs(catalog[j - 1] - lam_pred) < np.abs(catalog[j] - lam_pred), j - 1, j)
    dist = np.abs(catalog[nearest] - lam_pred)
    ok = dist <= tol_ang
    # If two observed lines claim the same catalog line, keep the closer one
    pairs = {}
    for i in np.nonzero(ok)[0]:
        n = nearest[i]
        if n not in pairs or dist[i] < dist[pairs[n]]:
            pairs[n] = i
    obs = np.array(sorted(pairs.values()), dtype=int)
    cat = np.array([nearest[i] for i in obs], dtype=int)
    return obs, cat


@dataclass
class WaveSolution:
    poly: Polynomial # row -> air wavelength (A), no heliocentric correction
    rows: np.ndarray # centroids of matched arc lines
    lam: np.ndarray # catalog wavelengths of matched lines
    used: np.ndarray # bool: kept by the clipped fit
    n_found: int # arc lines found
    shift: int # reference -> observed shift (rows)
    stretch: float
    reference: str
    n_rows: int # rows in the frame
    notes: list = field(default_factory=list)
    sigmas: np.ndarray = None # Gaussian sigma (rows) of the matched arc lines

    @property
    def line_fwhm(self):
        """Median FWHM (A) of the arc lines kept by the fit: the resolution for a source filling the slit."""
        if self.sigmas is None or not self.used.any():
            return float("nan")
        disp = np.abs(self.poly.deriv()(self.rows[self.used]))
        return float(np.median(2.3548 * self.sigmas[self.used] * disp))

    @property
    def residuals(self):
        return self.lam - self.poly(self.rows)

    @property
    def rms_ang(self):
        return float(np.sqrt(np.mean(self.residuals[self.used] ** 2)))

    @property
    def rms_kms(self):
        return float(np.sqrt(np.mean((self.residuals[self.used] / self.lam[self.used]) ** 2)) * C_KMS)

    def summary(self):
        return (f"Wavelength solution: {self.used.sum()} of {len(self.rows)} matched lines used "
                f"({self.n_found} arc lines found), order {self.poly.degree()}, "
                f"RMS {self.rms_ang:.4f} A = {self.rms_kms:.2f} km/s, "
                f"range {self.poly(0):.1f}-{self.poly(self.n_rows - 1):.1f} A "
                f"(reference shift {self.shift:+d} rows, stretch {self.stretch:.3f}), "
                f"arc line FWHM {self.line_fwhm:.2f} A")

    def table(self):
        """Text table of matched lines for checking the identification by eye."""
        lines = ["# row        lam_catalog  lam_fit      resid_A   resid_kms  used"]
        for r, l, used in zip(self.rows, self.lam, self.used):
            fit = self.poly(r)
            lines.append(f"{r:10.3f}  {l:10.3f}  {fit:10.3f}  {l - fit:+8.4f}  {(l - fit) / l * C_KMS:+8.2f}   {int(used)}")
        return "\n".join(lines) + "\n"


def identify_lines(peak_rows, catalog, guess, order, tol_ang, clip_sigma, min_lines, max_iter=10):
    """
    Match observed lines to the catalog starting from `guess` (row -> lam), fit a
    polynomial with iterative clipping, re-match with the new fit, and repeat until
    the matched set stops changing. Returns (poly, obs_idx, cat_idx, used).
    """
    predict = guess
    prev = None
    for _ in range(max_iter):
        obs, cat = match_lines(peak_rows, predict(peak_rows), catalog, tol_ang)
        if len(obs) < min_lines:
            raise RuntimeError(f"Only {len(obs)} arc lines matched the catalog (need {min_lines}); "
                               "check the lamps, the reference solution, or [wavecal] search/match_tol")
        x, y = peak_rows[obs], catalog[cat]
        deg = min(order, len(obs) - 2)
        used = np.ones(len(obs), dtype=bool)
        for _ in range(10):
            poly = Polynomial.fit(x[used], y[used], deg)
            resid = y - poly(x)
            scale = max(mad_std(resid[used]), 1e-4)
            new_used = np.abs(resid) < clip_sigma * scale
            if np.array_equal(new_used, used) or new_used.sum() < min_lines:
                break
            used = new_used
        key = (tuple(obs), tuple(cat))
        if key == prev:
            break
        prev = key
        predict = poly
    return poly, obs, cat, used


def solve(arc_spec, lamps, reference, binning_y, wcfg):
    """Full wavelength solution for one extracted arc spectrum. `wcfg` is the [wavecal] config."""
    n_rows = len(arc_spec)
    poly0 = reference_polynomial(reference, binning_y)
    disp = abs(poly0.deriv()(n_rows / 2)) # A per row, roughly
    catalog_all = load_linelist(lamps)
    lam_lo, lam_hi = sorted([poly0(0), poly0(n_rows - 1)])
    margin = wcfg["search"] * disp + 0.05 * (lam_hi - lam_lo)
    catalog = catalog_all[(catalog_all > lam_lo - margin) & (catalog_all < lam_hi + margin)]
    catalog = remove_blends(catalog, 2.5 * disp)

    rows, amps, sigs, found = find_arc_lines(arc_spec, nsigma=wcfg["peak_sigma"])
    if len(rows) < wcfg["min_lines"]:
        raise RuntimeError(f"Only {len(rows)} arc lines found (need {wcfg['min_lines']})")
    line_note = (f"{found['found']} arc lines found (FWHM {found['fwhm']:.1f} rows); left out: {found['too_wide']} "
                 f"much wider, {found['too_narrow']} much narrower, {found['crowded']} next to a brighter line")
    guess, shift, stretch, _ = initial_guess(rows, amps, catalog, poly0, n_rows,
                                             wcfg["search"], wcfg["match_tol"])

    def attempt(order):
        poly, obs, cat, used = identify_lines(rows, catalog, guess, order, wcfg["match_tol"] * disp,
                                              wcfg["clip_sigma"], wcfg["min_lines"])
        return WaveSolution(poly=poly, rows=rows[obs], lam=catalog[cat], used=used, n_found=len(rows),
                            shift=shift, stretch=stretch, reference=reference["path"], n_rows=n_rows,
                            sigmas=sigs[obs])

    if wcfg["order"] != "auto":
        sol = attempt(wcfg["order"])
        sol.notes = [line_note]
    else:
        # Automatic order: the one that best predicts every matched line from the others
        best, notes = None, []
        for order in AUTO_ORDERS:
            try:
                sol = attempt(order)
            except RuntimeError:
                continue
            if sol.used.sum() < order + 5: # too few lines to judge this order
                continue
            score = order_score(sol, order, disp)
            notes.append(f"order {order}: {sol.used.sum()} of {len(sol.rows)} lines kept, "
                         f"capped leave-one-out RMS {score:.3f} rows")
            if best is None or score < best[0]:
                best = (score, sol)
        if best is None:
            raise RuntimeError("No polynomial order gave a usable wavelength solution")
        sol = best[1]
        sol.notes = [line_note] + notes + [f"chose order {sol.poly.degree()}"]

    if sol.rms_kms > wcfg["max_rms"]:
        raise RuntimeError(f"Wavelength solution RMS {sol.rms_kms:.1f} km/s exceeds [wavecal] max_rms "
                           f"({wcfg['max_rms']} km/s): the lines were probably misidentified. "
                           "Check the reference solution and lamps, or increase [wavecal] search. "
                           + sol.summary())
    return sol


AUTO_ORDERS = (2, 3, 4, 5, 6)


def order_score(sol, order, disp):
    """
    How well a polynomial of this order predicts the arc lines, in rows. Every matched
    line counts, not just the ones the fit kept: kept lines by their leave-one-out
    residual, clipped lines by their residual from the fit. Each line's residual is
    capped at 1 row, so a bad catalog line costs every order the same, while an order
    too stiff to follow the dispersion is penalized for the lines it had to clip.
    """
    r = np.empty(len(sol.rows))
    for i in range(len(sol.rows)):
        keep = sol.used.copy()
        keep[i] = False
        p = Polynomial.fit(sol.rows[keep], sol.lam[keep], order)
        r[i] = (sol.lam[i] - p(sol.rows[i])) / disp
    return float(np.sqrt(np.mean(np.minimum(r ** 2, 1.0))))


# ---------------------------------------------------------------------------
# flexure check with night-sky lines
# ---------------------------------------------------------------------------

def extract_sky(image, columns, aperture_half, buffer, background):
    """Sky spectrum per row: median of the background windows on both sides of the trace (e- per pixel)."""
    n_rows, n_cols = image.shape
    sky = np.full(n_rows, np.nan)
    for r in range(n_rows):
        c = int(columns[r])
        left = image[r, max(0, c - aperture_half - buffer - background):max(0, c - aperture_half - buffer)]
        right = image[r, min(n_cols, c + aperture_half + buffer + 1):min(n_cols, c + aperture_half + buffer + background + 1)]
        both = np.concatenate([left, right])
        if both.size:
            sky[r] = np.median(both)
    return sky


def sky_offsets(sky, poly, sky_lines, half_window=8):
    """
    Measured minus expected wavelength (A) of each night-sky line in the sky spectrum,
    using the arc solution `poly` (no heliocentric correction: sky lines are at rest
    relative to the telescope). Returns {line: offset} for lines that could be fit.
    """
    n = len(sky)
    rows_all = np.arange(n)
    lam = poly(rows_all)
    out = {}
    for line in sky_lines:
        if not (min(lam) < line < max(lam)):
            continue
        r0 = int(np.argmin(np.abs(lam - line)))
        lo, hi = max(0, r0 - half_window), min(n, r0 + half_window + 1)
        x, y = rows_all[lo:hi], sky[lo:hi]
        if not np.all(np.isfinite(y)):
            continue
        bck0 = np.median(y)
        try:
            popt, _ = curve_fit(_gauss, x, y, p0=[max(y.max() - bck0, 1.0), r0, 1.5, bck0], maxfev=2000,
                                bounds=([0, lo, 0.5, -np.inf], [np.inf, hi - 1, 5.0, np.inf]))
        except (RuntimeError, ValueError):
            continue
        # Require a real detection: amplitude well above the local scatter
        scatter = mad_std(y - _gauss(x, *popt))
        if popt[0] < 5 * max(scatter, 1e-6):
            continue
        out[line] = float(poly(popt[1]) - line)
    return out


# ---------------------------------------------------------------------------
# reference solution files
# ---------------------------------------------------------------------------

def format_reference(ref):
    """JSON text for a reference solution, one [row, wavelength] pair per line."""
    head = {k: v for k, v in ref.items() if k not in ("lines", "path")}
    text = json.dumps(head, indent=1)[:-2] + ',\n "lines": [\n'
    text += ",\n".join(f"  [{r}, {l}]" for r, l in ref["lines"])
    return text + "\n ]\n}\n"
