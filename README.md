# KARP — KOSMOS Astronomical Reduction Pipeline

KARP reduces long-slit spectra from KOSMOS on the APO 3.5 m. Starting from a night of
raw frames, it produces a wavelength-calibrated 1D spectrum for each object, with
propagated errors. It can also flux-calibrate the spectrum when a spectrophotometric
standard was observed, and continuum-normalize it.

```
raw frames ─► sort by header ─► overscan, bias, flat ─► cosmic rays ─► trace
          ─► extraction (optimal or boxcar) ─► wavelength calibration (arc + sky check)
          ─► per-frame spectra ─► scale + combine ─► [optional] continuum normalization
                                                 └──► [optional] flux calibration (standard star)
```

## Installation

```bash
git clone git@github.com:agleason031/ag_KARP.git
cd ag_KARP
conda env create -f environment.yml   # creates the "ag_karp" environment
conda activate ag_karp
pip install -e .                     # optional: adds a `karp` command
```

Requires Python ≥ 3.11. `pygame` is only needed if you turn on `[output] sound`.

## Quick start

1. Copy [example_config.toml](example_config.toml) and set `[paths] raw_dir` (the folder
   of raw `k.NNNN.fits` frames) and `output_dir`. Every other setting has a default.
2. Check which frames KARP will use:
   ```bash
   python ag_KARP.py my_night.toml --list
   ```
3. Reduce:
   ```bash
   python ag_KARP.py my_night.toml        # or: karp my_night.toml
   ```

KARP sorts frames by their headers (IMAGETYP, OBJNAME, DISPERSR, SLIT, binning and the lamp
keywords). Each object and setup becomes one group, with matching biases, flats and arcs.
For the arcs, KARP takes the nearest exposure for each lamp combination, plus any repeats
taken within an hour; repeats are median-combined, and the different lamps are added
together. If there are no flats, KARP skips the flat-field correction and says so. The
frame lists in `[frames]` can narrow or override all of this. Set `pattern = "**/*.fits"`
to include subdirectories. Files that aren't raw frames, such as IRAF `.ms.fits`
extractions, are skipped. Each group is written to `output_dir/<OBJNAME>/`.

## What each step does

| Step | Details |
|---|---|
| Calibration | Each amplifier's overscan is fit along the rows and subtracted, then the frame is trimmed and converted to electrons with that amplifier's gain. The master bias is a clipped median. The flats are bias-subtracted, scaled by their medians and median-combined; dividing by a 5×5 smoothed copy gives the pixel-to-pixel flat. The flat is set to 1 in the rows crossed by the grating's zeroth order, which are found automatically (`[flat]`). Master calibrations are cached in `OUT/` and rebuilt when the input frames or settings change. |
| Cosmic rays | L.A.Cosmic on each 2D frame (`[cosmic_rays]`). |
| Trace | Coarse peak tracking, Gaussian fits per row, then a clipped polynomial (`[trace]`). |
| Extraction | Sky fit per row across both background windows. Optimal extraction (Horne 1986) uses a smoothed data-derived profile and outlier rejection; boxcar is also available. Errors come from a per-pixel variance image (Poisson + read noise, through the flat). See `[extraction]`. |
| Wavelength calibration | The arc is extracted along each frame's trace and its lines are found and centroided. They are identified against NIST line lists starting from a stored reference solution, then fit with a clipped polynomial (order chosen automatically by default). Night-sky lines are measured to check flexure, which can be reported or corrected. A heliocentric correction is applied to each frame from its header. See `[wavecal]`. |
| Combine | Frames are interpolated onto the best frame's wavelength grid and scaled to it with a smooth function of wavelength. The default combination is an inverse-variance weighted mean with outlier rejection based on the errors; plain mean and median are also available. See `[combine]`. |
| Flux calibration (optional) | Uses a spectrophotometric standard observed with the same disperser, slit and binning, recognized by its OBJNAME. Its combined spectrum is converted to a count rate per Å and corrected for extinction (APO curve, mid-exposure airmass). It is compared with the tabulated flux (HST CALSPEC) at matched resolution: at each wavelength the sharper of the two is smoothed to the other's resolution, after a fitted wavelength offset between the standard and its tabulated spectrum is taken out. The ratio is the sensitivity, which is fit with a spline that leaves out line cores and telluric bands. It is applied to every object with that setup, using the standard nearest in time. Objects with no matching standard are not flux calibrated. See `[fluxcal]`. |
| Normalization (optional) | A cubic spline continuum with asymmetric sigma clipping and a line mask (`[normalization]`). It isn't required for a valid reduction; turn it off with `[steps] normalize = false`. |

The steps in `[steps]` can be run separately. For example, `reduce = false` re-combines or
re-normalizes earlier output without redoing the extraction.

## Output

In `output_dir/<OBJNAME>/OUT/`:

| File | Contents |
|---|---|
| `<obj>_<n>_extracted.fits` | Spectrum of frame *n*: `Lam` (Å, air, heliocentric), `Flux`, `Sigma`, `Sky` (e⁻ per pixel), `Row` (detector row). The header holds the extraction method, heliocentric velocity, arc-fit RMS and flexure offset. |
| `<obj>_combined.fits` | Combined spectrum: `Lam`, `Flux`, `Sigma`, `Mask`. The flux is in electrons on the reference frame's scale. The header holds the frames used, combine method and S/N at 5500 Å. |
| `<obj>_fluxed.fits` | Only when a standard was observed with this setup: `Lam`, `Flux`, `Sigma` (erg s⁻¹ cm⁻² Å⁻¹), `Mask`. The header records the standard, exposure time and airmass. |
| `<std>_sensfunc.fits` | In the standard's folder: the measured and fitted sensitivity (mag), the pixels used, and the tabulated flux on the same grid. |
| `<obj>_normalized.fits` | Only when normalization is on: `Flux` and `Sigma` divided by the continuum, plus `Continuum`, `Mask`, and `ContFit` (1 = pixel used in the continuum fit). |
| `<obj>_wavecal_<n>.txt` | The matched arc lines and the solution, for checking by eye. |
| `<obj>_master_bias.fits`, `<obj>_final_flat.fits` | Cached master calibrations. |
| `KARP_log.txt` | Full log of the run, rewritten each run. |
| `KARP_run_info.txt` | The frames and every setting used. |
| `*.png` | Diagnostic plots. `[output] verbose = true` adds many more, in `ImageNumber_<n>/` too. |

**Mask bits** (a pixel can carry several):

| Bit | Meaning |
|---|---|
| 1 | Normalized flux above the `[normalization]` clip thresholds (flagged only; values unchanged) |
| 2 | No data |
| 4 | At least one frame was rejected as an outlier at this pixel during the weighted combine (informational) |
| 8 | Fluxed spectrum only: the sensitivity was extended linearly past the fitted range (at most `max_extrapolate` Å) |

S/N is the median `Flux/Sigma` per pixel and is reported in the log at 5500 Å and in 100 Å
bands.

## Configuration

[example_config.toml](example_config.toml) lists every setting with its default and a short
explanation. Only `[paths]` is required. Unknown settings or values of the wrong type stop KARP
with an error before anything runs.

**Detector layout and binning.** KARP works out each frame's column layout (each amplifier's
data and overscan columns) from its own header, so any binning works without changing the
config. It uses the header's DSEC/BSEC sections when they fit the frame. Some KOSMOS headers
(e.g. the 2×1 test data) carry the unbinned sections whatever the binning; in that case the
layout is computed from BINX/BINY and the frame size. KOSMOS has two amplifiers, each with
1024/BINX data columns, followed by 50 overscan columns per amplifier, which don't bin. The
log records which source was used ([detector.py](detector.py)). `[detector] amp_data` and
`amp_overscan` override both.

**Sizes are physical, so one config works at any binning.** Widths across the slit
(`[extraction] aperture_arcsec`, `buffer_arcsec`, `background_arcsec`) are in arcsec, converted
with `[detector] pixel_scale` (0.258″ per unbinned pixel) and the frames' BINX. Sizes along the
dispersion (`profile_window`, `[wavecal] search` and `match_tol`, `[flat] zeroth_pad` and
row ranges) are in unbinned rows, divided by BINY. `[trace] col` is an unbinned CCD column.
The log lists the resulting sizes in pixels.

The other `[detector]` defaults are two amplifiers with a gain of
0.6 e⁻/DN from the spec sheet, and 5.7 e⁻ read noise. The flats suggest the gain may be closer
to 0.72–0.77 e⁻/DN; check this for your data.

## Wavelength calibration data

- `calib_data/linelists/`: Ar I, Ne I and Kr I air wavelengths and intensities from the NIST
  Atomic Spectra Database.
- `calib_data/wavecal/kosmos_<disperser>.json`: a rough reference solution for each grating,
  used as the starting point for line identification. Only `kosmos_blue.json` (blue grating)
  exists so far.

To add a reference for another grating or binning, start from a matched-line table, such as
a KARP `OUT/<obj>_wavecal_<n>.txt` or a hand-identified table with columns
`row, wavelength, -, -, -, used`, and run:

```bash
python wavecal.py make-reference OUT/OBJ_wavecal_61.txt --disperser red --binning-y 1 \
    --order 5 --lamps ARGON -o calib_data/wavecal/kosmos_red.json
```

## Flux standards and extinction

- `calib_data/standards/`: 15 HST CALSPEC standards (G191-B2B, BD+28 4211, Feige 34, Feige 110,
  HZ 2/4/43/44, GD 71, GD 153, BD+17 4708, AGK+81 266, GRW+70 5824, LDS 749B, HD 93521),
  converted to air wavelengths. `index.json` lists the OBJNAME aliases KARP recognizes; the
  match ignores case, spaces and punctuation. To add a CALSPEC star, add it to
  `fetch_calspec.py` and run that script. For a star that isn't in CALSPEC, set
  `[fluxcal] standard` to its OBJNAME and `reference` to a file of air wavelength (Å) and
  flux (erg s⁻¹ cm⁻² Å⁻¹).
- `calib_data/extinction/apo.txt`: the APO extinction curve (Davenport 2015, via pykosmos).
- `calib_data/masks/standard.txt`: line cores and telluric bands left out of the sensitivity fit.

The shape of a fluxed spectrum is reliable. The absolute level is only as good as the match
in slit losses (seeing, centering) and transparency between the standard and the target.
CALSPEC's optical spectra (HST STIS G430L/G750L) have a resolution of 5.5 Å FWHM below
about 5700 Å and 9.8 Å above, usually coarser than KOSMOS. KARP takes this from each file's
FWHM column, or uses these values when a file has none. KOSMOS's resolution is taken from the
arc line widths unless `[fluxcal] resolution` is set; the result depends little on it. The
fitted offset (standard's lines minus tabulated) is logged and also serves as a check of the
wavelength calibration. Science spectra are never smoothed. The log reports a closure test: the standard calibrated with its own sensitivity function,
compared with its tabulated flux.

## Continuum mask

`calib_data/masks/stellar.txt` lists the regions left out of the continuum fit by default:
Balmer lines, Ca II H & K, Na D and the telluric O₂ bands. The columns are center and
half-width in Å. Stars with broad Balmer wings may need wider regions, which you can supply
through `mask` or `extra_mask`.

## Regression tests

`tests/regression/regress.py` runs the whole pipeline on a reference data set and compares
every intermediate and final array with a saved snapshot:

```bash
python tests/regression/regress.py compare                 # against tests/regression/reference/
python tests/regression/regress.py compare --set combine.method='"median"'
python tests/regression/regress.py make-reference --note "why the output changed"
```

The raw test frames are expected in `test_data/in/`, which isn't tracked by git. Runs are
written to `tests/regression/runs/`.

## Known limitations

- Only the blue grating has a reference wavelength solution (made at 1-row binning; other
  binnings are converted automatically).
- Arcs taken far from the science pointing can be offset by flexure. `[wavecal] flexure =
  "correct"` shifts the solution by the night-sky line offset, but that's a single shift and
  can't fix an offset that changes with wavelength. Arcs taken at the science pointing are best.
- The Ne and Kr line-list intensity cutoffs haven't been tested on real arcs.
- On the test data, the [O I] 5577 Å sky line sits about 0.4 Å off the arc solution while
  6300 and 6364 Å agree, so flexure correction defaults to report-only.
- The arc and sky line-finding windows are set in binned pixels. Their right size depends
  on the line width, which depends on the slit as much as on the binning. They work at 2×1
  with the 0.87″ slit and at 2×2 with the 2.1″ slit.
