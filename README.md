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
git clone https://github.com/agleason031/ag_KARP.git
cd ag_KARP
conda env create -f environment.yml   # creates the "ag_karp" environment
conda activate ag_karp
pip install -e .                     # optional: adds the `karp` command
```

Requires Python ≥ 3.11. `pygame` is only needed if you turn on `[output] sound`.

## Quick start

1. Put the raw frames in [observations/](observations/README.md), one subdirectory per
   object (its name names the output), with shared calibrations in a subdirectory such as
   `cals/`. Copy [example_config.toml](example_config.toml), which reads `observations/` and
   writes to `reductions/` (both ignored by git). Every other setting has a default.
2. Check which frames KARP will use:
   ```bash
   python -m karp my_night.toml --list
   ```
3. Reduce:
   ```bash
   python -m karp my_night.toml
   ```

`karp my_night.toml` (after `pip install -e .`) and `python ag_KARP.py my_night.toml` do the
same thing.

KARP sorts frames by their headers (IMAGETYP, OBJNAME, DISPERSR, SLIT, binning and the lamp
keywords). Each object and setup becomes one group, with matching biases, flats and arcs.
For the arcs, KARP takes the nearest exposure for each lamp combination, plus any repeats
taken within an hour; repeats are median-combined, and the different lamps are added
together. If there are no flats, KARP skips the flat-field correction and says so. The
frame lists in `[frames]` can narrow or override all of this. Set `pattern = "**/*.fits"`
to include subdirectories. Files that aren't raw frames, such as IRAF `.ms.fits`
extractions, are skipped.

`[paths] layout` says how `raw_dir` is organized. With `"objects"` (as in the example config),
each subdirectory holds one object and gives its output folder and file names; subdirectories
without science frames, and files directly in `raw_dir`, are shared calibrations that an object
uses for any kind (bias, flat, arc) it has none of itself. With `"night"` (the default), all
frames are in `raw_dir` and each group is written to `output_dir/<OBJNAME>/`.

If one object fails, KARP logs the error, carries on with the others, lists the failures at
the end and exits with status 1. Python warnings from the libraries go to `KARP_log.txt`, not
the console.

## What each step does

| Step | Details |
|---|---|
| Calibration | Each amplifier's overscan is fit along the rows and subtracted, then the frame is trimmed and converted to electrons with that amplifier's gain. The master bias is a clipped median. The flats are bias-subtracted, scaled by their medians and median-combined; dividing by a 5×5 smoothed copy gives the pixel-to-pixel flat. The flat is set to 1 in the rows crossed by the grating's zeroth order, which are found automatically (`[flat]`). Master calibrations are saved in `calib/` and rebuilt when the input frames or settings change. |
| Cosmic rays | L.A.Cosmic on each 2D frame (`[cosmic_rays]`). |
| Trace | The star nearest the slit center is traced (see [Choosing the star](#choosing-the-star-in-the-slit)). Coarse peak tracking, Gaussian fits per row, then a clipped polynomial (`[trace]`). |
| Extraction | Sky fit per row across both background windows. Optimal extraction (Horne 1986) uses a smoothed data-derived profile and outlier rejection; boxcar is also available. Errors come from a per-pixel variance image (Poisson + read noise, through the flat). See `[extraction]`. |
| Wavelength calibration | The arc is extracted along each frame's trace and its lines are found and centroided. Features much wider or narrower than a typical line (ghosts, blends) and lines on the wing of a much brighter line are left out. They are identified against NIST line lists starting from a stored reference solution, then fit with a clipped polynomial (order chosen automatically by default). Night-sky lines are measured to check flexure, which can be reported or corrected. A heliocentric correction is applied to each frame from its header. See `[wavecal]`. |
| Combine | Frames are interpolated onto the best frame's wavelength grid and scaled to it with a smooth function of wavelength. The default combination is an inverse-variance weighted mean with outlier rejection based on the errors; plain mean and median are also available. See `[combine]`. |
| Flux calibration (optional) | Uses a spectrophotometric standard observed with the same disperser, slit and binning, recognized by its OBJNAME. Its combined spectrum is converted to a count rate per Å and corrected for extinction (APO curve, mid-exposure airmass). It is compared with the tabulated flux (HST CALSPEC) at matched resolution: at each wavelength the sharper of the two is smoothed to the other's resolution, after a fitted wavelength offset between the standard and its tabulated spectrum is taken out. The ratio is the sensitivity, which is fit with a spline that leaves out line cores and telluric bands. It is applied to every object with that setup, using the standard nearest in time. Objects with no matching standard are not flux calibrated. See `[fluxcal]`. |
| Normalization (optional) | A cubic spline continuum with asymmetric sigma clipping and a line mask (`[normalization]`). It isn't required for a valid reduction; turn it off with `[steps] normalize = false`. |

The steps in `[steps]` can be run separately. For example, `reduce = false` re-combines,
re-normalizes or re-flux-calibrates earlier output without redoing the extraction; the
biases, flats and arcs aren't needed for that, only the science frames' headers.

## Output

Each object gets its own folder, `output_dir/<name>/` (the subdirectory name, or OBJNAME).
File names start with the same name (`<obj>`), with any character other than letters, digits
and `._+-` replaced by `_` (OBJNAME `2005 GO136` gives `2005_GO136_combined.fits`):

```
<obj>_<n>_extracted.fits     each frame's spectrum
<obj>_combined.fits/.png     the combined spectrum
<obj>_normalized.fits/.png   continuum-normalized (if [steps] normalize)
<obj>_fluxed.fits/.png       flux-calibrated (if a standard was observed with this setup)
KARP_log.txt                 full log of the run (rewritten each run)
KARP_run_info.txt            the frames and every setting used
trace_choice.json            the star picked with --pick (reused by later runs)
calib/                       master bias, final flat; a standard's sensitivity function
diagnostics/                 frames.png (each frame / combined), continuum.png, sensfunc.png;
                             with verbose also the master bias and flat and zoomed views
diagnostics/frame_<n>/       per frame: slit.png (stars in the slit, which one is traced),
                             trace.png (fit, residuals, profile width), wavecal.png
                             (arc with identified lines, residuals), wavecal_lines.txt,
                             spectrum.png (flux, error, sky); with verbose also image.png (the
                             frame with its extraction windows) and cuts across the slit
```

| File | Contents |
|---|---|
| `<obj>_<n>_extracted.fits` | `Lam` (Å, air, heliocentric), `Flux`, `Sigma`, `Sky` (e⁻ per pixel), `Row` (detector row). The header holds the extraction method, heliocentric velocity, arc-fit RMS, arc line width and flexure offset. |
| `<obj>_combined.fits` | `Lam`, `Flux`, `Sigma`, `Mask`. The flux is in electrons on the reference frame's scale. The header holds the frames used, combine method and S/N at 5500 Å. |
| `<obj>_normalized.fits` | `Flux` and `Sigma` divided by the continuum, plus `Continuum`, `Mask`, and `ContFit` (1 = pixel used in the continuum fit). |
| `<obj>_fluxed.fits` | `Lam`, `Flux`, `Sigma` (erg s⁻¹ cm⁻² Å⁻¹), `Mask`. The header records the standard, exposure time and airmass. |
| `calib/<std>_sensfunc.fits` | In the standard's folder: the measured and fitted sensitivity (mag), the pixels used, and the tabulated flux on the same grid. |

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
log records which source was used ([karp/detector.py](karp/detector.py)). `[detector] amp_data` and
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

## Choosing the star in the slit

A long slit often holds more than one star. KARP lists the stars in each frame (in the log
and in `diagnostics/frame_<n>/slit.png`) and by default traces the one nearest the slit
center: `[trace] slit_center`, an unbinned CCD column, or the middle of the frame if that
isn't set. Set `select = "brightest"` to trace the brightest column instead, as KARP 1.0 did.
If a star brighter than the traced one is in the slit, KARP logs a warning.

To choose for yourself, either set the column per object in the config (unbinned CCD columns;
the names are as in `[frames] objects`):

```toml
[trace.cols]
j0653p0239 = 952
```

or let KARP ask while it runs:

```bash
python -m karp my_night.toml --pick              # ask for every object
python -m karp my_night.toml --pick j0741p0427   # only this one
```

On each object's first frame, KARP plots the slit profile, lists the stars with their offsets
from the slit center, and waits for a number, `c<column>` (a binned column), or Enter to keep
its own choice. The choice is used for all of that object's frames and saved to
`<output_dir>/<name>/trace_choice.json`, so later runs reuse it without asking. Delete the
file to forget the choice. In order of precedence: `--pick`, `[trace] cols`, the saved
choice, `[trace] col` (one column for every object), then `select`.

## Wavelength calibration data

- `karp/data/linelists/`: Ar I, Ne I and Kr I air wavelengths and intensities from the NIST
  Atomic Spectra Database.
- `karp/data/wavecal/kosmos_<disperser>.json`: a rough reference solution for each grating,
  used as the starting point for line identification. Only `kosmos_blue.json` (blue grating)
  exists so far.

To add a reference for another grating, start from a matched-line table, such as a KARP
`diagnostics/frame_<n>/wavecal_lines.txt` or a hand-identified table with columns
`row, wavelength, -, -, -, used`, and run:

```bash
python scripts/make_wavecal_reference.py OBJ/diagnostics/frame_61/wavecal_lines.txt \
    --disperser red --binning-y 1 --order 5 --lamps ARGON -o karp/data/wavecal/kosmos_red.json
```

## Flux standards and extinction

- `karp/data/standards/`: 15 HST CALSPEC standards (G191-B2B, BD+28 4211, Feige 34, Feige 110,
  HZ 2/4/43/44, GD 71, GD 153, BD+17 4708, AGK+81 266, GRW+70 5824, LDS 749B, HD 93521),
  converted to air wavelengths. `index.json` lists the OBJNAME aliases KARP recognizes; the
  match ignores case, spaces and punctuation. To add a CALSPEC star, add it to
  `scripts/fetch_calspec.py` and run that script. For a star that isn't in CALSPEC, set
  `[fluxcal] standard` to its OBJNAME and `reference` to a file of air wavelength (Å) and
  flux (erg s⁻¹ cm⁻² Å⁻¹).
- `karp/data/extinction/apo.txt`: the APO extinction curve (J. R. A. Davenport, via pykosmos).
- `karp/data/masks/standard.txt`: line cores and telluric bands left out of the sensitivity fit.

The shape of a fluxed spectrum is reliable. The absolute level is only as good as the match
in slit losses (seeing, centering) and transparency between the standard and the target.
CALSPEC's optical spectra (HST STIS G430L/G750L) have a resolution of 5.5 Å FWHM below
about 5700 Å and 9.8 Å above, usually coarser than KOSMOS. KARP takes this from each file's
FWHM column, or uses these values when a file has none. KOSMOS's resolution is taken from the
arc line widths unless `[fluxcal] resolution` is set; the result depends little on it. The
fitted offset (standard's lines minus tabulated) is logged and also serves as a check of the
wavelength calibration. Science spectra are never smoothed. The log reports a closure test:
the standard calibrated with its own sensitivity function, compared with its tabulated flux.

## Continuum mask

`karp/data/masks/stellar.txt` lists the regions left out of the continuum fit by default:
Balmer lines, Ca II H & K, Na D and the telluric O₂ bands. The columns are center and
half-width in Å. Stars with broad Balmer wings may need wider regions, which you can supply
through `mask` or `extra_mask`.

## Code layout

```
ag_KARP.py              thin wrapper: python ag_KARP.py config.toml
karp/                   the package
  cli.py                command line, logging, running each object (python -m karp)
  config.py             TOML settings, defaults and checks
  discover.py           sorting frames into objects/setups from their headers
  detector.py           amplifier layout from each header, binning conversions
  observation.py        heliocentric correction, mid-exposure airmass
  calibration.py        overscan/trim, arcs, master bias and flat, zeroth order
  pipeline.py           ObjectReduction: the steps for one object; OutputPaths
  trace.py              tracing the spectrum
  extract.py            sky fit, boxcar and optimal extraction
  wavecal.py            arc line identification, wavelength solution, sky-line check
  combine.py            scaling and combining frames
  normalize.py          continuum fit (optional step)
  fluxcal.py            sensitivity function and flux calibration (optional step)
  regions.py            line masks and spline knots shared by the two above
  plots.py              all plots
  paths.py              location of karp/data/
  data/                 line lists, wavelength references, masks, standards, extinction
scripts/                fetch_calspec.py, make_wavecal_reference.py
tests/unit/             unit tests on simulated data (pytest)
tests/regression/       end-to-end regression test
```

## Tests

Unit tests run in a few seconds and need no data: they build simulated frames, arcs and
spectra with a known answer (a tilted star on sky with cosmic rays, an argon arc from the
real line list, a standard star seen through a known sensitivity curve, ...) and check that
KARP recovers it. They also cover the settings checks, the detector layout and the frame
sorting. GitHub runs them on every push.

```bash
pip install -e ".[test]"
pytest
```

The regression test `tests/regression/regress.py` runs the whole pipeline on a reference data set and compares
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
- The Ne line list hasn't been tested on real arcs. The Kr list has been tested on one night
  with the blue grating and the 2.1″ slit; Kr has only two usable lines below 4270 Å, so an
  argon or neon arc as well makes the blue end of the solution firmer.
- The [O I] 5577 Å sky line comes out 0.4–0.6 Å below the 6300/6364 Å lines on both test
  nights (argon and krypton arcs), which points at the sky-line measurement, not the arc
  solution. The flexure check uses the median of the lines and defaults to report-only.
- Bright night-sky lines (e.g. [O I] 5577 Å, Na D) can leave residuals in the extracted
  spectrum, mainly for faint, sky-dominated targets: up to ~10σ at 5577 Å on the 2.1″-slit
  test night (J0543), ~4σ on the 0.87″ night. Ruled out as the main cause: the curvature of
  the lines across the slit (measured from the arc; sampling the background along it didn't
  help) and the slit illumination (5577 Å is flat along the slit to 0.3%).
  `[extraction] sky_order = 2` removes most of the 5577 Å residual but costs 6–18% S/N.
  Not yet explained: a step in the sky level at the amplifier boundary (column 512), which
  the right-hand background window crosses when the trace is near column 480, about
  −25 e⁻ on the J0543 frames (sky ~130–190 e⁻).
- The arc and sky line-finding windows are set in binned pixels. Their right size depends
  on the line width, which depends on the slit as much as on the binning. They work at 2×1
  with the 0.87″ slit and at 2×2 with the 2.1″ slit.

## Data credits

KARP ships with these data files (`karp/data/`):

- **Arc line lists** (`linelists/`): Ar I, Ne I and Kr I wavelengths and relative
  intensities from the NIST Atomic Spectra Database (Kramida, A., Ralchenko, Yu., Reader, J.
  & NIST ASD Team, NIST Atomic Spectra Database, https://physics.nist.gov/asd,
  doi:10.18434/T4W30F; retrieved 2026-11-13).
- **Flux standards** (`standards/`): spectra from the HST CALSPEC database (Bohlin, Gordon &
  Tremblay 2014, PASP 126, 711; Bohlin, Hubeny & Rauch 2020, AJ 160, 21;
  https://archive.stsci.edu/hlsps/reference-atlases/cdbs/current_calspec/),
  converted to air wavelengths with the formula of Morton (2000, ApJS 130, 403). Each file's
  header names the CALSPEC file it came from.
- **APO extinction curve** (`extinction/apo.txt`): the curve prepared by J. R. A. Davenport
  (2015, credited in the file; not a journal publication) and distributed with
  [pykosmos](https://github.com/jradavenport/pykosmos) (MIT license, included as
  `extinction/LICENSE-pykosmos.txt`).
- **Wavelength reference** (`wavecal/kosmos_blue.json`) and **line masks** (`masks/`): made
  for KARP.

Please cite these sources as well as KARP when they contribute to your results.

## License and citation

KARP is released under the BSD 3-Clause license ([LICENSE](LICENSE)). If you use it in
published work, please cite it using [CITATION.cff](CITATION.cff) (GitHub's "Cite this
repository" button gives BibTeX and APA formats).

KARP was created by Alex Gleason, based on original code by Chase L. Smith and Max Moe.
The optimal extraction follows Horne (1986, PASP 98, 609).
