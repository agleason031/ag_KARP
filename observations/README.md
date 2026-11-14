# observations/

Put raw KOSMOS data here. Everything in this folder except this file is ignored by git.

One subdirectory per object, holding that object's science frames and, ideally, its
calibrations (searched recursively, so a `cals/` folder inside is fine). The subdirectory's
name is used for the object's output folder and file names:

```
observations/
  J0543/              science frames of J0543 (+ its biases, flats, arcs, in any subfolder)
  G191B2B/            a flux standard: its own subdirectory, like any other object
  cals/               no science frames: shared calibrations for every object
```

A subdirectory with no science frames (`IMAGETYP = Object`), such as `cals/` above, and any
files directly in `observations/`, are shared calibrations. An object uses them for each kind
(bias, flat, arc) it has none of in its own subdirectory. The usual matching still applies
(biases by binning; flats and arcs by disperser, slit and binning; arcs nearest in time).

Use it with:

```toml
[paths]
raw_dir = "observations"
output_dir = "reductions"
layout = "objects"
```

`python -m karp my_config.toml --list` shows what KARP found before anything is reduced.
A subdirectory holding several objects (e.g. a whole night) also works: each object then gets
`<subdirectory>_<OBJNAME>` as its name.
