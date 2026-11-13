"""
Find and group KOSMOS frames from their FITS headers.

Science frames (IMAGETYP = Object) are grouped by object (OBJNAME) and instrument
setup (disperser, slit, binning). Each group gets:
  - bias frames with the same binning,
  - flats with the same setup (none is allowed: the flat-field correction is skipped),
  - arcs (IMAGETYP = Comp with a lamp on) with the same setup: for each lamp
    combination, the arc nearest in time to the science frames plus the other arcs
    with those lamps taken within ARC_SEQUENCE_HOURS of it (repeated exposures, which
    are median-combined; the lamps' combined arcs are then added together).
The [frames] section of the config can restrict or override any of these.
Files that aren't raw frames (wrong shape, e.g. IRAF .ms.fits extractions) are skipped.
"""
import glob
import os
import re
from dataclasses import dataclass, field

import numpy as np
from astropy.io import fits
from astropy.time import Time

LAMPS = ("ARGON", "NEON", "KRYPTON")
ARC_SEQUENCE_HOURS = 1.0 # arcs with the same lamps this close to the chosen one are combined with it


@dataclass
class Frame:
    num: int
    path: str
    imagetyp: str # lower case: bias, flat, object, comp, ...
    objname: str
    setup: tuple # (disperser, slit, binning)
    lamps: frozenset # lamps that are on
    mjd: float # start of exposure
    exptime: float

    @property
    def binning(self):
        return self.setup[2]


@dataclass
class Group:
    objid: str
    setup: tuple
    science: dict # frame number -> path
    bias: list # paths
    flat: list # paths (may be empty: no flat-field correction)
    arcs: list # paths
    notes: list = field(default_factory=list) # warnings worth showing the user
    out_name: str = "" # output subdirectory name

    def describe(self):
        disp, slit, binning = self.setup
        nums = lambda paths: ", ".join(str(frame_number(p)) for p in paths) or "none"
        lines = [
            f"{self.objid}  (disperser {disp}, slit {slit}, binning {binning}) -> {self.out_name}/",
            f"  science: {', '.join(str(n) for n in self.science)}",
            f"  bias:    {nums(self.bias)}",
            f"  flat:    {nums(self.flat)}",
            f"  arc:     {nums(self.arcs)}",
        ]
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)


def frame_number(path):
    m = re.search(r"(\d+)\.fits?$", os.path.basename(path))
    return int(m.group(1)) if m else None


def read_frame(path, h=None):
    h = fits.getheader(path) if h is None else h
    scale = str(h.get("TIMESYS", "UTC")).strip().lower()
    t = Time(h["DATE-OBS"], scale=scale if scale in ("tai", "utc") else "utc")
    binning = f"{h.get('BINX', 1)}x{h.get('BINY', 1)}"
    return Frame(
        num=frame_number(path),
        path=path,
        imagetyp=str(h.get("IMAGETYP", "")).strip().lower(),
        objname=str(h.get("OBJNAME", "")).strip(),
        setup=(str(h.get("DISPERSR", "")).strip(), str(h.get("SLIT", "")).strip(), binning),
        lamps=frozenset(l for l in LAMPS if str(h.get(l, "off")).strip().lower() == "on"),
        mjd=t.utc.mjd,
        exptime=float(h.get("EXPTIME", 0.0)),
    )


def scan(raw_dir, pattern="*.fits", raw_width=None):
    """
    Read the headers of all matching frames in raw_dir, sorted by frame number.
    `pattern` may use ** to search subdirectories (e.g. "**/*.fits"). Files that aren't
    2D images `raw_width` columns wide (the raw frame width, overscan included) are
    skipped. Returns (frames, skipped), skipped being [(path, reason)].
    """
    paths = sorted(glob.glob(os.path.join(raw_dir, pattern), recursive=True))
    if not paths:
        raise FileNotFoundError(f"No files matching {pattern} in {raw_dir}")
    frames, skipped = [], []
    for p in paths:
        h = fits.getheader(p)
        if h.get("NAXIS") != 2 or (raw_width is not None and h.get("NAXIS1") != raw_width):
            shape = "x".join(str(h.get(f"NAXIS{i}")) for i in range(h.get("NAXIS", 0), 0, -1)) or "no data"
            skipped.append((p, f"not a raw frame (shape {shape})"))
            continue
        frames.append(read_frame(p, h))
    missing = [f.path for f in frames if f.num is None]
    if missing:
        raise ValueError(f"Can't get a frame number from these file names: {missing}")
    nums = [f.num for f in frames]
    dupes = sorted({n for n in nums if nums.count(n) > 1})
    if dupes:
        raise ValueError(f"Frame numbers {dupes} appear in more than one file; narrow paths.pattern "
                         "or use [frames] exclude")
    return sorted(frames, key=lambda f: f.num), skipped


def build_groups(frames, fcfg):
    """
    Group science frames and attach calibrations. `fcfg` is the [frames] config section.
    Returns (groups, problems): problems lists objects that couldn't be set up.
    """
    exclude = set(fcfg["exclude"])
    by_num = {f.num: f for f in frames}
    usable = [f for f in frames if f.num not in exclude]

    def pick(nums, kind):
        missing = [n for n in nums if n not in by_num]
        if missing:
            raise ValueError(f"[frames] {kind}: frames {missing} not found")
        return [by_num[n] for n in nums]

    science = [f for f in usable if f.imagetyp == "object"]
    if fcfg["science"]:
        science = pick(fcfg["science"], "science")
    if fcfg["objects"]:
        science = [f for f in science if f.objname in fcfg["objects"]]

    keys = sorted({(f.objname, f.setup) for f in science})
    groups, problems = [], []
    for objid, setup in keys:
        sci = [f for f in science if f.objname == objid and f.setup == setup]
        notes = []

        if fcfg["bias"]:
            bias = pick(fcfg["bias"], "bias")
        else:
            bias = [f for f in usable if f.imagetyp == "bias" and f.binning == setup[2]]
        for f in bias:
            if f.lamps:
                notes.append(f"bias {f.num} has lamp(s) {', '.join(sorted(f.lamps))} on in its header")

        if fcfg["flat"]:
            flat = pick(fcfg["flat"], "flat")
        else:
            flat = [f for f in usable if f.imagetyp == "flat" and f.setup == setup]

        if not flat:
            notes.append("no flats: the flat-field correction is skipped")

        mid = np.median([f.mjd for f in sci])
        if fcfg["arc"]:
            arcs = pick(fcfg["arc"], "arc")
        else:
            arcs = []
            candidates = [f for f in usable if f.imagetyp == "comp" and f.setup == setup and f.lamps]
            for lamps in sorted({f.lamps for f in candidates}, key=sorted):
                same = [f for f in candidates if f.lamps == lamps]
                nearest = min(same, key=lambda f: abs(f.mjd - mid))
                arcs += [f for f in same if abs(f.mjd - nearest.mjd) * 24 <= ARC_SEQUENCE_HOURS]
            arcs.sort(key=lambda f: f.num)
        if arcs:
            gap = min(abs(f.mjd - mid) for f in arcs) * 24
            if gap > 1:
                notes.append(f"the nearest arc is {gap:.1f} h from the science frames")

        missing = [name for name, val in (("bias", bias), ("arc", arcs)) if not val]
        if missing:
            problems.append(f"{objid} {setup}: no matching {', '.join(missing)} frames; skipped")
            continue
        groups.append(Group(
            objid=objid, setup=setup,
            science={f.num: f.path for f in sci},
            bias=[f.path for f in bias], flat=[f.path for f in flat],
            arcs=[f.path for f in arcs], notes=notes,
        ))

    # Output directory per group: the object name, plus the setup if an object
    # was observed with more than one setup
    counts = {}
    for g in groups:
        counts[g.objid] = counts.get(g.objid, 0) + 1
    for g in groups:
        name = g.objid if counts[g.objid] == 1 else f"{g.objid}_{g.setup[0]}_{g.setup[1]}_{g.setup[2]}"
        g.out_name = re.sub(r"[^A-Za-z0-9._+-]", "_", name)
    return groups, problems


def discover(cfg):
    """Scan cfg['paths']['raw_dir'] and return (groups, problems)."""
    raw_width = max(o1 for _, o1 in cfg["detector"]["amp_overscan"]) # overscan runs to the frame edge
    frames, skipped = scan(cfg["paths"]["raw_dir"], cfg["paths"]["pattern"], raw_width)
    groups, problems = build_groups(frames, cfg["frames"])
    return groups, problems + [f"{os.path.relpath(p, cfg['paths']['raw_dir'])}: {why}" for p, why in skipped]
