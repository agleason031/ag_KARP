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

[paths] layout says how raw_dir is organized:
  "night"    all frames in raw_dir (pattern may include ** for subdirectories); output
             folders are named after OBJNAME.
  "objects"  each subdirectory of raw_dir holds one object's frames, searched
             recursively, and its name is used for the output folder and files.
             Subdirectories with no science frames (e.g. cals/), and files directly in
             raw_dir, are a shared pool of calibrations: an object uses them for any kind
             (bias, flat, arc) it has none of in its own subdirectory.
"""
import glob
import os
import re
from dataclasses import dataclass, field

import numpy as np
from astropy.io import fits
from astropy.time import Time

from . import detector

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
    prefix: str = "" # start of the output file names (default: objid)
    source: str = "" # raw_dir subdirectory the frames came from (layout = "objects")

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


def scan(raw_dir, pattern="*.fits", det=None, required=True):
    """
    Read the headers of all matching frames in raw_dir, sorted by frame number.
    `pattern` may use ** to search subdirectories (e.g. "**/*.fits"). Files that aren't
    raw KOSMOS frames (not 2D, or a size that doesn't fit the detector layout for their
    binning; det is the [detector] config) are skipped. Returns (frames, skipped),
    skipped being [(path, reason)]. With required=False, no matching files is not an error.
    """
    paths = sorted(glob.glob(os.path.join(raw_dir, pattern), recursive=True))
    if not paths and required:
        raise FileNotFoundError(f"No files matching {pattern} in {raw_dir}")
    frames, skipped = [], []
    for p in paths:
        h = fits.getheader(p)
        shape = "x".join(str(h.get(f"NAXIS{i}")) for i in range(h.get("NAXIS", 0), 0, -1)) or "no data"
        if h.get("NAXIS") != 2:
            skipped.append((p, f"not a raw frame (shape {shape})"))
            continue
        if det is not None:
            try:
                geom = detector.frame_geometry(h, det)
                if max(c1 for _, c1 in geom["amp_overscan"]) > h["NAXIS1"]:
                    raise detector.GeometryError("the [detector] columns run past the frame edge")
            except detector.GeometryError as err:
                skipped.append((p, f"not a raw frame (shape {shape}): {err}"))
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


def build_groups(frames, fcfg, pool=(), need_cals=True):
    """
    Group science frames and attach calibrations. `fcfg` is the [frames] config section.
    `pool`: shared calibration frames, used for a kind (bias, flat, arc) only when
    `frames` has none that match; frame numbers in [frames] refer to `frames` only.
    need_cals: biases and arcs are required (they are only used when frames are reduced).
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

    def auto(kind, select, notes):
        """Matching frames of this object, or else from the shared pool."""
        own = [f for f in usable if select(f)]
        if own or not pool:
            return own
        shared = [f for f in pool if select(f)]
        if shared:
            notes.append(f"{kind} from the shared calibrations: "
                         + ", ".join(sorted({os.path.basename(os.path.dirname(f.path)) + "/" for f in shared})))
        return shared

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
            bias = auto("bias", lambda f: f.imagetyp == "bias" and f.binning == setup[2], notes)
        for f in bias:
            if f.lamps:
                notes.append(f"bias {f.num} has lamp(s) {', '.join(sorted(f.lamps))} on in its header")

        if fcfg["flat"]:
            flat = pick(fcfg["flat"], "flat")
        else:
            flat = auto("flat", lambda f: f.imagetyp == "flat" and f.setup == setup, notes)

        if not flat:
            notes.append("no flats: the flat-field correction is skipped")

        mid = np.median([f.mjd for f in sci])
        if fcfg["arc"]:
            arcs = pick(fcfg["arc"], "arc")
        else:
            arcs = []
            candidates = auto("arcs", lambda f: f.imagetyp == "comp" and f.setup == setup and bool(f.lamps), notes)
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
        if missing and need_cals:
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
        g.out_name = safe_name(name)
        g.prefix = safe_name(g.objid) # file names: "2005 GO136" -> 2005_GO136_combined.fits
    return groups, problems


def safe_name(name):
    """A name usable for files and folders."""
    return re.sub(r"[^A-Za-z0-9._+-]", "_", name)


def discover_object_dirs(cfg):
    """
    layout = "objects": one object per subdirectory of raw_dir, plus shared calibrations
    (see the module docstring). Returns (groups, problems, skipped files).
    """
    raw, det, fcfg = cfg["paths"]["raw_dir"], cfg["detector"], cfg["frames"]
    if not os.path.isdir(raw):
        raise FileNotFoundError(f"[paths] raw_dir {raw} is not a directory")
    pattern = cfg["paths"]["pattern"]
    sub_pattern = pattern if "**" in pattern else os.path.join("**", pattern)

    pool, problems = [], []
    top, skipped = scan(raw, pattern.replace("**/", ""), det, required=False)
    for f in top:
        if f.imagetyp == "object":
            problems.append(f"{os.path.basename(f.path)}: science frame directly in {raw}, not in an "
                            "object's subdirectory; skipped")
        else:
            pool.append(f)
    objects = {}
    for d in sorted(os.listdir(raw)):
        if d.startswith(".") or not os.path.isdir(os.path.join(raw, d)):
            continue
        frames, sk = scan(os.path.join(raw, d), sub_pattern, det, required=False)
        skipped += sk
        if any(f.imagetyp == "object" for f in frames):
            objects[d] = frames
        else:
            pool += frames
    if not objects:
        raise FileNotFoundError(f"No subdirectory of {raw} has science frames (IMAGETYP = Object); "
                                'with [paths] layout = "objects" each object needs its own subdirectory')

    groups = []
    for d, frames in objects.items():
        # [frames] objects may name the subdirectory instead of the OBJNAME
        sub_fcfg = dict(fcfg, objects=[]) if d in fcfg["objects"] else fcfg
        found, probs = build_groups(frames, sub_fcfg, pool, cfg["steps"]["reduce"])
        problems += [f"{d}/: {p}" for p in probs]
        for g in found:
            # The subdirectory names the output, plus the object and setup if it holds several
            name = d if len(found) == 1 else f"{d}_{g.out_name}"
            g.out_name = g.prefix = safe_name(name)
            g.source = d
        groups += found
    return groups, problems, skipped


def discover(cfg):
    """Scan cfg['paths']['raw_dir'] and return (groups, problems)."""
    raw = cfg["paths"]["raw_dir"]
    if cfg["paths"]["layout"] == "objects":
        groups, problems, skipped = discover_object_dirs(cfg)
    else:
        frames, skipped = scan(raw, cfg["paths"]["pattern"], cfg["detector"])
        groups, problems = build_groups(frames, cfg["frames"], need_cals=cfg["steps"]["reduce"])
    return groups, problems + [f"{os.path.relpath(p, raw)}: {why}" for p, why in skipped]
