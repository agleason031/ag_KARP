"""
Find and group KOSMOS frames from their FITS headers.

Science frames (IMAGETYP = Object) are grouped by object (OBJNAME) and instrument
setup (disperser, slit, binning). Each group gets:
  - bias frames with the same binning,
  - flats with the same setup,
  - the argon arc (IMAGETYP = Comp, ARGON = on) with the same setup that is nearest
    in time to the group's science frames.
The [frames] section of the config can restrict or override any of these.
"""
import glob
import os
import re
from dataclasses import dataclass, field

import numpy as np
from astropy.io import fits
from astropy.time import Time

LAMPS = ("ARGON", "NEON", "KRYPTON")


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
    flat: list # paths
    arc: str # path
    arc_num: int
    notes: list = field(default_factory=list) # warnings worth showing the user
    out_name: str = "" # output subdirectory name

    def describe(self):
        disp, slit, binning = self.setup
        nums = lambda paths: ", ".join(str(frame_number(p)) for p in paths)
        lines = [
            f"{self.objid}  (disperser {disp}, slit {slit}, binning {binning}) -> {self.out_name}/",
            f"  science: {', '.join(str(n) for n in self.science)}",
            f"  bias:    {nums(self.bias)}",
            f"  flat:    {nums(self.flat)}",
            f"  arc:     {self.arc_num}",
        ]
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)


def frame_number(path):
    m = re.search(r"(\d+)\.fits?$", os.path.basename(path))
    return int(m.group(1)) if m else None


def read_frame(path):
    h = fits.getheader(path)
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


def scan(raw_dir, pattern="*.fits"):
    """Read the headers of all matching frames in raw_dir, sorted by frame number."""
    paths = sorted(glob.glob(os.path.join(raw_dir, pattern)))
    if not paths:
        raise FileNotFoundError(f"No files matching {pattern} in {raw_dir}")
    frames = [read_frame(p) for p in paths]
    missing = [f.path for f in frames if f.num is None]
    if missing:
        raise ValueError(f"Can't get a frame number from these file names: {missing}")
    return sorted(frames, key=lambda f: f.num)


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

        if fcfg["arc"] is not None:
            arc = pick([fcfg["arc"]], "arc")[0]
        else:
            arcs = [f for f in usable if f.imagetyp == "comp" and f.setup == setup and "ARGON" in f.lamps]
            mid = np.median([f.mjd for f in sci])
            arc = min(arcs, key=lambda f: abs(f.mjd - mid)) if arcs else None
            if arc is not None and len(arcs) > 1:
                notes.append(f"{len(arcs)} argon arcs match; using {arc.num}, the nearest in time")
            if arc is not None and abs(arc.mjd - mid) * 24 > 1:
                notes.append(f"arc {arc.num} is {abs(arc.mjd - mid) * 24:.1f} h from the science frames")

        missing = [name for name, val in (("bias", bias), ("flat", flat), ("argon arc", arc)) if not val]
        if missing:
            problems.append(f"{objid} {setup}: no matching {', '.join(missing)} frames; skipped")
            continue
        groups.append(Group(
            objid=objid, setup=setup,
            science={f.num: f.path for f in sci},
            bias=[f.path for f in bias], flat=[f.path for f in flat],
            arc=arc.path, arc_num=arc.num, notes=notes,
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
    frames = scan(cfg["paths"]["raw_dir"], cfg["paths"]["pattern"])
    return build_groups(frames, cfg["frames"])
