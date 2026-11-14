"""Sorting frames into objects with their calibrations, from headers only."""
import pytest
from astropy.io import fits

from karp.config import merge
from karp.discover import Frame, build_groups, discover

FCFG = merge({"paths": {"raw_dir": "r", "output_dir": "o"}})["frames"]
BLUE = ("blue", "0.87-ctr", "2x2")
RED = ("red", "0.87-ctr", "2x2")


def frame(num, kind, obj="", setup=BLUE, lamps=(), mjd=60000.0, path=None):
    return Frame(num=num, path=path or f"cals/k.{num:04d}.fits", imagetyp=kind, objname=obj, setup=setup,
                 lamps=frozenset(lamps), mjd=mjd, exptime=60.0)


def night():
    return [
        frame(1, "bias"), frame(2, "bias"), frame(3, "bias", setup=("blue", "0.87-ctr", "1x1")),
        frame(4, "flat"), frame(5, "flat", setup=RED),
        frame(6, "comp", lamps=["ARGON"], mjd=60000.00), frame(7, "comp", lamps=["ARGON"], mjd=60000.02),
        frame(8, "comp", lamps=["ARGON"], mjd=60000.30), frame(9, "comp", lamps=["NEON"], mjd=60000.31),
        frame(10, "object", "STAR", mjd=60000.30), frame(11, "object", "STAR", mjd=60000.31),
        frame(12, "object", "OTHER", setup=RED, mjd=60000.5),
    ]


def nums(paths):
    return sorted(int(p.split(".")[-2]) for p in paths)


def test_calibrations_matched_by_setup_and_time():
    groups, problems = build_groups(night(), FCFG)
    star = next(g for g in groups if g.objid == "STAR")
    assert sorted(star.science) == [10, 11]
    assert nums(star.bias) == [1, 2]  # same binning only
    assert nums(star.flat) == [4]  # same setup only
    # per lamp combination: the nearest arc plus those within an hour of it
    assert nums(star.arcs) == [8, 9]
    # OTHER (red) has a flat but no arc with its setup
    assert [g.objid for g in groups] == ["STAR"]
    assert any("OTHER" in p and "arc" in p for p in problems)


def test_exclude_and_explicit_frames():
    fcfg = dict(FCFG, exclude=[8], science=[10])
    star = build_groups(night(), fcfg)[0][0]
    assert sorted(star.science) == [10]
    assert nums(star.arcs) == [6, 7, 9]  # 8 excluded: the next nearest argon arcs
    with pytest.raises(ValueError, match="not found"):
        build_groups(night(), dict(FCFG, bias=[99]))


def test_shared_pool_used_only_when_missing():
    own = [frame(10, "object", "STAR"), frame(20, "comp", lamps=["ARGON"], path="STAR/k.0020.fits")]
    pool = [frame(1, "bias"), frame(4, "flat"), frame(6, "comp", lamps=["ARGON"])]
    g = build_groups(own, FCFG, pool)[0][0]
    assert g.arcs == ["STAR/k.0020.fits"]  # its own arc wins
    assert nums(g.bias) == [1] and nums(g.flat) == [4]
    assert any("shared" in n for n in g.notes)


def test_calibrations_not_needed_without_reduce():
    only_science = [frame(10, "object", "STAR")]
    assert build_groups(only_science, FCFG)[0] == []
    assert len(build_groups(only_science, FCFG, need_cals=False)[0]) == 1


def test_file_names_made_safe():
    frames = night() + [frame(13, "object", "2005 GO136", mjd=60000.3)]
    g = next(g for g in build_groups(frames, FCFG)[0] if g.objid == "2005 GO136")
    assert g.out_name == g.prefix == "2005_GO136"


def write_frame(path, kind, obj="", lamps=(), date="2024-09-29T03:00:00"):
    """Header-only FITS file of a 2x2-binned KOSMOS frame."""
    h = fits.Header()
    h["SIMPLE"], h["BITPIX"], h["NAXIS"] = True, 16, 2
    h["NAXIS1"], h["NAXIS2"] = 1124, 2048
    h["BINX"], h["BINY"] = 2, 2
    h["IMAGETYP"], h["OBJNAME"], h["DISPERSR"], h["SLIT"] = kind, obj, "blue", "2.1-ctr"
    h["DATE-OBS"], h["EXPTIME"] = date, 60.0
    for lamp in ("ARGON", "NEON", "KRYPTON"):
        h[lamp] = "on" if lamp in lamps else "off"
    path.parent.mkdir(parents=True, exist_ok=True)
    h.tofile(path)


@pytest.mark.filterwarnings("ignore::astropy.utils.exceptions.AstropyUserWarning")  # header-only files
def test_objects_layout(tmp_path):
    raw = tmp_path / "observations"
    write_frame(raw / "J0543" / "j0543.0032.fits", "Object", "J0543")
    write_frame(raw / "J0543" / "cals" / "arc.0040.fits", "Comp", lamps=["KRYPTON"])
    write_frame(raw / "Std Star" / "g191.0024.fits", "Object", "G191-B2B")
    write_frame(raw / "cals" / "bias.0001.fits", "Bias")
    write_frame(raw / "cals" / "arc.0012.fits", "Comp", lamps=["ARGON"])
    write_frame(raw / "flat.0005.fits", "Flat")  # directly in raw_dir: shared
    write_frame(raw / "stray.0099.fits", "Object", "STRAY")  # science outside a subdirectory
    cfg = merge({"paths": {"raw_dir": str(raw), "output_dir": "o", "layout": "objects"}})
    groups, problems = discover(cfg)
    by_name = {g.out_name: g for g in groups}
    assert sorted(by_name) == ["J0543", "Std_Star"]
    j = by_name["J0543"]
    assert j.prefix == "J0543" and j.objid == "J0543" and j.source == "J0543"
    assert nums(j.arcs) == [40]  # its own arcs, not the shared ones
    assert nums(j.bias) == [1] and nums(j.flat) == [5]  # shared
    assert nums(by_name["Std_Star"].arcs) == [12]
    assert any("stray" in p for p in problems)

    # [frames] objects may name the subdirectory
    cfg["frames"]["objects"] = ["Std Star"]
    assert [g.out_name for g in discover(cfg)[0]] == ["Std_Star"]
