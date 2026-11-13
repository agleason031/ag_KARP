"""
KOSMOS detector layout for a raw frame: which columns of each amplifier hold data
and which hold overscan, for any binning.

The layout comes from the frame's own header when it can be trusted: DSECnn (data)
and BSECnn (bias/overscan) give each amplifier's sections in raw-frame pixels. Some
headers carry the unbinned (1x1) sections whatever the binning (e.g. 2x1 frames with
BSEC starting at column 2049 of a 1124-column frame); then the layout is computed from
BINX/BINY and the frame size instead, using the KOSMOS readout format:
  - two amplifiers splitting the CCD (4096 x 2048 unbinned, rows x columns) at its
    middle column; amplifier 1 reads the left half,
  - each amplifier's data columns (1024 / BINX of them) side by side, followed by the
    overscan strips (one per amplifier, in the same order, a fixed number of columns
    that doesn't change with binning: 50 for KOSMOS),
  - 4096 / BINY rows.
[detector] amp_data / amp_overscan in the config override both.
"""
import re

CCD_COLS, CCD_ROWS, N_AMPS = 2048, 4096, 2
_SECTION = re.compile(r"\[\s*(\d+)\s*:\s*(\d+)\s*,\s*(\d+)\s*:\s*(\d+)\s*\]")


class GeometryError(ValueError):
    pass


def parse_section(text):
    """IRAF/FITS section '[x1:x2,y1:y2]' (1-based, inclusive) -> (c0, c1, r0, r1), 0-based, end-exclusive."""
    m = _SECTION.fullmatch(str(text).strip())
    if not m:
        raise GeometryError(f"Can't read section {text!r}")
    x1, x2, y1, y2 = map(int, m.groups())
    return min(x1, x2) - 1, max(x1, x2), min(y1, y2) - 1, max(y1, y2)


def binning(header):
    return int(header.get("BINX", header.get("CCDBIN1", 1))), int(header.get("BINY", header.get("CCDBIN2", 1)))


def from_header_sections(header):
    """
    Layout from DSECnn/BSECnn, or None if the header has no such sections or they
    don't fit the frame (data sections must tile the columns before the overscan, cover
    the same rows, and lie inside the frame).
    """
    n_cols, n_rows = header.get("NAXIS1"), header.get("NAXIS2")
    keys = sorted(k for k in header if re.fullmatch(r"DSEC\d+", k))
    if not keys or any(f"B{k[1:]}" not in header for k in keys):
        return None
    try:
        amps = [(parse_section(header[k]), parse_section(header[f"B{k[1:]}"])) for k in keys]
    except GeometryError:
        return None
    amps.sort(key=lambda a: a[0][0]) # left to right
    for (dc0, dc1, dr0, dr1), (bc0, bc1, br0, br1) in amps:
        if not (0 <= dc0 < dc1 <= n_cols and 0 <= bc0 < bc1 <= n_cols and 0 <= dr0 < dr1 <= n_rows
                and 0 <= br0 < br1 <= n_rows):
            return None
    rows = {(d[2], d[3]) for d, _ in amps}
    data_edges = [(d[0], d[1]) for d, _ in amps]
    tiled = all(data_edges[i][1] == data_edges[i + 1][0] for i in range(len(data_edges) - 1))
    if len(rows) != 1 or not tiled or data_edges[0][0] != 0:
        return None
    return {"amp_data": [[d[0], d[1]] for d, _ in amps], "amp_overscan": [[b[0], b[1]] for _, b in amps],
            "rows": list(rows.pop()), "source": "header DSEC/BSEC"}


def from_binning(header):
    """Layout computed from BINX/BINY and the frame size (see the module docstring)."""
    binx, biny = binning(header)
    n_cols, n_rows = header.get("NAXIS1"), header.get("NAXIS2")
    per_amp = CCD_COLS // N_AMPS // binx
    n_data = per_amp * N_AMPS
    overscan = (n_cols - n_data) // N_AMPS if n_cols else 0
    if overscan <= 0 or n_data + overscan * N_AMPS != n_cols or n_rows != CCD_ROWS // biny:
        raise GeometryError(f"A {n_cols} x {n_rows} frame doesn't match the KOSMOS layout for "
                            f"binning {binx}x{biny} ({n_data} data columns + overscan, "
                            f"{CCD_ROWS // biny} rows); set [detector] amp_data and amp_overscan")
    return {"amp_data": [[i * per_amp, (i + 1) * per_amp] for i in range(N_AMPS)],
            "amp_overscan": [[n_data + i * overscan, n_data + (i + 1) * overscan] for i in range(N_AMPS)],
            "rows": [0, n_rows], "source": f"computed from binning {binx}x{biny}"}


def settings(det):
    """The [detector] settings that define how raw frames become trimmed electron images."""
    return {k: det[k] for k in ("amp_data", "amp_overscan", "overscan_skip", "gain", "overscan_order")}


def frame_geometry(header, det):
    """
    Layout of one raw frame: dict with amp_data and amp_overscan ([[c0, c1], ...],
    0-based, end-exclusive, the first det["overscan_skip"] overscan columns of each
    strip left out), rows [r0, r1] to keep, gain (e-/DN per amplifier),
    overscan_order, and source (where the layout came from).
    det is the [detector] config section.
    """
    if det["amp_data"] is not None:
        geom = {"amp_data": det["amp_data"], "amp_overscan": det["amp_overscan"],
                "rows": [0, header.get("NAXIS2")], "source": "[detector] config"}
    else:
        geom = from_header_sections(header) or from_binning(header)
    skip = det["overscan_skip"]
    geom["amp_overscan"] = [[o0 + skip, o1] for o0, o1 in geom["amp_overscan"]]
    if any(o1 - o0 < 5 for o0, o1 in geom["amp_overscan"]):
        raise GeometryError(f"Overscan strips {geom['amp_overscan']} are too narrow after skipping {skip} columns")
    if len(det["gain"]) != len(geom["amp_data"]):
        raise GeometryError(f"[detector] gain has {len(det['gain'])} entries for {len(geom['amp_data'])} amplifiers")
    geom.update(gain=det["gain"], overscan_order=det["overscan_order"])
    return geom


def describe(geom):
    amps = "; ".join(f"amp {i + 1}: data columns {d0}-{d1 - 1}, overscan {o0}-{o1 - 1}"
                     for i, ((d0, d1), (o0, o1)) in enumerate(zip(geom["amp_data"], geom["amp_overscan"])))
    return f"Detector layout ({geom['source']}): {amps}; rows {geom['rows'][0]}-{geom['rows'][1] - 1}"


def to_binned(value, binning, odd=False, minimum=1):
    """
    A size given in unbinned pixels, in pixels of a frame with this binning (rounded,
    at least `minimum`; made odd if `odd`, for running filters).
    """
    n = max(minimum, int(round(value / binning)))
    return n + 1 if odd and n % 2 == 0 else n


def setup_binning(setup):
    """(binx, biny) from a discover setup tuple (disperser, slit, "BINXxBINY")."""
    binx, biny = setup[2].split("x")
    return int(binx), int(biny)
