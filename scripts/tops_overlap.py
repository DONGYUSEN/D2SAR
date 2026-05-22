"""tops_overlap — Materialize top/bottom overlap slices between adjacent burst pairs.

Algorithm:
  1. For each adjacent pair in CommonBurstSelection.pairs:
       - top overlap: lines in burst i that overlap with burst i+1
         (burst i azimuth time is earlier → lines near burst i's end)
       - bottom overlap: lines in burst i+1 that overlap with burst i
         (burst i+1 azimuth time is later → lines near burst i+1's start)
  2. Compute valid-window intersection per sample to get first_sample/num_samples.
  3. Convert sensing time boundaries to line indices via _sensing_to_line.

No imports from strip/tops_insar backends.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np

from scripts.tops_model import (
    BurstRadarGrid,
    CommonBurstPair,
    CommonBurstSelection,
    OverlapPair,
    OverlapSlice,
)


UTC = timezone.utc

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "materialize_overlaps",
    "overlaps_to_dict",
    "write_overlaps_json",
    "read_overlap_window",
]


def materialize_overlaps(common: CommonBurstSelection) -> list[OverlapPair]:
    """Materialize top/bottom overlap slices for all adjacent burst pairs.

    Parameters
    ----------
    common : CommonBurstSelection
        Common-burst selection from ``match_common_bursts``.

    Returns
    -------
    list[OverlapPair]
        One OverlapPair per adjacent burst pair.  An empty list is returned
        when there are fewer than 2 common bursts.

    Raises
    ------
    ValueError
        If ``common`` has fewer than 1 burst pair.
    """
    if common.number_of_common_bursts == 0:
        raise ValueError("Empty common burst selection")

    pairs: list[OverlapPair] = []

    for i in range(len(common.pairs) - 1):
        top_slice = _extract_overlap_slice(common, i, is_top=True)
        bot_slice = _extract_overlap_slice(common, i, is_top=False)

        # Only include non-empty slices
        if top_slice.num_lines > 0 or bot_slice.num_lines > 0:
            pairs.append(
                OverlapPair(
                    pair_index=i,
                    top=top_slice,
                    bottom=bot_slice,
                )
            )

    return pairs


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_overlap_slice(
    common: CommonBurstSelection,
    pair_idx: int,
    is_top: bool,
) -> OverlapSlice:
    """Extract the overlap slice between adjacent burst pair_idx and pair_idx+1.

    The overlap is computed from the **sensing-time intersection** between
    the reference burst of pair_idx and the reference burst of pair_idx+1.
    This matches the ISCE2 ``runSubsetOverlaps.py`` algorithm:

        overlap_start = burst_b.sensing_start   (later burst first line)
        overlap_end   = min(burst_a.sensing_stop, burst_b.sensing_stop)

    Top slice (is_top=True)  → lines at the END   of the earlier burst (burst_a),
    where it overlaps with burst_b.
    Bottom slice (is_top=False) → lines at the START of the later burst (burst_b),
    where it overlaps with burst_a.

    Both slices have the same ``num_lines`` so the ESD double-difference
    ``top_ifg × conj(bot_ifg)`` compares the same physical ground area.

    Parameters
    ----------
    common : CommonBurstSelection
    pair_idx : int
        Index of the first adjacent pair.
    is_top : bool
        True → slice from burst at pair_idx  (azimuth-earlier).
        False → slice from burst at pair_idx+1 (azimuth-later).

    Returns
    -------
    OverlapSlice
        Coordinates of the overlap region, always relative to the
        **reference burst of the selected pair's SLC window**.
        ``num_lines`` may be 0 if the windows do not intersect.
    """
    pair_a = common.pairs[pair_idx]
    pair_b = common.pairs[pair_idx + 1]

    burst_a = pair_a.reference  # earlier burst (lower azimuth time)
    burst_b = pair_b.reference  # later burst

    dt = burst_a.azimuth_time_interval  # same for both bursts in a swath

    # --- ISCE2-style sensing-time overlap computation ---
    # Line in burst_a where burst_b's line 0 starts
    b0_in_a = round(
        (burst_b.identity.sensing_start - burst_a.identity.sensing_start)
        .total_seconds() / dt
    )
    # Where burst_b's valid window starts, expressed in burst_a's coordinates
    ol_start_in_a = b0_in_a + burst_b.valid_window.first_line
    # Overlap length bounded by both valid windows
    ol_end_in_a = min(
        burst_a.valid_line_stop,
        ol_start_in_a + burst_b.valid_window.num_lines,
    )
    n_lines = max(0, ol_end_in_a - ol_start_in_a)

    if is_top:
        # Slice from the earlier burst — overlap is near its END
        target = burst_a
        first_line = ol_start_in_a
    else:
        # Slice from the later burst — overlap is near its START (valid window start)
        target = burst_b
        first_line = burst_b.valid_line_start

    num_lines = n_lines

    # Sample intersection from valid windows
    a_fs = burst_a.valid_window.first_sample
    a_ns = burst_a.valid_window.num_samples
    b_fs = burst_b.valid_window.first_sample
    b_ns = burst_b.valid_window.num_samples
    first_sample = max(a_fs, b_fs)
    last_sample  = min(a_fs + a_ns, b_fs + b_ns)
    num_samples  = max(0, last_sample - first_sample)

    # Sensing time for the slice
    sensing_start = target.azimuth_time_at_line(first_line)
    line_stop_dt = target.azimuth_time_at_line(first_line + num_lines if num_lines > 0 else first_line)
    if line_stop_dt <= sensing_start:
        line_stop_dt = sensing_start + timedelta(microseconds=1)

    return OverlapSlice(
        burst_pair=pair_a if is_top else pair_b,
        is_top=is_top,
        first_line=first_line,
        num_lines=num_lines,
        first_sample=first_sample,
        num_samples=num_samples,
        sensing_start=sensing_start,
        sensing_stop=line_stop_dt,
    )


def _sensing_to_line(burst: BurstRadarGrid, sensing: datetime) -> int:
    """Convert a UTC sensing time to an absolute line index in the burst.

    Parameters
    ----------
    burst : BurstRadarGrid
    sensing : datetime
        UTC sensing time.

    Returns
    -------
    int
        Line index (0-based relative to the full measurement image).
    """
    dt = (sensing - burst.identity.sensing_start).total_seconds()
    return round(dt / burst.azimuth_time_interval)


def _line_valid_range(burst: BurstRadarGrid) -> tuple[int, int]:
    """Return (first, stop) line indices of the valid window (absolute coords).

    The stop index is one-past-the-last valid line.
    """
    return burst.valid_line_start, burst.valid_line_stop


def _valid_window_intersect(b1: BurstRadarGrid, b2: BurstRadarGrid) -> Tuple[int, int]:
    """Intersect the valid windows of two bursts in the sample dimension.

    Parameters
    ----------
    b1 : BurstRadarGrid
    b2 : BurstRadarGrid

    Returns
    -------
    tuple[int, int]
        (first_sample, num_samples) of the intersection.
        ``num_samples`` is 0 if there is no overlap.
    """
    vw1 = b1.valid_window
    vw2 = b2.valid_window

    first = max(vw1.first_sample, vw2.first_sample)
    last = min(vw1.sample_stop, vw2.sample_stop)

    num = max(0, last - first)
    return first, num


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------

def overlaps_to_dict(pairs: Sequence[OverlapPair]) -> dict:
    """Convert a sequence of OverlapPairs to a JSON-serialisable dict.

    Output format matches the PLAN.md specification:

    .. code-block:: python

        {
            "overlap_count": 2,
            "overlaps": [
                {
                    "pair_index": 0,
                    "is_top": True,
                    "first_line": 1300,
                    "num_lines": 200,
                    "first_sample": 500,
                    "num_samples": 24000,
                    "sensing_start": "2024-01-01T00:00:01.000Z",
                    "sensing_stop": "2024-01-01T00:00:01.500Z"
                },
                { ... bottom slice ... }
            ]
        }
    """
    slices: list[dict] = []

    for op in pairs:
        for is_top, slc in [(True, op.top), (False, op.bottom)]:
            slices.append({
                "pair_index": op.pair_index,
                "is_top": is_top,
                "first_line": slc.first_line,
                "num_lines": slc.num_lines,
                "first_sample": slc.first_sample,
                "num_samples": slc.num_samples,
                "sensing_start": slc.sensing_start.isoformat().replace("+00:00", "Z"),
                "sensing_stop": slc.sensing_stop.isoformat().replace("+00:00", "Z"),
            })

    return {
        "overlap_count": len(slices),
        "overlaps": slices,
    }


def write_overlaps_json(pairs: Sequence[OverlapPair], path: Path) -> None:
    """Write ``pairs`` to ``path`` as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(overlaps_to_dict(pairs), fh, indent=2)
        fh.write("\n")


# ---------------------------------------------------------------------------
# GDAL-based window read
# ---------------------------------------------------------------------------

def read_overlap_window(
    tiff_path: Path,
    overlap_slice: OverlapSlice,
    *, band: int = 1,
) -> np.ndarray | None:
    """Read the specified overlap window pixels from a Sentinel-1 TOPS full-swath TIFF.

    Parameters
    ----------
    tiff_path : Path
        Path to the full-swath TIFF file.
    overlap_slice : OverlapSlice
        OverlapSlice defining the window to read (absolute line/sample coordinates).
    band : int, default=1
        Band number to read (1-indexed for GDAL).

    Returns
    -------
    np.ndarray or None
        Array of shape (num_lines, num_samples) with dtype float32/float64 or
        complex64 for complex (ENVI) data.  Returns None on failure (missing file,
        GDAL unavailable, or invalid dimensions).
    """
    try:
        from osgeo import gdal
    except ImportError:
        log.warning("GDAL not available; read_overlap_window returns None")
        return None

    if not tiff_path.exists():
        log.warning("TIFF file not found: %s", tiff_path)
        return None

    if overlap_slice.num_lines <= 0 or overlap_slice.num_samples <= 0:
        log.warning(
            "Invalid overlap dimensions: num_lines=%d, num_samples=%d",
            overlap_slice.num_lines,
            overlap_slice.num_samples,
        )
        return None

    if overlap_slice.first_line < 0 or overlap_slice.first_sample < 0:
        log.warning(
            "Negative window origin: first_line=%d, first_sample=%d",
            overlap_slice.first_line,
            overlap_slice.first_sample,
        )
        return None

    try:
        ds = gdal.Open(str(tiff_path), gdal.GA_ReadOnly)
        if ds is None:
            log.warning("GDAL failed to open: %s", tiff_path)
            return None

        if band < 1 or band > ds.RasterCount:
            log.warning("Invalid band %d for %s", band, tiff_path)
            return None

        if overlap_slice.first_sample + overlap_slice.num_samples > ds.RasterXSize:
            return None
        if overlap_slice.first_line + overlap_slice.num_lines > ds.RasterYSize:
            return None

        band_obj = ds.GetRasterBand(band)
        if band_obj is None:
            log.warning("Missing band %d in %s", band, tiff_path)
            return None

        data = band_obj.ReadAsArray(
            overlap_slice.first_sample,
            overlap_slice.first_line,
            overlap_slice.num_samples,
            overlap_slice.num_lines,
        )
        if data is None:
            log.warning("ReadAsArray returned None for window in: %s", tiff_path)
            return None

        return np.asarray(data)

    except Exception as exc:  # pragma: no cover
        log.warning("Error reading overlap window from %s: %s", tiff_path, exc)
        return None
