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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence, Tuple

from scripts.tops_model import (
    BurstRadarGrid,
    CommonBurstPair,
    CommonBurstSelection,
    OverlapPair,
    OverlapSlice,
)


UTC = timezone.utc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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
    """Extract the top or bottom overlap slice between pair_idx and pair_idx+1.

    Overlap geometry (with reference burst = common.pairs[i].reference):
      - "top" burst: burst at index i  (earlier azimuth time)
      - "bottom" burst: burst at index i+1 (later azimuth time)

    Line range convention:
      - top overlap:  [burst2.valid_line_start,  burst1.valid_line_start - 1]
        (upper part of burst1, lines that precede burst2's start)
      - bottom overlap: [burst2.valid_line_stop + 1, burst1.valid_line_stop]
        (lower part of burst1, lines that follow burst2's end)

    Parameters
    ----------
    common : CommonBurstSelection
    pair_idx : int
        Index of the first burst in the pair (pairs[pair_idx] ↔ pairs[pair_idx+1]).
    is_top : bool
        True → extract top overlap slice (from burst at index pair_idx).
        False → extract bottom overlap slice (from burst at index pair_idx).

    Returns
    -------
    OverlapSlice
        Coordinates of the overlap region.  ``num_lines`` may be 0 if the
        bursts do not overlap in the valid window.
    """
    pair_a = common.pairs[pair_idx]
    pair_b = common.pairs[pair_idx + 1]

    # Reference bursts at the overlap interface
    burst1 = pair_a.reference  # earlier in azimuth (top)
    burst2 = pair_b.reference  # later in azimuth (bottom)

    if is_top:
        # Top overlap: lines from burst1 that overlap with burst2
        # line range = [burst2.valid_line_start, burst1.valid_line_start - 1]
        first_line = burst2.valid_line_start
        line_stop = burst1.valid_line_start
    else:
        # Bottom overlap: lines from burst1 that overlap with burst2
        # line range = [burst2.valid_line_stop + 1, burst1.valid_line_stop]
        first_line = burst2.valid_line_stop + 1
        line_stop = burst1.valid_line_stop

    num_lines = max(0, line_stop - first_line)

    # Valid-window sample intersection
    first_sample, num_samples = _valid_window_intersect(burst1, burst2)

    # Sensing time bounds: derive from line indices via azimuth_time_at_line.
    # OverlapSlice requires sensing_stop > sensing_start.  Guard against
    # numerical equality (e.g. zero-width line range or very close timestamps).
    sensing_start_dt = burst1.azimuth_time_at_line(first_line)
    line_stop_dt = burst1.azimuth_time_at_line(line_stop)
    if line_stop_dt <= sensing_start_dt:
        line_stop_dt = sensing_start_dt + timedelta(microseconds=1)

    return OverlapSlice(
        burst_pair=pair_a,
        is_top=is_top,
        first_line=first_line,
        num_lines=num_lines,
        first_sample=first_sample,
        num_samples=num_samples,
        sensing_start=sensing_start_dt,
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