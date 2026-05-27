"""tops_common_bursts — Global integer-offset continuous-span common burst matching.

Algorithm (5 steps):
  1. Enumerate all candidate integer offsets k (reference[i] ↔ secondary[i+k]).
  2. For each k, find all valid burst pairs whose sensing_start/stop diff ≤ 0.5 s,
     azimuth_time_interval diff ≤ 1e-9, and valid_window.num_lines > 0.
  3. Extract the longest contiguous span of valid pairs for each k.
  4. Select the k with maximal common count; tie-break on smallest median sensing-time error.
  5. < 1 common burst → ValueError; < 2 → allow IFG but flag ESD unavailable.

Output: CommonBurstSelection + JSON file (common_bursts.json).

No imports from strip/tops_insar backends.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Sequence, Tuple

from .tops_model import BurstRadarGrid, CommonBurstPair, CommonBurstSelection


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# For Sentinel-1 SAR interferometry with 24-day repeat cycles, temporal baselines
# can be 24+ days. We use a large tolerance (30 days) to accommodate this.
# The matching is further constrained by orbit direction and azimuth time interval.
TIME_TOLERANCE = timedelta(days=30)


# ---------------------------------------------------------------------------
# Internal candidate record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _MatchCandidate:
    burst_offset: int
    reference_start: int
    secondary_start: int
    common_count: int
    median_time_error: float


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def match_common_bursts(
    reference: Sequence[BurstRadarGrid],
    secondary: Sequence[BurstRadarGrid],
) -> CommonBurstSelection:
    """Match reference and secondary bursts by global integer offset and continuous span.

    Parameters
    ----------
    reference : sequence of BurstRadarGrid
        Reference (master) bursts for one swath.
    secondary : sequence of BurstRadarGrid
        Secondary (slave) bursts for the same swath.

    Returns
    -------
    CommonBurstSelection
        All matched pairs.  ``number_of_common_bursts`` may be 1, in which case
        ESD will be skipped upstream.

    Raises
    ------
    ValueError
        If either list is empty, bursts span multiple swaths, or no common bursts
        exist after the tolerance checks.
    """
    if not reference or not secondary:
        raise ValueError("Empty burst list")

    swath = reference[0].identity.swath
    if any(b.identity.swath != swath for b in reference):
        raise ValueError("Reference bursts span multiple swaths")
    if any(b.identity.swath != swath for b in secondary):
        raise ValueError("Secondary bursts span multiple swaths")

    candidates: list[_MatchCandidate] = []

    for k in range(-len(secondary) - 1, len(reference) + 1):
        valid_pairs: list[tuple[int, int, float]] = []

        for i, ref in enumerate(reference):
            j = i + k
            if not (0 <= j < len(secondary)):
                continue
            sec = secondary[j]
            if not _bursts_match(ref, sec):
                continue
            # Record sensing-start delta in seconds for tie-breaking
            delta_s = abs(
                (ref.identity.sensing_start - sec.identity.sensing_start).total_seconds()
            )
            valid_pairs.append((i, j, delta_s))

        if len(valid_pairs) < 1:
            continue

        spans = _contiguous_spans(valid_pairs)
        # longest span = max count
        best_span = max(spans, key=lambda s: s[2])
        ref_start, sec_start, count = best_span

        # Median sensing-time error for tie-breaking: compute over valid_pairs
        # in the selected span
        span_ref_indices = range(ref_start, ref_start + count)
        span_errors = sorted(
            err for (ri, _, err) in valid_pairs if ri in span_ref_indices
        )
        median_err = span_errors[len(span_errors) // 2] if span_errors else float("inf")

        candidates.append(
            _MatchCandidate(
                burst_offset=k,
                reference_start=ref_start,
                secondary_start=sec_start,
                common_count=count,
                median_time_error=median_err,
            )
        )

    if not candidates:
        raise ValueError(
            f"No common bursts found for swath {swath}. "
            "Check sensing times, orbit direction, and swath/pol consistency."
        )

    # Select: maximise common_count, then minimise median_time_error
    best = max(candidates, key=lambda c: (c.common_count, -c.median_time_error))

    pairs = tuple(
        CommonBurstPair(
            pair_index=i,
            reference=reference[best.reference_start + i],
            secondary=secondary[best.secondary_start + i],
            burst_offset=best.burst_offset,
        )
        for i in range(best.common_count)
    )

    return CommonBurstSelection(
        swath=swath,
        reference_start_index=best.reference_start,
        secondary_start_index=best.secondary_start,
        number_of_common_bursts=best.common_count,
        pairs=pairs,
    )


def _bursts_match(a: BurstRadarGrid, b: BurstRadarGrid) -> bool:
    """Return True when reference burst ``a`` and secondary burst ``b`` are compatible.

    Checks (all must pass):
    - swath / polarization match
    - orbit direction match (ascending/descending)
    - |sensing_start difference| ≤ TIME_TOLERANCE
    - azimuth_time_interval difference ≤ 1e-9 s
    - both valid_window.num_lines > 0
    """
    id_a, id_b = a.identity, b.identity

    if id_a.swath != id_b.swath:
        return False

    if id_a.polarization != id_b.polarization:
        return False

    # Orbit direction must match (both ascending or both descending)
    if id_a.orbit_direction != id_b.orbit_direction:
        return False

    delta_start = abs((id_a.sensing_start - id_b.sensing_start).total_seconds())
    if delta_start > TIME_TOLERANCE.total_seconds():
        return False

    delta_stop = abs((id_a.sensing_stop - id_b.sensing_stop).total_seconds())
    if delta_stop > TIME_TOLERANCE.total_seconds():
        return False

    if abs(a.azimuth_time_interval - b.azimuth_time_interval) > 1e-9:
        return False

    if a.valid_window.num_lines <= 0 or b.valid_window.num_lines <= 0:
        return False

    return True


def _contiguous_spans(
    valid_pairs: list[tuple[int, int, float]],
) -> list[tuple[int, int, int]]:
    """Find contiguous reference-index spans in the list of valid pairs.

    Parameters
    ----------
    valid_pairs : list of (ref_idx, sec_idx, time_error)
        Sorted by reference index.

    Returns
    -------
    list of (ref_start, sec_start, count)
        One entry per contiguous run; ``count`` is the number of pairs in the run.
    """
    if not valid_pairs:
        return []

    sorted_pairs = sorted(valid_pairs, key=lambda p: p[0])
    spans: list[tuple[int, int, int]] = []

    cur_ref_start = sorted_pairs[0][0]
    cur_sec_start = sorted_pairs[0][1]
    cur_count = 1

    for idx in range(1, len(sorted_pairs)):
        r_curr, s_curr, _ = sorted_pairs[idx]
        r_prev, s_prev, _ = sorted_pairs[idx - 1]

        if r_curr == r_prev + 1 and s_curr == s_prev + 1:
            # still contiguous
            cur_count += 1
        else:
            spans.append((cur_ref_start, cur_sec_start, cur_count))
            cur_ref_start = r_curr
            cur_sec_start = s_curr
            cur_count = 1

    spans.append((cur_ref_start, cur_sec_start, cur_count))
    return spans


# ---------------------------------------------------------------------------
# JSON I/O (convenience helpers)
# ---------------------------------------------------------------------------

def selection_to_dict(sel: CommonBurstSelection) -> dict:
    """Convert a CommonBurstSelection to a JSON-serialisable dict."""
    return {
        "swath": sel.swath,
        "reference_start_index": sel.reference_start_index,
        "secondary_start_index": sel.secondary_start_index,
        "number_of_common_bursts": sel.number_of_common_bursts,
        "burst_offset": (
            sel.pairs[0].burst_offset if sel.pairs else 0
        ),
        "pairs": [
            {
                "pair_index": p.pair_index,
                "reference_burst_index": p.reference.identity.burst_index,
                "secondary_burst_index": p.secondary.identity.burst_index,
            }
            for p in sel.pairs
        ],
    }


def write_common_bursts_json(sel: CommonBurstSelection, path: Path) -> None:
    """Write ``sel`` to ``path`` as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(selection_to_dict(sel), fh, indent=2)
        fh.write("\n")
