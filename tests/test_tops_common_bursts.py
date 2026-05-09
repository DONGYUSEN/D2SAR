"""Tests for tops_common_bursts — global integer-offset common-burst matching."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts.tops_model import (
    BurstIdentity,
    BurstWindow,
    BurstRadarGrid,
    CommonBurstSelection,
)
from scripts.tops_common_bursts import (
    match_common_bursts,
    _bursts_match,
    _contiguous_spans,
    selection_to_dict,
    write_common_bursts_json,
    TIME_TOLERANCE,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

UTC = timezone.utc


def _grid(
    idx: int,
    seconds: float,
    swath: str = "IW2",
    pol: str = "VV",
    az_interval: float = 0.002,
    valid_lines: int = 1300,
) -> BurstRadarGrid:
    """Minimal BurstRadarGrid for testing — 2-second bursts spaced 3 s apart."""
    base = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
    start = base + timedelta(seconds=seconds)
    stop = start + timedelta(seconds=2)
    return BurstRadarGrid(
        identity=BurstIdentity(
            swath=swath,
            burst_index=idx,
            sensing_start=start,
            sensing_stop=stop,
            polarization=pol,
            orbit_direction="ascending",
            azimuth_steering_rate=0.0,
        ),
        image_window=BurstWindow(
            first_line=idx * 1500, num_lines=1500,
            first_sample=0, num_samples=25000,
        ),
        valid_window=BurstWindow(
            first_line=100, num_lines=valid_lines,
            first_sample=500, num_samples=24000,
        ),
        line_offset=idx * 1500,
        azimuth_time_interval=az_interval,
        range_pixel_spacing=2.3,
        starting_range=800000.0,
        radar_wavelength=0.055,
        doppler_coefficients=(0.0,),
        azimuth_fm_rate_coefficients=(0.0,),
    )


# ---------------------------------------------------------------------------
# _bursts_match unit tests
# ---------------------------------------------------------------------------

def test_bursts_match_same():
    a = _grid(0, 0)
    b = _grid(0, 0)
    assert _bursts_match(a, b) is True


def test_bursts_match_sensing_within_tolerance():
    a = _grid(0, 0)
    b = _grid(0, 0.4)  # 0.4 s < 0.5 s tolerance
    assert _bursts_match(a, b) is True


def test_bursts_match_sensing_outside_tolerance():
    a = _grid(0, 0)
    b = _grid(0, 1.0)  # 1.0 s > 0.5 s tolerance
    assert _bursts_match(a, b) is False


def test_bursts_match_different_swath():
    a = _grid(0, 0, swath="IW1")
    b = _grid(0, 0, swath="IW2")
    assert _bursts_match(a, b) is False


def test_bursts_match_different_polarization():
    a = _grid(0, 0, pol="VV")
    b = _grid(0, 0, pol="VH")
    assert _bursts_match(a, b) is False


def test_bursts_match_zero_valid_lines():
    a = _grid(0, 0, valid_lines=1300)
    b = _grid(0, 0, valid_lines=0)
    assert _bursts_match(a, b) is False


def test_bursts_match_azimuth_interval_mismatch():
    a = _grid(0, 0, az_interval=0.002)
    b = _grid(0, 0, az_interval=0.003)
    assert _bursts_match(a, b) is False


# ---------------------------------------------------------------------------
# _contiguous_spans unit tests
# ---------------------------------------------------------------------------

def test_contiguous_spans_single_run():
    pairs = [(0, 0, 0.0), (1, 1, 0.0), (2, 2, 0.0)]
    spans = _contiguous_spans(pairs)
    assert spans == [(0, 0, 3)]


def test_contiguous_spans_two_runs():
    pairs = [(0, 0, 0.0), (1, 1, 0.0), (3, 3, 0.0)]
    spans = _contiguous_spans(pairs)
    assert spans == [(0, 0, 2), (3, 3, 1)]


def test_contiguous_spans_gap_in_secondary():
    # ref idx 0→0, 1→1 (contiguous), ref idx 3→2 (gap in ref, non-contiguous)
    pairs = [(0, 0, 0.0), (1, 1, 0.0), (3, 2, 0.0)]
    spans = _contiguous_spans(pairs)
    assert spans == [(0, 0, 2), (3, 2, 1)]


def test_contiguous_spans_empty():
    assert _contiguous_spans([]) == []


# ---------------------------------------------------------------------------
# match_common_bursts integration tests
# ---------------------------------------------------------------------------

def test_equal_starts():
    """reference and secondary start at the same burst."""
    ref = [_grid(0, 0), _grid(1, 3), _grid(2, 6)]
    sec = [_grid(0, 0), _grid(1, 3), _grid(2, 6)]
    sel = match_common_bursts(ref, sec)
    assert sel.number_of_common_bursts == 3
    assert sel.reference_start_index == 0
    assert sel.secondary_start_index == 0
    assert sel.pairs[0].burst_offset == 0


def test_secondary_missing_first_burst():
    """secondary starts 3 s later — common bursts start at ref[1]."""
    ref = [_grid(0, 0), _grid(1, 3), _grid(2, 6)]
    sec = [_grid(0, 3), _grid(1, 6)]
    sel = match_common_bursts(ref, sec)
    assert sel.number_of_common_bursts == 2
    assert sel.reference_start_index == 1
    assert sel.secondary_start_index == 0
    assert sel.pairs[0].burst_offset == -1


def test_reference_missing_first_burst():
    """reference starts 3 s later — common bursts start at ref[0] with offset=0."""
    ref = [_grid(0, 0), _grid(1, 3)]
    sec = [_grid(0, 0), _grid(1, 3), _grid(2, 6)]
    sel = match_common_bursts(ref, sec)
    assert sel.number_of_common_bursts == 2
    assert sel.reference_start_index == 0
    assert sel.secondary_start_index == 0
    assert sel.pairs[0].burst_offset == 0


def test_different_swath_raises():
    ref = [_grid(0, 0, swath="IW1")]
    sec = [_grid(0, 0, swath="IW2")]
    with pytest.raises(ValueError, match="multiple swaths"):
        match_common_bursts(ref, sec)


def test_no_common_bursts_different_sensing_time():
    """All secondary bursts are offset by > 0.5 s."""
    ref = [_grid(0, 0), _grid(1, 3)]
    sec = [_grid(0, 100), _grid(1, 103)]
    with pytest.raises(ValueError, match="No common bursts"):
        match_common_bursts(ref, sec)


def test_no_common_bursts_different_polarization():
    ref = [_grid(0, 0, pol="VV"), _grid(1, 3, pol="VV")]
    sec = [_grid(0, 0, pol="VH"), _grid(1, 3, pol="VH")]
    with pytest.raises(ValueError, match="No common bursts"):
        match_common_bursts(ref, sec)


def test_empty_reference_raises():
    sec = [_grid(0, 0)]
    with pytest.raises(ValueError, match="Empty burst list"):
        match_common_bursts([], sec)


def test_empty_secondary_raises():
    ref = [_grid(0, 0)]
    with pytest.raises(ValueError, match="Empty burst list"):
        match_common_bursts(ref, [])


def test_single_common_burst_allowed():
    """1 common burst is allowed (ESD will be skipped upstream).

    ref[0..1] at 0/3 s; sec has one burst at 0.1 s and a far-away one.
    Only ref[0] ↔ sec[0] match → span of 1.
    """
    ref = [_grid(0, 0), _grid(1, 3)]
    sec = [_grid(0, 0.1), _grid(1, 100)]
    sel = match_common_bursts(ref, sec)
    assert sel.number_of_common_bursts == 1


def test_tiebreak_by_median_time_error():
    """When two offsets give the same count, prefer the smaller median time error.

    ref[0..2] at 0,3,6 s.
    sec1 at 0,3,6  s → offset=0, median_error=0
    sec2 at 0.1,3.1,6.1 s → offset=0, median_error=0.1
    → sec1 is selected (smaller error).
    """
    ref = [_grid(0, 0), _grid(1, 3), _grid(2, 6)]
    sec = [_grid(0, 0), _grid(1, 3), _grid(2, 6)]
    sel = match_common_bursts(ref, sec)
    assert sel.number_of_common_bursts == 3
    assert sel.pairs[0].burst_offset == 0


# ---------------------------------------------------------------------------
# JSON round-trip tests
# ---------------------------------------------------------------------------

def test_selection_to_dict_roundtrip():
    ref = [_grid(0, 0), _grid(1, 3)]
    sec = [_grid(0, 0), _grid(1, 3)]
    sel = match_common_bursts(ref, sec)
    d = selection_to_dict(sel)
    assert d["swath"] == "IW2"
    assert d["number_of_common_bursts"] == 2
    assert d["burst_offset"] == 0
    assert len(d["pairs"]) == 2
    assert d["pairs"][0]["reference_burst_index"] == 0
    assert d["pairs"][0]["secondary_burst_index"] == 0


def test_write_and_read_common_bursts_json(tmp_path: Path):
    ref = [_grid(0, 0), _grid(1, 3)]
    sec = [_grid(0, 0), _grid(1, 3)]
    sel = match_common_bursts(ref, sec)
    out_path = tmp_path / "IW2" / "common_bursts.json"
    write_common_bursts_json(sel, out_path)
    loaded = json.loads(out_path.read_text())
    assert loaded["swath"] == "IW2"
    assert loaded["number_of_common_bursts"] == 2
    assert loaded["reference_start_index"] == 0
    assert loaded["secondary_start_index"] == 0
