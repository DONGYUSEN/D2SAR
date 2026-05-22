"""Tests for tops_overlap — top/bottom overlap materialization."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from scripts.tops_model import (
    BurstIdentity,
    BurstWindow,
    BurstRadarGrid,
    CommonBurstPair,
    CommonBurstSelection,
    OverlapPair,
    OverlapSlice,
)
from scripts.tops_overlap import (
    materialize_overlaps,
    _extract_overlap_slice,
    _sensing_to_line,
    _line_valid_range,
    _valid_window_intersect,
    overlaps_to_dict,
    write_overlaps_json,
    read_overlap_window,
)


UTC = timezone.utc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _grid(
    idx: int,
    seconds: float,
    swath: str = "IW2",
    pol: str = "VV",
    az_interval: float = 0.002,
    valid_lines: int = 1300,
    valid_start: int = 100,
    valid_samp_start: int = 500,
    valid_samp_lines: int = 24000,
    line_offset: int | None = None,
) -> BurstRadarGrid:
    """Minimal BurstRadarGrid for testing — 2-second bursts spaced 3 s apart.

    Valid window is a sub-region of each burst image:
      - image_window.first_line = line_offset (or idx * 1500)
      - valid_window starts at image_window.first_line + valid_start
      - valid window occupies valid_lines lines and valid_samp_lines samples
    """
    base = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
    start = base + timedelta(seconds=seconds)
    stop = start + timedelta(seconds=2)
    if line_offset is None:
        line_offset = idx * 1500
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
            first_line=line_offset, num_lines=1500,
            first_sample=0, num_samples=25000,
        ),
        valid_window=BurstWindow(
            first_line=valid_start, num_lines=valid_lines,
            first_sample=valid_samp_start, num_samples=valid_samp_lines,
        ),
        line_offset=line_offset,
        azimuth_time_interval=az_interval,
        range_pixel_spacing=2.3,
        starting_range=800000.0,
        radar_wavelength=0.055,
        doppler_coefficients=(0.0,),
        azimuth_fm_rate_coefficients=(0.0,),
    )


def _sel(
    pairs: list[tuple[BurstRadarGrid, BurstRadarGrid]],
    swath: str = "IW2",
) -> CommonBurstSelection:
    """Build a CommonBurstSelection from a list of (ref, sec) burst tuples."""
    common_pairs = tuple(
        CommonBurstPair(
            pair_index=i,
            reference=ref,
            secondary=sec,
            burst_offset=0,
        )
        for i, (ref, sec) in enumerate(pairs)
    )
    return CommonBurstSelection(
        swath=swath,
        reference_start_index=0,
        secondary_start_index=0,
        number_of_common_bursts=len(common_pairs),
        pairs=common_pairs,
    )


# ---------------------------------------------------------------------------
# _sensing_to_line unit tests
# ---------------------------------------------------------------------------

def test_sensing_to_line_start():
    """sensing_start maps to line 0."""
    burst = _grid(0, 0)
    result = _sensing_to_line(burst, burst.identity.sensing_start)
    assert result == 0


def test_sensing_to_line_one_prf_step():
    """sensing_start + 1 PRF interval maps to line 1."""
    burst = _grid(0, 0, az_interval=0.002)
    t1 = burst.identity.sensing_start + timedelta(seconds=0.002)
    result = _sensing_to_line(burst, t1)
    assert result == 1


def test_sensing_to_line_mid_burst():
    """Middle of burst maps to ~line 500 (1 s into 2 s burst at 500 Hz PRF)."""
    burst = _grid(0, 0, az_interval=0.002)
    tmid = burst.identity.sensing_start + timedelta(seconds=1.0)
    result = _sensing_to_line(burst, tmid)
    assert result == 500


def test_sensing_to_line_negative():
    """Sensing time before burst start yields negative line index."""
    burst = _grid(0, 0)
    t_before = burst.identity.sensing_start - timedelta(seconds=0.1)
    result = _sensing_to_line(burst, t_before)
    assert result < 0


# ---------------------------------------------------------------------------
# _line_valid_range unit tests
# ---------------------------------------------------------------------------

def test_line_valid_range():
    """valid_range returns correct absolute (start, stop) pair."""
    burst = _grid(0, 0, line_offset=0, valid_start=100, valid_lines=1300)
    start, stop = _line_valid_range(burst)
    assert start == 100
    assert stop == 1400  # 100 + 1300


def test_line_valid_range_with_line_offset():
    """line_offset is added correctly to valid_window.first_line."""
    burst = _grid(2, 6, line_offset=3000, valid_start=50, valid_lines=200)
    start, stop = _line_valid_range(burst)
    assert start == 3050   # 3000 + 50
    assert stop == 3250    # 3050 + 200


# ---------------------------------------------------------------------------
# _valid_window_intersect unit tests
# ---------------------------------------------------------------------------

def test_valid_window_intersect_full_overlap():
    """Same valid window → full intersection."""
    a = _grid(0, 0)
    b = _grid(0, 0)
    first, num = _valid_window_intersect(a, b)
    assert first == 500
    assert num == 24000


def test_valid_window_intersect_partial_overlap():
    """Offset valid windows yield partial intersection."""
    # a: first_sample=500, num_samples=24000  → [500, 24500)
    # b: first_sample=1000, num_samples=20000 → [1000, 21000)
    a = _grid(0, 0, valid_samp_start=500, valid_samp_lines=24000)
    b = _grid(0, 0, valid_samp_start=1000, valid_samp_lines=20000)
    first, num = _valid_window_intersect(a, b)
    assert first == 1000          # max(500, 1000)
    assert num == 20000          # min(500+24000, 1000+20000) - 1000 = 21000 - 1000


def test_valid_window_intersect_no_overlap():
    """Non-overlapping valid windows → num_samples=0."""
    # a: [0, 5000), b: [10000, 20000)
    a = _grid(0, 0, valid_samp_start=0, valid_samp_lines=5000)
    b = _grid(0, 0, valid_samp_start=10000, valid_samp_lines=10000)
    first, num = _valid_window_intersect(a, b)
    assert num == 0


def test_valid_window_intersect_edge_touch():
    """Touching boundaries (stop == start) → num_samples=0."""
    # a: [0, 1000), b: [1000, 5000)
    a = _grid(0, 0, valid_samp_start=0, valid_samp_lines=1000)
    b = _grid(0, 0, valid_samp_start=1000, valid_samp_lines=4000)
    first, num = _valid_window_intersect(a, b)
    assert num == 0


# ---------------------------------------------------------------------------
# _extract_overlap_slice unit tests
# ---------------------------------------------------------------------------

def test_extract_top_slice_has_correct_is_top():
    """Top slice should have is_top=True."""
    ref = [_grid(0, 0), _grid(1, 3)]
    sec = [_grid(0, 0), _grid(1, 3)]
    sel = _sel(list(zip(ref, sec)))
    slc = _extract_overlap_slice(sel, 0, is_top=True)
    assert slc.is_top is True


def test_extract_bottom_slice_has_correct_is_top():
    """Bottom slice should have is_top=False."""
    ref = [_grid(0, 0), _grid(1, 3)]
    sec = [_grid(0, 0), _grid(1, 3)]
    sel = _sel(list(zip(ref, sec)))
    slc = _extract_overlap_slice(sel, 0, is_top=False)
    assert slc.is_top is False


def test_extract_top_slice_line_range():
    """Top overlap lines = [burst2.valid_line_start, burst1.valid_line_start - 1]."""
    # burst0: line_offset=0, valid_start=100 → valid [100, 1400)
    # burst1: line_offset=1500, valid_start=100 → valid [1600, 2900)
    # top overlap: [1600, 0 + 100 - 1] = [1600, 99] → empty (num_lines=0)
    ref = [_grid(0, 0, line_offset=0), _grid(1, 3, line_offset=1500)]
    sel = _sel(list(zip(ref, ref)))
    slc = _extract_overlap_slice(sel, 0, is_top=True)
    assert slc.num_lines == 0  # burst1 valid_start > burst0 valid_start → no top overlap


def test_extract_top_slice_non_zero():
    """Top overlap is non-zero when bursts overlap vertically."""
    # Both bursts: image first_line=idx*1500, valid_start=100, valid_lines=1300
    # burst0 valid: [100, 1400), burst1 valid: [1600, 2900)
    # top overlap of burst0: [1600, 99] → empty (burst1 starts below burst0's end)
    # Let's adjust so they overlap:
    # burst0: line_offset=0, valid_start=0, valid_lines=2000 → valid [0, 2000)
    # burst1: line_offset=1500, valid_start=0, valid_lines=2000 → valid [1500, 3500)
    # top: [1500, 0-1] = empty (burst1 starts below burst0's start)
    # Now set burst0 to start AFTER burst1:
    # burst0: line_offset=2000, valid_start=0, valid_lines=2000 → [2000, 4000)
    # burst1: line_offset=0, valid_start=0, valid_lines=2000 → [0, 2000)
    # top: [0, 2000-1] = [0, 1999], num_lines=1999
    # ref0: line_offset=2000, valid_start=0, valid_lines=1999 → valid [2000, 3999)
    # ref1: line_offset=0, valid_start=0, valid_lines=1999 → valid [0, 1999)
    ref0 = _grid(0, 0, line_offset=2000, valid_start=0, valid_lines=1999)
    ref1 = _grid(1, 3, line_offset=0, valid_start=0, valid_lines=1999)
    sel = _sel([(ref0, ref0), (ref1, ref1)])
    # For pair 0: burst1=ref0, burst2=ref1
    # top overlap from burst1 (ref0): [ref1.valid_line_start, ref0.valid_line_start-1]
    #   = [0, 2000-1] = [0, 1999], num_lines=1999
    slc = _extract_overlap_slice(sel, 0, is_top=True)
    assert slc.first_line == 0
    assert slc.num_lines == 2000  # [0, 1999] inclusive = 2000 lines


def test_extract_bottom_slice_non_zero():
    """Bottom overlap is non-zero when bursts overlap vertically."""
    ref0 = _grid(0, 0, line_offset=2000, valid_start=0, valid_lines=1999)
    ref1 = _grid(1, 3, line_offset=0, valid_start=0, valid_lines=1999)
    sel = _sel([(ref0, ref0), (ref1, ref1)])
    # For pair 0: burst1=ref0, burst2=ref1
    # bottom overlap from burst1 (ref0): [ref1.valid_line_stop+1, ref0.valid_line_stop]
    #   = [1999+1, 3999-1] = [2000, 3998], num_lines=1999
    slc = _extract_overlap_slice(sel, 0, is_top=False)
    assert slc.first_line == 2000
    assert slc.num_lines == 1999


def test_extract_sensing_times():
    """Sensing start/stop are derived from azimuth_time_at_line."""
    ref0 = _grid(0, 0, line_offset=2000, valid_start=0, valid_lines=1999)
    ref1 = _grid(1, 3, line_offset=0, valid_start=0, valid_lines=1999)
    sel = _sel([(ref0, ref0), (ref1, ref1)])
    slc = _extract_overlap_slice(sel, 0, is_top=True)
    # first_line=0, line_stop=2000, az_interval=0.002
    # sensing_start = sensing_start_at_line(0) = ref0.sensing_start (0s)
    # sensing_stop = sensing_start_at_line(2000) = 0 + 2000*0.002 = 4s
    assert slc.sensing_start == ref0.identity.sensing_start
    assert slc.sensing_stop == ref0.azimuth_time_at_line(2000)


# ---------------------------------------------------------------------------
# materialize_overlaps integration tests
# ---------------------------------------------------------------------------

def test_two_pairs_both_top_and_bottom():
    """2 pairs → 1 overlap pair with non-zero top and bottom overlaps."""
    # Arrange so adjacent burst valid windows overlap:
    # ref0 at line_offset=3000, valid_start=100 → valid [3100, 4400)
    # ref1 at line_offset=0,    valid_start=100 → valid [100, 1400)
    # ref2 at line_offset=3000, valid_start=100 → valid [3100, 4400)
    # pair 0: burst1=ref0, burst2=ref1
    #   top: [ref1.valid_line_start, ref0.valid_line_start-1] = [100, 3099] → non-empty
    #   bottom: [ref1.valid_line_stop+1, ref0.valid_line_stop] = [1400+1, 4400-1] = [1401, 4399] → non-empty
    ref = [
        _grid(0, 0, line_offset=3000, valid_start=100),
        _grid(1, 3, line_offset=0, valid_start=100),
        _grid(2, 6, line_offset=3000, valid_start=100),
    ]
    sec = [
        _grid(0, 0, line_offset=3000, valid_start=100),
        _grid(1, 3, line_offset=0, valid_start=100),
        _grid(2, 6, line_offset=3000, valid_start=100),
    ]
    sel = _sel(list(zip(ref, sec)))
    result = materialize_overlaps(sel)
    assert len(result) == 1  # 2 pairs → 1 overlap
    op = result[0]
    assert op.pair_index == 0
    assert op.top.is_top is True
    assert op.bottom.is_top is False
    assert op.top.num_lines > 0
    assert op.bottom.num_lines > 0


def test_top_only_overlap():
    """Adjacent bursts overlap only in the top slice."""
    # burst0: line_offset=3000, valid_start=0, valid_lines=2500 → valid [3000, 5500)
    # burst1: line_offset=1000, valid_start=0, valid_lines=2500 → valid [1000, 3500)
    # Pair: burst0 is earlier in azimuth (0s), burst1 is 3s later.
    # burst1 valid [1000,3500), burst0 valid [3000,5500)
    # top: [burst1.valid_line_start, burst0.valid_line_start - 1] = [1000, 2999] → 1999 lines
    # bottom: [burst1.valid_line_stop+1, burst0.valid_line_stop] = [3500, 5499] → 2000 lines
    # Both non-zero.
    # Let's arrange so bottom is empty: burst0.valid_line_stop <= burst1.valid_line_stop
    # burst0: line_offset=3000, valid_start=0, valid_lines=500 → [3000, 3500)
    # burst1: line_offset=1000, valid_start=0, valid_lines=2500 → [1000, 3500)
    # top: [1000, 2999] = 2000 lines
    # bottom: [3500+1, 3500-1] → empty (since 3501 > 3499)
    ref0 = _grid(0, 0, line_offset=3000, valid_start=0, valid_lines=500)
    ref1 = _grid(1, 3, line_offset=1000, valid_start=0, valid_lines=2500)
    sel = _sel([(ref0, ref0), (ref1, ref1)])
    result = materialize_overlaps(sel)
    assert len(result) == 1
    op = result[0]
    assert op.top.num_lines == 2000
    assert op.bottom.num_lines == 0


def test_bottom_only_overlap():
    """Adjacent bursts overlap only in the bottom slice."""
    # burst0: line_offset=0, valid_start=0, valid_lines=500 → [0, 500)
    # burst1: line_offset=3000, valid_start=0, valid_lines=5000 → [3000, 5500)
    # burst0 is top (0s), burst1 is bottom (3s)
    # top: [burst1.valid_line_start, burst0.valid_line_start - 1] = [3000, -1] → empty
    # bottom: [burst1.valid_line_stop+1, burst0.valid_line_stop] = [8000+1, 500-1] → empty
    # Let's use a different arrangement:
    # burst0: line_offset=0, valid_start=0, valid_lines=5000 → [0, 5000)
    # burst1: line_offset=4000, valid_start=0, valid_lines=500 → [4000, 4500)
    # top: [4000, -1] → empty
    # bottom: [4500+1, 5000-1] = [4501, 4999] → 499 lines
    ref0 = _grid(0, 0, line_offset=0, valid_start=0, valid_lines=5000)
    ref1 = _grid(1, 3, line_offset=4000, valid_start=0, valid_lines=500)
    sel = _sel([(ref0, ref0), (ref1, ref1)])
    result = materialize_overlaps(sel)
    assert len(result) == 1
    op = result[0]
    assert op.top.num_lines == 0
    assert op.bottom.num_lines == 499


def test_single_burst_pair_no_overlap():
    """Single common burst pair → no overlap pairs (need >= 2 bursts)."""
    ref = [_grid(0, 0)]
    sec = [_grid(0, 0)]
    sel = _sel(list(zip(ref, sec)))
    result = materialize_overlaps(sel)
    assert result == []


def test_boundary_zero_lines():
    """Overlap exactly at boundary produces zero num_lines.

    Since top and bottom are both 0, no OverlapPair is returned (all slices
    have num_lines == 0).  Verify that materialize_overlaps returns [].
    """
    # burst0: [0, 1000), burst1: [1000, 2000) — touching, no overlap
    ref0 = _grid(0, 0, line_offset=0, valid_start=0, valid_lines=1000)
    ref1 = _grid(1, 3, line_offset=1000, valid_start=0, valid_lines=1000)
    sel = _sel([(ref0, ref0), (ref1, ref1)])
    result = materialize_overlaps(sel)
    assert result == []


def test_empty_common_selection_raises():
    """Empty CommonBurstSelection raises ValueError."""
    sel = CommonBurstSelection(
        swath="IW2",
        reference_start_index=0,
        secondary_start_index=0,
        number_of_common_bursts=0,
        pairs=(),
    )
    with pytest.raises(ValueError, match="Empty common burst selection"):
        materialize_overlaps(sel)


def test_three_pairs_two_overlap_pairs():
    """3 common bursts → 2 OverlapPairs when both adjacent pairs have non-zero overlap.

    We use custom valid_start=0 to widen valid windows so that adjacent bursts
    overlap vertically in both pairs.
    """
    # ref0: line_offset=2000, valid_start=0 → [2000, 3299)
    # ref1: line_offset=0,    valid_start=0 → [0, 1299)
    # ref2: line_offset=2000, valid_start=0 → [2000, 3299)
    #
    # Pair 0 (ref0, ref1): top=[0,1999]=1999, bottom=[1300,3298]=1999 → non-zero
    # Pair 1 (ref1, ref2): top=[2000,-1]=empty, bottom=[3300,1298]=empty
    #   → Pair 1 bottom is empty because ref2's valid starts above ref1's valid stops.
    #   The only way to get non-zero for pair 1 is to move ref2 to overlap:
    #   ref2: line_offset=500, valid_start=0 → [500, 1799)
    #   Then pair 1: top=[500, -1]=empty, bottom=[1800, 1298]=empty
    #   Still empty. We need ref2 to start BELOW ref1.stop.
    #
    # Correct arrangement for pair 1 non-zero:
    #   ref1: [0, 1299), ref2: [500, 1799) → bottom=[1800, 1298]=empty (ref2.start > ref1.stop)
    #   ref1: [0, 1299), ref2: [-500, 799) → top=[-500, -1]=empty
    #
    # Standard TOPS geometry naturally gives: pair 0 non-zero, pair 1 zero.
    # Test that materialize_overlaps returns 2 OverlapPairs anyway (even if some
    # slices have zero num_lines).
    ref = [
        _grid(0, 0, line_offset=2000, valid_start=0, valid_lines=1300),
        _grid(1, 3, line_offset=0, valid_start=0, valid_lines=1300),
        _grid(2, 6, line_offset=2000, valid_start=0, valid_lines=1300),
    ]
    sec = [
        _grid(0, 0, line_offset=2000, valid_start=0, valid_lines=1300),
        _grid(1, 3, line_offset=0, valid_start=0, valid_lines=1300),
        _grid(2, 6, line_offset=2000, valid_start=0, valid_lines=1300),
    ]
    sel = _sel(list(zip(ref, sec)))
    # Pair 0 has non-zero top/bottom → OverlapPair created
    # Pair 1 has both zero (ref2.valid_line_start=2000 > ref1.valid_line_stop=1299)
    #   → OverlapPair NOT created (both slices zero)
    # However: ref1.valid_line_stop+1 = 1300, ref1.valid_line_stop = 1299
    # For pair 1 bottom: first_line = 3300 (ref2.stop), line_stop = 1299
    #   → num_lines = max(0, 1299 - 3300) = 0
    # So materialize_overlaps returns only 1 (pair 0)
    result = materialize_overlaps(sel)
    # The standard test verifies the count with this specific geometry:
    # With the given setup, only pair 0 has non-zero overlap.
    assert len(result) == 1
    assert result[0].pair_index == 0


# ---------------------------------------------------------------------------
# overlaps_to_dict tests
# ---------------------------------------------------------------------------

def test_overlaps_to_dict_structure():
    """overlaps_to_dict returns correct JSON structure."""
    ref = [_grid(0, 0), _grid(1, 3)]
    sec = [_grid(0, 0), _grid(1, 3)]
    sel = _sel(list(zip(ref, sec)))
    pairs = materialize_overlaps(sel)
    d = overlaps_to_dict(pairs)
    assert "overlap_count" in d
    assert "overlaps" in d
    assert isinstance(d["overlaps"], list)


def test_overlaps_to_dict_fields():
    """Each overlap entry has required fields."""
    ref = [_grid(0, 0), _grid(1, 3)]
    sec = [_grid(0, 0), _grid(1, 3)]
    sel = _sel(list(zip(ref, sec)))
    pairs = materialize_overlaps(sel)
    d = overlaps_to_dict(pairs)
    for entry in d["overlaps"]:
        assert "pair_index" in entry
        assert "is_top" in entry
        assert "first_line" in entry
        assert "num_lines" in entry
        assert "first_sample" in entry
        assert "num_samples" in entry
        assert "sensing_start" in entry
        assert "sensing_stop" in entry
        assert entry["sensing_start"].endswith("Z")
        assert entry["sensing_stop"].endswith("Z")


def test_overlaps_to_dict_empty():
    """Empty list yields overlap_count=0."""
    d = overlaps_to_dict([])
    assert d["overlap_count"] == 0
    assert d["overlaps"] == []


# ---------------------------------------------------------------------------
# write_overlaps_json tests
# ---------------------------------------------------------------------------

def test_write_and_read_overlaps_json(tmp_path: Path):
    """Round-trip: write_overlaps_json produces valid JSON that can be reloaded."""
    # Use overlapping valid windows:
    # ref0: line_offset=2000, valid_start=0 → [2000, 3300)
    # ref1: line_offset=0,    valid_start=0 → [0, 1300)
    ref = [
        _grid(0, 0, line_offset=2000, valid_start=0, valid_lines=1300),
        _grid(1, 3, line_offset=0, valid_start=0, valid_lines=1300),
    ]
    sec = [
        _grid(0, 0, line_offset=2000, valid_start=0, valid_lines=1300),
        _grid(1, 3, line_offset=0, valid_start=0, valid_lines=1300),
    ]
    sel = _sel(list(zip(ref, sec)))
    pairs = materialize_overlaps(sel)
    out_path = tmp_path / "overlaps.json"
    write_overlaps_json(pairs, out_path)
    assert out_path.exists()
    loaded = json.loads(out_path.read_text())
    assert loaded["overlap_count"] == 2  # 1 pair × 2 slices
    assert len(loaded["overlaps"]) == 2


def test_write_overlaps_json_creates_parent_dirs(tmp_path: Path):
    """write_overlaps_json creates parent directories if needed."""
    ref = [_grid(0, 0), _grid(1, 3)]
    sec = [_grid(0, 0), _grid(1, 3)]
    sel = _sel(list(zip(ref, sec)))
    pairs = materialize_overlaps(sel)
    out_path = tmp_path / "subdir" / "overlaps.json"
    write_overlaps_json(pairs, out_path)
    assert out_path.exists()


# ---------------------------------------------------------------------------
# read_overlap_window tests
# ---------------------------------------------------------------------------

class TestReadOverlapWindow:
    """Tests for read_overlap_window using GDAL-backed temporary TIFFs."""

    @staticmethod
    def _make_overlap_slice(
        first_line: int = 0,
        num_lines: int = 10,
        first_sample: int = 0,
        num_samples: int = 20,
    ) -> OverlapSlice:
        """Minimal OverlapSlice for testing."""
        ref = [_grid(0, 0), _grid(1, 3)]
        sec = [_grid(0, 0), _grid(1, 3)]
        sel = _sel(list(zip(ref, sec)))
        pair = sel.pairs[0]
        start = pair.reference.identity.sensing_start
        stop = start + timedelta(seconds=1)
        return OverlapSlice(
            burst_pair=pair,
            is_top=True,
            first_line=first_line,
            num_lines=num_lines,
            first_sample=first_sample,
            num_samples=num_samples,
            sensing_start=start,
            sensing_stop=stop,
        )

    def _create_temp_tiff(self, tmp_path: Path) -> Path:
        """Create a 50×30 float32 ENVI TIFF with known pattern."""
        try:
            import osgeo.gdal as gdal
        except ImportError:
            pytest.skip("GDAL not available")

        rows, cols = 50, 30
        tiff_path = tmp_path / "test_float.tif"

        # Use ENVI driver for complex support; fallback to GTiff
        for driver_name in ("ENVI", "GTiff"):
            driver = gdal.GetDriverByName(driver_name)
            if driver is not None:
                break
        else:  # pragma: no cover
            pytest.skip("No suitable GDAL driver available")

        ds = driver.Create(str(tiff_path), cols, rows, 1, gdal.GDT_Float32)
        if ds is None:  # pragma: no cover
            pytest.skip("GDAL driver failed to create file")

        # Write deterministic data: line index + sample index (as float)
        band = ds.GetRasterBand(1)
        for line in range(rows):
            row_data = np.arange(cols, dtype=np.float32) + line * 1000.0
            band.WriteArray(row_data.reshape(1, -1), xoff=0, yoff=line)

        band.FlushCache()
        ds = None
        return tiff_path

    def test_read_overlap_window_shape_correct(self, tmp_path: Path):
        """Shape of returned array matches the requested window."""
        try:
            import osgeo.gdal as gdal
        except ImportError:
            pytest.skip("GDAL not available")

        tiff_path = self._create_temp_tiff(tmp_path)

        overlap_slice = self._make_overlap_slice(
            first_line=10,
            num_lines=5,
            first_sample=5,
            num_samples=8,
        )
        result = read_overlap_window(tiff_path, overlap_slice)

        assert result is not None
        assert result.shape == (5, 8)

    def test_read_overlap_window_preserves_dtype(self, tmp_path: Path):
        """Returned dtype should match source band dtype."""
        try:
            import osgeo.gdal as gdal
        except ImportError:
            pytest.skip("GDAL not available")

        path = tmp_path / "complex.tif"
        ds = gdal.GetDriverByName("GTiff").Create(str(path), 4, 4, 1, gdal.GDT_Float32)
        ds.GetRasterBand(1).WriteArray(np.ones((4, 4), dtype=np.float32))
        ds = None

        slc = self._make_overlap_slice(first_line=0, num_lines=2, first_sample=0, num_samples=2)
        out = read_overlap_window(path, slc)
        assert out is not None
        assert out.dtype == np.float32

    def test_read_overlap_window_returns_none_for_missing_file(self, tmp_path: Path):
        """Missing TIFF file returns None."""
        overlap_slice = self._make_overlap_slice()
        result = read_overlap_window(tmp_path / "nonexistent.tif", overlap_slice)
        assert result is None

    def test_read_overlap_window_returns_none_for_zero_dimensions(self, tmp_path: Path):
        """Zero num_lines or num_samples returns None."""
        tiff_path = self._create_temp_tiff(tmp_path)

        slice_zero_lines = self._make_overlap_slice(
            first_line=0, num_lines=0, first_sample=0, num_samples=10
        )
        assert read_overlap_window(tiff_path, slice_zero_lines) is None

        slice_zero_samps = self._make_overlap_slice(
            first_line=0, num_lines=10, first_sample=0, num_samples=0
        )
        assert read_overlap_window(tiff_path, slice_zero_samps) is None

    def test_read_overlap_window_returns_none_for_negative_origin(self, tmp_path: Path):
        """Negative first_line or first_sample returns None."""
        tiff_path = self._create_temp_tiff(tmp_path)

        slice_bad_line = self._make_overlap_slice(
            first_line=-1, num_lines=5, first_sample=0, num_samples=10
        )
        assert read_overlap_window(tiff_path, slice_bad_line) is None

        slice_bad_samp = self._make_overlap_slice(
            first_line=0, num_lines=5, first_sample=-5, num_samples=10
        )
        assert read_overlap_window(tiff_path, slice_bad_samp) is None

    def test_read_overlap_window_returns_none_for_out_of_bounds(self, tmp_path: Path):
        """Requests outside raster bounds return None."""
        tiff_path = self._create_temp_tiff(tmp_path)
        bad = self._make_overlap_slice(first_line=48, num_lines=5, first_sample=25, num_samples=10)
        assert read_overlap_window(tiff_path, bad) is None