"""Tests for tops_merge — valid-window-aware burst mosaic."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import numpy as np
import pytest

from scripts.tops_model import (
    BurstIdentity,
    BurstRadarGrid,
    BurstWindow,
    MergeResult,
)
from scripts.tops_merge import merge_bursts, plan_merge_segments


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _identity(idx: int) -> BurstIdentity:
    return BurstIdentity(
        swath="IW1",
        burst_index=idx,
        sensing_start=datetime(2024, 1, 1, 0, 0, idx * 3, tzinfo=timezone.utc),
        sensing_stop=datetime(2024, 1, 1, 0, 0, idx * 3 + 2, tzinfo=timezone.utc),
        polarization="VV",
        orbit_direction="ascending",
        azimuth_steering_rate=0.0,
    )


def _grid(idx: int, line_offset: int = 0) -> BurstRadarGrid:
    return BurstRadarGrid(
        identity=_identity(idx),
        image_window=BurstWindow(
            first_line=line_offset, num_lines=1500,
            first_sample=0, num_samples=25000,
        ),
        valid_window=BurstWindow(
            first_line=100, num_lines=1300,
            first_sample=500, num_samples=24000,
        ),
        line_offset=line_offset,
        azimuth_time_interval=0.002,
        range_pixel_spacing=2.3,
        starting_range=800000.0,
        radar_wavelength=0.05546576,
        doppler_coefficients=(0.0,),
        azimuth_fm_rate_coefficients=(0.0,),
    )


def _make_ifg(h: int, w: int, phase: float = 0.0) -> np.ndarray:
    """Return a constant-phase interferogram (all pixels identical phase)."""
    return np.full((h, w), np.exp(1j * phase), dtype=np.complex64)


def _make_coh(h: int, w: int, value: float = 0.9) -> np.ndarray:
    return np.full((h, w), value, dtype=np.float32)


# ---------------------------------------------------------------------------
# Test: single burst pass-through
# ---------------------------------------------------------------------------

def test_single_burst_passthrough():
    """A single burst fills the output exactly."""
    h, w = 1300, 24000
    # Use image_window that starts at 0 so valid data lands at line 0 in output
    burst = _grid(0, line_offset=0)

    # Provide valid_window matching the IFG shape
    vw = BurstWindow(first_line=0, num_lines=h, first_sample=0, num_samples=w)
    ifgs = [_make_ifg(h, w, phase=0.5)]
    coherences = [_make_coh(h, w, 0.95)]
    valid_windows = [vw]
    bursts = [burst]

    out_ifg = np.zeros((h, w), dtype=np.complex64)
    out_coh = np.zeros((h, w), dtype=np.float32)

    result = merge_bursts(
        ifgs, coherences, bursts, valid_windows,
        seam_regions=[],
        out_ifg=out_ifg, out_coh=out_coh,
    )

    np.testing.assert_array_almost_equal(np.angle(out_ifg), 0.5, decimal=4)
    np.testing.assert_array_almost_equal(out_coh, 0.95, decimal=4)
    assert result.gap_pixel_count == 0
    assert result.top_contribution_count == h * w
    assert result.bottom_contribution_count == 0
    assert len(result.segments) == 1


# ---------------------------------------------------------------------------
# Test: radar-domain placement from sensing time and slant range
# ---------------------------------------------------------------------------

def test_two_bursts_with_zero_image_offsets_are_placed_by_rd_coordinates():
    """Burst mosaicking uses RD coordinates, not only image_window offsets."""
    h, w = 4, 5

    # Both bursts have image_window.first_line=0.  If merging uses only that
    # field, they collapse onto the same rows.  Their sensing_start values are
    # 3 s apart and azimuth_time_interval is 1 s/line, so burst 1 belongs at
    # output row 3 in the merged RD grid, matching tops_rtc.py/ISCE2 logic.
    burst0 = replace(_grid(0, line_offset=0), azimuth_time_interval=1.0)
    burst1 = replace(_grid(1, line_offset=0), azimuth_time_interval=1.0)
    vw = BurstWindow(first_line=0, num_lines=h, first_sample=0, num_samples=w)

    out_ifg = np.zeros((7, w), dtype=np.complex64)
    out_coh = np.zeros((7, w), dtype=np.float32)

    result = merge_bursts(
        [_make_ifg(h, w, phase=0.1), _make_ifg(h, w, phase=0.2)],
        [_make_coh(h, w, 0.8), _make_coh(h, w, 0.9)],
        [burst0, burst1],
        [vw, vw],
        seam_regions=[],
        out_ifg=out_ifg,
        out_coh=out_coh,
    )

    assert result.segments[0].output_line_start == 0
    assert result.segments[1].output_line_start == 3
    np.testing.assert_array_almost_equal(np.angle(out_ifg[0:3, :]), 0.1, decimal=4)
    assert out_coh[0, 0] == 0.8
    # Row 3 is an overlap: avg method is not configurable yet, so current merge
    # averages both contributions in overlaps.
    assert out_coh[3, 0] == pytest.approx(0.85)
    np.testing.assert_array_almost_equal(np.angle(out_ifg[4:7, :]), 0.2, decimal=4)


# ---------------------------------------------------------------------------
# Test: two bursts stitched correctly
# ---------------------------------------------------------------------------

def test_two_bursts_stitch():
    """Two adjacent bursts are placed at correct offsets and fill the mosaic."""
    h, w = 1300, 24000

    # Burst 0: image_window.first_line=0, valid_window.first_line=0 → lands at line 0
    burst0 = _grid(0, line_offset=0)
    vw0 = BurstWindow(first_line=0, num_lines=h, first_sample=0, num_samples=w)

    # Burst 1: image_window.first_line=1500, valid_window.first_line=0 → lands at line 1500
    burst1 = _grid(1, line_offset=1500)
    vw1 = BurstWindow(first_line=0, num_lines=h, first_sample=0, num_samples=w)

    # Output height must cover both bursts with no clipping
    out_h = burst1.image_window.first_line + h  # = 2800
    out_w = w

    ifgs = [
        _make_ifg(h, w, phase=0.1),
        _make_ifg(h, w, phase=0.2),
    ]
    coherences = [
        _make_coh(h, w, 0.8),
        _make_coh(h, w, 0.85),
    ]
    valid_windows = [vw0, vw1]
    bursts = [burst0, burst1]

    out_ifg = np.zeros((out_h, out_w), dtype=np.complex64)
    out_coh = np.zeros((out_h, out_w), dtype=np.float32)

    result = merge_bursts(
        ifgs, coherences, bursts, valid_windows,
        seam_regions=[],
        out_ifg=out_ifg, out_coh=out_coh,
    )

    # Burst 0 occupies lines 0..1300; Burst 1 occupies lines 1500..2800
    # Gap of 200 lines in between (lines 1300..1500)
    assert result.gap_pixel_count == 200 * out_w
    assert result.top_contribution_count == h * w
    assert result.bottom_contribution_count == h * w
    assert len(result.segments) == 2

    # Check phases are placed correctly
    phase0 = np.angle(out_ifg[0:h, 0:w])
    np.testing.assert_array_almost_equal(phase0, 0.1, decimal=4)
    phase1 = np.angle(out_ifg[1500:2800, 0:w])
    np.testing.assert_array_almost_equal(phase1, 0.2, decimal=4)


# ---------------------------------------------------------------------------
# Test: seam phase diff and coherence drop
# ---------------------------------------------------------------------------

def test_seam_diagnostics():
    """Seam region statistics are computed and returned."""
    h, w = 1300, 24000

    burst0 = _grid(0, line_offset=0)
    vw0 = BurstWindow(first_line=0, num_lines=h, first_sample=0, num_samples=w)

    burst1 = _grid(1, line_offset=1500)
    vw1 = BurstWindow(first_line=0, num_lines=h, first_sample=0, num_samples=w)

    out_h = 2800
    out_w = w

    ifgs = [
        _make_ifg(h, w, phase=0.3),
        _make_ifg(h, w, phase=0.5),
    ]
    coherences = [
        _make_coh(h, w, 0.9),
        _make_coh(h, w, 0.7),
    ]
    valid_windows = [vw0, vw1]
    bursts = [burst0, burst1]

    # Seam between burst0 and burst1: 5-pixel wide band at lines 1295-1300
    # and 1500-1505 for the overlapping seam computation
    seam_regions = [
        (1295, 0, 5, w),    # bottom of burst0 seam
        (1500, 0, 5, w),    # top of burst1 seam
    ]

    out_ifg = np.zeros((out_h, out_w), dtype=np.complex64)
    out_coh = np.zeros((out_h, out_w), dtype=np.float32)

    result = merge_bursts(
        ifgs, coherences, bursts, valid_windows,
        seam_regions=seam_regions,
        out_ifg=out_ifg, out_coh=out_coh,
    )

    assert result.seam_phase_diff_median is not None
    assert result.seam_coherence_drop is not None
    # seam_coherence_drop is mean of seam coherences: mean([0.9, 0.7]) = 0.8
    assert abs(result.seam_coherence_drop - 0.8) < 0.05


# ---------------------------------------------------------------------------
# Test: gap detection
# ---------------------------------------------------------------------------

def test_gap_detection():
    """Gap pixels (no burst contribution) are counted correctly."""
    h, w = 1300, 24000

    # Only one burst, output is larger than burst valid window
    burst = _grid(0, line_offset=0)
    vw = BurstWindow(first_line=0, num_lines=h, first_sample=0, num_samples=w)
    out_h = h + 500  # extra 500 lines of zeros
    out_w = w

    ifgs = [_make_ifg(h, w)]
    coherences = [_make_coh(h, w)]
    valid_windows = [vw]
    bursts = [burst]

    out_ifg = np.zeros((out_h, out_w), dtype=np.complex64)
    out_coh = np.zeros((out_h, out_w), dtype=np.float32)

    result = merge_bursts(
        ifgs, coherences, bursts, valid_windows,
        seam_regions=[],
        out_ifg=out_ifg, out_coh=out_coh,
    )

    assert result.gap_pixel_count == 500 * out_w


# ---------------------------------------------------------------------------
# Test: coherence propagation
# ---------------------------------------------------------------------------

def test_coherence_propagation():
    """Coherence values are placed correctly in the output mosaic."""
    h, w = 1300, 24000

    burst0 = _grid(0, line_offset=0)
    vw0 = BurstWindow(first_line=0, num_lines=h, first_sample=0, num_samples=w)

    burst1 = _grid(1, line_offset=1500)
    vw1 = BurstWindow(first_line=0, num_lines=h, first_sample=0, num_samples=w)

    out_h = 2800
    out_w = w

    ifgs = [
        _make_ifg(h, w, phase=0.1),
        _make_ifg(h, w, phase=0.2),
    ]
    coherences = [
        _make_coh(h, w, 0.7),
        _make_coh(h, w, 0.5),
    ]
    valid_windows = [vw0, vw1]
    bursts = [burst0, burst1]

    out_ifg = np.zeros((out_h, out_w), dtype=np.complex64)
    out_coh = np.zeros((out_h, out_w), dtype=np.float32)

    result = merge_bursts(
        ifgs, coherences, bursts, valid_windows,
        seam_regions=[],
        out_ifg=out_ifg, out_coh=out_coh,
    )

    # Non-gap region coherence
    assert out_coh[200, 1000] == 0.7
    assert out_coh[1700, 1000] == 0.5
    # Gap region is zero: lines 1300..1499 (between burst0 at 0..1300 and burst1 at 1500..2800)
    assert out_coh[1350, 1000] == 0.0


# ---------------------------------------------------------------------------
# Test: zero-length seam
# ---------------------------------------------------------------------------

def test_zero_length_seam():
    """Empty seam region list is handled without error."""
    h, w = 1300, 24000
    burst = _grid(0, line_offset=0)
    vw = BurstWindow(first_line=0, num_lines=h, first_sample=0, num_samples=w)

    ifgs = [_make_ifg(h, w)]
    coherences = [_make_coh(h, w)]
    valid_windows = [vw]
    bursts = [burst]

    out_ifg = np.zeros((h, w), dtype=np.complex64)
    out_coh = np.zeros((h, w), dtype=np.float32)

    result = merge_bursts(
        ifgs, coherences, bursts, valid_windows,
        seam_regions=[],   # empty
        out_ifg=out_ifg, out_coh=out_coh,
    )

    # No seam statistics computed
    assert result.seam_phase_diff_median == 0.0
    assert result.seam_coherence_drop == 0.0


# ---------------------------------------------------------------------------
# Test: seam out-of-bounds → warning skip
# ---------------------------------------------------------------------------

def test_seam_out_of_bounds_skipped():
    """Out-of-bounds seam regions are skipped with a warning (no crash)."""
    h, w = 1300, 24000
    burst = _grid(0, line_offset=0)
    vw = BurstWindow(first_line=0, num_lines=h, first_sample=0, num_samples=w)

    ifgs = [_make_ifg(h, w)]
    coherences = [_make_coh(h, w)]
    valid_windows = [vw]
    bursts = [burst]

    out_ifg = np.zeros((h, w), dtype=np.complex64)
    out_coh = np.zeros((h, w), dtype=np.float32)

    # Seam outside output bounds
    seam_regions = [
        (999999, 0, 10, 10),    # row out-of-bounds
        (0, 999999, 10, 10),    # col out-of-bounds
        (-5, 0, 5, 5),          # negative start
    ]

    result = merge_bursts(
        ifgs, coherences, bursts, valid_windows,
        seam_regions=seam_regions,
        out_ifg=out_ifg, out_coh=out_coh,
    )

    # Should complete without error; seam stats are zero
    assert result.seam_phase_diff_median == 0.0
    assert result.gap_pixel_count == 0


# ---------------------------------------------------------------------------
# Test: shape mismatch raises ValueError
# ---------------------------------------------------------------------------

def test_shape_mismatch_raises():
    """Input IFG shape mismatch with valid_window raises ValueError."""
    vw = _grid(0).valid_window
    burst = _grid(0, line_offset=0)
    h, w = 1300, 24000

    # Bad valid_window claims different num_lines than the IFG being passed
    bad_vw = BurstWindow(first_line=0, num_lines=9999, first_sample=0, num_samples=w)

    out_ifg = np.zeros((h, w), dtype=np.complex64)
    out_coh = np.zeros((h, w), dtype=np.float32)

    with pytest.raises(ValueError, match="ifg shape .* != expected"):
        merge_bursts(
            [_make_ifg(h, w)],
            [_make_coh(h, w)],
            [burst],
            [bad_vw],
            seam_regions=[],
            out_ifg=out_ifg,
            out_coh=out_coh,
        )


def test_mismatched_lengths_raises():
    """Mismatched list lengths raise ValueError."""
    burst = _grid(0, line_offset=0)
    h, w = 1300, 24000
    vw = BurstWindow(first_line=0, num_lines=h, first_sample=0, num_samples=w)

    ifgs = [_make_ifg(h, w)]
    coherences: list[np.ndarray] = []
    valid_windows = [vw]
    bursts = [burst]

    out_ifg = np.zeros((h, w), dtype=np.complex64)
    out_coh = np.zeros((h, w), dtype=np.float32)

    with pytest.raises(ValueError, match="ifgs and coherences must have the same length"):
        merge_bursts(
            ifgs, coherences, bursts, valid_windows,
            seam_regions=[],
            out_ifg=out_ifg, out_coh=out_coh,
        )


# ---------------------------------------------------------------------------
# Test: plan_merge_segments
# ---------------------------------------------------------------------------

def test_plan_merge_segments():
    """plan_merge_segments returns sorted MergeSegment tuple."""
    burst0 = _grid(0, line_offset=0)
    burst1 = _grid(1, line_offset=1500)
    burst2 = _grid(2, line_offset=3000)
    bursts = [burst1, burst0, burst2]   # deliberately unsorted
    h, w = 1300, 24000
    vw = BurstWindow(first_line=0, num_lines=h, first_sample=0, num_samples=w)
    valid_windows = [vw, vw, vw]
    out_h = 4500
    out_w = w

    segs = plan_merge_segments(bursts, valid_windows, out_h, out_w)

    assert len(segs) == 3
    # Must be sorted by output_line_start
    assert segs[0].output_line_start < segs[1].output_line_start < segs[2].output_line_start
    # Burst indices preserved
    assert segs[0].burst_index == 1
    assert segs[1].burst_index == 0
    assert segs[2].burst_index == 2


def test_plan_merge_segments_bounds_clamp():
    """Segments are clamped to output bounds."""
    # Burst with image_window.first_line=100, valid_window.first_line=0
    # → lands at output line 100, extends to line 100+1300=1400
    # Set output height to 1100 so burst would go to 1400, clamped to 1100
    burst = _grid(0, line_offset=100)
    vw = BurstWindow(first_line=0, num_lines=1300, first_sample=0, num_samples=24000)
    out_h = 1100   # fits 100-1100 of the burst, clips last 300 lines
    out_w = 25000

    segs = plan_merge_segments([burst], [vw], out_h, out_w)

    # output_line_start=100, output should be clamped to h=1000 (out_h-out_line_start)
    assert segs[0].output_num_lines == out_h - segs[0].output_line_start
    # samples fit within output (valid_window has 24000, fits in 25000)
    assert segs[0].output_num_samples == 24000


# ---------------------------------------------------------------------------
# Test: MergeResult is returned with correct fields
# ---------------------------------------------------------------------------

def test_merge_result_fields():
    """MergeResult contains all expected fields."""
    h, w = 1300, 24000
    burst = _grid(0, line_offset=0)
    vw = BurstWindow(first_line=0, num_lines=h, first_sample=0, num_samples=w)

    ifgs = [_make_ifg(h, w)]
    coherences = [_make_coh(h, w)]
    valid_windows = [vw]
    bursts = [burst]

    out_ifg = np.zeros((h, w), dtype=np.complex64)
    out_coh = np.zeros((h, w), dtype=np.float32)

    result = merge_bursts(
        ifgs, coherences, bursts, valid_windows,
        seam_regions=[],
        out_ifg=out_ifg, out_coh=out_coh,
    )

    assert isinstance(result, MergeResult)
    assert "seam_phase_diff_median" in result.__dict__
    assert "gap_pixel_count" in result.__dict__
    assert "top_contribution_count" in result.__dict__
    assert "bottom_contribution_count" in result.__dict__
    assert "segments" in result.__dict__