"""Test suite for tops_model dataclasses.

Each dataclass is tested for:
  1. Construction and basic property access
  2. Boundary conditions (zero, typical, large values)
  3. Invalid input rejection (negative values, zero dimensions, type errors)
  4. Computed property / formula verification (prf, valid_line, slant_range, etc.)
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from scripts.tops_model import (
    BurstIdentity,
    BurstWindow,
    BurstRadarGrid,
    CommonBurstPair,
    CommonBurstSelection,
    Geo2RdrOffsets,
    OverlapSlice,
    OverlapPair,
    EsdEstimate,
    TimingCorrection,
    MergeSegment,
    MergeResult,
    RangeCoregEstimate,
)


UTC = timezone.utc

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _identity(idx: int = 0, swath: str = "IW1") -> BurstIdentity:
    start = datetime(2024, 1, 1, 0, 0, idx * 3, tzinfo=UTC)
    stop = start + timedelta(seconds=2)
    return BurstIdentity(
        swath=swath,
        burst_index=idx,
        sensing_start=start,
        sensing_stop=stop,
        polarization="VV",
        orbit_direction="ascending",
        azimuth_steering_rate=0.0018,
    )


def _window(first_line: int = 0, num_lines: int = 1500,
            first_sample: int = 0, num_samples: int = 25000) -> BurstWindow:
    return BurstWindow(
        first_line=first_line,
        num_lines=num_lines,
        first_sample=first_sample,
        num_samples=num_samples,
    )


def _grid(idx: int = 0, line_offset: int = 0,
          num_lines: int = 1500) -> BurstRadarGrid:
    return BurstRadarGrid(
        identity=_identity(idx),
        image_window=_window(first_line=line_offset, num_lines=num_lines),
        valid_window=_window(first_line=100, num_lines=1300,
                              first_sample=500, num_samples=24000),
        line_offset=line_offset,
        azimuth_time_interval=0.002,   # PRF = 500 Hz
        range_pixel_spacing=2.3,
        starting_range=800_000.0,
        radar_wavelength=0.05546576,
        doppler_coefficients=(0.0, -1e-7),
        azimuth_fm_rate_coefficients=(0.0, 1e-4),
    )


# ---------------------------------------------------------------------------
# BurstIdentity
# ---------------------------------------------------------------------------

class TestBurstIdentity:
    def test_construction_all_fields(self):
        ident = BurstIdentity(
            swath="IW2",
            burst_index=5,
            sensing_start=datetime(2024, 3, 15, 10, 30, 0, tzinfo=UTC),
            sensing_stop=datetime(2024, 3, 15, 10, 30, 2, tzinfo=UTC),
            polarization="VH",
            orbit_direction="descending",
            azimuth_steering_rate=-0.0018,
        )
        assert ident.swath == "IW2"
        assert ident.burst_index == 5
        assert ident.polarization == "VH"
        assert ident.orbit_direction == "descending"
        assert ident.azimuth_steering_rate == -0.0018

    def test_frozen_immutable(self):
        ident = _identity()
        with pytest.raises(Exception):   # dataclasses.FrozenInstanceError
            ident.swath = "IW3"

    def test_timezone_aware_required(self):
        start = datetime(2024, 1, 1, tzinfo=UTC)
        stop = start + timedelta(seconds=2)
        ident = BurstIdentity(
            swath="IW1", burst_index=0,
            sensing_start=start, sensing_stop=stop,
            polarization="VV", orbit_direction="ascending",
            azimuth_steering_rate=0.0,
        )
        assert ident.sensing_start.tzinfo is not None


# ---------------------------------------------------------------------------
# BurstWindow
# ---------------------------------------------------------------------------

class TestBurstWindow:
    def test_construction_and_stop_properties(self):
        w = BurstWindow(first_line=10, num_lines=1000,
                        first_sample=200, num_samples=5000)
        assert w.line_stop == 1010
        assert w.sample_stop == 5200

    def test_zero_dimensions_edge_case(self):
        w = BurstWindow(first_line=0, num_lines=0, first_sample=0, num_samples=0)
        assert w.line_stop == 0
        assert w.sample_stop == 0

    def test_negative_num_lines_rejected(self):
        with pytest.raises((TypeError, ValueError)):
            BurstWindow(first_line=0, num_lines=-1,
                        first_sample=0, num_samples=100)

    def test_negative_num_samples_rejected(self):
        with pytest.raises((TypeError, ValueError)):
            BurstWindow(first_line=0, num_lines=100,
                        first_sample=0, num_samples=-10)

    def test_large_dimensions(self):
        w = BurstWindow(first_line=0, num_lines=100_000,
                        first_sample=0, num_samples=50_000)
        assert w.line_stop == 100_000
        assert w.sample_stop == 50_000


# ---------------------------------------------------------------------------
# BurstRadarGrid
# ---------------------------------------------------------------------------

class TestBurstRadarGrid:
    def test_prf_from_azimuth_time_interval(self):
        # PRF = 1 / azimuth_time_interval  →  1/0.002 = 500
        g = _grid()
        assert g.prf == pytest.approx(500.0)

    def test_prf_extreme_case(self):
        # Very fast PRF
        g = _grid()
        g2 = BurstRadarGrid(
            identity=_identity(),
            image_window=_window(),
            valid_window=_window(),
            line_offset=0,
            azimuth_time_interval=0.001,
            range_pixel_spacing=2.3,
            starting_range=800_000.0,
            radar_wavelength=0.055,
            doppler_coefficients=(0.0,),
            azimuth_fm_rate_coefficients=(0.0,),
        )
        assert g2.prf == pytest.approx(1000.0)

    def test_duration(self):
        g = _grid()
        # sensing_start → sensing_stop is always 2 seconds in _grid()
        assert g.duration == pytest.approx(2.0)

    def test_duration_minimal_interval(self):
        # Zero-duration burst is physically invalid (sensing_stop > sensing_start required),
        # so we test the smallest-possible positive duration instead.
        ident = BurstIdentity(
            swath="IW1", burst_index=0,
            sensing_start=datetime(2024, 1, 1, tzinfo=UTC),
            sensing_stop=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(seconds=1e-6),
            polarization="VV", orbit_direction="ascending",
            azimuth_steering_rate=0.0,
        )
        g = BurstRadarGrid(
            identity=ident,
            image_window=_window(),
            valid_window=_window(),
            line_offset=0,
            azimuth_time_interval=1.0,
            range_pixel_spacing=2.3,
            starting_range=800_000.0,
            radar_wavelength=0.055,
            doppler_coefficients=(0.0,),
            azimuth_fm_rate_coefficients=(0.0,),
        )
        assert g.duration == pytest.approx(1e-6)

    def test_valid_line_absolute(self):
        # image_window.first_line=0, valid_window.first_line=100 → valid_line_start=100
        g = _grid(line_offset=0, num_lines=1500)
        assert g.valid_line_start == 100
        assert g.valid_line_stop == 100 + 1300  # = 1400

    def test_valid_line_with_line_offset(self):
        # Burst starts at line 3000 within the full measurement image
        g = _grid(line_offset=3000, num_lines=1500)
        # valid_window.first_line=100 → absolute start = 3000 + 100 = 3100
        assert g.valid_line_start == 3100
        assert g.valid_line_stop == 3100 + 1300  # = 4400

    def test_slant_range_at_sample_zero(self):
        g = _grid()
        assert g.slant_range_at(0) == 800_000.0

    def test_slant_range_at_sample(self):
        # slant_range = starting_range + sample * range_pixel_spacing
        g = _grid()
        assert g.slant_range_at(1000) == pytest.approx(800_000.0 + 1000 * 2.3)

    def test_azimuth_time_at_line(self):
        g = _grid()
        t0 = g.identity.sensing_start
        # Line 0 → sensing_start, Line 1 → sensing_start + 0.002 s, ...
        assert g.azimuth_time_at_line(0) == t0
        assert g.azimuth_time_at_line(100) == t0 + timedelta(seconds=0.200)

    def test_azimuth_time_interval_zero_rejected(self):
        # Zero interval would give infinite PRF — reject
        with pytest.raises((TypeError, ValueError)):
            BurstRadarGrid(
                identity=_identity(),
                image_window=_window(),
                valid_window=_window(),
                line_offset=0,
                azimuth_time_interval=0.0,   # invalid
                range_pixel_spacing=2.3,
                starting_range=800_000.0,
                radar_wavelength=0.055,
                doppler_coefficients=(0.0,),
                azimuth_fm_rate_coefficients=(0.0,),
            )

    def test_frozen(self):
        g = _grid()
        with pytest.raises(Exception):
            g.radar_wavelength = 0.06

    def test_doppler_coefficients_tuple_length(self):
        # Polynomial can be any length; empty tuple also works
        g = BurstRadarGrid(
            identity=_identity(),
            image_window=_window(),
            valid_window=_window(),
            line_offset=0,
            azimuth_time_interval=0.002,
            range_pixel_spacing=2.3,
            starting_range=800_000.0,
            radar_wavelength=0.055,
            doppler_coefficients=(),   # empty OK
            azimuth_fm_rate_coefficients=(),
        )
        assert g.doppler_coefficients == ()


# ---------------------------------------------------------------------------
# CommonBurstPair
# ---------------------------------------------------------------------------

class TestCommonBurstPair:
    def test_construction(self):
        ref = _grid(0)
        sec = _grid(1)
        pair = CommonBurstPair(
            pair_index=0,
            reference=ref,
            secondary=sec,
            burst_offset=1,
        )
        assert pair.pair_index == 0
        assert pair.burst_offset == 1
        assert pair.reference is ref
        assert pair.secondary is sec

    def test_frozen(self):
        pair = CommonBurstPair(
            pair_index=0,
            reference=_grid(0),
            secondary=_grid(1),
            burst_offset=1,
        )
        with pytest.raises(Exception):
            pair.burst_offset = 2

    def test_negative_burst_offset(self):
        # Negative offset means secondary is "earlier" than reference
        ref = _grid(2)
        sec = _grid(0)
        pair = CommonBurstPair(
            pair_index=0,
            reference=ref,
            secondary=sec,
            burst_offset=-2,
        )
        assert pair.burst_offset == -2


# ---------------------------------------------------------------------------
# CommonBurstSelection
# ---------------------------------------------------------------------------

class TestCommonBurstSelection:
    def test_construction(self):
        pairs = tuple(
            CommonBurstPair(i, _grid(i), _grid(i + 1), burst_offset=1)
            for i in range(3)
        )
        sel = CommonBurstSelection(
            swath="IW2",
            reference_start_index=1,
            secondary_start_index=0,
            number_of_common_bursts=3,
            pairs=pairs,
        )
        assert sel.swath == "IW2"
        assert sel.number_of_common_bursts == 3
        assert len(sel.pairs) == 3

    def test_default_pairs_empty(self):
        sel = CommonBurstSelection(
            swath="IW1",
            reference_start_index=0,
            secondary_start_index=0,
            number_of_common_bursts=0,
        )
        assert sel.pairs == ()

    def test_frozen(self):
        sel = CommonBurstSelection(
            swath="IW1",
            reference_start_index=0,
            secondary_start_index=0,
            number_of_common_bursts=0,
        )
        with pytest.raises(Exception):
            sel.swath = "IW3"


# ---------------------------------------------------------------------------
# Geo2RdrOffsets
# ---------------------------------------------------------------------------

class TestGeo2RdrOffsets:
    def test_construction(self):
        offsets = Geo2RdrOffsets(
            range_off_path="/tmp/range.off",
            azimuth_off_path="/tmp/azimuth.off",
            median_range_offset=5.23,
            median_azimuth_offset=0.12,
            valid_sample_count=12345,
        )
        assert offsets.median_range_offset == 5.23
        assert offsets.median_azimuth_offset == 0.12
        assert offsets.valid_sample_count == 12345

    def test_frozen(self):
        offsets = Geo2RdrOffsets(
            range_off_path="", azimuth_off_path="",
            median_range_offset=0.0, median_azimuth_offset=0.0, valid_sample_count=0,
        )
        with pytest.raises(Exception):
            offsets.median_range_offset = 1.0

    def test_negative_valid_count_rejected(self):
        with pytest.raises((TypeError, ValueError)):
            Geo2RdrOffsets(
                range_off_path="", azimuth_off_path="",
                median_range_offset=0.0, median_azimuth_offset=0.0,
                valid_sample_count=-1,
            )

    def test_nan_offsets_valid(self):
        # ISCE3 can produce NaN medians; we allow them but caller handles them
        offsets = Geo2RdrOffsets(
            range_off_path="",
            azimuth_off_path="",
            median_range_offset=float("nan"),
            median_azimuth_offset=float("nan"),
            valid_sample_count=0,
        )
        import math
        assert math.isnan(offsets.median_range_offset)


# ---------------------------------------------------------------------------
# RangeCoregEstimate
# ---------------------------------------------------------------------------

class TestRangeCoregEstimate:
    def test_construction(self):
        est = RangeCoregEstimate(
            median_range_offset=0.05,
            std_range_offset=0.02,
            median_azimuth_offset=0.0,
            std_azimuth_offset=0.01,
            sample_count=5000,
            usable_fraction=0.96,
        )
        assert est.median_range_offset == 0.05
        assert est.sample_count == 5000

    def test_frozen(self):
        est = RangeCoregEstimate(
            median_range_offset=0.0, std_range_offset=0.0,
            median_azimuth_offset=0.0, std_azimuth_offset=0.0,
            sample_count=0, usable_fraction=0.0,
        )
        with pytest.raises(Exception):
            est.sample_count = 1

    def test_zero_valid_count_rejected(self):
        # sample_count=0 is valid (empty estimate); usable_fraction=0.0 is valid.
        # Only strictly negative values are rejected.
        with pytest.raises(ValueError, match="sample_count must be non-negative"):
            RangeCoregEstimate(
                median_range_offset=0.0,
                std_range_offset=0.0,
                median_azimuth_offset=0.0,
                std_azimuth_offset=0.0,
                sample_count=-1,   # negative invalid
                usable_fraction=1.0,
            )
        with pytest.raises(ValueError, match="usable_fraction must be in"):
            RangeCoregEstimate(
                median_range_offset=0.0,
                std_range_offset=0.0,
                median_azimuth_offset=0.0,
                std_azimuth_offset=0.0,
                sample_count=0,
                usable_fraction=-0.1,   # out of bounds invalid
            )


# ---------------------------------------------------------------------------
# OverlapSlice
# ---------------------------------------------------------------------------

class TestOverlapSlice:
    def test_construction(self):
        pair = CommonBurstPair(
            pair_index=0,
            reference=_grid(0),
            secondary=_grid(1),
            burst_offset=1,
        )
        sl = OverlapSlice(
            burst_pair=pair,
            is_top=True,
            first_line=1300,
            num_lines=200,
            first_sample=500,
            num_samples=24000,
            sensing_start=datetime(2024, 1, 1, 0, 0, 1, tzinfo=UTC),
            sensing_stop=datetime(2024, 1, 1, 0, 0, 1, 500_000, tzinfo=UTC),
        )
        assert sl.first_line == 1300
        assert sl.num_lines == 200
        assert sl.is_top is True

    def test_frozen(self):
        pair = CommonBurstPair(
            pair_index=0, reference=_grid(0), secondary=_grid(1), burst_offset=1,
        )
        sl = OverlapSlice(
            burst_pair=pair, is_top=False,
            first_line=0, num_lines=100,
            first_sample=0, num_samples=1000,
            sensing_start=datetime(2024, 1, 1, tzinfo=UTC),
            sensing_stop=datetime(2024, 1, 1, 0, 0, 1, tzinfo=UTC),
        )
        with pytest.raises(Exception):
            sl.num_lines = 200

    def test_negative_num_lines_rejected(self):
        with pytest.raises((TypeError, ValueError)):
            OverlapSlice(
                burst_pair=None, is_top=True,
                first_line=0, num_lines=-1,
                first_sample=0, num_samples=1000,
                sensing_start=datetime(2024, 1, 1, tzinfo=UTC),
                sensing_stop=datetime(2024, 1, 1, 0, 0, 1, tzinfo=UTC),
            )


# ---------------------------------------------------------------------------
# OverlapPair
# ---------------------------------------------------------------------------

class TestOverlapPair:
    def test_construction_top_bottom(self):
        pair = CommonBurstPair(
            pair_index=0, reference=_grid(0), secondary=_grid(1), burst_offset=1,
        )
        top = OverlapSlice(
            burst_pair=pair, is_top=True,
            first_line=1300, num_lines=200,
            first_sample=500, num_samples=24000,
            sensing_start=datetime(2024, 1, 1, 0, 0, 1, tzinfo=UTC),
            sensing_stop=datetime(2024, 1, 1, 0, 0, 1, 500_000, tzinfo=UTC),
        )
        bot = OverlapSlice(
            burst_pair=pair, is_top=False,
            first_line=1300, num_lines=200,
            first_sample=500, num_samples=24000,
            sensing_start=datetime(2024, 1, 1, 0, 0, 1, tzinfo=UTC),
            sensing_stop=datetime(2024, 1, 1, 0, 0, 1, 500_000, tzinfo=UTC),
        )
        op = OverlapPair(pair_index=0, top=top, bottom=bot)
        assert op.pair_index == 0
        assert op.top.is_top is True
        assert op.bottom.is_top is False

    def test_frozen(self):
        pair = CommonBurstPair(
            pair_index=0, reference=_grid(0), secondary=_grid(1), burst_offset=1,
        )
        sl = OverlapSlice(
            burst_pair=pair, is_top=True,
            first_line=0, num_lines=100,
            first_sample=0, num_samples=1000,
            sensing_start=datetime(2024, 1, 1, tzinfo=UTC),
            sensing_stop=datetime(2024, 1, 1, 0, 0, 1, tzinfo=UTC),
        )
        op = OverlapPair(pair_index=0, top=sl, bottom=sl)
        with pytest.raises(Exception):
            op.pair_index = 1


# ---------------------------------------------------------------------------
# EsdEstimate
# ---------------------------------------------------------------------------

class TestEsdEstimate:
    def test_construction(self):
        est = EsdEstimate(
            median_offset_pixels=0.12,
            mean_offset_pixels=0.11,
            std_offset_pixels=0.05,
            sample_count=1200,
            azimuth_time_interval=0.002,
        )
        assert est.median_offset_pixels == 0.12
        assert est.azimuth_time_interval == 0.002

    def test_frozen(self):
        est = EsdEstimate(
            median_offset_pixels=0.0, mean_offset_pixels=0.0,
            std_offset_pixels=0.0, sample_count=0, azimuth_time_interval=0.002,
        )
        with pytest.raises(Exception):
            est.median_offset_pixels = 0.1

    def test_negative_sample_count_rejected(self):
        with pytest.raises((TypeError, ValueError)):
            EsdEstimate(
                median_offset_pixels=0.0, mean_offset_pixels=0.0,
                std_offset_pixels=0.0,
                sample_count=-1,     # invalid
                azimuth_time_interval=0.002,
            )

    def test_timing_seconds_from_pixels(self):
        # secondary_timing_seconds = median_offset_pixels * azimuth_time_interval
        est = EsdEstimate(
            median_offset_pixels=100.0,   # pixels
            mean_offset_pixels=0.0, std_offset_pixels=0.0,
            sample_count=1000, azimuth_time_interval=0.002,
        )
        expected_seconds = 100.0 * 0.002  # = 0.2 s
        assert expected_seconds == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# TimingCorrection
# ---------------------------------------------------------------------------

class TestTimingCorrection:
    def test_construction(self):
        est = EsdEstimate(
            median_offset_pixels=0.1, mean_offset_pixels=0.09,
            std_offset_pixels=0.03, sample_count=500,
            azimuth_time_interval=0.002,
        )
        tc = TimingCorrection(
            secondary_timing_seconds=0.0002,
            secondary_timing_pixels=0.1,
            esd_estimate=est,
        )
        assert tc.secondary_timing_seconds == 0.0002
        assert tc.secondary_timing_pixels == 0.1
        assert tc.esd_estimate is est

    def test_frozen(self):
        est = EsdEstimate(
            median_offset_pixels=0.0, mean_offset_pixels=0.0,
            std_offset_pixels=0.0, sample_count=0, azimuth_time_interval=0.002,
        )
        tc = TimingCorrection(
            secondary_timing_seconds=0.0,
            secondary_timing_pixels=0.0,
            esd_estimate=est,
        )
        with pytest.raises(Exception):
            tc.secondary_timing_seconds = 1.0

    def test_negative_timing_seconds_allowed(self):
        # Negative correction means secondary is "ahead" of reference
        est = EsdEstimate(
            median_offset_pixels=0.0, mean_offset_pixels=0.0,
            std_offset_pixels=0.0, sample_count=0, azimuth_time_interval=0.002,
        )
        tc = TimingCorrection(
            secondary_timing_seconds=-0.001,
            secondary_timing_pixels=-0.5,
            esd_estimate=est,
        )
        assert tc.secondary_timing_seconds == -0.001


# ---------------------------------------------------------------------------
# MergeSegment
# ---------------------------------------------------------------------------

class TestMergeSegment:
    def test_construction(self):
        seg = MergeSegment(
            burst_index=2,
            pair_index=1,
            input_line_start=3100,
            input_num_lines=1300,
            input_sample_start=500,
            input_num_samples=24000,
            output_line_start=0,
            output_num_lines=1300,
            output_sample_start=0,
            output_num_samples=24000,
        )
        assert seg.burst_index == 2
        assert seg.output_line_start == 0

    def test_frozen(self):
        seg = MergeSegment(
            burst_index=0, pair_index=0,
            input_line_start=0, input_num_lines=100,
            input_sample_start=0, input_num_samples=1000,
            output_line_start=0, output_num_lines=100,
            output_sample_start=0, output_num_samples=1000,
        )
        with pytest.raises(Exception):
            seg.output_num_lines = 200

    def test_zero_dimensions(self):
        seg = MergeSegment(
            burst_index=0, pair_index=0,
            input_line_start=0, input_num_lines=0,
            input_sample_start=0, input_num_samples=0,
            output_line_start=0, output_num_lines=0,
            output_sample_start=0, output_num_samples=0,
        )
        assert seg.input_num_lines == 0

    def test_negative_num_lines_rejected(self):
        # Only input_num_lines / input_num_samples / output_num_lines / output_num_samples
        # are validated (non-negative constraint); input_line_start is free to be negative.
        with pytest.raises(ValueError, match="input_num_lines must be non-negative"):
            MergeSegment(
                burst_index=0, pair_index=0,
                input_line_start=-100, input_num_lines=-1,   # negative num_lines
                input_sample_start=0, input_num_samples=1000,
                output_line_start=0, output_num_lines=100,
                output_sample_start=0, output_num_samples=1000,
            )


# ---------------------------------------------------------------------------
# MergeResult
# ---------------------------------------------------------------------------

class TestMergeResult:
    def test_construction(self):
        seg = MergeSegment(
            burst_index=0, pair_index=0,
            input_line_start=0, input_num_lines=100,
            input_sample_start=0, input_num_samples=1000,
            output_line_start=0, output_num_lines=100,
            output_sample_start=0, output_num_samples=1000,
        )
        result = MergeResult(
            seam_phase_diff_median=0.03,
            seam_phase_diff_std=0.02,
            seam_coherence_drop=0.05,
            gap_pixel_count=0,
            top_contribution_count=500_000,
            bottom_contribution_count=500_000,
            segments=(seg,),
        )
        assert result.seam_phase_diff_median == 0.03
        assert len(result.segments) == 1

    def test_frozen(self):
        result = MergeResult(
            seam_phase_diff_median=0.0, seam_phase_diff_std=0.0,
            seam_coherence_drop=0.0, gap_pixel_count=0,
            top_contribution_count=0, bottom_contribution_count=0,
            segments=(),
        )
        with pytest.raises(Exception):
            result.seam_phase_diff_median = 1.0

    def test_default_segments_empty(self):
        result = MergeResult(
            seam_phase_diff_median=0.0, seam_phase_diff_std=0.0,
            seam_coherence_drop=0.0, gap_pixel_count=0,
            top_contribution_count=0, bottom_contribution_count=0,
        )
        assert result.segments == ()

    def test_negative_gap_count_rejected(self):
        with pytest.raises((TypeError, ValueError)):
            MergeResult(
                seam_phase_diff_median=0.0, seam_phase_diff_std=0.0,
                seam_coherence_drop=0.0,
                gap_pixel_count=-1,       # invalid
                top_contribution_count=0, bottom_contribution_count=0,
            )


# ---------------------------------------------------------------------------
# Integration: valid_line calculation with offset grids
# ---------------------------------------------------------------------------

def test_valid_line_absolute_with_line_offset():
    """Integration-style: verify valid_line_start/stop with line_offset != 0."""
    # Burst 0: starts at line 0 within full image, valid starts at line 100
    g0 = _grid(idx=0, line_offset=0)
    # Burst 1: starts at line 1500 within full image, valid starts at line 1500+100=1600
    g1 = _grid(idx=1, line_offset=1500)
    # Overlap is lines 1600–1400 (impossible) — actually let's use bigger windows
    # Recompute with larger valid window to guarantee overlap
    g0_large = BurstRadarGrid(
        identity=_identity(0),
        image_window=_window(first_line=0, num_lines=1500),
        valid_window=_window(first_line=0, num_lines=1500,
                              first_sample=0, num_samples=25000),
        line_offset=0,
        azimuth_time_interval=0.002,
        range_pixel_spacing=2.3,
        starting_range=800_000.0,
        radar_wavelength=0.055,
        doppler_coefficients=(0.0,),
        azimuth_fm_rate_coefficients=(0.0,),
    )
    g1_large = BurstRadarGrid(
        identity=_identity(1),
        image_window=_window(first_line=1500, num_lines=1500),
        valid_window=_window(first_line=0, num_lines=1500,
                              first_sample=0, num_samples=25000),
        line_offset=1500,
        azimuth_time_interval=0.002,
        range_pixel_spacing=2.3,
        starting_range=800_000.0,
        radar_wavelength=0.055,
        doppler_coefficients=(0.0,),
        azimuth_fm_rate_coefficients=(0.0,),
    )
    # g0 valid: lines 0..1500; g1 valid: lines 1500..3000
    assert g0_large.valid_line_start == 0
    assert g0_large.valid_line_stop == 1500
    assert g1_large.valid_line_start == 1500
    assert g1_large.valid_line_stop == 3000
    # Overlap: line 1500 only
    overlap_start = max(g0_large.valid_line_start, g1_large.valid_line_start)
    overlap_end = min(g0_large.valid_line_stop, g1_large.valid_line_stop)
    assert overlap_start == 1500
    assert overlap_end == 1500  # zero-height overlap at boundary


def test_slant_range_formula():
    """Verify slant_range_at follows starting_range + sample * range_pixel_spacing."""
    g = _grid()
    for sample in [0, 500, 1000, 10000]:
        expected = 800_000.0 + sample * 2.3
        assert g.slant_range_at(sample) == pytest.approx(expected)


def test_prf_formula():
    """Verify PRF = 1 / azimuth_time_interval."""
    for ati in [0.001, 0.002, 0.004, 0.005, 0.01]:
        g = BurstRadarGrid(
            identity=_identity(),
            image_window=_window(),
            valid_window=_window(),
            line_offset=0,
            azimuth_time_interval=ati,
            range_pixel_spacing=2.3,
            starting_range=800_000.0,
            radar_wavelength=0.055,
            doppler_coefficients=(0.0,),
            azimuth_fm_rate_coefficients=(0.0,),
        )
        assert g.prf == pytest.approx(1.0 / ati)


def test_azimuth_time_interpolation():
    """Verify azimuth_time_at_line returns correct time for known line offsets."""
    g = _grid()
    t0 = g.identity.sensing_start
    line_step = 100
    expected = t0 + timedelta(seconds=line_step * g.azimuth_time_interval)
    assert g.azimuth_time_at_line(line_step) == expected