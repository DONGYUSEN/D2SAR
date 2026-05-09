"""Tests for tops_geometry — ISCE3 burst geometry adapters.

No imports from strip/tops_insar backends.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts.tops_model import (
    BurstIdentity,
    BurstWindow,
    BurstRadarGrid,
)
from scripts.tops_geometry import (
    S1RadarGrid,
    burst_to_radar_grid,
    build_isce3_orbit_from_safe,
    build_doppler_lut,
    run_geo2rdr_single_burst,
)


UTC = timezone.utc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_identity(idx: int = 0) -> BurstIdentity:
    return BurstIdentity(
        swath="IW2",
        burst_index=idx,
        sensing_start=datetime(2024, 1, 15, 4, 0, idx * 3, tzinfo=UTC),
        sensing_stop=datetime(2024, 1, 15, 4, 0, idx * 3 + 2, tzinfo=UTC),
        polarization="VV",
        orbit_direction="ascending",
        azimuth_steering_rate=0.0017,
    )


def _make_burst(idx: int = 0) -> BurstRadarGrid:
    return BurstRadarGrid(
        identity=_make_identity(idx),
        image_window=BurstWindow(first_line=idx * 1500, num_lines=1500,
                                first_sample=0, num_samples=25000),
        valid_window=BurstWindow(first_line=100, num_lines=1300,
                                 first_sample=500, num_samples=24000),
        line_offset=idx * 1500,
        azimuth_time_interval=0.002,   # prf = 500 Hz
        range_pixel_spacing=2.3,
        starting_range=800000.0,
        radar_wavelength=0.05546576,
        doppler_coefficients=(0.0,),
        azimuth_fm_rate_coefficients=(0.0,),
    )


# ---------------------------------------------------------------------------
# S1RadarGrid — construction & property tests
# ---------------------------------------------------------------------------

class TestS1RadarGridConstruction:
    """S1RadarGrid dataclass construction and property calculations."""

    def test_valid_construction(self):
        grid = S1RadarGrid(
            sensing_start=datetime(2024, 1, 15, 4, 0, 0, tzinfo=UTC),
            wavelength=0.05546576,
            prf=500.0,
            starting_range=800000.0,
            range_pixel_spacing=2.3,
            number_of_lines=1300,
            number_of_samples=24000,
            look_side="right",
        )
        assert grid.wavelength == 0.05546576
        assert grid.prf == 500.0
        assert grid.number_of_lines == 1300
        assert grid.number_of_samples == 24000
        assert grid.look_side == "right"

    def test_wavelength_validation_negative(self):
        with pytest.raises(ValueError, match="wavelength must be positive"):
            S1RadarGrid(
                sensing_start=datetime(2024, 1, 15, 4, 0, 0, tzinfo=UTC),
                wavelength=-0.055,
                prf=500.0,
                starting_range=800000.0,
                range_pixel_spacing=2.3,
                number_of_lines=1300,
                number_of_samples=24000,
            )

    def test_wavelength_validation_zero(self):
        with pytest.raises(ValueError, match="wavelength must be positive"):
            S1RadarGrid(
                sensing_start=datetime(2024, 1, 15, 4, 0, 0, tzinfo=UTC),
                wavelength=0.0,
                prf=500.0,
                starting_range=800000.0,
                range_pixel_spacing=2.3,
                number_of_lines=1300,
                number_of_samples=24000,
            )

    def test_prf_validation_negative(self):
        with pytest.raises(ValueError, match="prf must be positive"):
            S1RadarGrid(
                sensing_start=datetime(2024, 1, 15, 4, 0, 0, tzinfo=UTC),
                wavelength=0.055,
                prf=-500.0,
                starting_range=800000.0,
                range_pixel_spacing=2.3,
                number_of_lines=1300,
                number_of_samples=24000,
            )

    def test_prf_validation_zero(self):
        with pytest.raises(ValueError, match="prf must be positive"):
            S1RadarGrid(
                sensing_start=datetime(2024, 1, 15, 4, 0, 0, tzinfo=UTC),
                wavelength=0.055,
                prf=0.0,
                starting_range=800000.0,
                range_pixel_spacing=2.3,
                number_of_lines=1300,
                number_of_samples=24000,
            )

    def test_range_pixel_spacing_validation_negative(self):
        with pytest.raises(ValueError, match="range_pixel_spacing must be positive"):
            S1RadarGrid(
                sensing_start=datetime(2024, 1, 15, 4, 0, 0, tzinfo=UTC),
                wavelength=0.055,
                prf=500.0,
                starting_range=800000.0,
                range_pixel_spacing=-2.3,
                number_of_lines=1300,
                number_of_samples=24000,
            )

    def test_range_pixel_spacing_validation_zero(self):
        with pytest.raises(ValueError, match="range_pixel_spacing must be positive"):
            S1RadarGrid(
                sensing_start=datetime(2024, 1, 15, 4, 0, 0, tzinfo=UTC),
                wavelength=0.055,
                prf=500.0,
                starting_range=800000.0,
                range_pixel_spacing=0.0,
                number_of_lines=1300,
                number_of_samples=24000,
            )

    def test_number_of_lines_validation_zero(self):
        with pytest.raises(ValueError, match="number_of_lines must be positive"):
            S1RadarGrid(
                sensing_start=datetime(2024, 1, 15, 4, 0, 0, tzinfo=UTC),
                wavelength=0.055,
                prf=500.0,
                starting_range=800000.0,
                range_pixel_spacing=2.3,
                number_of_lines=0,
                number_of_samples=24000,
            )

    def test_number_of_samples_validation_zero(self):
        with pytest.raises(ValueError, match="number_of_samples must be positive"):
            S1RadarGrid(
                sensing_start=datetime(2024, 1, 15, 4, 0, 0, tzinfo=UTC),
                wavelength=0.055,
                prf=500.0,
                starting_range=800000.0,
                range_pixel_spacing=2.3,
                number_of_lines=1300,
                number_of_samples=0,
            )


# ---------------------------------------------------------------------------
# S1RadarGrid — formula verification (slant_range, azimuth_time)
# ---------------------------------------------------------------------------

class TestS1RadarGridFormulas:
    """Verify slant_range_at and azimuth_time_at_line formulas."""

    def test_slant_range_at_first_sample(self):
        """slant_range(sample=0) should equal starting_range."""
        grid = S1RadarGrid(
            sensing_start=datetime(2024, 1, 15, 4, 0, 0, tzinfo=UTC),
            wavelength=0.05546576,
            prf=500.0,
            starting_range=800000.0,
            range_pixel_spacing=2.3,
            number_of_lines=1300,
            number_of_samples=24000,
        )
        assert grid.slant_range_at(0) == 800000.0

    def test_slant_range_at_mid_sample(self):
        """slant_range(10000) = 800000 + 10000 * 2.3 = 823000."""
        grid = S1RadarGrid(
            sensing_start=datetime(2024, 1, 15, 4, 0, 0, tzinfo=UTC),
            wavelength=0.05546576,
            prf=500.0,
            starting_range=800000.0,
            range_pixel_spacing=2.3,
            number_of_lines=1300,
            number_of_samples=24000,
        )
        assert grid.slant_range_at(10000) == pytest.approx(823000.0)

    def test_slant_range_at_last_sample(self):
        """slant_range(23999) = 800000 + 23999 * 2.3."""
        grid = S1RadarGrid(
            sensing_start=datetime(2024, 1, 15, 4, 0, 0, tzinfo=UTC),
            wavelength=0.05546576,
            prf=500.0,
            starting_range=800000.0,
            range_pixel_spacing=2.3,
            number_of_lines=1300,
            number_of_samples=24000,
        )
        expected = 800000.0 + 23999 * 2.3
        assert grid.slant_range_at(23999) == pytest.approx(expected)

    def test_azimuth_time_at_line_zero(self):
        """azimuth_time_at_line(0) should equal sensing_start."""
        t0 = datetime(2024, 1, 15, 4, 0, 0, tzinfo=UTC)
        grid = S1RadarGrid(
            sensing_start=t0,
            wavelength=0.05546576,
            prf=500.0,
            starting_range=800000.0,
            range_pixel_spacing=2.3,
            number_of_lines=1300,
            number_of_samples=24000,
        )
        assert grid.azimuth_time_at_line(0) == t0

    def test_azimuth_time_at_line_one(self):
        """t(1) = sensing_start + 1/prf = sensing_start + 0.002 s."""
        t0 = datetime(2024, 1, 15, 4, 0, 0, tzinfo=UTC)
        grid = S1RadarGrid(
            sensing_start=t0,
            wavelength=0.05546576,
            prf=500.0,
            starting_range=800000.0,
            range_pixel_spacing=2.3,
            number_of_lines=1300,
            number_of_samples=24000,
        )
        expected = datetime(2024, 1, 15, 4, 0, 0, 2000, tzinfo=UTC)
        assert grid.azimuth_time_at_line(1) == expected

    def test_azimuth_time_at_line_large(self):
        """t(1299) = sensing_start + 1299 / 500 = sensing_start + 2.598 s."""
        t0 = datetime(2024, 1, 15, 4, 0, 0, tzinfo=UTC)
        grid = S1RadarGrid(
            sensing_start=t0,
            wavelength=0.05546576,
            prf=500.0,
            starting_range=800000.0,
            range_pixel_spacing=2.3,
            number_of_lines=1300,
            number_of_samples=24000,
        )
        t_expected = t0.timestamp() + 1299 / 500.0
        result = grid.azimuth_time_at_line(1299)
        assert result.timestamp() == pytest.approx(t_expected)


# ---------------------------------------------------------------------------
# S1RadarGrid — boundary cases
# ---------------------------------------------------------------------------

class TestS1RadarGridBoundary:
    """Edge-case inputs: zero views, large sample counts, etc."""

    def test_minimum_valid_grid(self):
        """One line, one sample — smallest possible valid grid."""
        grid = S1RadarGrid(
            sensing_start=datetime(2024, 1, 15, 4, 0, 0, tzinfo=UTC),
            wavelength=0.055,
            prf=500.0,
            starting_range=800000.0,
            range_pixel_spacing=2.3,
            number_of_lines=1,
            number_of_samples=1,
        )
        assert grid.slant_range_at(0) == 800000.0
        t1 = grid.azimuth_time_at_line(1)
        t0 = grid.sensing_start
        assert abs((t1 - t0).total_seconds() - 1 / 500.0) < 1e-9

    def test_very_large_sample_index(self):
        """slant_range_at(1_000_000) — verify no overflow."""
        grid = S1RadarGrid(
            sensing_start=datetime(2024, 1, 15, 4, 0, 0, tzinfo=UTC),
            wavelength=0.055,
            prf=500.0,
            starting_range=800000.0,
            range_pixel_spacing=2.3,
            number_of_lines=1300,
            number_of_samples=24000,
        )
        sr = grid.slant_range_at(1_000_000)
        assert sr == pytest.approx(800000.0 + 1_000_000 * 2.3)

    def test_very_large_line_index(self):
        """azimuth_time_at_line(10_000_000) — verify no overflow."""
        grid = S1RadarGrid(
            sensing_start=datetime(2024, 1, 15, 4, 0, 0, tzinfo=UTC),
            wavelength=0.055,
            prf=500.0,
            starting_range=800000.0,
            range_pixel_spacing=2.3,
            number_of_lines=1300,
            number_of_samples=24000,
        )
        t = grid.azimuth_time_at_line(10_000_000)
        expected_ts = grid.sensing_start.timestamp() + 10_000_000 / 500.0
        assert t.timestamp() == pytest.approx(expected_ts)


# ---------------------------------------------------------------------------
# burst_to_radar_grid — adapter from BurstRadarGrid
# ---------------------------------------------------------------------------

class TestBurstToRadarGrid:
    """burst_to_radar_grid conversion and prf derivation."""

    def test_prf_derivation(self):
        """prf = 1 / azimuth_time_interval = 1 / 0.002 = 500."""
        burst = _make_burst(idx=0)
        grid = burst_to_radar_grid(burst)
        assert grid.prf == pytest.approx(1.0 / 0.002)  # 500.0

    def test_wavelength_forwarded(self):
        burst = _make_burst(idx=0)
        grid = burst_to_radar_grid(burst)
        assert grid.wavelength == burst.radar_wavelength

    def test_range_params_forwarded(self):
        burst = _make_burst(idx=0)
        grid = burst_to_radar_grid(burst)
        assert grid.starting_range == burst.starting_range
        assert grid.range_pixel_spacing == burst.range_pixel_spacing

    def test_sensing_start_forwarded(self):
        burst = _make_burst(idx=0)
        grid = burst_to_radar_grid(burst)
        assert grid.sensing_start == burst.identity.sensing_start

    def test_valid_window_dimensions_used(self):
        """number_of_lines/samples come from valid_window, not image_window."""
        burst = _make_burst(idx=0)
        assert burst.image_window.num_lines == 1500
        assert burst.valid_window.num_lines == 1300
        grid = burst_to_radar_grid(burst)
        assert grid.number_of_lines == 1300
        assert grid.number_of_samples == 24000

    def test_invalid_azimuth_time_interval_raises(self):
        # BurstRadarGrid itself guards against azimuth_time_interval <= 0,
        # so the error is raised at BurstRadarGrid construction time.
        with pytest.raises(ValueError, match="azimuth_time_interval must be positive"):
            BurstRadarGrid(
                identity=_make_identity(),
                image_window=BurstWindow(first_line=0, num_lines=1500,
                                         first_sample=0, num_samples=25000),
                valid_window=BurstWindow(first_line=100, num_lines=1300,
                                         first_sample=500, num_samples=24000),
                line_offset=0,
                azimuth_time_interval=0.0,   # invalid
                range_pixel_spacing=2.3,
                starting_range=800000.0,
                radar_wavelength=0.05546576,
                doppler_coefficients=(0.0,),
                azimuth_fm_rate_coefficients=(0.0,),
            )

    def test_negative_azimuth_time_interval_raises(self):
        # Same as above: error raised at BurstRadarGrid construction.
        with pytest.raises(ValueError, match="azimuth_time_interval must be positive"):
            BurstRadarGrid(
                identity=_make_identity(),
                image_window=BurstWindow(first_line=0, num_lines=1500,
                                         first_sample=0, num_samples=25000),
                valid_window=BurstWindow(first_line=100, num_lines=1300,
                                         first_sample=500, num_samples=24000),
                line_offset=0,
                azimuth_time_interval=-0.002,
                range_pixel_spacing=2.3,
                starting_range=800000.0,
                radar_wavelength=0.05546576,
                doppler_coefficients=(0.0,),
                azimuth_fm_rate_coefficients=(0.0,),
            )


# ---------------------------------------------------------------------------
# Spike functions — NotImplementedError
# ---------------------------------------------------------------------------

class TestSpikeFunctions:
    """Spike stubs raise NotImplementedError with descriptive messages."""

    def test_build_isce3_orbit_from_safe_raises(self):
        with pytest.raises(NotImplementedError, match="spike stub"):
            build_isce3_orbit_from_safe(
                Path("/fake/safe"),
                datetime(2024, 1, 15, tzinfo=UTC),
                datetime(2024, 1, 15, tzinfo=UTC),
            )

    def test_build_doppler_lut_raises(self):
        with pytest.raises(NotImplementedError, match="spike stub"):
            build_doppler_lut(_make_burst())

    def test_run_geo2rdr_single_burst_raises(self):
        with pytest.raises(NotImplementedError, match="spike stub"):
            run_geo2rdr_single_burst(
                _make_burst(idx=0),
                _make_burst(idx=1),
                dem_path=Path("/fake/dem.tif"),
                work_dir=Path("/tmp/geo2rdr_work"),
                use_gpu=False,
            )


# ---------------------------------------------------------------------------
# S1RadarGrid — frozen immutability
# ---------------------------------------------------------------------------

class TestS1RadarGridImmutability:
    """S1RadarGrid is frozen; mutation raises FrozenInstanceError."""

    def test_frozen_cannot_mutate_wavelength(self):
        grid = S1RadarGrid(
            sensing_start=datetime(2024, 1, 15, 4, 0, 0, tzinfo=UTC),
            wavelength=0.05546576,
            prf=500.0,
            starting_range=800000.0,
            range_pixel_spacing=2.3,
            number_of_lines=1300,
            number_of_samples=24000,
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            grid.wavelength = 0.1
