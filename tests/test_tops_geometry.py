"""Tests for tops_geometry — ISCE3 burst geometry adapters.

No imports from strip/tops_insar backends.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
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
    _geo2rdr_valid_mask,
    run_geo2rdr_single_burst,
    _parse_eof_state_vectors,
    _find_eof_file,
    _resolve_orbit_file_for_safe,
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
# Spike functions — ISCE3 integration tests
# ---------------------------------------------------------------------------

# Path to real EOF file for testing
_EOF_PATH = Path(__file__).parents[1] / "result" / "orbits" / "S1A_OPER_AUX_POEORB_OPOD_20230715T080803_V20230624T225942_20230626T005942.EOF"


class TestParseEofStateVectors:
    """Test EOF parsing directly (no ISCE3 dependency)."""

    def test_parse_eof_returns_list_of_tuples(self):
        """_parse_eof_state_vectors returns list of (datetime, x, y, z, vx, vy, vz)."""
        if not _EOF_PATH.exists():
            pytest.skip(f"EOF file not found: {_EOF_PATH}")

        state_vectors = _parse_eof_state_vectors(_EOF_PATH)
        assert isinstance(state_vectors, list)
        assert len(state_vectors) > 0
        assert all(len(sv) == 7 for sv in state_vectors)

    def test_parse_eof_sorted_by_datetime(self):
        """State vectors are sorted by datetime."""
        if not _EOF_PATH.exists():
            pytest.skip(f"EOF file not found: {_EOF_PATH}")

        state_vectors = _parse_eof_state_vectors(_EOF_PATH)
        datetimes = [sv[0] for sv in state_vectors]
        assert datetimes == sorted(datetimes)

    def test_parse_eof_positions_valid(self):
        """State vector positions are realistic (ECEF in km scale)."""
        if not _EOF_PATH.exists():
            pytest.skip(f"EOF file not found: {_EOF_PATH}")

        state_vectors = _parse_eof_state_vectors(_EOF_PATH)
        for dt, x, y, z, vx, vy, vz in state_vectors:
            # ECEF positions should be ~7000 km magnitude
            assert -1e7 < x < 1e7
            assert -1e7 < y < 1e7
            assert -1e7 < z < 1e7
            # Velocities should be ~7000 m/s scale
            assert -1e4 < vx < 1e4
            assert -1e4 < vy < 1e4
            assert -1e4 < vz < 1e4


class TestFindEofFile:
    """Test EOF file discovery."""

    def test_find_eof_file_returns_none_for_empty_dir(self, tmp_path):
        """_find_eof_file returns None when no EOF files exist."""
        result = _find_eof_file(tmp_path)
        assert result is None

    def test_find_eof_file_finds_poeorb(self, tmp_path):
        """_find_eof_file finds POEORB files."""
        # Create a fake EOF file
        fake_eof = tmp_path / "aux" / "S1A_OPER_AUX_POEORB_TEST.EOF"
        fake_eof.parent.mkdir()
        fake_eof.write_text("<?xml version='1.0'?><Earth_Explorer_File></Earth_Explorer_File>")

        result = _find_eof_file(tmp_path)
        assert result is not None
        assert result.name == "S1A_OPER_AUX_POEORB_TEST.EOF"

class TestResolveOrbitFileForSafe:
    """Test orbit auto-download fallback."""

    def test_downloads_when_local_orbit_is_missing(self, tmp_path):
        downloaded = tmp_path / "orbits" / "S1A_OPER_AUX_POEORB_TEST.EOF"
        downloaded.parent.mkdir()
        downloaded.write_text("<?xml version='1.0'?><Earth_Explorer_File></Earth_Explorer_File>")

        with patch("scripts.tops_geometry._find_eof_file", return_value=None):
            with patch("scripts.tops_geometry.resolve_orbit_for_product") as resolve_mock:
                resolve_mock.return_value = SimpleNamespace(path=str(downloaded))
                result = _resolve_orbit_file_for_safe(
                    tmp_path / "S1A_IW_SLC__1SDV_20230625T114146_20230625T114213_049142_05E8CA_CCD3.SAFE",
                    orbit_dir=None,
                )

        assert result == downloaded
        resolve_mock.assert_called_once()
        _, kwargs = resolve_mock.call_args
        assert kwargs["download"] is True
        assert kwargs["orbit_dir"] == tmp_path / "orbits"


class TestBuildIsce3OrbitFromSafe:
    """Test ISCE3 orbit building from SAFE."""

    def test_raises_not_implemented_when_isce3_unavailable(self):
        """Raises NotImplementedError when ISCE3 C++ bindings are not available."""
        with patch("scripts.tops_geometry._get_isce3", side_effect=NotImplementedError("isce3 C++ bindings are not available")):
            with pytest.raises(NotImplementedError):
                build_isce3_orbit_from_safe(
                    Path("/fake/safe"),
                    datetime(2024, 1, 15, tzinfo=UTC),
                    datetime(2024, 1, 15, tzinfo=UTC),
                )

    def test_raises_file_not_found_when_auto_download_fails(self):
        """Raises FileNotFoundError when local EOF is missing and download fails."""
        with patch("scripts.tops_geometry._find_eof_file", return_value=None):
            with patch("scripts.tops_geometry.resolve_orbit_for_product", return_value=None):
                with pytest.raises(FileNotFoundError, match="auto-download"):
                    _resolve_orbit_file_for_safe(
                        Path("/fake/S1A_IW_SLC__1SDV_20240115T040000_20240115T040002_TEST.SAFE"),
                        orbit_dir=None,
                    )


class TestBuildDopplerLut:
    """Test Doppler LUT building."""

    def test_raises_not_implemented_when_isce3_unavailable(self):
        """Raises NotImplementedError when ISCE3 C++ bindings are not available."""
        with patch("scripts.tops_geometry._get_isce3", side_effect=NotImplementedError("isce3 C++ bindings are not available")):
            with pytest.raises(NotImplementedError):
                build_doppler_lut(_make_burst())


class TestRunGeo2RdrSingleBurst:
    """Test Geo2Rdr single burst execution."""

    def test_raises_not_implemented_when_isce3_unavailable(self):
        """Raises NotImplementedError when ISCE3 C++ bindings are not available."""
        with patch("scripts.tops_geometry._get_isce3", side_effect=NotImplementedError("isce3 C++ bindings are not available")):
            with pytest.raises(NotImplementedError):
                run_geo2rdr_single_burst(
                    _make_burst(idx=0),
                    _make_burst(idx=1),
                    dem_path=Path("/fake/dem.tif"),
                    work_dir=Path("/tmp/geo2rdr_work"),
                )

class TestGeo2RdrValidMask:
    """Validate ISCE3 Geo2Rdr NULL_VALUE handling."""

    def test_null_value_cells_are_excluded(self):
        range_offsets = np.array([[0.0, -1.0e6], [1.5, 2.0]])
        azimuth_offsets = np.array([[0.0, 3.0], [-1.0e6, 4.0]])
        mask = _geo2rdr_valid_mask(range_offsets, azimuth_offsets)
        assert mask.dtype == bool
        assert mask.tolist() == [[True, False], [False, True]]


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
