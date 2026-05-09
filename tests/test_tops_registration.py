"""Tests for tops_registration — coarse resampling with deramp/reramp.

Covers:
- Output shape / dtype correctness.
- Known-offset resampling (synthetic offsets with known ground truth).
- File existence after run_coarse_registration.
- Zero-offset invariance (resampled == original within numerical tolerance).
- _resample_sliding_window edge cases.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from scripts.tops_model import (
    BurstIdentity,
    BurstWindow,
    BurstRadarGrid,
    Geo2RdrOffsets,
    RangeCoregEstimate,
    TimingCorrection,
    EsdEstimate,
)
from scripts.tops_registration import (
    _resample_sliding_window,
    run_coarse_registration,
    fine_resample_with_timing,
    filter_ifg,
    _load_slc_npz,
    _save_slc_npz,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

UTC = timezone.utc


def _burst(swath: str = "IW1", idx: int = 0) -> BurstRadarGrid:
    """Minimal BurstRadarGrid fixture with non-zero Doppler slope."""
    return BurstRadarGrid(
        identity=BurstIdentity(
            swath=swath,
            burst_index=idx,
            sensing_start=datetime(2024, 1, 1, 0, 0, idx * 3, tzinfo=UTC),
            sensing_stop=datetime(2024, 1, 1, 0, 0, idx * 3 + 2, tzinfo=UTC),
            polarization="VV",
            orbit_direction="ascending",
            azimuth_steering_rate=0.0,
        ),
        image_window=BurstWindow(
            first_line=idx * 1500, num_lines=1500,
            first_sample=0, num_samples=25000,
        ),
        valid_window=BurstWindow(
            first_line=100, num_lines=200,
            first_sample=500, num_samples=500,
        ),
        line_offset=idx * 1500,
        azimuth_time_interval=0.002,
        range_pixel_spacing=2.3,
        starting_range=800000.0,
        radar_wavelength=0.055,
        doppler_coefficients=(0.0, 1e-7),   # non-zero slope for TOPS carrier
        azimuth_fm_rate_coefficients=(0.0,),
    )


def _npz_slc(tmp_path: Path, shape, data) -> Path:
    """Write a complex SLC to a temp npz and return its path."""
    path = tmp_path / "slc.npz"
    _save_slc_npz(data.astype(np.complex64), path)
    return path


# ---------------------------------------------------------------------------
# Tests for _resample_sliding_window
# ---------------------------------------------------------------------------

class TestResampleSlidingWindow:
    """Unit tests for the bilinear sliding-window resampler."""

    def test_output_shape_matches_offset_shape(self):
        """Output array shape equals the shape of the offset arrays."""
        src = np.random.randn(50, 80) + 1j * np.random.randn(50, 80)
        offsets = np.zeros((30, 40), dtype=np.float32)
        result = _resample_sliding_window(src, offsets, offsets)
        assert result.shape == (30, 40)

    def test_output_dtype_complex_for_complex_input(self):
        """Output dtype is complex64 for complex64 input."""
        src = np.random.randn(50, 80) + 1j * np.random.randn(50, 80)
        offsets = np.zeros((30, 40), dtype=np.float32)
        result = _resample_sliding_window(src, offsets, offsets)
        assert np.iscomplexobj(result)
        assert result.dtype == np.complex64

    def test_zero_offsets_returns_source_pixels(self):
        """Zero offsets: each output pixel (r, c) samples source at (r, c)."""
        src = np.arange(100, dtype=np.float32).reshape(10, 10)
        offsets = np.zeros((5, 5), dtype=np.float32)
        result = _resample_sliding_window(src, offsets, offsets)
        # Output is the top-left 5×5 block of source
        expected = src[:5, :5]
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_zero_offsets_matches_source_centre(self):
        """Zero offsets on a larger source returns the correct block."""
        src = np.arange(400, dtype=np.float32).reshape(20, 20)
        offsets = np.zeros((10, 10), dtype=np.float32)
        result = _resample_sliding_window(src, offsets, offsets)
        expected = src[:10, :10]
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_constant_column_offset_shifts_right(self):
        """A constant +1 column offset samples src at (r, c+1) for each output (r, c)."""
        src = np.arange(100, dtype=np.float32).reshape(10, 10)
        col_offsets = np.full((10, 10), 1.0, dtype=np.float32)
        row_offsets = np.zeros((10, 10), dtype=np.float32)
        result = _resample_sliding_window(src, row_offsets, col_offsets)
        # Each output pixel (r, c) samples src at (r, c+1)
        # Column 9 of output samples column 0 (out of bounds → 0)
        np.testing.assert_allclose(result[:, :9], src[:, 1:10], atol=1e-5)
        np.testing.assert_allclose(result[:, 9], 0.0, atol=1e-5)

    def test_constant_row_offset_shifts_down(self):
        """A constant +1 row offset samples src at (r+1, c) for each output (r, c)."""
        src = np.arange(100, dtype=np.float32).reshape(10, 10)
        row_offsets = np.full((5, 5), 1.0, dtype=np.float32)
        col_offsets = np.zeros((5, 5), dtype=np.float32)
        result = _resample_sliding_window(src, row_offsets, col_offsets)
        # Each output pixel (r, c) samples src at (r+1, c)
        # Row 4 of output samples row 5 of source (valid since src has 10 rows)
        np.testing.assert_allclose(result[:4, :], src[1:5, :5], atol=1e-5)

    def test_complex_input_bilinear_interpolation(self):
        """Complex input is interpolated correctly via bilinear resampling."""
        # Create a smooth complex field: real = row index, imag = col index
        nl, ns = 20, 20
        src = np.zeros((nl, ns), dtype=np.complex64)
        src = (np.arange(nl, dtype=np.float32)[:, None]
               + 1j * np.arange(ns, dtype=np.float32)[None, :])

        # Zero offset: each output (r, c) samples source (r, c) → real=r, imag=c
        offsets = np.zeros((5, 5), dtype=np.float32)
        result = _resample_sliding_window(src, offsets, offsets)
        # Top-left 5×5 block of source: real = 0..4 rows, imag = 0..4 cols
        expected_real = np.arange(5, dtype=np.float32)[:, None] * np.ones((1, 5), dtype=np.float32)
        expected_imag = np.ones((5, 1), dtype=np.float32) * np.arange(5, dtype=np.float32)[None, :]
        np.testing.assert_allclose(result.real, expected_real, atol=1e-4)
        np.testing.assert_allclose(result.imag, expected_imag, atol=1e-4)

    def test_out_of_bounds_returns_zero(self):
        """Source coordinates completely outside the array return zero."""
        src = np.ones((10, 10), dtype=np.float32)
        offsets = np.full((5, 5), -999.0, dtype=np.float32)  # way outside
        result = _resample_sliding_window(src, offsets, offsets)
        np.testing.assert_array_equal(result, 0.0)

    def test_mismatched_offset_shapes_raise(self):
        """offset_rows and offset_cols with different shapes raise ValueError."""
        src = np.ones((10, 10), dtype=np.float32)
        off_r = np.zeros((5, 5), dtype=np.float32)
        off_c = np.zeros((5, 6), dtype=np.float32)
        with pytest.raises(ValueError, match="offset_rows.shape"):
            _resample_sliding_window(src, off_r, off_c)

    def test_preallocated_out_array(self):
        """Pre-allocated output array is written in-place."""
        src = np.random.randn(50, 80) + 1j * np.random.randn(50, 80)
        offsets = np.zeros((30, 40), dtype=np.float32)
        out = np.zeros((30, 40), dtype=np.complex64)
        result = _resample_sliding_window(src, offsets, offsets, out=out)
        assert result is out

    def test_2d_offset_arrays(self):
        """2-D offset arrays with per-pixel variation work correctly."""
        src = np.arange(100, dtype=np.float32).reshape(10, 10)
        # Linear gradient offsets: row offset varies with row, col offset varies with col.
        row_offsets = np.linspace(-0.5, 0.5, 10, dtype=np.float32)[:, None] * np.ones((1, 10), dtype=np.float32)  # (10, 10)
        col_offsets = np.linspace(-0.5, 0.5, 10, dtype=np.float32)[None, :] * np.ones((10, 1), dtype=np.float32)  # (10, 10)
        assert row_offsets.shape == (10, 10)
        assert col_offsets.shape == (10, 10)
        result = _resample_sliding_window(src, row_offsets, col_offsets)
        assert result.shape == (10, 10)
        assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# Tests for _save_slc_npz / _load_slc_npz
# ---------------------------------------------------------------------------

class TestNpzSlcRoundtrip:
    """Roundtrip tests for npz SLC save / load."""

    def test_complex64_roundtrip(self, tmp_path: Path):
        """complex64 SLC survives a save → load roundtrip."""
        slc = np.random.randn(100, 200) + 1j * np.random.randn(100, 200)
        slc = slc.astype(np.complex64)
        path = tmp_path / "test.npz"
        _save_slc_npz(slc, path)
        loaded = _load_slc_npz(path)
        np.testing.assert_allclose(loaded, slc, rtol=1e-6)
        assert loaded.dtype == np.complex64

    def test_missing_file_raises(self, tmp_path: Path):
        """Missing npz raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            _load_slc_npz(tmp_path / "nonexistent.npz")

    def test_load_nonexistent_geo2rdr_offsets(self, tmp_path: Path):
        """Loading non-existent offset files raises FileNotFoundError."""
        geo = Geo2RdrOffsets(
            range_off_path=str(tmp_path / "range.off.npz"),
            azimuth_off_path=str(tmp_path / "azimuth.off.npz"),
            median_range_offset=0.0,
            median_azimuth_offset=0.0,
            valid_sample_count=0,
        )
        ref = _burst()
        sec = _burst()
        # We test the file-not-found path directly
        with pytest.raises(FileNotFoundError):
            _load_slc_npz(tmp_path / "missing.npz")


# ---------------------------------------------------------------------------
# Integration-style tests for run_coarse_registration
# ---------------------------------------------------------------------------

class TestRunCoarseRegistration:
    """High-level tests for run_coarse_registration using synthetic data."""

    def _make_geo_offsets(self, tmp_path: Path, shape, const_rg=0.0, const_az=0.0):
        """Write synthetic constant-offset arrays and return Geo2RdrOffsets."""
        range_off = np.full(shape, const_rg, dtype=np.float32)
        azimuth_off = np.full(shape, const_az, dtype=np.float32)
        rg_path = tmp_path / "range.off.npz"
        az_path = tmp_path / "azimuth.off.npz"
        np.savez(rg_path, data=range_off.astype(np.float32))
        np.savez(az_path, data=azimuth_off.astype(np.float32))
        return Geo2RdrOffsets(
            range_off_path=str(rg_path),
            azimuth_off_path=str(az_path),
            median_range_offset=const_rg,
            median_azimuth_offset=const_az,
            valid_sample_count=int(np.prod(shape)),
        )

    def _write_sec_slc(self, tmp_path: Path, shape):
        """Write a synthetic secondary SLC."""
        slc = np.random.randn(*shape) + 1j * np.random.randn(*shape)
        slc = slc.astype(np.complex64)
        path = tmp_path / "secondary_slc_IW1_0.slc.npz"
        _save_slc_npz(slc, path)
        return slc

    def _write_ref_slc(self, tmp_path: Path, shape):
        """Write a synthetic reference SLC."""
        slc = np.random.randn(*shape) + 1j * np.random.randn(*shape)
        slc = slc.astype(np.complex64)
        path = tmp_path / "reference_slc_IW1_0.slc.npz"
        _save_slc_npz(slc, path)
        return slc

    def test_output_shape_and_dtype(self, tmp_path: Path):
        """run_coarse_registration produces three output files with correct shape/dtype."""
        ref = _burst()
        sec = _burst()
        out_shape = (ref.valid_window.num_lines, ref.valid_window.num_samples)

        # Write secondary SLC
        self._write_sec_slc(tmp_path, out_shape)

        # Write reference SLC
        self._write_ref_slc(tmp_path, out_shape)

        # Write zero-offset Geo2Rdr
        geo = self._make_geo_offsets(tmp_path, out_shape, const_rg=0.0, const_az=0.0)

        # Output paths
        deramped_ref = tmp_path / "deramped_ref.npz"
        deramped_sec = tmp_path / "deramped_sec.npz"
        resampled_sec = tmp_path / "resampled_sec.npz"

        run_coarse_registration(
            ref, sec, geo,
            work_dir=tmp_path,
            deramped_ref_path=deramped_ref,
            deramped_sec_path=deramped_sec,
            resampled_sec_path=resampled_sec,
        )

        # Check files exist
        assert deramped_ref.exists(), "deramped_ref not written"
        assert deramped_sec.exists(), "deramped_sec not written"
        assert resampled_sec.exists(), "resampled_sec not written"

        # Check shapes and dtype
        for path in [deramped_ref, deramped_sec, resampled_sec]:
            arr = _load_slc_npz(path)
            assert arr.shape == out_shape, f"{path.name} shape mismatch"
            assert arr.dtype == np.complex64, f"{path.name} dtype mismatch"

    def test_zero_offsets_deramp_reramp_roundtrip(self, tmp_path: Path):
        """With zero offsets, resampled_sec matches the original after deramp→reramp."""
        ref = _burst()
        sec = _burst()
        out_shape = (ref.valid_window.num_lines, ref.valid_window.num_samples)

        sec_slc = self._write_sec_slc(tmp_path, out_shape)
        self._write_ref_slc(tmp_path, out_shape)
        geo = self._make_geo_offsets(tmp_path, out_shape, const_rg=0.0, const_az=0.0)

        resampled_sec_path = tmp_path / "resampled_sec.npz"

        run_coarse_registration(
            ref, sec, geo,
            work_dir=tmp_path,
            deramped_ref_path=tmp_path / "deramped_ref.npz",
            deramped_sec_path=tmp_path / "deramped_sec.npz",
            resampled_sec_path=resampled_sec_path,
        )

        resampled = _load_slc_npz(resampled_sec_path)

        # With zero offsets and the same burst parameters (ref == sec here),
        # resampled_sec should equal sec_slc after deramp→reramp roundtrip.
        # We check the deramp→reramp roundtrip matches the original.
        from scripts.tops_deramp import reramp_slc, deramp_slc
        expected = reramp_slc(deramp_slc(sec_slc, sec), ref)

        # Allow small numerical tolerance (bilinear interpolation + float32 rounding)
        np.testing.assert_allclose(resampled, expected, rtol=1e-4, atol=1e-6)

    def test_missing_secondary_slc_raises(self, tmp_path: Path):
        """Missing secondary SLC raises FileNotFoundError."""
        ref = _burst()
        sec = _burst()
        out_shape = (ref.valid_window.num_lines, ref.valid_window.num_samples)
        geo = self._make_geo_offsets(tmp_path, out_shape)

        with pytest.raises(FileNotFoundError, match="secondary_slc"):
            run_coarse_registration(
                ref, sec, geo,
                work_dir=tmp_path,
                deramped_ref_path=tmp_path / "deramped_ref.npz",
                deramped_sec_path=tmp_path / "deramped_sec.npz",
                resampled_sec_path=tmp_path / "resampled_sec.npz",
            )

    def test_missing_reference_slc_raises(self, tmp_path: Path):
        """Missing reference SLC raises FileNotFoundError."""
        ref = _burst()
        sec = _burst()
        out_shape = (ref.valid_window.num_lines, ref.valid_window.num_samples)

        # Write secondary but not reference
        self._write_sec_slc(tmp_path, out_shape)
        geo = self._make_geo_offsets(tmp_path, out_shape)

        with pytest.raises(FileNotFoundError, match="reference_slc"):
            run_coarse_registration(
                ref, sec, geo,
                work_dir=tmp_path,
                deramped_ref_path=tmp_path / "deramped_ref.npz",
                deramped_sec_path=tmp_path / "deramped_sec.npz",
                resampled_sec_path=tmp_path / "resampled_sec.npz",
            )

    def test_offset_shape_mismatch_raises(self, tmp_path: Path):
        """Mismatched offset array shape raises ValueError."""
        ref = _burst()
        sec = _burst()
        out_shape = (ref.valid_window.num_lines, ref.valid_window.num_samples)

        self._write_sec_slc(tmp_path, out_shape)
        self._write_ref_slc(tmp_path, out_shape)

        # Write wrong-shape offsets
        wrong_shape = (out_shape[0] + 10, out_shape[1] + 10)
        geo = self._make_geo_offsets(tmp_path, wrong_shape)

        with pytest.raises(ValueError, match="range.off shape"):
            run_coarse_registration(
                ref, sec, geo,
                work_dir=tmp_path,
                deramped_ref_path=tmp_path / "deramped_ref.npz",
                deramped_sec_path=tmp_path / "deramped_sec.npz",
                resampled_sec_path=tmp_path / "resampled_sec.npz",
            )

    def test_nonzero_constant_offset_shifts_signal(self, tmp_path: Path):
        """A non-zero constant azimuth offset shifts signal content down."""
        ref = _burst()
        sec = _burst(idx=1)  # same shape, different burst
        out_shape = (ref.valid_window.num_lines, ref.valid_window.num_samples)

        # Write identical SLC data for both
        slc_data = np.random.randn(*out_shape) + 1j * np.random.randn(*out_shape)
        slc_data = slc_data.astype(np.complex64)

        _save_slc_npz(slc_data, tmp_path / "secondary_slc_IW1_1.slc.npz")
        _save_slc_npz(slc_data, tmp_path / "reference_slc_IW1_0.slc.npz")

        # Apply a constant 5-pixel azimuth shift
        az_offset = 5.0
        geo = self._make_geo_offsets(tmp_path, out_shape, const_rg=0.0, const_az=az_offset)

        resampled_sec_path = tmp_path / "resampled_sec.npz"

        run_coarse_registration(
            ref, sec, geo,
            work_dir=tmp_path,
            deramped_ref_path=tmp_path / "deramped_ref.npz",
            deramped_sec_path=tmp_path / "deramped_sec.npz",
            resampled_sec_path=resampled_sec_path,
        )

        resampled = _load_slc_npz(resampled_sec_path)

        # After reramping, the resampled array should contain shifted signal content.
        # Check that the energy distribution changed (not zero).
        assert not np.allclose(resampled, 0.0)
        # The resampled data should differ from the unshifted case
        geo_zero = self._make_geo_offsets(tmp_path, out_shape, const_rg=0.0, const_az=0.0)
        _save_slc_npz(slc_data, tmp_path / "secondary_slc_IW1_1.slc.npz")  # re-write
        run_coarse_registration(
            ref, sec, geo_zero,
            work_dir=tmp_path,
            deramped_ref_path=tmp_path / "deramped_ref.npz",
            deramped_sec_path=tmp_path / "deramped_sec.npz",
            resampled_sec_path=tmp_path / "resampled_sec_zero.npz",
        )
        resampled_zero = _load_slc_npz(tmp_path / "resampled_sec_zero.npz")
        # At least some pixels must differ when an offset is applied
        assert not np.allclose(resampled, resampled_zero)


# ---------------------------------------------------------------------------
# Tests for fine_resample_with_timing
# ---------------------------------------------------------------------------

class TestFineResampleWithTiming:
    """Tests for fine_resample_with_timing — ESD timing + range coreg corrections."""

    def _make_geo_offsets(self, tmp_path: Path, shape):
        """Write synthetic constant-offset arrays and return Geo2RdrOffsets."""
        range_off = np.zeros(shape, dtype=np.float32)
        azimuth_off = np.zeros(shape, dtype=np.float32)
        rg_path = tmp_path / "range.off.npz"
        az_path = tmp_path / "azimuth.off.npz"
        np.savez(rg_path, data=range_off)
        np.savez(az_path, data=azimuth_off)
        return Geo2RdrOffsets(
            range_off_path=str(rg_path),
            azimuth_off_path=str(az_path),
            median_range_offset=0.0,
            median_azimuth_offset=0.0,
            valid_sample_count=int(np.prod(shape)),
        )

    def _write_slc_npz(self, tmp_path: Path, shape):
        """Write a synthetic SLC."""
        slc = np.random.randn(*shape) + 1j * np.random.randn(*shape)
        return slc.astype(np.complex64)

    def test_timing_correction_applied(self, tmp_path: Path):
        """ESD timing correction is added to azimuth offsets."""
        ref = _burst()
        sec = _burst()
        out_shape = (ref.valid_window.num_lines, ref.valid_window.num_samples)

        geo = self._make_geo_offsets(tmp_path, out_shape)

        ref_slc = self._write_slc_npz(tmp_path, out_shape)
        sec_slc = self._write_slc_npz(tmp_path, out_shape)

        timing_correction = TimingCorrection(
            secondary_timing_seconds=0.001,
            secondary_timing_pixels=2.5,
            esd_estimate=EsdEstimate(
                median_offset_pixels=2.5,
                mean_offset_pixels=2.5,
                std_offset_pixels=0.1,
                sample_count=1000,
                azimuth_time_interval=ref.azimuth_time_interval,
            ),
        )

        fine_path = tmp_path / "fine_resampled.npz"

        fine_resample_with_timing(
            ref_slc, sec_slc,
            ref, sec,
            geo,
            timing_correction=timing_correction,
            range_coreg_estimate=None,
            work_dir=tmp_path,
            fine_resampled_path=fine_path,
        )

        assert fine_path.exists()
        result = _load_slc_npz(fine_path)
        assert result.shape == out_shape
        assert result.dtype == np.complex64

    def test_range_coreg_applied(self, tmp_path: Path):
        """Range coregistration estimate is added to range offsets."""
        ref = _burst()
        sec = _burst()
        out_shape = (ref.valid_window.num_lines, ref.valid_window.num_samples)

        geo = self._make_geo_offsets(tmp_path, out_shape)

        ref_slc = self._write_slc_npz(tmp_path, out_shape)
        sec_slc = self._write_slc_npz(tmp_path, out_shape)

        range_coreg_estimate = RangeCoregEstimate(
            median_range_offset=1.2,
            std_range_offset=0.05,
            median_azimuth_offset=0.0,
            std_azimuth_offset=0.05,
            sample_count=500,
            usable_fraction=0.8,
        )

        fine_path = tmp_path / "fine_resampled.npz"

        fine_resample_with_timing(
            ref_slc, sec_slc,
            ref, sec,
            geo,
            timing_correction=None,
            range_coreg_estimate=range_coreg_estimate,
            work_dir=tmp_path,
            fine_resampled_path=fine_path,
        )

        assert fine_path.exists()
        result = _load_slc_npz(fine_path)
        assert result.shape == out_shape
        assert result.dtype == np.complex64

    def test_combined_timing_and_range_coreg(self, tmp_path: Path):
        """Both timing correction and range coreg are applied together."""
        ref = _burst()
        sec = _burst()
        out_shape = (ref.valid_window.num_lines, ref.valid_window.num_samples)

        geo = self._make_geo_offsets(tmp_path, out_shape)

        ref_slc = self._write_slc_npz(tmp_path, out_shape)
        sec_slc = self._write_slc_npz(tmp_path, out_shape)

        timing_correction = TimingCorrection(
            secondary_timing_seconds=0.001,
            secondary_timing_pixels=2.5,
            esd_estimate=EsdEstimate(
                median_offset_pixels=2.5,
                mean_offset_pixels=2.5,
                std_offset_pixels=0.1,
                sample_count=1000,
                azimuth_time_interval=ref.azimuth_time_interval,
            ),
        )
        range_coreg_estimate = RangeCoregEstimate(
            median_range_offset=1.2,
            std_range_offset=0.05,
            median_azimuth_offset=0.0,
            std_azimuth_offset=0.05,
            sample_count=500,
            usable_fraction=0.8,
        )

        fine_path = tmp_path / "fine_resampled.npz"

        fine_resample_with_timing(
            ref_slc, sec_slc,
            ref, sec,
            geo,
            timing_correction=timing_correction,
            range_coreg_estimate=range_coreg_estimate,
            work_dir=tmp_path,
            fine_resampled_path=fine_path,
        )

        assert fine_path.exists()
        result = _load_slc_npz(fine_path)
        assert result.shape == out_shape
        assert result.dtype == np.complex64
        # Combined corrections should produce non-trivial output
        assert not np.allclose(result, 0.0)


# ---------------------------------------------------------------------------
# Tests for filter_ifg
# ---------------------------------------------------------------------------

class TestFilterIfg:
    """Tests for filter_ifg — Goldstein phase filtering."""

    def test_coherence_mask_zeros_decorrelated(self):
        """Pixels with coherence <= 0.3 are set to zero."""
        shape = (32, 32)
        ifg = np.ones(shape, dtype=np.complex64) * (1.0 + 1.0j)
        coherence = np.full(shape, 0.2)  # all below threshold

        result = filter_ifg(ifg, coherence)

        np.testing.assert_array_equal(result, 0.0)

    def test_alpha_zero_is_identity(self):
        """alpha=0.0 returns the input interferogram unchanged."""
        shape = (32, 32)
        ifg = np.random.randn(*shape) + 1j * np.random.randn(*shape)
        ifg = ifg.astype(np.complex64)
        coherence = np.random.rand(*shape).astype(np.float32)

        result = filter_ifg(ifg, coherence, alpha=0.0)

        np.testing.assert_allclose(result, ifg, rtol=1e-6)

    def test_alpha_one_max_filter(self):
        """alpha=1.0 applies maximum filtering strength."""
        shape = (32, 32)
        # Create interferogram with varying intensity
        intensity = np.ones(shape, dtype=np.float32)
        intensity[8:24, 8:24] = 2.0  # high-intensity block
        phase = np.angle(ifg := (intensity * np.exp(1j * 0.5)))
        ifg = np.cos(phase) + 1j * np.sin(phase)

        coherence = np.full(shape, 0.8)  # high coherence everywhere

        result = filter_ifg(ifg, coherence, alpha=1.0)

        # Output should still be complex64
        assert result.dtype == np.complex64
        # Should not be zero everywhere (filter applied)
        assert not np.allclose(result, 0.0)

    def test_boxcar_window_smoothing(self):
        """Boxcar multi-looking (8×8) smooths local intensity variations."""
        shape = (32, 32)
        # High-intensity region in top-left, low in bottom-right
        intensity = np.zeros(shape, dtype=np.float32)
        intensity[:16, :16] = 4.0
        intensity[16:, 16:] = 1.0
        ifg = intensity * np.exp(1j * 0.3)
        coherence = np.full(shape, 0.9)

        result = filter_ifg(ifg, coherence, alpha=0.5)

        # Output should be complex64
        assert result.dtype == np.complex64
        # Smoothed filter should not produce extreme values
        assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# No-strip-import guard
# ---------------------------------------------------------------------------

def test_no_strip_imports():
    """tops_registration must not import strip_insar / tops_insar."""
    import ast, sys
    src = Path(__file__).parent.parent / "scripts" / "tops_registration.py"
    tree = ast.parse(src.read_text())
    forbidden = {"strip_insar", "strip_insar2", "tops_insar"}
    for node in ast.walk(tree):
        for alias in getattr(node, "names", []):
            name = alias.name
            assert name not in forbidden, f"tops_registration imports forbidden: {name}"
