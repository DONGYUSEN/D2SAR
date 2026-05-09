"""Tests for tops_utils — shared pure-math utilities."""

from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta

import numpy as np
import pytest

from scripts.tops_model import BurstWindow
from scripts.tops_utils import (
    robust_median_with_mad,
    evaluate_polynomial,
    intersect_windows,
    ensure_utc,
    circular_mean,
    safe_divide,
    unwrap_phase_2d,
    geocode_image,
    pad_to_tile,
    estimate_memory_usage,
    pad_window_to_grid,
)


# =============================================================================
# robust_median_with_mad
# =============================================================================

class TestRobustMedianWithMad:
    def test_1d_no_outlier(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = robust_median_with_mad(data)
        assert result == pytest.approx(3.0)

    def test_1d_with_outlier(self):
        # 50 is a huge outlier; MAD clipping excludes it.
        # After exclusion: [1,2,3,4,5]; median = 3
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 50.0])
        result = robust_median_with_mad(data)
        assert result == pytest.approx(3.0)

    def test_2d_axis_none(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = robust_median_with_mad(data, axis=None)
        assert result == pytest.approx(2.5)

    def test_2d_axis_0(self):
        data = np.array([[1.0, 100.0], [3.0, 4.0], [5.0, 6.0]])
        result = robust_median_with_mad(data, axis=0)
        # col 0: data=[1,3,5], median=3, MAD=2, threshold=6 → [1,3,5] all pass → median=3
        # col 1: data=[100,4,6], median=6, MAD=2, threshold=6 → 100 has |diff|=94>6, excluded
        #         remaining [4,6], median=5
        expected = np.array([3.0, 5.0])
        np.testing.assert_allclose(result, expected)

    def test_all_same_value(self):
        data = np.array([7.0, 7.0, 7.0, 7.0])
        result = robust_median_with_mad(data)
        assert result == pytest.approx(7.0)

    def test_scalar_like_input(self):
        result = robust_median_with_mad(np.array(5.0))
        assert result == pytest.approx(5.0)


# =============================================================================
# evaluate_polynomial
# =============================================================================

class TestEvaluatePolynomial:
    def test_constant(self):
        coeffs = [3.0]
        x = np.array([0.0, 1.0, 2.0])
        result = evaluate_polynomial(coeffs, x)
        np.testing.assert_allclose(result, [3.0, 3.0, 3.0])

    def test_linear(self):
        # p(x) = 1 + 2*x
        coeffs = [1.0, 2.0]
        x = np.array([0.0, 1.0, 2.0])
        result = evaluate_polynomial(coeffs, x)
        np.testing.assert_allclose(result, [1.0, 3.0, 5.0])

    def test_quadratic(self):
        # p(x) = 1 + x + x^2
        coeffs = [1.0, 1.0, 1.0]
        x = np.array([0.0, 1.0, 2.0, 3.0])
        result = evaluate_polynomial(coeffs, x)
        np.testing.assert_allclose(result, [1.0, 3.0, 7.0, 13.0])

    def test_empty_coeffs(self):
        coeffs = []
        x = np.array([0.5, 1.5])
        result = evaluate_polynomial(coeffs, x)
        np.testing.assert_allclose(result, [0.0, 0.0])

    def test_scalar_input(self):
        coeffs = [1.0, 2.0, 3.0]
        result = evaluate_polynomial(coeffs, np.array(2.0))
        # 1 + 2*2 + 3*4 = 1 + 4 + 12 = 17
        assert result == pytest.approx(17.0)

    def test_negative_coeffs(self):
        coeffs = [-1.0, 0.5]
        x = np.array([4.0])
        result = evaluate_polynomial(coeffs, x)
        # -1 + 0.5*4 = 1
        assert result == pytest.approx(1.0)


# =============================================================================
# intersect_windows
# =============================================================================

class TestIntersectWindows:
    def test_full_overlap(self):
        w1 = BurstWindow(first_line=10, num_lines=100, first_sample=20, num_samples=200)
        w2 = BurstWindow(first_line=10, num_lines=100, first_sample=20, num_samples=200)
        result = intersect_windows(w1, w2)
        assert result.first_line == 10
        assert result.num_lines == 100
        assert result.first_sample == 20
        assert result.num_samples == 200

    def test_partial_overlap(self):
        w1 = BurstWindow(first_line=0, num_lines=100, first_sample=0, num_samples=100)
        w2 = BurstWindow(first_line=50, num_lines=100, first_sample=50, num_samples=100)
        result = intersect_windows(w1, w2)
        assert result.first_line == 50
        assert result.num_lines == 50
        assert result.first_sample == 50
        assert result.num_samples == 50

    def test_no_overlap_returns_sentinel(self):
        # Line overlap → 50, but no sample overlap: [0,50) vs [50,150)
        w1 = BurstWindow(first_line=0, num_lines=50, first_sample=0, num_samples=50)
        w2 = BurstWindow(first_line=50, num_lines=50, first_sample=50, num_samples=100)
        result = intersect_windows(w1, w2)
        # samples: max(0,50)=50, min(50,150)=50 → num_samples=0 → sentinel
        assert result.first_line == -1
        assert result.num_samples == 0

    def test_overlap_in_one_dimension_only(self):
        # w1: lines [0,100), samples [0,50)
        # w2: lines [100,150), samples [0,100)
        # → lines don't overlap but samples do → num_lines=0 → sentinel
        w1 = BurstWindow(first_line=0, num_lines=100, first_sample=0, num_samples=50)
        w2 = BurstWindow(first_line=100, num_lines=50, first_sample=0, num_samples=100)
        result = intersect_windows(w1, w2)
        assert result.first_line == -1
        assert result.num_lines == 0

    def test_one_pixel_overlap(self):
        w1 = BurstWindow(first_line=0, num_lines=10, first_sample=0, num_samples=10)
        w2 = BurstWindow(first_line=9, num_lines=10, first_sample=9, num_samples=10)
        result = intersect_windows(w1, w2)
        assert result.first_line == 9
        assert result.num_lines == 1
        assert result.first_sample == 9
        assert result.num_samples == 1


# =============================================================================
# ensure_utc
# =============================================================================

class TestEnsureUtc:
    def test_naive_datetime(self):
        dt = datetime(2024, 6, 15, 12, 30, 0)
        result = ensure_utc(dt)
        assert result.tzinfo == timezone.utc
        assert result.year == 2024

    def test_already_utc(self):
        dt = datetime(2024, 6, 15, 12, 30, 0, tzinfo=timezone.utc)
        result = ensure_utc(dt)
        assert result.tzinfo == timezone.utc

    def test_different_timezone_converts(self):
        importZone = timezone(timedelta(hours=5))
        dt = datetime(2024, 6, 15, 12, 30, 0, tzinfo=importZone)
        result = ensure_utc(dt)
        assert result.tzinfo == timezone.utc
        # 12:30 IST → 07:00 UTC
        assert result.hour == 7

    def test_with_microseconds(self):
        dt = datetime(2024, 6, 15, 12, 30, 0, 123456)
        result = ensure_utc(dt)
        assert result.microsecond == 123456


# =============================================================================
# circular_mean
# =============================================================================

class TestCircularMean:
    def test_scalar_empty_like(self):
        result = circular_mean(np.array([]))
        # mean of empty complex array is nan → angle is nan
        assert math.isnan(result)

    def test_single_phase(self):
        result = circular_mean(np.array([0.0]))
        assert result == pytest.approx(0.0)

    def test_uniform_distribution(self):
        # Four phases uniformly spaced at π/2 → exp sum ≈ 0 → angle returns 0
        phases = np.array([0.0, np.pi / 2, np.pi, -np.pi / 2])
        result = circular_mean(phases)
        assert abs(result) < 0.01

    def test_all_same_phase(self):
        phases = np.array([np.pi / 4, np.pi / 4, np.pi / 4])
        result = circular_mean(phases)
        assert result == pytest.approx(np.pi / 4)

    def test_opposite_phases_cancel(self):
        phases = np.array([0.0, np.pi])
        result = circular_mean(phases)
        assert abs(result) < 0.01

    def test_axis_0(self):
        phases = np.array([[0.0, np.pi / 4], [0.0, np.pi / 4]])
        result = circular_mean(phases, axis=0)
        np.testing.assert_allclose(result, [0.0, np.pi / 4])

    def test_axis_1(self):
        # Each row [π/4, π/4, π/4] averages to π/4
        phases = np.array([[np.pi / 4, np.pi / 4, np.pi / 4],
                           [np.pi / 4, np.pi / 4, np.pi / 4]])
        result = circular_mean(phases, axis=1)
        np.testing.assert_allclose(result, [np.pi / 4, np.pi / 4])

    def test_wrapped_result_range(self):
        # mean of [179°, -179°] should be close to 180° (not 0°)
        phases = np.array([179.0 * np.pi / 180, -179.0 * np.pi / 180])
        result = circular_mean(phases)
        assert abs(abs(result) - np.pi) < 0.1


# =============================================================================
# safe_divide
# =============================================================================

class TestSafeDivide:
    def test_normal_division(self):
        result = safe_divide(10.0, 2.0)
        assert result == pytest.approx(5.0)

    def test_zero_denominator_returns_fill(self):
        result = safe_divide(10.0, 0.0, fill=-99.0)
        assert result == pytest.approx(-99.0)

    def test_nan_denominator_returns_fill(self):
        result = safe_divide(10.0, float("nan"), fill=-77.0)
        assert result == pytest.approx(-77.0)

    def test_array_division(self):
        a = np.array([6.0, 10.0, 20.0])
        b = np.array([2.0, 5.0, 4.0])
        result = safe_divide(a, b)
        np.testing.assert_allclose(result, [3.0, 2.0, 5.0])

    def test_broadcasting(self):
        a = np.array([10.0, 20.0])
        b = 2.0
        result = safe_divide(a, b)
        np.testing.assert_allclose(result, [5.0, 10.0])

    def test_mixed_zero_in_array(self):
        a = np.array([10.0, 20.0, 30.0])
        b = np.array([2.0, 0.0, 5.0])
        result = safe_divide(a, b, fill=999.0)
        np.testing.assert_allclose(result, [5.0, 999.0, 6.0])


# =============================================================================
# unwrap_phase_2d
# =============================================================================

class TestUnwrapPhase2d:
    def test_already_unwrapped_flat(self):
        phase = np.zeros((3, 4))
        result = unwrap_phase_2d(phase)
        np.testing.assert_allclose(result, phase)

    def test_wrapped_phase_reconstruction(self):
        # unwrapped = [-3π, -π, π, 3π] → wrapped = [π, -π, π, -π]
        unwrapped = np.array([[-3.0, -1.0, 1.0, 3.0]])
        wrapped = ((unwrapped + np.pi) % (2 * np.pi)) - np.pi
        result = unwrap_phase_2d(wrapped)
        np.testing.assert_allclose(result, unwrapped, atol=1e-10)

    def test_wrapped_phase_reconstruction_vertical(self):
        # Three rows; each row starts π/4 higher than the previous.
        # True phase jumps are π/4 < π, so wrapped differences unambiguously
        # encode the unwrapped cumulative phase at each row boundary.
        # Row 0: [0], Row 1: [π/4], Row 2: [π/2]
        unwrapped = np.array([[0.0], [np.pi / 4], [np.pi / 2]])
        wrapped = ((unwrapped + np.pi) % (2.0 * np.pi)) - np.pi
        result = unwrap_phase_2d(wrapped)
        np.testing.assert_allclose(result, unwrapped, atol=1e-10)

    def test_non_2d_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            unwrap_phase_2d(np.array([0.0, 1.0]))

    def test_multiple_wraps(self):
        # Two rows, four columns. Row1 is offset by 0.5π (π/2) from Row0.
        # True phase jump = π/2 < π, so both horizontal and vertical unwrap
        # correctly resolve all phases.
        # Wrapped: Row0=[0, 0.1π, 0.2π, 0.3π], Row1=[0.5π, 0.6π, 0.7π, 0.8π]
        unwrapped = np.array([
            [0.0, 0.1 * np.pi, 0.2 * np.pi, 0.3 * np.pi],
            [0.5 * np.pi, 0.6 * np.pi, 0.7 * np.pi, 0.8 * np.pi],
        ])
        wrapped = ((unwrapped + np.pi) % (2.0 * np.pi)) - np.pi
        result = unwrap_phase_2d(wrapped)
        np.testing.assert_allclose(result, unwrapped, atol=1e-10)

    def test_output_shape_same_as_input(self):
        phase = np.random.default_rng(0).uniform(-np.pi, np.pi, (7, 11))
        result = unwrap_phase_2d(phase)
        assert result.shape == phase.shape


# =============================================================================
# geocode_image
# =============================================================================

class TestGeocodeImage:
    def test_gdal_not_available_raises(self, monkeypatch):
        # Simulate GDAL import failure
        import builtins
        orig_import = builtins.__import__

        def _fail(name, *args, **kwargs):
            if name.startswith("osgeo"):
                raise ImportError("simulated")
            return orig_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fail)
        with pytest.raises(NotImplementedError, match="requires GDAL"):
            geocode_image(np.zeros((5, 5)), (0, 1, 0, 0, 0, -1), "EPSG:4326")


# =============================================================================
# pad_to_tile
# =============================================================================

class TestPadToTile:
    def test_already_tile_sized(self):
        data = np.zeros((10, 20))
        result = pad_to_tile(data, 5)
        assert result.shape == (10, 20)

    def test_pad_lines(self):
        data = np.zeros((7, 10))
        result = pad_to_tile(data, 5)
        assert result.shape == (10, 10)

    def test_pad_both_axes(self):
        data = np.zeros((7, 9))
        result = pad_to_tile(data, 5)
        assert result.shape == (10, 10)

    def test_data_preserved(self):
        rng = np.random.default_rng(42)
        data = rng.random((3, 4))
        result = pad_to_tile(data, 5)
        np.testing.assert_array_equal(result[:3, :4], data)
        assert result[3:, :].sum() == 0.0

    def test_tile_size_one(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = pad_to_tile(data, 1)
        assert result.shape == (2, 2)

    def test_tile_size_zero_raises(self):
        with pytest.raises(ValueError, match="tile_size"):
            pad_to_tile(np.zeros((3, 3)), 0)

    def test_dtype_preserved(self):
        data = np.array([[1, 2], [3, 4]], dtype=np.int16)
        result = pad_to_tile(data, 5)
        assert result.dtype == np.int16


# =============================================================================
# estimate_memory_usage
# =============================================================================

class TestEstimateMemoryUsage:
    def test_float64_scalar(self):
        result = estimate_memory_usage((5,), np.float64)
        assert result == 5 * 8

    def test_2d_complex64(self):
        result = estimate_memory_usage((100, 200), np.complex64)
        assert result == 100 * 200 * 8

    def test_empty_shape(self):
        result = estimate_memory_usage((), np.float32)
        assert result == 4

    def test_dtype_string(self):
        result = estimate_memory_usage((10, 10), "float32")
        assert result == 10 * 10 * 4

    def test_dtype_object(self):
        result = estimate_memory_usage((50,), float)
        assert result == 50 * 8


# =============================================================================
# pad_window_to_grid
# =============================================================================

class TestPadWindowToGrid:
    def test_window_already_in_grid(self):
        win = BurstWindow(first_line=10, num_lines=100, first_sample=20, num_samples=200)
        fl, nl, fs, ns = pad_window_to_grid(win, 500, 1000)
        assert fl == 10
        assert nl == 100
        assert fs == 20
        assert ns == 200

    def test_window_partially_outside_clamped(self):
        # window extends below 0
        win = BurstWindow(first_line=-10, num_lines=30, first_sample=5, num_samples=10)
        fl, nl, fs, ns = pad_window_to_grid(win, 500, 1000)
        assert fl == 0
        assert nl == 20

    def test_window_partially_outside_right(self):
        win = BurstWindow(first_line=480, num_lines=30, first_sample=980, num_samples=50)
        fl, nl, fs, ns = pad_window_to_grid(win, 500, 1000)
        assert fl == 480
        assert nl == 20
        assert fs == 980
        assert ns == 20

    def test_window_exceeds_grid_completely(self):
        # window partially overlaps grid from below:
        # first_line=-100, num_lines=200 → line_stop=100
        # grid_lines=500 → clamp fl=0, ls=100 → nl=100
        win = BurstWindow(first_line=-100, num_lines=200, first_sample=-100, num_samples=200)
        fl, nl, fs, ns = pad_window_to_grid(win, 500, 1000)
        assert fl == 0   # clamped from -100
        assert nl == 100  # overlap with grid is [0,100)
        assert fs == 0
        assert ns == 100

    def test_zero_size_window(self):
        # Zero-size window at a valid position — stays valid (nl=0, ns=0, fl=100, fs=100)
        win = BurstWindow(first_line=100, num_lines=0, first_sample=100, num_samples=0)
        fl, nl, fs, ns = pad_window_to_grid(win, 500, 1000)
        assert fl == 100
        assert nl == 0
        assert fs == 100
        assert ns == 0

    def test_exact_boundary(self):
        win = BurstWindow(first_line=0, num_lines=500, first_sample=0, num_samples=1000)
        fl, nl, fs, ns = pad_window_to_grid(win, 500, 1000)
        assert fl == 0
        assert nl == 500
        assert fs == 0
        assert ns == 1000