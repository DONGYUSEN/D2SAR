"""Tests for tops_deramp — TOPS azimuth carrier phase model."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure scripts/ is on the import path for the worktree root
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timezone

import numpy as np
import pytest

from scripts.tops_deramp import deramp_slc, reramp_slc, tophu_phase
from scripts.tops_model import (
    BurstIdentity,
    BurstRadarGrid,
    BurstWindow,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_burst(
    *,
    first_line: int = 0,
    num_lines: int = 1500,
    first_sample: int = 0,
    num_samples: int = 25000,
    doppler_coefficients: tuple[float, ...] = (0.0,),
    azimuth_time_interval: float = 0.002,
) -> BurstRadarGrid:
    """Minimal BurstRadarGrid for testing."""
    sensing_start = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    sensing_stop = sensing_start + timedelta(seconds=num_lines * azimuth_time_interval)
    return BurstRadarGrid(
        identity=BurstIdentity(
            swath="IW1",
            burst_index=0,
            sensing_start=sensing_start,
            sensing_stop=sensing_stop,
            polarization="VV",
            orbit_direction="ascending",
            azimuth_steering_rate=0.0,
        ),
        image_window=BurstWindow(
            first_line=0,
            num_lines=num_lines,
            first_sample=0,
            num_samples=num_samples,
        ),
        valid_window=BurstWindow(
            first_line=first_line,
            num_lines=num_lines,
            first_sample=first_sample,
            num_samples=num_samples,
        ),
        line_offset=0,
        azimuth_time_interval=azimuth_time_interval,
        range_pixel_spacing=2.3,
        starting_range=800000.0,
        radar_wavelength=0.05546576,
        doppler_coefficients=doppler_coefficients,
        azimuth_fm_rate_coefficients=(0.0,),
    )


from datetime import timedelta


# ---------------------------------------------------------------------------
# tophu_phase — unit tests
# ---------------------------------------------------------------------------

class TestTophuPhase:
    def test_zero_doppler(self):
        """Zero Doppler → zero phase everywhere."""
        burst = make_burst(doppler_coefficients=(0.0, 0.0))
        lines = np.array([0, 1, 2, 999])
        samples = np.array([0, 100, 500, 1000])
        phi = tophu_phase(burst, lines, samples)
        assert phi.shape == (4,)
        np.testing.assert_allclose(phi, 0.0, atol=1e-6)

    def test_dc_only(self):
        """Only DC term: phi = -2π * f_DC * line_index / prf."""
        prf = 500.0  # Hz
        f_DC = 100.0  # Hz
        burst = make_burst(
            azimuth_time_interval=1.0 / prf,
            doppler_coefficients=(f_DC,),
        )
        lines = np.array([0, 1, 2], dtype=np.intp).reshape(-1, 1)
        samples = np.zeros_like(lines)
        phi = tophu_phase(burst, lines, samples)
        expected = -2.0 * np.pi * f_DC * (lines.flatten() / prf)
        np.testing.assert_allclose(phi.flatten(), expected, atol=1e-6)

    def test_linear_doppler(self):
        """Linear Doppler: phi depends on both line and sample."""
        prf = 500.0
        f_DC = 0.0
        f_DR = 10.0  # Hz / sample
        burst = make_burst(
            azimuth_time_interval=1.0 / prf,
            doppler_coefficients=(f_DC, f_DR),
        )
        # single row: line_index = 10
        line_arr = np.full((1, 5), 10, dtype=np.intp)
        sample_arr = np.arange(5, dtype=np.intp).reshape(1, -1)
        phi = tophu_phase(burst, line_arr, sample_arr)
        t_l = 10.0 / prf
        expected = -2.0 * np.pi * (f_DC + f_DR * np.arange(5)) * t_l
        np.testing.assert_allclose(phi.flatten(), expected, atol=1e-6)

    def test_broadcast_2d(self):
        """lines (3,1) + samples (1,4) → (3,4) output."""
        burst = make_burst(doppler_coefficients=(500.0, 1.0))
        lines = np.array([[0], [1], [2]], dtype=np.intp)   # (3,1)
        samples = np.array([[10, 20, 30, 40]], dtype=np.intp)  # (1,4)
        phi = tophu_phase(burst, lines, samples)
        assert phi.shape == (3, 4)

    def test_scalar_inputs(self):
        """0-D (scalar) line/sample → 0-D output."""
        burst = make_burst(doppler_coefficients=(100.0,))
        # Pass as 0-D numpy arrays (scalar case)
        phi = tophu_phase(burst, np.array(5), np.array(0))
        assert phi.shape == ()  # scalar
        # also verify the value is correct
        expected = -2.0 * np.pi * 100.0 * (5.0 / burst.prf)
        np.testing.assert_allclose(float(phi), expected, atol=1e-5)

    def test_incompatible_shapes_raises(self):
        """Incompatible lines/samples shapes raise ValueError."""
        burst = make_burst()
        lines = np.array([0, 1, 2])
        samples = np.array([0, 1, 2, 3])
        with pytest.raises(ValueError, match="not broadcast-compatible"):
            tophu_phase(burst, lines, samples)

    def test_phase_dtype_float32(self):
        """Output phase is float32."""
        burst = make_burst(doppler_coefficients=(50.0, 0.5))
        lines = np.arange(10, dtype=np.intp).reshape(-1, 1)
        samples = np.arange(5, dtype=np.intp).reshape(1, -1)
        phi = tophu_phase(burst, lines, samples)
        assert phi.dtype == np.float32


# ---------------------------------------------------------------------------
# deramp_slc / reramp_slc — roundtrip tests
# ---------------------------------------------------------------------------

class TestDerampReramp:
    def test_roundtrip_identity(self):
        """deramp → reramp recovers original SLC within numerical tolerance."""
        burst = make_burst(
            num_lines=100,
            num_samples=200,
            doppler_coefficients=(200.0, 1.0),
            azimuth_time_interval=0.002,
        )
        rng = np.random.default_rng(42)
        slc = (rng.random((100, 200)) + 1j * rng.random((100, 200))).astype(np.complex64)

        deramped = deramp_slc(slc, burst)
        recovered = reramp_slc(deramped, burst)

        # Relative amplitude error should be negligible
        rel_err = np.abs(slc - recovered) / (np.abs(slc) + 1e-30)
        assert rel_err.max() < 1e-5, f"Roundtrip error too large: {rel_err.max()}"

    def test_deramp_reramp_complex_multiplication_reference(self):
        """Compare against a pure-numpy reference implementation."""
        burst = make_burst(
            num_lines=50,
            num_samples=80,
            doppler_coefficients=(150.0, 2.5),
            azimuth_time_interval=0.0025,
        )
        rng = np.random.default_rng(123)
        slc = (rng.random((50, 80)) + 1j * rng.random((50, 80))).astype(np.complex64)

        # Reference: explicit phase + exp
        nl, ns = slc.shape
        lines = np.arange(nl, dtype=np.intp).reshape(-1, 1)
        samples = np.arange(ns, dtype=np.intp).reshape(1, -1)
        phi_ref = -2.0 * np.pi * (
            burst.doppler_coefficients[0]
            + (burst.doppler_coefficients[1] if len(burst.doppler_coefficients) > 1 else 0.0)
            * samples.astype(np.float64)
        ) * (lines.astype(np.float64) / burst.prf)
        exp_phase_ref = np.exp(1j * phi_ref.astype(np.float32))

        # Implementation under test
        deramped_iut = deramp_slc(slc, burst)

        np.testing.assert_allclose(
            deramped_iut,
            slc * exp_phase_ref,
            rtol=1e-5,
            err_msg="deramp_slc complex multiplication disagrees with reference",
        )

    def test_reramp_zero_doppler(self):
        """Zero Doppler → reramp_slc returns copy of deramped unchanged."""
        burst = make_burst(
            num_lines=30,
            num_samples=40,
            doppler_coefficients=(0.0, 0.0),
        )
        rng = np.random.default_rng(7)
        slc = (rng.random((30, 40)) + 1j * rng.random((30, 40))).astype(np.complex64)

        deramped = deramp_slc(slc, burst)
        reramped = reramp_slc(deramped, burst)

        np.testing.assert_allclose(reramped, slc, rtol=1e-6)

    def test_deramp_zero_doppler_no_change(self):
        """Zero Doppler → deramp is a no-op (output equals input)."""
        burst = make_burst(
            num_lines=20,
            num_samples=30,
            doppler_coefficients=(0.0, 0.0),
        )
        rng = np.random.default_rng(99)
        slc = (rng.random((20, 30)) + 1j * rng.random((20, 30))).astype(np.complex64)

        deramped = deramp_slc(slc, burst)
        np.testing.assert_allclose(deramped, slc, rtol=1e-6)

    def test_output_dtype_complex64(self):
        """Output dtype is complex64 when input is complex64."""
        burst = make_burst(
            num_lines=10,
            num_samples=10,
            doppler_coefficients=(100.0,),
        )
        slc = np.ones((10, 10), dtype=np.complex64)
        out = deramp_slc(slc, burst)
        assert out.dtype == np.complex64

    def test_output_dtype_complex128(self):
        """Output dtype is complex128 when input is complex128."""
        burst = make_burst(
            num_lines=10,
            num_samples=10,
            doppler_coefficients=(100.0,),
        )
        slc = np.ones((10, 10), dtype=np.complex128)
        out = deramp_slc(slc, burst)
        assert out.dtype == np.complex128

    def test_inplace_output(self):
        """With out= argument the result is written into the provided array."""
        burst = make_burst(
            num_lines=20,
            num_samples=30,
            doppler_coefficients=(300.0, 0.5),
        )
        rng = np.random.default_rng(55)
        slc = (rng.random((20, 30)) + 1j * rng.random((20, 30))).astype(np.complex64)
        ref = deramp_slc(slc, burst)

        buf = np.empty_like(slc)
        result = deramp_slc(slc, burst, out=buf)

        assert result is buf
        np.testing.assert_allclose(result, ref)

    def test_reramp_inplace(self):
        """reramp_slc also respects the out= parameter."""
        burst = make_burst(
            num_lines=20,
            num_samples=30,
            doppler_coefficients=(300.0, 0.5),
        )
        rng = np.random.default_rng(55)
        slc = (rng.random((20, 30)) + 1j * rng.random((20, 30))).astype(np.complex64)
        deramped = deramp_slc(slc, burst)
        ref = reramp_slc(deramped, burst)

        buf = np.empty_like(deramped)
        result = reramp_slc(deramped, burst, out=buf)

        assert result is buf
        np.testing.assert_allclose(result, ref)

    def test_1d_slc_raises(self):
        """Non-2D slc raises ValueError."""
        burst = make_burst(num_lines=10, num_samples=10)
        slc_1d = np.ones(10, dtype=np.complex64)
        with pytest.raises(ValueError, match="must be 2-D"):
            deramp_slc(slc_1d, burst)

    def test_3d_slc_raises(self):
        """3D slc raises ValueError."""
        burst = make_burst(num_lines=10, num_samples=10)
        slc_3d = np.ones((5, 10, 10), dtype=np.complex64)
        with pytest.raises(ValueError, match="must be 2-D"):
            deramp_slc(slc_3d, burst)

    def test_zero_prf_raises(self):
        """burst.prf <= 0 raises ZeroDivisionError when accessing prf in deramp_slc."""
        burst = make_burst(num_lines=10, num_samples=10)
        # Patch prf to 0 (frozen dataclass — use object.__setattr__)
        object.__setattr__(burst, "azimuth_time_interval", 0.0)
        slc = np.ones((10, 10), dtype=np.complex64)
        # burst.prf raises ZeroDivisionError on 1.0/0.0
        with pytest.raises(ZeroDivisionError):
            deramp_slc(slc, burst)

    def test_out_wrong_shape_raises(self):
        """out= with wrong shape raises ValueError."""
        burst = make_burst(num_lines=10, num_samples=10)
        slc = np.ones((10, 10), dtype=np.complex64)
        bad_out = np.empty((5, 5), dtype=np.complex64)
        with pytest.raises(ValueError, match="out.shape"):
            deramp_slc(slc, burst, out=bad_out)

    def test_out_real_raises(self):
        """out= with real dtype raises ValueError."""
        burst = make_burst(num_lines=10, num_samples=10)
        slc = np.ones((10, 10), dtype=np.complex64)
        bad_out = np.empty((10, 10), dtype=np.float32)
        with pytest.raises(ValueError, match="out.dtype must be complex"):
            deramp_slc(slc, burst, out=bad_out)

    def test_phase_2d_shape_from_1d_broadcast(self):
        """1-D lines (N,1) + 1-D samples (1,M) → (N,M) output."""
        burst = make_burst(
            num_lines=5,
            num_samples=4,
            doppler_coefficients=(100.0, 0.0),
        )
        lines = np.arange(5, dtype=np.intp).reshape(-1, 1)   # (5, 1)
        samples = np.arange(4, dtype=np.intp).reshape(1, -1)  # (1, 4)
        phi = tophu_phase(burst, lines, samples)
        assert phi.shape == (5, 4)

    def test_phase_increases_with_line(self):
        """Phase should increase linearly with line index for positive DC."""
        f_DC = 100.0
        burst = make_burst(
            azimuth_time_interval=0.002,
            doppler_coefficients=(f_DC, 0.0),
            num_lines=100,
            num_samples=1,
        )
        lines = np.arange(100, dtype=np.intp).reshape(-1, 1)
        samples = np.zeros((1, 1), dtype=np.intp)
        phi = tophu_phase(burst, lines, samples)
        t = np.arange(100) / burst.prf
        expected = -2.0 * np.pi * f_DC * t
        np.testing.assert_allclose(phi.flatten(), expected, atol=1e-5)
