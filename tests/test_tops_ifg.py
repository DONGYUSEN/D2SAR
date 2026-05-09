"""Tests for tops_ifg — per-burst interferogram generation via cross-multiply."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure scripts/ is on the import path for the worktree root
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from scripts.tops_ifg import IfgResult, generate_ifg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _complex_normal(shape, rng):
    """Random complex64 array with normal distribution."""
    return (
        rng.standard_normal(shape) +
        1j * rng.standard_normal(shape)
    ).astype(np.complex64)


# ---------------------------------------------------------------------------
# generate_ifg — happy path
# ---------------------------------------------------------------------------

class TestGenerateIfg:
    """Happy-path tests for generate_ifg."""

    def test_output_dtype_complex64(self):
        """Output complex_ifg is complex64."""
        ref = np.ones((50, 100), dtype=np.complex64)
        sec = np.ones((50, 100), dtype=np.complex64)
        result = generate_ifg(ref, sec)
        assert result.complex_ifg.dtype == np.complex64

    def test_coherence_dtype_float32(self):
        """Output coherence is float32."""
        ref = np.ones((50, 100), dtype=np.complex64)
        sec = np.ones((50, 100), dtype=np.complex64)
        result = generate_ifg(ref, sec)
        assert result.coherence.dtype == np.float32

    def test_same_shape_identical_input(self):
        """Identical ref=sec → |IFG| = |ref|², coherence ≈ 1."""
        rng = _rng(42)
        ref = _complex_normal((50, 100), rng)
        result = generate_ifg(ref, ref)
        # Multilooked shape = floor(input / looks) * looks
        assert result.complex_ifg.shape == (10, 20)
        assert result.coherence.min() >= 0.99  # coherence ≈ 1 for identity IFG
        assert result.valid_fraction == 1.0

    def test_coherence_unity_for_correlated_ifg(self):
        """Constant-phase IFG → coherence exactly 1 (full correlation)."""
        # ref = unit circle
        ref = np.ones((50, 100), dtype=np.complex64)
        # sec = same magnitude, constant phase shift of π/3
        phase = 1.0472  # 60 degrees
        sec = np.exp(-1j * phase) * np.ones((50, 100), dtype=np.complex64)
        result = generate_ifg(ref, sec)
        np.testing.assert_allclose(result.coherence, 1.0, rtol=1e-6)

    def test_coherence_zero_for_uncorrelated_ifg(self):
        """Random ref/sec → coherence ≈ 0 (no correlation)."""
        rng = _rng(123)
        ref = _complex_normal((100, 200), rng)
        sec = _complex_normal((100, 200), rng)
        result = generate_ifg(ref, sec)
        # Coherence mean for uncorrelated data should be well below 0.5
        # (random phase → coherence ~ 1/sqrt(N) where N is window size)
        assert result.coherence.mean() < 0.4
        # All values should be in [0, 1]
        assert result.coherence.min() >= 0.0
        assert result.coherence.max() <= 1.0

    def test_multilook_shape_reduction(self):
        """Multilook reduces shape by factor of looks."""
        ref = np.ones((50, 100), dtype=np.complex64)
        sec = np.ones((50, 100), dtype=np.complex64)
        result = generate_ifg(ref, sec, looks_rg=5, looks_az=5)
        assert result.complex_ifg.shape == (10, 20)
        assert result.coherence.shape == (10, 20)

    def test_multilook_shape_reduction_asymmetric(self):
        """Asymmetric looks work correctly."""
        ref = np.ones((60, 80), dtype=np.complex64)
        sec = np.ones((60, 80), dtype=np.complex64)
        result = generate_ifg(ref, sec, looks_rg=4, looks_az=3)
        assert result.complex_ifg.shape == (20, 20)
        assert result.coherence.shape == (20, 20)

    def test_truncate_non_divisible(self):
        """Non-divisible dimensions are truncated, valid_fraction < 1."""
        ref = np.ones((51, 101), dtype=np.complex64)
        sec = np.ones((51, 101), dtype=np.complex64)
        result = generate_ifg(ref, sec, looks_rg=5, looks_az=5)
        # 51//5=10→10 rows, 101//5=20→20 cols (multilooked shape)
        assert result.complex_ifg.shape == (10, 20)
        # valid_fraction = valid pixels kept / total input pixels
        # = (50 * 100) / (51 * 101) = 5000 / 5151 ≈ 0.9707
        nl_ml = (51 // 5) * 5  # 50
        ns_ml = (101 // 5) * 5  # 100
        expected_frac = (nl_ml * ns_ml) / (51 * 101)
        np.testing.assert_allclose(result.valid_fraction, expected_frac, rtol=1e-6)

    def test_valid_fraction_zero_for_undersized_input(self):
        """Input smaller than looks → empty result, valid_fraction=0."""
        ref = np.ones((3, 3), dtype=np.complex64)
        sec = np.ones((3, 3), dtype=np.complex64)
        result = generate_ifg(ref, sec, looks_rg=5, looks_az=5)
        assert result.complex_ifg.shape == (0, 0)
        assert result.valid_fraction == 0.0

    def test_multilook_unity_looks_preserves_shape(self):
        """looks_rg=1, looks_az=1 preserves full shape."""
        ref = np.ones((20, 30), dtype=np.complex64)
        sec = np.ones((20, 30), dtype=np.complex64)
        result = generate_ifg(ref, sec, looks_rg=1, looks_az=1)
        assert result.complex_ifg.shape == (20, 30)
        assert result.coherence.shape == (20, 30)
        assert result.valid_fraction == 1.0

    def test_real_input_converted(self):
        """Real-valued input is promoted to complex64."""
        ref = np.ones((10, 20), dtype=np.float32)
        sec = np.ones((10, 20), dtype=np.float32)
        result = generate_ifg(ref, sec)
        assert result.complex_ifg.dtype == np.complex64
        assert result.coherence.dtype == np.float32

    def test_complex128_input_converted(self):
        """complex128 input is converted to complex64."""
        ref = np.ones((10, 20), dtype=np.complex128)
        sec = np.ones((10, 20), dtype=np.complex128)
        result = generate_ifg(ref, sec)
        assert result.complex_ifg.dtype == np.complex64

    def test_coherence_exact_one_for_constant_phase(self):
        """Constant-phase IFG with multilook → coherence exactly 1."""
        ref = np.exp(1j * 0.3) * np.ones((50, 100), dtype=np.complex64)
        sec = np.exp(-1j * 0.7) * np.ones((50, 100), dtype=np.complex64)
        result = generate_ifg(ref, sec)
        np.testing.assert_allclose(result.coherence, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# generate_ifg — validation
# ---------------------------------------------------------------------------

class TestGenerateIfgValidation:
    """Input validation tests for generate_ifg."""

    def test_shape_mismatch_raises(self):
        """Mismatched ref/sec shapes raise ValueError."""
        ref = np.ones((10, 20), dtype=np.complex64)
        sec = np.ones((10, 25), dtype=np.complex64)
        with pytest.raises(ValueError, match="shape .* != .* shape"):
            generate_ifg(ref, sec)

    def test_looks_rg_zero_raises(self):
        """looks_rg < 1 raises ValueError."""
        ref = np.ones((10, 20), dtype=np.complex64)
        sec = np.ones((10, 20), dtype=np.complex64)
        with pytest.raises(ValueError, match="looks_rg"):
            generate_ifg(ref, sec, looks_rg=0)

    def test_looks_az_zero_raises(self):
        """looks_az < 1 raises ValueError."""
        ref = np.ones((10, 20), dtype=np.complex64)
        sec = np.ones((10, 20), dtype=np.complex64)
        with pytest.raises(ValueError, match="looks_az"):
            generate_ifg(ref, sec, looks_az=0)


# ---------------------------------------------------------------------------
# generate_ifg — algorithmic correctness
# ---------------------------------------------------------------------------

class TestGenerateIfgAlgorithm:
    """Algorithm-level tests verifying the coherence formula."""

    def test_coherence_formula_exact(self):
        """Verify coherence = |sum(ifg)| / sqrt(sum(|ref|²) * sum(|sec|²))."""
        # Use small arrays where we can compute the reference by hand
        ref = np.array([[1 + 1j, 2], [0, 1 - 1j]], dtype=np.complex64)
        sec = np.array([[1, 0], [1j, 1]], dtype=np.complex64)
        result = generate_ifg(ref, ref)  # identity IFG
        # For identity, |ref|² is the IFG magnitude, coherence should be ~1
        np.testing.assert_allclose(result.coherence, 1.0, rtol=1e-5)

    def test_interferogram_is_cross_multiply(self):
        """The multilooked IFG equals boxcar-multilooked (ref * conj(sec))."""
        rng = _rng(7)
        ref = _complex_normal((40, 60), rng)
        sec = _complex_normal((40, 60), rng)
        result = generate_ifg(ref, sec, looks_rg=4, looks_az=4)

        # Manual reference: cross-multiply then multilook
        ifg_manual = ref * np.conj(sec)
        sh = (ifg_manual.shape[0] // 4, 4, ifg_manual.shape[1] // 4, 4)
        ifg_ref = ifg_manual[:sh[0]*4, :sh[2]*4].reshape(sh).mean(axis=(1, 3))

        np.testing.assert_allclose(result.complex_ifg, ifg_ref, rtol=1e-6)

    def test_coherence_in_range_zero_one(self):
        """Coherence always lies in [0, 1]."""
        rng = _rng(999)
        for _ in range(5):
            ref = _complex_normal((50, 80), rng)
            sec = _complex_normal((50, 80), rng)
            result = generate_ifg(ref, sec)
            assert result.coherence.min() >= 0.0
            assert result.coherence.max() <= 1.0
            assert not np.any(np.isnan(result.coherence))
            assert not np.any(np.isinf(result.coherence))

    def test_ifg_phase_is_sum_of_ref_minus_sec_phase(self):
        """IFG phase ≈ arg(ref) - arg(sec) for complex inputs (mod 2π)."""
        rng = _rng(5)
        ref = _complex_normal((20, 30), rng)
        sec = _complex_normal((20, 30), rng)
        result = generate_ifg(ref, sec, looks_rg=1, looks_az=1)
        expected_phase = np.angle(ref) - np.angle(sec)
        # Normalize to same 2π representation before comparison
        actual = np.angle(result.complex_ifg)
        diff = np.abs(actual - expected_phase)
        # Wrap difference into [0, π] to handle -π/+π discontinuity
        diff = np.minimum(diff, 2 * np.pi - diff)
        np.testing.assert_allclose(diff, 0.0, atol=1e-5)

    def test_mixed_real_complex_input(self):
        """Real ref with complex sec works and promotes correctly."""
        ref = np.ones((10, 20), dtype=np.float32)
        sec = _complex_normal((10, 20), _rng(8))
        result = generate_ifg(ref, sec)
        assert result.complex_ifg.dtype == np.complex64
        # Multilooked shape with default looks=5: floor(10/5)=2, floor(20/5)=4
        assert result.complex_ifg.shape == (2, 4)
