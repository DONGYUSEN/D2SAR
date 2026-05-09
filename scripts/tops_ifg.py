"""tops_ifg — Per-burst interferogram generation via cross-multiply.

Algorithm:
1. Cross-multiply: ifg = ref * conj(sec)
2. Multilook ifg, |ref|², |sec|² via boxcar
3. Coherence: γ = |sum(ifg)| / sqrt(sum(|ref|²) * sum(|sec|²))
4. Output same shape as multilooked input

Dependency: tops_model only (no strip/tops_insar imports).
"""

from __future__ import annotations

__all__ = [
    "IfgResult",
    "generate_ifg",
]

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IfgResult:
    """Result of per-burst interferogram generation.

    Attributes
    ----------
    complex_ifg : np.ndarray
        Complex interferogram (complex64), multilooked.
    coherence : np.ndarray
        Complex coherence magnitude (float32), same shape as complex_ifg.
    valid_fraction : float
        Fraction of pixels within integer multilook boundaries (0–1).
    """
    complex_ifg: np.ndarray      # complex64
    coherence: np.ndarray         # float32
    valid_fraction: float         # 0.0–1.0


def generate_ifg(
    ref: np.ndarray,
    sec: np.ndarray,
    *,
    looks_rg: int = 5,
    looks_az: int = 5,
) -> IfgResult:
    """Generate a multilooked interferogram and coherence from two SLC bursts.

    Parameters
    ----------
    ref : np.ndarray
        Reference SLC (complex64 or real). 2-D.
    sec : np.ndarray
        Secondary SLC (complex64 or real). 2-D, same shape as ``ref``.
    looks_rg : int, default 5
        Number of range looks (boxcar window width).
    looks_az : int, default 5
        Number of azimuth looks (boxcar window height).

    Returns
    -------
    IfgResult
        - complex_ifg : multilooked complex interferogram (complex64)
        - coherence   : coherence γ (float32)
        - valid_fraction : fraction of pixels retained after truncation

    Raises
    ------
    ValueError
        If ``ref`` and ``sec`` shapes differ, or if looks < 1.

    Algorithm
    ---------
    1. Cross-multiply: ifg = ref * conj(sec)           (element-wise)
    2. Boxcar multilook for ifg, |ref|², |sec|²
    3. Coherence:
       γ[i,j] = |sum(ifg_ij)| / sqrt(sum(|ref_ij|²) * sum(|sec_ij|²))
    4. Shape truncates to floor(n / looks) * looks

    Notes
    -----
    - Output dtype is always complex64 (inputs promoted if needed).
    - Uses pure NumPy (no ISCE3 dependency).
    """
    # --- Input validation ----------------------------------------------------
    if ref.shape != sec.shape:
        raise ValueError(
            f"ref shape {ref.shape} != sec shape {sec.shape}"
        )
    if looks_rg < 1 or looks_az < 1:
        raise ValueError(
            f"looks_rg={looks_rg} and looks_az={looks_az} must be >= 1"
        )

    nl, ns = ref.shape

    # --- Promote to complex64 ------------------------------------------------
    ref_c = ref.astype(np.complex64)
    sec_c = sec.astype(np.complex64)

    # --- Cross-multiply -------------------------------------------------------
    ifg = ref_c * np.conj(sec_c)  # (nl, ns), complex64

    # --- Multilook dimensions (truncate to integer multiple) ----------------
    nl_ml = (nl // looks_az) * looks_az
    ns_ml = (ns // looks_rg) * looks_rg
    valid_fraction = (nl_ml * ns_ml) / (nl * ns) if (nl * ns) > 0 else 0.0

    if nl_ml < looks_az or ns_ml < looks_rg:
        # No complete multilook window; return empty result
        empty_ifg = np.empty((0, 0), dtype=np.complex64)
        return IfgResult(
            complex_ifg=empty_ifg,
            coherence=np.empty((0, 0), dtype=np.float32),
            valid_fraction=valid_fraction,
        )

    # Slice to truncated region
    ifg_trim = ifg[:nl_ml, :ns_ml]
    ref_trim = ref_c[:nl_ml, :ns_ml]
    sec_trim = sec_c[:nl_ml, :ns_ml]

    # --- Boxcar multilook helper --------------------------------------------
    def _boxcar(arr: np.ndarray) -> np.ndarray:
        """2-D boxcar multilook: mean over looks_az × looks_rg windows."""
        sh = arr.shape
        az, rg = looks_az, looks_rg
        sh2 = (sh[0] // az, az, sh[1] // rg, rg)
        return arr.reshape(sh2).mean(axis=(1, 3))

    # Multilook each component
    ifg_ml = _boxcar(ifg_trim)                        # complex64
    ref_sq_ml = _boxcar(np.abs(ref_trim) ** 2)        # float32
    sec_sq_ml = _boxcar(np.abs(sec_trim) ** 2)        # float32

    # --- Coherence -----------------------------------------------------------
    # γ = |sum(ifg)| / sqrt(sum(|ref|²) * sum(|sec|²))
    numerator = np.abs(ifg_ml)
    denominator = np.sqrt(ref_sq_ml * sec_sq_ml)

    # Avoid division by zero
    coherence = np.where(denominator > 0.0, numerator / denominator, 0.0)
    coherence = coherence.astype(np.float32)

    return IfgResult(
        complex_ifg=ifg_ml,
        coherence=coherence,
        valid_fraction=float(valid_fraction),
    )
