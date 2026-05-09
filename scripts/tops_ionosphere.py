"""tops_ionosphere — Optional split-band ionospheric correction for Sentinel-1 TOPS.

Algorithm (split-band):
1. Split-band IFG: use ``tops_ifg.generate_ifg`` to produce high/low frequency IFGs.
2. Boxcar multilook both IFGs.
3. Phase ratio: phi_disp = (f_high/f_low) * phi_high - phi_low.
4. Dispersive phase: ionospheric_phase = -phi_disp / (f_high - f_low).
5. Median filtering to suppress noise.

Dependency: tops_ifg, tops_model only (no strip/tops_insar imports).
"""

from __future__ import annotations

__all__ = [
    "estimate_ionospheric_phase",
    "remove_ionospheric_phase",
    "write_ionosphere_summary",
]

from pathlib import Path

import numpy as np

from scripts.tops_ifg import generate_ifg
from scripts.tops_model import BurstRadarGrid


# ---------------------------------------------------------------------------
# Sentinel-1 C-band frequencies (Hz)
# ---------------------------------------------------------------------------
# Primary operating band: ~5.405 GHz
# Alternate band: ~5.335 GHz (from annotation XML, or f_high - 70 MHz fallback)
S1_F_HIGH_HZ = 5.405e9    # Sentinel-1 primary band centre (Hz)
S1_F_LOW_HZ  = 5.335e9     # Sentinel-1 alternate band centre (Hz)
S1_F_SEP_HZ  = S1_F_HIGH_HZ - S1_F_LOW_HZ   # ≈ 70 MHz

# ---------------------------------------------------------------------------
# Phase estimation
# ---------------------------------------------------------------------------

def estimate_ionospheric_phase(
    slc_high: np.ndarray,
    slc_low: np.ndarray,
    burst: BurstRadarGrid,
    looks_rg: int,
    looks_az: int,
) -> np.ndarray:
    """Estimate ionospheric dispersive phase via split-band IFG.

    Parameters
    ----------
    slc_high : np.ndarray
        SLC from the high-frequency sub-band (complex64 or complex128).
    slc_low : np.ndarray
        SLC from the low-frequency sub-band (same shape as ``slc_high``).
    burst : BurstRadarGrid
        Burst radar-grid parameters (for future metadata use).
    looks_rg : int
        Number of boxcar range looks.
    looks_az : int
        Number of boxcar azimuth looks.

    Returns
    -------
    np.ndarray
        2-D ionospheric phase in radians (float32), same shape as the
        multilooked IFG output.

    Raises
    ------
    ValueError
        - ``slc_high`` and ``slc_low`` shapes differ.
        - Fewer than 128 valid (finite) pixels remain after filtering.
        - Any pixel exceeds ±5 radians (indicates unwrapping or SNR issue).

    Algorithm
    ---------
    1. Validate shapes match.
    2. Generate high- and low-frequency IFGs via ``generate_ifg``.
    3. Boxcar multilook both IFGs (handled inside ``generate_ifg``).
    4. Extract wrapped phase from each IFG: ``phi = angle(ifg)``.
    5. Compute dispersive phase ratio:
       ``phi_disp = (f_high/f_low) * phi_high - phi_low``.
    6. Normalise: ``ionospheric_phase = -phi_disp / (f_high - f_low)``.
    7. Median-filter to suppress noise.
    8. Validate: ≥128 valid pixels and |phase| ≤ 5 rad.
    """
    # --- Input validation ---
    if slc_high.shape != slc_low.shape:
        raise ValueError(
            f"slc_high shape {slc_high.shape} != slc_low shape {slc_low.shape}"
        )
    if looks_rg < 1 or looks_az < 1:
        raise ValueError(
            f"looks_rg={looks_rg} and looks_az={looks_az} must be >= 1"
        )

    f_high = S1_F_HIGH_HZ
    f_low  = S1_F_LOW_HZ
    freq_sep = f_high - f_low  # ≈ 70 MHz

    # --- Generate split-band IFGs ---
    high_result = generate_ifg(slc_high, slc_low, looks_rg=looks_rg, looks_az=looks_az)
    low_result  = generate_ifg(slc_low, slc_high, looks_rg=looks_rg, looks_az=looks_az)

    # Extract multilooked complex IFGs
    high_ifg = high_result.complex_ifg
    low_ifg  = low_result.complex_ifg

    # --- Phase extraction ---
    phi_high = np.angle(high_ifg)   # radians, same shape as high_ifg
    phi_low  = np.angle(low_ifg)

    # --- Dispersive phase ratio ---
    phi_disp = (f_high / f_low) * phi_high - phi_low

    # --- Normalise to ionospheric phase (radians) ---
    ionospheric_phase = -phi_disp / freq_sep

    # --- NaN-aware median filtering (suppress phase noise) ---
    # Use NaN-aware median so input NaN values are replaced, not passed through.
    phase_f = ionospheric_phase.astype(np.float64)

    try:
        from scipy.ndimage import generic_filter
    except ImportError:
        generic_filter = None

    if generic_filter is not None:
        def _nanmedian_3x3(values):
            # generic_filter passes a flat 1-D window; use nanmedian
            finite = [v for v in values if np.isfinite(v)]
            if len(finite) == 0:
                return 0.0
            return float(np.median(finite))

        ionospheric_phase = generic_filter(
            phase_f,
            _nanmedian_3x3,
            size=3,
            mode="nearest",
        ).astype(np.float32)
    else:
        # Fallback: pure-numpy NaN-aware 3×3 median
        _filt = np.zeros_like(phase_f)
        pad = np.pad(phase_f, pad_width=1, mode="edge")
        for i in range(ionospheric_phase.shape[0]):
            for j in range(ionospheric_phase.shape[1]):
                window = pad[i : i + 3, j : j + 3].ravel()
                finite_vals = window[np.isfinite(window)]
                _filt[i, j] = float(np.median(finite_vals)) if finite_vals.size > 0 else 0.0
        ionospheric_phase = _filt.astype(np.float32)

    # --- NaN guard: replace any remaining NaN with 0 before validation ---
    ionospheric_phase = np.where(np.isfinite(ionospheric_phase), ionospheric_phase, 0.0)

    # --- Validation ---
    valid = np.isfinite(ionospheric_phase)
    if valid.sum() < 128:
        raise ValueError(
            f"Fewer than 128 valid pixels ({valid.sum():d}) for ionospheric phase"
        )

    abs_phase = np.abs(ionospheric_phase)
    bad = valid & (abs_phase > 5.0)
    if bad.any():
        raise ValueError(
            f"Ionospheric phase magnitude exceeds ±5 rad at {bad.sum()} pixels"
        )

    return ionospheric_phase


# ---------------------------------------------------------------------------
# Phase removal
# ---------------------------------------------------------------------------

def remove_ionospheric_phase(
    merged_ifg: np.ndarray,
    iono_phase: np.ndarray,
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Remove ionospheric dispersive phase from a merged interferogram.

    The ionospheric phase is injected by:
        ifg_corrected = merged_ifg * exp(-j * iono_phase)

    Parameters
    ----------
    merged_ifg : np.ndarray
        Complex merged interferogram (complex64 or complex128).
    iono_phase : np.ndarray
        Ionospheric phase in radians (float32), same shape as ``merged_ifg``.
    out : np.ndarray | None, optional
        Pre-allocated output array.  If ``None`` a new complex64 array is
        allocated and returned.

    Returns
    -------
    np.ndarray
        Ionospheric-phase-corrected complex interferogram.

    Raises
    ------
    ValueError
        If ``merged_ifg`` and ``iono_phase`` shapes differ.
    """
    if merged_ifg.shape != iono_phase.shape:
        raise ValueError(
            f"merged_ifg shape {merged_ifg.shape} != iono_phase shape {iono_phase.shape}"
        )

    if out is None:
        result = np.empty_like(merged_ifg)
    else:
        if out.shape != merged_ifg.shape:
            raise ValueError(
                f"out shape {out.shape} != merged_ifg shape {merged_ifg.shape}"
            )
        result = out

    # Convert phase to complex carrier: exp(-j * phase)
    phi = iono_phase.astype(np.float64)
    carrier = np.exp(-1j * phi).astype(merged_ifg.dtype)
    np.multiply(merged_ifg, carrier, out=result)
    return result


# ---------------------------------------------------------------------------
# Diagnostics writer
# ---------------------------------------------------------------------------

def write_ionosphere_summary(
    iono_phase: np.ndarray,
    path: Path,
) -> None:
    """Write a JSON diagnostic summary of the ionospheric phase.

    Parameters
    ----------
    iono_phase : np.ndarray
        2-D ionospheric phase array (radians).
    path : Path
        Destination JSON path.
    """
    iono_phase = np.asarray(iono_phase, dtype=np.float32)
    valid = np.isfinite(iono_phase)
    data = iono_phase[valid]

    import json

    payload = {
        "shape": list(iono_phase.shape),
        "mean_radians": float(np.mean(data)) if data.size > 0 else None,
        "std_radians": float(np.std(data)) if data.size > 0 else None,
        "min_radians": float(np.min(data)) if data.size > 0 else None,
        "max_radians": float(np.max(data)) if data.size > 0 else None,
        "nan_count": int(np.sum(~valid)),
        "valid_pixel_count": int(valid.sum()),
        "f_high_hz": float(S1_F_HIGH_HZ),
        "f_low_hz": float(S1_F_LOW_HZ),
        "frequency_separation_hz": float(S1_F_SEP_HZ),
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
