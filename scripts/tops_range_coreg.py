"""tops_range_coreg — Range coregistration residual estimation from overlap interferogram.

This module extracts range-direction misregistration offsets from a burst-overlap
interferogram using phase-gradient analysis.  No dependencies on strip_insar,
strip_insar2, or tops_insar.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TextIO

import numpy as np

from scripts.tops_model import RangeCoregEstimate

# Physical constants
SPEED_OF_LIGHT = 299792458.0  # m/s  (exact by SI definition)

__all__ = [
    "estimate_range_coreg",
    "inject_range_coreg",
    "write_range_coreg_summary",
    "SPEED_OF_LIGHT",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def estimate_range_coreg(
    overlap_ifg: np.ndarray,
    overlap_coherence: np.ndarray,
    *,
    coherence_threshold: float = 0.3,
    max_expected_offset: float = 1.0,
    looks_rg: int = 5,
    looks_az: int = 5,
    radar_wavelength: float = 0.05546576,
    range_pixel_spacing: float = 2.3295622,
) -> tuple[np.ndarray, np.ndarray, RangeCoregEstimate]:
    """Estimate range (and azimuth) coregistration offsets from an overlap interferogram.

    Parameters
    ----------
    overlap_ifg : np.ndarray
        Complex overlap interferogram (2-D, dtype complex64 or complex128).
        Shape: (azimuth_lines, range_samples).
    overlap_coherence : np.ndarray
        Per-pixel coherence estimate for the overlap region (same shape as overlap_ifg).
    coherence_threshold : float, default 0.3
        Minimum coherence for a pixel to be considered valid.
    max_expected_offset : float, default 1.0
        Maximum expected offset in pixels.  Pixels whose offset magnitude exceeds
        this value are rejected as outliers.
    looks_rg : int, default 5
        Number of range looks for the boxcar multilook before phase-gradient estimation.
    looks_az : int, default 5
        Number of azimuth looks for the boxcar multilook.
    radar_wavelength : float, default 0.05546576
        Radar wavelength in metres (C-band ≈ 5.55 cm).
    range_pixel_spacing : float, default 2.3295622
        Range pixel spacing in metres.

    Returns
    -------
    range_offset : np.ndarray
        2-D float32 array of per-pixel range-direction offsets in pixels.
        Shape matches ``overlap_ifg`` (post-multilook, i.e. reduced resolution).
    azimuth_offset : np.ndarray
        2-D float32 array of per-pixel azimuth-direction offsets in pixels.
        Same shape as range_offset.
    estimate : RangeCoregEstimate
        Dataclass with robust statistics over the valid pixel population.

    Raises
    ------
    ValueError
        If ``overlap_ifg`` has fewer than 2 dimensions, if ``overlap_coherence``
        does not match its shape, if no valid pixels remain after coherence
        masking, or if all offsets are rejected as outliers.
    """
    # ---- input validation ----------------------------------------------------
    if overlap_ifg.ndim != 2:
        raise ValueError(
            f"overlap_ifg must be 2-D, got {overlap_ifg.ndim}-D"
        )
    if overlap_coherence.shape != overlap_ifg.shape:
        raise ValueError(
            f"overlap_coherence shape {overlap_coherence.shape} "
            f"does not match overlap_ifg shape {overlap_ifg.shape}"
        )
    if looks_rg < 1 or looks_az < 1:
        raise ValueError(f"looks_rg={looks_rg} and looks_az={looks_az} must be ≥ 1")

    # ---- physical constants for offset conversion ---------------------------
    f_center = SPEED_OF_LIGHT / radar_wavelength  # Hz

    # ---- step 1: coherence mask -----------------------------------------------
    valid_mask = (overlap_coherence >= coherence_threshold) & np.isfinite(overlap_ifg)

    # ---- step 2: multilook ------------------------------------------------------
    # Multilook is applied only to compute phase gradient on a smoother field.
    # We compute it on the complex IFG so the gradient sees the correct phase.
    ifg_ml = _boxcar_multilook_complex(overlap_ifg, looks_az, looks_rg)
    mask_ml = _boxcar_multilook_bool(valid_mask, looks_az, looks_rg)

    # ---- step 3: phase gradient (rad/pixel) ------------------------------------
    # range direction: compare adjacent columns
    dphi_dr = _phase_gradient_axis(ifg_ml, axis=1)   # rad per range pixel
    # azimuth direction: compare adjacent rows
    dphi_da = _phase_gradient_axis(ifg_ml, axis=0)   # rad per azimuth pixel

    # ---- step 4: offset conversion ---------------------------------------------
    # offset (pixels) = phase_gradient / (4π * f_center) * (pixel_spacing / 1)
    #                  = phase_gradient / (4π) * wavelength / pixel_spacing
    # Because f_center = c/λ, so 4π*f_center = 4π*c/λ
    # Then: offset = dphi / (4π*c/λ) * pixel_spacing
    #             = dphi * λ / (4π * c) * pixel_spacing
    #             = dphi / (4π) * wavelength / pixel_spacing
    scale = radar_wavelength / (4.0 * np.pi * range_pixel_spacing)

    range_offset_ml = dphi_dr * scale
    azimuth_offset_ml = dphi_da * scale

    # ---- step 5: outlier rejection --------------------------------------------
    # Reject pixels whose offset magnitude exceeds max_expected_offset
    range_oob = np.abs(range_offset_ml) > max_expected_offset
    az_oob = np.abs(azimuth_offset_ml) > max_expected_offset

    # Valid for statistics: in-bounds AND coherent (mask_ml True)
    stat_valid = mask_ml & ~range_oob & ~az_oob

    n_total = int(np.sum(mask_ml))
    if n_total == 0:
        raise ValueError(
            "No valid pixels after coherence masking (threshold="
            f"{coherence_threshold}). Check overlap_ifg and overlap_coherence."
        )

    n_stat = int(np.sum(stat_valid))
    if n_stat == 0:
        raise ValueError(
            f"All {n_total} coherent pixels were rejected as outliers "
            f"(max_expected_offset={max_expected_offset} px). "
            "Increase max_expected_offset or lower coherence_threshold."
        )

    # ---- step 6: robust statistics --------------------------------------------
    flat_r = range_offset_ml[stat_valid].astype(np.float64)
    flat_a = azimuth_offset_ml[stat_valid].astype(np.float64)

    median_range = float(np.nanmedian(flat_r))
    std_range = float(np.nanstd(flat_r))
    median_az = float(np.nanmedian(flat_a))
    std_az = float(np.nanstd(flat_a))

    usable_fraction = float(n_stat) / float(n_total)

    estimate = RangeCoregEstimate(
        median_range_offset=median_range,
        std_range_offset=std_range,
        median_azimuth_offset=median_az,
        std_azimuth_offset=std_az,
        sample_count=n_stat,
        usable_fraction=usable_fraction,
    )

    return (
        range_offset_ml.astype(np.float32),
        azimuth_offset_ml.astype(np.float32),
        estimate,
    )


def inject_range_coreg(
    fine_range_off: np.ndarray,
    correction_px: float,
) -> np.ndarray:
    """Add a range-coregistration correction to a fine range-offset raster.

    Parameters
    ----------
    fine_range_off : np.ndarray
        Fine range offset raster (2-D float32 or float64), in pixels.
    correction_px : float
        Range correction to add, in pixels.  Positive values shift the
        secondary further in range relative to the reference.

    Returns
    -------
    np.ndarray
        Corrected offset raster (float32), same shape and dtype convention
        as ``fine_range_off``.
    """
    if fine_range_off.ndim != 2:
        raise ValueError(f"fine_range_off must be 2-D, got {fine_range_off.ndim}-D")
    corrected = fine_range_off.astype(np.float32) + np.float32(correction_px)
    return corrected


def write_range_coreg_summary(
    dest: str | Path | TextIO,
    estimate: RangeCoregEstimate,
    *,
    radar_wavelength: float = 0.05546576,
) -> None:
    """Write ``range_coreg_summary.json`` with statistics and metadata.

    Parameters
    ----------
    dest : str, Path, or TextIO
        Output path or file handle.  ``.json`` extension is added automatically
        if ``dest`` is a path without one.
    estimate : RangeCoregEstimate
        Coregistration estimate returned by ``estimate_range_coreg``.
    radar_wavelength : float
        Radar wavelength in metres (written into the JSON for traceability).
    """
    path: Path | None = None
    if isinstance(dest, (str, Path)):
        path = Path(dest)
        if path.suffix.lower() != ".json":
            path = path.with_suffix(".json")

    payload = {
        "median_range_offset": estimate.median_range_offset,
        "std_range_offset": estimate.std_range_offset,
        "median_azimuth_offset": estimate.median_azimuth_offset,
        "std_azimuth_offset": estimate.std_azimuth_offset,
        "sample_count": estimate.sample_count,
        "usable_fraction": estimate.usable_fraction,
        "wavelength": radar_wavelength,
    }

    if path is not None:
        path.write_text(json.dumps(payload, indent=2))
    else:
        # dest is TextIO
        json.dump(payload, dest, indent=2)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _boxcar_multilook_complex(
    arr: np.ndarray,
    az_looks: int,
    rg_looks: int,
) -> np.ndarray:
    """Boxcar-multilook a 2-D complex array (averaging real and imag separately)."""
    nl, ns = arr.shape
    # Pad to an exact multiple of looks
    pl = (az_looks - nl % az_looks) % az_looks
    ps = (rg_looks - ns % rg_looks) % rg_looks
    if pl or ps:
        arr = np.pad(arr, [(0, pl), (0, ps)], mode="edge")
    nl2, ns2 = arr.shape
    ol = nl2 // az_looks
    os_ = ns2 // rg_looks
    out = (
        arr.reshape(ol, az_looks, os_, rg_looks)
        .mean(axis=3)
        .mean(axis=1)
    )
    return out


def _boxcar_multilook_bool(
    mask: np.ndarray,
    az_looks: int,
    rg_looks: int,
) -> np.ndarray:
    """Boxcar-multilook a 2-D boolean mask (fraction of valid pixels ≥ 0.5)."""
    nl, ns = mask.shape
    pl = (az_looks - nl % az_looks) % az_looks
    ps = (rg_looks - ns % rg_looks) % rg_looks
    if pl or ps:
        mask = np.pad(mask, [(0, pl), (0, ps)], mode="constant", constant_values=False)
    nl2, ns2 = mask.shape
    ol = nl2 // az_looks
    os_ = ns2 // rg_looks
    # Fraction valid per look window
    fraction = (
        mask.reshape(ol, az_looks, os_, rg_looks)
        .mean(axis=3)
        .mean(axis=1)
    )
    return fraction >= 0.5


def _phase_gradient_axis(
    ifg: np.ndarray,
    axis: int,
) -> np.ndarray:
    """Phase gradient along ``axis`` using ``angle(c[r]*conj(c[r-1]))``.

    For axis=1 (range): gradient[r,c] = angle(ifg[r,c] * conj(ifg[r,c-1]))
    For axis=0 (azimuth): gradient[r,c] = angle(ifg[r,c] * conj(ifg[r-1,c]))

    Edge columns/rows are set to NaN.
    """
    if axis == 1:
        # range direction: shift right then left
        cur = ifg[:, 1:]          # [0, 1, ..., ns-1] → ns-1 elements
        prev = ifg[:, :-1]        # [0, 1, ..., ns-2] → ns-1 elements
        gradient = np.angle(cur * np.conj(prev))
        # pad right edge with NaN to match original width
        result = np.full(ifg.shape, np.nan, dtype=np.float64)
        result[:, :-1] = gradient
    elif axis == 0:
        # azimuth direction: shift down then up
        cur = ifg[1:, :]          # [1, 2, ..., nl-1] → nl-1 elements
        prev = ifg[:-1, :]        # [0, 1, ..., nl-2] → nl-1 elements
        gradient = np.angle(cur * np.conj(prev))
        result = np.full(ifg.shape, np.nan, dtype=np.float64)
        result[:-1, :] = gradient
    else:
        raise ValueError(f"axis must be 0 or 1, got {axis}")
    return result
