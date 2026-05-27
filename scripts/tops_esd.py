"""tops_esd — Elevation-dependent antenna pointing (ESD) timing correction.

Algorithm
---------
1. Multilook the overlap interferogram (boxcar in azimuth).
2. Compute the phase raster; average along azimuth → line-averaged phase per column.
3. Robust median of the line-averaged phase → offset angle.
4. Convert angle → pixels using the Sentinel-1 C-band wavelength and PRF.
5. Convert pixels → seconds; apply as a phase ramp to the secondary SLC.

Key formulas
------------
- offset_pixels = median(azimuth_averaged_angle) * wavelength_m * prf / (4π * c)
- secondary_timing_seconds = offset_pixels / prf
- wavelength_m = 0.055465  (Sentinel-1 C-band)
- c = 299_792_458.0  m/s
- f_center ≈ 5.405e9 Hz

No imports from strip/tops_insar backends.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from .tops_model import BurstRadarGrid, EsdEstimate, TimingCorrection

# Physical constants (Sentinel-1 C-band)
SPEED_OF_LIGHT: float = 299_792_458.0   # m/s
SENTINEL1_WAVELENGTH: float = 0.055465  # metres (C-band)

# Maximum expected azimuth misregistration before flagging a warning
MAX_OFFSET_PIXELS: float = 2.0


def _boxcar_multilook(
    arr: np.ndarray,
    az_looks: int,
    rg_looks: int,
) -> np.ndarray:
    """Boxcar (block-average) multilook.

    Parameters
    ----------
    arr : np.ndarray
        2-D complex or real array.
    az_looks : int
        Number of azimuth looks (must divide arr.shape[0]).
    rg_looks : int
        Number of range looks (must divide arr.shape[1]).

    Returns
    -------
    np.ndarray
        Multilooked array with shape (arr.shape[0] // az_looks,
        arr.shape[1] // rg_looks).
    """
    nl, ns = arr.shape
    if nl % az_looks != 0:
        nl = (nl // az_looks) * az_looks
        arr = arr[:nl]
    if ns % rg_looks != 0:
        ns = (ns // rg_looks) * rg_looks
        arr = arr[:, :ns]
    sh2 = (nl // az_looks, az_looks, ns // rg_looks, rg_looks)
    return arr.reshape(sh2).mean(axis=(1, 3))


def estimate_esd_timing(
    overlap_ifg: np.ndarray,
    coherence: np.ndarray,
    *,
    looks_az: int = 5,
    az_time_interval: float = 0.002,
) -> EsdEstimate:
    """Estimate azimuth timing offset from a pre-computed overlap interferogram.

    The overlap interferogram is ``top_SLC * conj(bottom_SLC)`` as produced
    by the ``tops_ifg`` stage.  This function multilooks it, extracts a
    frequency raster, and returns a robust pixel-offset estimate.

    Parameters
    ----------
    overlap_ifg : np.ndarray
        Complex overlap interferogram, shape (L, S).  Must have at least
        one line and one sample.  Zero-filled or all-NaN arrays will raise
        ``ValueError``.
    coherence : np.ndarray
        Real-valued coherence raster, same shape as ``overlap_ifg``.
        Used to compute ``mean_coherence`` for quality assessment.
    looks_az : int, default 5
        Number of azimuth looks for boxcar multilooking.

    Returns
    -------
    EsdEstimate
        Dataclass containing:

        - ``median_offset_pixels`` — robust azimuth misregistration (pixels)
        - ``mean_offset_pixels``   — arithmetic mean offset (pixels)
        - ``std_offset_pixels``    — standard deviation of offsets (pixels)
        - ``mean_coherence``       — mean coherence over the ESD window (0..1)
        - ``sample_count``         — number of range columns contributing
        - ``azimuth_time_interval`` — placeholder (set to 0.0 here; the caller
          provides the real value when building ``TimingCorrection``)

    Raises
    ------
    ValueError
        If ``overlap_ifg`` has zero area, all-zero values, or all NaN values.
    RuntimeError
        If no finite offsets remain after masking (should not occur for
        normally-constructed input).

    Warns
    -----
    UserWarning
        If the median offset exceeds ±2.0 pixels (unusual but not fatal).
    """
    if overlap_ifg.ndim != 2:
        raise ValueError(f"overlap_ifg must be 2-D, got {overlap_ifg.ndim}-D")
    if overlap_ifg.size == 0:
        raise ValueError("overlap_ifg is empty")
    if not np.any(np.isfinite(overlap_ifg)):
        raise ValueError("overlap_ifg contains no finite values")
    if np.abs(overlap_ifg).max() == 0.0:
        raise ValueError("overlap_ifg is all zeros")

    log = logging.getLogger("tops_esd")

    # Step 1: azimuth boxcar multilook
    ifg_ml = _boxcar_multilook(overlap_ifg, az_looks=looks_az, rg_looks=1)

    # Step 2: frequency raster = phase averaged along azimuth per range column
    # wrapped phase (rad), shape (n_az_ml, n_rg)
    phase_ml = np.angle(ifg_ml)

    # Average along azimuth → one value per range column
    # Use circular mean: mean of exp(j*phase) is more robust than mean of phase
    phasor_ml = np.exp(1j * phase_ml)
    mean_phasor = phasor_ml.mean(axis=0)           # (n_rg,)
    line_averaged_angle = np.angle(mean_phasor)    # (n_rg,) radians

    # Step 3: robust median offset (angle → pixels)
    offset_angle = float(np.median(line_averaged_angle))

    # Step 4: convert angle to pixels
    # angle per pixel = 4π * f_center / PRF  (from SAR geometry)
    #   where f_center = c / wavelength
    # => offset_pixels = offset_angle / (4π * f_center / PRF)
    #                   = offset_angle * wavelength * PRF / (4π * c)
    f_center = SPEED_OF_LIGHT / SENTINEL1_WAVELENGTH
    prf = 1.0 / az_time_interval
    offset_pixels = offset_angle * SENTINEL1_WAVELENGTH * prf / (4.0 * np.pi * SPEED_OF_LIGHT)

    # Per-column pixel offsets (for mean / std statistics)
    col_offsets = line_averaged_angle * SENTINEL1_WAVELENGTH * prf / (4.0 * np.pi * SPEED_OF_LIGHT)
    col_offsets = col_offsets[np.isfinite(col_offsets)]

    if col_offsets.size == 0:
        raise RuntimeError("ESD: no finite offsets after angle-to-pixel conversion")

    mean_pixels = float(np.mean(col_offsets))
    std_pixels = float(np.std(col_offsets))

    if abs(offset_pixels) > MAX_OFFSET_PIXELS:
        log.warning(
            "ESD median offset %.3f pixels exceeds ±%.1f pixel threshold "
            "(wavelength=%.6f m, PRF=%.1f Hz). Proceeding anyway.",
            offset_pixels, MAX_OFFSET_PIXELS, SENTINEL1_WAVELENGTH, prf,
        )

    # Step 5: compute mean coherence over the ESD window
    coh_finite = coherence[np.isfinite(coherence)]
    mean_coh = float(np.mean(coh_finite)) if coh_finite.size > 0 else 0.0

    return EsdEstimate(
        median_offset_pixels=float(offset_pixels),
        mean_offset_pixels=mean_pixels,
        std_offset_pixels=std_pixels,
        mean_coherence=mean_coh,
        sample_count=int(col_offsets.size),
        azimuth_time_interval=az_time_interval,
    )


def compute_esd_timing_correction(
    esd_estimate: EsdEstimate,
    az_time_interval: float,
) -> TimingCorrection:
    """Convert an ESD pixel estimate to a secondary-timing correction.

    Parameters
    ----------
    esd_estimate : EsdEstimate
        Result from ``estimate_esd_timing``.
    az_time_interval : float
        Burst azimuth time interval (seconds per line), i.e. 1/PRF.

    Returns
    -------
    TimingCorrection
        Contains:

        - ``secondary_timing_seconds`` — correction in seconds
        - ``secondary_timing_pixels``   — correction in pixels (same as median)
        - ``esd_estimate``              — the input estimate with the correct
          ``azimuth_time_interval`` embedded
    """
    if az_time_interval <= 0.0:
        raise ValueError(f"az_time_interval must be positive, got {az_time_interval}")

    pixels = esd_estimate.median_offset_pixels
    seconds = pixels * az_time_interval

    # Attach the real az_time_interval to the estimate (replace placeholder)
    corrected_estimate = EsdEstimate(
        median_offset_pixels=pixels,
        mean_offset_pixels=esd_estimate.mean_offset_pixels,
        std_offset_pixels=esd_estimate.std_offset_pixels,
        mean_coherence=esd_estimate.mean_coherence,
        sample_count=esd_estimate.sample_count,
        azimuth_time_interval=az_time_interval,
    )

    return TimingCorrection(
        secondary_timing_seconds=seconds,
        secondary_timing_pixels=pixels,
        esd_estimate=corrected_estimate,
    )


def apply_esd_correction(
    slc: np.ndarray,
    burst: BurstRadarGrid,
    correction: TimingCorrection,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply ESD timing correction as a Doppler-frequency-dependent phase ramp.

    The correction is:

        phase_correction(l, s) = -2π * f_D(s) * dt

    where ``dt = correction.secondary_timing_seconds`` and
    ``f_D(s)`` is the Doppler centroid at slant range sample ``s``.
    This is equivalent to a azimuth time shift of ``dt`` seconds.

    Parameters
    ----------
    slc : np.ndarray
        2-D complex SLC to correct (reference or already-coregistered secondary).
    burst : BurstRadarGrid
        Radar geometry parameters for the burst.
    correction : TimingCorrection
        ESD timing correction from ``compute_esd_timing_correction``.
    out : np.ndarray, optional
        Output buffer.  If provided, must have the same shape and dtype as ``slc``.
        If None, a new array is allocated.

    Returns
    -------
    np.ndarray
        Corrected SLC (same shape/dtype as ``slc``).  If ``out`` was supplied,
        this is the same object as ``out``.
    """
    if slc.ndim != 2:
        raise ValueError(f"slc must be 2-D, got {slc.ndim}-D")

    if out is None:
        out = np.empty_like(slc)
    else:
        if out.shape != slc.shape:
            raise ValueError(f"out.shape {out.shape} != slc.shape {slc.shape}")
        if out.dtype != slc.dtype:
            raise ValueError(f"out.dtype {out.dtype} != slc.dtype {slc.dtype}")

    dt = correction.secondary_timing_seconds

    # Build slant-range per sample index
    ns = slc.shape[1]
    slant_range = burst.starting_range + np.arange(ns, dtype=np.float64) * burst.range_pixel_spacing

    # Doppler centroid at each sample: f_D(s) = sum_k doppler_coefficients[k] * s^k
    dopp = burst.doppler_coefficients
    if not dopp:
        f_doppler = np.zeros(ns, dtype=np.float64)
    else:
        # Evaluate polynomial: f_D(s) = d[0] + d[1]*s + d[2]*s^2 + ...
        # Use Horner's method for numerical stability
        f_doppler = np.empty(ns, dtype=np.float64)
        for k in range(len(dopp)):
            if k == 0:
                f_doppler[:] = dopp[0]
            else:
                f_doppler[:] = f_doppler * slant_range + dopp[k]

    # Phase ramp: phi(l, s) = -2π * f_D(s) * dt
    # The azimuth index l cancels out — the ramp is constant along azimuth
    phi = (-2.0 * np.pi * f_doppler * dt).astype(np.float32)

    # Broadcast phi across azimuth dimension and apply as phase multiply
    ramp = np.exp(1j * phi[np.newaxis, :])  # shape (1, ns) → broadcasts to (nl, ns)
    out[:] = slc * ramp

    return out


def write_esd_summary(
    esd_estimate: EsdEstimate,
    path: Path,
) -> None:
    """Write ESD estimate to a JSON diagnostic file.

    Parameters
    ----------
    esd_estimate : EsdEstimate
        Result from ``estimate_esd_timing``.
    path : Path
        Output JSON file path.  Intermediate directories are created as needed.
    """
    import json

    path.parent.mkdir(parents=True, exist_ok=True)

    az_interval = esd_estimate.azimuth_time_interval
    median_pix = esd_estimate.median_offset_pixels

    payload = {
        "median_offset_pixels": float(median_pix),
        "std_offset_pixels": float(esd_estimate.std_offset_pixels),
        "mean_offset_pixels": float(esd_estimate.mean_offset_pixels),
        "sample_count": int(esd_estimate.sample_count),
        "azimuth_time_interval": float(az_interval),
        "secondary_timing_seconds": float(median_pix * az_interval),
    }

    with path.open("w") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
