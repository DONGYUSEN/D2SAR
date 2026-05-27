"""tops_deramp — Sentinel-1 TOPS azimuth carrier phase model.

Provides deramp / reramp roundtrip functions for Sentinel-1 TOPS-mode SLC
bursts.  The azimuth carrier phase is modelled as a linear Doppler model:

    phi(l, s) = -2π * (f_DC + f_DR * s) * t(l)
              = -2π * (f_DC + f_DR * s) * (line_index / prf)

where
    line_index = l - burst.valid_window.first_line   (relative to valid window)
    f_DR       = doppler_coefficients[1]  (Hz / sample)   [0 if not present]
    f_DC       = doppler_coefficients[0]  (Hz)
    prf        = 1 / azimuth_time_interval  (Hz)

deramp  : slc_out = slc      * exp(+1j * phi)
reramp  : slc_out = deramped * exp(-1j * phi)

Dependency: tops_model only (no strip/tops_insar imports).
"""

from __future__ import annotations

__all__ = [
    "deramp_slc",
    "reramp_slc",
    "tophu_phase",
]

import numpy as np

from .tops_model import BurstRadarGrid


def tophu_phase(
    burst: BurstRadarGrid,
    lines: np.ndarray,
    samples: np.ndarray,
) -> np.ndarray:
    """Compute the TOPS azimuth carrier phase at the given pixel coordinates.

    Parameters
    ----------
    burst : BurstRadarGrid
        Burst radar parameters (prf, doppler_coefficients, valid_window, …).
    lines : np.ndarray
        Integer array of line (row) indices **relative to the
        burst's valid window** (i.e. 0 = first line of valid_window).
        Can be 0-D (scalar), 1-D, or 2-D.  For 1-D inputs a 2-D grid is
        produced via broadcasting.
    samples : np.ndarray
        Integer array of sample (column) indices **relative to the burst's
        valid window** (i.e. 0 = first sample of valid_window).
        Must broadcast to a compatible shape with ``lines``.

    Returns
    -------
    np.ndarray
        Float32 array of phase values in radians with the broadcast shape of
        ``lines`` and ``samples``.

    Raises
    ------
    ValueError
        If ``lines`` and ``samples`` are 1-D with incompatible lengths
        (i.e. cannot broadcast to a common shape).
    """
    # --- Doppler coefficients ------------------------------------------------
    dopp = burst.doppler_coefficients
    f_DC = float(dopp[0]) if dopp else 0.0
    f_DR = float(dopp[1]) if len(dopp) > 1 else 0.0

    # Convert to arrays; keep scalars as 0-D
    lines_arr = np.asarray(lines, dtype=np.intp)
    samples_arr = np.asarray(samples, dtype=np.intp)

    # Determine broadcast shape; raise if incompatible
    try:
        broadcast_shape = np.broadcast_shapes(lines_arr.shape, samples_arr.shape)
    except ValueError as exc:
        raise ValueError(
            f"lines {lines_arr.shape} and samples {samples_arr.shape} "
            "are not broadcast-compatible"
        ) from exc

    # Broadcast to final shape
    lines_bc = np.broadcast_to(lines_arr, broadcast_shape, subok=True)
    samples_bc = np.broadcast_to(samples_arr, broadcast_shape, subok=True)

    # line_index relative to valid window (float64 for precision)
    line_index = lines_bc.astype(np.float64, copy=False)
    # t(l) in seconds
    t_line = line_index / burst.prf  # (s)

    # Doppler frequency as function of range sample
    f_D = f_DC + f_DR * samples_bc.astype(np.float64, copy=False)  # (Hz)

    # Carrier phase: phi = -2π * f_D * t(l)
    phi = -2.0 * np.pi * f_D * t_line

    return phi.astype(np.float32, copy=False)


def deramp_slc(
    slc: np.ndarray,
    burst: BurstRadarGrid,
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Remove the TOPS azimuth carrier phase from an SLC burst.

    Parameters
    ----------
    slc : np.ndarray
        2-D complex array (complex64 or complex128) representing the
        raw (ramped) TOPS-mode SLC data.
    burst : BurstRadarGrid
        Burst radar parameters.
    out : np.ndarray, optional
        Pre-allocated output array.  Must have the same shape and a
        complex dtype.  If None (default) a new array is allocated.

    Returns
    -------
    np.ndarray
        Deramped SLC: ``slc * exp(+1j * phi)``.
        Same shape as ``slc``; dtype is complex64 if input is complex64,
        otherwise complex128.

    Raises
    ------
    ValueError
        If ``slc`` is not 2-D, ``burst.prf <= 0``, or
        if ``out`` is provided with the wrong shape.
    """
    # --- Input validation ----------------------------------------------------
    if slc.ndim != 2:
        raise ValueError(f"slc must be 2-D, got {slc.ndim}-D")

    if burst.prf <= 0:
        raise ValueError(f"burst.prf must be positive, got {burst.prf}")

    if out is not None:
        if out.shape != slc.shape:
            raise ValueError(f"out.shape {out.shape} != slc.shape {slc.shape}")
        if not np.issubdtype(out.dtype, np.complexfloating):
            raise ValueError(f"out.dtype must be complex, got {out.dtype}")

    # --- Compute phase -------------------------------------------------------
    nl, ns = slc.shape

    # Grid of line/sample indices **relative to valid window**
    lines_arr = np.arange(nl, dtype=np.intp).reshape(-1, 1)   # (nl, 1)
    samples_arr = np.arange(ns, dtype=np.intp).reshape(1, -1)  # (1, ns)

    phi = tophu_phase(burst, lines_arr, samples_arr)  # (nl, ns), float32

    # --- Complex multiplication ----------------------------------------------
    if out is None:
        out_arr = np.empty_like(slc)
    else:
        out_arr = out

    # exp(+1j * phi) then multiply: use np.multiply with out= for efficiency
    exp_phase = np.exp(1j * phi)  # complex64
    np.multiply(slc, exp_phase, out=out_arr)

    return out_arr


def reramp_slc(
    deramped: np.ndarray,
    burst: BurstRadarGrid,
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Restore the TOPS azimuth carrier phase on a deramped SLC burst.

    This is the inverse of ``deramp_slc``: reramp applies ``exp(-1j * phi)``
    where ``phi`` is the same carrier phase computed from ``burst``.

    Parameters
    ----------
    deramped : np.ndarray
        2-D complex array (complex64 or complex128) representing the
        deramped TOPS-mode SLC data.
    burst : BurstRadarGrid
        Burst radar parameters.
    out : np.ndarray, optional
        Pre-allocated output array.  Must have the same shape and a
        complex dtype.  If None (default) a new array is allocated.

    Returns
    -------
    np.ndarray
        Reramped SLC: ``deramped * exp(-1j * phi)``.
        Same shape as ``deramped``; dtype is complex64 if input is complex64,
        otherwise complex128.

    Raises
    ------
    ValueError
        If ``deramped`` is not 2-D, ``burst.prf <= 0``, or
        if ``out`` is provided with the wrong shape.
    """
    # --- Input validation ----------------------------------------------------
    if deramped.ndim != 2:
        raise ValueError(f"deramped must be 2-D, got {deramped.ndim}-D")

    if burst.prf <= 0:
        raise ValueError(f"burst.prf must be positive, got {burst.prf}")

    if out is not None:
        if out.shape != deramped.shape:
            raise ValueError(f"out.shape {out.shape} != deramped.shape {deramped.shape}")
        if not np.issubdtype(out.dtype, np.complexfloating):
            raise ValueError(f"out.dtype must be complex, got {out.dtype}")

    # --- Compute phase -------------------------------------------------------
    nl, ns = deramped.shape

    lines_arr = np.arange(nl, dtype=np.intp).reshape(-1, 1)
    samples_arr = np.arange(ns, dtype=np.intp).reshape(1, -1)

    phi = tophu_phase(burst, lines_arr, samples_arr)  # (nl, ns), float32

    # --- Complex multiplication ----------------------------------------------
    if out is None:
        out_arr = np.empty_like(deramped)
    else:
        out_arr = out

    # exp(-1j * phi) then multiply
    exp_phase = np.exp(-1j * phi)  # complex64
    np.multiply(deramped, exp_phase, out=out_arr)

    return out_arr
