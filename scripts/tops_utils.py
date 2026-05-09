"""tops_utils — Shared pure-math / engineering utilities for TOPS InSAR processing.

No domain logic lives here; only numerical helpers and types that are used
by multiple pipeline modules.
"""

from __future__ import annotations

__all__ = [
    "robust_median_with_mad",
    "evaluate_polynomial",
    "intersect_windows",
    "ensure_utc",
    "circular_mean",
    "safe_divide",
    "unwrap_phase_2d",
    "geocode_image",
    "pad_to_tile",
    "estimate_memory_usage",
    "pad_window_to_grid",
]

import math
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from scripts.tops_model import BurstWindow


# ---------------------------------------------------------------------------
# Robust statistics
# ---------------------------------------------------------------------------

def robust_median_with_mad(data: np.ndarray, *, axis=None) -> np.ndarray | float:
    """Median with MAD-based outlier rejection.

    MAD = median(|x - median(x)|).  Values whose absolute deviation from
    the median exceeds 3 × MAD are excluded before re-taking the median.

    Parameters
    ----------
    data : np.ndarray
        1-D or 2-D array of real numbers.
    axis : int | None, optional
        Axis along which to compute.  If None, the whole array is flattened.

    Returns
    -------
    np.ndarray | float
        Robust median (scalar when axis=None, otherwise array along ``axis``).
    """
    data = np.asarray(data, dtype=np.float64)
    if data.size == 0:
        raise ValueError("data must not be empty")
    med = np.median(data, axis=axis)
    # MAD
    if axis is None:
        mad = np.median(np.abs(data - med))
    else:
        med_exp = np.expand_dims(med, axis=axis)
        mad = np.median(np.abs(data - med_exp), axis=axis)
    # 3-sigma-like threshold via MAD scale factor
    threshold = 3.0 * mad + 1e-12
    if axis is None:
        masked_data = data[np.abs(data - med) <= threshold]
        if masked_data.size == 0:
            return float(med)
        return float(np.median(masked_data))
    else:
        mask = np.abs(data - np.expand_dims(med, axis=axis)) <= threshold
        # Use nanmedian to ignore masked (False) positions
        # Replace masked values with NaN so np.nanmedian ignores them
        masked_arr = np.where(mask, data, np.nan)
        result = np.nanmedian(masked_arr, axis=axis)
        if result.ndim == 0:
            return float(result)
        return result


# ---------------------------------------------------------------------------
# Polynomial evaluation
# ---------------------------------------------------------------------------

def evaluate_polynomial(coeffs: list[float], x: np.ndarray) -> np.ndarray:
    """Evaluate a polynomial with coefficients ``coeffs`` at points ``x``.

    Uses Horner's method:  c[0] + c[1]*x + c[2]*x² + ... + c[n]*xⁿ.
    When ``x`` is a scalar the result is a scalar.

    Parameters
    ----------
    coeffs : list[float]
        Polynomial coefficients in order of increasing degree.
        ``coeffs[0]`` is the constant term.  Empty list → array of zeros.
    x : np.ndarray
        Points at which to evaluate.

    Returns
    -------
    np.ndarray
        Polynomial values, same shape as ``x``.
    """
    if not coeffs:
        return np.zeros_like(x, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    result = np.full_like(x, coeffs[-1], dtype=np.float64)
    for c in reversed(coeffs[:-1]):
        result *= x
        result += c
    return result


# ---------------------------------------------------------------------------
# BurstWindow utilities
# ---------------------------------------------------------------------------

def intersect_windows(w1: BurstWindow, w2: BurstWindow) -> BurstWindow:
    """Pixel-coordinate intersection of two windows.

    If the windows do not overlap in either dimension, returns a window with
    ``first_line = -1`` and ``num_lines = 0`` (i.e. invalid sentinel).

    Parameters
    ----------
    w1, w2 : BurstWindow
        Input windows.

    Returns
    -------
    BurstWindow
        Intersection of ``w1`` and ``w2``.
    """
    line_start = max(w1.first_line, w2.first_line)
    line_stop = min(w1.line_stop, w2.line_stop)
    sample_start = max(w1.first_sample, w2.first_sample)
    sample_stop = min(w1.sample_stop, w2.sample_stop)
    num_lines = max(0, line_stop - line_start)
    num_samples = max(0, sample_stop - sample_start)
    # Empty sentinel
    if num_lines == 0 or num_samples == 0:
        return BurstWindow(first_line=-1, num_lines=0,
                            first_sample=-1, num_samples=0)
    return BurstWindow(first_line=line_start, num_lines=num_lines,
                       first_sample=sample_start, num_samples=num_samples)


def pad_window_to_grid(
    win: BurstWindow,
    grid_lines: int,
    grid_samples: int,
) -> tuple[int, int, int, int]:
    """Pad a BurstWindow so it fits entirely inside a grid of ``grid_lines`` × ``grid_samples``.

    Coordinates are clamped to [0, grid_lines) / [0, grid_samples) so that
    the returned window is a valid window inside the grid.  This is used to
    expand a burst window to grid boundaries before reading / resampling.

    Parameters
    ----------
    win : BurstWindow
        Input window (any coordinates).
    grid_lines : int
        Number of lines in the destination grid.
    grid_samples : int
        Number of samples in the destination grid.

    Returns
    -------
    tuple[int, int, int, int]
        ``(first_line, num_lines, first_sample, num_samples)`` of the
        padded window.
    """
    fl = max(0, min(win.first_line, grid_lines - 1))
    ls = max(0, min(win.line_stop, grid_lines))
    nl = max(0, ls - fl)
    fs = max(0, min(win.first_sample, grid_samples - 1))
    ss = max(0, min(win.sample_stop, grid_samples))
    ns = max(0, ss - fs)
    return fl, nl, fs, ns


# ---------------------------------------------------------------------------
# Datetime utilities
# ---------------------------------------------------------------------------

def ensure_utc(dt: datetime) -> datetime:
    """Convert ``dt`` to a timezone-aware UTC datetime.

    - If ``dt`` is already tz-aware with any timezone, replace its timezone
      with ``datetime.timezone.utc``.
    - If ``dt`` has no ``tzinfo`` (naive), it is interpreted as UTC and
      ``timezone.utc`` is attached.
    - ``tzinfo=None`` raises ``ValueError``.

    Parameters
    ----------
    dt : datetime
        Input datetime.

    Returns
    -------
    datetime
        Timezone-aware UTC datetime.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# Circular statistics
# ---------------------------------------------------------------------------

def circular_mean(phases: np.ndarray, *, axis=None) -> np.ndarray | float:
    """Circular mean of phase values in radians.

    Uses the complex exponential representation:
        mean = angle(mean(exp(j*phase)))

    Parameters
    ----------
    phases : np.ndarray
        Phase values in radians, any shape.
    axis : int | None, optional
        Axis along which to compute.  If None, flatten first.

    Returns
    -------
    np.ndarray | float
        Circular mean in [-π, π).  Scalar when ``axis=None``.
    """
    phases = np.asarray(phases, dtype=np.float64)
    mean_complex = np.mean(np.exp(1j * phases), axis=axis)
    # Guard against degenerate case (all phases cancel → zero mean)
    if np.ndim(mean_complex) == 0:
        if np.abs(mean_complex) < 1e-12:
            return 0.0
        result: np.ndarray | float = float(np.angle(mean_complex))
    else:
        # Vector case: small-magnitude columns → angle undefined → set to 0
        small = np.abs(mean_complex) < 1e-12
        raw = np.angle(mean_complex)
        result = np.where(small, 0.0, raw)
    return result


# ---------------------------------------------------------------------------
# Safe arithmetic
# ---------------------------------------------------------------------------

def safe_divide(a, b, fill: float = 0.0) -> np.ndarray | float:
    """Element-wise division with zero-division protection.

    Wherever ``b`` is zero (or NaN) the result is replaced with ``fill``.
    Broadcasting follows NumPy's standard rules.

    Parameters
    ----------
    a : array_like
        Numerator.
    b : array_like
        Denominator.
    fill : float, default 0.0
        Value to use where ``b`` is zero or NaN.

    Returns
    -------
    np.ndarray | float
        ``a / b`` with ``fill`` substitution.
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    # NaN is treated like zero (replace with fill before division)
    b_arr = np.where(np.isnan(b_arr), 0.0, b_arr)
    result = np.full_like(a_arr, fill, dtype=np.float64)
    mask = b_arr != 0.0
    with np.errstate(divide='ignore', invalid='ignore'):
        np.divide(a_arr, b_arr, out=result, where=mask)
    return result


# ---------------------------------------------------------------------------
# Phase unwrapping
# ---------------------------------------------------------------------------

def unwrap_phase_2d(phase: np.ndarray, /) -> np.ndarray:
    """2-D phase unwrapping via seeded row-wise horizontal sweep.

    Step 1 — horizontal: for each row, cumulative sum of wrapped differences
             starting from the first pixel.
    Step 2 — vertical: starting from row 0 (which is already a correct
             reference), propagate absolute phase levels row by row.
             For each row l>0, compute the phase gap at the first *non-border*
             column between the current wrapped difference and the previous
             row's unwrapped value.  Round to the nearest multiple of 2π and
             apply that correction to the whole row.

    Using column 1 (rather than column 0) as the reference avoids the ±π
    ambiguity that occurs when the true phase equals ±π and wraps to 0.

    Parameters
    ----------
    phase : np.ndarray
        2-D wrapped phase in radians (shape ``(lines, samples)``).

    Returns
    -------
    np.ndarray
        Unwrapped phase with same shape as ``phase``.
    """
    phase = np.asarray(phase, dtype=np.float64)
    if phase.ndim != 2:
        raise ValueError(f"unwrap_phase_2d expects 2-D array, got {phase.ndim}D")

    lines, samples = phase.shape

    # Step 1 — horizontal unwrapping per row (relative to first pixel)
    unwrapped = np.zeros_like(phase)
    for l in range(lines):
        row = phase[l]
        if samples > 1:
            diff = np.diff(row)
            # Wrap differences to [-π, π]
            wrap = (diff + np.pi) % (2.0 * np.pi) - np.pi
            # Cumulative sum: start from row[0], accumulate wrapped diffs
            unwrapped[l, 0] = row[0]
            unwrapped[l, 1:] = row[0] + np.cumsum(wrap)
        else:
            unwrapped[l, 0] = row[0]

    # Step 2 — vertical level propagation
    # For each row l>0, use column 1 (not 0) as reference because
    # column 0 may be at ±π and wrap to 0, creating a ±π ambiguity.
    # Column 1 is the cumulative sum of one wrapped diff, which carries
    # the correct relative offset from column 0.
    TWO_PI = 2.0 * np.pi
    for l in range(1, lines):
        if samples > 1:
            # Use column 1: gap between row l's wrapped col1 and
            # row l-1's unwrapped col1 tells us how many 2π jumps to add.
            gap = phase[l, 1] - unwrapped[l - 1, 1]
            k = round(gap / TWO_PI)
            correction = k * TWO_PI
        else:
            # Single column: compare at column 0
            gap = phase[l, 0] - unwrapped[l - 1, 0]
            k = round(gap / TWO_PI)
            correction = k * TWO_PI
        unwrapped[l, :] += correction

    return unwrapped


# ---------------------------------------------------------------------------
# Geocoding
# ---------------------------------------------------------------------------

def geocode_image(
    data: np.ndarray,
    geo_transform: tuple,
    crs: str,
    /,
) -> np.ndarray:
    """Geocode a raster image via GDAL warp.

    Parameters
    ----------
    data : np.ndarray
        Input raster (2-D or 3-D).
    geo_transform : tuple
        GDAL geotransform tuple ``(ul_x, w_e, rot_1, ul_y, rot_2, n_s)``.
    crs : str
        Target CRS (e.g. ``"EPSG:4326"``).

    Returns
    -------
    np.ndarray
        Geocoded image.

    Raises
    ------
    NotImplementedError
        When GDAL is not available.
    """
    try:
        from osgeo import gdal, gdalconst
    except ImportError:
        raise NotImplementedError(
            "geocode_image requires GDAL (osgeo). Install with: pip install GDAL"
        )

    import tempfile, os

    # Write to temporary in-memory raster
    with tempfile.TemporaryDirectory() as tmp:
        src_path = os.path.join(tmp, "src.tif")
        dst_path = os.path.join(tmp, "dst.tif")

        # Create source in-memory dataset
        if data.ndim == 2:
            nbands, height, width = 1, *data.shape
        else:
            nbands, height, width = data.shape

        driver = gdal.GetDriverByName("GTiff")
        src_ds = driver.Create(src_path, width, height, nbands, gdal.GDT_Float32)
        src_ds.SetGeoTransform(geo_transform)
        src_ds.SetProjection(crs)
        for b in range(nbands):
            band = src_ds.GetRasterBand(b + 1)
            if data.ndim == 2:
                band.WriteArray(data)
            else:
                band.WriteArray(data[b])
            band.FlushCache()
        src_ds = None

        # Warp
        warp_ds = gdal.Warp(dst_path, src_path,
                            dstSRS=crs,
                            resampleAlg="bilinear")
        result = warp_ds.ReadAsArray()
        warp_ds = None

    return result


# ---------------------------------------------------------------------------
# Padding utilities
# ---------------------------------------------------------------------------

def pad_to_tile(data: np.ndarray, tile_size: int) -> np.ndarray:
    """Pad ``data`` to the next integer multiple of ``tile_size`` on both axes.

    Padding is applied symmetrically (or as close as possible) on both sides,
    using zero-fill.

    Parameters
    ----------
    data : np.ndarray
        Input array.
    tile_size : int
        Target tile dimension (must be positive).

    Returns
    -------
    np.ndarray
        Padded array with shape that is a multiple of ``tile_size`` along
        each axis.

    Raises
    ------
    ValueError
        When ``tile_size < 1``.
    """
    if tile_size < 1:
        raise ValueError(f"tile_size must be >= 1, got {tile_size}")

    data = np.asarray(data)
    new_shape = []
    for size in data.shape:
        remainder = size % tile_size
        if remainder == 0:
            new_size = size
        else:
            new_size = size + (tile_size - remainder)
        new_shape.append(new_size)

    padded = np.zeros(new_shape, dtype=data.dtype)
    slices = tuple(slice(0, s) for s in data.shape)
    padded[slices] = data
    return padded


# ---------------------------------------------------------------------------
# Memory estimation
# ---------------------------------------------------------------------------

def estimate_memory_usage(shape, dtype) -> int:
    """Estimate memory footprint of a NumPy array.

    Parameters
    ----------
    shape : tuple[int, ...]
        Array shape.
    dtype : np.dtype | str | type
        NumPy dtype or dtype string.

    Returns
    -------
    int
        Estimated number of bytes.
    """
    dtype = np.dtype(dtype)
    total = dtype.itemsize
    for dim in shape:
        total *= dim
    return int(total)