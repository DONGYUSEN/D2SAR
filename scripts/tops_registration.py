"""tops_registration — Sentinel-1 TOPS coarse / fine resampling with deramp/reramp.

Orchestrates the full registration pipeline per burst pair::

    Geo2Rdr offsets
      ↓
    deramp(reference)  →  deramped_ref
      ↓
    deramp(secondary)  →  deramped_sec
      ↓
    resample(deramped_sec → ref_grid, offsets)  →  deramped_resampled_sec
      ↓
    reramp(deramped_resampled_sec)  →  resampled_sec (coreg)
      ↓
    [ESD timing + range coreg — future steps]
      ↓
    fine resample (future)

ISCE3 C++ bindings are not required for the coarse resampling step;
pure NumPy + SciPy are used instead.

Dependency order: tops_model, tops_deramp.
No imports from strip/tops_insar backends.
"""

from __future__ import annotations

__all__ = [
    "run_coarse_registration",
    "fine_resample_with_timing",
    "filter_ifg",
    "_resample_sliding_window",
]

import logging
from pathlib import Path

import numpy as np

from scripts.tops_model import BurstRadarGrid, Geo2RdrOffsets, RangeCoregEstimate, TimingCorrection
from scripts.tops_deramp import deramp_slc, reramp_slc

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_slc_npz(path: str | Path) -> np.ndarray:
    """Load a complex64 SLC from a numpy .npz file.

    The file must contain ``data`` (the complex array) and ``shape``
    (a tuple used for reshape if needed, though shape is inferred from
    the array itself).

    Parameters
    ----------
    path : str | Path
        Path to the .npz file.

    Returns
    -------
    np.ndarray
        Complex64 array with shape matching the stored data.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SLC file not found: {path}")
    with np.load(path) as npz:
        data = npz["data"]
    # Ensure complex64
    if data.dtype == np.complex128:
        data = data.astype(np.complex64)
    return data


def _save_slc_npz(slc: np.ndarray, path: str | Path) -> None:
    """Save a complex64 SLC to a numpy .npz file.

    Parameters
    ----------
    slc : np.ndarray
        2-D complex array (complex64 or complex128).
    path : str | Path
        Output .npz path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = slc.astype(np.complex64)
    shape = data.shape
    np.savez(path, data=data, shape=np.array(shape))
    log.debug("Saved SLC %s  shape=%s  dtype=%s", path, shape, data.dtype)


# ---------------------------------------------------------------------------
# Core resampling: bilinear sliding window
# ---------------------------------------------------------------------------

def _resample_sliding_window(
    src: np.ndarray,
    offset_rows: np.ndarray,
    offset_cols: np.ndarray,
    *,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Resample a source 2-D array using per-pixel sub-pixel offsets (bilinear).

    For each output pixel (r_out, c_out) the source coordinate is::

        r_src = r_out + offset_rows[r_out, c_out]
        c_src = c_out + offset_cols[r_out, c_out]

    Uses ``scipy.ndimage.map_coordinates`` with ``order=1`` (bilinear).

    Parameters
    ----------
    src : np.ndarray
        2-D source array (complex or real).
    offset_rows : np.ndarray
        2-D float array of row (azimuth) offsets in pixel units.
        Shape must match the desired output shape.
    offset_cols : np.ndarray
        2-D float array of column (range) offsets in pixel units.
        Shape must match ``offset_rows``.
    out : np.ndarray, optional
        Pre-allocated output array.  If None, a new array is allocated.

    Returns
    -------
    np.ndarray
        Resampled array with shape equal to the offset array shapes.
        dtype is complex64 if ``src`` is complex, float32 otherwise.
    """
    # --- Validate input -------------------------------------------------------
    if src.ndim != 2:
        raise ValueError(f"src must be 2-D, got {src.ndim}-D")
    if offset_rows.shape != offset_cols.shape:
        raise ValueError(
            f"offset_rows.shape {offset_rows.shape} != "
            f"offset_cols.shape {offset_cols.shape}"
        )
    out_shape = offset_rows.shape

    if out is None:
        if np.iscomplexobj(src):
            out_arr = np.empty(out_shape, dtype=np.complex64)
        else:
            out_arr = np.empty(out_shape, dtype=np.float32)
    else:
        if out.shape != out_shape:
            raise ValueError(f"out.shape {out.shape} != expected {out_shape}")
        out_arr = out

    n_rows, n_cols = out_shape

    # --- Compute source coordinates --------------------------------------------
    # For each output pixel (r_out, c_out):
    #   r_src = r_out + offset_rows[r_out, c_out]
    #   c_src = c_out + offset_cols[r_out, c_out]
    # The offset arrays broadcast over the full output shape.
    offset_rows_bc = np.broadcast_to(offset_rows, out_shape).astype(np.float64)
    offset_cols_bc = np.broadcast_to(offset_cols, out_shape).astype(np.float64)

    # Grid of output pixel coordinates
    row_grid = np.arange(n_rows, dtype=np.float64).reshape(-1, 1)  # (n_rows, 1)
    col_grid = np.arange(n_cols, dtype=np.float64).reshape(1, -1)   # (1, n_cols)

    # Source coordinates (r_src, c_src) — shape (2, n_rows, n_cols)
    r_src = row_grid + offset_rows_bc   # (n_rows, n_cols)
    c_src = col_grid + offset_cols_bc   # (n_rows, n_cols)

    # Flatten to (2, n_rows * n_cols) for map_coordinates
    src_coords = np.stack([r_src.ravel(), c_src.ravel()], axis=0)  # (2, N)

    # --- Bilinear interpolation -------------------------------------------------
    try:
        from scipy.ndimage import map_coordinates
    except ImportError as exc:
        raise ImportError(
            "scipy is required for coarse resampling. "
            "Install: pip install scipy"
        ) from exc

    if np.iscomplexobj(src):
        # scipy.ndimage.map_coordinates natively handles complex arrays
        # (real and imag are interpolated independently, documented since v1.6.0).
        mapped = map_coordinates(
            src,
            src_coords,
            order=1,
            mode="constant",
            cval=0.0,   # real; scipy extends real and imag identically for complex input
            prefilter=True,
        )  # shape (N,), dtype complex
        out_arr[:] = mapped.reshape(out_shape).astype(np.complex64)
    else:
        mapped = map_coordinates(
            src,
            src_coords,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=True,
        )  # shape (N,)
        out_arr[:] = mapped.reshape(out_shape).astype(out_arr.dtype)

    return out_arr


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_coarse_registration(
    ref_burst: BurstRadarGrid,
    sec_burst: BurstRadarGrid,
    geo2rdr_offsets: Geo2RdrOffsets,
    *,
    work_dir: Path | str,
    deramped_ref_path: Path | str,
    deramped_sec_path: Path | str,
    resampled_sec_path: Path | str,
) -> None:
    """Coarse coregistration of a secondary SLC onto a reference SLC using
    Geo2Rdr offsets and TOPS deramp/reramp.

    Algorithm
    ---------
    1. Load secondary SLC from disk (format: numpy .npz, complex64).
    2. Deramp reference SLC: ``ref_der = deramp_slc(ref_slc, ref_burst)``.
    3. Deramp secondary SLC: ``sec_der = deramp_slc(sec_slc, sec_burst)``.
    4. Read Geo2Rdr range/azimuth offset grids from disk.
    5. Resample deramped secondary onto the reference grid:
       ``sec_der_resampled = _resample_sliding_window(sec_der, az_off, rg_off)``.
    6. Reramp the resampled secondary back onto the TOPS carrier:
       ``resampled_sec = reramp_slc(sec_der_resampled, ref_burst)``.
    7. Save three products to ``work_dir``:
       - ``deramped_ref``: reference deramped SLC
       - ``deramped_sec``: secondary deramped SLC
       - ``resampled_sec``: secondary coregistered SLC (deramp → resamp → reramp)

    Parameters
    ----------
    ref_burst : BurstRadarGrid
        Reference burst metadata.
    sec_burst : BurstRadarGrid
        Secondary burst metadata.
    geo2rdr_offsets : Geo2RdrOffsets
        Geo2Rdr offsets produced by ``tops_geometry.run_geo2rdr_single_burst``.
        Contains paths to ``range.off`` and ``azimuth.off`` numpy arrays.
    work_dir : Path | str
        Working directory for this burst pair.  Created if it does not exist.
    deramped_ref_path : Path | str
        Output path for the deramped reference SLC (.npz, complex64).
    deramped_sec_path : Path | str
        Output path for the deramped secondary SLC (.npz, complex64).
    resampled_sec_path : Path | str
        Output path for the resampled (coregistered) secondary SLC (.npz, complex64).

    Raises
    ------
    FileNotFoundError
        If the secondary SLC, range offset, or azimuth offset files are not found.
    ValueError
        If offset array shapes do not match the reference burst valid window shape.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Expected output shape
    out_shape = (
        ref_burst.valid_window.num_lines,
        ref_burst.valid_window.num_samples,
    )

    # -------------------------------------------------------------------------
    # Step 1: Load secondary SLC from disk
    # -------------------------------------------------------------------------
    # The secondary SLC must be stored as npz in the burst work directory.
    # Convention: {work_dir}/secondary_slc_{swath}_{burst_idx}.slc.npz
    sec_slc = _load_slc_npz(
        work_dir / f"secondary_slc_{sec_burst.identity.swath}_{sec_burst.identity.burst_index}.slc.npz"
    )

    log.info(
        "run_coarse_registration  ref=%s[%d]  sec=%s[%d]  shape=%s",
        ref_burst.identity.swath, ref_burst.identity.burst_index,
        sec_burst.identity.swath, sec_burst.identity.burst_index,
        out_shape,
    )

    # -------------------------------------------------------------------------
    # Step 2: Deramp reference
    # -------------------------------------------------------------------------
    # We need the reference SLC. If it was already written by a previous stage
    # (e.g. from a TIF window read), load it; otherwise we raise.
    ref_slc_path = (
        work_dir
        / f"reference_slc_{ref_burst.identity.swath}_{ref_burst.identity.burst_index}.slc.npz"
    )
    if ref_slc_path.exists():
        ref_slc = _load_slc_npz(ref_slc_path)
    else:
        raise FileNotFoundError(
            f"Reference SLC not found at {ref_slc_path}. "
            "Ensure the reference SLC has been written to disk before "
            "running coarse registration."
        )

    ref_der = deramp_slc(ref_slc, ref_burst)
    _save_slc_npz(ref_der, deramped_ref_path)

    # -------------------------------------------------------------------------
    # Step 3: Deramp secondary
    # -------------------------------------------------------------------------
    sec_der = deramp_slc(sec_slc, sec_burst)
    _save_slc_npz(sec_der, deramped_sec_path)

    # -------------------------------------------------------------------------
    # Step 4: Load Geo2Rdr offsets
    # -------------------------------------------------------------------------
    range_off_path = Path(geo2rdr_offsets.range_off_path)
    azimuth_off_path = Path(geo2rdr_offsets.azimuth_off_path)

    if not range_off_path.exists():
        raise FileNotFoundError(f"range.off not found: {range_off_path}")
    if not azimuth_off_path.exists():
        raise FileNotFoundError(f"azimuth.off not found: {azimuth_off_path}")

    with np.load(range_off_path) as npz:
        range_off = npz["data"].astype(np.float64)
    with np.load(azimuth_off_path) as npz:
        azimuth_off = npz["data"].astype(np.float64)

    if range_off.shape != out_shape:
        raise ValueError(
            f"range.off shape {range_off.shape} != expected {out_shape}"
        )
    if azimuth_off.shape != out_shape:
        raise ValueError(
            f"azimuth.off shape {azimuth_off.shape} != expected {out_shape}"
        )

    log.debug(
        "Offsets loaded  range=[%.4f, %.4f]  azimuth=[%.4f, %.4f]  shape=%s",
        float(np.nanmin(range_off)), float(np.nanmax(range_off)),
        float(np.nanmin(azimuth_off)), float(np.nanmax(azimuth_off)),
        range_off.shape,
    )

    # -------------------------------------------------------------------------
    # Step 5: Resample deramped secondary onto reference grid
    # -------------------------------------------------------------------------
    # Offsets are in PIXEL units.  We apply them as:
    #   src_row = target_row + azimuth_off
    #   src_col = target_col + range_off
    # Using map_coordinates with order=1 (bilinear).
    sec_der_resampled = _resample_sliding_window(
        sec_der,
        offset_rows=azimuth_off,
        offset_cols=range_off,
    )
    # Ensure output is complex64
    sec_der_resampled = sec_der_resampled.astype(np.complex64)

    # -------------------------------------------------------------------------
    # Step 6: Reramp onto the reference burst TOPS carrier
    # -------------------------------------------------------------------------
    resampled_sec = reramp_slc(sec_der_resampled, ref_burst)
    resampled_sec = resampled_sec.astype(np.complex64)

    # -------------------------------------------------------------------------
    # Step 7: Save outputs
    # -------------------------------------------------------------------------
    _save_slc_npz(resampled_sec, resampled_sec_path)

    log.info(
        "Coarse registration complete  ref=%s[%d]  sec=%s[%d]  "
        "resampled_sec=%s  dtype=%s",
        ref_burst.identity.swath, ref_burst.identity.burst_index,
        sec_burst.identity.swath, sec_burst.identity.burst_index,
        resampled_sec_path,
        resampled_sec.dtype,
    )


# ---------------------------------------------------------------------------
# Fine resampling with ESD timing + range coregistration
# ---------------------------------------------------------------------------

def fine_resample_with_timing(
    ref_slc: np.ndarray,
    sec_slc: np.ndarray,
    ref_burst: BurstRadarGrid,
    sec_burst: BurstRadarGrid,
    coarse_offsets: Geo2RdrOffsets,
    timing_correction: TimingCorrection | None,
    range_coreg_estimate: RangeCoregEstimate | None,
    *,
    work_dir: Path | str,
    fine_resampled_path: Path | str,
) -> None:
    """Fine coregistration of a secondary SLC onto a reference SLC after coarse
    registration, applying ESD timing and range coregistration corrections.

    This function is called after ``run_coarse_registration`` to refine the
    coregistration using:
    - ESD-derived azimuth timing correction (added to azimuth offsets)
    - Range coregistration estimate (added to range offsets)

    Algorithm
    ---------
    1. Deramp secondary SLC: ``sec_der = deramp_slc(sec_slc, sec_burst)``.
    2. Load coarse Geo2Rdr range and azimuth offset grids from disk.
    3. Apply ESD timing correction: ``azimuth_off += timing_correction.secondary_timing_pixels``.
    4. Apply range coreg correction: ``range_off += range_coreg_estimate.median_range_offset``.
    5. Resample deramped secondary using corrected offsets:
       ``sec_der_fine = _resample_sliding_window(sec_der, azimuth_off, range_off)``.
    6. Reramp the fine-resampled secondary onto the reference burst TOPS carrier:
       ``fine_resampled_sec = reramp_slc(sec_der_fine, ref_burst)``.
    7. Save the fine resampled secondary SLC to ``fine_resampled_path``.

    Parameters
    ----------
    ref_slc : np.ndarray
        Reference SLC (complex64 array).
    sec_slc : np.ndarray
        Secondary SLC (complex64 array).
    ref_burst : BurstRadarGrid
        Reference burst metadata.
    sec_burst : BurstRadarGrid
        Secondary burst metadata.
    coarse_offsets : Geo2RdrOffsets
        Geo2Rdr offsets from the coarse registration step.
        Contains paths to ``range.off`` and ``azimuth.off`` numpy arrays.
    timing_correction : TimingCorrection | None
        ESD-derived timing correction. If None, no timing correction is applied.
        ``timing_correction.secondary_timing_pixels`` is added to the azimuth offsets.
    range_coreg_estimate : RangeCoregEstimate | None
        Range coregistration estimate from overlap interferogram. If None,
        no range correction is applied.
        ``range_coreg_estimate.median_range_offset`` is added to the range offsets.
    work_dir : Path | str
        Working directory.  Created if it does not exist.
    fine_resampled_path : Path | str
        Output path for the fine resampled secondary SLC (.npz, complex64).

    Raises
    ------
    FileNotFoundError
        If the range or azimuth offset files from coarse registration are not found.
    ValueError
        If offset array shapes do not match the reference burst valid window shape.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    out_shape = (
        ref_burst.valid_window.num_lines,
        ref_burst.valid_window.num_samples,
    )

    log.info(
        "fine_resample_with_timing  ref=%s[%d]  sec=%s[%d]  shape=%s  "
        "timing=%s  range_coreg=%s",
        ref_burst.identity.swath, ref_burst.identity.burst_index,
        sec_burst.identity.swath, sec_burst.identity.burst_index,
        out_shape,
        timing_correction is not None,
        range_coreg_estimate is not None,
    )

    # -------------------------------------------------------------------------
    # Step 1: Deramp secondary
    # -------------------------------------------------------------------------
    sec_der = deramp_slc(sec_slc, sec_burst)

    # -------------------------------------------------------------------------
    # Step 2: Load coarse offsets from disk
    # -------------------------------------------------------------------------
    range_off_path = Path(coarse_offsets.range_off_path)
    azimuth_off_path = Path(coarse_offsets.azimuth_off_path)

    if not range_off_path.exists():
        raise FileNotFoundError(f"range.off not found: {range_off_path}")
    if not azimuth_off_path.exists():
        raise FileNotFoundError(f"azimuth.off not found: {azimuth_off_path}")

    with np.load(range_off_path) as npz:
        range_off = npz["data"].astype(np.float64).copy()
    with np.load(azimuth_off_path) as npz:
        azimuth_off = npz["data"].astype(np.float64).copy()

    if range_off.shape != out_shape:
        raise ValueError(
            f"range.off shape {range_off.shape} != expected {out_shape}"
        )
    if azimuth_off.shape != out_shape:
        raise ValueError(
            f"azimuth.off shape {azimuth_off.shape} != expected {out_shape}"
        )

    # -------------------------------------------------------------------------
    # Step 3: Apply ESD timing correction to azimuth offsets
    # -------------------------------------------------------------------------
    if timing_correction is not None:
        azimuth_off = azimuth_off + timing_correction.secondary_timing_pixels
        log.debug(
            "Applied ESD timing correction: +%.4f pixels",
            timing_correction.secondary_timing_pixels,
        )

    # -------------------------------------------------------------------------
    # Step 4: Apply range coregistration correction
    # -------------------------------------------------------------------------
    if range_coreg_estimate is not None:
        range_off = range_off + range_coreg_estimate.median_range_offset
        log.debug(
            "Applied range coreg correction: +%.4f pixels",
            range_coreg_estimate.median_range_offset,
        )

    # -------------------------------------------------------------------------
    # Step 5: Resample deramped secondary with corrected offsets
    # -------------------------------------------------------------------------
    sec_der_fine = _resample_sliding_window(
        sec_der,
        offset_rows=azimuth_off,
        offset_cols=range_off,
    )
    sec_der_fine = sec_der_fine.astype(np.complex64)

    # -------------------------------------------------------------------------
    # Step 6: Reramp onto the reference burst TOPS carrier
    # -------------------------------------------------------------------------
    fine_resampled_sec = reramp_slc(sec_der_fine, ref_burst)
    fine_resampled_sec = fine_resampled_sec.astype(np.complex64)

    # -------------------------------------------------------------------------
    # Step 7: Save the fine resampled secondary
    # -------------------------------------------------------------------------
    _save_slc_npz(fine_resampled_sec, fine_resampled_path)

    log.info(
        "Fine resampling complete  ref=%s[%d]  sec=%s[%d]  "
        "fine_resampled=%s  dtype=%s",
        ref_burst.identity.swath, ref_burst.identity.burst_index,
        sec_burst.identity.swath, sec_burst.identity.burst_index,
        fine_resampled_path,
        fine_resampled_sec.dtype,
    )


# ---------------------------------------------------------------------------
# Goldstein phase filtering
# ---------------------------------------------------------------------------

def filter_ifg(
    ifg: np.ndarray, coherence: np.ndarray, *, alpha: float = 0.5
) -> np.ndarray:
    """Apply Goldstein phase filtering to an interferogram.

    The filter adapts the strength of filtering to local coherence, smoothing
    areas of high coherence while preserving detail in decorrelated regions.

    Algorithm
    ---------
    1. Compute local intensity ``I = |ifg|`` for pixels where ``coherence > 0.3``.
    2. Apply boxcar multi-looking to ``I`` with a window of 8×8 pixels.
    3. Compute filter power ``F = I_ml ** alpha`` where ``I_ml`` is the
       multi-looked intensity.
    4. Apply the filter: ``ifg_filtered = ifg * F / mean(F)``.

    Parameters
    ----------
    ifg : np.ndarray
        2-D complex interferogram (complex64 or complex128).
    coherence : np.ndarray
        2-D float array of coherence values (same shape as ``ifg``).
        Values should be in the range [0, 1].
    alpha : float, optional
        Filter exponent controlling the strength of filtering.
        - ``alpha = 0.0``: no filtering (identity operation).
        - ``alpha = 1.0``: maximum filtering.
        - Default: 0.5.

    Returns
    -------
    np.ndarray
        Filtered interferogram (complex64).  Pixels with ``coherence <= 0.3``
        are set to zero.

    Raises
    ------
    ValueError
        If ``ifg`` and ``coherence`` have different shapes.
    ValueError
        If ``alpha`` is outside the range [0, 1].
    """
    if ifg.shape != coherence.shape:
        raise ValueError(
            f"ifg.shape {ifg.shape} != coherence.shape {coherence.shape}"
        )
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    # --- Identity case: no filtering ----------------------------------------
    if alpha == 0.0:
        return ifg.astype(np.complex64)

    # --- Compute intensity for coherent pixels --------------------------------
    intensity = np.abs(ifg).astype(np.float32)

    # Mask: only filter coherent regions (coherence > 0.3)
    mask = coherence > 0.3
    intensity_filtered = np.where(mask, intensity, 0.0)

    # --- Boxcar multi-looking (8×8 window) ---------------------------------
    # Simple explicit convolution (no scipy dependency for this step)
    window = 8
    from scipy.ndimage import uniform_filter

    intensity_ml = uniform_filter(
        intensity_filtered,
        size=window,
        mode="constant",
        cval=0.0,
    )

    # Normalize by the number of valid (non-zero) pixels in the window
    # for edge handling similar to the mean filter
    count_ml = uniform_filter(
        mask.astype(np.float32),
        size=window,
        mode="constant",
        cval=0.0,
    )
    # Avoid division by zero
    count_ml = np.maximum(count_ml, 1e-8)
    intensity_ml = intensity_ml / count_ml

    # --- Compute filter strength ----------------------------------------------
    filter_power = np.power(intensity_ml, alpha, where=intensity_ml > 0)
    filter_power = np.maximum(filter_power, 0.0)  # ensure non-negative

    # --- Normalize and apply filter ------------------------------------------
    mean_power = float(np.mean(filter_power))
    if mean_power == 0.0:
        return np.full_like(ifg, 0.0, dtype=np.complex64)

    filter_factor = filter_power / mean_power

    # Apply filter and zero out decorrelated pixels
    ifg_filtered = ifg.astype(np.complex64) * filter_factor.astype(np.complex64)
    ifg_filtered = np.where(mask, ifg_filtered, 0.0 + 0.0j)

    return ifg_filtered.astype(np.complex64)
