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
    "_resample_sliding_window",
]

import logging
from pathlib import Path

import numpy as np

from scripts.tops_model import BurstRadarGrid, Geo2RdrOffsets
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
