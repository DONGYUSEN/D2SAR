from __future__ import annotations

import gc
import json
import os
import subprocess
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
from osgeo import gdal
from scipy import interpolate, ndimage, signal

GEO2RDR_OFFSET_NODATA = -999999.0
GEO2RDR_OFFSET_INVALID_LOW = -1.0e5


def _gdal_dtype_to_numpy(gdal_dtype: int) -> np.dtype:
    mapping = {
        gdal.GDT_Byte: np.uint8,
        gdal.GDT_Int16: np.int16,
        gdal.GDT_UInt16: np.uint16,
        gdal.GDT_Int32: np.int32,
        gdal.GDT_UInt32: np.uint32,
        gdal.GDT_Float32: np.float32,
        gdal.GDT_Float64: np.float64,
        gdal.GDT_CInt16: np.complex64,
        gdal.GDT_CInt32: np.complex64,
        gdal.GDT_CFloat32: np.complex64,
        gdal.GDT_CFloat64: np.complex128,
    }
    return np.dtype(mapping.get(gdal_dtype, np.float32))


def _read_band_array(
    band,
    xoff: int = 0,
    yoff: int = 0,
    xsize: int | None = None,
    ysize: int | None = None,
    *,
    dtype: np.dtype | type | None = None,
) -> np.ndarray:
    if xsize is None:
        xsize = band.XSize
    if ysize is None:
        ysize = band.YSize
    np_dtype = np.dtype(dtype) if dtype is not None else _gdal_dtype_to_numpy(band.DataType)
    if np.issubdtype(np_dtype, np.complexfloating):
        buf_type = gdal.GDT_CFloat32 if np_dtype == np.dtype(np.complex64) else gdal.GDT_CFloat64
    elif np_dtype == np.dtype(np.float64):
        buf_type = gdal.GDT_Float64
    elif np_dtype == np.dtype(np.uint8):
        buf_type = gdal.GDT_Byte
    else:
        buf_type = gdal.GDT_Float32
        np_dtype = np.dtype(np.float32)
    raw = band.ReadRaster(
        int(xoff),
        int(yoff),
        int(xsize),
        int(ysize),
        buf_xsize=int(xsize),
        buf_ysize=int(ysize),
        buf_type=buf_type,
    )
    if raw is None:
        raise RuntimeError("failed to read raster block")
    return np.frombuffer(raw, dtype=np_dtype).reshape(int(ysize), int(xsize)).copy()


def _polyfit2d(data: np.ndarray) -> list[list[float]]:
    if data is None:
        return [[0.0, 0.0], [0.0, 0.0]]
    arr = np.asarray(data)
    if arr.ndim != 2 or arr.size == 0:
        return [[0.0, 0.0], [0.0, 0.0]]
    rows, cols = arr.shape
    yy, xx = np.mgrid[0:rows, 0:cols]
    a = np.column_stack(
        [
            np.ones(rows * cols),
            xx.ravel(),
            yy.ravel(),
            (xx * yy).ravel(),
        ]
    )
    coeffs, *_ = np.linalg.lstsq(a, arr.ravel().astype(np.float64), rcond=None)
    return [
        [float(coeffs[0]), float(coeffs[1])],
        [float(coeffs[2]), float(coeffs[3])],
    ]


def _summarize_fit_quality(
    azimuth_offset: np.ndarray,
    range_offset: np.ndarray,
) -> dict:
    azimuth_offset = np.asarray(azimuth_offset, dtype=np.float32)
    range_offset = np.asarray(range_offset, dtype=np.float32)
    az_rms = float(np.sqrt(np.mean(np.square(azimuth_offset)))) if azimuth_offset.size else 0.0
    rg_rms = float(np.sqrt(np.mean(np.square(range_offset)))) if range_offset.size else 0.0
    max_abs = max(
        float(np.max(np.abs(azimuth_offset))) if azimuth_offset.size else 0.0,
        float(np.max(np.abs(range_offset))) if range_offset.size else 0.0,
    )
    retry_recommended = bool(max_abs >= 20.0 or az_rms >= 8.0 or rg_rms >= 8.0)
    return {
        "azimuth_rms": az_rms,
        "range_rms": rg_rms,
        "max_abs_offset": max_abs,
        "retry_recommended": retry_recommended,
    }


def _read_matching_raster(path: str | Path) -> np.ndarray:
    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"failed to open matching raster: {path}")
    try:
        arr = _read_band_array(ds.GetRasterBand(1))
    finally:
        ds = None
    if np.iscomplexobj(arr):
        arr = np.abs(arr)
    arr = np.asarray(arr, dtype=np.float32)
    arr[~np.isfinite(arr)] = 0.0
    return arr


def _geo2rdr_offset_valid_mask(arr: np.ndarray) -> np.ndarray:
    clean = np.asarray(arr)
    valid = np.isfinite(clean)
    valid &= clean != GEO2RDR_OFFSET_NODATA
    valid &= clean >= GEO2RDR_OFFSET_INVALID_LOW
    return valid


def _sanitize_geo2rdr_offset_array(arr: np.ndarray, *, replacement: float = 0.0) -> np.ndarray:
    """Apply ISCE2-style Geo2Rdr offset validity rules before resample/flatten use."""
    clean = np.asarray(arr)
    valid = _geo2rdr_offset_valid_mask(clean)
    if np.all(valid):
        return clean
    clean = np.array(clean, copy=True)
    clean[~valid] = replacement
    return clean


def _estimate_offset_mean_from_raster(path: str | Path) -> float:
    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"failed to open offset raster: {path}")
    try:
        arr = _sanitize_geo2rdr_offset_array(
            np.asarray(_read_band_array(ds.GetRasterBand(1), dtype=np.float32), dtype=np.float32)
        )
    finally:
        ds = None
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise RuntimeError(f"offset raster has no finite values: {path}")
    return float(np.mean(finite, dtype=np.float64))


def _select_cpu_dense_match_plan(effective_resolution: float) -> dict:
    resolution = float(effective_resolution or 0.0)
    if resolution > 0.0 and resolution < 5.0:
        candidates = [
            {"window_size": (512, 512), "search_range": (128, 128)},
            {"window_size": (256, 256), "search_range": (64, 64)},
            {"window_size": (128, 128), "search_range": (64, 64)},
        ]
    elif resolution > 0.0 and resolution < 10.0:
        candidates = [
            {"window_size": (256, 256), "search_range": (64, 64)},
            {"window_size": (128, 128), "search_range": (64, 64)},
            {"window_size": (64, 64), "search_range": (32, 32)},
        ]
    else:
        candidates = [
            {"window_size": (128, 128), "search_range": (64, 64)},
            {"window_size": (64, 64), "search_range": (32, 32)},
        ]
    return {
        "gross_search_range": (3, 3),
        "candidates": candidates,
    }


def _phase_correlate_search_window(
    reference_patch: np.ndarray,
    secondary_patch: np.ndarray,
    search_range: tuple[int, int],
) -> tuple[float, float, float, float]:
    ref = np.asarray(reference_patch, dtype=np.float32)
    sec = np.asarray(secondary_patch, dtype=np.float32)
    search_down = int(search_range[0])
    search_across = int(search_range[1])
    ref_h, ref_w = ref.shape
    if sec.shape != (ref_h + 2 * search_down, ref_w + 2 * search_across):
        return 0.0, 0.0, 0.0, 0.0

    ref_std = float(np.std(ref))
    if ref_std < 1.0e-6:
        return 0.0, 0.0, 0.0, 0.0

    ref_norm = (ref - float(np.mean(ref))) / ref_std
    scores = np.full((2 * search_down + 1, 2 * search_across + 1), -np.inf, dtype=np.float32)
    for row_shift in range(-search_down, search_down + 1):
        row0 = row_shift + search_down
        row1 = row0 + ref_h
        for col_shift in range(-search_across, search_across + 1):
            col0 = col_shift + search_across
            col1 = col0 + ref_w
            sec_patch = sec[row0:row1, col0:col1]
            sec_std = float(np.std(sec_patch))
            if sec_std < 1.0e-6:
                continue
            sec_norm = (sec_patch - float(np.mean(sec_patch))) / sec_std
            scores[row_shift + search_down, col_shift + search_across] = float(
                np.mean(ref_norm * sec_norm)
            )

    finite_scores = scores[np.isfinite(scores)]
    if finite_scores.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    peak = np.unravel_index(int(np.nanargmax(scores)), scores.shape)
    peak_value = float(scores[peak])
    if not np.isfinite(peak_value):
        return 0.0, 0.0, 0.0, 0.0

    sorted_scores = np.sort(finite_scores)
    second_best = float(sorted_scores[-2]) if sorted_scores.size >= 2 else -1.0
    row_offset = float(peak[0] - search_down)
    col_offset = float(peak[1] - search_across)
    quality = max(0.0, peak_value * 100.0) + max(0.0, (peak_value - second_best) * 25.0)
    return row_offset, col_offset, quality, max(0.0, min(1.0, peak_value))


def _plan_matching_grid(
    *,
    rows: int,
    cols: int,
    window_size: tuple[int, int],
    search_range: tuple[int, int],
    skip: tuple[int, int],
    gross_offset: tuple[float, float] = (0.0, 0.0),
    max_windows: int = 4096,
    max_window_down: int = 60,
    max_window_across: int = 40,
) -> dict:
    gross_az = int(np.rint(float(gross_offset[0] if gross_offset else 0.0)))
    gross_rg = int(np.rint(float(gross_offset[1] if gross_offset else 0.0)))
    margin_az = 2 * int(search_range[0]) + int(window_size[0]) + 2 * int(abs(gross_az))
    margin_rg = 2 * int(search_range[1]) + int(window_size[1]) + 2 * int(abs(gross_rg))
    skip_down = max(1, int(skip[0]))
    skip_across = max(1, int(skip[1]))

    def _counts(curr_skip_down: int, curr_skip_across: int) -> tuple[int, int]:
        return (
            max(1, (int(rows) - margin_az) // max(1, int(curr_skip_down))),
            max(1, (int(cols) - margin_rg) // max(1, int(curr_skip_across))),
        )

    while True:
        number_window_down, number_window_across = _counts(skip_down, skip_across)
        if (
            number_window_down <= int(max_window_down)
            and number_window_across <= int(max_window_across)
            and number_window_down * number_window_across <= int(max_windows)
        ):
            break
        needs_down = number_window_down > int(max_window_down)
        needs_across = number_window_across > int(max_window_across)
        if needs_down and (number_window_down >= number_window_across or not needs_across):
            skip_down += max(1, skip[0])
            continue
        if needs_across:
            skip_across += max(1, skip[1])
            continue
        if number_window_down * number_window_across > int(max_windows):
            if number_window_down >= number_window_across:
                skip_down += max(1, skip[0])
            else:
                skip_across += max(1, skip[1])

    start_down = int(abs(gross_az)) + int(search_range[0])
    start_across = int(abs(gross_rg)) + int(search_range[1])
    return {
        "gross_az": gross_az,
        "gross_rg": gross_rg,
        "margin_az": int(margin_az),
        "margin_rg": int(margin_rg),
        "skip_down": int(skip_down),
        "skip_across": int(skip_across),
        "number_window_down": int(number_window_down),
        "number_window_across": int(number_window_across),
        "reference_start_pixel_down": int(start_down),
        "reference_start_pixel_across": int(start_across),
    }


def _run_cpu_dense_candidate(
    *,
    master: np.ndarray,
    slave: np.ndarray,
    rows: int,
    cols: int,
    cand_window: tuple[int, int],
    cand_search: tuple[int, int],
    skip: tuple[int, int],
    gross_az: float,
    gross_rg: float,
    max_windows: int,
    max_window_down: int,
    max_window_across: int,
    quality_threshold: float = 18.0,
    interpolate_dense: bool = True,
) -> dict:
    grid = _plan_matching_grid(
        rows=rows,
        cols=cols,
        window_size=cand_window,
        search_range=cand_search,
        skip=skip,
        gross_offset=(gross_az, gross_rg),
        max_windows=max_windows,
        max_window_down=max_window_down,
        max_window_across=max_window_across,
    )
    skip_down = int(grid["skip_down"])
    skip_across = int(grid["skip_across"])
    number_window_down = int(grid["number_window_down"])
    number_window_across = int(grid["number_window_across"])
    start_down = int(grid["reference_start_pixel_down"])
    start_across = int(grid["reference_start_pixel_across"])
    row_sparse = np.full((number_window_down, number_window_across), np.nan, dtype=np.float32)
    col_sparse = np.full((number_window_down, number_window_across), np.nan, dtype=np.float32)
    snr = np.full((number_window_down, number_window_across), np.nan, dtype=np.float32)
    correlation = np.full((number_window_down, number_window_across), np.nan, dtype=np.float32)
    covariance_az = np.full((number_window_down, number_window_across), np.nan, dtype=np.float32)
    covariance_rg = np.full((number_window_down, number_window_across), np.nan, dtype=np.float32)
    valid_points = 0
    rejected_low_quality = 0
    rejected_boundary = 0
    rejected_invalid = 0
    row_coords = (
        start_down
        + np.arange(number_window_down, dtype=np.float64) * skip_down
        + (float(cand_window[0]) / 2.0)
    )
    col_coords = (
        start_across
        + np.arange(number_window_across, dtype=np.float64) * skip_across
        + (float(cand_window[1]) / 2.0)
    )
    for row_idx in range(number_window_down):
        row0 = start_down + row_idx * skip_down
        row1 = row0 + int(cand_window[0])
        sr0 = row0 + int(np.rint(gross_az)) - int(cand_search[0])
        sr1 = row1 + int(np.rint(gross_az)) + int(cand_search[0])
        if sr0 < 0 or sr1 > rows:
            continue
        for col_idx in range(number_window_across):
            col0 = start_across + col_idx * skip_across
            col1 = col0 + int(cand_window[1])
            sc0 = col0 + int(np.rint(gross_rg)) - int(cand_search[1])
            sc1 = col1 + int(np.rint(gross_rg)) + int(cand_search[1])
            if sc0 < 0 or sc1 > cols:
                continue
            ref = master[row0:row1, col0:col1]
            sec = slave[sr0:sr1, sc0:sc1]
            match = _phase_correlate_search_window(ref, sec, cand_search)
            if len(match) >= 4:
                drow, dcol, peak_snr, peak_correlation = match
            else:
                drow, dcol, peak_snr = match
                peak_correlation = max(0.0, min(1.0, float(peak_snr) / 100.0))
            if peak_snr < quality_threshold:
                rejected_low_quality += 1
                continue
            if (
                abs(float(drow)) >= float(cand_search[0])
                or abs(float(dcol)) >= float(cand_search[1])
            ):
                rejected_boundary += 1
                continue
            if (not np.isfinite(drow)) or (not np.isfinite(dcol)):
                rejected_invalid += 1
                continue
            row_sparse[row_idx, col_idx] = drow
            col_sparse[row_idx, col_idx] = dcol
            snr[row_idx, col_idx] = peak_snr
            correlation[row_idx, col_idx] = peak_correlation
            covariance_az[row_idx, col_idx] = max(0.0, 1.0 - peak_snr / 100.0)
            covariance_rg[row_idx, col_idx] = max(0.0, 1.0 - peak_snr / 100.0)
            valid_points += 1

    diagnostics = {
        "engine": "cpu-template-search",
        "gross_offset": {"azimuth": float(gross_az), "range": float(gross_rg)},
        "window_size_height": int(cand_window[0]),
        "window_size_width": int(cand_window[1]),
        "search_range": [int(cand_search[0]), int(cand_search[1])],
        "number_window_down": int(number_window_down),
        "number_window_across": int(number_window_across),
        "reference_start_pixel_down": int(start_down),
        "reference_start_pixel_across": int(start_across),
        "skip_down": int(skip_down),
        "skip_across": int(skip_across),
        "quality_threshold": float(quality_threshold),
        "valid_points": int(valid_points),
        "candidate_points": int(number_window_down * number_window_across),
        "rejected_low_quality": int(rejected_low_quality),
        "rejected_boundary": int(rejected_boundary),
        "rejected_invalid": int(rejected_invalid),
        "row_coords": row_coords.tolist(),
        "col_coords": col_coords.tolist(),
        "row_sparse_shape": list(row_sparse.shape),
        "col_sparse_shape": list(col_sparse.shape),
        "row_sparse_stats": _summarize_numeric_array(row_sparse),
        "col_sparse_stats": _summarize_numeric_array(col_sparse),
        "snr_stats": _summarize_numeric_array(snr),
        "correlation_stats": _summarize_numeric_array(correlation),
        "covariance_az_stats": _summarize_numeric_array(covariance_az),
        "covariance_rg_stats": _summarize_numeric_array(covariance_rg),
    }
    min_valid_points = 1
    if valid_points < min_valid_points:
        diagnostics["status"] = "failed"
        diagnostics["reason"] = "insufficient_valid_points"
        diagnostics["min_valid_points"] = int(min_valid_points)
        return {
            "score": (
                -1,
                float(diagnostics["snr_stats"].get("median", 0.0) or 0.0) if diagnostics["snr_stats"] else 0.0,
                0.0,
                int(cand_window[0]),
            ),
            "row_offset": None,
            "col_offset": None,
            "details": {
                "metadata": diagnostics,
                "row_sparse": row_sparse,
                "col_sparse": col_sparse,
                "row_coords": row_coords,
                "col_coords": col_coords,
                "snr": snr,
                "correlation": correlation,
                "covariance_az": covariance_az,
                "covariance_rg": covariance_rg,
                "diagnostics": diagnostics,
            },
        }
    prepared, common_sparse_fit = _prepare_sparse_offsets_for_dense_model(
        {
            "metadata": diagnostics,
            "row_sparse": row_sparse,
            "col_sparse": col_sparse,
            "row_coords": row_coords,
            "col_coords": col_coords,
            "snr": snr,
            "correlation": correlation,
            "covariance_az": covariance_az,
            "covariance_rg": covariance_rg,
            "diagnostics": diagnostics,
        },
        out_shape=(rows, cols),
        min_points=min_valid_points,
    )
    diagnostics["common_sparse_fit"] = common_sparse_fit
    if not common_sparse_fit.get("success", False):
        diagnostics["status"] = "failed"
        diagnostics["reason"] = common_sparse_fit.get("reason")
        return {
            "score": (
                -1,
                float(diagnostics["snr_stats"].get("median", 0.0) or 0.0) if diagnostics["snr_stats"] else 0.0,
                0.0,
                int(cand_window[0]),
            ),
            "row_offset": None,
            "col_offset": None,
            "details": {
                "metadata": diagnostics,
                "row_sparse": row_sparse,
                "col_sparse": col_sparse,
                "row_coords": row_coords,
                "col_coords": col_coords,
                "snr": snr,
                "correlation": correlation,
                "covariance_az": covariance_az,
                "covariance_rg": covariance_rg,
                "diagnostics": diagnostics,
            },
        }

    row_offset = None if not interpolate_dense else np.asarray(prepared["row_offset"], dtype=np.float32)
    col_offset = None if not interpolate_dense else np.asarray(prepared["col_offset"], dtype=np.float32)
    diagnostics["status"] = "ok"
    diagnostics["reason"] = None
    prepared["diagnostics"] = diagnostics
    score = (
        int(common_sparse_fit["fit"]["final_inliers"]),
        float(diagnostics["snr_stats"].get("median", 0.0) or 0.0) if diagnostics["snr_stats"] else 0.0,
        -float(np.nanstd(prepared["row_sparse"]) + np.nanstd(prepared["col_sparse"])),
        int(cand_window[0]),
    )
    return {
        "score": score,
        "row_offset": row_offset,
        "col_offset": col_offset,
        "details": {
            "metadata": diagnostics,
            "row_sparse": np.asarray(prepared["row_sparse"], dtype=np.float32),
            "col_sparse": np.asarray(prepared["col_sparse"], dtype=np.float32),
            "row_coords": row_coords,
            "col_coords": col_coords,
            "snr": None if prepared.get("snr") is None else np.asarray(prepared["snr"], dtype=np.float32),
            "correlation": np.asarray(prepared["correlation"], dtype=np.float32),
            "covariance_az": covariance_az,
            "covariance_rg": covariance_rg,
            "row_offset": row_offset,
            "col_offset": col_offset,
            "diagnostics": diagnostics,
        },
    }


def run_cpu_dense_offsets(
    *,
    master_slc_path: str,
    slave_slc_path: str,
    output_dir: str | Path,
    window_size: tuple[int, int] = (64, 64),
    search_range: tuple[int, int] = (20, 20),
    skip: tuple[int, int] = (32, 32),
    max_windows: int = 4096,
    gross_offset: tuple[float, float] = (0.0, 0.0),
    window_candidates: list[dict] | None = None,
    return_details: bool = False,
) -> tuple[np.ndarray | None, np.ndarray | None] | tuple[np.ndarray | None, np.ndarray | None, dict | None]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    master_local = master_slc_path
    slave_local = slave_slc_path
    master_cleanup = None
    slave_cleanup = None
    try:
        master_local, master_cleanup = _ensure_local_tiff(master_slc_path)
        slave_local, slave_cleanup = _ensure_local_tiff(slave_slc_path)
        if master_local is None or slave_local is None:
            failure = {
                "metadata": None,
                "row_sparse": np.zeros((0, 0), dtype=np.float32),
                "col_sparse": np.zeros((0, 0), dtype=np.float32),
                "row_coords": np.zeros((0,), dtype=np.float64),
                "col_coords": np.zeros((0,), dtype=np.float64),
                "snr": np.zeros((0, 0), dtype=np.float32),
                "correlation": np.zeros((0, 0), dtype=np.float32),
                "covariance_az": np.zeros((0, 0), dtype=np.float32),
                "covariance_rg": np.zeros((0, 0), dtype=np.float32),
                "diagnostics": {
                    "engine": "cpu-template-search",
                    "status": "failed",
                    "reason": "unavailable_input",
                    "valid_points": 0,
                },
            }
            return (None, None, failure) if return_details else (None, None)
        master = _read_matching_raster(master_local)
        slave = _read_matching_raster(slave_local)
        if master.shape != slave.shape:
            raise RuntimeError(
                f"dense match shape mismatch: master={master.shape} slave={slave.shape}"
            )
        rows, cols = master.shape
        candidates = list(window_candidates or [{"window_size": window_size, "search_range": search_range}])
        best_result = None
        best_score = None
        evaluated_candidates = []
        gross_az = float(gross_offset[0] if gross_offset else 0.0)
        gross_rg = float(gross_offset[1] if gross_offset else 0.0)
        selection = {
            "strategy": "staged-preview" if len(candidates) > 1 else "single-candidate",
            "preview_max_window_down": 8,
            "preview_max_window_across": 8,
            "selected_index": None,
            "selected_window_size": None,
            "selected_search_range": None,
        }

        preview_results = []
        for candidate_idx, candidate in enumerate(candidates):
            cand_window = tuple(candidate.get("window_size", window_size))
            cand_search = tuple(candidate.get("search_range", search_range))
            preview = _run_cpu_dense_candidate(
                master=master,
                slave=slave,
                rows=rows,
                cols=cols,
                cand_window=cand_window,
                cand_search=cand_search,
                skip=skip,
                gross_az=gross_az,
                gross_rg=gross_rg,
                max_windows=64,
                max_window_down=8,
                max_window_across=8,
                interpolate_dense=False,
            )
            preview["candidate_index"] = int(candidate_idx)
            preview_results.append(preview)
            preview_diag = preview["details"]["diagnostics"]
            preview_diag["candidate_index"] = int(candidate_idx)
            preview_diag["evaluation_stage"] = "preview"
            evaluated_candidates.append(preview_diag)
            if preview["details"]["diagnostics"].get("status") != "ok":
                continue
            if best_result is None or preview["score"] > best_score:
                best_score = preview["score"]
                best_result = preview

        if best_result is not None:
            selected_index = int(best_result["candidate_index"])
            selected = candidates[selected_index]
            selected_window = tuple(selected.get("window_size", window_size))
            selected_search = tuple(selected.get("search_range", search_range))
            selection.update(
                {
                    "selected_index": selected_index,
                    "selected_window_size": [int(selected_window[0]), int(selected_window[1])],
                    "selected_search_range": [int(selected_search[0]), int(selected_search[1])],
                }
            )
            final = _run_cpu_dense_candidate(
                master=master,
                slave=slave,
                rows=rows,
                cols=cols,
                cand_window=selected_window,
                cand_search=selected_search,
                skip=skip,
                gross_az=gross_az,
                gross_rg=gross_rg,
                max_windows=min(int(max_windows), 2400),
                max_window_down=60,
                max_window_across=40,
                interpolate_dense=True,
            )
            final["details"]["diagnostics"]["candidate_index"] = selected_index
            final["details"]["diagnostics"]["evaluation_stage"] = "final"
            final["details"]["diagnostics"]["candidate_selection"] = selection
            if final["details"]["diagnostics"].get("status") == "ok":
                best_score = final["score"]
                best_result = final
                evaluated_candidates.append(final["details"]["diagnostics"])
            else:
                evaluated_candidates.append(final["details"]["diagnostics"])
                best_result = None

        if best_result is None:
            failure = {
                "metadata": None,
                "row_sparse": np.zeros((0, 0), dtype=np.float32),
                "col_sparse": np.zeros((0, 0), dtype=np.float32),
                "row_coords": np.zeros((0,), dtype=np.float64),
                "col_coords": np.zeros((0,), dtype=np.float64),
                "snr": np.zeros((0, 0), dtype=np.float32),
                "correlation": np.zeros((0, 0), dtype=np.float32),
                "covariance_az": np.zeros((0, 0), dtype=np.float32),
                "covariance_rg": np.zeros((0, 0), dtype=np.float32),
                "diagnostics": {
                    "engine": "cpu-template-search",
                    "status": "failed",
                    "reason": "no_valid_candidate",
                    "gross_offset": {"azimuth": gross_az, "range": gross_rg},
                    "candidate_evaluations": evaluated_candidates,
                    "candidate_selection": selection,
                    "valid_points": 0,
                },
            }
            return (None, None, failure) if return_details else (None, None)

        row_offset = best_result["row_offset"]
        col_offset = best_result["col_offset"]
        details = best_result["details"]
        details["diagnostics"]["candidate_evaluations"] = evaluated_candidates
        np.save(out_dir / "dense_row_offsets.npy", row_offset.astype(np.float32))
        np.save(out_dir / "dense_col_offsets.npy", col_offset.astype(np.float32))
        if return_details:
            return row_offset, col_offset, details
        return row_offset, col_offset
    except Exception:
        _write_ampcor_log(out_dir / "cpu_dense_match.exception.log", traceback.format_exc())
        failure = {
            "metadata": None,
            "row_sparse": np.zeros((0, 0), dtype=np.float32),
            "col_sparse": np.zeros((0, 0), dtype=np.float32),
            "row_coords": np.zeros((0,), dtype=np.float64),
            "col_coords": np.zeros((0,), dtype=np.float64),
            "snr": np.zeros((0, 0), dtype=np.float32),
            "correlation": np.zeros((0, 0), dtype=np.float32),
            "covariance_az": np.zeros((0, 0), dtype=np.float32),
            "covariance_rg": np.zeros((0, 0), dtype=np.float32),
            "diagnostics": {
                "engine": "cpu-template-search",
                "status": "failed",
                "reason": "exception",
                "valid_points": 0,
            },
        }
        return (None, None, failure) if return_details else (None, None)
    finally:
        if master_cleanup is not None:
            master_cleanup()
        if slave_cleanup is not None:
            slave_cleanup()


def _write_offset_raster(
    path: Path,
    data: np.ndarray | None,
    *,
    dtype: int | None = None,
) -> str:
    """Write offset raster. If data is None, creates a 1x1 zero raster (legacy/caller must handle size mismatch)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if dtype is None:
        dtype = gdal.GDT_Float32
        if data is not None and np.asarray(data).dtype == np.dtype(np.float64):
            dtype = gdal.GDT_Float64
    np_dtype = np.float64 if dtype == gdal.GDT_Float64 else np.float32
    if data is None:
        ds = gdal.GetDriverByName("GTiff").Create(str(path), 1, 1, 1, dtype)
        ds.GetRasterBand(1).WriteRaster(
            0,
            0,
            1,
            1,
            np.zeros((1, 1), dtype=np_dtype).tobytes(),
            buf_xsize=1,
            buf_ysize=1,
            buf_type=dtype,
        )
        ds = None
        return str(path)
    rows, cols = data.shape
    ds = gdal.GetDriverByName("GTiff").Create(
        str(path), cols, rows, 1, dtype, options=["COMPRESS=LZW", "TILED=YES"]
    )
    band = ds.GetRasterBand(1)
    block_rows = 512
    for row0 in range(0, rows, block_rows):
        nrows = min(block_rows, rows - row0)
        block = np.ascontiguousarray(data[row0 : row0 + nrows, :], dtype=np_dtype)
        band.WriteRaster(
            0,
            row0,
            cols,
            nrows,
            block.tobytes(),
            buf_xsize=cols,
            buf_ysize=nrows,
            buf_type=dtype,
        )
    band.FlushCache()
    ds = None
    return str(path)


def _write_zero_offset_rasters(
    path: Path, width: int, length: int
) -> tuple[str, str]:
    """Write zero range/azimuth offset rasters matching the given image dimensions.

    Returns (rg_offset_path, az_offset_path).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    zero_rg = np.zeros((length, width), dtype=np.float32)
    zero_az = np.zeros((length, width), dtype=np.float32)
    rg_path = _write_offset_raster(path.parent / "range.off.tif", zero_rg)
    az_path = _write_offset_raster(path.parent / "azimuth.off.tif", zero_az)
    return rg_path, az_path


def _raster_shape(path: str | Path) -> tuple[int, int] | None:
    try:
        ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    except Exception:
        return None
    if ds is None:
        return None
    rows = ds.RasterYSize
    cols = ds.RasterXSize
    ds = None
    if rows <= 0 or cols <= 0:
        return None
    return rows, cols


def _load_offset_dataset_for_resample(
    path: str | Path,
    out_shape: tuple[int, int],
    *,
    replacement: float = 0.0,
):
    """Load an offset dataset for ISCE3 resample.

    Supports either:
    - GDAL-readable single-band rasters
    - raw float64 binary rasters matching ISCE3 `range.off` / `azimuth.off`
    """
    offset_path = Path(path)
    rows, cols = out_shape

    expected_bytes = rows * cols * np.dtype(np.float64).itemsize
    if offset_path.is_file() and offset_path.stat().st_size == expected_bytes:
        arr = np.memmap(offset_path, dtype=np.float64, mode="r", shape=out_shape)
        return _sanitize_geo2rdr_offset_array(arr, replacement=replacement)

    ds = gdal.Open(str(offset_path), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"unable to open offset dataset: {offset_path}")
    try:
        if ds.RasterYSize != rows or ds.RasterXSize != cols:
            raise RuntimeError(
                f"offset shape mismatch for {offset_path}: "
                f"expected {out_shape}, got {(ds.RasterYSize, ds.RasterXSize)}"
            )
        return _sanitize_geo2rdr_offset_array(
            np.array(_read_band_array(ds.GetRasterBand(1), dtype=np.float64), dtype=np.float64),
            replacement=replacement,
        )
    finally:
        ds = None


def _load_offset_dataset_with_valid_mask(
    path: str | Path,
    out_shape: tuple[int, int],
    *,
    replacement: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Load an offset raster and return sanitized values plus original valid mask."""
    offset_path = Path(path)
    rows, cols = out_shape
    expected_bytes = rows * cols * np.dtype(np.float64).itemsize
    if offset_path.is_file() and offset_path.stat().st_size == expected_bytes:
        arr = np.memmap(offset_path, dtype=np.float64, mode="r", shape=out_shape)
        valid = _geo2rdr_offset_valid_mask(arr)
        clean = np.array(arr, dtype=np.float32, copy=True)
        clean[~valid] = replacement
        return clean, valid

    ds = gdal.Open(str(offset_path), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"unable to open offset dataset: {offset_path}")
    try:
        if ds.RasterYSize != rows or ds.RasterXSize != cols:
            raise RuntimeError(
                f"offset shape mismatch for {offset_path}: "
                f"expected {out_shape}, got {(ds.RasterYSize, ds.RasterXSize)}"
            )
        arr = np.asarray(_read_band_array(ds.GetRasterBand(1), dtype=np.float64), dtype=np.float64)
        valid = _geo2rdr_offset_valid_mask(arr)
        clean = np.array(arr, dtype=np.float32, copy=True)
        clean[~valid] = replacement
        return clean, valid
    finally:
        ds = None


def _write_varying_gross_offset_file(
    *,
    range_offset_path: str | Path,
    azimuth_offset_path: str | Path,
    output_path: str | Path,
    full_shape: tuple[int, int],
    window_size: tuple[int, int] = (64, 64),
    search_range: tuple[int, int] = (20, 20),
    skip: tuple[int, int] = (32, 32),
) -> dict:
    rows, cols = full_shape
    range_offsets = np.asarray(
        _load_offset_dataset_for_resample(range_offset_path, (rows, cols)),
        dtype=np.float64,
    )
    azimuth_offsets = np.asarray(
        _load_offset_dataset_for_resample(azimuth_offset_path, (rows, cols)),
        dtype=np.float64,
    )

    margin = int(
        np.ceil(
            max(
                float(np.nanmax(np.abs(range_offsets))),
                float(np.nanmax(np.abs(azimuth_offsets))),
            )
        )
    )
    start_down = margin + int(search_range[0])
    start_across = margin + int(search_range[1])
    margin_rg = 2 * margin + 2 * int(search_range[1]) + int(window_size[1])
    margin_az = 2 * margin + 2 * int(search_range[0]) + int(window_size[0])
    number_window_down = max(1, (rows - margin_az) // max(1, int(skip[0])))
    number_window_across = max(1, (cols - margin_rg) // max(1, int(skip[1])))

    row_coords = start_down + np.arange(number_window_down, dtype=np.int32) * int(skip[0])
    col_coords = start_across + np.arange(number_window_across, dtype=np.int32) * int(skip[1])
    row_coords = np.clip(row_coords, 0, rows - 1)
    col_coords = np.clip(col_coords, 0, cols - 1)

    gross = np.empty((number_window_down * number_window_across, 2), dtype=np.int32)
    idx = 0
    for row in row_coords:
        for col in col_coords:
            gross[idx, 0] = int(np.rint(azimuth_offsets[int(row), int(col)]))
            gross[idx, 1] = int(np.rint(range_offsets[int(row), int(col)]))
            idx += 1

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gross.tofile(output_path)
    return {
        "gross_offset_filepath": str(output_path),
        "number_window_down": int(number_window_down),
        "number_window_across": int(number_window_across),
        "reference_start_pixel_down": int(start_down),
        "reference_start_pixel_across": int(start_across),
        "skip_down": int(skip[0]),
        "skip_across": int(skip[1]),
        "margin": int(margin),
    }


def _create_empty_dataset(
    filename: str | Path,
    width: int,
    length: int,
    bands: int,
    dtype,
    *,
    interleave: str = "bip",
    file_type: str = "ENVI",
) -> None:
    driver = gdal.GetDriverByName(file_type)
    driver.Create(
        str(filename),
        xsize=width,
        ysize=length,
        bands=bands,
        eType=dtype,
        options=[f"INTERLEAVE={interleave}"],
    )


def _allocate_raw_bip_float32(
    path: str | Path,
    length: int,
    width: int,
    bands: int,
) -> None:
    """Preallocate a raw float32 BIP file for PyCuAmpcor outputs.

    PyCuAmpcor writes directly into flat binary files. Using raw preallocation
    avoids relying on optional GDAL drivers such as ENVI, which are not always
    available in the runtime container.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mm = np.memmap(path, dtype=np.float32, mode="w+", shape=(length, width, bands))
    mm[:] = 0.0
    mm.flush()
    del mm


def _write_ampcor_log(path: str | Path, text: str | None) -> None:
    Path(path).write_text("" if text is None else text, encoding="utf-8")


def _interpolate_sparse_offsets(
    sparse_offsets: np.ndarray,
    row_coords: np.ndarray,
    col_coords: np.ndarray,
    out_shape: tuple[int, int],
) -> np.ndarray:
    out_rows, out_cols = out_shape
    if sparse_offsets.size == 0:
        return np.zeros(out_shape, dtype=np.float32)

    target_rows = np.arange(out_rows, dtype=np.float64)
    target_cols = np.arange(out_cols, dtype=np.float64)
    row_coords = np.asarray(row_coords, dtype=np.float64)
    col_coords = np.asarray(col_coords, dtype=np.float64)

    if sparse_offsets.shape[1] == 1:
        rowwise = np.repeat(sparse_offsets[:, :1], out_cols, axis=1)
    else:
        rowwise = np.empty((sparse_offsets.shape[0], out_cols), dtype=np.float32)
        for idx in range(sparse_offsets.shape[0]):
            rowwise[idx, :] = np.interp(
                target_cols,
                col_coords,
                sparse_offsets[idx, :],
                left=float(sparse_offsets[idx, 0]),
                right=float(sparse_offsets[idx, -1]),
            ).astype(np.float32)

    if sparse_offsets.shape[0] == 1:
        return np.repeat(rowwise[:1, :], out_rows, axis=0).astype(np.float32)

    dense = np.empty(out_shape, dtype=np.float32)
    for col_idx in range(out_cols):
        dense[:, col_idx] = np.interp(
            target_rows,
            row_coords,
            rowwise[:, col_idx],
            left=float(rowwise[0, col_idx]),
            right=float(rowwise[-1, col_idx]),
        ).astype(np.float32)
    return dense


def _design_matrix(
    rows_norm: np.ndarray,
    cols_norm: np.ndarray,
    *,
    model: str = "quadratic",
) -> np.ndarray:
    terms = [
        np.ones_like(rows_norm),
    ]
    if model in {"plane", "bilinear", "quadratic"}:
        terms.extend([rows_norm, cols_norm])
    if model in {"bilinear", "quadratic"}:
        terms.append(rows_norm * cols_norm)
    if model == "quadratic":
        terms.extend([rows_norm * rows_norm, cols_norm * cols_norm])
    return np.column_stack(terms)


def _normalize_dense_fit_model(model: str | None) -> str:
    normalized = str(model or "").strip().lower()
    if normalized not in {"auto", "constant", "plane", "bilinear", "quadratic"}:
        return "plane"
    return normalized


def _min_points_for_sparse_model(model: str) -> int:
    return {
        "constant": 1,
        "plane": 3,
        "bilinear": 4,
        "quadratic": 6,
    }.get(str(model), 3)


def _candidate_sparse_fit_models(requested_model: str) -> list[str]:
    if requested_model == "auto":
        return ["quadratic", "bilinear", "plane", "constant"]
    if requested_model == "quadratic":
        return ["quadratic", "bilinear", "plane", "constant"]
    if requested_model == "bilinear":
        return ["bilinear", "plane", "constant"]
    if requested_model == "plane":
        return ["plane", "constant"]
    return ["constant"]


def _evaluate_offset_surface(
    coeffs: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    row_mean: float,
    row_std: float,
    col_mean: float,
    col_std: float,
    model: str,
) -> np.ndarray:
    rows_norm = (np.asarray(rows, dtype=np.float64) - float(row_mean)) / float(row_std)
    cols_norm = (np.asarray(cols, dtype=np.float64) - float(col_mean)) / float(col_std)
    return _design_matrix(rows_norm, cols_norm, model=model) @ np.asarray(coeffs, dtype=np.float64)


def _choose_sparse_fit_model(point_count: int) -> tuple[str, int, dict]:
    requested_model = _normalize_dense_fit_model(os.environ.get("D2SAR_DENSE_FIT_MODEL", "plane"))
    return _choose_sparse_fit_model_for_request(point_count, requested_model)


def _choose_sparse_fit_model_for_request(
    point_count: int,
    requested_model: str,
) -> tuple[str, int, dict]:
    requested_model = _normalize_dense_fit_model(requested_model)
    candidates = _candidate_sparse_fit_models(requested_model)
    points = int(point_count)
    for candidate in candidates:
        min_points = _min_points_for_sparse_model(candidate)
        if points >= int(min_points):
            return candidate, int(min_points), {
                "requested_model": requested_model,
                "candidate_models": candidates,
                "selected_model": candidate,
                "fallback_used": bool(candidate != requested_model),
            }
    selected = candidates[-1]
    return selected, int(_min_points_for_sparse_model(selected)), {
        "requested_model": requested_model,
        "candidate_models": candidates,
        "selected_model": selected,
        "fallback_used": bool(selected != requested_model),
    }


def _iterative_sparse_fit(
    *,
    rows: np.ndarray,
    cols: np.ndarray,
    az: np.ndarray,
    rg: np.ndarray,
    model: str,
    min_points: int,
    iterations: int,
    initial_mask: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray, dict]:
    rows_mean = float(np.mean(rows))
    cols_mean = float(np.mean(cols))
    rows_std = max(float(np.std(rows)), 1.0)
    cols_std = max(float(np.std(cols)), 1.0)
    A = _design_matrix((rows - rows_mean) / rows_std, (cols - cols_mean) / cols_std, model=model)
    base_mask = np.ones(rows.shape[0], dtype=bool)
    if initial_mask is not None:
        base_mask &= np.asarray(initial_mask, dtype=bool)
    mask = base_mask.copy()

    if int(np.count_nonzero(mask)) < int(min_points):
        return None, None, np.zeros(rows.shape[0], dtype=bool), {
            "success": False,
            "reason": "insufficient_points_for_initial_mask",
            "model": model,
            "iterations_requested": int(iterations),
            "iterations_completed": 0,
            "initial_points": int(np.count_nonzero(mask)),
            "final_inliers": 0,
            "min_points": int(min_points),
            "iteration_details": [],
        }

    iteration_details = []
    coeff_az = None
    coeff_rg = None
    for iter_idx in range(int(iterations)):
        current_points = int(np.count_nonzero(mask))
        if current_points < int(min_points):
            break
        coeff_az, *_ = np.linalg.lstsq(A[mask], az[mask], rcond=None)
        coeff_rg, *_ = np.linalg.lstsq(A[mask], rg[mask], rcond=None)
        pred_az = A @ coeff_az
        pred_rg = A @ coeff_rg
        residual = np.sqrt((az - pred_az) ** 2 + (rg - pred_rg) ** 2)
        core = residual[mask]
        med = float(np.median(core))
        mad = float(np.median(np.abs(core - med)))
        sigma = max(1.4826 * mad, 1.0e-6)
        threshold = float(med + 2.5 * sigma)
        new_mask = residual <= threshold
        new_mask &= base_mask
        kept_points = int(np.count_nonzero(new_mask))
        iteration_details.append(
            {
                "iteration": int(iter_idx + 1),
                "input_points": int(current_points),
                "kept_points": int(kept_points),
                "median_residual": med,
                "mad_residual": mad,
                "sigma_residual": sigma,
                "threshold": threshold,
            }
        )
        if kept_points < int(min_points):
            break
        if np.array_equal(new_mask, mask):
            mask = new_mask
            break
        mask = new_mask

    final_inliers = int(np.count_nonzero(mask))
    if final_inliers < int(min_points) or coeff_az is None or coeff_rg is None:
        return None, None, np.zeros(rows.shape[0], dtype=bool), {
            "success": False,
            "reason": "insufficient_points_after_iterative_fit",
            "model": model,
            "iterations_requested": int(iterations),
            "iterations_completed": int(len(iteration_details)),
            "initial_points": int(np.count_nonzero(base_mask)),
            "final_inliers": int(final_inliers),
            "min_points": int(min_points),
            "iteration_details": iteration_details,
        }

    coeff_az, *_ = np.linalg.lstsq(A[mask], az[mask], rcond=None)
    coeff_rg, *_ = np.linalg.lstsq(A[mask], rg[mask], rcond=None)
    return coeff_az, coeff_rg, mask, {
        "success": True,
        "reason": None,
        "model": model,
        "iterations_requested": int(iterations),
        "iterations_completed": int(len(iteration_details)),
        "initial_points": int(np.count_nonzero(base_mask)),
        "final_inliers": int(final_inliers),
        "min_points": int(min_points),
        "row_mean": rows_mean,
        "row_std": rows_std,
        "col_mean": cols_mean,
        "col_std": cols_std,
        "azimuth_coefficients": [float(v) for v in coeff_az.tolist()],
        "range_coefficients": [float(v) for v in coeff_rg.tolist()],
        "iteration_details": iteration_details,
    }


def _fit_sparse_offset_model(
    rows: np.ndarray,
    cols: np.ndarray,
    az: np.ndarray,
    rg: np.ndarray,
    *,
    iterations: int,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray, dict]:
    model, min_points, model_selection = _choose_sparse_fit_model(rows.size)
    if rows.size < int(min_points):
        return None, None, np.zeros(rows.shape[0], dtype=bool), {
            "success": False,
            "reason": "insufficient_points_after_quality_filter",
            "model": model,
            "model_selection": model_selection,
            "iterations_requested": int(iterations),
            "iterations_completed": 0,
            "initial_points": int(rows.size),
            "final_inliers": 0,
            "min_points": int(min_points),
            "iteration_details": [],
        }

    prefit_enabled = str(os.environ.get("D2SAR_DENSE_PREFIT_ENABLE", "1")).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    prefit_requested = _normalize_dense_fit_model(
        os.environ.get("D2SAR_DENSE_PREFIT_MODEL", "quadratic")
    )
    if prefit_requested == "auto":
        prefit_requested = "quadratic"

    prefit_diag: dict | None = None
    seed_mask: np.ndarray | None = None
    if prefit_enabled and prefit_requested != "constant":
        prefit_model, prefit_min_points, prefit_selection = _choose_sparse_fit_model_for_request(
            rows.size,
            prefit_requested,
        )
        _, _, pre_mask, prefit_core = _iterative_sparse_fit(
            rows=rows,
            cols=cols,
            az=az,
            rg=rg,
            model=prefit_model,
            min_points=prefit_min_points,
            iterations=int(iterations),
            initial_mask=None,
        )
        prefit_diag = {**prefit_core, "model_selection": prefit_selection}
        if prefit_core.get("success", False):
            seed_mask = pre_mask

    coeff_az, coeff_rg, mask, fit_diag = _iterative_sparse_fit(
        rows=rows,
        cols=cols,
        az=az,
        rg=rg,
        model=model,
        min_points=min_points,
        iterations=int(iterations),
        initial_mask=seed_mask,
    )
    fit_diag["model_selection"] = model_selection
    fit_diag["prefit"] = prefit_diag
    if not fit_diag.get("success", False) or coeff_az is None or coeff_rg is None:
        return None, None, np.zeros(rows.shape[0], dtype=bool), fit_diag
    return coeff_az, coeff_rg, mask, fit_diag


def _prepare_sparse_offsets_for_dense_model(
    dense_match_details: dict,
    *,
    out_shape: tuple[int, int],
    min_points: int = 6,
    iterations: int = 5,
    snr_threshold: float = 2.0,
    correlation_threshold: float = 0.3,
) -> tuple[dict, dict]:
    row_sparse = np.asarray(dense_match_details["row_sparse"], dtype=np.float32)
    col_sparse = np.asarray(dense_match_details["col_sparse"], dtype=np.float32)
    row_coords = np.asarray(dense_match_details["row_coords"], dtype=np.float64)
    col_coords = np.asarray(dense_match_details["col_coords"], dtype=np.float64)
    snr = dense_match_details.get("snr")
    snr_arr = None if snr is None else np.asarray(snr, dtype=np.float32)
    correlation = dense_match_details.get("correlation")
    correlation_source = "input"
    if correlation is None:
        if snr_arr is not None:
            correlation_arr = np.clip(snr_arr / 100.0, 0.0, 1.0).astype(np.float32)
            correlation_source = "derived_from_snr"
        else:
            correlation_arr = np.ones_like(row_sparse, dtype=np.float32)
            correlation_source = "assumed_unity"
    else:
        correlation_arr = np.asarray(correlation, dtype=np.float32)

    finite_mask = (
        np.isfinite(row_sparse)
        & np.isfinite(col_sparse)
        & (True if snr_arr is None else np.isfinite(snr_arr))
        & np.isfinite(correlation_arr)
    )
    input_points = int(np.count_nonzero(finite_mask))
    quality_mask = finite_mask.copy()
    if snr_arr is not None:
        quality_mask &= snr_arr > float(snr_threshold)
    quality_mask &= correlation_arr > float(correlation_threshold)
    quality_points = int(np.count_nonzero(quality_mask))
    preview_model, preview_min_points, preview_model_selection = _choose_sparse_fit_model(
        quality_points
    )

    screened_row = np.full_like(row_sparse, np.nan, dtype=np.float32)
    screened_col = np.full_like(col_sparse, np.nan, dtype=np.float32)
    screened_snr = None if snr_arr is None else np.full_like(snr_arr, np.nan, dtype=np.float32)
    screened_correlation = np.full_like(correlation_arr, np.nan, dtype=np.float32)

    diagnostics = {
        "success": False,
        "reason": None,
        "input_points": int(input_points),
        "quality_filter": {
            "input_points": int(input_points),
            "kept_points": int(quality_points),
            "snr_threshold": float(snr_threshold),
            "correlation_threshold": float(correlation_threshold),
            "correlation_source": correlation_source,
        },
        "fit": {
            "success": False,
            "reason": None,
            "model": preview_model,
            "model_selection": preview_model_selection,
            "iterations_requested": int(iterations),
            "iterations_completed": 0,
            "initial_points": int(quality_points),
            "final_inliers": 0,
            "min_points": int(preview_min_points),
            "iteration_details": [],
        },
    }

    if quality_points <= 0:
        diagnostics["reason"] = "insufficient_points_after_quality_filter"
        prepared = dict(dense_match_details)
        prepared["row_sparse"] = screened_row
        prepared["col_sparse"] = screened_col
        prepared["row_coords"] = row_coords
        prepared["col_coords"] = col_coords
        prepared["snr"] = screened_snr
        prepared["correlation"] = screened_correlation
        prepared["row_offset"] = np.zeros(out_shape, dtype=np.float32)
        prepared["col_offset"] = np.zeros(out_shape, dtype=np.float32)
        diagnostics_to_write = dict(dense_match_details.get("diagnostics") or {})
        diagnostics_to_write["common_sparse_fit"] = diagnostics
        prepared["diagnostics"] = diagnostics_to_write
        return prepared, diagnostics

    sample_indices = np.argwhere(quality_mask)
    sample_rows = np.array([row_coords[r] for r, _ in sample_indices], dtype=np.float64)
    sample_cols = np.array([col_coords[c] for _, c in sample_indices], dtype=np.float64)
    sample_az = np.array([row_sparse[r, c] for r, c in sample_indices], dtype=np.float64)
    sample_rg = np.array([col_sparse[r, c] for r, c in sample_indices], dtype=np.float64)
    coeff_az, coeff_rg, inlier_mask, fit_diagnostics = _fit_sparse_offset_model(
        sample_rows,
        sample_cols,
        sample_az,
        sample_rg,
        iterations=int(iterations),
    )
    diagnostics["fit"] = fit_diagnostics
    if not fit_diagnostics.get("success", False):
        diagnostics["reason"] = fit_diagnostics.get("reason")
        prepared = dict(dense_match_details)
        prepared["row_sparse"] = screened_row
        prepared["col_sparse"] = screened_col
        prepared["row_coords"] = row_coords
        prepared["col_coords"] = col_coords
        prepared["snr"] = screened_snr
        prepared["correlation"] = screened_correlation
        prepared["row_offset"] = np.zeros(out_shape, dtype=np.float32)
        prepared["col_offset"] = np.zeros(out_shape, dtype=np.float32)
        diagnostics_to_write = dict(dense_match_details.get("diagnostics") or {})
        diagnostics_to_write["common_sparse_fit"] = diagnostics
        prepared["diagnostics"] = diagnostics_to_write
        return prepared, diagnostics

    for keep, (r, c) in zip(inlier_mask.tolist(), sample_indices.tolist()):
        if not keep:
            continue
        screened_row[r, c] = float(row_sparse[r, c])
        screened_col[r, c] = float(col_sparse[r, c])
        screened_correlation[r, c] = float(correlation_arr[r, c])
        if screened_snr is not None:
            screened_snr[r, c] = float(snr_arr[r, c])

    dense_rows = np.arange(int(out_shape[0]), dtype=np.float64)
    dense_cols = np.arange(int(out_shape[1]), dtype=np.float64)
    yy, xx = np.meshgrid(dense_rows, dense_cols, indexing="ij")
    row_offset = _evaluate_offset_surface(
        coeff_az,
        yy.ravel(),
        xx.ravel(),
        row_mean=float(fit_diagnostics["row_mean"]),
        row_std=float(fit_diagnostics["row_std"]),
        col_mean=float(fit_diagnostics["col_mean"]),
        col_std=float(fit_diagnostics["col_std"]),
        model=str(fit_diagnostics["model"]),
    ).reshape(out_shape).astype(np.float32)
    col_offset = _evaluate_offset_surface(
        coeff_rg,
        yy.ravel(),
        xx.ravel(),
        row_mean=float(fit_diagnostics["row_mean"]),
        row_std=float(fit_diagnostics["row_std"]),
        col_mean=float(fit_diagnostics["col_mean"]),
        col_std=float(fit_diagnostics["col_std"]),
        model=str(fit_diagnostics["model"]),
    ).reshape(out_shape).astype(np.float32)

    diagnostics["success"] = True
    diagnostics["reason"] = None
    diagnostics_to_write = dict(dense_match_details.get("diagnostics") or {})
    diagnostics_to_write["common_sparse_fit"] = diagnostics

    prepared = dict(dense_match_details)
    prepared["row_sparse"] = screened_row
    prepared["col_sparse"] = screened_col
    prepared["row_coords"] = row_coords
    prepared["col_coords"] = col_coords
    prepared["snr"] = screened_snr
    prepared["correlation"] = screened_correlation
    prepared["row_offset"] = row_offset
    prepared["col_offset"] = col_offset
    prepared["diagnostics"] = diagnostics_to_write
    return prepared, diagnostics


def _default_rubbersheet_params() -> dict:
    return {
        "culling_metric": "median_filter",
        "median_filter_size_range": 9,
        "median_filter_size_azimuth": 9,
        "threshold": 0.75,
        "mask_refine_enabled": True,
        "mask_refine_min_neighbors": 5,
        "mask_refine_filter_size": 5,
        "outlier_filling_method": "fill_smoothed",
        "fill_smoothed": {"iterations": 1, "kernel_size": 3},
        "interpolation_method": "linear",
        "offsets_filter": "boxcar",
        "boxcar": {"filter_size_range": 5, "filter_size_azimuth": 5},
        "snr_threshold": 3.0,
    }


def _compute_mad_mask(
    offset: np.ndarray,
    window_az: int,
    window_rg: int,
    threshold: float,
) -> np.ndarray:
    median_off = ndimage.median_filter(
        np.asarray(offset, dtype=np.float32),
        size=(int(window_az), int(window_rg)),
        mode="nearest",
    )
    mad = np.abs(np.asarray(offset, dtype=np.float32) - median_off)
    return mad > float(threshold)


def _remove_pixels_with_many_nans(
    offset: np.ndarray,
    *,
    kernel_size: int,
    min_valid_neighbors: int,
) -> np.ndarray:
    out = np.array(offset, dtype=np.float32, copy=True)
    nan_mask = np.isnan(out)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.int32)
    center = kernel_size // 2
    kernel[center, center] = 0
    valid_neighbor_count = ndimage.convolve(
        (~nan_mask).astype(np.int32),
        kernel,
        mode="constant",
        cval=0,
    )
    out[valid_neighbor_count < int(min_valid_neighbors)] = np.nan
    return out


def _fill_nan_with_mean(
    arr_in: np.ndarray,
    arr_ref: np.ndarray,
    neighborhood_size: int,
) -> np.ndarray:
    filled = np.array(arr_in, dtype=np.float32, copy=True)
    nan_mask = np.isnan(filled)
    if not np.any(nan_mask):
        return filled
    kernel = np.ones((neighborhood_size, neighborhood_size), dtype=np.float32)
    ref = np.asarray(arr_ref, dtype=np.float32)
    masked_ref = np.where(np.isnan(ref), 0.0, ref)
    local_sum = ndimage.convolve(masked_ref, kernel, mode="constant", cval=0.0)
    valid_counts = ndimage.convolve(
        np.isfinite(ref).astype(np.float32),
        kernel,
        mode="constant",
        cval=0.0,
    )
    valid_counts[valid_counts == 0] = np.nan
    local_mean = local_sum / valid_counts
    filled[nan_mask] = local_mean[nan_mask]
    return filled


def _fill_outliers_holes(offset: np.ndarray, params: dict) -> np.ndarray:
    method = params.get("outlier_filling_method", "fill_smoothed")
    if method == "nearest_neighbor":
        invalid = np.isnan(offset)
        if not np.any(invalid):
            return np.asarray(offset, dtype=np.float32)
        _, indices = ndimage.distance_transform_edt(
            invalid,
            return_distances=True,
            return_indices=True,
        )
        return np.asarray(offset, dtype=np.float32)[tuple(indices)]
    if method != "fill_smoothed":
        raise ValueError(f"unsupported outlier filling method: {method}")
    kernel_size = int(params["fill_smoothed"]["kernel_size"])
    iterations = int(params["fill_smoothed"]["iterations"])
    filled = _fill_nan_with_mean(offset, offset, kernel_size)
    while np.isnan(filled).any() and iterations > 0:
        iterations -= 1
        filled = _fill_nan_with_mean(filled, filled, kernel_size)
    return filled


def _interpolate_nan_offsets(offset: np.ndarray, method: str) -> np.ndarray:
    valid_mask = np.isfinite(offset)
    if not np.any(valid_mask):
        return np.zeros_like(offset, dtype=np.float32)
    if method == "no_interpolation":
        out = np.array(offset, dtype=np.float32, copy=True)
        out[~valid_mask] = 0.0
        return out
    yy, xx = np.nonzero(valid_mask)
    values = np.asarray(offset, dtype=np.float32)[valid_mask]
    grid_x, grid_y = np.meshgrid(np.arange(offset.shape[1]), np.arange(offset.shape[0]))
    if values.size == 1:
        return np.full_like(offset, float(values[0]), dtype=np.float32)
    if method == "nearest":
        interp = interpolate.NearestNDInterpolator((xx, yy), values)
        return np.asarray(interp(grid_x, grid_y), dtype=np.float32)
    if method == "linear":
        if values.size < 4:
            interp = interpolate.NearestNDInterpolator((xx, yy), values)
            return np.asarray(interp(grid_x, grid_y), dtype=np.float32)
        interp = interpolate.LinearNDInterpolator((xx, yy), values, fill_value=np.nan)
        out = np.asarray(interp(grid_x, grid_y), dtype=np.float32)
        if np.isnan(out).any():
            nearest = interpolate.NearestNDInterpolator((xx, yy), values)
            out[np.isnan(out)] = np.asarray(nearest(grid_x, grid_y), dtype=np.float32)[np.isnan(out)]
        return out
    if method == "cubic":
        if values.size < 4:
            interp = interpolate.NearestNDInterpolator((xx, yy), values)
            return np.asarray(interp(grid_x, grid_y), dtype=np.float32)
        out = interpolate.griddata((xx, yy), values, (grid_x, grid_y), method="cubic", fill_value=np.nan)
        out = np.asarray(out, dtype=np.float32)
        if np.isnan(out).any():
            nearest = interpolate.NearestNDInterpolator((xx, yy), values)
            out[np.isnan(out)] = np.asarray(nearest(grid_x, grid_y), dtype=np.float32)[np.isnan(out)]
        return out
    raise ValueError(f"unsupported interpolation method: {method}")


def _filter_offset_field(offset: np.ndarray, params: dict) -> np.ndarray:
    filter_type = params.get("offsets_filter", "boxcar")
    if filter_type == "none":
        return np.asarray(offset, dtype=np.float32)
    if filter_type == "boxcar":
        window_rg = int(params["boxcar"]["filter_size_range"])
        window_az = int(params["boxcar"]["filter_size_azimuth"])
        kernel = np.ones((window_az, window_rg), dtype=np.float32) / float(window_az * window_rg)
        return np.asarray(signal.convolve2d(offset, kernel, mode="same", boundary="symm"), dtype=np.float32)
    if filter_type == "median":
        return np.asarray(
            ndimage.median_filter(
                offset,
                size=(
                    int(params["median"]["filter_size_azimuth"]),
                    int(params["median"]["filter_size_range"]),
                ),
                mode="nearest",
            ),
            dtype=np.float32,
        )
    if filter_type == "gaussian":
        return np.asarray(
            ndimage.gaussian_filter(
                offset,
                sigma=(
                    float(params["gaussian"]["sigma_azimuth"]),
                    float(params["gaussian"]["sigma_range"]),
                ),
                mode="nearest",
            ),
            dtype=np.float32,
        )
    raise ValueError(f"unsupported offsets filter: {filter_type}")


def _rubbersheet_dense_offsets(
    *,
    row_sparse: np.ndarray,
    col_sparse: np.ndarray,
    row_coords: np.ndarray,
    col_coords: np.ndarray,
    out_shape: tuple[int, int],
    snr: np.ndarray | None = None,
    covariance_az: np.ndarray | None = None,
    covariance_rg: np.ndarray | None = None,
    params: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    cfg = _default_rubbersheet_params()
    if params:
        cfg.update({k: v for k, v in params.items() if k not in {"fill_smoothed", "boxcar"}})
        if "fill_smoothed" in params:
            cfg["fill_smoothed"] = {**cfg["fill_smoothed"], **params["fill_smoothed"]}
        if "boxcar" in params:
            cfg["boxcar"] = {**cfg["boxcar"], **params["boxcar"]}

    row_work = np.asarray(row_sparse, dtype=np.float32).copy()
    col_work = np.asarray(col_sparse, dtype=np.float32).copy()
    metric = cfg["culling_metric"]
    if metric == "snr":
        if snr is None:
            raise ValueError("snr culling requires snr grid")
        mask = np.asarray(snr, dtype=np.float32) < float(cfg["threshold"])
    elif metric == "covariance":
        if covariance_az is None or covariance_rg is None:
            raise ValueError("covariance culling requires covariance grids")
        mask = (np.asarray(covariance_az) > float(cfg["threshold"])) | (
            np.asarray(covariance_rg) > float(cfg["threshold"])
        )
    else:
        mask = _compute_mad_mask(
            row_work,
            int(cfg["median_filter_size_azimuth"]),
            int(cfg["median_filter_size_range"]),
            float(cfg["threshold"]),
        ) | _compute_mad_mask(
            col_work,
            int(cfg["median_filter_size_azimuth"]),
            int(cfg["median_filter_size_range"]),
            float(cfg["threshold"]),
        )
        if snr is not None:
            mask |= np.asarray(snr, dtype=np.float32) < float(cfg["snr_threshold"])

    row_work[mask] = np.nan
    col_work[mask] = np.nan
    masked_fraction_initial = float(np.count_nonzero(mask) / mask.size) if mask.size else 0.0

    if cfg.get("mask_refine_enabled", False):
        refine_mask = _compute_mad_mask(
            np.nan_to_num(row_work, nan=0.0),
            int(cfg["mask_refine_filter_size"]),
            int(cfg["mask_refine_filter_size"]),
            float(cfg["threshold"]),
        ) | _compute_mad_mask(
            np.nan_to_num(col_work, nan=0.0),
            int(cfg["mask_refine_filter_size"]),
            int(cfg["mask_refine_filter_size"]),
            float(cfg["threshold"]),
        )
        row_work[refine_mask] = np.nan
        col_work[refine_mask] = np.nan
        row_work = _remove_pixels_with_many_nans(
            row_work,
            kernel_size=int(cfg["mask_refine_filter_size"]),
            min_valid_neighbors=int(cfg["mask_refine_min_neighbors"]),
        )
        col_work = _remove_pixels_with_many_nans(
            col_work,
            kernel_size=int(cfg["mask_refine_filter_size"]),
            min_valid_neighbors=int(cfg["mask_refine_min_neighbors"]),
        )

    row_filled = _fill_outliers_holes(row_work, cfg)
    col_filled = _fill_outliers_holes(col_work, cfg)
    row_interp = _interpolate_nan_offsets(row_filled, cfg["interpolation_method"])
    col_interp = _interpolate_nan_offsets(col_filled, cfg["interpolation_method"])
    row_dense = _interpolate_sparse_offsets(row_interp, row_coords, col_coords, out_shape)
    col_dense = _interpolate_sparse_offsets(col_interp, row_coords, col_coords, out_shape)
    row_dense = _filter_offset_field(row_dense, cfg)
    col_dense = _filter_offset_field(col_dense, cfg)

    diagnostics = {
        "culling_metric": metric,
        "threshold": float(cfg["threshold"]),
        "snr_threshold": float(cfg["snr_threshold"]),
        "masked_fraction_initial": masked_fraction_initial,
        "masked_count_initial": int(np.count_nonzero(mask)),
        "total_count": int(mask.size),
        "nan_count_after_refine_row": int(np.count_nonzero(~np.isfinite(row_work))),
        "nan_count_after_refine_col": int(np.count_nonzero(~np.isfinite(col_work))),
        "interpolation_method": cfg["interpolation_method"],
        "offsets_filter": cfg["offsets_filter"],
        "row_sparse_stats": _summarize_numeric_array(row_sparse),
        "col_sparse_stats": _summarize_numeric_array(col_sparse),
        "row_dense_stats": _summarize_numeric_array(row_dense),
        "col_dense_stats": _summarize_numeric_array(col_dense),
        "snr_stats": None if snr is None else _summarize_numeric_array(snr),
    }
    return row_dense.astype(np.float32), col_dense.astype(np.float32), diagnostics


def _summarize_numeric_array(data: np.ndarray | None) -> dict | None:
    if data is None:
        return None
    arr = np.asarray(data, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"count": int(arr.size), "finite_count": 0}
    return {
        "count": int(arr.size),
        "finite_count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "std": float(np.std(finite)),
    }


def _is_vsi_path(path: str) -> bool:
    """Check if path is a GDAL VSI virtual filesystem path."""
    return path.startswith("/vsizip/") or path.startswith("/vsitar/")


def _input_path_available(path: str | Path) -> bool:
    path_str = str(path)
    return _is_vsi_path(path_str) or Path(path_str).exists()


def _maybe_convert_complex_int16(path: str) -> tuple[str, callable | None]:
    try:
        src_ds = gdal.Open(path, gdal.GA_ReadOnly)
        if src_ds is None:
            return path, None
        src_band = src_ds.GetRasterBand(1)
        dtype = src_band.DataType
        if dtype != gdal.GDT_CInt16:
            src_ds = None
            return path, None
        rows, cols = src_ds.RasterYSize, src_ds.RasterXSize
        data = _read_band_array(src_band, dtype=np.complex64).astype(np.complex64)
        src_ds = None

        tmp = tempfile.NamedTemporaryFile(suffix=".conv.tif", delete=False)
        tmp.close()
        converted_tif = tmp.name
        driver = gdal.GetDriverByName("GTiff")
        conv_ds = driver.Create(
            converted_tif,
            cols,
            rows,
            1,
            gdal.GDT_CFloat32,
            options=["COMPRESS=LZW", "TILED=YES"],
        )
        conv_ds.WriteRaster(0, 0, cols, rows, data.tobytes())
        conv_ds = None
        del data
        gc.collect()

        def cleanup():
            try:
                Path(converted_tif).unlink(missing_ok=True)
            except OSError:
                pass

        return converted_tif, cleanup
    except Exception:
        return path, None


def _ensure_local_tiff(vsi_path: str) -> tuple[str, None] | tuple[None, None]:
    if not _is_vsi_path(vsi_path):
        return _maybe_convert_complex_int16(vsi_path)
    try:
        import zipfile
        prefix = "/vsizip/"
        if not vsi_path.startswith(prefix):
            return vsi_path, None
        rest = vsi_path[len(prefix):].lstrip("/")
        zip_end = rest.find(".zip")
        if zip_end < 0:
            return vsi_path, None
        zip_end += 4
        zip_path = "/" + rest[:zip_end]
        member = rest[zip_end + 1:]
        if not member:
            return vsi_path, None
        with zipfile.ZipFile(zip_path, "r") as zf:
            raw_bytes = zf.read(member)
        tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
        tmp.write(raw_bytes)
        tmp.close()
        tmp_path = tmp.name
        del raw_bytes
        gc.collect()
        converted_path, converted_cleanup = _maybe_convert_complex_int16(tmp_path)
        if converted_path != tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            tmp_path = converted_path
        def cleanup():
            for p in [tmp_path]:
                try:
                    Path(p).unlink(missing_ok=True)
                except OSError:
                    pass
            if converted_cleanup is not None:
                converted_cleanup()
        return tmp_path, cleanup
    except Exception:
        return None, None


def _extract_to_temp_tiff(vsi_path: str) -> tuple[str, str | None]:
    return _ensure_local_tiff(vsi_path)


def _copy_raster(src: str, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if Path(src).exists():
        shutil.copyfile(src, dst)
    elif _is_vsi_path(src):
        local_path, cleanup_cb = _extract_to_temp_tiff(src)
        if local_path is not None:
            shutil.copyfile(local_path, dst)
            if cleanup_cb:
                cleanup_cb()
        else:
            dst.write_bytes(b"")
    else:
        dst.write_bytes(b"")
    return str(dst)


def _copy_raster_to_envi_complex64(
    src: str,
    dst: str | Path,
) -> str:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.unlink(missing_ok=True)
    Path(f"{dst}.hdr").unlink(missing_ok=True)

    local_src = src
    cleanup_cb = None
    if _is_vsi_path(src):
        local_src, cleanup_cb = _extract_to_temp_tiff(src)
        if local_src is None:
            raise RuntimeError(f"failed to materialize source raster: {src}")

    try:
        src_ds = gdal.Open(str(local_src), gdal.GA_ReadOnly)
        if src_ds is None:
            raise RuntimeError(f"failed to open source raster for ENVI copy: {src}")
        try:
            translated = gdal.Translate(
                str(dst),
                src_ds,
                format="ENVI",
                outputType=gdal.GDT_CFloat32,
                creationOptions=["INTERLEAVE=BIP"],
            )
            if translated is None:
                raise RuntimeError(f"gdal.Translate failed for {src} -> {dst}")
            translated = None
        finally:
            src_ds = None
        return str(dst)
    finally:
        if cleanup_cb is not None:
            cleanup_cb()


def _materialize_ampcor_input(
    src: str,
    dst_base: str | Path,
) -> str:
    """Materialize a complex64 raster for Ampcor input.

    Prefer ENVI to match ISCE3 workflows, but fall back to local GTiff when the
    runtime GDAL build does not provide ENVI support.
    """
    try:
        return _copy_raster_to_envi_complex64(src, dst_base)
    except Exception:
        dst_base = Path(dst_base)
        tif_path = dst_base.with_suffix(".tif")
        return _copy_raster(src, tif_path)


def _coerce_offset_dataset_for_resample(
    dataset: str | Path | np.ndarray | None,
    out_shape: tuple[int, int],
) -> np.ndarray:
    if dataset is None:
        return np.zeros(out_shape, dtype=np.float32)
    if isinstance(dataset, np.ndarray):
        if dataset.shape != out_shape:
            raise RuntimeError(
                f"offset array shape mismatch: expected {out_shape}, got {dataset.shape}"
            )
        return np.asarray(dataset)
    loaded = _load_offset_dataset_for_resample(dataset, out_shape)
    return np.asarray(loaded)


def run_resamp_isce3_v2(
    *,
    input_slc_path: str,
    output_slc_path: str,
    radar_grid,
    doppler,
    ref_radar_grid,
    rg_offset_dataset: str | Path | np.ndarray | None = None,
    az_offset_dataset: str | Path | np.ndarray | None = None,
    use_gpu: bool = False,
    block_size_az: int = 256,
    block_size_rg: int = 0,
) -> tuple[bool, str]:
    """Run ISCE3 v2 blockwise SLC resampling with optional GPU execution."""
    try:
        from isce3.core import LUT2d
        from isce3.image.v2.resample_slc import resample_slc_blocks
        from isce3.io.gdal.gdal_raster import GDALRaster

        out_length = ref_radar_grid.length
        out_width = ref_radar_grid.width
        out_shape = (out_length, out_width)

        if block_size_rg == 0:
            block_size_rg = out_width

        rg_offsets = _coerce_offset_dataset_for_resample(rg_offset_dataset, out_shape)
        az_offsets = _coerce_offset_dataset_for_resample(az_offset_dataset, out_shape)

        out_path = Path(output_slc_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.unlink(missing_ok=True)
        Path(f"{out_path}.aux.xml").unlink(missing_ok=True)

        writer = GDALRaster.create_dataset_file(
            filepath=out_path,
            dtype=np.complex64,
            shape=out_shape,
            num_bands=1,
            driver_name="GTiff",
        )
        if writer is None:
            return False, "isce3-v2-resample_slc_blocks writer allocation failed"

        reader_path = str(Path(input_slc_path))
        if use_gpu:
            local_path, cleanup_cb = _extract_to_temp_tiff(reader_path)
        else:
            local_path, cleanup_cb = _ensure_local_tiff(reader_path)
        if local_path is None:
            del writer
            gc.collect()
            return False, "isce3-v2-resample_slc_blocks input raster unavailable"
        try:
            reader_ds = GDALRaster(filepath=local_path, band=1)
            resample_slc_blocks(
                output_resampled_slcs=[writer],
                input_slcs=[reader_ds],
                az_offsets_dataset=az_offsets,
                rg_offsets_dataset=rg_offsets,
                input_radar_grid=radar_grid,
                doppler=doppler if doppler is not None else LUT2d(),
                block_size_az=block_size_az,
                block_size_rg=block_size_rg,
                fill_value=np.complex64(0.0 + 0.0j),
                quiet=True,
                with_gpu=use_gpu,
            )
            del writer
            del reader_ds
            gc.collect()
            if not out_path.exists() or out_path.stat().st_size == 0:
                return False, "isce3-v2-resample_slc_blocks produced empty output"
            backend = "gpu" if use_gpu else "cpu"
            return True, f"isce3-v2-resample_slc_blocks-{backend}"
        finally:
            if cleanup_cb is not None:
                cleanup_cb()
    except Exception as exc:
        backend = "gpu" if use_gpu else "cpu"
        return False, f"isce3-v2-resample_slc_blocks-{backend} unavailable: {exc}"


def _import_legacy_resampslc():
    """Import legacy ISCE3 ResampSlc bindings across package layouts."""
    try:
        from isce3.image.ResampSlc import ResampSlc
    except Exception:
        from isce3.image import ResampSlc
    try:
        from isce3.io.Raster import Raster
    except Exception:
        from isce3.io import Raster
    return ResampSlc, Raster


def _make_legacy_resampslc(radar_grid, doppler, ref_radar_grid):
    from isce3.core import LUT2d

    ResampSlc, _ = _import_legacy_resampslc()
    doppler = doppler if doppler is not None else LUT2d()
    invalid = np.complex64(0.0 + 0.0j)
    try:
        return ResampSlc(
            rdr_grid=radar_grid,
            doppler=doppler,
            invalid_value=invalid,
            ref_rdr_grid=ref_radar_grid,
        )
    except TypeError:
        try:
            from isce3.core import Poly2d

            return ResampSlc(radar_grid, doppler, Poly2d(), Poly2d(), invalid, ref_radar_grid)
        except TypeError:
            if ref_radar_grid is not None:
                return ResampSlc(radar_grid, ref_radar_grid, doppler, invalid)
            return ResampSlc(radar_grid, doppler, invalid)


def run_resamp_isce3_legacy(
    *,
    input_slc_path: str,
    output_slc_path: str,
    radar_grid,
    doppler,
    ref_radar_grid,
    rg_offset_dataset: str | Path | np.ndarray | None = None,
    az_offset_dataset: str | Path | np.ndarray | None = None,
    block_size_az: int = 256,
) -> tuple[bool, str]:
    """Run legacy ISCE3 ResampSlc without using Python gdal_array helpers."""
    cleanup_cb = None
    try:
        _, Raster = _import_legacy_resampslc()

        out_length = int(ref_radar_grid.length)
        out_width = int(ref_radar_grid.width)
        out_shape = (out_length, out_width)
        rg_offsets = _coerce_offset_dataset_for_resample(rg_offset_dataset, out_shape)
        az_offsets = _coerce_offset_dataset_for_resample(az_offset_dataset, out_shape)

        reader_path, cleanup_cb = _ensure_local_tiff(str(input_slc_path))
        if reader_path is None:
            return False, "isce3-legacy-ResampSlc input raster unavailable"

        out_path = Path(output_slc_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.unlink(missing_ok=True)
        Path(f"{out_path}.aux.xml").unlink(missing_ok=True)

        with tempfile.TemporaryDirectory(prefix="isce3_resamp_offsets_", dir=str(out_path.parent)) as tmpdir:
            tmpdir_path = Path(tmpdir)
            rg_path = _write_offset_raster(tmpdir_path / "range.off.tif", np.asarray(rg_offsets, dtype=np.float32))
            az_path = _write_offset_raster(tmpdir_path / "azimuth.off.tif", np.asarray(az_offsets, dtype=np.float32))

            resamp = _make_legacy_resampslc(radar_grid, doppler, ref_radar_grid)
            if block_size_az > 0:
                try:
                    resamp.lines_per_tile = int(block_size_az)
                except Exception:
                    try:
                        resamp.linesPerTile = int(block_size_az)
                    except Exception:
                        pass

            input_raster = Raster(str(reader_path))
            output_raster = Raster(str(out_path), out_width, out_length, 1, int(gdal.GDT_CFloat32), "GTiff")
            rg_raster = Raster(str(rg_path))
            az_raster = Raster(str(az_path))
            resamp.resamp(input_raster, output_raster, rg_raster, az_raster, 1, False)
            for raster_obj in (output_raster, input_raster, rg_raster, az_raster):
                try:
                    raster_obj.close_dataset()
                except Exception:
                    pass

        gc.collect()
        if not out_path.exists() or out_path.stat().st_size == 0:
            return False, "isce3-legacy-ResampSlc produced empty output"
        return True, "isce3-legacy-ResampSlc-cpu"
    except Exception as exc:
        return False, f"isce3-legacy-ResampSlc unavailable: {exc}"
    finally:
        if cleanup_cb is not None:
            cleanup_cb()


def run_resamp_isce3(
    *,
    input_slc_path: str,
    output_slc_path: str,
    radar_grid,
    doppler,
    ref_radar_grid,
    rg_offset_dataset: str | Path | np.ndarray | None = None,
    az_offset_dataset: str | Path | np.ndarray | None = None,
    use_gpu: bool = False,
    block_size_az: int = 256,
    block_size_rg: int = 0,
) -> tuple[bool, str]:
    """Run ISCE3 resampling, falling back when v2 imports gdal_array internally."""
    ok, method = run_resamp_isce3_v2(
        input_slc_path=input_slc_path,
        output_slc_path=output_slc_path,
        radar_grid=radar_grid,
        doppler=doppler,
        ref_radar_grid=ref_radar_grid,
        rg_offset_dataset=rg_offset_dataset,
        az_offset_dataset=az_offset_dataset,
        use_gpu=use_gpu,
        block_size_az=block_size_az,
        block_size_rg=block_size_rg,
    )
    if ok:
        return ok, method

    legacy_ok, legacy_method = run_resamp_isce3_legacy(
        input_slc_path=input_slc_path,
        output_slc_path=output_slc_path,
        radar_grid=radar_grid,
        doppler=doppler,
        ref_radar_grid=ref_radar_grid,
        rg_offset_dataset=rg_offset_dataset,
        az_offset_dataset=az_offset_dataset,
        block_size_az=block_size_az,
    )
    if legacy_ok:
        return True, f"{legacy_method} (v2 failed: {method})"
    return False, f"{method}; legacy failed: {legacy_method}"


def run_coarse_resamp_isce3_v2(
    *,
    slave_slc_path: str,
    coarse_coreg_slave_path: str,
    radar_grid,
    doppler,
    ref_radar_grid,
    rg_offset_path: str | None = None,
    az_offset_path: str | None = None,
    use_gpu: bool = False,
    block_size_az: int = 256,
    block_size_rg: int = 0,
) -> bool:
    """Run coarse SLC resampling using ISCE3 v2, with legacy ResampSlc fallback.

    Uses the provided Geo2Rdr offsets when available; otherwise zero offsets are
    supplied by the resampling wrapper. Output is radar-domain (NOT geocoded).

    Parameters
    ----------
    slave_slc_path : str
        Path to input secondary SLC (complex).
    coarse_coreg_slave_path : str
        Path to output coarsely coregistered secondary SLC.
    radar_grid : RadarGridParameters
        Radar grid of the input secondary SLC.
    doppler : LUT2d
        Doppler centroid of secondary SLC.
    ref_radar_grid : RadarGridParameters
        Reference radar grid defining output geometry.
    use_gpu : bool
        Whether to use GPU resampling.
    block_size_az : int
        Azimuth block size for processing.
    block_size_rg : int
        Range block size (0 = full line).

    Returns
    -------
    bool
        True if resampling succeeded, False otherwise.
    """
    ok, method = run_resamp_isce3(
        input_slc_path=slave_slc_path,
        output_slc_path=coarse_coreg_slave_path,
        radar_grid=radar_grid,
        doppler=doppler,
        ref_radar_grid=ref_radar_grid,
        rg_offset_dataset=rg_offset_path,
        az_offset_dataset=az_offset_path,
        use_gpu=use_gpu,
        block_size_az=block_size_az,
        block_size_rg=block_size_rg,
    )
    if not ok:
        print(f"[run_coarse_resamp_isce3_v2] failed: {method}")
    elif "legacy" in method:
        print(f"[run_coarse_resamp_isce3_v2] fallback: {method}", flush=True)
    return ok


def run_pycuampcor_dense_offsets(
    *,
    master_slc_path: str,
    slave_slc_path: str,
    output_dir: str | Path,
    gpu_id: int = 0,
    window_size: tuple[int, int] = (64, 64),
    search_range: tuple[int, int] = (20, 20),
    skip: tuple[int, int] = (32, 32),
    gross_offset_filepath: str | Path | None = None,
    reference_start_pixel_down: int | None = None,
    reference_start_pixel_across: int | None = None,
    number_window_down: int | None = None,
    number_window_across: int | None = None,
    timeout_seconds: int = 60 * 60,
    return_details: bool = False,
) -> tuple[np.ndarray | None, np.ndarray | None] | tuple[np.ndarray | None, np.ndarray | None, dict | None]:
    shape = _raster_shape(master_slc_path)
    if shape is None:
        raise RuntimeError(f"failed to open master SLC for offsets: {master_slc_path}")
    rows, cols = shape
    planned_grid = _plan_matching_grid(
        rows=rows,
        cols=cols,
        window_size=window_size,
        search_range=search_range,
        skip=skip,
        gross_offset=(0.0, 0.0),
        max_windows=2400,
        max_window_down=60,
        max_window_across=40,
    )
    explicit_window_grid = (
        reference_start_pixel_down is not None
        and reference_start_pixel_across is not None
        and number_window_down is not None
        and number_window_across is not None
    )
    effective_skip_down = int(skip[0]) if explicit_window_grid else int(planned_grid["skip_down"])
    effective_skip_across = int(skip[1]) if explicit_window_grid else int(planned_grid["skip_across"])

    try:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        reference_slc_path = out_dir / "reference.slc"
        secondary_slc_path = out_dir / "secondary.slc"
        master_local = _materialize_ampcor_input(master_slc_path, reference_slc_path)
        slave_local = _materialize_ampcor_input(slave_slc_path, secondary_slc_path)
        if not Path(master_local).exists() or not Path(slave_local).exists():
            return (None, None, None) if return_details else (None, None)
        payload = {
            "master_slc_path": master_local,
            "slave_slc_path": slave_local,
            "rows": rows,
            "cols": cols,
            "gpu_id": gpu_id,
            "window_size": list(window_size),
            "search_range": list(search_range),
            "skip": [effective_skip_down, effective_skip_across],
            "algorithm": "frequency",
            "raw_data_oversampling_factor": 2,
            "deramping_method": "complex",
            "deramping_axis": "azimuth",
            "correlation_statistics_zoom": 21,
            "correlation_surface_zoom": 8,
            "correlation_surface_oversampling_factor": 64,
            "correlation_surface_oversampling_method": "fft",
            "windows_batch_range": 10,
            "windows_batch_azimuth": 1,
            "cuda_streams": 2,
            "gross_offset_filepath": None if gross_offset_filepath is None else str(gross_offset_filepath),
            "reference_start_pixel_down": int(planned_grid["reference_start_pixel_down"]) if reference_start_pixel_down is None else int(reference_start_pixel_down),
            "reference_start_pixel_across": int(planned_grid["reference_start_pixel_across"]) if reference_start_pixel_across is None else int(reference_start_pixel_across),
            "number_window_down": int(planned_grid["number_window_down"]) if number_window_down is None else int(number_window_down),
            "number_window_across": int(planned_grid["number_window_across"]) if number_window_across is None else int(number_window_across),
            "output_dir": str(out_dir),
        }
        helper = r'''
import json, sys
from pathlib import Path
import numpy as np
cfg = json.loads(sys.argv[1])
from isce3.cuda.matchtemplate import PyCuAmpcor

def allocate_raw_bip_float32(filename, width, length, bands):
    mm = np.memmap(filename, dtype=np.float32, mode="w+", shape=(length, width, bands))
    mm[:] = 0.0
    mm.flush()
    del mm

ampcor = PyCuAmpcor()
ampcor.deviceID = int(cfg["gpu_id"])
ampcor.useMmap = 1
ampcor.referenceImageName = cfg["master_slc_path"]
ampcor.secondaryImageName = cfg["slave_slc_path"]
ampcor.referenceImageHeight = int(cfg["rows"])
ampcor.referenceImageWidth = int(cfg["cols"])
ampcor.secondaryImageHeight = int(cfg["rows"])
ampcor.secondaryImageWidth = int(cfg["cols"])
ampcor.windowSizeHeight = int(cfg["window_size"][0])
ampcor.windowSizeWidth = int(cfg["window_size"][1])
ampcor.halfSearchRangeDown = int(cfg["search_range"][0])
ampcor.halfSearchRangeAcross = int(cfg["search_range"][1])
ampcor.skipSampleDown = int(cfg["skip"][0])
ampcor.skipSampleAcross = int(cfg["skip"][1])
ampcor.referenceStartPixelDownStatic = int(cfg["reference_start_pixel_down"]) if cfg["reference_start_pixel_down"] is not None else int(cfg["search_range"][0])
ampcor.referenceStartPixelAcrossStatic = int(cfg["reference_start_pixel_across"]) if cfg["reference_start_pixel_across"] is not None else int(cfg["search_range"][1])
margin_rg = 2 * ampcor.halfSearchRangeAcross + ampcor.windowSizeWidth
margin_az = 2 * ampcor.halfSearchRangeDown + ampcor.windowSizeHeight
ampcor.numberWindowDown = int(cfg["number_window_down"]) if cfg["number_window_down"] is not None else max(1, (int(cfg["rows"]) - margin_az) // max(1, int(cfg["skip"][0])))
ampcor.numberWindowAcross = int(cfg["number_window_across"]) if cfg["number_window_across"] is not None else max(1, (int(cfg["cols"]) - margin_rg) // max(1, int(cfg["skip"][1])))
ampcor.algorithm = 0 if cfg["algorithm"] == "frequency" else 1
ampcor.rawDataOversamplingFactor = int(cfg["raw_data_oversampling_factor"])
if cfg["deramping_method"] == "magnitude":
    ampcor.derampMethod = 0
elif cfg["deramping_method"] == "complex":
    ampcor.derampMethod = 1
else:
    ampcor.derampMethod = 2
if cfg["deramping_axis"] == "azimuth":
    ampcor.derampAxis = 0
elif cfg["deramping_axis"] == "range":
    ampcor.derampAxis = 1
else:
    ampcor.derampAxis = 2
ampcor.corrStatWindowSize = int(cfg["correlation_statistics_zoom"])
ampcor.corrSurfaceZoomInWindow = int(cfg["correlation_surface_zoom"])
ampcor.corrSurfaceOverSamplingFactor = int(cfg["correlation_surface_oversampling_factor"])
ampcor.corrSurfaceOverSamplingMethod = 0 if cfg["correlation_surface_oversampling_method"] == "fft" else 1
ampcor.numberWindowAcrossInChunk = int(cfg["windows_batch_range"])
ampcor.numberWindowDownInChunk = int(cfg["windows_batch_azimuth"])
ampcor.nStreams = int(cfg["cuda_streams"])
out_dir = Path(cfg["output_dir"])
ampcor.offsetImageName = str(out_dir / "dense_offsets")
ampcor.grossOffsetImageName = str(out_dir / "gross_offsets")
ampcor.snrImageName = str(out_dir / "snr")
ampcor.covImageName = str(out_dir / "covariance")
ampcor.corrImageName = str(out_dir / "correlation_peak")
ampcor.setupParams()
ampcor.setConstantGrossOffset(0, 0)
if cfg["gross_offset_filepath"]:
    gross_offset = np.fromfile(cfg["gross_offset_filepath"], dtype=np.int32)
    windows_number = ampcor.numberWindowAcross * ampcor.numberWindowDown
    if gross_offset.size != 2 * windows_number:
        raise RuntimeError(
            "The input gross offset does not match the offset width*offset length"
        )
    gross_offset = gross_offset.reshape(windows_number, 2)
    ampcor.setVaryingGrossOffset(gross_offset[:, 0], gross_offset[:, 1])
ampcor.checkPixelInImageRange()
allocate_raw_bip_float32(out_dir / "dense_offsets", ampcor.numberWindowAcross, ampcor.numberWindowDown, 2)
allocate_raw_bip_float32(out_dir / "gross_offsets", ampcor.numberWindowAcross, ampcor.numberWindowDown, 2)
allocate_raw_bip_float32(out_dir / "snr", ampcor.numberWindowAcross, ampcor.numberWindowDown, 1)
allocate_raw_bip_float32(out_dir / "covariance", ampcor.numberWindowAcross, ampcor.numberWindowDown, 3)
allocate_raw_bip_float32(out_dir / "correlation_peak", ampcor.numberWindowAcross, ampcor.numberWindowDown, 1)
ampcor.runAmpcor()
print(json.dumps({
    "number_window_down": int(ampcor.numberWindowDown),
    "number_window_across": int(ampcor.numberWindowAcross),
    "reference_start_pixel_down": int(ampcor.referenceStartPixelDownStatic),
    "reference_start_pixel_across": int(ampcor.referenceStartPixelAcrossStatic),
    "skip_down": int(ampcor.skipSampleDown),
    "skip_across": int(ampcor.skipSampleAcross),
    "window_size_height": int(ampcor.windowSizeHeight),
    "window_size_width": int(ampcor.windowSizeWidth)
}))
'''
        env = os.environ.copy()
        try:
            result = subprocess.run(
                [sys.executable, "-c", helper, json.dumps(payload)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            _write_ampcor_log(out_dir / "pycuampcor.stdout.log", getattr(exc, "stdout", None))
            _write_ampcor_log(out_dir / "pycuampcor.stderr.log", getattr(exc, "stderr", None))
            raise
        _write_ampcor_log(out_dir / "pycuampcor.stdout.log", result.stdout)
        _write_ampcor_log(out_dir / "pycuampcor.stderr.log", result.stderr)
        if result.returncode != 0:
            return (None, None, None) if return_details else (None, None)
        metadata = json.loads(result.stdout.strip().splitlines()[-1])
        dense_offsets_path = out_dir / "dense_offsets"
        raw = np.fromfile(dense_offsets_path, dtype=np.float32)
        expected = (
            int(metadata["number_window_down"])
            * int(metadata["number_window_across"])
            * 2
        )
        if raw.size != expected:
            return (None, None, None) if return_details else (None, None)

        sparse = raw.reshape(
            int(metadata["number_window_down"]),
            int(metadata["number_window_across"]),
            2,
        )
        row_sparse = sparse[:, :, 0]
        col_sparse = sparse[:, :, 1]
        snr_raw = np.fromfile(out_dir / "snr", dtype=np.float32)
        snr = None
        if snr_raw.size == int(metadata["number_window_down"]) * int(metadata["number_window_across"]):
            snr = snr_raw.reshape(
                int(metadata["number_window_down"]),
                int(metadata["number_window_across"]),
            )
        covariance_raw = np.fromfile(out_dir / "covariance", dtype=np.float32)
        covariance_az = None
        covariance_rg = None
        expected_cov = int(metadata["number_window_down"]) * int(metadata["number_window_across"]) * 3
        if covariance_raw.size == expected_cov:
            covariance = covariance_raw.reshape(
                int(metadata["number_window_down"]),
                int(metadata["number_window_across"]),
                3,
            )
            covariance_az = covariance[:, :, 0]
            covariance_rg = covariance[:, :, 1]
        correlation = None
        correlation_path = out_dir / "correlation_peak"
        if correlation_path.exists():
            correlation_raw = np.fromfile(correlation_path, dtype=np.float32)
            expected_corr = int(metadata["number_window_down"]) * int(metadata["number_window_across"])
            if correlation_raw.size == expected_corr:
                correlation = correlation_raw.reshape(
                    int(metadata["number_window_down"]),
                    int(metadata["number_window_across"]),
                )
        row_coords = (
            int(metadata["reference_start_pixel_down"])
            + np.arange(int(metadata["number_window_down"]), dtype=np.float64)
            * int(metadata["skip_down"])
        )
        col_coords = (
            int(metadata["reference_start_pixel_across"])
            + np.arange(int(metadata["number_window_across"]), dtype=np.float64)
            * int(metadata["skip_across"])
        )
        row_offset = _interpolate_sparse_offsets(
            row_sparse,
            row_coords,
            col_coords,
            (rows, cols),
        )
        col_offset = _interpolate_sparse_offsets(
            col_sparse,
            row_coords,
            col_coords,
            (rows, cols),
        )
        np.save(out_dir / "dense_row_offsets.npy", row_offset.astype(np.float32))
        np.save(out_dir / "dense_col_offsets.npy", col_offset.astype(np.float32))
        diagnostics = {
            **metadata,
            "row_coords": row_coords.tolist(),
            "col_coords": col_coords.tolist(),
            "row_sparse_shape": list(row_sparse.shape),
            "col_sparse_shape": list(col_sparse.shape),
            "row_sparse_stats": _summarize_numeric_array(row_sparse),
            "col_sparse_stats": _summarize_numeric_array(col_sparse),
            "snr_stats": _summarize_numeric_array(snr),
            "correlation_stats": _summarize_numeric_array(correlation),
            "covariance_az_stats": _summarize_numeric_array(covariance_az),
            "covariance_rg_stats": _summarize_numeric_array(covariance_rg),
        }
        if gross_offset_filepath is not None:
            diagnostics["gross_offset_filepath"] = str(gross_offset_filepath)
        details = {
            "metadata": metadata,
            "row_sparse": row_sparse.astype(np.float32),
            "col_sparse": col_sparse.astype(np.float32),
            "row_coords": row_coords.astype(np.float64),
            "col_coords": col_coords.astype(np.float64),
            "snr": None if snr is None else snr.astype(np.float32),
            "correlation": None if correlation is None else correlation.astype(np.float32),
            "covariance_az": None if covariance_az is None else covariance_az.astype(np.float32),
            "covariance_rg": None if covariance_rg is None else covariance_rg.astype(np.float32),
            "diagnostics": diagnostics,
        }
        prepared, common_sparse_fit = _prepare_sparse_offsets_for_dense_model(
            details,
            out_shape=(rows, cols),
            min_points=6,
        )
        prepared["diagnostics"]["common_sparse_fit"] = common_sparse_fit
        if common_sparse_fit.get("success", False):
            row_offset = np.asarray(prepared["row_offset"], dtype=np.float32)
            col_offset = np.asarray(prepared["col_offset"], dtype=np.float32)
            details = prepared
        if return_details:
            return row_offset, col_offset, details
        return row_offset, col_offset
    except Exception:
        _write_ampcor_log(out_dir / "pycuampcor.exception.log", traceback.format_exc())
        return (None, None, None) if return_details else (None, None)
    finally:
        pass


def _resamp_with_isce3(
    *,
    input_slc: str,
    output_slc: str,
    rg_offset_raster: str | None = None,
    az_offset_raster: str | None = None,
    rg_offset_dataset: str | Path | np.ndarray | None = None,
    az_offset_dataset: str | Path | np.ndarray | None = None,
    radar_grid=None,
    doppler=None,
    azimuth_carrier=None,
    range_carrier=None,
    ref_radar_grid=None,
    use_gpu: bool,
) -> tuple[bool, str]:
    if radar_grid is None or ref_radar_grid is None:
        return False, "missing radar grid for isce3 resamp"

    return run_resamp_isce3(
        input_slc_path=input_slc,
        output_slc_path=output_slc,
        radar_grid=radar_grid,
        doppler=doppler,
        ref_radar_grid=ref_radar_grid,
        rg_offset_dataset=rg_offset_dataset if rg_offset_dataset is not None else rg_offset_raster,
        az_offset_dataset=az_offset_dataset if az_offset_dataset is not None else az_offset_raster,
        use_gpu=use_gpu,
        block_size_az=100,
        block_size_rg=0,
    )


def write_registration_outputs(
    *,
    stage_path: str | Path,
    slave_slc_path: str,
    coarse_coreg_slave_path: str | None = None,
    coarse_rg_offset_path: str | Path | None = None,
    coarse_az_offset_path: str | Path | None = None,
    row_offset=None,
    col_offset=None,
    dense_match_details: dict | None = None,
    source: str = "isce3-local-wrapper",
    use_gpu: bool = False,
    radar_grid=None,
    doppler=None,
    azimuth_carrier=None,
    range_carrier=None,
    ref_radar_grid=None,
) -> dict:
    stage_path = Path(stage_path)
    stage_path.mkdir(parents=True, exist_ok=True)

    coarse_coreg_slave = (
        Path(coarse_coreg_slave_path)
        if coarse_coreg_slave_path is not None
        else stage_path / "coarse_coreg_slave.tif"
    )
    if coarse_coreg_slave_path is None:
        if radar_grid is not None and ref_radar_grid is not None:
            ok = run_coarse_resamp_isce3_v2(
                slave_slc_path=slave_slc_path,
                coarse_coreg_slave_path=str(coarse_coreg_slave),
                radar_grid=radar_grid,
                doppler=doppler,
                ref_radar_grid=ref_radar_grid,
                use_gpu=use_gpu,
            )
            coarse_resamp_method = "isce3-v2-resample_slc_blocks"
            if not ok:
                _copy_raster(slave_slc_path, coarse_coreg_slave)
        else:
            _copy_raster(slave_slc_path, coarse_coreg_slave)
            coarse_resamp_method = "copy"
    else:
        coarse_resamp_method = "precomputed"

    grid_shape = None
    for candidate in (row_offset, col_offset):
        if isinstance(candidate, np.ndarray) and candidate.ndim == 2:
            grid_shape = candidate.shape
            break
    if grid_shape is None and ref_radar_grid is not None:
        grid_shape = (int(ref_radar_grid.length), int(ref_radar_grid.width))
    if grid_shape is None and radar_grid is not None:
        grid_shape = (int(radar_grid.length), int(radar_grid.width))
    if grid_shape is None:
        grid_shape = _raster_shape(coarse_coreg_slave) or _raster_shape(slave_slc_path) or (1, 1)

    if row_offset is None:
        row_offset = np.zeros(grid_shape, dtype=np.float32)
    if col_offset is None:
        col_offset = np.zeros(grid_shape, dtype=np.float32)

    dense_diag = dense_match_details.get("diagnostics") if dense_match_details else None
    dense_match_success = True
    dense_match_reason = None
    effective_source = source
    if dense_diag is not None:
        dense_match_success = str(dense_diag.get("status", "ok")).strip().lower() == "ok"
        dense_match_reason = dense_diag.get("reason")
        if not dense_match_success:
            effective_source = "fine_match_failed"

    raw_row_residual = np.asarray(row_offset, dtype=np.float32)
    raw_col_residual = np.asarray(col_offset, dtype=np.float32)
    residual_row = raw_row_residual
    residual_col = raw_col_residual
    residual_forced_zero = False
    dense_model_diagnostics = None
    prepared_dense_match_details = dense_match_details
    if dense_match_details and dense_match_success:
        try:
            prepared_dense_match_details, dense_model_diagnostics = _prepare_sparse_offsets_for_dense_model(
                dense_match_details,
                out_shape=grid_shape,
            )
            if dense_model_diagnostics.get("success", False):
                residual_row = np.asarray(prepared_dense_match_details["row_offset"], dtype=np.float32)
                residual_col = np.asarray(prepared_dense_match_details["col_offset"], dtype=np.float32)
            else:
                dense_match_success = False
                dense_match_reason = dense_model_diagnostics.get("reason")
                effective_source = "fine_match_failed"
        except Exception as exc:
            dense_model_diagnostics = {"enabled": True, "failed": str(exc)}
            dense_match_success = False
            dense_match_reason = f"dense_model_prepare_failed: {exc}"
            effective_source = "fine_match_failed"
            residual_row = raw_row_residual
            residual_col = raw_col_residual

    # Dense matching failure must not contaminate final offsets.
    if dense_match_details is not None and not dense_match_success:
        residual_row = np.zeros(grid_shape, dtype=np.float32)
        residual_col = np.zeros(grid_shape, dtype=np.float32)
        residual_forced_zero = True

    _write_offset_raster(stage_path / "azimuth_residual_raw.off.tif", raw_row_residual)
    _write_offset_raster(stage_path / "range_residual_raw.off.tif", raw_col_residual)
    az_residual_path = Path(_write_offset_raster(stage_path / "azimuth_residual.off.tif", residual_row))
    rg_residual_path = Path(_write_offset_raster(stage_path / "range_residual.off.tif", residual_col))

    coarse_row = np.zeros(grid_shape, dtype=np.float32)
    coarse_col = np.zeros(grid_shape, dtype=np.float32)
    coarse_valid_mask = np.ones(grid_shape, dtype=bool)
    has_coarse_offsets = coarse_rg_offset_path is not None and coarse_az_offset_path is not None
    if has_coarse_offsets:
        coarse_col, coarse_col_valid = _load_offset_dataset_with_valid_mask(
            coarse_rg_offset_path,
            grid_shape,
        )
        coarse_row, coarse_row_valid = _load_offset_dataset_with_valid_mask(
            coarse_az_offset_path,
            grid_shape,
        )
        coarse_valid_mask = coarse_col_valid & coarse_row_valid

    final_row_offset = coarse_row + residual_row
    final_col_offset = coarse_col + residual_col
    if has_coarse_offsets and not np.all(coarse_valid_mask):
        final_row_offset = np.array(final_row_offset, dtype=np.float32, copy=True)
        final_col_offset = np.array(final_col_offset, dtype=np.float32, copy=True)
        final_row_offset[~coarse_valid_mask] = GEO2RDR_OFFSET_NODATA
        final_col_offset[~coarse_valid_mask] = GEO2RDR_OFFSET_NODATA

    rg_offset_path = Path(
        _write_offset_raster(
            stage_path / "range.off.tif",
            np.asarray(final_col_offset, dtype=np.float64),
            dtype=gdal.GDT_Float64,
        )
    )
    az_offset_path = Path(
        _write_offset_raster(
            stage_path / "azimuth.off.tif",
            np.asarray(final_row_offset, dtype=np.float64),
            dtype=gdal.GDT_Float64,
        )
    )

    if dense_model_diagnostics is not None:
        (stage_path / "dense_model_diagnostics.json").write_text(
            json.dumps(dense_model_diagnostics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    diagnostics_to_write = (
        prepared_dense_match_details.get("diagnostics")
        if prepared_dense_match_details is not None
        else None
    )
    if diagnostics_to_write is not None:
        (stage_path / "pycuampcor_offsets_diagnostics.json").write_text(
            json.dumps(diagnostics_to_write, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    fine_coreg_slave = stage_path / "fine_coreg_slave.tif"
    fine_resamp_method = "copy"
    fine_input = (
        str(slave_slc_path)
        if has_coarse_offsets
        else str(coarse_coreg_slave if coarse_coreg_slave.exists() else Path(slave_slc_path))
    )
    fine_radar_grid = radar_grid if has_coarse_offsets and radar_grid is not None else ref_radar_grid
    fine_input_mode = "original_slave_slc" if has_coarse_offsets else "coarse_coreg_slave"
    nonzero_offsets = bool(
        np.any(np.abs(final_row_offset) > 0.0)
        or np.any(np.abs(final_col_offset) > 0.0)
    )
    if nonzero_offsets and _input_path_available(fine_input) and ref_radar_grid is not None:
        attempts = [use_gpu] if not use_gpu else [True, False]
        attempt_messages: list[str] = []
        ok = False
        for attempt_use_gpu in attempts:
            ok, attempt_method = _resamp_with_isce3(
                input_slc=str(fine_input),
                output_slc=str(fine_coreg_slave),
                rg_offset_dataset=final_col_offset,
                az_offset_dataset=final_row_offset,
                radar_grid=fine_radar_grid,
                doppler=doppler,
                azimuth_carrier=azimuth_carrier,
                range_carrier=range_carrier,
                ref_radar_grid=ref_radar_grid,
                use_gpu=attempt_use_gpu,
            )
            attempt_messages.append(attempt_method)
            if ok:
                fine_resamp_method = attempt_method
                if use_gpu and not attempt_use_gpu:
                    fine_resamp_method = (
                        f"{attempt_method} (gpu failed: {attempt_messages[0]})"
                    )
                break
        if not ok:
            fine_resamp_method = (
                attempt_messages[-1] if attempt_messages else fine_resamp_method
            )
            _copy_raster(str(fine_input), fine_coreg_slave)
    else:
        _copy_raster(str(fine_input), fine_coreg_slave)

    registration_model = stage_path / "registration_model.json"
    model = {
        "source": effective_source,
        "coarse_offsets": {
            "azimuth_path": None if coarse_az_offset_path is None else str(coarse_az_offset_path),
            "range_path": None if coarse_rg_offset_path is None else str(coarse_rg_offset_path),
            "azimuth_stats": _summarize_numeric_array(coarse_row),
            "range_stats": _summarize_numeric_array(coarse_col),
        },
        "residual_offsets": {
            "azimuth_path": str(az_residual_path),
            "range_path": str(rg_residual_path),
            "azimuth_raw_path": str(stage_path / "azimuth_residual_raw.off.tif"),
            "range_raw_path": str(stage_path / "range_residual_raw.off.tif"),
            "azimuth_stats": _summarize_numeric_array(residual_row),
            "range_stats": _summarize_numeric_array(residual_col),
        },
        "fine_fit": {
            "azimuth_poly": _polyfit2d(final_row_offset),
            "range_poly": _polyfit2d(final_col_offset),
        },
        "coarse_resamp_method": coarse_resamp_method,
        "fine_resamp_method": fine_resamp_method,
        "fine_resample_input": fine_input_mode,
        "use_gpu": use_gpu,
        "offset_shape": {"rows": int(grid_shape[0]), "cols": int(grid_shape[1])},
        "offset_magnitude": {
            "azimuth_max_abs": float(np.max(np.abs(final_row_offset))) if final_row_offset.size else 0.0,
            "range_max_abs": float(np.max(np.abs(final_col_offset))) if final_col_offset.size else 0.0,
        },
        "final_offsets": {
            "azimuth_stats": _summarize_numeric_array(final_row_offset),
            "range_stats": _summarize_numeric_array(final_col_offset),
        },
        "fit_quality": _summarize_fit_quality(residual_row, residual_col),
        "dense_match": {
            "success": bool(dense_match_success),
            "reason": dense_match_reason,
            "engine": None if dense_diag is None else dense_diag.get("engine"),
            "valid_points": None if dense_diag is None else dense_diag.get("valid_points"),
            "residual_forced_zero": bool(residual_forced_zero),
        },
        "dense_model": dense_model_diagnostics,
    }
    registration_model.write_text(json.dumps(model, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "coarse_coreg_slave": str(coarse_coreg_slave),
        "fine_coreg_slave": str(fine_coreg_slave),
        "registration_model": str(registration_model),
        "range_offsets": str(rg_offset_path),
        "azimuth_offsets": str(az_offset_path),
        "range_residual_offsets": str(rg_residual_path),
        "azimuth_residual_offsets": str(az_residual_path),
        "coarse_offsets_staged": True,
    }
