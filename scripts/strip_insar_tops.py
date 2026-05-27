"""strip_insar_tops — TOPS burst processing for Sentinel-1 InSAR.

Extracted from strip_insar.py for modularity.  Imported lazily by
strip_insar.py's ``process_strip_insar`` when ``tops_mode=True``.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from osgeo import gdal

# Import shared internals from strip_insar core
from strip_insar import (
    _gdal_path_exists,
    _load_cached_array,
    _load_cached_stage_outputs,
    _load_stage_output_path,
    _multilook_mean,
    _parse_datetime,
    _read_slc_block_as_complex,
    _run_slave_geo2rdr_from_master_topo,
    _save_stage_array,
    _select_registered_slave_slc,
    _write_complex_gtiff,
    _write_custom_stage_record,
    _write_float_gtiff,
    _write_stage_outputs_record,
    load_stage_record,
    resolve_manifest_data_path,
    resolve_manifest_metadata_path,
    stage_dir,
    utc_now_iso,
)

import tops_rtc


log = logging.getLogger(__name__)


# ── TOPS dataclasses ──────────────────────────────────────────────────────────


# ── TOPS-specific utility functions ────────────────────────────────────

def _slice_offset_to_valid_window(offset: np.ndarray | None, burst: TopsBurstInfo) -> np.ndarray | None:
    if offset is None:
        return None
    arr = np.asarray(offset)
    if arr.ndim != 2:
        return None
    row0 = int(burst.first_valid_line)
    col0 = int(burst.first_valid_sample)
    rows = int(burst.num_valid_lines)
    cols = int(burst.num_valid_samples)
    if rows <= 0 or cols <= 0:
        return None
    row1 = row0 + rows
    col1 = col0 + cols
    if row0 < 0 or col0 < 0 or row1 > arr.shape[0] or col1 > arr.shape[1]:
        row0 = max(0, min(arr.shape[0], row0))
        col0 = max(0, min(arr.shape[1], col0))
        row1 = max(row0, min(arr.shape[0], row1))
        col1 = max(col0, min(arr.shape[1], col1))
    out = np.asarray(arr[row0:row1, col0:col1], dtype=np.float32)
    return out if out.size else None




def _estimate_phasecorr_shift(
    reference: np.ndarray,
    secondary: np.ndarray,
    *,
    max_rows: int = 1024,
    max_cols: int = 2048,
    stride_az: int = 2,
    stride_rg: int = 4,
) -> tuple[float, float, float]:
    ref = np.asarray(reference, dtype=np.complex64)
    sec = np.asarray(secondary, dtype=np.complex64)
    rows = min(ref.shape[0], sec.shape[0])
    cols = min(ref.shape[1], sec.shape[1])
    if rows < 16 or cols < 16:
        return 0.0, 0.0, 0.0
    ref = ref[:rows, :cols]
    sec = sec[:rows, :cols]

    valid = (
        np.isfinite(ref.real)
        & np.isfinite(ref.imag)
        & np.isfinite(sec.real)
        & np.isfinite(sec.imag)
        & (np.abs(ref) > 0)
        & (np.abs(sec) > 0)
    )
    if not np.any(valid):
        return 0.0, 0.0, 0.0

    row_ids = np.where(np.any(valid, axis=1))[0]
    col_ids = np.where(np.any(valid, axis=0))[0]
    if row_ids.size < 16 or col_ids.size < 16:
        return 0.0, 0.0, 0.0

    r0 = int(row_ids[0])
    r1 = int(row_ids[-1]) + 1
    c0 = int(col_ids[0])
    c1 = int(col_ids[-1]) + 1
    if (r1 - r0) > max_rows:
        center = (r0 + r1) // 2
        half = max_rows // 2
        r0 = max(0, center - half)
        r1 = min(rows, r0 + max_rows)
        r0 = max(0, r1 - max_rows)
    if (c1 - c0) > max_cols:
        center = (c0 + c1) // 2
        half = max_cols // 2
        c0 = max(0, center - half)
        c1 = min(cols, c0 + max_cols)
        c0 = max(0, c1 - max_cols)

    ref_crop = ref[r0:r1, c0:c1]
    sec_crop = sec[r0:r1, c0:c1]
    valid_crop = valid[r0:r1, c0:c1]

    ref_obs = np.abs(ref_crop).astype(np.float32)
    sec_obs = np.abs(sec_crop).astype(np.float32)
    ref_obs[~valid_crop] = 0.0
    sec_obs[~valid_crop] = 0.0
    ref_obs = ref_obs[:: max(1, stride_az), :: max(1, stride_rg)]
    sec_obs = sec_obs[:: max(1, stride_az), :: max(1, stride_rg)]
    if ref_obs.shape[0] < 16 or ref_obs.shape[1] < 16:
        return 0.0, 0.0, 0.0

    ref_obs = ref_obs - float(np.mean(ref_obs, dtype=np.float64))
    sec_obs = sec_obs - float(np.mean(sec_obs, dtype=np.float64))

    fa = np.fft.fft2(ref_obs)
    fb = np.fft.fft2(sec_obs)
    cps = fa * np.conj(fb)
    denom = np.abs(cps)
    denom[denom <= 1.0e-9] = 1.0
    cps /= denom
    corr = np.fft.ifft2(cps)
    corr_abs = np.abs(corr)
    peak = int(np.argmax(corr_abs))
    pr, pc = np.unravel_index(peak, corr_abs.shape)
    pval = float(corr_abs[pr, pc])
    mean_corr = float(np.mean(corr_abs, dtype=np.float64)) + 1.0e-9
    snr = pval / mean_corr

    def _subpixel_parabola(v_m1: float, v0: float, v_p1: float) -> float:
        denom = (v_m1 - 2.0 * v0 + v_p1)
        if abs(denom) < 1.0e-9:
            return 0.0
        return 0.5 * (v_m1 - v_p1) / denom

    nrows, ncols = corr_abs.shape
    r_m1 = corr_abs[(pr - 1) % nrows, pc]
    r_0 = corr_abs[pr, pc]
    r_p1 = corr_abs[(pr + 1) % nrows, pc]
    c_m1 = corr_abs[pr, (pc - 1) % ncols]
    c_0 = corr_abs[pr, pc]
    c_p1 = corr_abs[pr, (pc + 1) % ncols]
    dr_sub = _subpixel_parabola(float(r_m1), float(r_0), float(r_p1))
    dc_sub = _subpixel_parabola(float(c_m1), float(c_0), float(c_p1))

    dy = float(pr if pr <= nrows // 2 else pr - nrows) + float(dr_sub)
    dx = float(pc if pc <= ncols // 2 else pc - ncols) + float(dc_sub)
    dy *= float(max(1, stride_az))
    dx *= float(max(1, stride_rg))
    return float(dy), float(dx), float(snr)




def _read_complex_slc_valid_window(slc_path: str, burst: TopsBurstInfo) -> np.ndarray:
    ds = gdal.Open(str(slc_path), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"failed to open SLC: {slc_path}")
    try:
        row0, row1, col0, col1 = _burst_valid_window_from_info(burst)
        rows = max(0, row1 - row0)
        cols = max(0, col1 - col0)
        if rows <= 0 or cols <= 0:
            return np.zeros((0, 0), dtype=np.complex64)
        if ds.RasterCount >= 2:
            i = _read_band_array(ds.GetRasterBand(1), col0, row0, cols, rows).astype(np.float32)
            q = _read_band_array(ds.GetRasterBand(2), col0, row0, cols, rows).astype(np.float32)
            return (i + 1j * q).astype(np.complex64)
        real = _read_band_array(ds.GetRasterBand(1), col0, row0, cols, rows)
        return np.asarray(real, dtype=np.complex64)
    finally:
        ds = None


# --- TOPS code moved to strip_insar_tops.py ---


@dataclass
class TopsBurstInfo:
    """Container for TOPS burst metadata."""
    burst_index: int
    line_offset: int
    number_of_lines: int
    number_of_samples: int
    first_valid_line: int
    num_valid_lines: int
    first_valid_sample: int
    num_valid_samples: int
    sensing_start: str | None = None
    azimuth_time_interval: float | None = None
    radar_wavelength: float | None = None


@dataclass
class TopsOverlapInfo:
    """Container for TOPS burst overlap metadata."""
    previous_burst_index: int
    next_burst_index: int
    estimated_overlap_lines: int
    overlap_start_line: int | None = None

# ── TOPS burst utility functions ─────────────────────────────────────

def _load_array(path: str | Path) -> np.ndarray:
    """Load a 2-D array from .npy or GDAL-readable raster."""
    p = Path(path)
    if p.suffix.lower() == ".npy":
        return np.load(str(p))
    ds = gdal.Open(str(p), gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(str(p))
    try:
        band = ds.GetRasterBand(1)
        arr = band.ReadAsArray()
    finally:
        ds = None
    return np.asarray(arr)


def _boxcar_mean2d(array: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Simple boxcar (uniform) 2-D mean filter."""
    from scipy.ndimage import uniform_filter
    return uniform_filter(array.astype(np.float64), size=window_size, mode="reflect")


def _burst_indices_from_infos(bursts: list[Any]) -> set[int]:
    indices = [int(getattr(burst, "burst_index", 0) or 0) for burst in bursts]
    if not indices:
        return set()
    if min(indices) == 0:
        return {idx + 1 for idx in indices}
    return {idx for idx in indices if idx > 0}


def _burst_grid_range_start(grid: dict[str, Any]) -> float:
    if "startingRange" in grid:
        return float(grid["startingRange"])
    if "rangeTimeFirstPixel" in grid:
        return float(grid["rangeTimeFirstPixel"]) * 299792458.0 / 2.0
    return 0.0


def _merged_burst_grid_json(bursts: list[dict[str, Any]]) -> dict[str, Any]:
    merged_fn = getattr(tops_rtc, "_merged_radar_grid_json", None)
    if callable(merged_fn):
        return merged_fn(bursts)
    if not bursts:
        raise ValueError("at least one burst is required")
    grids = [burst["radargrid"] for burst in bursts]
    dt = float(grids[0].get("rowSpacing", 0.0))
    dr = float(grids[0].get("columnSpacing", 0.0))
    if dt <= 0.0 or dr <= 0.0:
        raise ValueError("burst radar grid rowSpacing and columnSpacing must be positive")
    ref_start = min(float(grid["sensingStartGPSTime"]) for grid in grids)
    ref_range = min(_burst_grid_range_start(grid) for grid in grids)
    out_rows = 0
    out_cols = 0
    for grid in grids:
        row_off = int(round((float(grid["sensingStartGPSTime"]) - ref_start) / dt))
        col_off = int(round((_burst_grid_range_start(grid) - ref_range) / dr))
        first_line = int(grid.get("firstValidLine", 0))
        num_lines = int(grid.get("numValidLines", grid.get("numberOfRows", 0)))
        first_sample = int(grid.get("firstValidSample", 0))
        num_samples = int(grid.get("numValidSamples", grid.get("numberOfColumns", 0)))
        number_rows = int(grid.get("numberOfRows", first_line + num_lines))
        number_cols = int(grid.get("numberOfColumns", first_sample + num_samples))
        out_rows = max(out_rows, row_off + number_rows)
        out_cols = max(out_cols, col_off + number_cols)

    merged = dict(grids[0])
    merged.update(
        {
            "source": "sentinel-1-merged-tops-bursts",
            "burstCount": len(bursts),
            "numberOfRows": int(out_rows),
            "numberOfColumns": int(out_cols),
            "sensingStartGPSTime": float(ref_start),
            "startingRange": float(ref_range),
            "firstValidLine": 0,
            "numValidLines": int(out_rows),
            "firstValidSample": 0,
            "numValidSamples": int(out_cols),
        }
    )
    if "rangeTimeFirstPixel" in merged:
        import isce3.core

        merged["rangeTimeFirstPixel"] = 2.0 * float(ref_range) / isce3.core.speed_of_light
    return merged


def _merge_burst_scalar_fields(
    bursts: list[dict[str, Any]],
    data_paths: list[str],
    *,
    output_path: Path,
    fill_value: float = np.nan,
    invalid_mask_fn=None,
) -> str:
    if not bursts:
        raise ValueError("at least one burst is required")
    if len(bursts) != len(data_paths):
        raise ValueError("burst metadata and data path lengths do not match")

    merged_grid = _merged_burst_grid_json(bursts)
    rows = int(merged_grid["numberOfRows"])
    cols = int(merged_grid["numberOfColumns"])
    ref_start = float(merged_grid["sensingStartGPSTime"])
    ref_range = _burst_grid_range_start(merged_grid)
    dt = float(bursts[0]["radargrid"].get("rowSpacing", 0.0))
    dr = float(bursts[0]["radargrid"].get("columnSpacing", 0.0))
    if dt <= 0.0 or dr <= 0.0:
        raise ValueError("burst radar grid rowSpacing and columnSpacing must be positive")

    out = np.full((rows, cols), fill_value, dtype=np.float64)
    valid_mask = np.zeros((rows, cols), dtype=bool)
    for burst, data_path in zip(bursts, data_paths, strict=False):
        arr = np.asarray(_load_array(data_path), dtype=np.float64)
        grid = burst["radargrid"]
        row_off = int(round((float(grid["sensingStartGPSTime"]) - ref_start) / dt))
        col_off = int(round((_burst_grid_range_start(grid) - ref_range) / dr))
        first_line = int(grid.get("firstValidLine", 0))
        num_lines = int(grid.get("numValidLines", grid.get("numberOfRows", arr.shape[0])))
        first_sample = int(grid.get("firstValidSample", 0))
        num_samples = int(grid.get("numValidSamples", grid.get("numberOfColumns", arr.shape[1])))
        dst_row = row_off + first_line
        dst_col = col_off + first_sample
        src_rows = min(arr.shape[0], num_lines)
        src_cols = min(arr.shape[1], num_samples)
        if src_rows <= 0 or src_cols <= 0:
            continue
        dst_row_end = min(rows, dst_row + src_rows)
        dst_col_end = min(cols, dst_col + src_cols)
        src_rows = dst_row_end - dst_row
        src_cols = dst_col_end - dst_col
        if src_rows <= 0 or src_cols <= 0:
            continue
        src = arr[:src_rows, :src_cols]
        valid = invalid_mask_fn(src) if invalid_mask_fn is not None else np.isfinite(src)
        target = out[dst_row:dst_row_end, dst_col:dst_col_end]
        target_mask = valid_mask[dst_row:dst_row_end, dst_col:dst_col_end]
        write_mask = valid & ~target_mask
        if np.any(write_mask):
            target[write_mask] = src[write_mask]
            target_mask[write_mask] = True
            out[dst_row:dst_row_end, dst_col:dst_col_end] = target
            valid_mask[dst_row:dst_row_end, dst_col:dst_col_end] = target_mask

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return _write_float_gtiff(output_path, out.astype(np.float32), dtype=gdal.GDT_Float32, nodata=np.nan)


def _write_burst_topo_mosaic(
    burst_records: list[dict[str, Any]],
    *,
    output_dir: Path,
    name: str,
) -> str:
    if not burst_records:
        raise ValueError("at least one burst record is required")

    output_dir.mkdir(parents=True, exist_ok=True)
    x_paths = [str(Path(rec["topo_dir"]) / "x.tif") for rec in burst_records]
    y_paths = [str(Path(rec["topo_dir"]) / "y.tif") for rec in burst_records]
    z_paths = [str(Path(rec["topo_dir"]) / "z.tif") for rec in burst_records]
    bursts = [rec["burst"] for rec in burst_records]

    merged_x = _merge_burst_scalar_fields(
        bursts,
        x_paths,
        output_path=output_dir / "x.tif",
    )
    merged_y = _merge_burst_scalar_fields(
        bursts,
        y_paths,
        output_path=output_dir / "y.tif",
    )
    merged_z = _merge_burst_scalar_fields(
        bursts,
        z_paths,
        output_path=output_dir / "z.tif",
    )

    return _build_topo_vrt(output_dir, epsg=4326)



def _prepare_tops_burst_plan(
    *,
    context: PairContext,
    master_bursts: list[TopsBurstInfo],
    output_dir: Path,
) -> dict[str, Any]:
    slc_path = resolve_manifest_data_path(
        context.master_manifest_path, (context.master_manifest.get("slc") or {}).get("path")
    )
    if not slc_path:
        raise RuntimeError("missing master slc path")

    acq_start = (
        context.master_acq_data.get("startTimeUTC")
        or context.master_acq_data.get("start_time_utc")
        or context.master_acq_data.get("startTime")
        or context.master_acq_data.get("start_time")
    )
    start_dt = _parse_datetime(str(acq_start)) if acq_start else None
    gps_epoch = datetime(1980, 1, 6, tzinfo=timezone.utc)
    acq_start_gps = (
        (start_dt - gps_epoch).total_seconds() if start_dt is not None else float(context.master_acq_data.get("startGPSTime", 0.0) or 0.0)
    )

    row_spacing = float(context.master_rg_data.get("rowSpacing", 0.0) or 0.0)
    if row_spacing <= 0:
        row_spacing = float(context.master_acq_data.get("prf", 0.0) or 0.0)
        row_spacing = (1.0 / row_spacing) if row_spacing > 0 else 0.0
    col_spacing = float(context.master_rg_data.get("columnSpacing", 0.0) or 0.0)
    if col_spacing <= 0:
        col_spacing = 2.32956

    bursts: list[dict[str, Any]] = []
    for burst in master_bursts:
        burst_index = int(burst.burst_index)
        line_offset = int(burst.line_offset)
        nrows = int(burst.number_of_lines)
        ncols = int(burst.number_of_samples)
        fvl = int(burst.first_valid_line)
        nvl = int(burst.num_valid_lines)
        fvs = int(burst.first_valid_sample)
        nvs = int(burst.num_valid_samples)
        burst_start = acq_start_gps + line_offset * row_spacing
        burst_dir = output_dir / f"burst_{burst_index:03d}"
        bursts.append(
            {
                "burstIndex": burst_index + 1,
                "radargrid": {
                    "source": "sentinel-1-burst",
                    "burstIndex": burst_index + 1,
                    "numberOfRows": nrows,
                    "numberOfColumns": ncols,
                    "rowSpacing": row_spacing,
                    "columnSpacing": col_spacing,
                    "rangeTimeFirstPixel": float(context.master_rg_data.get("rangeTimeFirstPixel", 0.0) or 0.0),
                    "startingRange": float(context.master_rg_data.get("startingRange", 0.0) or 0.0),
                    "prf": float(context.master_acq_data.get("prf", 0.0) or 0.0),
                    "wavelength": float(getattr(burst, "radar_wavelength", None) or context.wavelength or 0.0),
                    "sensingStartGPSTime": float(burst_start),
                    "lineOffset": line_offset,
                    "firstValidLine": fvl,
                    "numValidLines": nvl,
                    "firstValidSample": fvs,
                    "numValidSamples": nvs,
                    "swath": context.master_manifest.get("tops", {}).get("swath"),
                    "polarisation": context.master_manifest.get("tops", {}).get("polarisation"),
                    "lookDirection": context.master_acq_data.get("lookDirection", "RIGHT"),
                },
                "doppler": {"coefficients": [0.0], "t0": 0.0},
                "slcWindow": {
                    "xoff": 0,
                    "yoff": line_offset,
                    "xsize": ncols,
                    "ysize": nrows,
                    "validWindow": {
                        "xoff": fvs,
                        "yoff": fvl,
                        "xsize": nvs,
                        "ysize": nvl,
                    },
                },
                "outputs": {
                    "directory": str(burst_dir),
                    "topo_h5": str(burst_dir / "topo.h5"),
                    "amplitude_h5": str(burst_dir / "topo.h5"),
                    "metadata_json": str(burst_dir / "metadata.json"),
                },
            }
        )

    plan = {
        "version": "1.0",
        "mode": "tops-burst-geometry",
        "sensor": "sentinel-1",
        "swath": context.master_manifest.get("tops", {}).get("swath"),
        "polarisation": context.master_manifest.get("tops", {}).get("polarisation"),
        "input_manifest": str(context.master_manifest_path),
        "slc_path": str(slc_path),
        "burst_count": len(bursts),
        "bursts": bursts,
    }
    plan_path = output_dir / "tops_burst_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    plan["plan_path"] = str(plan_path)
    return plan


def run_tops_burst_geometry_stage(
    context: PairContext,
    *,
    master_bursts: list[TopsBurstInfo],
    slave_bursts: list[TopsBurstInfo],
    gpu_mode: str,
    gpu_id: int,
    block_rows: int,
) -> tuple[dict[str, Any], str, str | None]:
    if not master_bursts:
        raise RuntimeError("tops burst geometry requires non-empty master_bursts")
    if not slave_bursts:
        raise RuntimeError("tops burst geometry requires non-empty slave_bursts")

    p0_dir = stage_dir(context.pair_dir, "p0")
    p1_dir = stage_dir(context.pair_dir, "p1")
    p0_dir.mkdir(parents=True, exist_ok=True)
    p1_dir.mkdir(parents=True, exist_ok=True)

    plan = _prepare_tops_burst_plan(context=context, master_bursts=master_bursts, output_dir=p0_dir)
    plan_path = str(plan["plan_path"])
    topo_result = tops_rtc.compute_burst_topo(
        plan_path,
        context.resolved_dem,
        burst_limit=len(master_bursts),
        block_rows=block_rows,
        orbit_interp=context.orbit_interp,
        use_gpu=(gpu_mode != "cpu"),
        gpu_id=gpu_id,
    )
    burst_results = topo_result.get("bursts", []) if isinstance(topo_result, dict) else []
    if len(burst_results) != len(master_bursts):
        raise RuntimeError("burst topo result count does not match master bursts")

    slave_index = {int(b.burst_index): b for b in slave_bursts}
    burst_geo2rdr_records: list[dict[str, Any]] = []
    for burst_info, topo_item in zip(master_bursts, burst_results, strict=False):
        burst_index = int(burst_info.burst_index)
        slave_info = slave_index.get(burst_index, burst_info)
        topo_h5 = str(topo_item.get("topo_h5") or topo_item.get("amplitude_h5") or "")
        if not topo_h5:
            raise RuntimeError("missing burst topo HDF in tops_rtc output")
        topo_dir = p1_dir / f"burst_{burst_index:03d}" / "topo_rasters"
        topo_dir.mkdir(parents=True, exist_ok=True)
        with h5py.File(topo_h5, "r") as f:
            lon = np.asarray(f["longitude"][:], dtype=np.float64)
            lat = np.asarray(f["latitude"][:], dtype=np.float64)
            hgt = np.asarray(f["height"][:], dtype=np.float64)
        x_path = _write_float_gtiff(topo_dir / "x.tif", lon.astype(np.float64), dtype=gdal.GDT_Float64, nodata=np.nan)
        y_path = _write_float_gtiff(topo_dir / "y.tif", lat.astype(np.float64), dtype=gdal.GDT_Float64, nodata=np.nan)
        z_path = _write_float_gtiff(topo_dir / "z.tif", hgt.astype(np.float64), dtype=gdal.GDT_Float64, nodata=np.nan)
        topo_vrt = _build_topo_vrt(topo_dir, epsg=4326)

        burst_offset_dir = p1_dir / f"burst_{burst_index:03d}" / "geo2rdr"
        burst_offset_dir.mkdir(parents=True, exist_ok=True)
        coarse_rg_offset_path, coarse_az_offset_path = _run_slave_geo2rdr_from_master_topo(
            master_topo_vrt_path=topo_vrt,
            slave_orbit_data=context.slave_orbit_data,
            slave_acq_data=context.slave_acq_data,
            slave_rg_data=context.slave_rg_data,
            slave_dop_data=context.slave_dop_data,
            output_dir=burst_offset_dir,
            use_gpu=(gpu_mode != "cpu"),
            gpu_id=gpu_id,
            block_rows=block_rows,
            orbit_interp=context.orbit_interp,
        )
        burst_geo2rdr_records.append(
            {
                "burst_index": burst_index,
                "slave_burst_index": int(slave_info.burst_index),
                "topo_h5": topo_h5,
                "topo_vrt": topo_vrt,
                "topo_x": x_path,
                "topo_y": y_path,
                "topo_z": z_path,
                "coarse_geo2rdr_range_offsets": coarse_rg_offset_path,
                "coarse_geo2rdr_azimuth_offsets": coarse_az_offset_path,
            }
        )

    p0_outputs = {
        "tops_burst_plan": plan_path,
        "tops_burst_topo": str(p0_dir),
        "burst_count": len(master_bursts),
    }
    _write_stage_outputs_record(
        output_dir=context.pair_dir,
        stage="p0",
        master_manifest_path=context.master_manifest_path,
        slave_manifest_path=context.slave_manifest_path,
        backend_used="cpu",
        output_files=p0_outputs,
    )
    p1_outputs = {
        "tops_burst_geo2rdr_json": str(p1_dir / "burst_geo2rdr.json"),
        "burst_geo2rdr_records": burst_geo2rdr_records,
        "burst_count": len(master_bursts),
    }
    Path(p1_outputs["tops_burst_geo2rdr_json"]).write_text(
        json.dumps(p1_outputs, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_stage_outputs_record(
        output_dir=context.pair_dir,
        stage="p1",
        master_manifest_path=context.master_manifest_path,
        slave_manifest_path=context.slave_manifest_path,
        backend_used="cpu",
        output_files=p1_outputs,
    )
    return p1_outputs, "cpu", None



def _burst_valid_window_from_info(burst: TopsBurstInfo) -> tuple[int, int, int, int]:
    row0 = int(burst.line_offset) + int(burst.first_valid_line)
    row1 = row0 + int(burst.num_valid_lines)
    col0 = int(burst.first_valid_sample)
    col1 = col0 + int(burst.num_valid_samples)
    return row0, row1, col0, col1


def _align_tops_valid_window(reference: TopsBurstInfo, secondary: TopsBurstInfo) -> tuple[TopsBurstInfo, TopsBurstInfo]:
    ref_first_line = int(reference.first_valid_line)
    ref_last_line = ref_first_line + int(reference.num_valid_lines) - 1
    ref_first_sample = int(reference.first_valid_sample)
    ref_last_sample = ref_first_sample + int(reference.num_valid_samples) - 1

    sec_first_line = int(secondary.first_valid_line)
    sec_last_line = sec_first_line + int(secondary.num_valid_lines) - 1
    sec_first_sample = int(secondary.first_valid_sample)
    sec_last_sample = sec_first_sample + int(secondary.num_valid_samples) - 1

    first_line = max(ref_first_line, sec_first_line)
    last_line = min(ref_last_line, sec_last_line)
    first_sample = max(ref_first_sample, sec_first_sample)
    last_sample = min(ref_last_sample, sec_last_sample)

    num_lines = max(0, last_line - first_line + 1)
    num_samples = max(0, last_sample - first_sample + 1)

    ref_out = TopsBurstInfo(
        burst_index=int(reference.burst_index),
        line_offset=int(reference.line_offset),
        number_of_lines=int(reference.number_of_lines),
        number_of_samples=int(reference.number_of_samples),
        first_valid_line=first_line,
        num_valid_lines=num_lines,
        first_valid_sample=first_sample,
        num_valid_samples=num_samples,
        sensing_start=reference.sensing_start,
        azimuth_time_interval=reference.azimuth_time_interval,
        radar_wavelength=reference.radar_wavelength,
    )
    sec_out = TopsBurstInfo(
        burst_index=int(secondary.burst_index),
        line_offset=int(secondary.line_offset),
        number_of_lines=int(secondary.number_of_lines),
        number_of_samples=int(secondary.number_of_samples),
        first_valid_line=first_line,
        num_valid_lines=num_lines,
        first_valid_sample=first_sample,
        num_valid_samples=num_samples,
        sensing_start=secondary.sensing_start,
        azimuth_time_interval=secondary.azimuth_time_interval,
        radar_wavelength=secondary.radar_wavelength,
    )
    return ref_out, sec_out



def run_burst_ifg_stage(
    context: PairContext,
    *,
    master_bursts: list[TopsBurstInfo],
    slave_bursts: list[TopsBurstInfo],
    range_looks: int = 1,
    azimuth_looks: int = 1,
    esd_azimuth_offsets: list[float] | None = None,
) -> tuple[dict, str, str | None]:
    processing_options = {
        "range_looks": int(range_looks),
        "azimuth_looks": int(azimuth_looks),
        "esd_azimuth_offsets": [float(v) for v in esd_azimuth_offsets] if esd_azimuth_offsets is not None else None,
    }
    cached_outputs = _load_cached_stage_outputs(
        context.pair_dir,
        "p2",
        required_keys=("burst_interferograms", "burst_coherences"),
        expected_processing_options=processing_options,
    )
    if cached_outputs is not None:
        return cached_outputs, "cache", None

    master_slc = resolve_manifest_data_path(
        context.master_manifest_path,
        context.master_manifest["slc"]["path"],
    )
    slave_slc = _select_registered_slave_slc(
        load_stage_record(context.pair_dir, "p1").get("output_files") if load_stage_record(context.pair_dir, "p1") else None,
        resolve_manifest_data_path(
            context.slave_manifest_path,
            context.slave_manifest["slc"]["path"],
        ),
    )

    if not master_slc or not slave_slc:
        raise RuntimeError("missing master/slave SLC path for burst IFG stage")

    slave_index = {int(b.burst_index): b for b in slave_bursts}
    bursts = list(master_bursts)
    if not bursts:
        raise RuntimeError("burst IFG stage requires non-empty master_bursts")


def run_tops_burst_fine_offsets_stage(
    context: PairContext,
    *,
    master_bursts: list[TopsBurstInfo],
    slave_bursts: list[TopsBurstInfo],
) -> tuple[dict[str, Any], str, str | None]:
    p1_record = load_stage_record(context.pair_dir, "p1") or {}
    p1_outputs = dict(p1_record.get("output_files") or {})
    burst_geo2rdr_records = p1_outputs.get("burst_geo2rdr_records") or []
    if not isinstance(burst_geo2rdr_records, list):
        burst_geo2rdr_records = []

    master_slc = resolve_manifest_data_path(context.master_manifest_path, (context.master_manifest.get("slc") or {}).get("path"))
    slave_slc = resolve_manifest_data_path(context.slave_manifest_path, (context.slave_manifest.get("slc") or {}).get("path"))
    if not master_slc or not slave_slc:
        return p1_outputs, "cpu", "missing master/slave SLC path for burst fine registration"
    if not _gdal_path_exists(master_slc) or not _gdal_path_exists(slave_slc):
        return p1_outputs, "cpu", "master/slave SLC not accessible; skip burst fine registration"

    slave_index = {int(b.burst_index): b for b in slave_bursts}
    coarse_index: dict[int, tuple[np.ndarray | None, np.ndarray | None]] = {}
    for item in burst_geo2rdr_records:
        if not isinstance(item, dict):
            continue
        idx = int(item.get("burst_index", -1))
        if idx < 0:
            continue
        rg_off = _load_float_offset_raster(item.get("coarse_geo2rdr_range_offsets"))
        az_off = _load_float_offset_raster(item.get("coarse_geo2rdr_azimuth_offsets"))
        coarse_index[idx] = (rg_off, az_off)

    fine_records: list[dict[str, Any]] = []
    for mb in master_bursts:
        sb = slave_index.get(int(mb.burst_index), mb)
        mb_aligned, sb_aligned = _align_tops_valid_window(mb, sb)
        m_win = _read_complex_slc_valid_window(master_slc, mb_aligned)
        s_win = _read_complex_slc_valid_window(slave_slc, sb_aligned)
        rg_off, az_off = coarse_index.get(int(mb.burst_index), (None, None))
        rg_win = _slice_offset_to_valid_window(rg_off, mb_aligned)
        az_win = _slice_offset_to_valid_window(az_off, mb_aligned)
        if rg_win is not None or az_win is not None:
            s_base = _resample_complex_with_offsets(s_win, rg_win, az_win)
        else:
            s_base = np.asarray(s_win, dtype=np.complex64)

        rows = min(m_win.shape[0], s_base.shape[0])
        cols = min(m_win.shape[1], s_base.shape[1])
        window_info = {
            "master_rows": int(m_win.shape[0]),
            "master_cols": int(m_win.shape[1]),
            "slave_rows": int(s_win.shape[0]),
            "slave_cols": int(s_win.shape[1]),
            "aligned_rows": rows,
            "aligned_cols": cols,
        }
        if rows < 32 or cols < 32:
            fine_records.append(
                {
                    "burst_index": int(mb.burst_index),
                    "fine_azimuth_offset_px": 0.0,
                    "fine_range_offset_px": 0.0,
                    "snr": 0.0,
                    "method": "spectral-phasecorr",
                    "status": "insufficient_window",
                    "window": window_info,
                    "coarse_offset_applied": rg_off is not None or az_off is not None,
                }
            )
            continue
        dy, dx, snr = _estimate_phasecorr_shift(m_win[:rows, :cols], s_base[:rows, :cols])
        if not np.isfinite(dy) or not np.isfinite(dx):
            dy = 0.0
            dx = 0.0
        if abs(float(dy)) > 64.0 or abs(float(dx)) > 64.0:
            dy = 0.0
            dx = 0.0
        fine_records.append(
            {
                "burst_index": int(mb.burst_index),
                "fine_azimuth_offset_px": float(dy),
                "fine_range_offset_px": float(dx),
                "snr": float(snr),
                "method": "spectral-phasecorr",
                "status": "ok",
                "window": window_info,
                "coarse_offset_applied": rg_off is not None or az_off is not None,
            }
        )

    fine_json_path = stage_dir(context.pair_dir, "p1") / "burst_fine_offsets.json"
    fine_json_payload = {
        "method": "spectral-phasecorr",
        "burst_count": len(fine_records),
        "records": fine_records,
        "summary": {
            "total_bursts": len(fine_records),
            "ok_count": sum(1 for r in fine_records if r.get("status") == "ok"),
            "insufficient_window_count": sum(1 for r in fine_records if r.get("status") == "insufficient_window"),
            "all_offsets_applied": all(r.get("coarse_offset_applied", False) for r in fine_records),
        },
    }
    fine_json_path.write_text(json.dumps(fine_json_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    p1_outputs["burst_fine_offsets_json"] = str(fine_json_path)
    p1_outputs["burst_fine_offsets"] = fine_records

    _write_stage_outputs_record(
        output_dir=context.pair_dir,
        stage="p1",
        master_manifest_path=context.master_manifest_path,
        slave_manifest_path=context.slave_manifest_path,
        backend_used="cpu",
        output_files=p1_outputs,
        processing_options={"burst_fine_offset_method": "spectral-phasecorr"},
    )
    return p1_outputs, "cpu", None


def _inject_constant_azimuth_esd_offset(
    *,
    azimuth_offset_path: str,
    output_path: Path,
    esd_azimuth_offset_px: float,
) -> str:
    ds = gdal.Open(str(azimuth_offset_path), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"failed to open coarse azimuth offset raster: {azimuth_offset_path}")
    try:
        arr = _read_band_array(ds.GetRasterBand(1), dtype=np.float64).astype(np.float64)
    finally:
        ds = None
    valid = np.isfinite(arr)
    valid &= arr != GEO2RDR_OFFSET_NODATA
    valid &= arr >= GEO2RDR_OFFSET_INVALID_LOW
    arr_out = arr.copy()
    arr_out[valid] += float(esd_azimuth_offset_px)
    return _write_float_gtiff(output_path, arr_out.astype(np.float32), nodata=float(GEO2RDR_OFFSET_NODATA))


def _apply_esd_azimuth_shift_to_slc(
    *,
    input_slc_path: str,
    output_dir: Path,
    azimuth_shift_pixels: float,
) -> str:
    ds = gdal.Open(str(input_slc_path), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"failed to open SLC for ESD shift: {input_slc_path}")
    try:
        rows = int(ds.RasterYSize)
        cols = int(ds.RasterXSize)
        arr = _read_slc_block_as_complex(ds, 0, rows, cols).astype(np.complex64)
    finally:
        ds = None
    if rows <= 0 or cols <= 0:
        raise RuntimeError("invalid SLC shape for ESD shift")

    y = np.arange(rows, dtype=np.float32)
    src_y = y - np.float32(azimuth_shift_pixels)
    src0 = np.floor(src_y).astype(np.int32)
    frac = (src_y - src0).astype(np.float32)
    src1 = src0 + 1
    valid = (src0 >= 0) & (src1 < rows)

    shifted = np.zeros_like(arr, dtype=np.complex64)
    if np.any(valid):
        idx0 = src0[valid]
        idx1 = src1[valid]
        w = frac[valid][:, None].astype(np.float32)
        shifted[valid, :] = (1.0 - w) * arr[idx0, :] + w * arr[idx1, :]

    out_path = Path(output_dir) / "slave_esd_shifted.tif"
    _write_complex_gtiff(out_path, shifted)
    return str(out_path)


def _load_tops_bursts_from_manifest(manifest_path: Path, manifest: dict) -> list[dict]:
    tops_meta_path = resolve_manifest_metadata_path(manifest_path, manifest, "tops")
    if not tops_meta_path:
        return []
    try:
        data = json.loads(Path(tops_meta_path).read_text(encoding="utf-8"))
    except Exception:
        return []
    bursts = data.get("bursts", [])
    return bursts if isinstance(bursts, list) else []


def _sinc_interpolate_kernel(x: np.ndarray, a: int = 3) -> np.ndarray:
    """加窗 sinc 插值核（Lanczos 核）。

    Args:
        x: 输入坐标数组
        a: Lanczos 参数，决定截断半径

    Returns:
        插值核权重
    """
    x = np.asarray(x, dtype=np.float32)
    result = np.zeros_like(x, dtype=np.float32)
    small = np.abs(x) < 1.0e-6
    result[small] = 1.0
    nonzero = ~small & (np.abs(x) <= float(a))
    if np.any(nonzero):
        x_nz = x[nonzero]
        arg = np.pi * x_nz
        window = 2.0 * a * np.sin(arg) * np.sin(arg / a) / (arg * arg + 1.0e-10)
        sinc_term = np.sin(arg) / (arg + 1.0e-10)
        result[nonzero] = window * sinc_term
    return result


def _resample_complex_with_offsets(
    slave: np.ndarray,
    range_off: np.ndarray | None,
    az_off: np.ndarray | None,
    *,
    method: str = "lanczos",
    lanczos_a: int = 3,
) -> np.ndarray:
    """使用 offset 场对 slave SLC 进行重采样。

    Args:
        slave: 输入的复数 SLC 数据
        range_off: range 方向偏移量（像素）
        az_off: azimuth 方向偏移量（像素）
        method: 插值方法，"bilinear" 或 "lanczos"
        lanczos_a: Lanczos 核的截断半径

    Returns:
        重采样后的复数数据
    """
    from scipy.ndimage import map_coordinates

    src = np.asarray(slave, dtype=np.complex64)
    rows, cols = src.shape[:2]
    if rows <= 0 or cols <= 0:
        return src.copy()
    if range_off is None and az_off is None:
        return src.copy()

    rg = np.zeros((rows, cols), dtype=np.float32) if range_off is None else np.asarray(range_off, dtype=np.float32)
    az = np.zeros((rows, cols), dtype=np.float32) if az_off is None else np.asarray(az_off, dtype=np.float32)
    rows = min(rows, rg.shape[0], az.shape[0])
    cols = min(cols, rg.shape[1], az.shape[1])
    src = src[:rows, :cols]
    rg = rg[:rows, :cols]
    az = az[:rows, :cols]

    yy, xx = np.meshgrid(np.arange(rows, dtype=np.float32), np.arange(cols, dtype=np.float32), indexing="ij")
    src_y = yy - az
    src_x = xx - rg

    if method == "bilinear":
        finite = np.isfinite(src_y) & np.isfinite(src_x)
        y0 = np.zeros((rows, cols), dtype=np.int32)
        x0 = np.zeros((rows, cols), dtype=np.int32)
        y0[finite] = np.floor(src_y[finite]).astype(np.int32)
        x0[finite] = np.floor(src_x[finite]).astype(np.int32)
        y1 = y0 + 1
        x1 = x0 + 1
        wy = (src_y - y0).astype(np.float32)
        wx = (src_x - x0).astype(np.float32)

        valid = finite & (y0 >= 0) & (x0 >= 0) & (y1 < rows) & (x1 < cols)
        out = np.zeros((rows, cols), dtype=np.complex64)
        if not np.any(valid):
            return out

        flat = src.reshape(-1)
        idx00 = y0[valid] * cols + x0[valid]
        idx01 = y0[valid] * cols + x1[valid]
        idx10 = y1[valid] * cols + x0[valid]
        idx11 = y1[valid] * cols + x1[valid]
        w00 = (1.0 - wy[valid]) * (1.0 - wx[valid])
        w01 = (1.0 - wy[valid]) * wx[valid]
        w10 = wy[valid] * (1.0 - wx[valid])
        w11 = wy[valid] * wx[valid]
        out[valid] = (
            flat[idx00] * w00 + flat[idx01] * w01 + flat[idx10] * w10 + flat[idx11] * w11
        ).astype(np.complex64)
        return out

    # Lanczos (sinc) interpolation using scipy.ndimage.map_coordinates
    # 将复数数组拆分为实部和虚部分别插值
    finite = np.isfinite(src_y) & np.isfinite(src_x)
    if not np.any(finite):
        return np.zeros((rows, cols), dtype=np.complex64)

    # 坐标（scipy 的坐标顺序是 (x, y) 即 (cols, rows)）
    coords = np.array([src_x.flatten(), src_y.flatten()], dtype=np.float64)

    # 分别对实部和虚部进行插值
    order = min(int(lanczos_a), 5)  # scipy 的 order 参数范围是 0-5
    real_interp = map_coordinates(src.real, coords, order=order, mode="constant", cval=0.0)
    imag_interp = map_coordinates(src.imag, coords, order=order, mode="constant", cval=0.0)

    # 处理无效像素
    valid_mask = finite.flatten()
    real_interp[~valid_mask] = 0.0
    imag_interp[~valid_mask] = 0.0

    out = (real_interp + 1j * imag_interp).reshape(rows, cols).astype(np.complex64)
    return out



def _repair_burst_seams_isce2_style(
    *,
    interferogram: np.ndarray,
    coherence: np.ndarray,
    segments: list[tuple[int, int, int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    if not segments:
        return interferogram, coherence
    out_ifg = interferogram.copy()
    out_coh = coherence.copy()
    rows, cols = out_coh.shape
    valid_mask = np.zeros((rows, cols), dtype=bool)
    for y0, y1, x0, x1 in segments:
        valid_mask[y0:y1, x0:x1] = True

    # Fill small gaps between adjacent burst valid windows using boundary interpolation.
    for i in range(len(segments) - 1):
        py0, py1, px0, px1 = segments[i]
        cy0, cy1, cx0, cx1 = segments[i + 1]
        gy0 = py1
        gy1 = cy0
        if gy1 <= gy0:
            continue
        ox0 = max(px0, cx0)
        ox1 = min(px1, cx1)
        if ox1 <= ox0:
            continue
        if py1 - 1 < 0 or cy0 >= rows:
            continue
        top_ifg = out_ifg[py1 - 1, ox0:ox1]
        bot_ifg = out_ifg[cy0, ox0:ox1]
        top_coh = out_coh[py1 - 1, ox0:ox1]
        bot_coh = out_coh[cy0, ox0:ox1]
        gap = gy1 - gy0
        for r in range(gap):
            t = float(r + 1) / float(gap + 1)
            rr = gy0 + r
            out_ifg[rr, ox0:ox1] = (1.0 - t) * top_ifg + t * bot_ifg
            out_coh[rr, ox0:ox1] = ((1.0 - t) * top_coh + t * bot_coh).astype(np.float32)
            valid_mask[rr, ox0:ox1] = True

    out_ifg[~valid_mask] = np.complex64(0.0 + 0.0j)
    out_coh[~valid_mask] = np.float32(0.0)
    return out_ifg, out_coh


def _collect_burst_seam_diagnostics(
    *,
    interferogram: np.ndarray,
    coherence: np.ndarray,
    segments: list[tuple[int, int, int, int]],
) -> dict[str, Any]:
    rows, cols = interferogram.shape[:2]
    seam_records: list[dict[str, Any]] = []
    if not segments:
        return {
            "rows": int(rows),
            "cols": int(cols),
            "segment_count": 0,
            "seam_count": 0,
            "seams": seam_records,
        }

    coh_arr = np.asarray(coherence, dtype=np.float32)
    for i in range(len(segments) - 1):
        py0, py1, px0, px1 = segments[i]
        cy0, cy1, cx0, cx1 = segments[i + 1]
        seam_row_top = int(py1 - 1)
        seam_row_bottom = int(cy0)
        overlap_x0 = int(max(px0, cx0))
        overlap_x1 = int(min(px1, cx1))
        gap_lines = int(max(0, cy0 - py1))
        valid = (
            seam_row_top >= 0
            and seam_row_bottom < rows
            and overlap_x1 > overlap_x0
            and seam_row_top < rows
        )
        record: dict[str, Any] = {
            "pair_index": int(i),
            "top_segment": [int(py0), int(py1), int(px0), int(px1)],
            "bottom_segment": [int(cy0), int(cy1), int(cx0), int(cx1)],
            "gap_lines": int(gap_lines),
            "seam_row_top": int(seam_row_top),
            "seam_row_bottom": int(seam_row_bottom),
            "overlap_x0": int(overlap_x0),
            "overlap_x1": int(overlap_x1),
            "valid": bool(valid),
        }
        if valid:
            top = coh_arr[seam_row_top, overlap_x0:overlap_x1]
            bottom = coh_arr[seam_row_bottom, overlap_x0:overlap_x1]
            top_mean = float(np.nanmean(top)) if top.size else float("nan")
            bottom_mean = float(np.nanmean(bottom)) if bottom.size else float("nan")
            record["top_coherence_mean"] = top_mean if np.isfinite(top_mean) else None
            record["bottom_coherence_mean"] = bottom_mean if np.isfinite(bottom_mean) else None
            if top.size and bottom.size:
                jump_mean = float(np.nanmean(np.abs(top - bottom)))
                record["coherence_jump_mean"] = jump_mean if np.isfinite(jump_mean) else None
        seam_records.append(record)

    return {
        "rows": int(rows),
        "cols": int(cols),
        "segment_count": int(len(segments)),
        "seam_count": int(len(seam_records)),
        "seams": seam_records,
    }


def _merge_tops_burst_interferograms(
    burst_interferograms: list[np.ndarray],
    burst_coherences: list[np.ndarray],
    bursts: list[TopsBurstInfo],
    overlap_pairs: list[tuple[int, int, int]] | None = None,
    output_dir: Path | None = None,
    *,
    method: str = "top",
    use_esd_offsets: bool = False,
    esd_azimuth_offsets: list[float] | None = None,
    az_reference_offsets: list[list[int]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if not bursts:
        raise ValueError("bursts must not be empty")
    if len(burst_interferograms) != len(bursts) or len(burst_coherences) != len(bursts):
        raise ValueError("burst data and burst metadata lengths do not match")

    method = method.lower().strip()
    if method not in {"top", "bot", "avg"}:
        raise ValueError(f"unsupported merge method: {method}")

    def _burst_attr(burst: TopsBurstInfo | dict, name: str, default: int | float = 0):
        if isinstance(burst, dict):
            return burst.get(name, default)
        return getattr(burst, name, default)

    def _shift_azimuth(arr: np.ndarray, az_shift: float) -> np.ndarray:
        if abs(float(az_shift)) < 1.0e-6:
            return np.asarray(arr)
        src = np.asarray(arr)
        rows = int(src.shape[0])
        if rows <= 0:
            return src.copy()
        y = np.arange(rows, dtype=np.float32)
        src_y = y - np.float32(az_shift)
        src0 = np.floor(src_y).astype(np.int32)
        frac = (src_y - src0).astype(np.float32)
        src1 = src0 + 1
        valid = (src0 >= 0) & (src1 < rows)
        out = np.zeros_like(src)
        if np.any(valid):
            idx0 = src0[valid]
            idx1 = src1[valid]
            w = frac[valid].astype(np.float32)
            w = w.reshape((-1,) + (1,) * (src.ndim - 1))
            out[valid] = (1.0 - w) * src[idx0] + w * src[idx1]
        return out

    def _offset_values(value) -> list[float]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, np.ndarray)):
            out: list[float] = []
            for item in value:
                out.extend(_offset_values(item))
            return out
        try:
            return [float(value)]
        except Exception:
            return []

    bursts_norm = [burst if isinstance(burst, TopsBurstInfo) else TopsBurstInfo(**burst) for burst in bursts]
    if use_esd_offsets and esd_azimuth_offsets is not None and len(esd_azimuth_offsets) < len(bursts_norm):
        raise ValueError("esd_azimuth_offsets length is shorter than bursts length")
    if az_reference_offsets is not None and len(az_reference_offsets) < len(bursts_norm):
        raise ValueError("az_reference_offsets length is shorter than bursts length")

    row0s: list[int] = []
    row1s: list[int] = []
    col0s: list[int] = []
    col1s: list[int] = []
    for burst in bursts_norm:
        burst_idx = int(_burst_attr(burst, "burst_index", len(row0s)))
        ref_offset = 0
        if az_reference_offsets is not None and 0 <= burst_idx < len(az_reference_offsets):
            ref_vals = [float(v) for v in az_reference_offsets[burst_idx] if np.isfinite(float(v))]
            if ref_vals:
                ref_offset = int(round(float(np.mean(ref_vals))))
        row0 = int(_burst_attr(burst, "line_offset", 0)) + int(_burst_attr(burst, "first_valid_line", 0)) + ref_offset
        row1 = row0 + int(_burst_attr(burst, "num_valid_lines", _burst_attr(burst, "number_of_lines", 0)))
        col0 = int(_burst_attr(burst, "first_valid_sample", 0))
        col1 = col0 + int(_burst_attr(burst, "num_valid_samples", _burst_attr(burst, "number_of_samples", 0)))
        row0s.append(row0)
        row1s.append(row1)
        col0s.append(col0)
        col1s.append(col1)

    base_row = min(row0s)
    base_col = min(col0s)
    out_rows = max(row1s) - base_row
    out_cols = max(col1s) - base_col
    if out_rows <= 0 or out_cols <= 0:
        raise RuntimeError("invalid merged burst dimensions")

    out_ifg = np.zeros((out_rows, out_cols), dtype=np.complex64)
    out_coh = np.zeros((out_rows, out_cols), dtype=np.float32)
    coverage = np.zeros((out_rows, out_cols), dtype=np.uint8)

    overlap_lookup: dict[tuple[int, int], int] = {}
    if overlap_pairs:
        for prev_idx, next_idx, overlap_lines in overlap_pairs:
            overlap_lookup[(int(prev_idx), int(next_idx))] = max(0, int(overlap_lines))

    burst_arrays_ifg = [np.asarray(arr, dtype=np.complex64) for arr in burst_interferograms]
    burst_arrays_coh = [np.asarray(arr, dtype=np.float32) for arr in burst_coherences]
    if use_esd_offsets and esd_azimuth_offsets is not None:
        burst_arrays_ifg = [
            _shift_azimuth(arr, float(esd_azimuth_offsets[i])) for i, arr in enumerate(burst_arrays_ifg)
        ]
        burst_arrays_coh = [
            np.asarray(_shift_azimuth(arr, float(esd_azimuth_offsets[i])), dtype=np.float32)
            for i, arr in enumerate(burst_arrays_coh)
        ]

    def _place(burst_idx: int, y0: int, y1: int, x0: int, x1: int) -> tuple[int, int, int, int]:
        dst_y0 = max(0, y0 - base_row)
        dst_y1 = min(y1 - base_row, out_ifg.shape[0])
        dst_x0 = max(0, x0 - base_col)
        dst_x1 = min(x1 - base_col, out_ifg.shape[1])
        src = burst_arrays_ifg[burst_idx]
        src_coh = burst_arrays_coh[burst_idx]
        # Clamp source to available data
        src_rows = min(dst_y1 - dst_y0, src.shape[0])
        src_cols = min(dst_x1 - dst_x0, src.shape[1])
        if src_rows > 0 and src_cols > 0:
            out_ifg[dst_y0:dst_y0 + src_rows, dst_x0:dst_x0 + src_cols] = src[:src_rows, :src_cols]
            out_coh[dst_y0:dst_y0 + src_rows, dst_x0:dst_x0 + src_cols] = src_coh[:src_rows, :src_cols]
            coverage[dst_y0:dst_y0 + src_rows, dst_x0:dst_x0 + src_cols] = np.maximum(
                coverage[dst_y0:dst_y0 + src_rows, dst_x0:dst_x0 + src_cols],
                np.uint8(1),
            )
        return dst_y0, dst_y0 + src_rows, dst_x0, dst_x0 + src_cols

    prev_bounds = _place(0, row0s[0], row1s[0], col0s[0], col1s[0])
    prev_idx = 0
    for burst_idx in range(1, len(bursts_norm)):
        y0, y1, x0, x1 = row0s[burst_idx], row1s[burst_idx], col0s[burst_idx], col1s[burst_idx]
        cur_ifg = burst_arrays_ifg[burst_idx]
        cur_coh = burst_arrays_coh[burst_idx]

        ov_lines = overlap_lookup.get((prev_idx, burst_idx))
        if ov_lines is None:
            ov_start = max(row0s[prev_idx], y0)
            ov_end = min(row1s[prev_idx], y1)
        else:
            ov_lines = max(0, min(int(ov_lines), row1s[prev_idx] - row0s[prev_idx], y1 - y0))
            ov_start = max(y0, row1s[prev_idx] - ov_lines)
            ov_end = min(y1, ov_start + ov_lines)
        x_start = max(col0s[prev_idx], x0)
        x_end = min(col1s[prev_idx], x1)
        if x_end <= x_start:
            _place(burst_idx, y0, y1, x0, x1)
            prev_idx = burst_idx
            prev_bounds = (y0 - base_row, y1 - base_row, x0 - base_col, x1 - base_col)
            continue

        # Unique head for the current burst.
        if y0 < ov_start:
            # Source slice bounds (clamped)
            head_src_y0 = 0
            head_src_y1 = max(0, min(ov_start - y0, cur_ifg.shape[0]))
            src_rows = head_src_y1 - head_src_y0
            # Destination slice bounds
            dst_y0 = max(0, y0 - base_row)
            dst_y1 = min(dst_y0 + src_rows, out_ifg.shape[0])
            src_rows = dst_y1 - dst_y0
            dst_x0 = max(0, x_start - base_col)
            dst_x1 = min(dst_x0 + (x_end - x_start), out_ifg.shape[1])
            copy_cols = dst_x1 - dst_x0
            if src_rows > 0 and copy_cols > 0:
                head = cur_ifg[head_src_y0:head_src_y0 + src_rows, :copy_cols]
                head_coh = cur_coh[head_src_y0:head_src_y0 + src_rows, :copy_cols]
                out_ifg[dst_y0:dst_y1, dst_x0:dst_x1] = head
                out_coh[dst_y0:dst_y1, dst_x0:dst_x1] = head_coh
                coverage[dst_y0:dst_y1, dst_x0:dst_x1] = 1

        if ov_end > ov_start:
            # Source slice bounds (clamped)
            top_y0 = max(0, min(ov_start - row0s[prev_idx], burst_arrays_ifg[prev_idx].shape[0]))
            top_y1 = max(0, min(ov_end - row0s[prev_idx], burst_arrays_ifg[prev_idx].shape[0]))
            bot_y0 = max(0, min(ov_start - y0, cur_ifg.shape[0]))
            bot_y1 = max(0, min(ov_end - y0, cur_ifg.shape[0]))
            xs_prev = max(0, min(x_start - col0s[prev_idx], burst_arrays_ifg[prev_idx].shape[1]))
            xe_prev = max(0, min(xs_prev + (x_end - x_start), burst_arrays_ifg[prev_idx].shape[1]))
            xs_cur = max(0, min(x_start - x0, cur_ifg.shape[1]))
            xe_cur = max(0, min(xs_cur + (x_end - x_start), cur_ifg.shape[1]))
            top_rows = top_y1 - top_y0
            bot_rows = bot_y1 - bot_y0
            cols = min(xe_prev - xs_prev, xe_cur - xs_cur, x_end - x_start)
            if top_rows > 0 and bot_rows > 0 and cols > 0:
                # Extract properly sized source blocks
                top_block = burst_arrays_ifg[prev_idx][top_y0:top_y0 + top_rows, xs_prev:xs_prev + cols]
                bot_block = cur_ifg[bot_y0:bot_y0 + bot_rows, xs_cur:xs_cur + cols]
                top_coh_block = burst_arrays_coh[prev_idx][top_y0:top_y0 + top_rows, xs_prev:xs_prev + cols]
                bot_coh_block = cur_coh[bot_y0:bot_y0 + bot_rows, xs_cur:xs_cur + cols]
                dst_y0 = max(0, ov_start - base_row)
                dst_x0 = max(0, x_start - base_col)
                # Determine actual copy size (smallest of source and available dest)
                copy_rows = min(top_block.shape[0], bot_block.shape[0], out_ifg.shape[0] - dst_y0)
                copy_cols = min(top_block.shape[1], bot_block.shape[1], out_ifg.shape[1] - dst_x0)
                if copy_rows > 0 and copy_cols > 0:
                    if method == "top":
                        out_ifg[dst_y0:dst_y0 + copy_rows, dst_x0:dst_x0 + copy_cols] = top_block[:copy_rows, :copy_cols]
                        out_coh[dst_y0:dst_y0 + copy_rows, dst_x0:dst_x0 + copy_cols] = top_coh_block[:copy_rows, :copy_cols]
                    elif method == "bot":
                        out_ifg[dst_y0:dst_y0 + copy_rows, dst_x0:dst_x0 + copy_cols] = bot_block[:copy_rows, :copy_cols]
                        out_coh[dst_y0:dst_y0 + copy_rows, dst_x0:dst_x0 + copy_cols] = bot_coh_block[:copy_rows, :copy_cols]
                    else:
                        out_ifg[dst_y0:dst_y0 + copy_rows, dst_x0:dst_x0 + copy_cols] = 0.5 * (top_block[:copy_rows, :copy_cols] + bot_block[:copy_rows, :copy_cols])
                        out_coh[dst_y0:dst_y0 + copy_rows, dst_x0:dst_x0 + copy_cols] = 0.5 * (top_coh_block[:copy_rows, :copy_cols] + bot_coh_block[:copy_rows, :copy_cols])
                    coverage[dst_y0:dst_y0 + copy_rows, dst_x0:dst_x0 + copy_cols] = 1

        # Unique tail for the current burst.
        if y1 > ov_end:
            # Source slice bounds (clamped to array)
            tail_src_y0 = max(0, min(ov_end - y0, cur_ifg.shape[0]))
            tail_src_y1 = max(0, min(y1 - y0, cur_ifg.shape[0]))
            src_rows = tail_src_y1 - tail_src_y0
            # Destination slice bounds
            dst_y0 = max(0, ov_end - base_row)
            dst_y1 = min(dst_y0 + src_rows, out_ifg.shape[0])
            src_rows = dst_y1 - dst_y0
            dst_x0 = max(0, x_start - base_col)
            dst_x1 = min(dst_x0 + (x_end - x_start), out_ifg.shape[1])
            copy_cols = dst_x1 - dst_x0
            if src_rows > 0 and copy_cols > 0:
                tail = cur_ifg[tail_src_y0:tail_src_y0 + src_rows, :copy_cols]
                tail_coh = cur_coh[tail_src_y0:tail_src_y0 + src_rows, :copy_cols]
                out_ifg[dst_y0:dst_y1, dst_x0:dst_x1] = tail
                out_coh[dst_y0:dst_y1, dst_x0:dst_x1] = tail_coh
                coverage[dst_y0:dst_y1, dst_x0:dst_x1] = 1

        prev_idx = burst_idx
        prev_bounds = (y0 - base_row, y1 - base_row, x0 - base_col, x1 - base_col)

    out_ifg[coverage == 0] = np.complex64(0.0 + 0.0j)
    out_coh[coverage == 0] = np.float32(0.0)
    return out_ifg, out_coh


def _tops_esd_stage_dir(output_dir: Path) -> Path:
    return stage_dir(output_dir, "p3") / "tops_esd"


def _estimate_esd_local_frequency(ifg: np.ndarray) -> np.ndarray:
    phase = np.unwrap(np.angle(np.asarray(ifg, dtype=np.complex64)), axis=0)
    if phase.shape[0] <= 1:
        return np.zeros_like(phase, dtype=np.float32)
    freq = np.gradient(phase, axis=0).astype(np.float32)
    try:
        freq = _boxcar_mean2d(freq, 5).astype(np.float32)
    except Exception:
        pass
    freq[np.abs(freq) < 1.0e-6] = np.nan
    return freq


def _compute_esd_spectral_diversity(
    overlap_ifgs: list[np.ndarray],
    overlap_cohs: list[np.ndarray],
    *,
    azimuth_looks: int = 5,
    range_looks: int = 15,
    coherence_threshold: float = 0.85,
    extra_esd_cycles: float = 0.0,
) -> tuple[float, float, float, np.ndarray]:
    if len(overlap_ifgs) != len(overlap_cohs):
        raise ValueError("overlap_ifgs and overlap_cohs must have the same length")

    extra_offset = float(extra_esd_cycles) * 2.0 * np.pi
    all_offsets: list[np.ndarray] = []

    for ifg, cor in zip(overlap_ifgs, overlap_cohs, strict=False):
        ifg_arr = np.asarray(ifg, dtype=np.complex64)
        cor_arr = np.asarray(cor, dtype=np.float32)
        rows = min(ifg_arr.shape[0], cor_arr.shape[0])
        cols = min(ifg_arr.shape[1], cor_arr.shape[1])
        if rows <= 0 or cols <= 0:
            continue
        ifg_arr = ifg_arr[:rows, :cols]
        cor_arr = cor_arr[:rows, :cols]

        if azimuth_looks > 1 or range_looks > 1:
            ifg_arr = _multilook_mean(ifg_arr, azimuth_looks, range_looks).astype(np.complex64)
            cor_arr = _multilook_mean(cor_arr, azimuth_looks, range_looks).astype(np.float32)

        rows, cols = ifg_arr.shape[:2]
        if rows <= 0 or cols <= 0:
            continue

        freq = _estimate_esd_local_frequency(ifg_arr)
        phase = np.angle(ifg_arr).astype(np.float32)
        off = np.full((rows, cols), np.nan, dtype=np.float32)
        valid_freq = np.isfinite(freq) & (np.abs(freq) > 1.0e-6)
        off[valid_freq] = (phase[valid_freq] + extra_offset) / freq[valid_freq]

        mask = (np.abs(ifg_arr) > 0) & (cor_arr > float(coherence_threshold)) & np.isfinite(off)
        if np.any(mask):
            all_offsets.append(off[mask].astype(np.float32))

    if not all_offsets:
        raise RuntimeError("Coherence threshold too strict. No points left for reliable ESD estimate")

    offsets = np.concatenate(all_offsets).astype(np.float64)
    return (
        float(np.median(offsets)),
        float(np.mean(offsets)),
        float(np.std(offsets)),
        offsets.astype(np.float32),
    )


def _load_tops_burst_patches_from_p2(
    context: PairContext,
    bursts: list[TopsBurstInfo],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    ifg = _load_cached_array(context.pair_dir, "p2", "interferogram")
    coh = _load_cached_array(context.pair_dir, "p2", "coherence")
    burst_ifg: list[np.ndarray] = []
    burst_coh: list[np.ndarray] = []
    for burst in bursts:
        y0 = int(burst.line_offset + burst.first_valid_line)
        y1 = y0 + int(burst.num_valid_lines)
        x0 = int(burst.first_valid_sample)
        x1 = x0 + int(burst.num_valid_samples)
        burst_ifg.append(np.asarray(ifg[y0:y1, x0:x1], dtype=np.complex64).copy())
        burst_coh.append(np.asarray(coh[y0:y1, x0:x1], dtype=np.float32).copy())
    return burst_ifg, burst_coh



def run_burst_merge_stage(
    context: PairContext,
    *,
    master_bursts: list[TopsBurstInfo],
    slave_bursts: list[TopsBurstInfo],
    overlap_pairs: list[dict] | None = None,
    use_topo_flattening: bool = False,
    do_burst_seam_repair: bool = True,
    esd_azimuth_offsets: list[float] | None = None,
    az_reference_offsets: list[list[int]] | None = None,
) -> tuple[dict, str, str | None]:
    processing_options = {
        "use_topo_flattening": bool(use_topo_flattening),
        "do_burst_seam_repair": bool(do_burst_seam_repair),
        "overlap_pairs": overlap_pairs if overlap_pairs is not None else None,
        "esd_azimuth_offsets": [float(v) for v in esd_azimuth_offsets] if esd_azimuth_offsets is not None else None,
        "az_reference_offsets": az_reference_offsets if az_reference_offsets is not None else None,
    }
    cached_outputs = _load_cached_stage_outputs(
        context.pair_dir,
        "p3",
        required_keys=("merged_interferogram", "merged_coherence"),
        expected_processing_options=processing_options,
    )
    if cached_outputs is not None:
        return cached_outputs, "cache", None

    existing_p3 = load_stage_record(context.pair_dir, "p3") or {}
    existing_p3_outputs = existing_p3.get("output_files") if isinstance(existing_p3.get("output_files"), dict) else {}

    p2_record = load_stage_record(context.pair_dir, "p2") or {}
    p2_outputs = p2_record.get("output_files") or {}

    def _burst_to_dict(burst: TopsBurstInfo) -> dict:
        return {
            "lineOffset": int(burst.line_offset),
            "numberOfLines": int(burst.number_of_lines),
            "numberOfSamples": int(burst.number_of_samples),
            "firstValidLine": int(burst.first_valid_line),
            "numValidLines": int(burst.num_valid_lines),
            "firstValidSample": int(burst.first_valid_sample),
            "numValidSamples": int(burst.num_valid_samples),
        }

    def _load_burst_arrays(kind: str) -> list[np.ndarray]:
        values = p2_outputs.get(kind)
        if isinstance(values, list) and values:
            out: list[np.ndarray] = []
            for value in values:
                try:
                    out.append(np.load(str(value)))
                except Exception:
                    out = []
                    break
            if out:
                return out
        p2_dir = stage_dir(context.pair_dir, "p2")
        patterns = (
            "*burst*interferogram*.npy" if kind == "burst_interferograms" else "*burst*coherence*.npy",
            "*burst*ifg*.npy" if kind == "burst_interferograms" else "*burst*coh*.npy",
        )
        paths: list[Path] = []
        for pattern in patterns:
            paths = sorted(p2_dir.glob(pattern))
            if paths:
                break
        if paths:
            return [np.load(str(path)) for path in paths]
        single_key = "interferogram" if kind == "burst_interferograms" else "coherence"
        return [np.load(str(p2_outputs.get(single_key) or _load_stage_output_path(context.pair_dir, "p2", single_key)))]

    if master_bursts:
        bursts = list(master_bursts)
    elif slave_bursts:
        bursts = list(slave_bursts)
    else:
        bursts = []

    if not bursts:
        interferograms = _load_burst_arrays("burst_interferograms")
        coherences = _load_burst_arrays("burst_coherences")
        if len(interferograms) != 1 or len(coherences) != 1:
            raise RuntimeError("TOPS burst merge requires burst metadata or per-burst inputs")
        merged_interferogram = np.asarray(interferograms[0], dtype=np.complex64)
        merged_coherence = np.asarray(coherences[0], dtype=np.float32)
        bursts = [TopsBurstInfo(0, 0, int(merged_interferogram.shape[0]), int(merged_interferogram.shape[1]), 0, int(merged_interferogram.shape[0]), 0, int(merged_interferogram.shape[1]))]
    else:
        interferograms = _load_burst_arrays("burst_interferograms")
        coherences = _load_burst_arrays("burst_coherences")
        if len(interferograms) == 1 and len(coherences) == 1 and len(bursts) > 1:
            full_ifg = np.asarray(interferograms[0], dtype=np.complex64)
            full_coh = np.asarray(coherences[0], dtype=np.float32)
            interferograms = []
            coherences = []
            for burst in bursts:
                y0 = int(burst.line_offset + burst.first_valid_line)
                y1 = y0 + int(burst.num_valid_lines)
                x0 = int(burst.first_valid_sample)
                x1 = x0 + int(burst.num_valid_samples)
                interferograms.append(full_ifg[y0:y1, x0:x1].copy())
                coherences.append(full_coh[y0:y1, x0:x1].copy())
        if len(interferograms) != len(bursts) or len(coherences) != len(bursts):
            raise RuntimeError("burst input count does not match burst metadata")
        overlap_tuples: list[tuple[int, int, int]] | None = None
        if overlap_pairs:
            overlap_tuples = []
            for item in overlap_pairs:
                if isinstance(item, dict):
                    prev_idx = int(item.get("previous_burst_index", item.get("prev_idx", 0)))
                    next_idx = int(item.get("next_burst_index", item.get("next_idx", prev_idx + 1)))
                    overlap_lines = int(item.get("estimated_overlap_lines", item.get("overlap_lines", 0)))
                else:
                    prev_idx, next_idx, overlap_lines = item
                overlap_tuples.append((int(prev_idx), int(next_idx), int(overlap_lines)))
        else:
            overlap_tuples = None

        merged_interferogram, merged_coherence = _merge_tops_burst_interferograms(
            interferograms,
            coherences,
            bursts,
            overlap_pairs=overlap_tuples,
            output_dir=context.pair_dir,
            method="avg" if use_topo_flattening else "top",
            use_esd_offsets=esd_azimuth_offsets is not None,
            esd_azimuth_offsets=esd_azimuth_offsets,
            az_reference_offsets=az_reference_offsets,
        )


def run_esd_estimation_stage(
    context: PairContext,
    *,
    master_bursts: list[TopsBurstInfo],
    slave_bursts: list[TopsBurstInfo],
    overlap_pairs: list[dict] | None = None,
    esd_azimuth_looks: int = 5,
    esd_range_looks: int = 15,
    esd_coherence_threshold: float = 0.85,
    extra_esd_cycles: float = 0.0,
) -> tuple[dict, str, str | None]:
    processing_options = {
        "esd_azimuth_looks": int(esd_azimuth_looks),
        "esd_range_looks": int(esd_range_looks),
        "esd_coherence_threshold": float(esd_coherence_threshold),
        "extra_esd_cycles": float(extra_esd_cycles),
        "overlap_pairs": overlap_pairs if overlap_pairs is not None else None,
    }
    esd_dir = _tops_esd_stage_dir(context.pair_dir)
    esd_dir.mkdir(parents=True, exist_ok=True)
    summary_path = esd_dir / "esd_summary.json"
    median_path = esd_dir / "median_offset.npy"
    mean_path = esd_dir / "mean_offset.npy"
    std_path = esd_dir / "std_offset.npy"
    offsets_path = esd_dir / "offsets.npy"
    if summary_path.is_file() and median_path.is_file() and mean_path.is_file() and std_path.is_file() and offsets_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if dict(summary.get("processing_options") or {}) == dict(processing_options):
                return {
                    "median_offset_px": str(median_path),
                    "mean_offset_px": str(mean_path),
                    "std_offset_px": str(std_path),
                    "offsets": str(offsets_path),
                    "summary": str(summary_path),
                }, "cache", None
        except Exception:
            pass

    p2_record = load_stage_record(context.pair_dir, "p2") or {}
    p2_outputs = p2_record.get("output_files") or {}
    overlap_ifg_paths = p2_outputs.get("overlap_ifgs") or []
    overlap_coh_paths = p2_outputs.get("overlap_cohs") or []
    overlap_ifgs: list[np.ndarray] = []
    overlap_cohs: list[np.ndarray] = []
    for ifg_path, coh_path in zip(overlap_ifg_paths, overlap_coh_paths, strict=False):
        if not ifg_path or not coh_path:
            continue
        ifg_p = Path(str(ifg_path))
        coh_p = Path(str(coh_path))
        if not ifg_p.is_file() or not coh_p.is_file():
            continue
        overlap_ifgs.append(np.asarray(np.load(ifg_p), dtype=np.complex64))
        overlap_cohs.append(np.asarray(np.load(coh_p), dtype=np.float32))

    if not overlap_ifgs or not overlap_cohs:
        raise RuntimeError("ESD estimation requires overlap burst interferograms from p2")

    median_offset, mean_offset, std_offset, all_offsets = _compute_esd_spectral_diversity(
        overlap_ifgs,
        overlap_cohs,
        azimuth_looks=esd_azimuth_looks,
        range_looks=esd_range_looks,
        coherence_threshold=esd_coherence_threshold,
        extra_esd_cycles=extra_esd_cycles,
    )

    np.save(median_path, np.asarray(median_offset, dtype=np.float32))
    np.save(mean_path, np.asarray(mean_offset, dtype=np.float32))
    np.save(std_path, np.asarray(std_offset, dtype=np.float32))
    np.save(offsets_path, all_offsets.astype(np.float32))
    summary = {
        "median_offset_px": float(median_offset),
        "mean_offset_px": float(mean_offset),
        "std_offset_px": float(std_offset),
        "n_valid_points": int(all_offsets.size),
        "processing_options": processing_options,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    output_files = {
        "median_offset_px": str(median_path),
        "mean_offset_px": str(mean_path),
        "std_offset_px": str(std_path),
        "offsets": str(offsets_path),
        "summary": str(summary_path),
    }
    _write_custom_stage_record(
        output_dir=context.pair_dir,
        stage="p3",
        master_manifest_path=context.master_manifest_path,
        slave_manifest_path=context.slave_manifest_path,
        backend_used="cpu",
        output_files=output_files,
        fallback_reason=None,
        upstream_stage_dependencies=["p2"],
        processing_options=processing_options,
    )
    return output_files, "cpu", None



