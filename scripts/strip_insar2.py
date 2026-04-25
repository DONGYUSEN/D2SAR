from __future__ import annotations

import argparse
import colorsys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from xml.sax.saxutils import escape

import h5py
import numpy as np
from osgeo import gdal, osr
from PIL import Image


ICU_DEFAULTS = {
    "seed": 0,
    "buffer_lines": 3700,
    "overlap_lines": 200,
    "use_phase_gradient_neutron": False,
    "use_intensity_neutron": False,
    "phase_gradient_window_size": 5,
    "neutron_phase_gradient_threshold": 3.0,
    "neutron_intensity_threshold": 8.0,
    "max_intensity_correlation_threshold": 0.8,
    "trees_number": 7,
    "max_branch_length": 64,
    "pixel_spacing_ratio": 1.0,
    "initial_correlation_threshold": 0.1,
    "max_correlation_threshold": 0.9,
    "correlation_threshold_increments": 0.1,
    "min_tile_area": 0.003125,
    "bootstrap_lines": 16,
    "min_overlap_area": 16,
    "phase_variance_threshold": 8.0,
}

SNAPHU_DEFAULTS = {
    "nlooks": 1.0,
    "cost_mode": "smooth",
    "initialization_method": "mcf",
    "min_conncomp_frac": 0.01,
    "phase_grad_window": (7, 7),
    "ntiles": (1, 1),
    "tile_overlap": (0, 0),
    "nproc": 1,
    "tile_cost_thresh": 500,
    "min_region_size": 300,
    "single_tile_reoptimize": True,
    "regrow_conncomps": True,
}


@dataclass
class PairContext:
    master_manifest_path: Path
    slave_manifest_path: Path
    master_manifest: dict
    slave_manifest: dict
    master_orbit_data: dict
    slave_orbit_data: dict
    master_acq_data: dict
    slave_acq_data: dict
    master_rg_data: dict
    slave_rg_data: dict
    master_dop_data: dict
    slave_dop_data: dict
    output_root: Path
    pair_name: str
    pair_dir: Path
    output_paths: dict[str, str]
    resolved_dem: str
    orbit_interp: str
    wavelength: float


STAGE_SEQUENCE = ("check", "prep", "crop", "p0", "p1", "p2", "p3", "p4", "p5", "p6")
STAGE_DIR_NAMES = {
    "check": "check",
    "prep": "prep",
    "crop": "crop",
    "p0": "p0_geo2rdr",
    "p1": "p1_dense_match",
    "p2": "p2_crossmul",
    "p3": "p3_unwrap",
    "p4": "p4_geocode",
    "p5": "p5_hdf",
    "p6": "p6_publish",
}
GEO2RDR_OFFSET_NODATA = -999999.0
GEO2RDR_OFFSET_INVALID_LOW = -1.0e5
NISAR_OFFSET_INVALID_VALUE = -1.0e6
_NISAR_REGISTRATION_MODULES: dict[str, object] | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_nisar_python_packages_on_path() -> Path | None:
    try:
        importlib.import_module("isce3")
        importlib.import_module("nisar")
        return None
    except Exception:
        pass

    packages_dir = _repo_root() / "isce3" / "python" / "packages"
    if packages_dir.is_dir() and str(packages_dir) not in sys.path:
        sys.path.insert(0, str(packages_dir))
    return packages_dir


def _load_nisar_registration_modules() -> dict[str, object]:
    global _NISAR_REGISTRATION_MODULES
    if _NISAR_REGISTRATION_MODULES is not None:
        return _NISAR_REGISTRATION_MODULES

    packages_dir = _ensure_nisar_python_packages_on_path()
    if packages_dir is not None and not packages_dir.is_dir():
        raise FileNotFoundError(f"NISAR python packages directory not found: {packages_dir}")

    from nisar.workflows.dense_offsets import create_empty_dataset, set_optional_attributes
    from nisar.workflows.helpers import sum_gdal_rasters
    from nisar.workflows.rubbersheet import (
        _filter_offsets,
        _interpolate_offsets,
        fill_outliers_holes,
        identify_outliers,
    )

    _NISAR_REGISTRATION_MODULES = {
        "create_empty_dataset": create_empty_dataset,
        "set_optional_attributes": set_optional_attributes,
        "sum_gdal_rasters": sum_gdal_rasters,
        "identify_outliers": identify_outliers,
        "fill_outliers_holes": fill_outliers_holes,
        "_interpolate_offsets": _interpolate_offsets,
        "_filter_offsets": _filter_offsets,
    }
    return _NISAR_REGISTRATION_MODULES


def _default_dense_offsets_cfg() -> dict:
    return {
        "window_range": 96,
        "window_azimuth": 96,
        "half_search_range": 24,
        "half_search_azimuth": 24,
        "skip_range": 48,
        "skip_azimuth": 48,
        "margin": 48,
        "gross_offset_range": None,
        "gross_offset_azimuth": None,
        "start_pixel_range": None,
        "start_pixel_azimuth": None,
        "offset_width": None,
        "offset_length": None,
        "cross_correlation_domain": "frequency",
        "slc_oversampling_factor": 2,
        "deramping_method": "magnitude",
        "deramping_axis": "both",
        "correlation_statistics_zoom": 21,
        "correlation_surface_zoom": 16,
        "correlation_surface_oversampling_factor": 32,
        "correlation_surface_oversampling_method": "fft",
        "windows_batch_range": 16,
        "windows_batch_azimuth": 16,
        "cuda_streams": 2,
        "use_gross_offsets": False,
        "gross_offset_filepath": None,
        "merge_gross_offset": False,
    }


def _default_rubbersheet_cfg() -> dict:
    return {
        "threshold": 2.5,
        "median_filter_size_range": 5,
        "median_filter_size_azimuth": 5,
        "culling_metric": "median_filter",
        "mask_refine_enabled": False,
        "mask_refine_filter_size": 5,
        "mask_refine_min_neighbors": 6,
        "outlier_filling_method": "fill_smoothed",
        "fill_smoothed": {
            "kernel_size": 7,
            "iterations": 4,
        },
        "interpolation_method": "linear",
        "offsets_filter": "median",
        "boxcar": {
            "filter_size_range": 5,
            "filter_size_azimuth": 5,
        },
        "median": {
            "filter_size_range": 5,
            "filter_size_azimuth": 5,
        },
        "gaussian": {
            "sigma_range": 1.0,
            "sigma_azimuth": 1.0,
        },
    }


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_stage_name(stage: str) -> str:
    if stage not in STAGE_SEQUENCE:
        raise ValueError(f"Unsupported stage '{stage}'")
    return stage


def work_dir(output_dir: str | Path) -> Path:
    return Path(output_dir) / "work"


def stage_dir(output_dir: str | Path, stage: str) -> Path:
    return work_dir(output_dir) / STAGE_DIR_NAMES[validate_stage_name(stage)]


def stage_json_path(output_dir: str | Path, stage: str) -> Path:
    return stage_dir(output_dir, stage) / "stage.json"


def success_marker_path(output_dir: str | Path, stage: str) -> Path:
    return stage_dir(output_dir, stage) / "SUCCESS"


def load_stage_record(output_dir: str | Path, stage: str) -> dict | None:
    path = stage_json_path(output_dir, stage)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_stage_record(output_dir: str | Path, stage: str, record: dict) -> Path:
    path = stage_json_path(output_dir, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def mark_stage_success(output_dir: str | Path, stage: str) -> Path:
    path = success_marker_path(output_dir, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("success\n", encoding="utf-8")
    return path


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


def _numpy_dtype_to_gdal(np_dtype: np.dtype | type) -> int:
    dtype = np.dtype(np_dtype)
    if dtype == np.dtype(np.uint8):
        return gdal.GDT_Byte
    if dtype == np.dtype(np.float64):
        return gdal.GDT_Float64
    if dtype == np.dtype(np.complex64):
        return gdal.GDT_CFloat32
    if dtype == np.dtype(np.complex128):
        return gdal.GDT_CFloat64
    return gdal.GDT_Float32


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
    buf_type = _numpy_dtype_to_gdal(np_dtype)
    if buf_type == gdal.GDT_Float32 and np_dtype != np.dtype(np.float32):
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


def _write_band_array(band, data: np.ndarray, xoff: int = 0, yoff: int = 0) -> None:
    arr = np.ascontiguousarray(data)
    buf_type = _numpy_dtype_to_gdal(arr.dtype)
    if buf_type == gdal.GDT_Float32 and arr.dtype != np.dtype(np.float32):
        arr = np.ascontiguousarray(arr, dtype=np.float32)
    rows, cols = arr.shape
    band.WriteRaster(
        int(xoff),
        int(yoff),
        int(cols),
        int(rows),
        arr.tobytes(),
        buf_xsize=int(cols),
        buf_ysize=int(rows),
        buf_type=buf_type,
    )


def resolve_manifest_data_path(manifest_path: str | Path, entry) -> str | None:
    if entry is None:
        return None
    manifest_dir = Path(manifest_path).parent.resolve()
    if isinstance(entry, dict):
        path_value = entry.get("path")
        if path_value is None:
            return None
        base_path = Path(path_value)
        resolved = base_path if base_path.is_absolute() else (manifest_dir / base_path)
        storage = str(entry.get("storage") or "").strip().lower()
        member = entry.get("member")
        if storage == "zip" and member:
            archive = resolved.as_posix()
            member_path = str(member).lstrip("/").replace("\\", "/")
            return f"/vsizip/{archive}/{member_path}"
        return str(resolved)
    entry_path = Path(str(entry))
    resolved = entry_path if entry_path.is_absolute() else (manifest_dir / entry_path)
    return str(resolved)


def resolve_manifest_metadata_path(
    manifest_path: str | Path,
    manifest: dict,
    key: str,
) -> Path:
    manifest_path = Path(manifest_path)
    metadata = manifest.get("metadata", {})
    entry = metadata.get(key)
    fallback_path = manifest_path.parent / "metadata" / f"{key}.json"
    if entry is not None:
        resolved = resolve_manifest_data_path(manifest_path, entry)
        if resolved is None:
            raise FileNotFoundError(f"metadata entry '{key}' is null in manifest")
        resolved_path = Path(resolved)
        if resolved_path.exists() or not fallback_path.exists():
            return resolved_path
    return fallback_path


def gps_to_datetime(ts: str) -> datetime:
    ts = ts.strip()
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    elif not (ts[-6:-5] in ("+", "-") and ts[-6:].count(":") == 3):
        ts = ts + "+00:00"
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def choose_orbit_interp(orbit_json: dict, acquisition_json: dict | None = None) -> str:
    try:
        state_vectors = orbit_json.get("stateVectors", [])
        if len(state_vectors) < 8:
            return "Legendre"
        times = []
        pos = []
        vel = []
        for sv in state_vectors:
            if "gpsTime" in sv and sv["gpsTime"] is not None:
                times.append(float(sv["gpsTime"]))
            else:
                gps_epoch = datetime(1980, 1, 6, tzinfo=timezone.utc)
                times.append((gps_to_datetime(sv["timeUTC"]) - gps_epoch).total_seconds())
            pos.append([sv["posX"], sv["posY"], sv["posZ"]])
            vel.append([sv["velX"], sv["velY"], sv["velZ"]])
        t = np.asarray(times, dtype=np.float64)
        pos = np.asarray(pos, dtype=np.float64)
        vel = np.asarray(vel, dtype=np.float64)
        if not (np.all(np.isfinite(t)) and np.all(np.isfinite(pos)) and np.all(np.isfinite(vel))):
            return "Legendre"
        dt = np.diff(t)
        if len(dt) == 0 or np.any(dt <= 0):
            return "Legendre"
        median_dt = float(np.median(dt))
        if median_dt <= 0 or float(np.max(np.abs(dt - median_dt))) > 1e-3:
            return "Legendre"
        if acquisition_json is not None:
            start_gps = acquisition_json.get("startGPSTime")
            stop_gps = acquisition_json.get("stopGPSTime")
            if start_gps is None or stop_gps is None:
                return "Legendre"
            margin = 2.0 * median_dt
            if float(start_gps) < t[0] + margin or float(stop_gps) > t[-1] - margin:
                return "Legendre"
        fd_vel = (pos[2:] - pos[:-2]) / (t[2:, None] - t[:-2, None])
        vel_mid = vel[1:-1]
        denom = np.linalg.norm(fd_vel, axis=1)
        valid = denom > 0
        if not np.any(valid):
            return "Legendre"
        rel_err = np.linalg.norm(vel_mid[valid] - fd_vel[valid], axis=1) / denom[valid]
        if float(np.median(rel_err)) <= 1e-5 and float(np.percentile(rel_err, 95)) <= 1e-4:
            return "Hermite"
        return "Legendre"
    except Exception:
        return "Legendre"


def _build_coregistration_doppler_lut():
    import isce3.core

    # Follow NISAR strip RSLC registration assumptions: use zero Doppler
    # through geo2rdr and resample stages.
    return isce3.core.LUT2d()


def load_scene_corners_with_fallback(
    manifest_path: str | Path,
    manifest: dict,
    ancillary_override=None,
):
    manifest_path = Path(manifest_path)
    scene_json = resolve_manifest_metadata_path(manifest_path, manifest, "scene")
    if scene_json.exists():
        with open(scene_json, encoding="utf-8") as f:
            scene = json.load(f)
        corners = scene.get("sceneCorners") or []
        if corners:
            return corners
    return []


def construct_orbit(orbit_json: dict, interp_method: str = "Hermite"):
    import isce3.core

    raw_datetimes = [gps_to_datetime(sv["timeUTC"]) for sv in orbit_json["stateVectors"]]
    if len(raw_datetimes) >= 3:
        raw_seconds = np.array([dt.timestamp() for dt in raw_datetimes], dtype=np.float64)
        dt_seconds = np.diff(raw_seconds)
        uniform_spacing = float(np.median(dt_seconds))
        if np.max(np.abs(dt_seconds - uniform_spacing)) < 1e-3:
            spacing_us = int(round(uniform_spacing * 1_000_000))
            raw_datetimes = [
                raw_datetimes[0] + timedelta(microseconds=i * spacing_us)
                for i in range(len(raw_datetimes))
            ]
    state_vectors = []
    for i, sv in enumerate(orbit_json["stateVectors"]):
        dt = isce3.core.DateTime(raw_datetimes[i])
        pos = np.array([sv["posX"], sv["posY"], sv["posZ"]], dtype=np.float64)
        vel = np.array([sv["velX"], sv["velY"], sv["velZ"]], dtype=np.float64)
        state_vectors.append(isce3.core.StateVector(dt, pos, vel))
    ref_dt = isce3.core.DateTime(gps_to_datetime(orbit_json["header"]["firstStateTimeUTC"]))
    method_map = {
        "Hermite": isce3.core.OrbitInterpMethod.HERMITE,
        "Legendre": isce3.core.OrbitInterpMethod.LEGENDRE,
    }
    return isce3.core.Orbit(state_vectors, ref_dt, method_map[interp_method])


def construct_doppler_lut2d(
    doppler_json: dict,
    *,
    radargrid_json: dict,
    acquisition_json: dict,
    orbit_json: dict,
):
    import isce3.core

    combined = doppler_json["combinedDoppler"]
    coeffs = combined["coefficients"]
    ref_range_time = float(combined["referencePoint"])
    starting_range = isce3.core.speed_of_light * float(radargrid_json["rangeTimeFirstPixel"]) / 2.0
    range_pixel_spacing = float(radargrid_json["columnSpacing"])
    width = int(radargrid_json["numberOfColumns"])
    x_coord = starting_range + range_pixel_spacing * np.arange(width + 1, dtype=np.float64)
    orbit_ref_dt = gps_to_datetime(orbit_json["header"]["firstStateTimeUTC"])
    gps_epoch = datetime(1980, 1, 6, tzinfo=timezone.utc)
    orbit_ref_gps = (orbit_ref_dt - gps_epoch).total_seconds()
    sensing_start = float(acquisition_json["startGPSTime"]) - orbit_ref_gps
    length = int(radargrid_json["numberOfRows"])
    prf = float(acquisition_json["prf"])
    sensing_end = sensing_start + length / prf
    y_coord = np.array([sensing_start, sensing_end], dtype=np.float64)
    data = np.zeros((len(y_coord), len(x_coord)), dtype=np.float64)
    range_times = 2.0 * x_coord / isce3.core.speed_of_light
    row = np.zeros(len(x_coord), dtype=np.float64)
    for j, c in enumerate(coeffs):
        row += float(c) * np.power(range_times - ref_range_time, j)
    data[:] = row[None, :]
    return isce3.core.LUT2d(
        xstart=float(x_coord[0]),
        ystart=float(y_coord[0]),
        dx=(float(x_coord[-1]) - float(x_coord[0])) / max(len(x_coord) - 1, 1),
        dy=(float(y_coord[-1]) - float(y_coord[0])) / max(len(y_coord) - 1, 1),
        data=data,
        method="bilinear",
        b_error=True,
    )


def construct_radar_grid(
    radargrid_json: dict,
    acquisition_json: dict,
    orbit_json: dict,
):
    import isce3.core
    import isce3.product

    sensing_start_abs_gps = acquisition_json["startGPSTime"]
    orbit_ref_dt = gps_to_datetime(orbit_json["header"]["firstStateTimeUTC"])
    gps_epoch = datetime(1980, 1, 6, tzinfo=timezone.utc)
    orbit_ref_gps = (orbit_ref_dt - gps_epoch).total_seconds()
    sensing_start_rel = sensing_start_abs_gps - orbit_ref_gps
    wavelength = isce3.core.speed_of_light / acquisition_json["centerFrequency"]
    prf = acquisition_json["prf"]
    r0 = isce3.core.speed_of_light * radargrid_json["rangeTimeFirstPixel"] / 2.0
    range_pixel_spacing = radargrid_json["columnSpacing"]
    look_raw = acquisition_json.get("lookDirection", "RIGHT").strip().upper()
    look_side = isce3.core.LookSide.Left if look_raw == "LEFT" else isce3.core.LookSide.Right
    return isce3.product.RadarGridParameters(
        sensing_start=sensing_start_rel,
        wavelength=wavelength,
        prf=prf,
        starting_range=r0,
        range_pixel_spacing=range_pixel_spacing,
        lookside=look_side,
        length=radargrid_json["numberOfRows"],
        width=radargrid_json["numberOfColumns"],
        ref_epoch=isce3.core.DateTime(orbit_ref_dt),
    )


def _copy_raster(src: str, dst: Path) -> str:
    src_ds = gdal.Open(str(src), gdal.GA_ReadOnly)
    if src_ds is None:
        raise RuntimeError(f"failed to open raster: {src}")
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.CreateCopy(str(dst), src_ds, strict=0)
    if out_ds is None:
        src_ds = None
        raise RuntimeError(f"failed to copy raster to {dst}")
    out_ds.FlushCache()
    out_ds = None
    src_ds = None
    return str(dst)


def _translate_raster(
    src: str | Path,
    dst: Path,
    *,
    driver: str,
    width: int | None = None,
    height: int | None = None,
    resample_alg: str = "nearest",
    output_type: int | None = None,
) -> str:
    src_ds = gdal.Open(str(src), gdal.GA_ReadOnly)
    if src_ds is None:
        raise RuntimeError(f"failed to open raster for translation: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    translate_kwargs = {
        "format": driver,
        "resampleAlg": resample_alg,
    }
    if width is not None:
        translate_kwargs["width"] = int(width)
    if height is not None:
        translate_kwargs["height"] = int(height)
    if output_type is not None:
        translate_kwargs["outputType"] = int(output_type)
    translated = gdal.Translate(str(dst), src_ds, **translate_kwargs)
    src_ds = None
    if translated is None:
        raise RuntimeError(f"failed to translate raster {src} -> {dst}")
    translated.FlushCache()
    translated = None
    return str(dst)


def _read_raster_array(path: str | Path, *, band: int = 1, dtype=np.float32) -> np.ndarray:
    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"failed to open raster: {path}")
    try:
        return np.asarray(_read_band_array(ds.GetRasterBand(band), dtype=dtype), dtype=dtype)
    finally:
        ds = None


def _normalize_offset_array(
    data: np.ndarray,
    *,
    invalid_value: float = NISAR_OFFSET_INVALID_VALUE,
) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64).copy()
    invalid = ~np.isfinite(arr)
    invalid |= arr == GEO2RDR_OFFSET_NODATA
    invalid |= arr <= GEO2RDR_OFFSET_INVALID_LOW
    arr[invalid] = invalid_value
    return arr


def _write_envi_single_band(path: Path, data: np.ndarray, *, dtype=gdal.GDT_Float64) -> str:
    arr = np.asarray(data)
    rows, cols = arr.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = gdal.GetDriverByName("ENVI").Create(str(path), cols, rows, 1, dtype)
    if ds is None:
        raise RuntimeError(f"failed to create ENVI raster: {path}")
    ds.GetRasterBand(1).WriteArray(arr)
    ds.FlushCache()
    ds = None
    return str(path)


def _write_offset_raster(path: Path, data: np.ndarray | None) -> str:
    arr = np.asarray(data if data is not None else np.zeros((1, 1), dtype=np.float32), dtype=np.float32)
    return _write_float_gtiff(path, arr, nodata=GEO2RDR_OFFSET_NODATA)


def _estimate_offset_mean_from_raster(path: str | Path) -> float:
    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"failed to open offset raster: {path}")
    try:
        arr = np.asarray(_read_band_array(ds.GetRasterBand(1), dtype=np.float32), dtype=np.float32)
    finally:
        ds = None
    valid = np.isfinite(arr) & (arr != GEO2RDR_OFFSET_NODATA) & (arr >= GEO2RDR_OFFSET_INVALID_LOW)
    if not np.any(valid):
        raise RuntimeError(f"offset raster has no finite values: {path}")
    return float(np.mean(arr[valid], dtype=np.float64))


def goldstein_filter(
    interferogram: np.ndarray,
    alpha: float = 0.5,
    window_size: int = 32,
    step: int | None = None,
) -> np.ndarray:
    if interferogram.ndim != 2:
        raise ValueError("goldstein_filter expects a 2D interferogram")
    if interferogram.dtype != np.complex64:
        interferogram = interferogram.astype(np.complex64)
    rows, cols = interferogram.shape
    if step is None:
        step = window_size // 2
    hanning = np.outer(np.hanning(window_size), np.hanning(window_size)).astype(np.complex64)
    filtered = np.zeros_like(interferogram, dtype=np.complex64)
    weight_sum = np.zeros((rows, cols), dtype=np.float64)
    for row_start in range(0, rows - window_size + 1, step):
        for col_start in range(0, cols - window_size + 1, step):
            window_data = interferogram[row_start:row_start + window_size, col_start:col_start + window_size].copy()
            spectrum = np.fft.fft2(window_data * hanning)
            psd = np.abs(spectrum) ** 2
            weight = np.power(psd + 1e-10, alpha / 2.0)
            filtered_window = np.fft.ifft2(spectrum * weight)
            filtered[row_start:row_start + window_size, col_start:col_start + window_size] += filtered_window
            weight_sum[row_start:row_start + window_size, col_start:col_start + window_size] += 1.0
    weight_sum[weight_sum == 0] = 1.0
    return (filtered / weight_sum).astype(np.complex64)


def load_manifest(manifest_path: str | Path) -> dict:
    manifest_path = Path(manifest_path)
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def _load_processing_metadata(manifest_path: Path) -> tuple[dict, dict, dict, dict, dict]:
    manifest = load_manifest(manifest_path)
    with open(
        resolve_manifest_metadata_path(manifest_path, manifest, "orbit"),
        encoding="utf-8",
    ) as f:
        orbit_data = json.load(f)
    with open(
        resolve_manifest_metadata_path(manifest_path, manifest, "acquisition"),
        encoding="utf-8",
    ) as f:
        acquisition_data = json.load(f)
    with open(
        resolve_manifest_metadata_path(manifest_path, manifest, "radargrid"),
        encoding="utf-8",
    ) as f:
        radargrid_data = json.load(f)
    with open(
        resolve_manifest_metadata_path(manifest_path, manifest, "doppler"),
        encoding="utf-8",
    ) as f:
        doppler_data = json.load(f)
    return manifest, orbit_data, acquisition_data, radargrid_data, doppler_data


def _resolve_dem_path(
    manifest_path: Path,
    manifest: dict,
    corners,
    dem_path: str | None,
    dem_cache_dir: str | None,
    dem_margin_deg: float,
) -> str:
    if dem_path is not None:
        return str(Path(dem_path))

    manifest_dem = (
        manifest.get("dem", {}).get("path")
        if isinstance(manifest.get("dem"), dict)
        else None
    )
    if manifest_dem is not None:
        resolved_manifest_dem = resolve_manifest_data_path(manifest_path, manifest_dem)
        if resolved_manifest_dem is not None:
            return resolved_manifest_dem
    raise FileNotFoundError(
        "DEM path is required in strip_insar2.py stage-1 migration when manifest has no dem.path"
    )


def _default_gpu_check(gpu_requested: bool | None, gpu_id: int) -> bool:
    try:
        from isce3.core.gpu_check import use_gpu
    except Exception:
        return False
    try:
        return bool(use_gpu(gpu_requested, gpu_id))
    except Exception:
        return False


def extract_scene_date(acquisition_data: dict, orbit_data: dict | None = None) -> str:
    for key in ("startTimeUTC", "start_time_utc", "start_time", "startTime"):
        value = acquisition_data.get(key)
        if value:
            dt = _parse_datetime(str(value))
            return dt.strftime("%Y%m%d")

    orbit_header = (orbit_data or {}).get("header", {})
    for key in ("firstStateTimeUTC", "startTimeUTC"):
        value = orbit_header.get(key)
        if value:
            dt = _parse_datetime(str(value))
            return dt.strftime("%Y%m%d")

    start_gps = acquisition_data.get("startGPSTime")
    if start_gps is not None:
        gps_epoch = datetime(1980, 1, 6, tzinfo=timezone.utc)
        dt = gps_epoch.fromtimestamp(gps_epoch.timestamp() + float(start_gps), tz=timezone.utc)
        return dt.strftime("%Y%m%d")

    raise ValueError("unable to determine scene date from acquisition/orbit metadata")


def build_pair_name(master_date: str, slave_date: str) -> str:
    return f"{master_date}_{slave_date}"


def build_output_paths(output_dir: str | Path, pair_name: str) -> dict[str, str]:
    output_dir = Path(output_dir)
    return {
        "interferogram_h5": str(output_dir / f"{pair_name}_insar.h5"),
        "avg_amplitude_tif": str(output_dir / f"{pair_name}_avg_amplitude_utm_geocoded.tif"),
        "avg_amplitude_png": str(output_dir / f"{pair_name}_avg_amplitude_utm_geocoded.png"),
        "avg_amplitude_kml": str(output_dir / f"{pair_name}_avg_amplitude_utm_geocoded.kml"),
        "interferogram_tif": str(output_dir / f"{pair_name}_interferogram_utm_geocoded.tif"),
        "interferogram_png": str(output_dir / f"{pair_name}_interferogram_wrapped_phase_utm_geocoded.png"),
        "interferogram_kml": str(output_dir / f"{pair_name}_interferogram_wrapped_phase_utm_geocoded.kml"),
        "filtered_interferogram_tif": str(output_dir / f"{pair_name}_filtered_interferogram_utm_geocoded.tif"),
        "filtered_interferogram_png": str(
            output_dir / f"{pair_name}_filtered_interferogram_wrapped_phase_utm_geocoded.png"
        ),
        "filtered_interferogram_kml": str(
            output_dir / f"{pair_name}_filtered_interferogram_wrapped_phase_utm_geocoded.kml"
        ),
        "coherence_tif": str(output_dir / f"{pair_name}_coherence_utm_geocoded.tif"),
        "coherence_png": str(output_dir / f"{pair_name}_coherence_utm_geocoded.png"),
        "coherence_kml": str(output_dir / f"{pair_name}_coherence_utm_geocoded.kml"),
        "unwrapped_phase_tif": str(output_dir / f"{pair_name}_unwrapped_phase_utm_geocoded.tif"),
        "unwrapped_phase_png": str(output_dir / f"{pair_name}_unwrapped_phase_utm_geocoded.png"),
        "unwrapped_phase_kml": str(output_dir / f"{pair_name}_unwrapped_phase_utm_geocoded.kml"),
        "los_displacement_tif": str(output_dir / f"{pair_name}_los_displacement_utm_geocoded.tif"),
        "los_displacement_png": str(output_dir / f"{pair_name}_los_displacement_utm_geocoded.png"),
        "los_displacement_kml": str(output_dir / f"{pair_name}_los_displacement_utm_geocoded.kml"),
    }


def get_wavelength(acquisition_json: dict) -> float:
    return 299792458.0 / float(acquisition_json["centerFrequency"])


def _starting_range_from_radargrid(radargrid_data: dict | None) -> float | None:
    if not isinstance(radargrid_data, dict):
        return None
    for key in ("startingRange", "starting_range"):
        value = radargrid_data.get(key)
        if value is None:
            continue
        try:
            out = float(value)
        except Exception:
            continue
        if np.isfinite(out):
            return out
    range_time_first = radargrid_data.get("rangeTimeFirstPixel")
    if range_time_first is None:
        return None
    try:
        range_time_first = float(range_time_first)
    except Exception:
        return None
    if not np.isfinite(range_time_first):
        return None
    return 0.5 * 299792458.0 * range_time_first


def _compute_ref_sec_starting_range_shift_m(
    master_radargrid_data: dict | None,
    slave_radargrid_data: dict | None,
) -> float:
    master_start = _starting_range_from_radargrid(master_radargrid_data)
    slave_start = _starting_range_from_radargrid(slave_radargrid_data)
    if master_start is None or slave_start is None:
        return 0.0
    shift = float(slave_start - master_start)
    return shift if np.isfinite(shift) else 0.0


def run_stage_with_fallback(
    *,
    stage_name: str,
    gpu_mode: str,
    gpu_id: int,
    gpu_runner,
    cpu_runner,
    gpu_check=None,
) -> tuple[object, str, str | None]:
    gpu_check = gpu_check or _default_gpu_check

    if gpu_mode != "cpu":
        try:
            if gpu_check(True if gpu_mode == "gpu" else None, gpu_id):
                try:
                    return gpu_runner(), "gpu", None
                except Exception as exc:
                    return cpu_runner(), "cpu", f"{stage_name} GPU failed: {exc}"
        except Exception as exc:
            return cpu_runner(), "cpu", f"{stage_name} GPU unavailable: {exc}"

    return cpu_runner(), "cpu", None


def load_pair_context(
    master_manifest_path: str | Path,
    slave_manifest_path: str | Path,
    *,
    output_root: str | Path,
    dem_path: str | None = None,
    dem_cache_dir: str | None = None,
    dem_margin_deg: float = 0.2,
) -> PairContext:
    master_manifest_path = Path(master_manifest_path)
    slave_manifest_path = Path(slave_manifest_path)
    (
        master_manifest,
        master_orbit_data,
        master_acq_data,
        master_rg_data,
        master_dop_data,
    ) = _load_processing_metadata(master_manifest_path)
    (
        slave_manifest,
        slave_orbit_data,
        slave_acq_data,
        slave_rg_data,
        slave_dop_data,
    ) = _load_processing_metadata(slave_manifest_path)

    master_date = extract_scene_date(master_acq_data, master_orbit_data)
    slave_date = extract_scene_date(slave_acq_data, slave_orbit_data)
    pair_name = build_pair_name(master_date, slave_date)

    output_root = Path(output_root)
    pair_dir = output_root / pair_name
    pair_dir.mkdir(parents=True, exist_ok=True)

    try:
        scene_corners = load_scene_corners_with_fallback(
            master_manifest_path,
            master_manifest,
        )
    except Exception:
        scene_corners = []
    resolved_dem = _resolve_dem_path(
        master_manifest_path,
        master_manifest,
        scene_corners,
        dem_path,
        dem_cache_dir,
        dem_margin_deg,
    )
    orbit_interp = choose_orbit_interp(master_orbit_data, master_acq_data)
    wavelength = get_wavelength(master_acq_data)

    return PairContext(
        master_manifest_path=master_manifest_path,
        slave_manifest_path=slave_manifest_path,
        master_manifest=master_manifest,
        slave_manifest=slave_manifest,
        master_orbit_data=master_orbit_data,
        slave_orbit_data=slave_orbit_data,
        master_acq_data=master_acq_data,
        slave_acq_data=slave_acq_data,
        master_rg_data=master_rg_data,
        slave_rg_data=slave_rg_data,
        master_dop_data=master_dop_data,
        slave_dop_data=slave_dop_data,
        output_root=output_root,
        pair_name=pair_name,
        pair_dir=pair_dir,
        output_paths=build_output_paths(pair_dir, pair_name),
        resolved_dem=resolved_dem,
        orbit_interp=orbit_interp,
        wavelength=wavelength,
    )


def _write_stage_outputs_record(
    *,
    output_dir: Path,
    stage: str,
    master_manifest_path: str | Path,
    slave_manifest_path: str | Path,
    backend_used: str,
    output_files: dict,
    fallback_reason: str | None = None,
) -> None:
    upstream = {
        "p0": ["prep"],
        "p1": ["p0"],
        "p2": ["p1"],
        "p3": ["p2"],
        "p4": ["p3"],
    }
    record = {
        "stage": stage,
        "input_manifests": {
            "master": str(master_manifest_path),
            "slave": str(slave_manifest_path),
        },
        "effective_crop": {},
        "backend_used": backend_used,
        "upstream_stage_dependencies": upstream.get(stage, []),
        "output_files": output_files,
        "start_time": utc_now_iso(),
        "end_time": utc_now_iso(),
        "success": True,
        "fallback_reason": fallback_reason,
    }
    write_stage_record(output_dir, stage, record)
    mark_stage_success(output_dir, stage)


def _write_custom_stage_record(
    *,
    output_dir: Path,
    stage: str,
    master_manifest_path: Path,
    slave_manifest_path: Path,
    backend_used: str,
    output_files: dict,
    fallback_reason: str | None = None,
    upstream_stage_dependencies: list[str] | None = None,
) -> None:
    record = {
        "stage": stage,
        "input_manifests": {
            "master": str(master_manifest_path),
            "slave": str(slave_manifest_path),
        },
        "effective_crop": {},
        "backend_used": backend_used,
        "upstream_stage_dependencies": upstream_stage_dependencies or [],
        "output_files": output_files,
        "start_time": utc_now_iso(),
        "end_time": utc_now_iso(),
        "success": True,
        "fallback_reason": fallback_reason,
    }
    write_stage_record(output_dir, stage, record)
    mark_stage_success(output_dir, stage)


def _save_stage_array(output_dir: Path, stage: str, name: str, arr: np.ndarray) -> str:
    path = stage_dir(output_dir, stage)
    path.mkdir(parents=True, exist_ok=True)
    array_path = path / f"{name}.npy"
    np.save(array_path, arr)
    return str(array_path)


def _load_stage_output_path(output_dir: Path, stage: str, key: str) -> str:
    record = load_stage_record(output_dir, stage) or {}
    path = record.get("output_files", {}).get(key)
    if not path:
        raise RuntimeError(f"Missing cached output '{key}' for stage '{stage}'")
    return str(path)


def _load_cached_array(output_dir: Path, stage: str, key: str) -> np.ndarray:
    return np.load(_load_stage_output_path(output_dir, stage, key))


def _load_cached_stage_outputs(
    output_dir: Path,
    stage: str,
    *,
    required_keys: tuple[str, ...],
) -> dict | None:
    if not success_marker_path(output_dir, stage).is_file():
        return None
    record = load_stage_record(output_dir, stage) or {}
    if not record.get("success"):
        return None
    output_files = record.get("output_files")
    if not isinstance(output_files, dict):
        return None
    for key in required_keys:
        value = output_files.get(key)
        if not value:
            return None
        try:
            if not Path(str(value)).exists():
                return None
        except Exception:
            return None
    return dict(output_files)


def _write_complex_gtiff(path: Path, data: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows, cols = data.shape
    ds = gdal.GetDriverByName("GTiff").Create(
        str(path),
        cols,
        rows,
        1,
        gdal.GDT_CFloat32,
        options=["COMPRESS=LZW", "TILED=YES"],
    )
    if ds is None:
        raise RuntimeError(f"failed to create raster: {path}")
    ds.GetRasterBand(1).WriteArray(np.asarray(data, dtype=np.complex64))
    ds.FlushCache()
    ds = None
    return str(path)


def _write_float_gtiff(path: Path, data: np.ndarray, *, dtype=gdal.GDT_Float32, nodata=None) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows, cols = data.shape
    ds = gdal.GetDriverByName("GTiff").Create(
        str(path),
        cols,
        rows,
        1,
        dtype,
        options=["COMPRESS=LZW", "TILED=YES"],
    )
    if ds is None:
        raise RuntimeError(f"failed to create raster: {path}")
    band = ds.GetRasterBand(1)
    if nodata is not None:
        band.SetNoDataValue(float(nodata))
    band.WriteArray(data)
    ds.FlushCache()
    ds = None
    return str(path)


def _open_slc_as_complex(slc_path: str) -> np.ndarray:
    ds = gdal.Open(slc_path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"failed to open SLC: {slc_path}")
    try:
        band1 = ds.GetRasterBand(1)
        if ds.RasterCount >= 2:
            band2 = ds.GetRasterBand(2)
            real = _read_band_array(band1, dtype=np.float32).astype(np.float32)
            imag = _read_band_array(band2, dtype=np.float32).astype(np.float32)
            out = real.astype(np.complex64) + 1j * imag.astype(np.complex64)
        else:
            out = _read_band_array(band1, dtype=np.complex64).astype(np.complex64)
    finally:
        ds = None
    return out


def _compute_slc_amplitude(slc_path: str) -> np.ndarray:
    slc = _open_slc_as_complex(slc_path)
    return np.abs(slc).astype(np.float32)


def _write_radar_amplitude_png(
    slc_path: str,
    output_png: str | Path,
) -> str:
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    amplitude = _compute_slc_amplitude(slc_path)
    valid = np.isfinite(amplitude) & (amplitude > 0)
    image = np.zeros(amplitude.shape, dtype=np.uint8)
    if np.any(valid):
        vals = 20.0 * np.log10(amplitude[valid])
        lo = float(np.percentile(vals, 2))
        hi = float(np.percentile(vals, 98))
        scaled = np.clip((20.0 * np.log10(amplitude[valid]) - lo) / (hi - lo + 1.0e-9), 0.0, 1.0)
        image[valid] = np.rint(scaled * 255.0).astype(np.uint8)
    Image.fromarray(image, mode="L").save(output_png)
    return str(output_png)


def _write_radar_wrapped_phase_png(
    interferogram: np.ndarray,
    output_png: str | Path,
) -> str:
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    phase = np.angle(np.asarray(interferogram, dtype=np.complex64))
    valid = np.isfinite(phase)
    hsv = np.zeros((*phase.shape, 3), dtype=np.uint8)
    if np.any(valid):
        hue = np.mod((phase[valid] + np.pi) / (2.0 * np.pi), 1.0)
        hsv[..., 0][valid] = np.rint(hue * 255.0).astype(np.uint8)
        hsv[..., 1][valid] = 255
        hsv[..., 2][valid] = 255
    Image.fromarray(hsv, mode="HSV").convert("RGB").save(output_png)
    return str(output_png)


def append_topo_coordinates_hdf(
    manifest_path: str,
    dem_path: str,
    output_h5: str,
    block_rows: int = 256,
    orbit_interp: str | None = None,
    use_gpu: bool = False,
    gpu_id: int = 0,
) -> str:
    import isce3.core
    import isce3.io

    manifest_path = Path(manifest_path)
    output_h5 = Path(output_h5)
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    with open(resolve_manifest_metadata_path(manifest_path, manifest, "radargrid"), encoding="utf-8") as f:
        radargrid_data = json.load(f)
    with open(resolve_manifest_metadata_path(manifest_path, manifest, "orbit"), encoding="utf-8") as f:
        orbit_data = json.load(f)
    with open(resolve_manifest_metadata_path(manifest_path, manifest, "acquisition"), encoding="utf-8") as f:
        acquisition_data = json.load(f)

    width = radargrid_data["numberOfColumns"]
    length = radargrid_data["numberOfRows"]
    if orbit_interp is None:
        orbit_interp = choose_orbit_interp(orbit_data, acquisition_data)
    orbit = construct_orbit(orbit_data, orbit_interp)
    radar_grid = construct_radar_grid(radargrid_data, acquisition_data, orbit_data)
    dem_raster = isce3.io.Raster(str(dem_path))

    if use_gpu:
        import isce3.cuda.core
        import isce3.cuda.geometry

        device = isce3.cuda.core.Device(gpu_id)
        isce3.cuda.core.set_device(device)
        rdr2geo_cls = isce3.cuda.geometry.Rdr2Geo
    else:
        import isce3.geometry

        rdr2geo_cls = isce3.geometry.Rdr2Geo

    topo = rdr2geo_cls(
        radar_grid,
        orbit,
        isce3.core.Ellipsoid(),
        isce3.core.LUT2d(),
        epsg_out=4326,
        compute_mask=True,
        lines_per_block=block_rows,
    )

    tmp_parent = output_h5.parent / ".topo_tmp"
    tmp_parent.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="d2sar_topo_gtiff_", dir=str(tmp_parent)))
    try:
        def make_raster(name: str, dtype: int = gdal.GDT_Float32):
            return isce3.io.Raster(
                str(workdir / f"{name}.tif"),
                radar_grid.width,
                radar_grid.length,
                1,
                dtype,
                "GTiff",
            )

        x_raster = make_raster("x")
        y_raster = make_raster("y")
        z_raster = make_raster("z")
        inc_raster = make_raster("inc")
        hdg_raster = make_raster("hdg")
        local_inc_raster = make_raster("localInc")
        local_psi_raster = make_raster("localPsi")
        simamp_raster = make_raster("simamp")
        layover_raster = make_raster("layoverShadowMask", gdal.GDT_Byte)
        los_e_raster = make_raster("los_east")
        los_n_raster = make_raster("los_north")

        topo.topo(
            dem_raster,
            x_raster,
            y_raster,
            z_raster,
            inc_raster,
            hdg_raster,
            local_inc_raster,
            local_psi_raster,
            simamp_raster,
            layover_raster,
            los_e_raster,
            los_n_raster,
        )
        for raster in [
            x_raster, y_raster, z_raster, inc_raster, hdg_raster, local_inc_raster,
            local_psi_raster, simamp_raster, layover_raster, los_e_raster, los_n_raster,
        ]:
            raster.close_dataset()

        lon_ds = gdal.Open(str(workdir / "x.tif"))
        lat_ds = gdal.Open(str(workdir / "y.tif"))
        hgt_ds = gdal.Open(str(workdir / "z.tif"))
        if lon_ds is None or lat_ds is None or hgt_ds is None:
            raise RuntimeError("failed to reopen topo GTiff rasters")

        with h5py.File(output_h5, "a") as f:
            for name in ("longitude", "latitude", "height"):
                if name in f:
                    del f[name]
            d_lon = f.create_dataset("longitude", shape=(length, width), dtype="f4", chunks=(min(block_rows, length), min(1024, width)), compression="gzip", shuffle=True)
            d_lat = f.create_dataset("latitude", shape=(length, width), dtype="f4", chunks=(min(block_rows, length), min(1024, width)), compression="gzip", shuffle=True)
            d_hgt = f.create_dataset("height", shape=(length, width), dtype="f4", chunks=(min(block_rows, length), min(1024, width)), compression="gzip", shuffle=True)
            f.attrs["coordinate_system"] = "EPSG:4326"
            f.attrs["longitude_units"] = "degrees_east"
            f.attrs["latitude_units"] = "degrees_north"
            f.attrs["height_units"] = "meters"
            f.attrs["coordinate_source"] = "rdr2geo_topo_with_validated_dem"

            lon_band = lon_ds.GetRasterBand(1)
            lat_band = lat_ds.GetRasterBand(1)
            hgt_band = hgt_ds.GetRasterBand(1)
            for row0 in range(0, length, block_rows):
                rows = min(block_rows, length - row0)
                d_lon[row0:row0 + rows, :] = _read_band_array(lon_band, 0, row0, width, rows).astype(np.float32)
                d_lat[row0:row0 + rows, :] = _read_band_array(lat_band, 0, row0, width, rows).astype(np.float32)
                d_hgt[row0:row0 + rows, :] = _read_band_array(hgt_band, 0, row0, width, rows).astype(np.float32)
        return str(output_h5)
    finally:
        dem_raster.close_dataset()
        shutil.rmtree(workdir, ignore_errors=True)
        if tmp_parent.exists() and not any(tmp_parent.iterdir()):
            tmp_parent.rmdir()


def point2epsg(lon: float, lat: float) -> int:
    if lon >= 180.0:
        lon -= 360.0
    if lat >= 75.0:
        return 3413
    if lat <= -75.0:
        return 3031
    if lat > 0:
        return 32601 + int(np.round((lon + 177) / 6.0))
    if lat < 0:
        return 32701 + int(np.round((lon + 177) / 6.0))
    raise ValueError(f"Could not determine projection for {lat},{lon}")


def append_utm_coordinates_hdf(output_h5: str, manifest_path: str, block_rows: int = 32) -> str:
    output_h5 = Path(output_h5)
    manifest_path = Path(manifest_path)
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    corners = load_scene_corners_with_fallback(manifest_path, manifest)
    if not corners:
        with h5py.File(output_h5, "a") as f:
            lon_ds = f["longitude"]
            lat_ds = f["latitude"]
            valid = np.isfinite(lon_ds[()]) & np.isfinite(lat_ds[()])
            if not np.any(valid):
                raise RuntimeError("cannot derive UTM coordinates without valid lon/lat")
            center_lon = float(np.nanmean(lon_ds[()][valid]))
            center_lat = float(np.nanmean(lat_ds[()][valid]))
    else:
        center_lon = sum(pt["lon"] for pt in corners) / len(corners)
        center_lat = sum(pt["lat"] for pt in corners) / len(corners)
    epsg = point2epsg(center_lon, center_lat)

    src = osr.SpatialReference()
    src.ImportFromEPSG(4326)
    src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    dst = osr.SpatialReference()
    dst.ImportFromEPSG(epsg)
    dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osr.CoordinateTransformation(src, dst)

    with h5py.File(output_h5, "a") as f:
        lon_ds = f["longitude"]
        lat_ds = f["latitude"]
        length, width = lon_ds.shape
        for name in ("utm_x", "utm_y"):
            if name in f:
                del f[name]
        utm_x = f.create_dataset("utm_x", shape=(length, width), dtype="f4", chunks=(min(block_rows, length), min(1024, width)), compression="gzip", shuffle=True)
        utm_y = f.create_dataset("utm_y", shape=(length, width), dtype="f4", chunks=(min(block_rows, length), min(1024, width)), compression="gzip", shuffle=True)
        f.attrs["utm_epsg"] = epsg
        f.attrs["utm_coordinate_source"] = "transformed_from_topo_driven_lonlat"
        for row0 in range(0, length, block_rows):
            rows = min(block_rows, length - row0)
            lon_block = lon_ds[row0:row0 + rows, :]
            lat_block = lat_ds[row0:row0 + rows, :]
            x_block = np.full((rows, width), np.nan, dtype=np.float32)
            y_block = np.full((rows, width), np.nan, dtype=np.float32)
            valid = np.isfinite(lon_block) & np.isfinite(lat_block) & (lon_block >= -180.0) & (lon_block <= 180.0) & (lat_block >= -90.0) & (lat_block <= 90.0)
            if np.any(valid):
                pts = np.column_stack([lon_block[valid], lat_block[valid]])
                transformed = np.asarray(transform.TransformPoints(pts[:, :2]), dtype=np.float64)
                x_block[valid] = transformed[:, 0].astype(np.float32)
                y_block[valid] = transformed[:, 1].astype(np.float32)
            utm_x[row0:row0 + rows, :] = x_block
            utm_y[row0:row0 + rows, :] = y_block
    return str(output_h5)


def compute_utm_output_shape(input_h5: str, resolution_meters: float, block_rows: int = 64) -> tuple[int, int]:
    input_h5 = Path(input_h5)
    with h5py.File(input_h5, "r") as f:
        x_ds = f["utm_x"]
        y_ds = f["utm_y"]
        length, _ = x_ds.shape
        x_min = np.inf
        x_max = -np.inf
        y_min = np.inf
        y_max = -np.inf
        for row0 in range(0, length, block_rows):
            rows = min(block_rows, length - row0)
            x = x_ds[row0:row0 + rows, :]
            y = y_ds[row0:row0 + rows, :]
            valid = np.isfinite(x) & np.isfinite(y)
            if np.any(valid):
                x_min = min(x_min, float(np.nanmin(x[valid])))
                x_max = max(x_max, float(np.nanmax(x[valid])))
                y_min = min(y_min, float(np.nanmin(y[valid])))
                y_max = max(y_max, float(np.nanmax(y[valid])))
    target_width = max(1, int(round((x_max - x_min) / resolution_meters)))
    target_height = max(1, int(round((y_max - y_min) / resolution_meters)))
    return target_width, target_height


def accumulate_utm_grid(
    input_h5: str,
    dataset_name: str,
    target_width: int,
    target_height: int | None = None,
    block_rows: int = 64,
) -> tuple[np.ndarray, dict]:
    input_h5 = Path(input_h5)
    with h5py.File(input_h5, "r") as f:
        x_ds = f["utm_x"]
        y_ds = f["utm_y"]
        amp_ds = f[dataset_name]
        length, width = amp_ds.shape
        utm_epsg = int(f.attrs["utm_epsg"])
        x_min = np.inf
        x_max = -np.inf
        y_min = np.inf
        y_max = -np.inf
        for row0 in range(0, length, block_rows):
            rows = min(block_rows, length - row0)
            x = x_ds[row0:row0 + rows, :]
            y = y_ds[row0:row0 + rows, :]
            valid = np.isfinite(x) & np.isfinite(y)
            if np.any(valid):
                x_min = min(x_min, float(np.nanmin(x[valid])))
                x_max = max(x_max, float(np.nanmax(x[valid])))
                y_min = min(y_min, float(np.nanmin(y[valid])))
                y_max = max(y_max, float(np.nanmax(y[valid])))
        aspect = (y_max - y_min) / max(x_max - x_min, 1e-9)
        if target_height is None:
            target_height = max(1, int(round(target_width * aspect)))
        sums = np.zeros((target_height, target_width), dtype=np.float64)
        counts = np.zeros((target_height, target_width), dtype=np.uint32)
        for row0 in range(0, length, block_rows):
            rows = min(block_rows, length - row0)
            x = x_ds[row0:row0 + rows, :]
            y = y_ds[row0:row0 + rows, :]
            data = amp_ds[row0:row0 + rows, :]
            valid = np.isfinite(x) & np.isfinite(y)
            if np.iscomplexobj(data):
                valid &= np.isfinite(data.real) & np.isfinite(data.imag)
                vals = np.abs(data[valid]).astype(np.float64)
            else:
                valid &= np.isfinite(data)
                vals = data[valid].astype(np.float64)
            if not np.any(valid):
                continue
            x_valid = x[valid]
            y_valid = y[valid]
            col = np.clip(((x_valid - x_min) / max(x_max - x_min, 1e-9) * (target_width - 1)).astype(np.int32), 0, target_width - 1)
            row = np.clip(((y_max - y_valid) / max(y_max - y_min, 1e-9) * (target_height - 1)).astype(np.int32), 0, target_height - 1)
            flat_idx = row * target_width + col
            np.add.at(sums.ravel(), flat_idx, vals)
            np.add.at(counts.ravel(), flat_idx, 1)
    out = np.full((target_height, target_width), np.nan, dtype=np.float64)
    mask = counts > 0
    out[mask] = sums[mask] / counts[mask]
    meta = {
        "utm_epsg": utm_epsg,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "target_width": target_width,
        "target_height": target_height,
        "x_res": (x_max - x_min) / max(target_width - 1, 1),
        "y_res": (y_max - y_min) / max(target_height - 1, 1),
    }
    return out, meta


def write_geocoded_geotiff(
    input_h5: str,
    output_tif: str,
    dataset_name: str,
    target_width: int,
    target_height: int | None = None,
    block_rows: int = 64,
) -> str:
    out, meta = accumulate_utm_grid(input_h5, dataset_name, target_width, target_height, block_rows)
    out = np.nan_to_num(out, nan=0.0).astype(np.float32)
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(str(output_tif), meta["target_width"], meta["target_height"], 1, gdal.GDT_Float32, options=["COMPRESS=LZW", "TILED=YES"])
    ds.SetGeoTransform([meta["x_min"], meta["x_res"], 0.0, meta["y_max"], 0.0, -meta["y_res"]])
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(meta["utm_epsg"])
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(0.0)
    _write_band_array(band, out)
    band.FlushCache()
    ds.FlushCache()
    ds = None
    return str(output_tif)


def write_geocoded_png(
    input_h5: str,
    output_png: str,
    dataset_name: str,
    target_width: int,
    target_height: int | None = None,
    block_rows: int = 64,
) -> str:
    out, _ = accumulate_utm_grid(input_h5, dataset_name, target_width, target_height, block_rows)
    img = np.zeros(out.shape, dtype=np.uint8)
    valid = np.isfinite(out)
    if np.any(valid):
        vals = out[valid]
        p2 = np.percentile(vals, 2)
        p98 = np.percentile(vals, 98)
        scaled = np.clip((vals - p2) / (p98 - p2 + 1.0e-9), 0.0, 1.0)
        img[valid] = (scaled * 255).astype(np.uint8)
    Image.fromarray(img, mode="L").save(output_png)
    return str(output_png)


def _accumulate_wrapped_phase_grid(
    input_h5: str,
    dataset_name: str,
    target_width: int,
    target_height: int | None = None,
    block_rows: int = 64,
) -> tuple[np.ndarray, dict]:
    input_h5 = Path(input_h5)
    with h5py.File(input_h5, "r") as f:
        x_ds = f["utm_x"]
        y_ds = f["utm_y"]
        ifg_ds = f[dataset_name]
        length, width = ifg_ds.shape
        utm_epsg = int(f.attrs["utm_epsg"])
        x_min = np.inf
        x_max = -np.inf
        y_min = np.inf
        y_max = -np.inf
        for row0 in range(0, length, block_rows):
            rows = min(block_rows, length - row0)
            x = x_ds[row0:row0 + rows, :]
            y = y_ds[row0:row0 + rows, :]
            valid = np.isfinite(x) & np.isfinite(y)
            if np.any(valid):
                x_min = min(x_min, float(np.nanmin(x[valid])))
                x_max = max(x_max, float(np.nanmax(x[valid])))
                y_min = min(y_min, float(np.nanmin(y[valid])))
                y_max = max(y_max, float(np.nanmax(y[valid])))
        aspect = (y_max - y_min) / max(x_max - x_min, 1e-9)
        if target_height is None:
            target_height = max(1, int(round(target_width * aspect)))
        strongest = np.zeros((target_height, target_width), dtype=np.complex64)
        strongest_amp = np.zeros((target_height, target_width), dtype=np.float32)
        counts = np.zeros((target_height, target_width), dtype=np.uint32)
        for row0 in range(0, length, block_rows):
            rows = min(block_rows, length - row0)
            x = x_ds[row0:row0 + rows, :]
            y = y_ds[row0:row0 + rows, :]
            ifg = ifg_ds[row0:row0 + rows, :]
            valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(ifg.real) & np.isfinite(ifg.imag)
            if not np.any(valid):
                continue
            x_valid = x[valid]
            y_valid = y[valid]
            ifg_valid = ifg[valid]
            col = np.clip(((x_valid - x_min) / max(x_max - x_min, 1e-9) * (target_width - 1)).astype(np.int32), 0, target_width - 1)
            row = np.clip(((y_max - y_valid) / max(y_max - y_min, 1e-9) * (target_height - 1)).astype(np.int32), 0, target_height - 1)
            flat_idx = row * target_width + col
            amps = np.abs(ifg_valid).astype(np.float32)
            strongest_flat = strongest.ravel()
            strongest_amp_flat = strongest_amp.ravel()
            for idx, amp, val in zip(flat_idx, amps, ifg_valid, strict=False):
                if amp > strongest_amp_flat[idx]:
                    strongest_amp_flat[idx] = amp
                    strongest_flat[idx] = np.complex64(val)
            np.add.at(counts.ravel(), flat_idx, 1)
    out = np.full((target_height, target_width), np.nan, dtype=np.float32)
    mask = counts > 0
    out[mask] = np.angle(strongest[mask]).astype(np.float32)
    meta = {
        "utm_epsg": utm_epsg,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "target_width": target_width,
        "target_height": target_height,
        "x_res": (x_max - x_min) / max(target_width - 1, 1),
        "y_res": (y_max - y_min) / max(target_height - 1, 1),
    }
    return out, meta


def write_wrapped_phase_geotiff(
    input_h5: str,
    output_tif: str,
    dataset_name: str,
    target_width: int,
    target_height: int | None = None,
    block_rows: int = 64,
) -> str:
    out, meta = _accumulate_wrapped_phase_grid(input_h5, dataset_name, target_width, target_height, block_rows)
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(str(output_tif), meta["target_width"], meta["target_height"], 1, gdal.GDT_Float32, options=["COMPRESS=LZW", "TILED=YES"])
    ds.SetGeoTransform([meta["x_min"], meta["x_res"], 0.0, meta["y_max"], 0.0, -meta["y_res"]])
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(meta["utm_epsg"])
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(np.nan)
    _write_band_array(band, out.astype(np.float32))
    band.FlushCache()
    ds.FlushCache()
    ds = None
    return str(output_tif)


def write_wrapped_phase_png(
    input_h5: str,
    output_png: str,
    dataset_name: str,
    target_width: int,
    target_height: int | None = None,
    block_rows: int = 64,
) -> str:
    phase, _ = _accumulate_wrapped_phase_grid(input_h5, dataset_name, target_width, target_height, block_rows)
    rgb = np.zeros((*phase.shape, 3), dtype=np.uint8)
    valid = np.isfinite(phase)
    if np.any(valid):
        hue = ((phase[valid] + np.pi) / (2.0 * np.pi)).astype(np.float64)
        colors = np.array([colorsys.hsv_to_rgb(float(h), 1.0, 1.0) for h in hue])
        rgb[valid] = (colors * 255.0).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(output_png)
    return str(output_png)


def write_ground_overlay_kml(
    *,
    image_path: str | Path,
    output_kml: str | Path,
    west: float,
    east: float,
    south: float,
    north: float,
    overlay_name: str | None = None,
) -> str:
    image_path = Path(image_path)
    output_kml = Path(output_kml)
    output_kml.parent.mkdir(parents=True, exist_ok=True)
    name = overlay_name or image_path.stem
    href = escape(image_path.name)
    content = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <GroundOverlay>
    <name>{escape(name)}</name>
    <Icon>
      <href>{href}</href>
    </Icon>
    <LatLonBox>
      <north>{float(north)}</north>
      <south>{float(south)}</south>
      <east>{float(east)}</east>
      <west>{float(west)}</west>
    </LatLonBox>
  </GroundOverlay>
</kml>
"""
    output_kml.write_text(content, encoding="utf-8")
    return str(output_kml)


def _to_geographic_bounds(*, projection_wkt: str, west: float, east: float, south: float, north: float) -> tuple[float, float, float, float]:
    src = osr.SpatialReference()
    src.ImportFromWkt(projection_wkt)
    src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    dst = osr.SpatialReference()
    dst.ImportFromEPSG(4326)
    dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    tx = osr.CoordinateTransformation(src, dst)
    corners = [
        tx.TransformPoint(float(west), float(north)),
        tx.TransformPoint(float(east), float(north)),
        tx.TransformPoint(float(east), float(south)),
        tx.TransformPoint(float(west), float(south)),
    ]
    lons = [float(c[0]) for c in corners]
    lats = [float(c[1]) for c in corners]
    return min(lons), max(lons), min(lats), max(lats)


def write_ground_overlay_kml_from_geotiff(
    *,
    tif_path: str | Path,
    image_path: str | Path,
    output_kml: str | Path,
    overlay_name: str | None = None,
) -> str:
    tif_path = Path(tif_path)
    ds = gdal.Open(str(tif_path), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"failed to open GeoTIFF for KML bounds: {tif_path}")
    gt = ds.GetGeoTransform(can_return_null=True)
    if gt is None:
        ds = None
        raise RuntimeError(f"GeoTIFF missing geotransform: {tif_path}")
    projection = ds.GetProjectionRef()
    width = int(ds.RasterXSize)
    height = int(ds.RasterYSize)
    ds = None
    west = float(gt[0])
    east = float(gt[0] + gt[1] * width)
    north = float(gt[3])
    south = float(gt[3] + gt[5] * height)
    west, east = sorted((west, east))
    south, north = sorted((south, north))
    if projection:
        west, east, south, north = _to_geographic_bounds(
            projection_wkt=projection, west=west, east=east, south=south, north=north
        )
    return write_ground_overlay_kml(
        image_path=image_path,
        output_kml=output_kml,
        west=west,
        east=east,
        south=south,
        north=north,
        overlay_name=overlay_name,
    )


def export_insar_products(
    *,
    input_h5: str | Path,
    output_paths: dict[str, str],
    resolution_meters: float,
    block_rows: int = 64,
    generate_kml: bool = True,
) -> dict[str, str]:
    input_h5 = str(input_h5)
    target_width, target_height = compute_utm_output_shape(input_h5, resolution_meters)
    exported: dict[str, str] = {}
    scalar_datasets = (
        ("avg_amplitude", "avg_amplitude_tif", "avg_amplitude_png"),
        ("coherence", "coherence_tif", "coherence_png"),
        ("unwrapped_phase", "unwrapped_phase_tif", "unwrapped_phase_png"),
        ("los_displacement", "los_displacement_tif", "los_displacement_png"),
    )
    for dataset_name, tif_key, png_key in scalar_datasets:
        tif_path = output_paths[tif_key]
        png_path = output_paths[png_key]
        write_geocoded_geotiff(input_h5, tif_path, dataset_name, target_width, target_height, block_rows)
        write_geocoded_png(input_h5, png_path, dataset_name, target_width, target_height, block_rows)
        exported[tif_key] = tif_path
        exported[png_key] = png_path
        if generate_kml:
            kml_key = png_key.replace("_png", "_kml")
            exported[kml_key] = write_ground_overlay_kml_from_geotiff(
                tif_path=tif_path,
                image_path=png_path,
                output_kml=output_paths[kml_key],
                overlay_name=Path(png_path).stem,
            )
    wrapped_datasets = (
        ("interferogram", "interferogram_tif", "interferogram_png"),
        ("filtered_interferogram", "filtered_interferogram_tif", "filtered_interferogram_png"),
    )
    for dataset_name, tif_key, png_key in wrapped_datasets:
        tif_path = output_paths[tif_key]
        png_path = output_paths[png_key]
        write_wrapped_phase_geotiff(input_h5, tif_path, dataset_name, target_width, target_height, block_rows)
        write_wrapped_phase_png(input_h5, png_path, dataset_name, target_width, target_height, block_rows)
        exported[tif_key] = tif_path
        exported[png_key] = png_path
        if generate_kml:
            kml_key = png_key.replace("_png", "_kml")
            exported[kml_key] = write_ground_overlay_kml_from_geotiff(
                tif_path=tif_path,
                image_path=png_path,
                output_kml=output_paths[kml_key],
                overlay_name=Path(png_path).stem,
            )
    return exported


def _construct_doppler_if_possible(
    doppler_data: dict | None,
    *,
    orbit_data: dict | None,
    acquisition_data: dict | None,
    radargrid_data: dict | None,
):
    if not doppler_data or "combinedDoppler" not in doppler_data:
        return None
    if orbit_data is None or acquisition_data is None or radargrid_data is None:
        return None
    try:
        return construct_doppler_lut2d(
            doppler_data,
            radargrid_json=radargrid_data,
            acquisition_json=acquisition_data,
            orbit_json=orbit_data,
        )
    except Exception:
        return None


def _build_topo_vrt(target_dir: Path, *, epsg: int) -> str:
    vrt_path = target_dir / "topo.vrt"
    x_path = target_dir / "x.tif"
    y_path = target_dir / "y.tif"
    z_path = target_dir / "z.tif"
    try:
        import isce3.io

        raster_list = [
            isce3.io.Raster(str(x_path)),
            isce3.io.Raster(str(y_path)),
            isce3.io.Raster(str(z_path)),
        ]
        output_vrt = isce3.io.Raster(str(vrt_path), raster_list)
        output_vrt.set_epsg(int(epsg))
        output_vrt.close_dataset()
        for raster in raster_list:
            raster.close_dataset()
    except Exception:
        vrt = gdal.BuildVRT(str(vrt_path), [str(x_path), str(y_path), str(z_path)], separate=True)
        if vrt is None:
            raise RuntimeError(f"failed to build topo vrt: {vrt_path}")
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(int(epsg))
        vrt.SetProjection(srs.ExportToWkt())
        vrt.FlushCache()
        vrt = None
    return str(vrt_path)


def _run_rdr2geo_topo(
    *,
    orbit_data: dict,
    acquisition_data: dict,
    radargrid_data: dict,
    dem_path: str,
    orbit_interp: str,
    use_gpu: bool,
    gpu_id: int,
    output_dir: Path,
    block_rows: int,
) -> str:
    import isce3.core
    import isce3.io

    orbit = construct_orbit(orbit_data, orbit_interp)
    radar_grid = construct_radar_grid(radargrid_data, acquisition_data, orbit_data)
    dem_raster = isce3.io.Raster(str(dem_path))

    if use_gpu:
        import isce3.cuda.core
        import isce3.cuda.geometry

        device = isce3.cuda.core.Device(gpu_id)
        isce3.cuda.core.set_device(device)
        rdr2geo_cls = isce3.cuda.geometry.Rdr2Geo
    else:
        import isce3.geometry

        rdr2geo_cls = isce3.geometry.Rdr2Geo

    output_dir.mkdir(parents=True, exist_ok=True)
    topo = rdr2geo_cls(
        radar_grid,
        orbit,
        isce3.core.Ellipsoid(),
        isce3.core.LUT2d(),
        epsg_out=4326,
        compute_mask=False,
        lines_per_block=block_rows,
    )

    x_raster = isce3.io.Raster(
        str(output_dir / "x.tif"), radar_grid.width, radar_grid.length, 1, gdal.GDT_Float64, "GTiff"
    )
    y_raster = isce3.io.Raster(
        str(output_dir / "y.tif"), radar_grid.width, radar_grid.length, 1, gdal.GDT_Float64, "GTiff"
    )
    z_raster = isce3.io.Raster(
        str(output_dir / "z.tif"), radar_grid.width, radar_grid.length, 1, gdal.GDT_Float64, "GTiff"
    )
    inc_raster = isce3.io.Raster(
        str(output_dir / "inc.tif"), radar_grid.width, radar_grid.length, 1, gdal.GDT_Float32, "GTiff"
    )
    hdg_raster = isce3.io.Raster(
        str(output_dir / "hdg.tif"), radar_grid.width, radar_grid.length, 1, gdal.GDT_Float32, "GTiff"
    )
    local_inc_raster = isce3.io.Raster(
        str(output_dir / "localInc.tif"), radar_grid.width, radar_grid.length, 1, gdal.GDT_Float32, "GTiff"
    )
    local_psi_raster = isce3.io.Raster(
        str(output_dir / "localPsi.tif"), radar_grid.width, radar_grid.length, 1, gdal.GDT_Float32, "GTiff"
    )
    simamp_raster = isce3.io.Raster(
        str(output_dir / "simamp.tif"), radar_grid.width, radar_grid.length, 1, gdal.GDT_Float32, "GTiff"
    )
    layover_raster = isce3.io.Raster(
        str(output_dir / "layoverShadowMask.tif"),
        radar_grid.width,
        radar_grid.length,
        1,
        gdal.GDT_Byte,
        "GTiff",
    )
    los_e_raster = isce3.io.Raster(
        str(output_dir / "los_east.tif"), radar_grid.width, radar_grid.length, 1, gdal.GDT_Float32, "GTiff"
    )
    los_n_raster = isce3.io.Raster(
        str(output_dir / "los_north.tif"), radar_grid.width, radar_grid.length, 1, gdal.GDT_Float32, "GTiff"
    )

    topo.topo(
        dem_raster,
        x_raster,
        y_raster,
        z_raster,
        inc_raster,
        hdg_raster,
        local_inc_raster,
        local_psi_raster,
        simamp_raster,
        layover_raster,
        los_e_raster,
        los_n_raster,
    )
    for raster in (
        x_raster,
        y_raster,
        z_raster,
        inc_raster,
        hdg_raster,
        local_inc_raster,
        local_psi_raster,
        simamp_raster,
        layover_raster,
        los_e_raster,
        los_n_raster,
    ):
        raster.close_dataset()
    dem_raster.close_dataset()
    return _build_topo_vrt(output_dir, epsg=int(topo.epsg_out))


def run_geo2rdr_stage(
    context: PairContext,
    *,
    gpu_mode: str,
    gpu_id: int,
    block_rows: int,
) -> tuple[dict, str, str | None]:
    cached_outputs = _load_cached_stage_outputs(
        context.pair_dir,
        "p0",
        required_keys=("master_topo", "slave_topo"),
    )
    if cached_outputs is not None:
        cached_outputs.setdefault("master_topo_vrt", cached_outputs.get("master_topo"))
        cached_outputs.setdefault("slave_topo_vrt", cached_outputs.get("slave_topo"))
        return cached_outputs, "cache", None

    p0_dir = stage_dir(context.pair_dir, "p0")

    def _gpu():
        master_topo = _run_rdr2geo_topo(
            orbit_data=context.master_orbit_data,
            acquisition_data=context.master_acq_data,
            radargrid_data=context.master_rg_data,
            dem_path=context.resolved_dem,
            orbit_interp=context.orbit_interp,
            use_gpu=True,
            gpu_id=gpu_id,
            output_dir=p0_dir / "master_topo",
            block_rows=block_rows,
        )
        slave_topo = _run_rdr2geo_topo(
            orbit_data=context.slave_orbit_data,
            acquisition_data=context.slave_acq_data,
            radargrid_data=context.slave_rg_data,
            dem_path=context.resolved_dem,
            orbit_interp=context.orbit_interp,
            use_gpu=True,
            gpu_id=gpu_id,
            output_dir=p0_dir / "slave_topo",
            block_rows=block_rows,
        )
        return {"master_topo": master_topo, "slave_topo": slave_topo}

    def _cpu():
        master_topo = _run_rdr2geo_topo(
            orbit_data=context.master_orbit_data,
            acquisition_data=context.master_acq_data,
            radargrid_data=context.master_rg_data,
            dem_path=context.resolved_dem,
            orbit_interp=context.orbit_interp,
            use_gpu=False,
            gpu_id=gpu_id,
            output_dir=p0_dir / "master_topo",
            block_rows=block_rows,
        )
        slave_topo = _run_rdr2geo_topo(
            orbit_data=context.slave_orbit_data,
            acquisition_data=context.slave_acq_data,
            radargrid_data=context.slave_rg_data,
            dem_path=context.resolved_dem,
            orbit_interp=context.orbit_interp,
            use_gpu=False,
            gpu_id=gpu_id,
            output_dir=p0_dir / "slave_topo",
            block_rows=block_rows,
        )
        return {"master_topo": master_topo, "slave_topo": slave_topo}

    output_files, backend_used, fallback_reason = run_stage_with_fallback(
        stage_name="rdr2geo",
        gpu_mode=gpu_mode,
        gpu_id=gpu_id,
        gpu_runner=_gpu,
        cpu_runner=_cpu,
    )
    output_files["master_topo_vrt"] = output_files["master_topo"]
    output_files["slave_topo_vrt"] = output_files["slave_topo"]
    _write_stage_outputs_record(
        output_dir=context.pair_dir,
        stage="p0",
        master_manifest_path=context.master_manifest_path,
        slave_manifest_path=context.slave_manifest_path,
        backend_used=backend_used,
        output_files=output_files,
        fallback_reason=fallback_reason,
    )
    return output_files, backend_used, fallback_reason


def _convert_geo2rdr_output_to_gtiff(
    *,
    source_path: Path,
    rows: int,
    cols: int,
    output_path: Path,
) -> str:
    ds = gdal.Open(str(source_path), gdal.GA_ReadOnly)
    if ds is not None:
        try:
            band = ds.GetRasterBand(1)
            arr = _read_band_array(band, dtype=np.float64).astype(np.float64)
        finally:
            ds = None
    else:
        arr = np.memmap(str(source_path), dtype=np.float64, mode="r", shape=(rows, cols))
        arr = np.asarray(arr, dtype=np.float64)
    return _write_offset_raster(output_path, arr.astype(np.float32))


def _run_slave_geo2rdr_from_master_topo(
    *,
    master_topo_vrt_path: str,
    slave_orbit_data: dict,
    slave_acq_data: dict,
    slave_rg_data: dict,
    slave_dop_data: dict | None,
    output_dir: Path,
    use_gpu: bool,
    gpu_id: int,
    block_rows: int = 256,
) -> tuple[str, str]:
    import isce3.io

    slave_orbit = construct_orbit(
        slave_orbit_data,
        choose_orbit_interp(slave_orbit_data, slave_acq_data),
    )
    slave_grid = construct_radar_grid(slave_rg_data, slave_acq_data, slave_orbit_data)
    doppler = _build_coregistration_doppler_lut()

    if use_gpu:
        import isce3.cuda.core
        import isce3.cuda.geometry

        device = isce3.cuda.core.Device(gpu_id)
        isce3.cuda.core.set_device(device)
        geo2rdr_cls = isce3.cuda.geometry.Geo2Rdr
    else:
        import isce3.geometry

        geo2rdr_cls = isce3.geometry.Geo2Rdr

    topo_raster = isce3.io.Raster(str(master_topo_vrt_path))
    artifacts_dir = output_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="strip_insar2_geo2rdr_", dir=str(artifacts_dir)) as tmpdir:
        geo2rdr = geo2rdr_cls(
            slave_grid,
            slave_orbit,
            isce3.core.Ellipsoid(),
            doppler,
            threshold=1.0e-8,
            numiter=50,
            lines_per_block=block_rows,
        )
        geo2rdr.geo2rdr(topo_raster, tmpdir)
        tmpdir_path = Path(tmpdir)
        rows = int(getattr(topo_raster, "length", slave_grid.length))
        cols = int(getattr(topo_raster, "width", slave_grid.width))
        range_path = _convert_geo2rdr_output_to_gtiff(
            source_path=tmpdir_path / "range.off",
            rows=rows,
            cols=cols,
            output_path=artifacts_dir / "coarse_geo2rdr_range.off",
        )
        azimuth_path = _convert_geo2rdr_output_to_gtiff(
            source_path=tmpdir_path / "azimuth.off",
            rows=rows,
            cols=cols,
            output_path=artifacts_dir / "coarse_geo2rdr_azimuth.off",
        )

    model = {
        "method": "nisar-style-geo2rdr-from-master-topo",
        "sign_convention": "slave_minus_master_pixel_index",
        "topo_raster": master_topo_vrt_path,
        "use_gpu": bool(use_gpu),
    }
    (artifacts_dir / "coarse_geo2rdr_model.json").write_text(
        json.dumps(model, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return range_path, azimuth_path


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
    from isce3.core import LUT2d
    from isce3.image.v2.resample_slc import resample_slc_blocks

    try:
        input_slc = _open_slc_as_complex(slave_slc_path)
        out_shape = (int(ref_radar_grid.length), int(ref_radar_grid.width))
        output_slc = np.full(out_shape, np.complex64(np.nan + 1j * np.nan), dtype=np.complex64)
        if rg_offset_path and Path(rg_offset_path).is_file():
            rg_ds = gdal.Open(str(rg_offset_path), gdal.GA_ReadOnly)
            rg_offsets = _read_band_array(rg_ds.GetRasterBand(1), dtype=np.float64).astype(np.float64)
            rg_ds = None
        else:
            rg_offsets = np.zeros(out_shape, dtype=np.float64)
        if az_offset_path and Path(az_offset_path).is_file():
            az_ds = gdal.Open(str(az_offset_path), gdal.GA_ReadOnly)
            az_offsets = _read_band_array(az_ds.GetRasterBand(1), dtype=np.float64).astype(np.float64)
            az_ds = None
        else:
            az_offsets = np.zeros(out_shape, dtype=np.float64)
        az_invalid = (~np.isfinite(az_offsets)) | (az_offsets == NISAR_OFFSET_INVALID_VALUE)
        az_invalid |= (az_offsets == GEO2RDR_OFFSET_NODATA) | (az_offsets <= GEO2RDR_OFFSET_INVALID_LOW)
        rg_invalid = (~np.isfinite(rg_offsets)) | (rg_offsets == NISAR_OFFSET_INVALID_VALUE)
        rg_invalid |= (rg_offsets == GEO2RDR_OFFSET_NODATA) | (rg_offsets <= GEO2RDR_OFFSET_INVALID_LOW)
        az_offsets[az_invalid] = np.nan
        rg_offsets[rg_invalid] = np.nan
        if block_size_rg == 0:
            block_size_rg = out_shape[1]
        resample_slc_blocks(
            output_resampled_slcs=[output_slc],
            input_slcs=[input_slc],
            az_offsets_dataset=az_offsets,
            rg_offsets_dataset=rg_offsets,
            input_radar_grid=radar_grid,
            doppler=doppler if doppler is not None else LUT2d(),
            block_size_az=block_size_az,
            block_size_rg=block_size_rg,
            quiet=True,
            with_gpu=use_gpu,
        )
        _write_complex_gtiff(Path(coarse_coreg_slave_path), output_slc)
        return True
    except Exception:
        return False


def _prepare_nisar_geo2rdr_offsets(
    *,
    coarse_rg_offset_path: str,
    coarse_az_offset_path: str,
    output_dir: Path,
) -> tuple[str, str]:
    geo2rdr_dir = output_dir / "geo2rdr" / "freqA"
    geo2rdr_dir.mkdir(parents=True, exist_ok=True)
    range_arr = _normalize_offset_array(_read_raster_array(coarse_rg_offset_path, dtype=np.float64))
    azimuth_arr = _normalize_offset_array(_read_raster_array(coarse_az_offset_path, dtype=np.float64))
    range_envi = _write_envi_single_band(geo2rdr_dir / "range.off", range_arr, dtype=gdal.GDT_Float64)
    azimuth_envi = _write_envi_single_band(geo2rdr_dir / "azimuth.off", azimuth_arr, dtype=gdal.GDT_Float64)
    return range_envi, azimuth_envi


def _run_nisar_dense_offsets(
    *,
    reference_slc_path: str,
    secondary_slc_path: str,
    output_dir: Path,
    use_gpu: bool,
    gpu_id: int,
) -> dict[str, str]:
    if use_gpu:
        try:
            return _run_nisar_dense_offsets_gpu_subprocess(
                reference_slc_path=reference_slc_path,
                secondary_slc_path=secondary_slc_path,
                output_dir=output_dir,
                gpu_id=gpu_id,
            )
        except Exception:
            return _run_nisar_dense_offsets_impl(
                reference_slc_path=reference_slc_path,
                secondary_slc_path=secondary_slc_path,
                output_dir=output_dir,
                use_gpu=False,
                gpu_id=gpu_id,
            )
    return _run_nisar_dense_offsets_impl(
        reference_slc_path=reference_slc_path,
        secondary_slc_path=secondary_slc_path,
        output_dir=output_dir,
        use_gpu=False,
        gpu_id=gpu_id,
    )


def _run_nisar_dense_offsets_gpu_subprocess(
    *,
    reference_slc_path: str,
    secondary_slc_path: str,
    output_dir: Path,
    gpu_id: int,
) -> dict[str, str]:
    marker = "__STRIP_INSAR2_DENSE_OFFSETS__="
    scripts_dir = Path(__file__).resolve().parent
    code = (
        "import json, sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "import strip_insar2\n"
        "result = strip_insar2._run_nisar_dense_offsets_impl(\n"
        "    reference_slc_path=sys.argv[2],\n"
        "    secondary_slc_path=sys.argv[3],\n"
        "    output_dir=Path(sys.argv[4]),\n"
        "    use_gpu=True,\n"
        "    gpu_id=int(sys.argv[5]),\n"
        ")\n"
        f"print('{marker}' + json.dumps(result, ensure_ascii=False))\n"
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(scripts_dir),
            str(reference_slc_path),
            str(secondary_slc_path),
            str(output_dir),
            str(gpu_id),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    if completed.returncode != 0:
        detail = (stdout + "\n" + stderr).strip()
        if len(detail) > 800:
            detail = detail[-800:]
        raise RuntimeError(
            f"dense_offsets GPU subprocess failed with exit code {completed.returncode}: {detail}"
        )
    for line in reversed(stdout.splitlines()):
        if line.startswith(marker):
            return json.loads(line[len(marker) :])
    raise RuntimeError("dense_offsets GPU subprocess returned no result marker")


def _run_nisar_dense_offsets_impl(
    *,
    reference_slc_path: str,
    secondary_slc_path: str,
    output_dir: Path,
    use_gpu: bool,
    gpu_id: int,
) -> dict[str, str]:
    modules = _load_nisar_registration_modules()
    import isce3
    import isce3.io
    import isce3.matchtemplate

    if use_gpu:
        import isce3.cuda.core
        import isce3.cuda.matchtemplate

    reference_envi = Path(output_dir / "reference.slc")
    secondary_envi = Path(output_dir / "secondary.slc")
    _translate_raster(reference_slc_path, reference_envi, driver="ENVI", output_type=gdal.GDT_CFloat32)
    _translate_raster(secondary_slc_path, secondary_envi, driver="ENVI", output_type=gdal.GDT_CFloat32)

    reference_raster = isce3.io.Raster(str(reference_envi))
    secondary_raster = isce3.io.Raster(str(secondary_envi))
    if use_gpu:
        device = isce3.cuda.core.Device(gpu_id)
        isce3.cuda.core.set_device(device)
        ampcor = isce3.cuda.matchtemplate.PyCuAmpcor()
        ampcor.deviceID = gpu_id
    else:
        ampcor = isce3.matchtemplate.PyCPUAmpcor()

    ampcor.useMmap = 1
    ampcor.referenceImageName = str(reference_envi)
    ampcor.referenceImageHeight = reference_raster.length
    ampcor.referenceImageWidth = reference_raster.width
    ampcor.secondaryImageName = str(secondary_envi)
    ampcor.secondaryImageHeight = secondary_raster.length
    ampcor.secondaryImageWidth = secondary_raster.width
    ampcor = modules["set_optional_attributes"](
        ampcor,
        _default_dense_offsets_cfg(),
        reference_raster.length,
        reference_raster.width,
    )

    dense_offsets_path = output_dir / "dense_offsets"
    gross_offsets_path = output_dir / "gross_offsets"
    snr_path = output_dir / "snr"
    covariance_path = output_dir / "covariance"
    correlation_peak_path = output_dir / "correlation_peak"
    ampcor.offsetImageName = str(dense_offsets_path)
    ampcor.grossOffsetImageName = str(gross_offsets_path)
    ampcor.snrImageName = str(snr_path)
    ampcor.covImageName = str(covariance_path)
    ampcor.corrImageName = str(correlation_peak_path)

    modules["create_empty_dataset"](
        str(dense_offsets_path),
        ampcor.numberWindowAcross,
        ampcor.numberWindowDown,
        2,
        gdal.GDT_Float32,
    )
    modules["create_empty_dataset"](
        str(gross_offsets_path),
        ampcor.numberWindowAcross,
        ampcor.numberWindowDown,
        2,
        gdal.GDT_Float32,
    )
    modules["create_empty_dataset"](
        str(snr_path),
        ampcor.numberWindowAcross,
        ampcor.numberWindowDown,
        1,
        gdal.GDT_Float32,
    )
    modules["create_empty_dataset"](
        str(covariance_path),
        ampcor.numberWindowAcross,
        ampcor.numberWindowDown,
        3,
        gdal.GDT_Float32,
    )
    modules["create_empty_dataset"](
        str(correlation_peak_path),
        ampcor.numberWindowAcross,
        ampcor.numberWindowDown,
        1,
        gdal.GDT_Float32,
    )

    ampcor.runAmpcor()
    return {
        "reference_slc": str(reference_envi),
        "secondary_slc": str(secondary_envi),
        "dense_offsets": str(dense_offsets_path),
        "gross_offsets": str(gross_offsets_path),
        "snr": str(snr_path),
        "covariance": str(covariance_path),
        "correlation_peak": str(correlation_peak_path),
    }


def _run_nisar_rubbersheet(
    *,
    dense_offsets_dir: Path,
    coarse_range_envi: str,
    coarse_azimuth_envi: str,
    output_dir: Path,
    ref_length: int,
    ref_width: int,
) -> dict[str, str]:
    modules = _load_nisar_registration_modules()
    rubbersheet_cfg = _default_rubbersheet_cfg()
    output_dir.mkdir(parents=True, exist_ok=True)

    culled_azimuth, culled_range = modules["identify_outliers"](str(dense_offsets_dir), rubbersheet_cfg)
    azimuth_filled = modules["fill_outliers_holes"](culled_azimuth, rubbersheet_cfg)
    range_filled = modules["fill_outliers_holes"](culled_range, rubbersheet_cfg)

    processed_offsets = {}
    for key, data in (("azimuth", azimuth_filled), ("range", range_filled)):
        if np.isnan(data).any():
            data = modules["_interpolate_offsets"](data, rubbersheet_cfg["interpolation_method"])
        data = modules["_filter_offsets"](data, rubbersheet_cfg)
        processed_offsets[key] = np.asarray(data, dtype=np.float64)

    culled_azimuth_path = _write_envi_single_band(
        output_dir / "culled_az_offsets",
        processed_offsets["azimuth"],
        dtype=gdal.GDT_Float64,
    )
    culled_range_path = _write_envi_single_band(
        output_dir / "culled_rg_offsets",
        processed_offsets["range"],
        dtype=gdal.GDT_Float64,
    )

    resampled_azimuth_path = output_dir / "resampled_az_offsets"
    resampled_range_path = output_dir / "resampled_rg_offsets"
    _translate_raster(
        culled_azimuth_path,
        resampled_azimuth_path,
        driver="ENVI",
        width=ref_width,
        height=ref_length,
        resample_alg="bilinear",
    )
    _translate_raster(
        culled_range_path,
        resampled_range_path,
        driver="ENVI",
        width=ref_width,
        height=ref_length,
        resample_alg="bilinear",
    )

    final_azimuth_path = output_dir / "azimuth.off"
    final_range_path = output_dir / "range.off"
    modules["sum_gdal_rasters"](
        coarse_azimuth_envi,
        str(resampled_azimuth_path),
        str(final_azimuth_path),
        invalid_value=NISAR_OFFSET_INVALID_VALUE,
    )
    modules["sum_gdal_rasters"](
        coarse_range_envi,
        str(resampled_range_path),
        str(final_range_path),
        invalid_value=NISAR_OFFSET_INVALID_VALUE,
    )

    return {
        "culled_azimuth_offsets": culled_azimuth_path,
        "culled_range_offsets": culled_range_path,
        "resampled_azimuth_offsets": str(resampled_azimuth_path),
        "resampled_range_offsets": str(resampled_range_path),
        "azimuth_offsets": str(final_azimuth_path),
        "range_offsets": str(final_range_path),
    }


def _run_nisar_registration_chain(
    *,
    context: PairContext,
    use_gpu: bool,
    gpu_id: int,
    p1_stage_path: Path,
) -> dict:
    slave_slc = resolve_manifest_data_path(
        context.slave_manifest_path,
        context.slave_manifest["slc"]["path"],
    )
    master_slc = resolve_manifest_data_path(
        context.master_manifest_path,
        context.master_manifest["slc"]["path"],
    )
    if not slave_slc or not master_slc:
        raise FileNotFoundError("master/slave SLC path missing in manifest")

    ref_radar_grid = construct_radar_grid(
        context.master_rg_data,
        context.master_acq_data,
        context.master_orbit_data,
    )
    slave_radar_grid = construct_radar_grid(
        context.slave_rg_data,
        context.slave_acq_data,
        context.slave_orbit_data,
    )
    slave_doppler = _build_coregistration_doppler_lut()

    p0_record = load_stage_record(context.pair_dir, "p0") or {}
    p0_outputs = p0_record.get("output_files", {})
    master_topo = p0_outputs.get("master_topo_vrt") or p0_outputs.get("master_topo")
    if not master_topo or not Path(str(master_topo)).exists():
        raise RuntimeError("p0 master_topo output is required before running p1")

    coarse_rg_offset_path, coarse_az_offset_path = _run_slave_geo2rdr_from_master_topo(
        master_topo_vrt_path=str(master_topo),
        slave_orbit_data=context.slave_orbit_data,
        slave_acq_data=context.slave_acq_data,
        slave_rg_data=context.slave_rg_data,
        slave_dop_data=context.slave_dop_data,
        output_dir=p1_stage_path,
        use_gpu=use_gpu,
        gpu_id=gpu_id,
    )
    coarse_range_envi, coarse_azimuth_envi = _prepare_nisar_geo2rdr_offsets(
        coarse_rg_offset_path=coarse_rg_offset_path,
        coarse_az_offset_path=coarse_az_offset_path,
        output_dir=p1_stage_path,
    )

    coarse_coreg_slave_path = str(p1_stage_path / "coarse_coreg_slave.tif")
    coarse_ok = run_coarse_resamp_isce3_v2(
        slave_slc_path=slave_slc,
        coarse_coreg_slave_path=coarse_coreg_slave_path,
        radar_grid=slave_radar_grid,
        doppler=slave_doppler,
        ref_radar_grid=ref_radar_grid,
        rg_offset_path=coarse_rg_offset_path,
        az_offset_path=coarse_az_offset_path,
        use_gpu=use_gpu,
    )
    if not coarse_ok:
        raise RuntimeError("NISAR coarse resample failed")

    dense_offsets_dir = p1_stage_path / "dense_offsets" / "freqA" / "HH"
    dense_outputs = _run_nisar_dense_offsets(
        reference_slc_path=master_slc,
        secondary_slc_path=coarse_coreg_slave_path,
        output_dir=dense_offsets_dir,
        use_gpu=use_gpu,
        gpu_id=gpu_id,
    )

    rubbersheet_dir = p1_stage_path / "rubbersheet_offsets" / "freqA" / "HH"
    rubbersheet_outputs = _run_nisar_rubbersheet(
        dense_offsets_dir=dense_offsets_dir,
        coarse_range_envi=coarse_range_envi,
        coarse_azimuth_envi=coarse_azimuth_envi,
        output_dir=rubbersheet_dir,
        ref_length=int(ref_radar_grid.length),
        ref_width=int(ref_radar_grid.width),
    )

    fine_coreg_slave_path = str(p1_stage_path / "fine_coreg_slave.tif")
    fine_ok = run_coarse_resamp_isce3_v2(
        slave_slc_path=slave_slc,
        coarse_coreg_slave_path=fine_coreg_slave_path,
        radar_grid=slave_radar_grid,
        doppler=slave_doppler,
        ref_radar_grid=ref_radar_grid,
        rg_offset_path=rubbersheet_outputs["range_offsets"],
        az_offset_path=rubbersheet_outputs["azimuth_offsets"],
        use_gpu=use_gpu,
    )
    if not fine_ok:
        raise RuntimeError("NISAR fine resample failed")

    range_offset_gtiff = _write_offset_raster(
        p1_stage_path / "range.off.tif",
        _read_raster_array(rubbersheet_outputs["range_offsets"], dtype=np.float64),
    )
    azimuth_offset_gtiff = _write_offset_raster(
        p1_stage_path / "azimuth.off.tif",
        _read_raster_array(rubbersheet_outputs["azimuth_offsets"], dtype=np.float64),
    )
    range_residual_gtiff = _write_offset_raster(
        p1_stage_path / "range_residual.off.tif",
        _read_raster_array(rubbersheet_outputs["resampled_range_offsets"], dtype=np.float64),
    )
    azimuth_residual_gtiff = _write_offset_raster(
        p1_stage_path / "azimuth_residual.off.tif",
        _read_raster_array(rubbersheet_outputs["resampled_azimuth_offsets"], dtype=np.float64),
    )

    registration_model = p1_stage_path / "registration_model.json"
    registration_model.write_text(
        json.dumps(
            {
                "source": "nisar-strip-registration-chain",
                "use_gpu": bool(use_gpu),
                "sequence": [
                    "geo2rdr",
                    "coarse_resample",
                    "dense_offsets",
                    "rubbersheet",
                    "fine_resample",
                ],
                "coarse_geo2rdr_range_offsets": coarse_rg_offset_path,
                "coarse_geo2rdr_azimuth_offsets": coarse_az_offset_path,
                "dense_offsets_dir": str(dense_offsets_dir),
                "rubbersheet_dir": str(rubbersheet_dir),
                "range_offsets": range_offset_gtiff,
                "azimuth_offsets": azimuth_offset_gtiff,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    outputs = {
        "coarse_coreg_slave": coarse_coreg_slave_path,
        "fine_coreg_slave": fine_coreg_slave_path,
        "registration_model": str(registration_model),
        "range_offsets": range_offset_gtiff,
        "azimuth_offsets": azimuth_offset_gtiff,
        "range_residual_offsets": range_residual_gtiff,
        "azimuth_residual_offsets": azimuth_residual_gtiff,
        "dense_offsets_dir": str(dense_offsets_dir),
        "rubbersheet_dir": str(rubbersheet_dir),
        "coarse_geo2rdr_range_offsets": coarse_rg_offset_path,
        "coarse_geo2rdr_azimuth_offsets": coarse_az_offset_path,
        "coarse_geo2rdr_range_offsets_envi": coarse_range_envi,
        "coarse_geo2rdr_azimuth_offsets_envi": coarse_azimuth_envi,
        **dense_outputs,
        **rubbersheet_outputs,
    }

    try:
        outputs["coarse_coreg_slave_png"] = _write_radar_amplitude_png(
            coarse_coreg_slave_path,
            context.pair_dir / "slave_coarse_coreg_fullres.png",
        )
    except Exception:
        pass
    try:
        outputs["fine_coreg_slave_png"] = _write_radar_amplitude_png(
            fine_coreg_slave_path,
            context.pair_dir / "slave_fine_coreg_fullres.png",
        )
    except Exception:
        pass

    offsets_path = p1_stage_path / "offsets.json"
    offsets_path.write_text(
        json.dumps(
            {
                "row_offset": azimuth_offset_gtiff,
                "col_offset": range_offset_gtiff,
                "row_residual_offset": azimuth_residual_gtiff,
                "col_residual_offset": range_residual_gtiff,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    outputs["offsets"] = str(offsets_path)

    coarse_model_path = p1_stage_path / "coarse_geo2rdr_model.json"
    if coarse_model_path.is_file():
        outputs["coarse_geo2rdr_model"] = str(coarse_model_path)
    return outputs


def run_resample_stage(
    context: PairContext,
    *,
    gpu_mode: str,
    gpu_id: int,
) -> tuple[dict, str, str | None]:
    cached_outputs = _load_cached_stage_outputs(
        context.pair_dir,
        "p1",
        required_keys=("fine_coreg_slave", "range_offsets", "azimuth_offsets"),
    )
    if cached_outputs is not None:
        return cached_outputs, "cache", None

    p1_stage_path = stage_dir(context.pair_dir, "p1")
    p1_stage_path.mkdir(parents=True, exist_ok=True)

    result, backend_used, fallback_reason = run_stage_with_fallback(
        stage_name="resample",
        gpu_mode=gpu_mode,
        gpu_id=gpu_id,
        gpu_runner=lambda: _run_nisar_registration_chain(
            context=context,
            use_gpu=True,
            gpu_id=gpu_id,
            p1_stage_path=p1_stage_path,
        ),
        cpu_runner=lambda: _run_nisar_registration_chain(
            context=context,
            use_gpu=False,
            gpu_id=gpu_id,
            p1_stage_path=p1_stage_path,
        ),
    )
    _write_stage_outputs_record(
        output_dir=context.pair_dir,
        stage="p1",
        master_manifest_path=context.master_manifest_path,
        slave_manifest_path=context.slave_manifest_path,
        backend_used=backend_used,
        output_files=result,
        fallback_reason=fallback_reason,
    )
    return result, backend_used, fallback_reason


def _select_registered_slave_slc(
    p1_outputs: dict | None,
    fallback_path: str,
    *,
    require_existing: bool = False,
) -> str:
    if not isinstance(p1_outputs, dict):
        return fallback_path
    for key in ("fine_coreg_slave", "coarse_coreg_slave"):
        candidate = p1_outputs.get(key)
        if not candidate:
            continue
        if require_existing and not Path(str(candidate)).is_file():
            continue
        return str(candidate)
    return fallback_path


def _prepare_crossmul_flatten_raster(
    *,
    output_dir: Path,
    range_offset_path: str | Path | None,
    radargrid_data: dict,
) -> tuple[str | None, str | None, str | None]:
    if not range_offset_path:
        return None, None, None
    source_path = Path(str(range_offset_path))
    if not source_path.is_file():
        return None, None, None

    rows = int(radargrid_data.get("numberOfRows", 0) or 0)
    cols = int(radargrid_data.get("numberOfColumns", 0) or 0)
    if rows <= 0 or cols <= 0:
        return None, None, None

    ds = gdal.Open(str(source_path), gdal.GA_ReadOnly)
    if ds is None:
        return None, None, None
    try:
        arr = _read_band_array(ds.GetRasterBand(1), dtype=np.float64).astype(np.float64)
    finally:
        ds = None
    if arr.shape != (rows, cols):
        return None, None, None

    valid = np.isfinite(arr)
    valid &= arr != GEO2RDR_OFFSET_NODATA
    valid &= arr >= GEO2RDR_OFFSET_INVALID_LOW
    model = str(os.environ.get("D2SAR_FLATTEN_OFFSET_MODEL", "full")).strip().lower()
    if model not in {"full", "plane"}:
        model = "full"

    clean = np.zeros_like(arr, dtype=np.float64)
    clean[valid] = arr[valid]
    if model == "plane" and np.any(valid):
        yy, xx = np.mgrid[0:rows, 0:cols]
        sample = np.column_stack(
            [
                np.ones(int(np.count_nonzero(valid)), dtype=np.float64),
                yy[valid].astype(np.float64),
                xx[valid].astype(np.float64),
            ]
        )
        coeffs, *_ = np.linalg.lstsq(sample, arr[valid].astype(np.float64), rcond=None)
        clean = coeffs[0] + coeffs[1] * yy + coeffs[2] * xx
        clean[~valid] = 0.0

    p2_dir = stage_dir(output_dir, "p2")
    p2_dir.mkdir(parents=True, exist_ok=True)
    flatten_path = _write_float_gtiff(
        p2_dir / "range_flatten.off.tif",
        clean.astype(np.float32),
        nodata=0.0,
    )
    mask_path = _write_float_gtiff(
        p2_dir / "range_flatten_valid_mask.tif",
        valid.astype(np.uint8),
        dtype=gdal.GDT_Byte,
        nodata=0,
    )
    return flatten_path, mask_path, model


def _build_crossmul_flatten_options(
    *,
    output_dir: Path,
    p1_outputs: dict | None,
    registered_slave_slc: str | None,
    radargrid_data: dict,
    slave_radargrid_data: dict | None,
    acquisition_data: dict,
) -> dict:
    flatten_source_mode = str(os.environ.get("D2SAR_FLATTEN_SOURCE", "final")).strip().lower()
    if flatten_source_mode not in {"coarse", "final", "auto", "none"}:
        flatten_source_mode = "final"

    selected_source = None
    selected_range_offset = None
    coarse_range_offset = None
    final_range_offset = None
    final_consistent = False

    if isinstance(p1_outputs, dict):
        coarse_candidate = p1_outputs.get("coarse_geo2rdr_range_offsets")
        final_candidate = p1_outputs.get("range_offsets")
        fine_coreg_slave = p1_outputs.get("fine_coreg_slave")
        if coarse_candidate and Path(str(coarse_candidate)).is_file():
            coarse_range_offset = str(coarse_candidate)
        if final_candidate and Path(str(final_candidate)).is_file():
            final_range_offset = str(final_candidate)
        if (
            final_range_offset is not None
            and fine_coreg_slave
            and registered_slave_slc
            and Path(str(fine_coreg_slave)).is_file()
        ):
            try:
                final_consistent = Path(str(fine_coreg_slave)).resolve() == Path(str(registered_slave_slc)).resolve()
            except Exception:
                final_consistent = str(fine_coreg_slave) == str(registered_slave_slc)

    if flatten_source_mode == "coarse":
        if coarse_range_offset:
            selected_source = "coarse_geo2rdr_range_offsets"
            selected_range_offset = coarse_range_offset
    elif flatten_source_mode == "final":
        if final_range_offset and final_consistent:
            selected_source = "p1_final_range_offsets"
            selected_range_offset = final_range_offset
        elif coarse_range_offset:
            selected_source = "coarse_geo2rdr_range_offsets"
            selected_range_offset = coarse_range_offset
    elif flatten_source_mode == "auto":
        if final_range_offset and final_consistent:
            selected_source = "p1_final_range_offsets"
            selected_range_offset = final_range_offset
        elif coarse_range_offset:
            selected_source = "coarse_geo2rdr_range_offsets"
            selected_range_offset = coarse_range_offset

    flatten_raster, flatten_mask_raster, flatten_model = _prepare_crossmul_flatten_raster(
        output_dir=output_dir,
        range_offset_path=selected_range_offset,
        radargrid_data=radargrid_data,
    )
    return {
        "flatten_raster": flatten_raster,
        "flatten_mask_raster": flatten_mask_raster,
        "range_pixel_spacing": float(radargrid_data.get("columnSpacing", 0.0) or 0.0),
        "wavelength": get_wavelength(acquisition_data),
        "flatten_source": selected_source,
        "flatten_model": flatten_model,
        "flatten_source_mode": flatten_source_mode,
        "flatten_starting_range_shift_m": _compute_ref_sec_starting_range_shift_m(
            radargrid_data,
            slave_radargrid_data,
        ),
    }


def _add_flatten_outputs(output_files: dict, flatten_options: dict) -> None:
    flatten_raster = flatten_options.get("flatten_raster")
    if not flatten_raster:
        return
    output_files["flatten_range_offsets"] = flatten_raster
    output_files["flatten_range_offsets_mask"] = flatten_options.get("flatten_mask_raster")
    output_files["flatten_method"] = "nisar-style-explicit-rangeoff-flatten"
    output_files["flatten_offset_model"] = flatten_options.get("flatten_model")
    output_files["flatten_offset_source_mode"] = flatten_options.get("flatten_source_mode")
    output_files["flatten_ref_sec_starting_range_shift_m"] = flatten_options.get(
        "flatten_starting_range_shift_m"
    )


def _estimate_coherence(
    master_slc: np.ndarray,
    slave_slc: np.ndarray,
    *,
    size: int = 5,
) -> np.ndarray:
    try:
        from scipy.ndimage import uniform_filter
    except Exception:
        uniform_filter = None

    if uniform_filter is None:
        coh = np.ones(master_slc.shape, dtype=np.float32)
        coh[~(np.isfinite(master_slc) & np.isfinite(slave_slc))] = 0.0
        return coh

    ifg = master_slc * np.conj(slave_slc)
    num_real = uniform_filter(ifg.real.astype(np.float32), size=size, mode="nearest")
    num_imag = uniform_filter(ifg.imag.astype(np.float32), size=size, mode="nearest")
    num = np.sqrt(num_real**2 + num_imag**2)
    pwr_m = uniform_filter(np.abs(master_slc).astype(np.float32) ** 2, size=size, mode="nearest")
    pwr_s = uniform_filter(np.abs(slave_slc).astype(np.float32) ** 2, size=size, mode="nearest")
    den = np.sqrt(np.maximum(pwr_m * pwr_s, 1.0e-9))
    coh = np.clip(num / den, 0.0, 1.0)
    coh[~np.isfinite(coh)] = 0.0
    return coh.astype(np.float32)


def _run_crossmul_cpu(
    *,
    master_slc_path: str,
    slave_slc_path: str,
    flatten_raster: str | None,
    flatten_mask_raster: str | None,
    range_pixel_spacing: float | None,
    wavelength: float | None,
    flatten_starting_range_shift_m: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    master_slc = _open_slc_as_complex(master_slc_path)
    slave_slc = _open_slc_as_complex(slave_slc_path)
    if master_slc.shape != slave_slc.shape:
        raise RuntimeError(
            f"shape mismatch for crossmul: master={master_slc.shape}, slave={slave_slc.shape}"
        )
    interferogram = master_slc * np.conj(slave_slc)

    if flatten_raster and wavelength and range_pixel_spacing is not None:
        ds = gdal.Open(str(flatten_raster), gdal.GA_ReadOnly)
        if ds is None:
            raise RuntimeError(f"failed to open flatten raster: {flatten_raster}")
        try:
            range_offset = _read_band_array(ds.GetRasterBand(1), dtype=np.float32).astype(np.float32)
        finally:
            ds = None
        mask = np.ones(range_offset.shape, dtype=bool)
        if flatten_mask_raster:
            mask_ds = gdal.Open(str(flatten_mask_raster), gdal.GA_ReadOnly)
            if mask_ds is not None:
                try:
                    mask = _read_band_array(mask_ds.GetRasterBand(1), dtype=np.uint8).astype(np.uint8) != 0
                finally:
                    mask_ds = None
        phase = (
            4.0
            * np.pi
            * (
                float(range_pixel_spacing) * range_offset.astype(np.float32)
                + float(flatten_starting_range_shift_m or 0.0)
            )
            / float(wavelength)
        )
        flatten_term = np.exp(-1j * phase.astype(np.float32)).astype(np.complex64)
        interferogram = interferogram.astype(np.complex64, copy=False)
        interferogram[mask] *= flatten_term[mask]

    coherence = _estimate_coherence(master_slc, slave_slc)
    return interferogram.astype(np.complex64), coherence


def run_crossmul_stage(
    context: PairContext,
    *,
    gpu_mode: str,
    gpu_id: int,
    block_rows: int,
) -> tuple[dict, str, str | None]:
    cached_outputs = _load_cached_stage_outputs(
        context.pair_dir,
        "p2",
        required_keys=("interferogram", "coherence"),
    )
    if cached_outputs is not None:
        return cached_outputs, "cache", None

    def _run_impl(use_gpu: bool) -> dict:
        master_slc = resolve_manifest_data_path(
            context.master_manifest_path,
            context.master_manifest["slc"]["path"],
        )
        slave_slc = resolve_manifest_data_path(
            context.slave_manifest_path,
            context.slave_manifest["slc"]["path"],
        )
        p1_record = load_stage_record(context.pair_dir, "p1") or {}
        p1_outputs = p1_record.get("output_files")
        registered_slave_slc = _select_registered_slave_slc(
            p1_outputs,
            slave_slc,
            require_existing=True,
        )
        flatten_options = _build_crossmul_flatten_options(
            output_dir=context.pair_dir,
            p1_outputs=p1_outputs,
            registered_slave_slc=registered_slave_slc,
            radargrid_data=context.master_rg_data,
            slave_radargrid_data=context.slave_rg_data,
            acquisition_data=context.master_acq_data,
        )
        if use_gpu:
            raise RuntimeError("manifest-based NISAR-style GPU crossmul is not enabled in strip_insar2")
        interferogram, coherence = _run_crossmul_cpu(
            master_slc_path=master_slc,
            slave_slc_path=registered_slave_slc,
            flatten_raster=flatten_options.get("flatten_raster"),
            flatten_mask_raster=flatten_options.get("flatten_mask_raster"),
            range_pixel_spacing=flatten_options.get("range_pixel_spacing"),
            wavelength=flatten_options.get("wavelength"),
            flatten_starting_range_shift_m=flatten_options.get("flatten_starting_range_shift_m"),
        )
        filtered_interferogram = goldstein_filter(interferogram.astype(np.complex64))
        output_files = {
            "interferogram": _save_stage_array(context.pair_dir, "p2", "interferogram", interferogram),
            "filtered_interferogram": _save_stage_array(
                context.pair_dir, "p2", "filtered_interferogram", filtered_interferogram
            ),
            "coherence": _save_stage_array(context.pair_dir, "p2", "coherence", coherence),
            "wrapped_phase_radar_png": _write_radar_wrapped_phase_png(
                interferogram,
                context.pair_dir / "wrapped_phase_radar.png",
            ),
        }
        _add_flatten_outputs(output_files, flatten_options)
        return output_files

    result, backend_used, fallback_reason = run_stage_with_fallback(
        stage_name="crossmul",
        gpu_mode=gpu_mode,
        gpu_id=gpu_id,
        gpu_runner=lambda: _run_impl(True),
        cpu_runner=lambda: _run_impl(False),
    )
    _write_stage_outputs_record(
        output_dir=context.pair_dir,
        stage="p2",
        master_manifest_path=context.master_manifest_path,
        slave_manifest_path=context.slave_manifest_path,
        backend_used=backend_used,
        output_files=result,
        fallback_reason=fallback_reason,
    )
    return result, backend_used, fallback_reason


def _build_icu_unwrapper():
    import isce3

    unwrap = isce3.unwrap.ICU()
    unwrap.corr_incr_thr = ICU_DEFAULTS["correlation_threshold_increments"]
    unwrap.buffer_lines = ICU_DEFAULTS["buffer_lines"]
    unwrap.overlap_lines = ICU_DEFAULTS["overlap_lines"]
    unwrap.use_phase_grad_neut = ICU_DEFAULTS["use_phase_gradient_neutron"]
    unwrap.use_intensity_neut = ICU_DEFAULTS["use_intensity_neutron"]
    unwrap.phase_grad_win_size = ICU_DEFAULTS["phase_gradient_window_size"]
    unwrap.neut_phase_grad_thr = ICU_DEFAULTS["neutron_phase_gradient_threshold"]
    unwrap.neut_intensity_thr = ICU_DEFAULTS["neutron_intensity_threshold"]
    unwrap.neut_correlation_thr = ICU_DEFAULTS["max_intensity_correlation_threshold"]
    unwrap.trees_number = ICU_DEFAULTS["trees_number"]
    unwrap.max_branch_length = ICU_DEFAULTS["max_branch_length"]
    unwrap.ratio_dxdy = ICU_DEFAULTS["pixel_spacing_ratio"]
    unwrap.init_corr_thr = ICU_DEFAULTS["initial_correlation_threshold"]
    unwrap.max_corr_thr = ICU_DEFAULTS["max_correlation_threshold"]
    unwrap.min_cc_area = ICU_DEFAULTS["min_tile_area"]
    unwrap.num_bs_lines = ICU_DEFAULTS["bootstrap_lines"]
    unwrap.min_overlap_area = ICU_DEFAULTS["min_overlap_area"]
    unwrap.phase_var_thr = ICU_DEFAULTS["phase_variance_threshold"]
    return unwrap


def _unwrap_with_icu(
    interferogram: np.ndarray,
    coherence: np.ndarray,
    scratch_dir: Path,
) -> np.ndarray:
    import isce3

    scratch_dir.mkdir(parents=True, exist_ok=True)
    igram_path = Path(_write_complex_gtiff(scratch_dir / "wrapped_igram.tif", interferogram))
    coh_path = Path(_write_float_gtiff(scratch_dir / "coherence.tif", coherence.astype(np.float32)))
    rows, cols = interferogram.shape
    driver = gdal.GetDriverByName("GTiff")
    unw_ds = driver.Create(str(scratch_dir / "unwrapped_phase.tif"), cols, rows, 1, gdal.GDT_Float32)
    cc_ds = driver.Create(str(scratch_dir / "connected_components.tif"), cols, rows, 1, gdal.GDT_Byte)
    if unw_ds is None or cc_ds is None:
        raise RuntimeError("failed to create ICU scratch rasters")
    unw_ds = None
    cc_ds = None

    icu = _build_icu_unwrapper()
    icu.unwrap(
        isce3.io.Raster(str(scratch_dir / "unwrapped_phase.tif"), update=True),
        isce3.io.Raster(str(scratch_dir / "connected_components.tif"), update=True),
        isce3.io.Raster(str(igram_path)),
        isce3.io.Raster(str(coh_path)),
        seed=ICU_DEFAULTS["seed"],
    )
    ds = gdal.Open(str(scratch_dir / "unwrapped_phase.tif"), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError("ICU did not produce unwrapped phase raster")
    try:
        result = _read_band_array(ds.GetRasterBand(1), dtype=np.float32).astype(np.float32)
    finally:
        ds = None
    if not np.any(np.isfinite(result)):
        raise RuntimeError("ICU produced no finite pixels")
    return result


def _unwrap_with_snaphu(
    interferogram: np.ndarray,
    coherence: np.ndarray,
    scratch_dir: Path,
) -> np.ndarray:
    import snaphu

    scratch_dir.mkdir(parents=True, exist_ok=True)
    unw = np.zeros(interferogram.shape, dtype=np.float32)
    conncomp = np.zeros(interferogram.shape, dtype=np.uint32)
    snaphu.unwrap(
        interferogram.astype(np.complex64),
        coherence.astype(np.float32),
        SNAPHU_DEFAULTS["nlooks"],
        unw=unw,
        conncomp=conncomp,
        cost=SNAPHU_DEFAULTS["cost_mode"],
        init=SNAPHU_DEFAULTS["initialization_method"],
        min_conncomp_frac=SNAPHU_DEFAULTS["min_conncomp_frac"],
        phase_grad_window=SNAPHU_DEFAULTS["phase_grad_window"],
        ntiles=SNAPHU_DEFAULTS["ntiles"],
        tile_overlap=SNAPHU_DEFAULTS["tile_overlap"],
        nproc=SNAPHU_DEFAULTS["nproc"],
        tile_cost_thresh=SNAPHU_DEFAULTS["tile_cost_thresh"],
        min_region_size=SNAPHU_DEFAULTS["min_region_size"],
        single_tile_reoptimize=SNAPHU_DEFAULTS["single_tile_reoptimize"],
        regrow_conncomps=SNAPHU_DEFAULTS["regrow_conncomps"],
        scratchdir=scratch_dir,
        delete_scratch=True,
    )
    if not np.any(np.isfinite(unw)):
        raise RuntimeError("SNAPHU produced no finite pixels")
    return unw.astype(np.float32)


def run_unwrap_stage(
    context: PairContext,
    *,
    unwrap_method: str,
    block_rows: int,
) -> tuple[dict, str, str | None]:
    cached_outputs = _load_cached_stage_outputs(
        context.pair_dir,
        "p3",
        required_keys=("unwrapped_phase",),
    )
    if cached_outputs is not None:
        return cached_outputs, "cache", None

    try:
        interferogram = _load_cached_array(context.pair_dir, "p2", "filtered_interferogram")
    except RuntimeError:
        interferogram = _load_cached_array(context.pair_dir, "p2", "interferogram")
    coherence = _load_cached_array(context.pair_dir, "p2", "coherence")

    with tempfile.TemporaryDirectory(prefix="strip_insar2_unwrap_", dir=str(context.pair_dir)) as tmpdir:
        scratch = Path(tmpdir)
        if unwrap_method == "icu":
            unwrapped_phase = _unwrap_with_icu(interferogram, coherence, scratch)
        elif unwrap_method == "snaphu":
            unwrapped_phase = _unwrap_with_snaphu(interferogram, coherence, scratch)
        else:
            raise ValueError(f"unsupported unwrap method: {unwrap_method}")

    output_files = {
        "unwrapped_phase": _save_stage_array(context.pair_dir, "p3", "unwrapped_phase", unwrapped_phase)
    }
    _write_stage_outputs_record(
        output_dir=context.pair_dir,
        stage="p3",
        master_manifest_path=context.master_manifest_path,
        slave_manifest_path=context.slave_manifest_path,
        backend_used="cpu",
        output_files=output_files,
    )
    return output_files, "cpu", None


def compute_los_displacement(unwrapped_phase: np.ndarray, wavelength: float) -> np.ndarray:
    return (np.asarray(unwrapped_phase, dtype=np.float32) * float(wavelength) / (4.0 * np.pi)).astype(
        np.float32
    )


def run_los_stage(context: PairContext) -> tuple[dict, str, str | None]:
    cached_outputs = _load_cached_stage_outputs(
        context.pair_dir,
        "p4",
        required_keys=("los_displacement",),
    )
    if cached_outputs is not None:
        return cached_outputs, "cache", None

    unwrapped_phase = _load_cached_array(context.pair_dir, "p3", "unwrapped_phase")
    los_displacement = compute_los_displacement(unwrapped_phase, context.wavelength)
    output_files = {
        "los_displacement": _save_stage_array(
            context.pair_dir, "p4", "los_displacement", los_displacement
        )
    }
    _write_stage_outputs_record(
        output_dir=context.pair_dir,
        stage="p4",
        master_manifest_path=context.master_manifest_path,
        slave_manifest_path=context.slave_manifest_path,
        backend_used="cpu",
        output_files=output_files,
    )
    return output_files, "cpu", None


def write_insar_hdf(
    master_slc_path: str,
    slave_slc_path: str,
    interferogram: np.ndarray,
    coherence: np.ndarray,
    unwrapped_phase: np.ndarray,
    los_displacement: np.ndarray,
    wavelength: float,
    unwrap_method: str,
    output_h5: str,
    block_rows: int = 256,
    filtered_interferogram: np.ndarray | None = None,
) -> str:
    master_amp = _compute_slc_amplitude(master_slc_path)
    slave_amp = _compute_slc_amplitude(slave_slc_path)
    avg_amplitude = ((master_amp + slave_amp) * 0.5).astype(np.float32)
    rows, cols = avg_amplitude.shape

    with h5py.File(output_h5, "w") as f:
        f.attrs["product_type"] = "insar_interferogram_fullres"
        f.attrs["width"] = cols
        f.attrs["length"] = rows
        f.attrs["wavelength"] = float(wavelength)
        f.attrs["unwrap_method"] = unwrap_method
        f.attrs["radiometry"] = "complex_interferogram"
        f.attrs["value_domain"] = "phase"
        f.attrs["los_displacement_convention"] = "positive = toward satellite"

        chunk = (min(block_rows, rows), min(1024, cols))
        f.create_dataset("avg_amplitude", data=avg_amplitude, dtype="f4", chunks=chunk, compression="gzip", shuffle=True)
        f.create_dataset("interferogram", data=interferogram.astype(np.complex64), dtype=np.complex64, chunks=chunk, compression="gzip", shuffle=True)
        f.create_dataset("coherence", data=coherence.astype(np.float32), dtype="f4", chunks=chunk, compression="gzip", shuffle=True)
        f.create_dataset(
            "filtered_interferogram",
            data=(filtered_interferogram if filtered_interferogram is not None else interferogram).astype(np.complex64),
            dtype=np.complex64,
            chunks=chunk,
            compression="gzip",
            shuffle=True,
        )
        f.create_dataset("unwrapped_phase", data=unwrapped_phase.astype(np.float32), dtype="f4", chunks=chunk, compression="gzip", shuffle=True)
        f.create_dataset("los_displacement", data=los_displacement.astype(np.float32), dtype="f4", chunks=chunk, compression="gzip", shuffle=True)
    return output_h5


def write_primary_product(
    context: PairContext,
    *,
    p1_outputs: dict,
    gpu_mode: str,
    gpu_id: int,
    block_rows: int,
    unwrap_method: str,
) -> tuple[str, str, str | None]:
    cached_outputs = _load_cached_stage_outputs(
        context.pair_dir,
        "p5",
        required_keys=("interferogram_h5",),
    )
    if cached_outputs is not None:
        return str(cached_outputs["interferogram_h5"]), "cache", None

    master_slc = resolve_manifest_data_path(
        context.master_manifest_path,
        context.master_manifest["slc"]["path"],
    )
    slave_slc = _select_registered_slave_slc(
        p1_outputs,
        resolve_manifest_data_path(
            context.slave_manifest_path,
            context.slave_manifest["slc"]["path"],
        ),
    )
    interferogram = _load_cached_array(context.pair_dir, "p2", "interferogram")
    filtered_interferogram = _load_cached_array(context.pair_dir, "p2", "filtered_interferogram")
    coherence = _load_cached_array(context.pair_dir, "p2", "coherence")
    unwrapped_phase = _load_cached_array(context.pair_dir, "p3", "unwrapped_phase")
    los_displacement = _load_cached_array(context.pair_dir, "p4", "los_displacement")
    output_h5 = context.output_paths["interferogram_h5"]

    write_insar_hdf(
        master_slc,
        slave_slc,
        interferogram,
        coherence,
        unwrapped_phase,
        los_displacement,
        context.wavelength,
        unwrap_method,
        output_h5,
        block_rows=block_rows,
        filtered_interferogram=filtered_interferogram,
    )

    def _gpu():
        return append_topo_coordinates_hdf(
            str(context.master_manifest_path),
            context.resolved_dem,
            output_h5,
            block_rows=block_rows,
            orbit_interp=context.orbit_interp,
            use_gpu=True,
            gpu_id=gpu_id,
        )

    def _cpu():
        return append_topo_coordinates_hdf(
            str(context.master_manifest_path),
            context.resolved_dem,
            output_h5,
            block_rows=block_rows,
            orbit_interp=context.orbit_interp,
            use_gpu=False,
        )

    _, backend_used, fallback_reason = run_stage_with_fallback(
        stage_name="coordinates",
        gpu_mode=gpu_mode,
        gpu_id=gpu_id,
        gpu_runner=_gpu,
        cpu_runner=_cpu,
    )
    append_utm_coordinates_hdf(
        output_h5,
        str(context.master_manifest_path),
        block_rows=min(block_rows, 64),
    )
    _write_custom_stage_record(
        output_dir=context.pair_dir,
        stage="p5",
        master_manifest_path=context.master_manifest_path,
        slave_manifest_path=context.slave_manifest_path,
        backend_used=backend_used,
        output_files={"interferogram_h5": output_h5},
        fallback_reason=fallback_reason,
        upstream_stage_dependencies=["p4"],
    )
    return output_h5, backend_used, fallback_reason


def process_strip_insar2(
    master_manifest_path: str | Path,
    slave_manifest_path: str | Path,
    *,
    output_root: str | Path = "results",
    gpu_mode: str = "auto",
    gpu_id: int = 0,
    unwrap_method: str = "icu",
    resolution_meters: float = 20.0,
    block_rows: int = 256,
    dem_path: str | None = None,
    dem_cache_dir: str | None = None,
    dem_margin_deg: float = 0.2,
    no_kml: bool = False,
) -> dict:
    context = load_pair_context(
        master_manifest_path,
        slave_manifest_path,
        output_root=output_root,
        dem_path=dem_path,
        dem_cache_dir=dem_cache_dir,
        dem_margin_deg=dem_margin_deg,
    )
    stage_backends: dict[str, str] = {}
    fallback_reasons: dict[str, str] = {}

    _, backend_used, fallback_reason = run_geo2rdr_stage(
        context,
        gpu_mode=gpu_mode,
        gpu_id=gpu_id,
        block_rows=block_rows,
    )
    stage_backends["geo2rdr"] = backend_used
    if fallback_reason:
        fallback_reasons["geo2rdr"] = fallback_reason

    p1_outputs, backend_used, fallback_reason = run_resample_stage(
        context,
        gpu_mode=gpu_mode,
        gpu_id=gpu_id,
    )
    stage_backends["resample"] = backend_used
    if fallback_reason:
        fallback_reasons["resample"] = fallback_reason

    _, backend_used, fallback_reason = run_crossmul_stage(
        context,
        gpu_mode=gpu_mode,
        gpu_id=gpu_id,
        block_rows=block_rows,
    )
    stage_backends["crossmul"] = backend_used
    if fallback_reason:
        fallback_reasons["crossmul"] = fallback_reason

    _, backend_used, fallback_reason = run_unwrap_stage(
        context,
        unwrap_method=unwrap_method,
        block_rows=block_rows,
    )
    stage_backends["unwrap"] = backend_used
    if fallback_reason:
        fallback_reasons["unwrap"] = fallback_reason

    _, backend_used, fallback_reason = run_los_stage(context)
    stage_backends["los"] = backend_used
    if fallback_reason:
        fallback_reasons["los"] = fallback_reason

    output_h5, backend_used, fallback_reason = write_primary_product(
        context,
        p1_outputs=p1_outputs,
        gpu_mode=gpu_mode,
        gpu_id=gpu_id,
        block_rows=block_rows,
        unwrap_method=unwrap_method,
    )
    stage_backends["product"] = backend_used
    if fallback_reason:
        fallback_reasons["product"] = fallback_reason

    exported = export_insar_products(
        input_h5=output_h5,
        output_paths=context.output_paths,
        resolution_meters=resolution_meters,
        block_rows=min(block_rows, 64),
        generate_kml=not no_kml,
    )
    _write_custom_stage_record(
        output_dir=context.pair_dir,
        stage="p6",
        master_manifest_path=context.master_manifest_path,
        slave_manifest_path=context.slave_manifest_path,
        backend_used="cpu",
        output_files=exported,
        fallback_reason=None,
        upstream_stage_dependencies=["p5"],
    )

    return {
        "pair_name": context.pair_name,
        "pair_dir": str(context.pair_dir),
        "output_paths": context.output_paths,
        "exports": exported,
        "stage_backends": stage_backends,
        "fallback_reasons": fallback_reasons,
    }


def _parse_datetime(value: str) -> datetime:
    value = value.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    elif "T" in value and "+" not in value[-6:] and "-" not in value[-6:]:
        value = value + "+00:00"
    return datetime.fromisoformat(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NISAR-style strip InSAR orchestrator for manifest-based strip pairs",
    )
    parser.add_argument("master_json")
    parser.add_argument("slave_json")
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--gpu-mode", choices=["auto", "gpu", "cpu"], default="auto")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--unwrap-method", choices=["icu", "snaphu"], default="icu")
    parser.add_argument("--resolution", type=float, default=20.0)
    parser.add_argument("--block-rows", type=int, default=256)
    parser.add_argument("--dem-path")
    parser.add_argument("--dem-cache-dir")
    parser.add_argument("--dem-margin-deg", type=float, default=0.2)
    parser.add_argument("--no-kml", action="store_true")
    args = parser.parse_args()

    result = process_strip_insar2(
        args.master_json,
        args.slave_json,
        output_root=args.output_root,
        gpu_mode=args.gpu_mode,
        gpu_id=args.gpu_id,
        unwrap_method=args.unwrap_method,
        resolution_meters=args.resolution,
        block_rows=args.block_rows,
        dem_path=args.dem_path,
        dem_cache_dir=args.dem_cache_dir,
        dem_margin_deg=args.dem_margin_deg,
        no_kml=args.no_kml,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
