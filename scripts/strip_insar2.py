from __future__ import annotations

import argparse
import colorsys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import py_compile
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
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
SNAPHU_RETRY_PROFILES = (
    ("default", {}),
    (
        "relaxed",
        {
            "cost_mode": "defo",
            "min_conncomp_frac": 0.001,
            "min_region_size": 100,
            "phase_grad_window": (5, 5),
            "single_tile_reoptimize": False,
        },
    ),
)
ISCE3_GEOMETRY_LINES_PER_BLOCK_DEFAULT = 1000
ISCE3_CROSSMUL_LINES_PER_BLOCK_DEFAULT = 1024


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
STAGE_LOG_LABELS = {
    "p0": "rdr2geo/topo",
    "p1": "resample/registration",
    "p2": "crossmul/filter",
    "p3": "unwrap",
    "p4": "los",
    "p5": "product/hdf",
    "p6": "export/publish",
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
        "window_range": 64,
        "window_azimuth": 64,
        "half_search_range": 20,
        "half_search_azimuth": 20,
        "skip_range": 32,
        "skip_azimuth": 32,
        "margin": 48,
        "gross_offset_range": None,
        "gross_offset_azimuth": None,
        "start_pixel_range": None,
        "start_pixel_azimuth": None,
        "offset_width": None,
        "offset_length": None,
        "cross_correlation_domain": "frequency",
        "slc_oversampling_factor": 2,
        "deramping_method": "complex",
        "deramping_axis": "azimuth",
        "correlation_statistics_zoom": 21,
        "correlation_surface_zoom": 8,
        "correlation_surface_oversampling_factor": 64,
        "correlation_surface_oversampling_method": "fft",
        "windows_batch_range": 10,
        "windows_batch_azimuth": 1,
        "cuda_streams": 2,
        "use_gross_offsets": False,
        "gross_offset_filepath": None,
        "merge_gross_offset": False,
    }


def _select_strip_dense_match_plan(effective_resolution: float) -> dict:
    module = importlib.import_module("insar_registration")
    return module._select_cpu_dense_match_plan(float(effective_resolution or 0.0))


def _run_strip_pycuampcor_dense_offsets(**kwargs):
    module = importlib.import_module("insar_registration")
    return module.run_pycuampcor_dense_offsets(**kwargs)


def _run_strip_cpu_dense_offsets(**kwargs):
    module = importlib.import_module("insar_registration")
    return module.run_cpu_dense_offsets(**kwargs)


def _write_strip_registration_outputs(**kwargs):
    module = importlib.import_module("insar_registration")
    return module.write_registration_outputs(**kwargs)


def _write_strip_varying_gross_offset_file(**kwargs):
    module = importlib.import_module("insar_registration")
    return module._write_varying_gross_offset_file(**kwargs)


def _strip_raster_shape(path: str | Path) -> tuple[int, int] | None:
    module = importlib.import_module("insar_registration")
    return module._raster_shape(path)


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


def _resolve_stage_block_rows(block_rows: int | None) -> tuple[int, int]:
    if block_rows is None:
        return (
            ISCE3_GEOMETRY_LINES_PER_BLOCK_DEFAULT,
            ISCE3_CROSSMUL_LINES_PER_BLOCK_DEFAULT,
        )
    rows = int(block_rows)
    return rows, rows


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


_CONSOLE_WRITE_LOCK = threading.Lock()
try:
    _CONSOLE_STDOUT_FD = os.dup(1)
except Exception:
    _CONSOLE_STDOUT_FD = None


def _console_write(text: str) -> None:
    if not isinstance(text, str):
        text = str(text)
    if _CONSOLE_STDOUT_FD is None:
        print(text, end="", flush=True)
        return
    data = text.encode("utf-8", errors="replace")
    with _CONSOLE_WRITE_LOCK:
        os.write(_CONSOLE_STDOUT_FD, data)


def _log_stage_status(
    stage: str,
    status: str,
    *,
    backend_used: str | None = None,
    fallback_reason: str | None = None,
) -> None:
    label = STAGE_LOG_LABELS.get(stage, stage)
    message = f"[{utc_now_iso()}] [{stage}] {status} {label}"
    if backend_used:
        message += f" backend={backend_used}"
    if fallback_reason:
        message += f" fallback={fallback_reason}"
    print(message, flush=True)


class _StageProgress:
    def __init__(self, stage: str):
        self.stage = stage
        self._active = False
        self._last_len = 0
        self._last_message = ""
        self._last_running_tick = -1

    def _render(self, message: str) -> None:
        padded = message
        if self._last_len > len(message):
            padded += " " * (self._last_len - len(message))
        _console_write(f"\r{padded}")
        self._active = True
        self._last_len = len(message)
        self._last_message = message

    def block(self, *, backend: str, current: int, total: int, detail: str) -> None:
        total = max(int(total), 1)
        current = min(max(int(current), 0), total)
        percent = 100.0 * float(current) / float(total)
        self._render(
            f"[{self.stage}] {backend.upper()} progress {current}/{total} blocks "
            f"({percent:.1f}%) {detail}"
        )

    def running(self, *, backend: str, detail: str, elapsed: float, force: bool = False) -> None:
        tick = int(max(float(elapsed), 0.0))
        if not force and tick == self._last_running_tick:
            return
        self._last_running_tick = tick
        self._render(f"[{self.stage}] {backend.upper()} running {detail} elapsed={tick}s")

    def close(self) -> None:
        if self._active:
            _console_write("\n")
        self._active = False
        self._last_len = 0
        self._last_message = ""
        self._last_running_tick = -1


def _run_with_running_progress(
    *,
    progress_reporter: _StageProgress | None,
    backend: str,
    detail: str,
    func,
    interval_s: float = 1.0,
):
    if progress_reporter is None:
        return func()

    start_time = time.monotonic()
    stop_event = threading.Event()

    def _ticker() -> None:
        progress_reporter.running(backend=backend, detail=detail, elapsed=0.0, force=True)
        while not stop_event.wait(max(float(interval_s), 0.05)):
            progress_reporter.running(
                backend=backend,
                detail=detail,
                elapsed=time.monotonic() - start_time,
            )

    thread = threading.Thread(target=_ticker, daemon=True)
    thread.start()
    try:
        return func()
    finally:
        stop_event.set()
        thread.join(timeout=max(float(interval_s), 0.05) + 0.2)


@contextmanager
def _silence_isce3_journal(log_path: str | Path):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        sys.stdout.flush()
    except Exception:
        pass
    try:
        sys.stderr.flush()
    except Exception:
        pass

    with open(log_path, "a", encoding="utf-8") as sink:
        saved_stdout = os.dup(1)
        saved_stderr = os.dup(2)
        try:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
            yield str(log_path)
        finally:
            try:
                sys.stdout.flush()
            except Exception:
                pass
            try:
                sys.stderr.flush()
            except Exception:
                pass
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)


def _run_with_silenced_journal(log_path: str | Path, func):
    with _silence_isce3_journal(log_path):
        return func()


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
    arr_in = data if data is not None else np.zeros((1, 1), dtype=np.float32)
    arr = np.asarray(arr_in)
    dtype = gdal.GDT_Float64 if arr.dtype == np.dtype(np.float64) else gdal.GDT_Float32
    if dtype == gdal.GDT_Float64:
        arr = np.asarray(arr, dtype=np.float64)
    else:
        arr = np.asarray(arr, dtype=np.float32)
    return _write_float_gtiff(path, arr, dtype=dtype, nodata=GEO2RDR_OFFSET_NODATA)


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


def _goldstein_filter_gpu(
    interferogram: np.ndarray,
    *,
    gpu_id: int,
    alpha: float = 0.5,
    window_size: int = 32,
    step: int | None = None,
    progress_reporter: _StageProgress | None = None,
) -> np.ndarray:
    cp = importlib.import_module("cupy")

    if interferogram.ndim != 2:
        raise ValueError("goldstein_filter expects a 2D interferogram")
    if window_size < 1:
        raise ValueError("window_size must be positive")
    if step is None:
        step = window_size // 2
    if step < 1:
        step = 1

    with cp.cuda.Device(int(gpu_id)):
        arr = cp.asarray(np.asarray(interferogram, dtype=np.complex64))
        rows, cols = arr.shape
        hanning = cp.outer(cp.hanning(window_size), cp.hanning(window_size)).astype(cp.complex64)
        filtered = cp.zeros_like(arr, dtype=cp.complex64)
        weight_sum = cp.zeros((rows, cols), dtype=cp.float32)

        row_starts = list(range(0, rows - window_size + 1, step))
        col_starts = list(range(0, cols - window_size + 1, step))
        total_blocks = max(len(row_starts) * len(col_starts), 1)
        current_block = 0

        for row_start in row_starts:
            for col_start in col_starts:
                window_data = arr[
                    row_start:row_start + window_size,
                    col_start:col_start + window_size,
                ]
                spectrum = cp.fft.fft2(window_data * hanning)
                psd = cp.abs(spectrum) ** 2
                weight = cp.power(psd + 1.0e-10, alpha / 2.0)
                filtered_window = cp.fft.ifft2(spectrum * weight)
                filtered[
                    row_start:row_start + window_size,
                    col_start:col_start + window_size,
                ] += filtered_window
                weight_sum[
                    row_start:row_start + window_size,
                    col_start:col_start + window_size,
                ] += 1.0
                current_block += 1
                if progress_reporter is not None:
                    progress_reporter.block(
                        backend="gpu",
                        current=current_block,
                        total=total_blocks,
                        detail="goldstein_filter",
                    )

        weight_sum = cp.where(weight_sum == 0, 1.0, weight_sum)
        result = (filtered / weight_sum).astype(cp.complex64)
        cp.cuda.runtime.deviceSynchronize()
        return cp.asnumpy(result)


def _run_goldstein_filter(
    *,
    interferogram: np.ndarray,
    use_gpu: bool,
    gpu_id: int,
    progress_reporter: _StageProgress | None = None,
) -> tuple[np.ndarray, str, str | None]:
    fallback_reason = None
    if use_gpu:
        try:
            filtered = _run_with_running_progress(
                progress_reporter=progress_reporter,
                backend="gpu",
                detail="goldstein_filter",
                func=lambda: _goldstein_filter_gpu(
                    interferogram,
                    gpu_id=gpu_id,
                    progress_reporter=progress_reporter,
                ),
            )
            return np.asarray(filtered, dtype=np.complex64), "gpu", None
        except Exception as exc:
            fallback_reason = str(exc)

    filtered = _run_with_running_progress(
        progress_reporter=progress_reporter,
        backend="cpu",
        detail="goldstein_filter",
        func=lambda: goldstein_filter(np.asarray(interferogram, dtype=np.complex64)),
    )
    return np.asarray(filtered, dtype=np.complex64), "cpu", fallback_reason


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


def _write_complex_envi(path: Path, data: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows, cols = data.shape
    ds = gdal.GetDriverByName("ENVI").Create(
        str(path),
        cols,
        rows,
        1,
        gdal.GDT_CFloat32,
    )
    if ds is None:
        raise RuntimeError(f"failed to create ENVI raster: {path}")
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
    block_rows: int = ISCE3_GEOMETRY_LINES_PER_BLOCK_DEFAULT,
    orbit_interp: str | None = None,
    use_gpu: bool = False,
    gpu_id: int = 0,
    progress_reporter: _StageProgress | None = None,
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

        x_raster = make_raster("x", gdal.GDT_Float64)
        y_raster = make_raster("y", gdal.GDT_Float64)
        z_raster = make_raster("z", gdal.GDT_Float64)
        inc_raster = make_raster("inc")
        hdg_raster = make_raster("hdg")
        local_inc_raster = make_raster("localInc")
        local_psi_raster = make_raster("localPsi")
        simamp_raster = make_raster("simamp")
        layover_raster = make_raster("layoverShadowMask", gdal.GDT_Byte)
        los_e_raster = make_raster("los_east")
        los_n_raster = make_raster("los_north")

        with _silence_isce3_journal(stage_dir(output_h5.parent, "p5") / "isce3_journal.log"):
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
            d_lon = f.create_dataset("longitude", shape=(length, width), dtype="f8", chunks=(min(block_rows, length), min(1024, width)), compression="gzip", shuffle=True)
            d_lat = f.create_dataset("latitude", shape=(length, width), dtype="f8", chunks=(min(block_rows, length), min(1024, width)), compression="gzip", shuffle=True)
            d_hgt = f.create_dataset("height", shape=(length, width), dtype="f8", chunks=(min(block_rows, length), min(1024, width)), compression="gzip", shuffle=True)
            f.attrs["coordinate_system"] = "EPSG:4326"
            f.attrs["longitude_units"] = "degrees_east"
            f.attrs["latitude_units"] = "degrees_north"
            f.attrs["height_units"] = "meters"
            f.attrs["coordinate_source"] = "rdr2geo_topo_with_validated_dem"

            lon_band = lon_ds.GetRasterBand(1)
            lat_band = lat_ds.GetRasterBand(1)
            hgt_band = hgt_ds.GetRasterBand(1)
            total_blocks = max((int(length) + int(block_rows) - 1) // int(block_rows), 1)
            for row0 in range(0, length, block_rows):
                rows = min(block_rows, length - row0)
                d_lon[row0:row0 + rows, :] = _read_band_array(lon_band, 0, row0, width, rows).astype(np.float64)
                d_lat[row0:row0 + rows, :] = _read_band_array(lat_band, 0, row0, width, rows).astype(np.float64)
                d_hgt[row0:row0 + rows, :] = _read_band_array(hgt_band, 0, row0, width, rows).astype(np.float64)
                if progress_reporter is not None:
                    progress_reporter.block(
                        backend="gpu" if use_gpu else "cpu",
                        current=(row0 // block_rows) + 1,
                        total=total_blocks,
                        detail="write_hdf_coordinates",
                    )
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
    value = np.ones(phase.shape, dtype=np.float64)
    try:
        avg_amplitude, _ = accumulate_utm_grid(
            input_h5,
            "avg_amplitude",
            target_width=phase.shape[1],
            target_height=phase.shape[0],
            block_rows=block_rows,
        )
        avg_amplitude, _ = prepare_display_grid(
            avg_amplitude,
            confidence_interval_pct=98.0,
        )
        amp_valid = np.isfinite(avg_amplitude) & (avg_amplitude > 0)
        if np.any(amp_valid):
            amp_db = 10.0 * np.log10(avg_amplitude[amp_valid])
            p2 = float(np.percentile(amp_db, 2))
            p98 = float(np.percentile(amp_db, 98))
            scaled = np.clip((amp_db - p2) / (p98 - p2 + 1.0e-9), 0.0, 1.0)
            value[:] = 0.15
            value[amp_valid] = np.clip(scaled, 0.15, 1.0)
    except Exception:
        pass
    valid = np.isfinite(phase)
    if np.any(valid):
        hue = ((phase[valid] + np.pi) / (2.0 * np.pi)).astype(np.float64)
        colors = np.array(
            [
                colorsys.hsv_to_rgb(float(h), 1.0, float(v))
                for h, v in zip(hue, value[valid], strict=False)
            ]
        )
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

    with _silence_isce3_journal(output_dir / "isce3_journal.log"):
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
    progress = _StageProgress("p0")

    def _gpu():
        def _impl():
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

        return _run_with_running_progress(
            progress_reporter=progress,
            backend="gpu",
            detail="rdr2geo/topo",
            func=_impl,
        )

    def _cpu():
        def _impl():
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

        return _run_with_running_progress(
            progress_reporter=progress,
            backend="cpu",
            detail="rdr2geo/topo",
            func=_impl,
        )

    try:
        output_files, backend_used, fallback_reason = run_stage_with_fallback(
            stage_name="rdr2geo",
            gpu_mode=gpu_mode,
            gpu_id=gpu_id,
            gpu_runner=_gpu,
            cpu_runner=_cpu,
        )
    finally:
        progress.close()
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
    return _write_offset_raster(output_path, arr)


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
    block_rows: int = ISCE3_GEOMETRY_LINES_PER_BLOCK_DEFAULT,
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
        with _silence_isce3_journal(artifacts_dir / "isce3_journal.log"):
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
    coarse_coreg_slave_gtiff_path: str | None = None,
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
        output_slc = np.zeros(out_shape, dtype=np.complex64)
        log_path = Path(coarse_coreg_slave_path).parent / "isce3_resample_journal.log"
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
        az_offsets[az_invalid] = NISAR_OFFSET_INVALID_VALUE
        rg_offsets[rg_invalid] = NISAR_OFFSET_INVALID_VALUE
        if block_size_rg == 0:
            block_size_rg = out_shape[1]
        _run_with_silenced_journal(
            log_path,
            lambda: resample_slc_blocks(
                output_resampled_slcs=[output_slc],
                input_slcs=[input_slc],
                az_offsets_dataset=az_offsets,
                rg_offsets_dataset=rg_offsets,
                input_radar_grid=radar_grid,
                doppler=doppler if doppler is not None else LUT2d(),
                block_size_az=block_size_az,
                block_size_rg=block_size_rg,
                fill_value=0.0 + 0.0j,
                quiet=True,
                with_gpu=use_gpu,
            ),
        )
        _write_complex_envi(Path(coarse_coreg_slave_path), output_slc)
        if coarse_coreg_slave_gtiff_path:
            _write_complex_gtiff(Path(coarse_coreg_slave_gtiff_path), output_slc)
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
    reference_envi = Path(output_dir / "reference.slc")
    secondary_envi = Path(output_dir / "secondary.slc")
    _copy_raster_to_envi_complex64(reference_slc_path, reference_envi)
    _copy_raster_to_envi_complex64(secondary_slc_path, secondary_envi)
    shape = _raster_shape(reference_envi)
    secondary_shape = _raster_shape(secondary_envi)
    if shape is None:
        raise RuntimeError(f"failed to open reference SLC for offsets: {reference_envi}")
    if secondary_shape is None:
        raise RuntimeError(f"failed to open secondary SLC for offsets: {secondary_envi}")
    if shape != secondary_shape:
        raise RuntimeError(
            f"dense offsets input shape mismatch: reference={shape} secondary={secondary_shape}"
        )
    rows, cols = shape

    import isce3.matchtemplate

    if use_gpu:
        import isce3.cuda.core
        import isce3.cuda.matchtemplate
        device = isce3.cuda.core.Device(gpu_id)
        isce3.cuda.core.set_device(device)
        ampcor = isce3.cuda.matchtemplate.PyCuAmpcor()
        ampcor.deviceID = gpu_id
    else:
        ampcor = isce3.matchtemplate.PyCPUAmpcor()

    ampcor.useMmap = 1
    ampcor.referenceImageName = str(reference_envi)
    ampcor.referenceImageHeight = int(rows)
    ampcor.referenceImageWidth = int(cols)
    ampcor.secondaryImageName = str(secondary_envi)
    ampcor.secondaryImageHeight = int(rows)
    ampcor.secondaryImageWidth = int(cols)
    _configure_ampcor_from_cfg(
        ampcor,
        cfg=_default_dense_offsets_cfg(),
        rows=int(rows),
        cols=int(cols),
        gross_offset_filepath=None,
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

    _create_empty_dataset(
        str(dense_offsets_path),
        ampcor.numberWindowAcross,
        ampcor.numberWindowDown,
        2,
        gdal.GDT_Float32,
    )
    _create_empty_dataset(
        str(gross_offsets_path),
        ampcor.numberWindowAcross,
        ampcor.numberWindowDown,
        2,
        gdal.GDT_Float32,
    )
    _create_empty_dataset(
        str(snr_path),
        ampcor.numberWindowAcross,
        ampcor.numberWindowDown,
        1,
        gdal.GDT_Float32,
    )
    _create_empty_dataset(
        str(covariance_path),
        ampcor.numberWindowAcross,
        ampcor.numberWindowDown,
        3,
        gdal.GDT_Float32,
    )
    _create_empty_dataset(
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

    coarse_coreg_slave_path = str(
        p1_stage_path / "coarse_resample_slc" / "freqA" / "HH" / "coregistered_secondary.slc"
    )
    coarse_coreg_slave_gtiff_path = str(p1_stage_path / "coarse_coreg_slave.tif")
    coarse_ok = run_coarse_resamp_isce3_v2(
        slave_slc_path=slave_slc,
        coarse_coreg_slave_path=coarse_coreg_slave_path,
        coarse_coreg_slave_gtiff_path=coarse_coreg_slave_gtiff_path,
        radar_grid=slave_radar_grid,
        doppler=slave_doppler,
        ref_radar_grid=ref_radar_grid,
        rg_offset_path=coarse_rg_offset_path,
        az_offset_path=coarse_az_offset_path,
        use_gpu=use_gpu,
    )
    if not coarse_ok:
        raise RuntimeError("NISAR coarse resample failed")

    range_res = float(context.master_rg_data.get("groundRangeResolution", 0.0) or 0.0)
    azimuth_res = float(context.master_rg_data.get("azimuthResolution", 0.0) or 0.0)
    effective_resolution = max(range_res, azimuth_res)
    dense_plan = _select_strip_dense_match_plan(effective_resolution)
    dense_offsets_dir = p1_stage_path / "dense_offsets" / "freqA" / "HH"
    dense_offsets_dir.mkdir(parents=True, exist_ok=True)
    gross_offset = (0.0, 0.0)
    if coarse_az_offset_path and coarse_rg_offset_path:
        try:
            gross_offset = (
                _estimate_offset_mean_from_raster(coarse_az_offset_path),
                _estimate_offset_mean_from_raster(coarse_rg_offset_path),
            )
        except Exception:
            gross_offset = (0.0, 0.0)

    row_offset = None
    col_offset = None
    dense_match_details = None
    dense_source = "pycuampcor" if use_gpu else "cpu-dense-match"
    dense_plan_record = {
        "effective_resolution": float(effective_resolution),
        "gross_offset": {
            "azimuth": float(gross_offset[0]),
            "range": float(gross_offset[1]),
        },
        **dense_plan,
    }
    (p1_stage_path / "dense_match_plan.json").write_text(
        json.dumps(dense_plan_record, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    try:
        if use_gpu:
            first_candidate = dense_plan["candidates"][0]
            pycuampcor_kwargs: dict[str, object] = {}
            use_gross_on_coarse_coreg = (
                str(os.environ.get("D2SAR_DENSE_GROSS_ON_COARSE_COREG", "0"))
                .strip()
                .lower()
                in {"1", "true", "yes", "on"}
            )
            if (
                use_gross_on_coarse_coreg
                and coarse_rg_offset_path is not None
                and coarse_az_offset_path is not None
            ):
                try:
                    full_shape = _strip_raster_shape(master_slc)
                    if full_shape is not None:
                        varying_cfg = _write_strip_varying_gross_offset_file(
                            range_offset_path=coarse_rg_offset_path,
                            azimuth_offset_path=coarse_az_offset_path,
                            output_path=dense_offsets_dir / "pycuampcor_varying_gross_offsets.bin",
                            full_shape=full_shape,
                            window_size=tuple(first_candidate["window_size"]),
                            search_range=tuple(first_candidate["search_range"]),
                            skip=(32, 32),
                        )
                        pycuampcor_kwargs = {
                            "gross_offset_filepath": varying_cfg.get("gross_offset_filepath"),
                            "reference_start_pixel_down": varying_cfg.get(
                                "reference_start_pixel_down"
                            ),
                            "reference_start_pixel_across": varying_cfg.get(
                                "reference_start_pixel_across"
                            ),
                            "number_window_down": varying_cfg.get("number_window_down"),
                            "number_window_across": varying_cfg.get("number_window_across"),
                        }
                        dense_plan_record["varying_gross_offsets"] = varying_cfg
                        (p1_stage_path / "dense_match_plan.json").write_text(
                            json.dumps(dense_plan_record, indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                except Exception as exc:
                    print(
                        f"[P1] varying gross offsets unavailable for pycuampcor: {exc}",
                        flush=True,
                    )
            row_offset, col_offset, dense_match_details = _run_strip_pycuampcor_dense_offsets(
                master_slc_path=master_slc,
                slave_slc_path=coarse_coreg_slave_path,
                output_dir=dense_offsets_dir,
                gpu_id=gpu_id,
                return_details=True,
                window_size=tuple(first_candidate["window_size"]),
                search_range=tuple(first_candidate["search_range"]),
                **pycuampcor_kwargs,
            )
        else:
            row_offset, col_offset, dense_match_details = _run_strip_cpu_dense_offsets(
                master_slc_path=master_slc,
                slave_slc_path=coarse_coreg_slave_path,
                output_dir=dense_offsets_dir,
                return_details=True,
                gross_offset=gross_offset,
                window_candidates=dense_plan["candidates"],
            )
    except NotImplementedError:
        row_offset = None
        col_offset = None
        dense_match_details = None
    except Exception:
        row_offset = None
        col_offset = None
        dense_match_details = None

    registration_outputs = _run_with_silenced_journal(
        p1_stage_path / "isce3_registration_journal.log",
        lambda: _write_strip_registration_outputs(
            stage_path=p1_stage_path,
            slave_slc_path=slave_slc,
            coarse_coreg_slave_path=coarse_coreg_slave_gtiff_path,
            coarse_rg_offset_path=coarse_rg_offset_path,
            coarse_az_offset_path=coarse_az_offset_path,
            row_offset=row_offset,
            col_offset=col_offset,
            dense_match_details=dense_match_details,
            source=dense_source if row_offset is not None and col_offset is not None else "coarse-resample-only",
            use_gpu=use_gpu,
            radar_grid=slave_radar_grid,
            doppler=slave_doppler,
            ref_radar_grid=ref_radar_grid,
        ),
    )

    fine_coreg_slave_tif_path = registration_outputs["fine_coreg_slave"]
    fine_coreg_slave_path = str(fine_coreg_slave_tif_path)
    fine_coreg_slave_gtiff_path = str(fine_coreg_slave_tif_path)
    registration_model = Path(registration_outputs["registration_model"])
    range_offset_gtiff = str(registration_outputs["range_offsets"])
    azimuth_offset_gtiff = str(registration_outputs["azimuth_offsets"])
    range_residual_gtiff = str(registration_outputs["range_residual_offsets"])
    azimuth_residual_gtiff = str(registration_outputs["azimuth_residual_offsets"])

    outputs = {
        "coarse_coreg_slave": coarse_coreg_slave_path,
        "fine_coreg_slave": fine_coreg_slave_path,
        "coarse_coreg_slave_tif": coarse_coreg_slave_gtiff_path,
        "fine_coreg_slave_tif": fine_coreg_slave_gtiff_path,
        "registration_model": str(registration_model),
        "range_offsets": range_offset_gtiff,
        "azimuth_offsets": azimuth_offset_gtiff,
        "range_residual_offsets": range_residual_gtiff,
        "azimuth_residual_offsets": azimuth_residual_gtiff,
        "dense_offsets_dir": str(dense_offsets_dir),
        "coarse_geo2rdr_range_offsets": coarse_rg_offset_path,
        "coarse_geo2rdr_azimuth_offsets": coarse_az_offset_path,
        "dense_match_plan": str(p1_stage_path / "dense_match_plan.json"),
    }

    try:
        outputs["coarse_coreg_slave_png"] = _write_radar_amplitude_png(
            coarse_coreg_slave_gtiff_path,
            context.pair_dir / "slave_coarse_coreg_fullres.png",
        )
    except Exception:
        pass
    try:
        outputs["fine_coreg_slave_png"] = _write_radar_amplitude_png(
            fine_coreg_slave_gtiff_path,
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
    progress = _StageProgress("p1")

    def _gpu():
        return _run_with_running_progress(
            progress_reporter=progress,
            backend="gpu",
            detail="resample/registration",
            func=lambda: _run_nisar_registration_chain(
                context=context,
                use_gpu=True,
                gpu_id=gpu_id,
                p1_stage_path=p1_stage_path,
            ),
        )

    def _cpu():
        return _run_with_running_progress(
            progress_reporter=progress,
            backend="cpu",
            detail="resample/registration",
            func=lambda: _run_nisar_registration_chain(
                context=context,
                use_gpu=False,
                gpu_id=gpu_id,
                p1_stage_path=p1_stage_path,
            ),
        )

    try:
        result, backend_used, fallback_reason = run_stage_with_fallback(
            stage_name="resample",
            gpu_mode=gpu_mode,
            gpu_id=gpu_id,
            gpu_runner=_gpu,
            cpu_runner=_cpu,
        )
    finally:
        progress.close()
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
    block_rows: int = ISCE3_CROSSMUL_LINES_PER_BLOCK_DEFAULT,
    progress_reporter: _StageProgress | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    master_slc = _open_slc_as_complex(master_slc_path)
    slave_slc = _open_slc_as_complex(slave_slc_path)
    if master_slc.shape != slave_slc.shape:
        raise RuntimeError(
            f"shape mismatch for crossmul: master={master_slc.shape}, slave={slave_slc.shape}"
        )
    interferogram = np.zeros(master_slc.shape, dtype=np.complex64)

    range_offset = None
    mask = None
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

    total_blocks = max((int(master_slc.shape[0]) + int(block_rows) - 1) // int(block_rows), 1)
    for row0 in range(0, master_slc.shape[0], block_rows):
        rows = min(block_rows, master_slc.shape[0] - row0)
        block = master_slc[row0 : row0 + rows] * np.conj(slave_slc[row0 : row0 + rows])
        block = block.astype(np.complex64, copy=False)
        if range_offset is not None:
            block_offsets = range_offset[row0 : row0 + rows].astype(np.float32, copy=False)
            phase = (
                4.0
                * np.pi
                * (
                    float(range_pixel_spacing) * block_offsets
                    + float(flatten_starting_range_shift_m or 0.0)
                )
                / float(wavelength)
            )
            flatten_term = np.exp(-1j * phase.astype(np.float32)).astype(np.complex64)
            block_mask = mask[row0 : row0 + rows] if mask is not None else None
            if block_mask is None:
                block *= flatten_term
            elif np.any(block_mask):
                block[block_mask] *= flatten_term[block_mask]
        interferogram[row0 : row0 + rows] = block
        if progress_reporter is not None:
            progress_reporter.block(
                backend="cpu",
                current=(row0 // block_rows) + 1,
                total=total_blocks,
                detail="crossmul",
            )

    coherence = _estimate_coherence(master_slc, slave_slc)
    return interferogram.astype(np.complex64), coherence


def _copy_raster_to_envi_complex64(
    src: str,
    dst: str | Path,
) -> str:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.unlink(missing_ok=True)
    Path(f"{dst}.hdr").unlink(missing_ok=True)

    src_ds = gdal.Open(str(src), gdal.GA_ReadOnly)
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
    if driver is None:
        raise RuntimeError(f"GDAL driver unavailable: {file_type}")
    ds = driver.Create(
        str(filename),
        xsize=int(width),
        ysize=int(length),
        bands=int(bands),
        eType=dtype,
        options=[f"INTERLEAVE={interleave}"],
    )
    if ds is None:
        raise RuntimeError(f"failed to create dataset: {filename}")
    ds.FlushCache()
    ds = None


def _raster_shape(path: str | Path) -> tuple[int, int] | None:
    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    if ds is None:
        return None
    try:
        rows = int(ds.RasterYSize)
        cols = int(ds.RasterXSize)
    finally:
        ds = None
    if rows <= 0 or cols <= 0:
        return None
    return rows, cols


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
            skip_down += max(1, int(skip[0]))
            continue
        if needs_across:
            skip_across += max(1, int(skip[1]))
            continue
        if number_window_down * number_window_across > int(max_windows):
            if number_window_down >= number_window_across:
                skip_down += max(1, int(skip[0]))
            else:
                skip_across += max(1, int(skip[1]))

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


def _configure_ampcor_from_cfg(
    ampcor,
    *,
    cfg: dict,
    rows: int,
    cols: int,
    gross_offset_filepath: str | None = None,
) -> dict:
    window_size = (int(cfg["window_azimuth"]), int(cfg["window_range"]))
    search_range = (int(cfg["half_search_azimuth"]), int(cfg["half_search_range"]))
    skip = (int(cfg["skip_azimuth"]), int(cfg["skip_range"]))
    planned_grid = _plan_matching_grid(
        rows=int(rows),
        cols=int(cols),
        window_size=window_size,
        search_range=search_range,
        skip=skip,
        gross_offset=(0.0, 0.0),
        max_windows=2400,
        max_window_down=60,
        max_window_across=40,
    )

    ampcor.windowSizeHeight = window_size[0]
    ampcor.windowSizeWidth = window_size[1]
    ampcor.halfSearchRangeDown = search_range[0]
    ampcor.halfSearchRangeAcross = search_range[1]
    ampcor.skipSampleDown = int(planned_grid["skip_down"])
    ampcor.skipSampleAcross = int(planned_grid["skip_across"])
    ampcor.referenceStartPixelDownStatic = int(planned_grid["reference_start_pixel_down"])
    ampcor.referenceStartPixelAcrossStatic = int(planned_grid["reference_start_pixel_across"])
    ampcor.numberWindowDown = int(planned_grid["number_window_down"])
    ampcor.numberWindowAcross = int(planned_grid["number_window_across"])
    ampcor.algorithm = 0 if cfg["cross_correlation_domain"] == "frequency" else 1
    ampcor.rawDataOversamplingFactor = int(cfg["slc_oversampling_factor"])
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
    ampcor.corrSurfaceOverSamplingMethod = (
        0 if cfg["correlation_surface_oversampling_method"] == "fft" else 1
    )
    ampcor.numberWindowAcrossInChunk = int(cfg["windows_batch_range"])
    ampcor.numberWindowDownInChunk = int(cfg["windows_batch_azimuth"])
    ampcor.nStreams = int(cfg["cuda_streams"])
    ampcor.setupParams()
    ampcor.setConstantGrossOffset(0, 0)
    if gross_offset_filepath:
        gross_offset = np.fromfile(gross_offset_filepath, dtype=np.int32)
        windows_number = ampcor.numberWindowAcross * ampcor.numberWindowDown
        if gross_offset.size != 2 * windows_number:
            raise RuntimeError(
                "The input gross offset does not match the offset width*offset length"
            )
        gross_offset = gross_offset.reshape(windows_number, 2)
        ampcor.setVaryingGrossOffset(gross_offset[:, 0], gross_offset[:, 1])
    ampcor.checkPixelInImageRange()
    return planned_grid


def _crossmul_isce3_gpu(
    *,
    master_slc_path: str,
    slave_slc_path: str,
    output_dir: Path,
    gpu_id: int,
    block_rows: int,
    flatten_raster: str | None = None,
    range_pixel_spacing: float | None = None,
    wavelength: float | None = None,
    flatten_starting_range_shift_m: float | None = None,
    progress_reporter: _StageProgress | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    return _crossmul_isce3_gpu_subprocess(
        master_slc_path=master_slc_path,
        slave_slc_path=slave_slc_path,
        output_dir=output_dir,
        gpu_id=gpu_id,
        block_rows=block_rows,
        flatten_raster=flatten_raster,
        range_pixel_spacing=range_pixel_spacing,
        wavelength=wavelength,
        flatten_starting_range_shift_m=flatten_starting_range_shift_m,
        progress_reporter=progress_reporter,
    )


def _run_crossmul_filter_gpu_pipeline(
    *,
    master_slc_path: str,
    slave_slc_path: str,
    output_dir: Path,
    gpu_id: int,
    block_rows: int,
    flatten_raster: str | None = None,
    range_pixel_spacing: float | None = None,
    wavelength: float | None = None,
    flatten_starting_range_shift_m: float | None = None,
    progress_reporter: _StageProgress | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str | None]:
    return _crossmul_filter_isce3_gpu_subprocess(
        master_slc_path=master_slc_path,
        slave_slc_path=slave_slc_path,
        output_dir=output_dir,
        gpu_id=gpu_id,
        block_rows=block_rows,
        flatten_raster=flatten_raster,
        range_pixel_spacing=range_pixel_spacing,
        wavelength=wavelength,
        flatten_starting_range_shift_m=flatten_starting_range_shift_m,
        progress_reporter=progress_reporter,
    )


def _crossmul_isce3_gpu_subprocess(
    *,
    master_slc_path: str,
    slave_slc_path: str,
    output_dir: Path,
    gpu_id: int,
    block_rows: int,
    flatten_raster: str | None = None,
    range_pixel_spacing: float | None = None,
    wavelength: float | None = None,
    flatten_starting_range_shift_m: float | None = None,
    progress_reporter: _StageProgress | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    output_dir = Path(output_dir)
    p2_dir = stage_dir(output_dir, "p2")
    p2_dir.mkdir(parents=True, exist_ok=True)

    ifg_path = p2_dir / "cuda_interferogram.int"
    coh_path = p2_dir / "cuda_coherence.bin"
    helper = p2_dir / "cuda_crossmul_helper.py"
    helper.write_text(
        (
            "import sys\n"
            "from pathlib import Path\n"
            "from osgeo import gdal\n"
            "import isce3.core\n"
            "import isce3.cuda.core\n"
            "import isce3.cuda.signal\n"
            "import isce3.io\n"
            "\n"
            "def _copy_raster_to_envi_complex64(src, dst):\n"
            "    dst = Path(dst)\n"
            "    dst.parent.mkdir(parents=True, exist_ok=True)\n"
            "    dst.unlink(missing_ok=True)\n"
            "    Path(f'{dst}.hdr').unlink(missing_ok=True)\n"
            "    src_ds = gdal.Open(str(src), gdal.GA_ReadOnly)\n"
            "    if src_ds is None:\n"
            "        raise RuntimeError(f'failed to open source raster for ENVI copy: {src}')\n"
            "    try:\n"
            "        translated = gdal.Translate(\n"
            "            str(dst),\n"
            "            src_ds,\n"
            "            format='ENVI',\n"
            "            outputType=gdal.GDT_CFloat32,\n"
            "            creationOptions=['INTERLEAVE=BIP'],\n"
            "        )\n"
            "        if translated is None:\n"
            "            raise RuntimeError(f'gdal.Translate failed for {src} -> {dst}')\n"
            "        translated = None\n"
            "    finally:\n"
            "        src_ds = None\n"
            "    return str(dst)\n"
            "\n"
            "master_slc_path, slave_slc_path, gpu_id, block_rows, ifg_path, coh_path, flatten_raster, range_pixel_spacing, wavelength, flatten_starting_range_shift_m = sys.argv[1:]\n"
            "gpu_id = int(gpu_id)\n"
            "block_rows = int(block_rows)\n"
            "flatten_raster = None if flatten_raster == '__NONE__' else flatten_raster\n"
            "range_pixel_spacing = None if range_pixel_spacing == '__NONE__' else float(range_pixel_spacing)\n"
            "wavelength = None if wavelength == '__NONE__' else float(wavelength)\n"
            "flatten_starting_range_shift_m = None if flatten_starting_range_shift_m == '__NONE__' else float(flatten_starting_range_shift_m)\n"
            "\n"
            "input_dir = Path(ifg_path).parent / 'cuda_inputs'\n"
            "input_dir.mkdir(parents=True, exist_ok=True)\n"
            "master_local = _copy_raster_to_envi_complex64(master_slc_path, input_dir / 'master.slc')\n"
            "slave_local = _copy_raster_to_envi_complex64(slave_slc_path, input_dir / 'slave.slc')\n"
            "\n"
            "device = isce3.cuda.core.Device(gpu_id)\n"
            "isce3.cuda.core.set_device(device)\n"
            "master_raster = isce3.io.Raster(str(master_local))\n"
            "slave_raster = isce3.io.Raster(str(slave_local))\n"
            "width = int(master_raster.width)\n"
            "length = int(master_raster.length)\n"
            "if int(slave_raster.width) != width or int(slave_raster.length) != length:\n"
            "    raise RuntimeError(f'CUDA crossmul input dimensions differ: master={length}x{width}, slave={slave_raster.length}x{slave_raster.width}')\n"
            "ifg_raster = isce3.io.Raster(str(ifg_path), width, length, 1, gdal.GDT_CFloat32, 'ENVI')\n"
            "coh_raster = isce3.io.Raster(str(coh_path), width, length, 1, gdal.GDT_Float32, 'ENVI')\n"
            "crossmul = isce3.cuda.signal.Crossmul()\n"
            "crossmul.range_looks = 1\n"
            "crossmul.az_looks = 1\n"
            "crossmul.lines_per_block = block_rows\n"
            "try:\n"
            "    crossmul.set_dopplers(isce3.core.LUT1d(), isce3.core.LUT1d())\n"
            "except Exception:\n"
            "    pass\n"
            "flatten_isce_raster = None\n"
            "if flatten_raster is not None:\n"
            "    if range_pixel_spacing is None or wavelength is None:\n"
            "        raise ValueError('range_pixel_spacing and wavelength are required for CUDA crossmul flattening')\n"
            "    crossmul.range_pixel_spacing = range_pixel_spacing\n"
            "    crossmul.wavelength = wavelength\n"
            "    crossmul.ref_sec_offset_starting_range_shift = float(flatten_starting_range_shift_m or 0.0)\n"
            "    flatten_isce_raster = isce3.io.Raster(str(flatten_raster))\n"
            "crossmul.crossmul(master_raster, slave_raster, ifg_raster, coh_raster, flatten_isce_raster)\n"
        ),
        encoding="utf-8",
    )
    py_compile.compile(str(helper), doraise=True)

    cmd = [
        sys.executable,
        "-X",
        "faulthandler",
        str(helper),
        str(master_slc_path),
        str(slave_slc_path),
        str(gpu_id),
        str(block_rows),
        str(ifg_path),
        str(coh_path),
        flatten_raster if flatten_raster is not None else "__NONE__",
        str(range_pixel_spacing) if range_pixel_spacing is not None else "__NONE__",
        str(wavelength) if wavelength is not None else "__NONE__",
        (
            str(flatten_starting_range_shift_m)
            if flatten_starting_range_shift_m is not None
            else "__NONE__"
        ),
    ]
    process = subprocess.Popen(
        cmd,
        cwd=str(Path(__file__).resolve().parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    start_time = datetime.now(timezone.utc)
    if progress_reporter is not None:
        progress_reporter.running(backend="gpu", detail="crossmul", elapsed=0.0, force=True)
    while True:
        try:
            stdout_text, stderr_text = process.communicate(timeout=1.0)
            break
        except subprocess.TimeoutExpired:
            if progress_reporter is not None:
                progress_reporter.running(
                    backend="gpu",
                    detail="crossmul",
                    elapsed=(datetime.now(timezone.utc) - start_time).total_seconds(),
                )
    class _Completed:
        pass
    result = _Completed()
    result.returncode = process.returncode
    result.stdout = stdout_text
    result.stderr = stderr_text
    if result.returncode != 0:
        stdout_tail = "\n".join(result.stdout.strip().splitlines()[-20:])
        stderr_tail = "\n".join(result.stderr.strip().splitlines()[-40:])
        raise RuntimeError(
            "gpu crossmul subprocess failed "
            f"with exit code {result.returncode}\nstdout:\n{stdout_tail}\nstderr:\n{stderr_tail}"
        )

    ifg_ds = gdal.Open(str(ifg_path), gdal.GA_ReadOnly)
    coh_ds = gdal.Open(str(coh_path), gdal.GA_ReadOnly)
    if ifg_ds is None or coh_ds is None:
        raise RuntimeError("failed to open CUDA crossmul outputs")
    try:
        interferogram = _read_band_array(ifg_ds.GetRasterBand(1), dtype=np.complex64).astype(np.complex64)
        coherence = _read_band_array(coh_ds.GetRasterBand(1), dtype=np.float32).astype(np.float32)
    finally:
        ifg_ds = None
        coh_ds = None
    return interferogram, coherence


def _crossmul_filter_isce3_gpu_subprocess(
    *,
    master_slc_path: str,
    slave_slc_path: str,
    output_dir: Path,
    gpu_id: int,
    block_rows: int,
    flatten_raster: str | None = None,
    range_pixel_spacing: float | None = None,
    wavelength: float | None = None,
    flatten_starting_range_shift_m: float | None = None,
    progress_reporter: _StageProgress | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str | None]:
    output_dir = Path(output_dir)
    p2_dir = stage_dir(output_dir, "p2")
    p2_dir.mkdir(parents=True, exist_ok=True)

    ifg_path = p2_dir / "cuda_interferogram.int"
    coh_path = p2_dir / "cuda_coherence.bin"
    filtered_path = p2_dir / "cuda_filtered_interferogram.int"
    helper = p2_dir / "cuda_crossmul_filter_helper.py"
    helper.write_text(
        (
            "import sys\n"
            "import time\n"
            "import json\n"
            "from pathlib import Path\n"
            "import numpy as np\n"
            "from osgeo import gdal\n"
            "import isce3.core\n"
            "import isce3.cuda.core\n"
            "import isce3.cuda.signal\n"
            "import isce3.io\n"
            "\n"
            "def _copy_raster_to_envi_complex64(src, dst):\n"
            "    dst = Path(dst)\n"
            "    dst.parent.mkdir(parents=True, exist_ok=True)\n"
            "    dst.unlink(missing_ok=True)\n"
            "    Path(f'{dst}.hdr').unlink(missing_ok=True)\n"
            "    src_ds = gdal.Open(str(src), gdal.GA_ReadOnly)\n"
            "    if src_ds is None:\n"
            "        raise RuntimeError(f'failed to open source raster for ENVI copy: {src}')\n"
            "    try:\n"
            "        translated = gdal.Translate(\n"
            "            str(dst), src_ds, format='ENVI', outputType=gdal.GDT_CFloat32,\n"
            "            creationOptions=['INTERLEAVE=BIP'],\n"
            "        )\n"
            "        if translated is None:\n"
            "            raise RuntimeError(f'gdal.Translate failed for {src} -> {dst}')\n"
            "        translated = None\n"
            "    finally:\n"
            "        src_ds = None\n"
            "    return str(dst)\n"
            "\n"
            "def _wait_for_min_file_size(path, expected_size, timeout=5.0, interval=0.1):\n"
            "    deadline = time.time() + timeout\n"
            "    target = Path(path)\n"
            "    while time.time() < deadline:\n"
            "        try:\n"
            "            if target.stat().st_size >= expected_size:\n"
            "                return\n"
            "        except FileNotFoundError:\n"
            "            pass\n"
            "        time.sleep(interval)\n"
            "    raise RuntimeError(\n"
            "        f'output file not fully written in time: {path} expected>={expected_size}'\n"
            "    )\n"
            "\n"
            "def _goldstein_filter_cpu(interferogram, alpha=0.5, window_size=32, step=None):\n"
            "    if step is None:\n"
            "        step = window_size // 2\n"
            "    if step < 1:\n"
            "        step = 1\n"
            "    data = np.asarray(interferogram, dtype=np.complex64)\n"
            "    rows, cols = data.shape\n"
            "    hanning = np.outer(np.hanning(window_size), np.hanning(window_size)).astype(np.complex64)\n"
            "    filtered = np.zeros_like(data, dtype=np.complex64)\n"
            "    weight_sum = np.zeros((rows, cols), dtype=np.float64)\n"
            "    for row_start in range(0, rows - window_size + 1, step):\n"
            "        for col_start in range(0, cols - window_size + 1, step):\n"
            "            window_data = data[row_start:row_start + window_size, col_start:col_start + window_size].copy()\n"
            "            spectrum = np.fft.fft2(window_data * hanning)\n"
            "            psd = np.abs(spectrum) ** 2\n"
            "            weight = np.power(psd + 1.0e-10, alpha / 2.0)\n"
            "            filtered_window = np.fft.ifft2(spectrum * weight)\n"
            "            filtered[row_start:row_start + window_size, col_start:col_start + window_size] += filtered_window\n"
            "            weight_sum[row_start:row_start + window_size, col_start:col_start + window_size] += 1.0\n"
            "    weight_sum[weight_sum == 0] = 1.0\n"
            "    return (filtered / weight_sum).astype(np.complex64)\n"
            "\n"
            "def _goldstein_filter_gpu(interferogram, alpha=0.5, window_size=32, step=None):\n"
            "    try:\n"
            "        import cupy as cp\n"
            "    except Exception as exc:\n"
            "        raise RuntimeError(f'cupy unavailable for combined gpu filter: {exc}')\n"
            "    if step is None:\n"
            "        step = window_size // 2\n"
            "    if step < 1:\n"
            "        step = 1\n"
            "    arr = cp.asarray(np.asarray(interferogram, dtype=np.complex64))\n"
            "    rows, cols = arr.shape\n"
            "    hanning = cp.outer(cp.hanning(window_size), cp.hanning(window_size)).astype(cp.complex64)\n"
            "    filtered = cp.zeros_like(arr, dtype=cp.complex64)\n"
            "    weight_sum = cp.zeros((rows, cols), dtype=cp.float32)\n"
            "    for row_start in range(0, rows - window_size + 1, step):\n"
            "        for col_start in range(0, cols - window_size + 1, step):\n"
            "            window_data = arr[row_start:row_start + window_size, col_start:col_start + window_size]\n"
            "            spectrum = cp.fft.fft2(window_data * hanning)\n"
            "            psd = cp.abs(spectrum) ** 2\n"
            "            weight = cp.power(psd + 1.0e-10, alpha / 2.0)\n"
            "            filtered_window = cp.fft.ifft2(spectrum * weight)\n"
            "            filtered[row_start:row_start + window_size, col_start:col_start + window_size] += filtered_window\n"
            "            weight_sum[row_start:row_start + window_size, col_start:col_start + window_size] += 1.0\n"
            "    weight_sum = cp.where(weight_sum == 0, 1.0, weight_sum)\n"
            "    result = (filtered / weight_sum).astype(cp.complex64)\n"
            "    cp.cuda.runtime.deviceSynchronize()\n"
            "    return cp.asnumpy(result)\n"
            "\n"
            "master_slc_path, slave_slc_path, gpu_id, block_rows, ifg_path, coh_path, filtered_path, flatten_raster, range_pixel_spacing, wavelength, flatten_starting_range_shift_m = sys.argv[1:]\n"
            "gpu_id = int(gpu_id)\n"
            "block_rows = int(block_rows)\n"
            "flatten_raster = None if flatten_raster == '__NONE__' else flatten_raster\n"
            "range_pixel_spacing = None if range_pixel_spacing == '__NONE__' else float(range_pixel_spacing)\n"
            "wavelength = None if wavelength == '__NONE__' else float(wavelength)\n"
            "flatten_starting_range_shift_m = None if flatten_starting_range_shift_m == '__NONE__' else float(flatten_starting_range_shift_m)\n"
            "input_dir = Path(ifg_path).parent / 'cuda_inputs'\n"
            "input_dir.mkdir(parents=True, exist_ok=True)\n"
            "master_local = _copy_raster_to_envi_complex64(master_slc_path, input_dir / 'master.slc')\n"
            "slave_local = _copy_raster_to_envi_complex64(slave_slc_path, input_dir / 'slave.slc')\n"
            "device = isce3.cuda.core.Device(gpu_id)\n"
            "isce3.cuda.core.set_device(device)\n"
            "master_raster = isce3.io.Raster(str(master_local))\n"
            "slave_raster = isce3.io.Raster(str(slave_local))\n"
            "width = int(master_raster.width)\n"
            "length = int(master_raster.length)\n"
            "if int(slave_raster.width) != width or int(slave_raster.length) != length:\n"
            "    raise RuntimeError(f'CUDA crossmul input dimensions differ: master={length}x{width}, slave={slave_raster.length}x{slave_raster.width}')\n"
            "ifg_raster = isce3.io.Raster(str(ifg_path), width, length, 1, gdal.GDT_CFloat32, 'ENVI')\n"
            "coh_raster = isce3.io.Raster(str(coh_path), width, length, 1, gdal.GDT_Float32, 'ENVI')\n"
            "crossmul = isce3.cuda.signal.Crossmul()\n"
            "crossmul.range_looks = 1\n"
            "crossmul.az_looks = 1\n"
            "crossmul.lines_per_block = block_rows\n"
            "try:\n"
            "    crossmul.set_dopplers(isce3.core.LUT1d(), isce3.core.LUT1d())\n"
            "except Exception:\n"
            "    pass\n"
            "flatten_isce_raster = None\n"
            "if flatten_raster is not None:\n"
            "    if range_pixel_spacing is None or wavelength is None:\n"
            "        raise ValueError('range_pixel_spacing and wavelength are required for CUDA crossmul flattening')\n"
            "    crossmul.range_pixel_spacing = range_pixel_spacing\n"
            "    crossmul.wavelength = wavelength\n"
            "    crossmul.ref_sec_offset_starting_range_shift = float(flatten_starting_range_shift_m or 0.0)\n"
            "    flatten_isce_raster = isce3.io.Raster(str(flatten_raster))\n"
            "crossmul.crossmul(master_raster, slave_raster, ifg_raster, coh_raster, flatten_isce_raster)\n"
            "for raster in (ifg_raster, coh_raster, master_raster, slave_raster, flatten_isce_raster):\n"
            "    if raster is not None:\n"
            "        close_dataset = getattr(raster, 'close_dataset', None)\n"
            "        if callable(close_dataset):\n"
            "            close_dataset()\n"
            "ifg_raster = None\n"
            "coh_raster = None\n"
            "master_raster = None\n"
            "slave_raster = None\n"
            "flatten_isce_raster = None\n"
            "_wait_for_min_file_size(ifg_path, width * length * np.dtype(np.complex64).itemsize)\n"
            "_wait_for_min_file_size(coh_path, width * length * np.dtype(np.float32).itemsize)\n"
            "ifg_ds = gdal.Open(str(ifg_path), gdal.GA_ReadOnly)\n"
            "if ifg_ds is None:\n"
            "    raise RuntimeError('failed to open CUDA crossmul interferogram for filtering')\n"
            "try:\n"
            "    interferogram = ifg_ds.GetRasterBand(1).ReadAsArray().astype(np.complex64)\n"
            "finally:\n"
            "    ifg_ds = None\n"
            "filter_backend = 'gpu'\n"
            "filter_fallback_reason = None\n"
            "try:\n"
            "    filtered = _goldstein_filter_gpu(interferogram)\n"
            "except Exception as exc:\n"
            "    filter_backend = 'cpu'\n"
            "    filter_fallback_reason = str(exc)\n"
            "    sys.stderr.write(f'[p2 helper] GPU goldstein failed, falling back to CPU: {exc}\\n')\n"
            "    filtered = _goldstein_filter_cpu(interferogram)\n"
            "print('D2SAR_FILTER_INFO=' + json.dumps({'backend': filter_backend, 'fallback': filter_fallback_reason}), flush=True)\n"
            "filtered_ds = gdal.GetDriverByName('ENVI').Create(str(filtered_path), width, length, 1, gdal.GDT_CFloat32)\n"
            "if filtered_ds is None:\n"
            "    raise RuntimeError(f'failed to create filtered interferogram raster: {filtered_path}')\n"
            "filtered_ds.GetRasterBand(1).WriteArray(filtered.astype(np.complex64))\n"
            "filtered_ds.FlushCache()\n"
            "filtered_ds = None\n"
        ),
        encoding="utf-8",
    )
    py_compile.compile(str(helper), doraise=True)

    cmd = [
        sys.executable,
        "-X",
        "faulthandler",
        str(helper),
        str(master_slc_path),
        str(slave_slc_path),
        str(gpu_id),
        str(block_rows),
        str(ifg_path),
        str(coh_path),
        str(filtered_path),
        flatten_raster if flatten_raster is not None else "__NONE__",
        str(range_pixel_spacing) if range_pixel_spacing is not None else "__NONE__",
        str(wavelength) if wavelength is not None else "__NONE__",
        (
            str(flatten_starting_range_shift_m)
            if flatten_starting_range_shift_m is not None
            else "__NONE__"
        ),
    ]
    process = subprocess.Popen(
        cmd,
        cwd=str(Path(__file__).resolve().parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    start_time = datetime.now(timezone.utc)
    if progress_reporter is not None:
        progress_reporter.running(
            backend="gpu",
            detail="crossmul+goldstein_filter",
            elapsed=0.0,
            force=True,
        )
    while True:
        try:
            stdout_text, stderr_text = process.communicate(timeout=1.0)
            break
        except subprocess.TimeoutExpired:
            if progress_reporter is not None:
                progress_reporter.running(
                    backend="gpu",
                    detail="crossmul+goldstein_filter",
                    elapsed=(datetime.now(timezone.utc) - start_time).total_seconds(),
                )
    class _Completed:
        pass
    result = _Completed()
    result.returncode = process.returncode
    result.stdout = stdout_text
    result.stderr = stderr_text
    if result.returncode != 0:
        stdout_tail = "\n".join(result.stdout.strip().splitlines()[-20:])
        stderr_tail = "\n".join(result.stderr.strip().splitlines()[-40:])
        raise RuntimeError(
            "gpu crossmul+filter subprocess failed "
            f"with exit code {result.returncode}\nstdout:\n{stdout_tail}\nstderr:\n{stderr_tail}"
        )

    helper_filter_backend = "gpu"
    helper_filter_fallback_reason = None
    for line in reversed(result.stdout.splitlines()):
        if not line.startswith("D2SAR_FILTER_INFO="):
            continue
        try:
            payload = json.loads(line.split("=", 1)[1])
        except Exception:
            break
        backend = str(payload.get("backend") or "gpu").strip().lower()
        helper_filter_backend = "cpu" if backend == "cpu" else "gpu"
        fallback_reason = payload.get("fallback")
        if fallback_reason:
            helper_filter_fallback_reason = str(fallback_reason)
        break

    ifg_ds = gdal.Open(str(ifg_path), gdal.GA_ReadOnly)
    coh_ds = gdal.Open(str(coh_path), gdal.GA_ReadOnly)
    filtered_ds = gdal.Open(str(filtered_path), gdal.GA_ReadOnly)
    if ifg_ds is None or coh_ds is None or filtered_ds is None:
        raise RuntimeError("failed to open CUDA crossmul/filter outputs")
    try:
        interferogram = _read_band_array(ifg_ds.GetRasterBand(1), dtype=np.complex64).astype(np.complex64)
        coherence = _read_band_array(coh_ds.GetRasterBand(1), dtype=np.float32).astype(np.float32)
        filtered_interferogram = _read_band_array(
            filtered_ds.GetRasterBand(1),
            dtype=np.complex64,
        ).astype(np.complex64)
    finally:
        ifg_ds = None
        coh_ds = None
        filtered_ds = None
    return (
        interferogram,
        coherence,
        filtered_interferogram,
        helper_filter_backend,
        helper_filter_fallback_reason,
    )


def _run_crossmul_and_filter(
    *,
    master_slc_path: str,
    slave_slc_path: str,
    use_gpu: bool,
    gpu_id: int,
    output_dir: Path,
    block_rows: int = ISCE3_CROSSMUL_LINES_PER_BLOCK_DEFAULT,
    flatten_raster: str | None = None,
    flatten_mask_raster: str | None = None,
    range_pixel_spacing: float | None = None,
    wavelength: float | None = None,
    flatten_starting_range_shift_m: float | None = None,
    progress_reporter: _StageProgress | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, str], str | None]:
    fallback_reasons: list[str] = []
    if use_gpu:
        try:
            (
                interferogram,
                coherence,
                filtered_interferogram,
                helper_filter_backend,
                helper_filter_fallback_reason,
            ) = _run_crossmul_filter_gpu_pipeline(
                master_slc_path=master_slc_path,
                slave_slc_path=slave_slc_path,
                output_dir=output_dir,
                gpu_id=gpu_id,
                block_rows=block_rows,
                flatten_raster=flatten_raster,
                range_pixel_spacing=range_pixel_spacing,
                wavelength=wavelength,
                flatten_starting_range_shift_m=flatten_starting_range_shift_m,
                progress_reporter=progress_reporter,
            )
            combined_fallback_reason = None
            if helper_filter_backend != "gpu" and helper_filter_fallback_reason:
                combined_fallback_reason = (
                    "combined_gpu_pipeline goldstein_filter->cpu: "
                    f"{helper_filter_fallback_reason}"
                )
            return (
                interferogram,
                coherence,
                filtered_interferogram,
                {"crossmul": "gpu", "goldstein_filter": helper_filter_backend},
                combined_fallback_reason,
            )
        except Exception as exc:
            fallback_reasons.append(f"combined_gpu_pipeline: {exc}")

    interferogram, coherence, crossmul_backend, crossmul_fallback_reason = _run_crossmul(
        master_slc_path=master_slc_path,
        slave_slc_path=slave_slc_path,
        use_gpu=use_gpu,
        gpu_id=gpu_id,
        output_dir=output_dir,
        block_rows=block_rows,
        flatten_raster=flatten_raster,
        flatten_mask_raster=flatten_mask_raster,
        range_pixel_spacing=range_pixel_spacing,
        wavelength=wavelength,
        flatten_starting_range_shift_m=flatten_starting_range_shift_m,
        progress_reporter=progress_reporter,
    )
    if crossmul_fallback_reason:
        fallback_reasons.append(f"crossmul: {crossmul_fallback_reason}")

    filtered_interferogram, filter_backend, filter_fallback_reason = _run_goldstein_filter(
        interferogram=interferogram,
        use_gpu=use_gpu,
        gpu_id=gpu_id,
        progress_reporter=progress_reporter,
    )
    if filter_fallback_reason:
        fallback_reasons.append(f"goldstein_filter: {filter_fallback_reason}")

    return (
        interferogram,
        coherence,
        filtered_interferogram,
        {"crossmul": crossmul_backend, "goldstein_filter": filter_backend},
        "; ".join(fallback_reasons) if fallback_reasons else None,
    )


def _run_crossmul(
    *,
    master_slc_path: str,
    slave_slc_path: str,
    use_gpu: bool,
    gpu_id: int,
    output_dir: Path,
    block_rows: int = ISCE3_CROSSMUL_LINES_PER_BLOCK_DEFAULT,
    flatten_raster: str | None = None,
    flatten_mask_raster: str | None = None,
    range_pixel_spacing: float | None = None,
    wavelength: float | None = None,
    flatten_starting_range_shift_m: float | None = None,
    progress_reporter: _StageProgress | None = None,
) -> tuple[np.ndarray, np.ndarray, str, str | None]:
    fallback_reason = None
    if use_gpu:
        try:
            # Invalid offsets are already zeroed in flatten_raster, so the CUDA path can
            # ignore the validity mask and still preserve "no-flatten" behavior there.
            interferogram, coherence = _crossmul_isce3_gpu(
                master_slc_path=master_slc_path,
                slave_slc_path=slave_slc_path,
                output_dir=output_dir,
                gpu_id=gpu_id,
                block_rows=block_rows,
                flatten_raster=flatten_raster,
                range_pixel_spacing=range_pixel_spacing,
                wavelength=wavelength,
                flatten_starting_range_shift_m=flatten_starting_range_shift_m,
                progress_reporter=progress_reporter,
            )
            return interferogram, coherence, "gpu", None
        except Exception as exc:
            fallback_reason = str(exc)

    interferogram, coherence = _run_crossmul_cpu(
        master_slc_path=master_slc_path,
        slave_slc_path=slave_slc_path,
        flatten_raster=flatten_raster,
        flatten_mask_raster=flatten_mask_raster,
        range_pixel_spacing=range_pixel_spacing,
        wavelength=wavelength,
        flatten_starting_range_shift_m=flatten_starting_range_shift_m,
        block_rows=block_rows,
        progress_reporter=progress_reporter,
    )
    return interferogram, coherence, "cpu", fallback_reason


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
    use_gpu = False
    if gpu_mode != "cpu":
        use_gpu = _default_gpu_check(True if gpu_mode == "gpu" else None, gpu_id)

    progress = _StageProgress("p2")
    try:
        interferogram, coherence, filtered_interferogram, backends, fallback_reason = _run_crossmul_and_filter(
            master_slc_path=master_slc,
            slave_slc_path=registered_slave_slc,
            use_gpu=use_gpu,
            gpu_id=gpu_id,
            output_dir=context.pair_dir,
            block_rows=block_rows,
            flatten_raster=flatten_options.get("flatten_raster"),
            flatten_mask_raster=flatten_options.get("flatten_mask_raster"),
            range_pixel_spacing=flatten_options.get("range_pixel_spacing"),
            wavelength=flatten_options.get("wavelength"),
            flatten_starting_range_shift_m=flatten_options.get("flatten_starting_range_shift_m"),
            progress_reporter=progress,
        )
    finally:
        progress.close()
    stage_backend = (
        "gpu"
        if backends.get("crossmul") == "gpu" and backends.get("goldstein_filter") == "gpu"
        else "cpu"
    )
    result = {
        "interferogram": _save_stage_array(context.pair_dir, "p2", "interferogram", interferogram),
        "filtered_interferogram": _save_stage_array(
            context.pair_dir, "p2", "filtered_interferogram", filtered_interferogram
        ),
        "coherence": _save_stage_array(context.pair_dir, "p2", "coherence", coherence),
        "crossmul_backend": backends.get("crossmul"),
        "goldstein_filter_backend": backends.get("goldstein_filter"),
        "wrapped_phase_radar_png": _write_radar_wrapped_phase_png(
            interferogram,
            context.pair_dir / "wrapped_phase_radar.png",
        ),
    }
    _add_flatten_outputs(result, flatten_options)
    _write_stage_outputs_record(
        output_dir=context.pair_dir,
        stage="p2",
        master_manifest_path=context.master_manifest_path,
        slave_manifest_path=context.slave_manifest_path,
        backend_used=stage_backend,
        output_files=result,
        fallback_reason=fallback_reason,
    )
    return result, stage_backend, fallback_reason


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
    config_overrides: dict | None = None,
) -> np.ndarray:
    import snaphu

    scratch_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(SNAPHU_DEFAULTS)
    if config_overrides:
        cfg.update(config_overrides)

    # Dynamic tiling: if image > 5000x5000, use 2x2 tiles with 600 overlap
    rows, cols = interferogram.shape
    if rows > 5000 and cols > 5000:
        cfg["ntiles"] = (2, 2)
        cfg["tile_overlap"] = (600, 600)
        print(f"[SNAPHU] Large image detected ({rows}x{cols}), using 2x2 tiles with 600 overlap")

    unw = np.zeros(interferogram.shape, dtype=np.float32)
    conncomp = np.zeros(interferogram.shape, dtype=np.uint32)
    snaphu.unwrap(
        interferogram.astype(np.complex64),
        coherence.astype(np.float32),
        cfg["nlooks"],
        unw=unw,
        conncomp=conncomp,
        cost=cfg["cost_mode"],
        init=cfg["initialization_method"],
        min_conncomp_frac=cfg["min_conncomp_frac"],
        phase_grad_window=cfg["phase_grad_window"],
        ntiles=cfg["ntiles"],
        tile_overlap=cfg["tile_overlap"],
        nproc=cfg["nproc"],
        tile_cost_thresh=cfg["tile_cost_thresh"],
        min_region_size=cfg["min_region_size"],
        single_tile_reoptimize=cfg["single_tile_reoptimize"],
        regrow_conncomps=cfg["regrow_conncomps"],
        scratchdir=scratch_dir,
        delete_scratch=True,
    )
    if not np.any(np.isfinite(unw)):
        raise RuntimeError("SNAPHU produced no finite pixels")
    return unw.astype(np.float32)


def _unwrap_with_snaphu_profiles(
    interferogram: np.ndarray,
    coherence: np.ndarray,
    scratch_dir: Path,
) -> tuple[np.ndarray, str | None]:
    errors: list[str] = []
    for profile_name, overrides in SNAPHU_RETRY_PROFILES:
        try:
            result = _unwrap_with_snaphu(
                interferogram,
                coherence,
                scratch_dir / profile_name,
                config_overrides=overrides,
            )
            fallback_reason = None
            if profile_name != "default":
                fallback_reason = f"SNAPHU profile={profile_name}"
            return result, fallback_reason
        except Exception as exc:
            errors.append(f"{profile_name}: {exc}")
    raise RuntimeError(f"SNAPHU failed across profiles: {'; '.join(errors)}")


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

    fallback_reason = None
    with tempfile.TemporaryDirectory(prefix="strip_insar2_unwrap_", dir=str(context.pair_dir)) as tmpdir:
        scratch = Path(tmpdir)
        if unwrap_method != "icu":
            raise ValueError(f"unsupported unwrap method: {unwrap_method}")
        try:
            unwrapped_phase = _unwrap_with_icu(interferogram, coherence, scratch / "icu")
        except Exception as icu_exc:
            try:
                unwrapped_phase, snaphu_fallback = _unwrap_with_snaphu_profiles(
                    interferogram,
                    coherence,
                    scratch / "snaphu",
                )
                fallback_reason = f"ICU failed, fell back to SNAPHU: {icu_exc}"
                if snaphu_fallback:
                    fallback_reason += f"; {snaphu_fallback}"
            except Exception as snaphu_exc:
                raise RuntimeError(
                    f"ICU failed ({icu_exc}); SNAPHU fallback also failed ({snaphu_exc})"
                ) from snaphu_exc

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
        fallback_reason=fallback_reason,
    )
    return output_files, "cpu", fallback_reason


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
    block_rows: int = ISCE3_GEOMETRY_LINES_PER_BLOCK_DEFAULT,
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
    progress = _StageProgress("p5")

    def _gpu():
        progress.running(backend="gpu", detail="rdr2geo/topo", elapsed=0.0, force=True)
        return append_topo_coordinates_hdf(
            str(context.master_manifest_path),
            context.resolved_dem,
            output_h5,
            block_rows=block_rows,
            orbit_interp=context.orbit_interp,
            use_gpu=True,
            gpu_id=gpu_id,
            progress_reporter=progress,
        )

    def _cpu():
        return append_topo_coordinates_hdf(
            str(context.master_manifest_path),
            context.resolved_dem,
            output_h5,
            block_rows=block_rows,
            orbit_interp=context.orbit_interp,
            use_gpu=False,
            progress_reporter=progress,
        )

    try:
        _, backend_used, fallback_reason = run_stage_with_fallback(
            stage_name="coordinates",
            gpu_mode=gpu_mode,
            gpu_id=gpu_id,
            gpu_runner=_gpu,
            cpu_runner=_cpu,
        )
    finally:
        progress.close()
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
    block_rows: int | None = None,
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

    geometry_block_rows, crossmul_block_rows = _resolve_stage_block_rows(block_rows)

    _log_stage_status("p0", "START")
    _, backend_used, fallback_reason = run_geo2rdr_stage(
        context,
        gpu_mode=gpu_mode,
        gpu_id=gpu_id,
        block_rows=geometry_block_rows,
    )
    stage_backends["geo2rdr"] = backend_used
    if fallback_reason:
        fallback_reasons["geo2rdr"] = fallback_reason
    _log_stage_status("p0", "DONE", backend_used=backend_used, fallback_reason=fallback_reason)

    _log_stage_status("p1", "START")
    p1_outputs, backend_used, fallback_reason = run_resample_stage(
        context,
        gpu_mode=gpu_mode,
        gpu_id=gpu_id,
    )
    stage_backends["resample"] = backend_used
    if fallback_reason:
        fallback_reasons["resample"] = fallback_reason
    _log_stage_status("p1", "DONE", backend_used=backend_used, fallback_reason=fallback_reason)

    _log_stage_status("p2", "START")
    _, backend_used, fallback_reason = run_crossmul_stage(
        context,
        gpu_mode=gpu_mode,
        gpu_id=gpu_id,
        block_rows=crossmul_block_rows,
    )
    stage_backends["crossmul"] = backend_used
    if fallback_reason:
        fallback_reasons["crossmul"] = fallback_reason
    _log_stage_status("p2", "DONE", backend_used=backend_used, fallback_reason=fallback_reason)

    _log_stage_status("p3", "START")
    _, backend_used, fallback_reason = run_unwrap_stage(
        context,
        unwrap_method=unwrap_method,
        block_rows=geometry_block_rows,
    )
    stage_backends["unwrap"] = backend_used
    if fallback_reason:
        fallback_reasons["unwrap"] = fallback_reason
    _log_stage_status("p3", "DONE", backend_used=backend_used, fallback_reason=fallback_reason)

    _log_stage_status("p4", "START")
    _, backend_used, fallback_reason = run_los_stage(context)
    stage_backends["los"] = backend_used
    if fallback_reason:
        fallback_reasons["los"] = fallback_reason
    _log_stage_status("p4", "DONE", backend_used=backend_used, fallback_reason=fallback_reason)

    _log_stage_status("p5", "START")
    output_h5, backend_used, fallback_reason = write_primary_product(
        context,
        p1_outputs=p1_outputs,
        gpu_mode=gpu_mode,
        gpu_id=gpu_id,
        block_rows=geometry_block_rows,
        unwrap_method=unwrap_method,
    )
    stage_backends["product"] = backend_used
    if fallback_reason:
        fallback_reasons["product"] = fallback_reason
    _log_stage_status("p5", "DONE", backend_used=backend_used, fallback_reason=fallback_reason)

    _log_stage_status("p6", "START")
    exported = export_insar_products(
        input_h5=output_h5,
        output_paths=context.output_paths,
        resolution_meters=resolution_meters,
        block_rows=min(geometry_block_rows, 64),
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
    _log_stage_status("p6", "DONE", backend_used="cpu")

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
    parser.add_argument("--unwrap-method", choices=["icu"], default="icu")
    parser.add_argument("--resolution", type=float, default=20.0)
    parser.add_argument("--block-rows", type=int, default=None)
    parser.add_argument("--dem-path")
    parser.add_argument("--dem-cache-dir")
    parser.add_argument("--dem-margin-deg", type=float, default=0.2)
    parser.add_argument("--no-kml", action="store_true")
    args = parser.parse_args()

    try:
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
    except Exception as exc:
        show_traceback = str(os.environ.get("D2SAR_STRIP_INSAR2_TRACEBACK", "0")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if show_traceback:
            traceback.print_exc()
        else:
            print(f"[strip_insar2] ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from None
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
