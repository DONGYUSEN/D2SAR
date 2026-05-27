"""strip_insar_io — I/O and shared utilities for the strip_insar pipeline."""
from __future__ import annotations

from common_processing import _gdal_dtype_to_numpy, _numpy_dtype_to_gdal, _read_band_array, _write_band_array, gps_to_datetime, point2epsg, load_scene_corners_with_fallback
from datetime import datetime, timedelta, timezone
from contextlib import contextmanager
import importlib
import json
import logging
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import h5py
import numpy as np
from osgeo import gdal, osr
from PIL import Image

from .strip_insar_types import (
    GEO2RDR_OFFSET_NODATA, GEO2RDR_OFFSET_INVALID_LOW, NISAR_OFFSET_INVALID_VALUE,
    ISCE3_GEOMETRY_LINES_PER_BLOCK_DEFAULT, ISCE3_CROSSMUL_LINES_PER_BLOCK_DEFAULT,
    STAGE_SEQUENCE, STAGE_DIR_NAMES, STAGE_LOG_LABELS,
    PairContext,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Extracted from strip_insar.py L200-342
# First function: _default_rubbersheet_cfg
# ---------------------------------------------------------------------------


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



# ---------------------------------------------------------------------------
# Extracted from strip_insar.py L343-541
# First function: validate_stage_name
# ---------------------------------------------------------------------------
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
    detail: str | None = None,
    label_override: str | None = None,
    backend_used: str | None = None,
    fallback_reason: str | None = None,
) -> None:
    label = str(label_override) if label_override else STAGE_LOG_LABELS.get(stage, stage)
    message = f"[{utc_now_iso()}] [{stage}] {status} {label}"
    if detail:
        message += f" {detail}"
    if backend_used:
        message += f" backend={backend_used}"
    if fallback_reason:
        message += f" fallback={fallback_reason}"
    print(message, flush=True)


def _log_prepare_info(section: str, message: str) -> None:
    print(f"[{utc_now_iso()}] [prep] {section} {message}", flush=True)


def _format_normalize_start_message(
    precheck: dict,
    *,
    master_acquisition: dict | None = None,
    slave_acquisition: dict | None = None,
) -> str:
    checks = precheck.get("checks") or {}
    reasons = []
    if (checks.get("prf") or {}).get("severity") == "warn":
        reasons.append("PRF 差异超标")
    if (checks.get("radar_grid") or {}).get("severity") == "warn":
        reasons.append("雷达网格有差异")
    if (checks.get("doppler") or {}).get("severity") == "warn":
        reasons.append("Doppler有差异")
    reason_text = "、".join(reasons) if reasons else "预检查要求"
    prf_text = ""
    if master_acquisition is not None and slave_acquisition is not None:
        master_prf = master_acquisition.get("prf")
        slave_prf = slave_acquisition.get("prf")
        if master_prf is not None and slave_prf is not None:
            prf_text = f" slave PRF {slave_prf} -> master PRF {master_prf};"
    return (
        f"---- 发现 {reason_text}，"
        f"开始归一化处理: {prf_text} "
    )


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

# ---------------------------------------------------------------------------
# Extracted from strip_insar.py L542-584
# First function: resolve_manifest_data_path
# ---------------------------------------------------------------------------
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


def resolve_manifest_data_path(manifest_path: str | Path, entry) -> str | None:

# ---------------------------------------------------------------------------
# Extracted from strip_insar.py L585-803
# First function: choose_orbit_interp
# ---------------------------------------------------------------------------
    def _remap_legacy_absolute_path(path_str: str) -> str:
        if not path_str.startswith("/"):
            return path_str
        candidates: list[Path] = []
        if path_str.startswith("/results/"):
            candidates.append(Path.cwd() / path_str.lstrip("/"))
        if path_str.startswith("/work/results/"):
            candidates.append(Path.cwd() / path_str[len("/work/") :].lstrip("/"))
        if path_str.startswith("/tmp/"):
            candidates.append(Path("/home/ysdong/Temp") / path_str[5:])
        if path_str.startswith("/temp/"):
            candidates.append(Path("/home/ysdong/Temp") / path_str[6:])
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate.exists():
                return str(candidate)
        return path_str

    def _resolve_archive_path(raw_path: str | Path) -> str:
        def _find_archive_candidate(path_str: str) -> str:
            path_obj = Path(path_str)
            candidates: list[Path] = [path_obj]
            name = path_obj.name
            if name:
                candidates.extend(
                    [
                        manifest_dir / name,
                        manifest_dir / "temp" / name,
                        manifest_dir.parent / name,
                        manifest_dir.parent / "temp" / name,
                        Path.cwd() / name,
                        Path("/results") / name,
                        Path("/work") / name,
                        Path("/tmp") / name,
                        Path("/temp") / name,
                    ]
                )
            for candidate in candidates:
                try:
                    if candidate.exists():
                        return str(candidate.resolve())
                except Exception:
                    continue
            return path_str

        path = Path(raw_path)
        resolved = path if path.is_absolute() else (manifest_dir / path)
        resolved = resolved.resolve()
        resolved_str = str(resolved)
        if not resolved.exists():
            remapped = _remap_legacy_absolute_path(resolved_str)
            if remapped != resolved_str:
                return remapped
            candidate = _find_archive_candidate(resolved_str)
            if candidate != resolved_str:
                return candidate
        return resolved_str

    def _split_archive_member(path: str, *, storage: str) -> tuple[str, str] | None:
        low = path.lower()
        markers = (".zip/",) if storage == "zip" else (".tar.gz/", ".tgz/", ".tar/")
        for marker in markers:
            idx = low.find(marker)
            if idx < 0:
                continue
            archive_end = idx + len(marker) - 1
            archive = path[:archive_end]
            member = path[archive_end + 1 :]
            if archive and member:
                return archive, member
        return None

    def _normalize_vsi_entry(vsi_path: str) -> str:
        if vsi_path.startswith("/vsitar/"):
            prefix = "/vsitar/"
            storage = "tar"
        elif vsi_path.startswith("/vsizip/"):
            prefix = "/vsizip/"
            storage = "zip"
        else:
            return vsi_path
        remainder = vsi_path[len(prefix) :]
        split = _split_archive_member(remainder, storage=storage)
        if split is None:
            return vsi_path
        archive, member = split
        archive_abs = _resolve_archive_path(archive)
        member_path = str(member).lstrip("/").replace("\\", "/")
        return f"{prefix}{archive_abs}/{member_path}"

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
        if storage == "tar" and member:
            archive = str(resolved.resolve()).replace("\\", "/")
            if not Path(archive).exists():
                archive = _remap_legacy_absolute_path(archive).replace("\\", "/")
            member_path = str(member).lstrip("/").replace("\\", "/")
            return f"/vsitar/{archive}/{member_path}"
        if storage == "zip" and member:
            archive = str(resolved.resolve()).replace("\\", "/")
            if not Path(archive).exists():
                archive = _remap_legacy_absolute_path(archive).replace("\\", "/")
            member_path = str(member).lstrip("/").replace("\\", "/")
            return f"/vsizip/{archive}/{member_path}"
        resolved_str = str(resolved.resolve())
        if not Path(resolved_str).exists():
            resolved_str = _remap_legacy_absolute_path(resolved_str)
        return resolved_str
    entry_str = str(entry)
    if entry_str.startswith("/vsitar/") or entry_str.startswith("/vsizip/"):
        return _normalize_vsi_entry(entry_str)
    entry_path = Path(entry_str)
    resolved = entry_path if entry_path.is_absolute() else (manifest_dir / entry_path)
    resolved_str = str(resolved.resolve())
    if not Path(resolved_str).exists():
        resolved_str = _remap_legacy_absolute_path(resolved_str)
    return resolved_str


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


def _normalize_looks(value: int | float | str | None, name: str) -> int:
    looks = int(value or 1)
    if looks < 1:
        raise ValueError(f"{name} must be >= 1")
    return looks


def _multilook_output_shape(shape: tuple[int, int], azimuth_looks: int, range_looks: int) -> tuple[int, int]:
    azimuth_looks = _normalize_looks(azimuth_looks, "azimuth_looks")
    range_looks = _normalize_looks(range_looks, "range_looks")
    rows = int(shape[0]) // azimuth_looks
    cols = int(shape[1]) // range_looks
    if rows < 1 or cols < 1:
        raise ValueError(
            f"multilook factors too large for shape {shape}: azimuth_looks={azimuth_looks}, range_looks={range_looks}"
        )
    return rows, cols


def _multilook_mean(arr: np.ndarray, azimuth_looks: int, range_looks: int) -> np.ndarray:
    azimuth_looks = _normalize_looks(azimuth_looks, "azimuth_looks")
    range_looks = _normalize_looks(range_looks, "range_looks")
    if azimuth_looks == 1 and range_looks == 1:
        return np.asarray(arr)
    rows_out, cols_out = _multilook_output_shape(arr.shape[:2], azimuth_looks, range_looks)
    trim_rows = rows_out * azimuth_looks
    trim_cols = cols_out * range_looks
    trimmed = np.asarray(arr)[:trim_rows, :trim_cols]
    reshaped = trimmed.reshape(rows_out, azimuth_looks, cols_out, range_looks, *trimmed.shape[2:])
    return reshaped.mean(axis=(1, 3))


def _round_up_half(value: float) -> float:
    return math.ceil(float(value) * 2.0) / 2.0


def _compute_default_resolution(
    manifest_path: str | Path,
    *,
    range_looks: int = 1,
    azimuth_looks: int = 1,
) -> float:
    """Compute default output resolution from radargrid metadata.

    Full-resolution default:
        2 * max(groundRangeResolution, azimuthResolution), rounded up to 0.5 m.

    Multilook default:
        max(groundRangeResolution * range_looks, azimuthResolution * azimuth_looks),
        rounded up to 0.5 m.
    """
    range_looks = _normalize_looks(range_looks, "range_looks")
    azimuth_looks = _normalize_looks(azimuth_looks, "azimuth_looks")

    manifest = load_manifest(manifest_path)
    radargrid_path = resolve_manifest_metadata_path(manifest_path, manifest, "radargrid")
    with open(radargrid_path, encoding="utf-8") as f:
        radargrid = json.load(f)

    range_res = float(radargrid.get("groundRangeResolution", 0.0) or 0.0)
    azimuth_res = float(radargrid.get("azimuthResolution", 0.0) or 0.0)
    max_res = max(range_res, azimuth_res)

    if range_looks == 1 and azimuth_looks == 1:
        return _round_up_half(2.0 * max_res)

    range_res_ml = range_res * float(range_looks)
    azimuth_res_ml = azimuth_res * float(azimuth_looks)
    return _round_up_half(max(range_res_ml, azimuth_res_ml))


def choose_orbit_interp(orbit_json: dict, acquisition_json: dict | None = None) -> str:

# ---------------------------------------------------------------------------
# Extracted from strip_insar.py L804-1004
# First function: _copy_raster
# ---------------------------------------------------------------------------
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
            if "position" in sv and isinstance(sv["position"], dict):
                p = sv["position"]
                pos.append([p["x"], p["y"], p["z"]])
            else:
                pos.append([sv["posX"], sv["posY"], sv["posZ"]])
            if "velocity" in sv and isinstance(sv["velocity"], dict):
                v = sv["velocity"]
                vel.append([v["x"], v["y"], v["z"]])
            else:
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


def _manifest_sensor_name(manifest: dict | None, acquisition_data: dict | None = None) -> str:
    manifest_sensor = ""
    if isinstance(manifest, dict):
        manifest_sensor = str(manifest.get("sensor", "")).strip().lower()
    if manifest_sensor:
        return manifest_sensor
    if isinstance(acquisition_data, dict):
        return str(acquisition_data.get("source", "")).strip().lower()
    return ""


def _choose_context_orbit_interp(
    master_manifest: dict,
    slave_manifest: dict,
    master_orbit_data: dict,
    master_acq_data: dict,
    slave_acq_data: dict | None = None,
) -> str:
    sensors = {
        _manifest_sensor_name(master_manifest, master_acq_data),
        _manifest_sensor_name(slave_manifest, slave_acq_data),
    }
    if "lutan" in sensors:
        return "Legendre"
    return choose_orbit_interp(master_orbit_data, master_acq_data)


def _build_coregistration_doppler_lut():
    import isce3.core

    # Follow NISAR strip RSLC registration assumptions: use zero Doppler
    # through geo2rdr and resample stages.
    return isce3.core.LUT2d()


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
        if "position" in sv and isinstance(sv["position"], dict):
            position = sv["position"]
            pos = np.array([position["x"], position["y"], position["z"]], dtype=np.float64)
        else:
            pos = np.array([sv["posX"], sv["posY"], sv["posZ"]], dtype=np.float64)
        if "velocity" in sv and isinstance(sv["velocity"], dict):
            velocity = sv["velocity"]
            vel = np.array([velocity["x"], velocity["y"], velocity["z"]], dtype=np.float64)
        else:
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

# ---------------------------------------------------------------------------
# Extracted from strip_insar.py L1005-1246
# First function: _load_processing_metadata
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Extracted from strip_insar.py L1241-1481
# First function: load_pair_context
# ---------------------------------------------------------------------------
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


def _load_rdr2geo_inputs_from_manifest(manifest_path: str | Path) -> tuple[dict, dict, dict]:
    manifest_path = Path(manifest_path)
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
    return orbit_data, acquisition_data, radargrid_data


def _resolve_dem_path(
    manifest_path: Path,
    manifest: dict,
    corners,
    dem_path: str | None,
    dem_cache_dir: str | None,
    dem_margin_deg: float,
) -> str:
    if dem_path is not None:
        return str(Path(dem_path).resolve())

    manifest_dem = (
        manifest.get("dem", {}).get("path")
        if isinstance(manifest.get("dem"), dict)
        else None
    )
    if manifest_dem is not None:
        resolved_manifest_dem = resolve_manifest_data_path(manifest_path, manifest_dem)
        if resolved_manifest_dem is not None:
            return resolved_manifest_dem

    # Fallback: auto-resolve DEM from scene corners when manifest has no dem.path.
    if corners:
        try:
            from dem_manager import resolve_dem_for_scene

            cache_dir = Path(dem_cache_dir) if dem_cache_dir else (manifest_path.parent / "dem")
            cache_dir.mkdir(parents=True, exist_ok=True)
            auto_dem_path, _ = resolve_dem_for_scene(
                corners,
                dem_path=None,
                output_dir=str(cache_dir),
                margin_deg=float(dem_margin_deg),
            )
            return str(auto_dem_path)
        except Exception as exc:
            raise FileNotFoundError(
                f"failed to auto-resolve DEM (no dem.path in manifest): {exc}"
            ) from exc

    raise FileNotFoundError(
        "DEM path is required: pass --dem or provide manifest.dem.path; auto DEM requires valid scene corners"
    )


def _default_gpu_check(gpu_requested: bool | None, gpu_id: int) -> bool:
    """Check GPU availability via shared gpu_utils, then ISCE3 gpu_check."""
    try:
        from scripts.gpu_utils import check_cuda_available
        if check_cuda_available(gpu_id):
            return True
    except Exception:
        pass
    try:
        from isce3.core.gpu_check import use_gpu
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
    product_dir = output_dir / f"{pair_name}_product"
    return {
        "interferogram_h5": str(output_dir / f"{pair_name}_insar.h5"),
        "avg_amplitude_tif": str(product_dir / f"{pair_name}_avg_amplitude_utm_geocoded.tif"),
        "avg_amplitude_png": str(product_dir / f"{pair_name}_avg_amplitude_utm_geocoded.png"),
        "avg_amplitude_kml": str(product_dir / f"{pair_name}_avg_amplitude_utm_geocoded.kml"),
        "interferogram_tif": str(product_dir / f"{pair_name}_interferogram_utm_geocoded.tif"),
        "interferogram_png": str(product_dir / f"{pair_name}_interferogram_wrapped_phase_utm_geocoded.png"),
        "interferogram_kml": str(product_dir / f"{pair_name}_interferogram_wrapped_phase_utm_geocoded.kml"),
        "filtered_interferogram_tif": str(product_dir / f"{pair_name}_filtered_interferogram_utm_geocoded.tif"),
        "filtered_interferogram_png": str(
            product_dir / f"{pair_name}_filtered_interferogram_wrapped_phase_utm_geocoded.png"
        ),
        "filtered_interferogram_kml": str(
            product_dir / f"{pair_name}_filtered_interferogram_wrapped_phase_utm_geocoded.kml"
        ),
        "coherence_tif": str(product_dir / f"{pair_name}_coherence_utm_geocoded.tif"),
        "coherence_png": str(product_dir / f"{pair_name}_coherence_utm_geocoded.png"),
        "coherence_kml": str(product_dir / f"{pair_name}_coherence_utm_geocoded.kml"),
        "unwrapped_phase_tif": str(product_dir / f"{pair_name}_unwrapped_phase_utm_geocoded.tif"),
        "unwrapped_phase_png": str(product_dir / f"{pair_name}_unwrapped_phase_utm_geocoded.png"),
        "unwrapped_phase_kml": str(product_dir / f"{pair_name}_unwrapped_phase_utm_geocoded.kml"),
        "los_displacement_tif": str(product_dir / f"{pair_name}_los_displacement_utm_geocoded.tif"),
        "los_displacement_png": str(product_dir / f"{pair_name}_los_displacement_utm_geocoded.png"),
        "los_displacement_kml": str(product_dir / f"{pair_name}_los_displacement_utm_geocoded.kml"),
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

# ---------------------------------------------------------------------------
# Extracted from strip_insar.py L1556-1913
# First function: make_raster
# ---------------------------------------------------------------------------
    *,
    output_dir: Path,
    stage: str,
    master_manifest_path: str | Path,
    slave_manifest_path: str | Path,
    backend_used: str,
    output_files: dict,
    fallback_reason: str | None = None,
    processing_options: dict | None = None,
) -> None:
    upstream = {
        "p0": ["prep"],
        "p1": ["p0"],
        "p2": ["p1"],
        "p3": ["p2"],
        "p4": ["p3"],
    }
    upstream_dependencies = upstream.get(stage, [])
    record = {
        "stage": stage,
        "input_manifests": {
            "master": str(master_manifest_path),
            "slave": str(slave_manifest_path),
        },
        "effective_crop": {},
        "backend_used": backend_used,
        "upstream_stage_dependencies": upstream_dependencies,
        "upstream_stage_tokens": {
            dep: (load_stage_record(output_dir, dep) or {}).get("end_time")
            for dep in upstream_dependencies
        },
        "output_files": output_files,
        "start_time": utc_now_iso(),
        "end_time": utc_now_iso(),
        "success": True,
        "fallback_reason": fallback_reason,
        "processing_options": dict(processing_options or {}),
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
    processing_options: dict | None = None,
) -> None:
    upstream_dependencies = upstream_stage_dependencies or []
    record = {
        "stage": stage,
        "input_manifests": {
            "master": str(master_manifest_path),
            "slave": str(slave_manifest_path),
        },
        "effective_crop": {},
        "backend_used": backend_used,
        "upstream_stage_dependencies": upstream_dependencies,
        "upstream_stage_tokens": {
            dep: (load_stage_record(output_dir, dep) or {}).get("end_time")
            for dep in upstream_dependencies
        },
        "output_files": output_files,
        "start_time": utc_now_iso(),
        "end_time": utc_now_iso(),
        "success": True,
        "fallback_reason": fallback_reason,
        "processing_options": dict(processing_options or {}),
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
    expected_processing_options: dict | None = None,
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
    if expected_processing_options is not None:
        processing_options = record.get("processing_options") or {}
        if dict(processing_options) != dict(expected_processing_options):
            return None
    upstream_dependencies = record.get("upstream_stage_dependencies") or []
    stored_tokens = record.get("upstream_stage_tokens") or {}
    if upstream_dependencies and stored_tokens:
        current_tokens = {
            dep: (load_stage_record(output_dir, dep) or {}).get("end_time")
            for dep in upstream_dependencies
        }
        if current_tokens != stored_tokens:
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


def _gdal_path_exists(path_str: str | Path) -> bool:
    """检测 GDAL 支持的路径是否存在，包括 /vsizip/ 和 /vsitar/ 路径。

    Args:
        path_str: 文件路径，可能是普通路径、/vsizip/... 或 /vsitar/... 格式

    Returns:
        True 如果路径可被 GDAL 打开，False otherwise
    """
    try:
        ds = gdal.Open(str(path_str), gdal.GA_ReadOnly)
        if ds is not None:
            ds.FlushCache()
            ds = None
            return True
        return False
    except Exception:
        return False


def _load_float_offset_raster(path: str | Path | None) -> np.ndarray | None:
    if not path:
        return None
    if not _gdal_path_exists(path):
        return None
    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    if ds is None:
        return None
    try:
        arr = _read_band_array(ds.GetRasterBand(1), dtype=np.float64).astype(np.float64)
    finally:
        ds = None
    valid = np.isfinite(arr)
    valid &= arr != GEO2RDR_OFFSET_NODATA
    valid &= arr > GEO2RDR_OFFSET_INVALID_LOW
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    out[valid] = arr[valid]
    return out


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
        from scripts.gpu_utils import init_cuda_device
        gpu_info = init_cuda_device(gpu_id, gpu_mode="auto")
        use_gpu = gpu_info.available
        if use_gpu:
            import isce3.cuda.geometry
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

# ---------------------------------------------------------------------------
# Extracted from strip_insar.py L1914-2201
# First function: accumulate_utm_grid
# ---------------------------------------------------------------------------
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


def append_topo_coordinates_hdf_from_vrt(
    topo_vrt_path: str | Path,
    output_h5: str | Path,
    *,
    block_rows: int = ISCE3_GEOMETRY_LINES_PER_BLOCK_DEFAULT,
    range_looks: int = 1,
    azimuth_looks: int = 1,
) -> str:
    topo_vrt_path = Path(topo_vrt_path)
    output_h5 = Path(output_h5)
    ds = gdal.Open(str(topo_vrt_path), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"failed to open topo VRT: {topo_vrt_path}")
    try:
        width = int(ds.RasterXSize)
        length = int(ds.RasterYSize)
        looked_length, looked_width = _multilook_output_shape((length, width), azimuth_looks, range_looks)
        trim_length = looked_length * azimuth_looks
        trim_width = looked_width * range_looks

        lon_band = ds.GetRasterBand(1)
        lat_band = ds.GetRasterBand(2)
        hgt_band = ds.GetRasterBand(3)

        output_block_rows = max(1, int(block_rows) // azimuth_looks)
        input_block_rows = output_block_rows * azimuth_looks
        with h5py.File(output_h5, "a") as f:
            for name in ("longitude", "latitude", "height"):
                if name in f:
                    del f[name]
            chunk_rows = min(max(1, output_block_rows), looked_length)
            chunk_cols = min(1024, looked_width)
            d_lon = f.create_dataset("longitude", shape=(looked_length, looked_width), dtype="f8", chunks=(chunk_rows, chunk_cols), compression="gzip", shuffle=True)
            d_lat = f.create_dataset("latitude", shape=(looked_length, looked_width), dtype="f8", chunks=(chunk_rows, chunk_cols), compression="gzip", shuffle=True)
            d_hgt = f.create_dataset("height", shape=(looked_length, looked_width), dtype="f8", chunks=(chunk_rows, chunk_cols), compression="gzip", shuffle=True)
            f.attrs["coordinate_system"] = "EPSG:4326"
            f.attrs["longitude_units"] = "degrees_east"
            f.attrs["latitude_units"] = "degrees_north"
            f.attrs["height_units"] = "meters"
            f.attrs["coordinate_source"] = "p0_master_topo_vrt_multilooked"

            for out_row0 in range(0, looked_length, output_block_rows):
                row0 = out_row0 * azimuth_looks
                rows = min(input_block_rows, trim_length - row0)
                out_rows = rows // azimuth_looks
                lon_block = _read_band_array(lon_band, 0, row0, trim_width, rows).astype(np.float64)
                lat_block = _read_band_array(lat_band, 0, row0, trim_width, rows).astype(np.float64)
                hgt_block = _read_band_array(hgt_band, 0, row0, trim_width, rows).astype(np.float64)
                d_lon[out_row0:out_row0 + out_rows, :] = _multilook_mean(lon_block, azimuth_looks, range_looks)
                d_lat[out_row0:out_row0 + out_rows, :] = _multilook_mean(lat_block, azimuth_looks, range_looks)
                d_hgt[out_row0:out_row0 + out_rows, :] = _multilook_mean(hgt_block, azimuth_looks, range_looks)
        return str(output_h5)
    finally:
        ds = None


def _append_multilooked_topo_coordinates_hdf(
    *,
    context: PairContext,
    output_h5: str | Path,
    block_rows: int,
    range_looks: int,
    azimuth_looks: int,
    use_gpu: bool,
    gpu_id: int,
) -> str:
    scratch_root = stage_dir(context.pair_dir, "p5") / "multilook_topo_tmp"
    scratch_root.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="d2sar_ml_rdr2geo_", dir=str(scratch_root)))
    try:
        orbit_data, acquisition_data, radargrid_data = _load_rdr2geo_inputs_from_manifest(
            context.master_manifest_path
        )
        topo_vrt = _run_rdr2geo_topo(
            orbit_data=orbit_data,
            acquisition_data=acquisition_data,
            radargrid_data=radargrid_data,
            dem_path=context.resolved_dem,
            orbit_interp=context.orbit_interp,
            use_gpu=use_gpu,
            gpu_id=gpu_id,
            output_dir=workdir,
            block_rows=block_rows,
        )
        return append_topo_coordinates_hdf_from_vrt(
            topo_vrt,
            output_h5,
            block_rows=block_rows,
            range_looks=range_looks,
            azimuth_looks=azimuth_looks,
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        if scratch_root.exists() and not any(scratch_root.iterdir()):
            scratch_root.rmdir()


def append_utm_coordinates_hdf(output_h5: str, manifest_path: str, block_rows: int = 32) -> str:
    output_h5 = Path(output_h5)
    manifest_path = Path(manifest_path)
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    with h5py.File(output_h5, "a") as f:
        lon_ds = f["longitude"]
        lat_ds = f["latitude"]
        valid = np.isfinite(lon_ds[()]) & np.isfinite(lat_ds[()])
        if np.any(valid):
            center_lon = float(np.nanmean(lon_ds[()][valid]))
            center_lat = float(np.nanmean(lat_ds[()][valid]))
        else:
            corners = load_scene_corners_with_fallback(manifest_path, manifest)
            if not corners:
                raise RuntimeError("cannot derive UTM coordinates without valid lon/lat or scene corners")
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

# ---------------------------------------------------------------------------
# Extracted from strip_insar.py L2346-2765
# First function: _construct_doppler_if_possible
# ---------------------------------------------------------------------------
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
    if dataset_name == "avg_amplitude":
        source_display = _stretch_source_grid_for_png(input_h5, dataset_name, stretch_percent=5.0)
        out, meta = _accumulate_source_grid_to_utm(
            input_h5, source_display, target_width, target_height, block_rows
        )
        img = np.zeros(out.shape, dtype=np.uint8)
        valid = np.isfinite(out)
        if np.any(valid):
            img[valid] = np.rint(np.clip(out[valid], 0.0, 255.0)).astype(np.uint8)
        Image.fromarray(img, mode="L").save(output_png)
        _write_png_georef_sidecars(output_png, meta)
        return str(output_png)

    out, meta = accumulate_utm_grid(input_h5, dataset_name, target_width, target_height, block_rows)
    img = np.zeros(out.shape, dtype=np.uint8)
    valid = np.isfinite(out)
    if np.any(valid):
        vals = out[valid]
        p2 = np.percentile(vals, 2)
        p98 = np.percentile(vals, 98)
        scaled = np.clip((vals - p2) / (p98 - p2 + 1.0e-9), 0.0, 1.0)
        img[valid] = (scaled * 255).astype(np.uint8)
    Image.fromarray(img, mode="L").save(output_png)
    _write_png_georef_sidecars(output_png, meta)
    return str(output_png)


def _write_png_georef_sidecars(output_png: str | Path, meta: dict) -> None:
    output_png = Path(output_png)
    x_res = float(meta["x_res"])
    y_res = float(meta["y_res"])
    x_min = float(meta["x_min"])
    y_max = float(meta["y_max"])
    utm_epsg = int(meta["utm_epsg"])

    worldfile = output_png.with_suffix(".pgw")
    worldfile.write_text(
        "\n".join(
            [
                str(x_res),
                "0.0",
                "0.0",
                str(-y_res),
                str(x_min + 0.5 * x_res),
                str(y_max - 0.5 * y_res),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(utm_epsg)
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    output_png.with_suffix(".prj").write_text(srs.ExportToWkt(), encoding="utf-8")


def _record_png_sidecars(exported: dict[str, str], png_key: str, png_path: str | Path) -> None:
    png_path = Path(png_path)
    pgw_path = png_path.with_suffix(".pgw")
    prj_path = png_path.with_suffix(".prj")
    base_key = png_key.removesuffix("_png")
    if pgw_path.exists():
        exported[f"{base_key}_pgw"] = str(pgw_path)
    if prj_path.exists():
        exported[f"{base_key}_prj"] = str(prj_path)


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
    phase, meta = _accumulate_wrapped_phase_grid(input_h5, dataset_name, target_width, target_height, block_rows)
    rgba = np.zeros((*phase.shape, 4), dtype=np.uint8)
    valid = np.isfinite(phase)
    if np.any(valid):
        hue = ((phase[valid] + np.pi) / (2.0 * np.pi)).astype(np.float64)
        colors = np.array(
            [
                colorsys.hsv_to_rgb(float(h), 1.0, 1.0)
                for h in hue
            ]
        )
        rgba[valid, :3] = (colors * 255.0).astype(np.uint8)
        rgba[valid, 3] = 255
    Image.fromarray(rgba, mode="RGBA").save(output_png)
    _write_png_georef_sidecars(output_png, meta)
    return str(output_png)


def write_unwrapped_phase_png(
    input_h5: str,
    output_png: str,
    dataset_name: str,
    target_width: int,
    target_height: int | None = None,
    block_rows: int = 64,
) -> str:
    phase, meta = accumulate_utm_grid(input_h5, dataset_name, target_width, target_height, block_rows)
    phase = np.mod(np.asarray(phase, dtype=np.float64) + np.pi, 2.0 * np.pi) - np.pi
    rgba = np.zeros((*phase.shape, 4), dtype=np.uint8)
    valid = np.isfinite(phase)
    if np.any(valid):
        hue = ((phase[valid] + np.pi) / (2.0 * np.pi)).astype(np.float64)
        colors = np.array(
            [
                colorsys.hsv_to_rgb(float(h), 1.0, 1.0)
                for h in hue
            ]
        )
        rgba[valid, :3] = (colors * 255.0).astype(np.uint8)
        rgba[valid, 3] = 255
    Image.fromarray(rgba, mode="RGBA").save(output_png)
    _write_png_georef_sidecars(output_png, meta)
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

    for key, value in output_paths.items():
        if key.endswith(("_tif", "_png", "_kml")):
            Path(value).parent.mkdir(parents=True, exist_ok=True)

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
        if dataset_name == "unwrapped_phase":
            write_unwrapped_phase_png(
                input_h5,
                png_path,
                dataset_name,
                target_width,
                target_height,
                block_rows,
            )
        else:
            write_geocoded_png(input_h5, png_path, dataset_name, target_width, target_height, block_rows)
        exported[tif_key] = tif_path
        exported[png_key] = png_path
        _record_png_sidecars(exported, png_key, png_path)
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
        _record_png_sidecars(exported, png_key, png_path)
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

# ---------------------------------------------------------------------------
# Extracted from strip_insar.py L2765-2825
# First function: _burst_grid_range_start
# ---------------------------------------------------------------------------
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


def _burst_indices_from_infos(bursts: list[Any]) -> set[int]:
    indices = [int(getattr(burst, "burst_index", 0) or 0) for burst in bursts]
    if not indices:
        return set()
    if min(indices) == 0:
        return {idx + 1 for idx in indices}
    return {idx for idx in indices if idx > 0}


def _burst_grid_range_start(grid: dict[str, Any]) -> float: