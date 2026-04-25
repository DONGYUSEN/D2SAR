"""Unified Tianyi/Lutan strip InSAR processor with GPU-first fallback.

Processing chain:
    P0: Geo2Rdr coarse coregistration
    P1: PyCuAmpcor dense matching (GPU only)
    P2: Crossmul interferogram + coherence (GPU/CPU)
    P3: Phase unwrapping (ICU/SNAPHU, CPU only)
    P4: Geocode products (GPU/CPU)
    P5: Multi-look output + GeoTIFF + PNG
"""

from __future__ import annotations

import json
import os
import py_compile
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np
from osgeo import gdal, osr
from PIL import Image

from common_processing import (
    gps_to_datetime,
    choose_orbit_interp,
    load_scene_corners_with_fallback,
    construct_orbit,
    construct_doppler_lut2d,
    construct_radar_grid,
    append_topo_coordinates_hdf,
    append_utm_coordinates_hdf,
    compute_utm_output_shape,
    accumulate_utm_grid,
    prepare_display_grid,
    write_geocoded_geotiff,
    write_geocoded_png,
    write_wrapped_phase_geotiff,
    write_wrapped_phase_png,
    resolve_manifest_data_path,
    resolve_manifest_metadata_path,
    resolve_dem_for_scene,
    dem_covers_scene_corners,
    fetch_dem,
    point2epsg,
)
from insar_crop import normalize_crop_request
from insar_precheck import run_compatibility_precheck
from insar_preprocess import build_preprocess_plan
from insar_filtering import goldstein_filter
from insar_registration import (
    run_pycuampcor_dense_offsets,
    run_cpu_dense_offsets,
    run_coarse_resamp_isce3_v2,
    write_registration_outputs,
    _ensure_local_tiff,
    _copy_raster_to_envi_complex64,
    _estimate_offset_mean_from_raster,
    _select_cpu_dense_match_plan,
    _write_varying_gross_offset_file,
    _raster_shape,
)
from insar_subset import build_cropped_manifest
from insar_stage_cache import (
    load_stage_record,
    mark_stage_success,
    resolve_requested_stages,
    stage_succeeded,
    stage_dir,
    utc_now_iso,
    write_stage_record,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_SENSORS = {"tianyi", "lutan"}

# Per-stage backend map for hybrid GPU pipeline
HYBRID_GPU_STAGE_BACKENDS = {
    "geo2rdr": "gpu",
    "pycuampcor": "gpu",
    "crossmul": "cpu",
    "unwrap": "cpu",  # No GPU unwrapping in ISCE3
    "geocode": "gpu",
    "output": "cpu",
}

CPU_STAGE_BACKENDS = {k: "cpu" for k in HYBRID_GPU_STAGE_BACKENDS}

EXPERIMENTAL_GPU_CROSSMUL_ENV = "D2SAR_ENABLE_EXPERIMENTAL_GPU_CROSSMUL"
ICU_TILE_SIZE_ENV = "D2SAR_ICU_TILE_SIZE"
ICU_TILE_OVERLAP_ENV = "D2SAR_ICU_TILE_OVERLAP"
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


# ---------------------------------------------------------------------------
# Phase Unwrapper Interface
# ---------------------------------------------------------------------------


class PhaseUnwrapper(ABC):
    """Abstract base for phase unwrapping methods."""

    @abstractmethod
    def unwrap(
        self,
        interferogram: np.ndarray,
        coherence: np.ndarray,
        radar_grid,
        orbit,
        dem_path: str,
        output_dir: str,
        block_rows: int = 256,
    ) -> np.ndarray:
        """Unwrap phase.

        Args:
            interferogram: complex64 array (rows, cols)
            coherence: float32 array (rows, cols), values in [0, 1]
            radar_grid: isce3.product.RadarGridParameters
            orbit: isce3.core.Orbit
            dem_path: path to DEM raster
            output_dir: working directory for temp files
            block_rows: block size for I/O

        Returns:
            unwrapped_phase: float32 array (rows, cols) in radians
        """
        pass


class ICUUnwrapper(PhaseUnwrapper):
    """ISCE3 ICU phase unwrapper (CPU only)."""

    def unwrap(
        self,
        interferogram: np.ndarray,
        coherence: np.ndarray,
        radar_grid,
        orbit,
        dem_path: str,
        output_dir: str,
        block_rows: int = 256,
    ) -> np.ndarray:
        tmp_dir = Path(output_dir)
        rows, cols = interferogram.shape
        tile_size = self._icu_tile_size()
        if rows > tile_size or cols > tile_size:
            print(
                f"[STRIP_INSAR] Using tiled ICU for large raster "
                f"({rows}x{cols}, tile_size={tile_size})"
            )
            return self._unwrap_tiled(
                interferogram,
                coherence,
                tmp_dir / "icu_tiles",
                tile_size=tile_size,
            )

        full_dir = tmp_dir / "icu_full"
        full_dir.mkdir(parents=True, exist_ok=True)
        try:
            return self._unwrap_once(
                interferogram,
                coherence,
                full_dir,
            )
        except RuntimeError as exc:
            if "ICU unwrapping produced" not in str(exc):
                raise
            print(f"[STRIP_INSAR] Full-frame ICU output invalid ({exc}); retrying tiled ICU")
            return self._unwrap_tiled(
                interferogram,
                coherence,
                tmp_dir / "icu_tiles",
                tile_size=tile_size,
            )

    def _unwrap_once(
        self,
        interferogram: np.ndarray,
        coherence: np.ndarray,
        output_dir: Path,
    ) -> np.ndarray:
        import isce3.io
        import isce3.unwrap
        from osgeo import gdal

        rows, cols = interferogram.shape
        tmp_dir = Path(output_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        int_path = tmp_dir / "unwrap_input.tif"
        coh_path = tmp_dir / "unwrap_coh.tif"
        unw_path = tmp_dir / "unw.int"
        ccl_path = tmp_dir / "ccl.cc"

        drv = gdal.GetDriverByName("GTiff")

        int_ds = drv.Create(str(int_path), cols, rows, 1, gdal.GDT_CFloat32)
        _write_band_array(int_ds.GetRasterBand(1), interferogram.astype(np.complex64))
        int_ds = None

        coh_ds = drv.Create(str(coh_path), cols, rows, 1, gdal.GDT_Float32)
        _write_band_array(coh_ds.GetRasterBand(1), coherence.astype(np.float32))
        coh_ds = None

        igram = isce3.io.Raster(str(int_path))
        corr = isce3.io.Raster(str(coh_path))
        unwRaster = isce3.io.Raster(
            str(unw_path), cols, rows, 1, gdal.GDT_Float32, "GTiff"
        )
        cclRaster = isce3.io.Raster(
            str(ccl_path), cols, rows, 1, gdal.GDT_Byte, "GTiff"
        )

        unwrapper = isce3.unwrap.ICU()

        unwrapper.unwrap(unwRaster, cclRaster, igram, corr)

        igram.close_dataset()
        corr.close_dataset()
        unwRaster.close_dataset()
        cclRaster.close_dataset()

        result_ds = gdal.Open(str(unw_path))
        result = _read_band_array(result_ds.GetRasterBand(1), dtype=np.float32).astype(np.float32)
        result_ds = None
        ccl_ds = gdal.Open(str(ccl_path))
        connected_components = _read_band_array(ccl_ds.GetRasterBand(1), dtype=np.uint8)
        ccl_ds = None
        if connected_components is None or not np.any(connected_components):
            raise RuntimeError("ICU unwrapping produced no connected component labels")
        # Zero-valued unwrapped phase is valid after flattening in quiet areas.
        # Use connected-component labels, not phase magnitude, to determine validity.
        result = result.astype(np.float32, copy=False)
        result[connected_components == 0] = np.nan
        finite_result = result[np.isfinite(result)]
        if finite_result.size == 0:
            raise RuntimeError("ICU unwrapping produced an empty unwrapped phase")

        return result

    def _unwrap_tiled(
        self,
        interferogram: np.ndarray,
        coherence: np.ndarray,
        output_dir: Path,
        *,
        tile_size: int | None = None,
    ) -> np.ndarray:
        rows, cols = interferogram.shape
        tile_size = int(tile_size or self._icu_tile_size())
        overlap = max(0, int(os.environ.get(ICU_TILE_OVERLAP_ENV, "128")))
        if overlap * 2 >= tile_size:
            overlap = tile_size // 4
        stride = max(1, tile_size - overlap)

        output_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = np.full((rows, cols), np.nan, dtype=np.float32)
        valid = np.zeros((rows, cols), dtype=bool)
        tile_index = 0
        for row0 in range(0, rows, stride):
            for col0 in range(0, cols, stride):
                row1 = min(rows, row0 + tile_size)
                col1 = min(cols, col0 + tile_size)
                if row1 - row0 < 32 or col1 - col0 < 32:
                    continue
                tile_index += 1
                tile_dir = output_dir / f"tile_{tile_index:05d}_r{row0}_c{col0}"
                try:
                    tile_unw = self._unwrap_once(
                        interferogram[row0:row1, col0:col1],
                        coherence[row0:row1, col0:col1],
                        tile_dir,
                    )
                except RuntimeError as exc:
                    print(
                        f"[STRIP_INSAR] ICU tile skipped r={row0}:{row1} c={col0}:{col1}: {exc}"
                    )
                    continue

                tile_valid = np.isfinite(tile_unw)
                if not np.any(tile_valid):
                    continue

                existing = valid[row0:row1, col0:col1]
                if np.any(existing & tile_valid):
                    diff = unwrapped[row0:row1, col0:col1][existing & tile_valid] - tile_unw[
                        existing & tile_valid
                    ]
                    diff = diff[np.isfinite(diff)]
                    if diff.size:
                        tile_unw = tile_unw + float(np.median(diff))

                write_mask = tile_valid & ~existing
                update = unwrapped[row0:row1, col0:col1]
                update[write_mask] = tile_unw[write_mask]
                unwrapped[row0:row1, col0:col1] = update
                valid[row0:row1, col0:col1] |= tile_valid

        if not np.any(valid):
            raise RuntimeError("tiled ICU unwrapping produced no valid pixels")
        unwrapped[~valid] = 0.0
        return unwrapped.astype(np.float32)

    def _icu_tile_size(self) -> int:
        return max(256, int(os.environ.get(ICU_TILE_SIZE_ENV, "2048")))


def _create_unwrapper(unwrap_method: str) -> PhaseUnwrapper:
    """Create a PhaseUnwrapper instance from method name.

    Parameters
    ----------
    unwrap_method : str
        One of "icu" or "snaphu".

    Returns
    -------
    PhaseUnwrapper
        Configured unwrapper instance.

    Raises
    ------
    ValueError
        If unwrap_method is not recognized.
    """
    if unwrap_method == "icu":
        return ICUUnwrapper()
    elif unwrap_method == "snaphu":
        return SNAPHUUnwrapper()
    else:
        raise ValueError(f"Unknown unwrap_method: {unwrap_method}")


class SNAPHUUnwrapper(PhaseUnwrapper):
    """SNAPHU phase unwrapping (CPU only, requires external SNAPHU)."""

    def unwrap(
        self,
        interferogram: np.ndarray,
        coherence: np.ndarray,
        radar_grid,
        orbit,
        dem_path: str,
        output_dir: str,
        block_rows: int = 256,
    ) -> np.ndarray:
        import shutil

        # Check SNAPHU availability
        if shutil.which("snaphu") is None:
            raise RuntimeError(
                "SNAPHU not found in PATH. Install SNAPHU or use --unwrap-method icu"
            )

        tmp_dir = Path(output_dir)
        rows, cols = interferogram.shape

        # Write interferogram (phase + amplitude in separate files for SNAPHU)
        phase = np.angle(interferogram).astype(np.float32)
        amp = np.abs(interferogram).astype(np.float32)

        int_path = tmp_dir / "snaphu_input_phase.tif"
        coh_path = tmp_dir / "snaphu_coherence.tif"
        unwrap_path = tmp_dir / "snaphu_unwrapped.tif"

        # Write wrapped phase
        from osgeo import gdal

        drv = gdal.GetDriverByName("GTiff")
        ds = drv.Create(str(int_path), cols, rows, 1, gdal.GDT_Float32)
        _write_band_array(ds.GetRasterBand(1), phase)
        ds = None

        # Write coherence
        ds = drv.Create(str(coh_path), cols, rows, 1, gdal.GDT_Float32)
        _write_band_array(ds.GetRasterBand(1), coherence.astype(np.float32))
        ds = None

        # Run SNAPHU
        cfg_file = tmp_dir / "snaphu.conf"
        cfg_file.write_text(
            f"OUTFILE {unwrap_path}\n"
            f"LINENUMBER {rows}\n"
            f"WIDTH {cols}\n"
            f"IGNOREFILE true\n"
            f"CORRFILE {coh_path}\n"
        )

        subprocess.run(
            ["snaphu", "-f", str(cfg_file), str(int_path)],
            check=True,
            cwd=str(tmp_dir),
        )

        # Read back
        ds = gdal.Open(str(unwrap_path))
        result = _read_band_array(ds.GetRasterBand(1), dtype=np.float32).astype(np.float32)
        ds = None

        return result


# ---------------------------------------------------------------------------
# Manifest utilities
# ---------------------------------------------------------------------------


def load_manifest(manifest_path: str | Path) -> dict:
    manifest_path = Path(manifest_path)
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def detect_sensor_from_manifest(manifest_path: str | Path) -> str:
    manifest = load_manifest(manifest_path)
    sensor = str(manifest.get("sensor", "")).strip().lower()
    if sensor not in SUPPORTED_SENSORS:
        raise ValueError(f"Unsupported sensor '{sensor}' in manifest")
    return sensor


def build_output_paths(output_dir: str | Path) -> dict[str, str]:
    output_dir = Path(output_dir)
    return {
        "interferogram_h5": str(output_dir / "interferogram_fullres.h5"),
        "interferogram_tif": str(output_dir / "interferogram_utm_geocoded.tif"),
        "coherence_tif": str(output_dir / "coherence_utm_geocoded.tif"),
        "unwrapped_phase_tif": str(output_dir / "unwrapped_phase_utm_geocoded.tif"),
        "los_displacement_tif": str(output_dir / "los_displacement_utm_geocoded.tif"),
        "interferogram_png": str(output_dir / "interferogram_utm_geocoded.png"),
        "filtered_interferogram_png": str(
            output_dir / "filtered_interferogram_utm_geocoded.png"
        ),
    }


# ---------------------------------------------------------------------------
# GPU memory utilities (mirrored from strip_rtc.py)
# ---------------------------------------------------------------------------


def _default_gpu_check(gpu_requested: bool | None, gpu_id: int) -> bool:
    from isce3.core.gpu_check import use_gpu

    return bool(use_gpu(gpu_requested, gpu_id))


def select_processing_backend(
    gpu_mode: str, gpu_id: int, gpu_check=None
) -> tuple[str, str]:
    gpu_mode = str(gpu_mode).strip().lower()
    if gpu_mode not in {"auto", "cpu", "gpu"}:
        raise ValueError(f"Unsupported gpu_mode '{gpu_mode}'")

    if gpu_mode == "cpu":
        return "cpu", "CPU mode forced by user"

    gpu_check = gpu_check or _default_gpu_check

    try:
        if gpu_mode == "gpu":
            if not gpu_check(True, gpu_id):
                raise ValueError("GPU requested but unavailable")
            return "gpu", f"GPU {gpu_id} explicitly requested and available"

        if gpu_check(None, gpu_id):
            return "gpu", f"GPU {gpu_id} available for strip processing"
    except Exception as exc:
        return "cpu", f"GPU unavailable, fallback to CPU: {exc}"

    return "cpu", "GPU unavailable, fallback to CPU"


def query_gpu_memory_info(gpu_id: int) -> dict[str, int] | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_id}",
                "--query-gpu=memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None

    line = (
        result.stdout.strip().splitlines()[0].strip() if result.stdout.strip() else ""
    )
    if not line:
        return None
    try:
        total_mib_str, free_mib_str = [p.strip() for p in line.split(",", 1)]
        mib = 1024 * 1024
        return {
            "total_bytes": int(float(total_mib_str) * mib),
            "free_bytes": int(float(free_mib_str) * mib),
        }
    except Exception:
        return None


def _is_experimental_gpu_crossmul_enabled() -> bool:
    value = os.getenv(EXPERIMENTAL_GPU_CROSSMUL_ENV, "").strip().lower()
    if value == "":
        return False
    return value in {"1", "true", "yes", "on"}


def choose_gpu_topo_block_rows(
    width: int,
    default_block_rows: int,
    memory_info: dict[str, int] | None,
    min_block_rows: int = 64,
    max_block_rows: int = 1024,
) -> tuple[int, str]:
    """Choose GPU block rows for geometry stages (Geo2Rdr, Crossmul)."""
    if memory_info is None:
        return (
            default_block_rows,
            "Default block_rows used; GPU memory info unavailable",
        )
    total_bytes = int(memory_info.get("total_bytes", 0))
    free_bytes = int(memory_info.get("free_bytes", 0))
    if width <= 0 or total_bytes <= 0 or free_bytes <= 0:
        return default_block_rows, "Default block_rows used; GPU memory info invalid"

    reserve_bytes = max(1024**3, int(0.20 * total_bytes))
    budget_bytes = min(
        int(0.40 * total_bytes),
        int(0.65 * free_bytes),
        free_bytes - reserve_bytes,
    )
    if budget_bytes <= 0:
        return (
            default_block_rows,
            "Default block_rows used; GPU memory budget unavailable",
        )

    # Crossmul + Geo2Rdr: ~40 bytes per cell (complex64 + float32)
    bytes_per_row_lower_bound = width * 40
    safety_factor = 4
    estimated_rows = budget_bytes // (bytes_per_row_lower_bound * safety_factor)
    estimated_rows = max(32, (estimated_rows // 32) * 32)
    block_rows_out = max(min_block_rows, min(max_block_rows, int(estimated_rows)))
    return (
        block_rows_out,
        f"Adaptive GPU block_rows={block_rows_out} from VRAM budget {budget_bytes} bytes",
    )


def _is_gpu_memory_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        token in text
        for token in (
            "out of memory",
            "cudaerrormemoryallocation",
            "memory allocation",
            "cuda memory",
            "cuda out of memory",
            "std::bad_alloc",
        )
    )


def _halve_block_rows(block_rows: int, min_block_rows: int = 64) -> int:
    halved = max(min_block_rows, block_rows // 2)
    halved = max(min_block_rows, (halved // 32) * 32)
    return halved


# ---------------------------------------------------------------------------
# Wavelength from radar grid
# ---------------------------------------------------------------------------


def get_wavelength(acquisition_json: dict) -> float:
    return 299792458.0 / acquisition_json["centerFrequency"]


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
    if not np.isfinite(shift):
        return 0.0
    return shift


# ---------------------------------------------------------------------------
# LOS displacement
# ---------------------------------------------------------------------------


def compute_los_displacement(
    unwrapped_phase: np.ndarray, wavelength: float
) -> np.ndarray:
    """Convert unwrapped phase (radians) to LOS displacement (meters).

    LOS displacement = phase * wavelength / (4 * pi)
    Positive = motion toward satellite.
    """
    return (unwrapped_phase * wavelength / (4.0 * np.pi)).astype(np.float32)


# ---------------------------------------------------------------------------
# HDF5 write helpers
# ---------------------------------------------------------------------------


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
    radar_grid,
    block_rows: int = 256,
    extra_attrs: dict | None = None,
    filtered_interferogram: np.ndarray | None = None,
    crop_request: dict | None = None,
) -> str:
    """Write InSAR products to HDF5."""
    # Compute average amplitude from master + slave
    master_amp = _compute_slc_amplitude(master_slc_path)
    slave_amp = _compute_slc_amplitude(slave_slc_path)
    master_amp = _apply_crop_to_array(master_amp, crop_request)
    slave_amp = _apply_crop_to_array(slave_amp, crop_request)
    avg_amplitude = ((master_amp + slave_amp) * 0.5).astype(np.float32)

    rows, cols = avg_amplitude.shape
    print(
        f"[DEBUG] write_insar_hdf: master_amp.shape={master_amp.shape}, slave_amp.shape={slave_amp.shape}, rows={rows}, cols={cols}"
    )

    with h5py.File(output_h5, "w") as f:
        f.attrs["product_type"] = "insar_interferogram_fullres"
        f.attrs["width"] = cols
        f.attrs["length"] = rows
        f.attrs["wavelength"] = float(wavelength)
        f.attrs["unwrap_method"] = unwrap_method
        f.attrs["radiometry"] = "complex_interferogram"
        f.attrs["value_domain"] = "phase"
        f.attrs["los_displacement_convention"] = "positive = toward satellite"
        for key, value in (extra_attrs or {}).items():
            if value is None:
                continue
            if isinstance(value, (dict, list, tuple)):
                f.attrs[key] = json.dumps(value, ensure_ascii=False)
            else:
                f.attrs[key] = value

        # avg_amplitude
        d_amp = f.create_dataset(
            "avg_amplitude",
            shape=(rows, cols),
            dtype="f4",
            chunks=(min(block_rows, rows), min(1024, cols)),
            compression="gzip",
            shuffle=True,
        )
        d_amp.attrs["long_name"] = "average SLC amplitude"
        d_amp.attrs["description"] = "(master_amplitude + slave_amplitude) / 2"
        d_amp.attrs["units"] = "linear_amplitude"

        # interferogram
        d_int = f.create_dataset(
            "interferogram",
            shape=(rows, cols),
            dtype=np.complex64,
            chunks=(min(block_rows, rows), min(1024, cols)),
            compression="gzip",
            shuffle=True,
        )
        d_int.attrs["long_name"] = "complex interferogram"
        d_int.attrs["description"] = "master * conj(slave)"
        d_int.attrs["units"] = "dimensionless"

        # coherence
        d_coh = f.create_dataset(
            "coherence",
            shape=(rows, cols),
            dtype="f4",
            chunks=(min(block_rows, rows), min(1024, cols)),
            compression="gzip",
            shuffle=True,
        )
        d_coh.attrs["long_name"] = "interferometric coherence"
        d_coh.attrs["description"] = "coherence estimate"
        d_coh.attrs["units"] = "dimensionless"
        d_coh.attrs["valid_min"] = 0.0
        d_coh.attrs["valid_max"] = 1.0

        d_filt = f.create_dataset(
            "filtered_interferogram",
            shape=(rows, cols),
            dtype=np.complex64,
            chunks=(min(block_rows, rows), min(1024, cols)),
            compression="gzip",
            shuffle=True,
        )
        d_filt.attrs["long_name"] = "goldstein filtered interferogram"
        d_filt.attrs["filter_method"] = "goldstein"

        # unwrapped_phase
        d_up = f.create_dataset(
            "unwrapped_phase",
            shape=(rows, cols),
            dtype="f4",
            chunks=(min(block_rows, rows), min(1024, cols)),
            compression="gzip",
            shuffle=True,
        )
        d_up.attrs["long_name"] = "unwrapped phase"
        d_up.attrs["description"] = "phase after 2D unwrapping"
        d_up.attrs["units"] = "radians"

        # los_displacement
        d_los = f.create_dataset(
            "los_displacement",
            shape=(rows, cols),
            dtype="f4",
            chunks=(min(block_rows, rows), min(1024, cols)),
            compression="gzip",
            shuffle=True,
        )
        d_los.attrs["long_name"] = "LOS line-of-sight displacement"
        d_los.attrs["description"] = "unwrapped_phase * wavelength / (4*pi)"
        d_los.attrs["units"] = "meters"

        # Write data
        d_amp[:] = avg_amplitude
        d_int[:] = interferogram.astype(np.complex64)
        d_coh[:] = coherence.astype(np.float32)
        d_filt[:] = (filtered_interferogram if filtered_interferogram is not None else interferogram).astype(np.complex64)
        d_up[:] = unwrapped_phase.astype(np.float32)
        d_los[:] = los_displacement.astype(np.float32)

    return output_h5


def _append_cropped_coordinates_hdf(
    output_h5: str,
    master_topo_path: str,
    crop_request: dict,
    block_rows: int = 64,
) -> str:
    from osgeo import gdal, osr

    window = _crop_window(crop_request)
    if window is None:
        return output_h5
    row0, col0, rows, cols = window

    ds = gdal.Open(master_topo_path)
    if ds is None:
        raise RuntimeError(f"failed to open cached topo raster: {master_topo_path}")

    lon_band = ds.GetRasterBand(1)
    lat_band = ds.GetRasterBand(2)
    hgt_band = ds.GetRasterBand(3)
    longitude = _read_band_array(lon_band, col0, row0, cols, rows, dtype=np.float32).astype(np.float32)
    latitude = _read_band_array(lat_band, col0, row0, cols, rows, dtype=np.float32).astype(np.float32)
    height = _read_band_array(hgt_band, col0, row0, cols, rows, dtype=np.float32).astype(np.float32)
    ds = None

    valid = np.isfinite(longitude) & np.isfinite(latitude)
    if np.any(valid):
        center_lon = float(np.nanmean(longitude[valid]))
        center_lat = float(np.nanmean(latitude[valid]))
    else:
        center_lon = 0.0
        center_lat = 0.0
    epsg = point2epsg(center_lon, center_lat)

    src = osr.SpatialReference()
    src.ImportFromEPSG(4326)
    src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    dst = osr.SpatialReference()
    dst.ImportFromEPSG(epsg)
    dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osr.CoordinateTransformation(src, dst)

    utm_x = np.full((rows, cols), np.nan, dtype=np.float32)
    utm_y = np.full((rows, cols), np.nan, dtype=np.float32)
    if np.any(valid):
        pts = np.column_stack([longitude[valid], latitude[valid]])
        transformed = np.asarray(transform.TransformPoints(pts[:, :2]), dtype=np.float64)
        utm_x[valid] = transformed[:, 0].astype(np.float32)
        utm_y[valid] = transformed[:, 1].astype(np.float32)

    with h5py.File(output_h5, "a") as f:
        for name in ("longitude", "latitude", "height", "utm_x", "utm_y"):
            if name in f:
                del f[name]
        chunk_rows = min(block_rows, rows)
        chunk_cols = min(1024, cols)
        d_lon = f.create_dataset("longitude", data=longitude, chunks=(chunk_rows, chunk_cols), compression="gzip", shuffle=True)
        d_lat = f.create_dataset("latitude", data=latitude, chunks=(chunk_rows, chunk_cols), compression="gzip", shuffle=True)
        d_hgt = f.create_dataset("height", data=height, chunks=(chunk_rows, chunk_cols), compression="gzip", shuffle=True)
        d_x = f.create_dataset("utm_x", data=utm_x, chunks=(chunk_rows, chunk_cols), compression="gzip", shuffle=True)
        d_y = f.create_dataset("utm_y", data=utm_y, chunks=(chunk_rows, chunk_cols), compression="gzip", shuffle=True)
        f.attrs["coordinate_system"] = "EPSG:4326"
        f.attrs["longitude_units"] = "degrees_east"
        f.attrs["latitude_units"] = "degrees_north"
        f.attrs["height_units"] = "meters"
        f.attrs["coordinate_source"] = "cropped_from_cached_master_topo"
        f.attrs["utm_epsg"] = epsg
        f.attrs["utm_coordinate_source"] = "transformed_from_cropped_cached_lonlat"
        d_lon.attrs["long_name"] = "longitude"
        d_lat.attrs["long_name"] = "latitude"
        d_hgt.attrs["long_name"] = "height"
        d_x.attrs["long_name"] = "utm_x"
        d_y.attrs["long_name"] = "utm_y"
    return output_h5


def _write_check_stage(
    *,
    output_dir: Path,
    master_manifest_path: Path,
    slave_manifest_path: Path,
    master_manifest: dict,
    slave_manifest: dict,
    master_acq_data: dict,
    slave_acq_data: dict,
    master_rg_data: dict,
    slave_rg_data: dict,
    master_dop_data: dict,
    slave_dop_data: dict,
    crop_request: dict,
    dc_policy: str,
    prf_policy: str,
    skip_precheck: bool,
) -> dict:
    stage = "check"
    stage_path = stage_dir(output_dir, stage)
    stage_path.mkdir(parents=True, exist_ok=True)
    started_at = utc_now_iso()
    precheck = run_compatibility_precheck(
        master_acquisition=master_acq_data,
        slave_acquisition=slave_acq_data,
        master_radargrid=master_rg_data,
        slave_radargrid=slave_rg_data,
        master_doppler=master_dop_data,
        slave_doppler=slave_dop_data,
        dc_policy=dc_policy,
        prf_policy=prf_policy,
        skip_precheck=skip_precheck,
    )
    (stage_path / "precheck.json").write_text(
        json.dumps(precheck, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    record = {
        "stage": stage,
        "input_manifests": {
            "master": str(master_manifest_path),
            "slave": str(slave_manifest_path),
        },
        "effective_crop": None,
        "backend_used": "cpu",
        "upstream_stage_dependencies": [],
        "output_files": {
            "precheck": str(stage_path / "precheck.json"),
        },
        "start_time": started_at,
        "end_time": utc_now_iso(),
        "success": precheck["overall_severity"] != "fatal",
        "fallback_reason": None,
        "plan": {
            "dc_policy": dc_policy,
            "prf_policy": prf_policy,
            "recommended_geometry_mode": precheck.get("recommended_geometry_mode"),
        },
    }
    write_stage_record(output_dir, stage, record)
    if not record["success"]:
        raise ValueError(
            f"Precheck failed with severity={precheck['overall_severity']}: {json.dumps(precheck['checks'], ensure_ascii=False)}"
        )
    mark_stage_success(output_dir, stage)
    return precheck


def _write_prep_stage(
    *,
    output_dir: Path,
    slave_manifest_path: Path,
    crop_request: dict,
    precheck: dict,
    master_acq_data: dict,
    master_rg_data: dict,
    slave_acq_data: dict,
    slave_rg_data: dict,
    slave_dop_data: dict,
) -> str:
    stage = "prep"
    stage_path = stage_dir(output_dir, stage)
    stage_path.mkdir(parents=True, exist_ok=True)
    started_at = utc_now_iso()
    preprocess_plan, normalized_slave_manifest = build_preprocess_plan(
        precheck=precheck,
        slave_manifest_path=slave_manifest_path,
        stage_dir=stage_path,
        master_acquisition=master_acq_data,
        master_radargrid=master_rg_data,
        slave_acquisition=slave_acq_data,
        slave_radargrid=slave_rg_data,
        slave_doppler=slave_dop_data,
    )
    (stage_path / "preprocess_plan.json").write_text(
        json.dumps(preprocess_plan, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    record = {
        "stage": stage,
        "input_manifests": {"slave": str(slave_manifest_path)},
        "effective_crop": None,
        "backend_used": "cpu",
        "upstream_stage_dependencies": ["check"],
        "output_files": {
            "preprocess_plan": str(stage_path / "preprocess_plan.json"),
            "normalized_slave_manifest": normalized_slave_manifest,
        },
        "start_time": started_at,
        "end_time": utc_now_iso(),
        "success": True,
        "fallback_reason": None,
    }
    write_stage_record(output_dir, stage, record)
    mark_stage_success(output_dir, stage)
    return normalized_slave_manifest


def _write_crop_stage(
    *,
    output_dir: Path,
    master_manifest_path: Path,
    normalized_slave_manifest_path: Path,
    crop_request: dict,
) -> dict:
    stage = "crop"
    stage_path = stage_dir(output_dir, stage)
    stage_path.mkdir(parents=True, exist_ok=True)
    started_at = utc_now_iso()
    window = crop_request.get("master_window") if crop_request.get("mode") != "full" else None
    subset_window = None if window is None else (
        int(window["row0"]),
        int(window["col0"]),
        int(window["rows"]),
        int(window["cols"]),
    )
    cropped_master_manifest = build_cropped_manifest(
        manifest_path=master_manifest_path,
        output_dir=stage_path,
        output_name="master_crop",
        window=subset_window,
    )
    cropped_slave_manifest = build_cropped_manifest(
        manifest_path=normalized_slave_manifest_path,
        output_dir=stage_path,
        output_name="slave_crop",
        window=subset_window,
    )
    master_manifest = load_manifest(master_manifest_path)
    slave_manifest = load_manifest(normalized_slave_manifest_path)
    cropped_master = load_manifest(cropped_master_manifest)
    cropped_slave = load_manifest(cropped_slave_manifest)
    master_slc_path = resolve_manifest_data_path(master_manifest_path, master_manifest["slc"]["path"])
    slave_slc_path = resolve_manifest_data_path(normalized_slave_manifest_path, slave_manifest["slc"]["path"])
    master_crop_slc = resolve_manifest_data_path(cropped_master_manifest, cropped_master["slc"]["path"])
    slave_crop_slc = resolve_manifest_data_path(cropped_slave_manifest, cropped_slave["slc"]["path"])
    def _try_write_preview(slc_path: str | None, png_path: Path) -> str | None:
        if not slc_path:
            return None
        try:
            return _write_radar_amplitude_png(slc_path, png_path)
        except Exception:
            return None

    master_normal_png = _try_write_preview(
        master_slc_path,
        output_dir / "master_normal_fullres.png",
    )
    slave_normal_png = _try_write_preview(
        slave_slc_path,
        output_dir / "slave_normal_fullres.png",
    )
    master_crop_png = _try_write_preview(
        master_crop_slc,
        output_dir / "master_crop_fullres.png",
    )
    slave_crop_png = _try_write_preview(
        slave_crop_slc,
        output_dir / "slave_crop_fullres.png",
    )
    (stage_path / "crop.json").write_text(
        json.dumps(crop_request, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    record = {
        "stage": stage,
        "input_manifests": {
            "master": str(master_manifest_path),
            "slave": str(normalized_slave_manifest_path),
        },
        "effective_crop": crop_request,
        "backend_used": "cpu",
        "upstream_stage_dependencies": ["prep"],
        "output_files": {
            "crop": str(stage_path / "crop.json"),
            "master_crop_manifest": cropped_master_manifest,
            "slave_crop_manifest": cropped_slave_manifest,
            **({"master_normal_fullres_png": master_normal_png} if master_normal_png else {}),
            **({"slave_normal_fullres_png": slave_normal_png} if slave_normal_png else {}),
            **({"master_crop_fullres_png": master_crop_png} if master_crop_png else {}),
            **({"slave_crop_fullres_png": slave_crop_png} if slave_crop_png else {}),
        },
        "start_time": started_at,
        "end_time": utc_now_iso(),
        "success": True,
        "fallback_reason": None,
    }
    write_stage_record(output_dir, stage, record)
    mark_stage_success(output_dir, stage)
    return {
        "master_manifest": cropped_master_manifest,
        "slave_manifest": cropped_slave_manifest,
        "crop_request": crop_request,
    }


def _write_synthetic_pipeline_stage_records(
    *,
    output_dir: Path,
    master_manifest_path: Path,
    slave_manifest_path: Path,
    crop_request: dict,
    backend_used: str,
    outputs: dict,
) -> None:
    pipeline_outputs = {
        "p0": {},
        "p1": {},
        "p2": {},
        "p3": {},
        "p4": {},
    }
    for stage in ("p0", "p1", "p2", "p3", "p4"):
        record = {
            "stage": stage,
            "input_manifests": {
                "master": str(master_manifest_path),
                "slave": str(slave_manifest_path),
            },
            "effective_crop": crop_request,
            "backend_used": backend_used,
            "upstream_stage_dependencies": ["prep"] if stage == "p0" else [f"p{int(stage[1]) - 1}"],
            "output_files": pipeline_outputs[stage],
            "start_time": utc_now_iso(),
            "end_time": utc_now_iso(),
            "success": True,
            "fallback_reason": None,
            "synthetic_record": True,
        }
        write_stage_record(output_dir, stage, record)
        mark_stage_success(output_dir, stage)


def _write_p5_stage_record(
    *,
    output_dir: Path,
    master_manifest_path: Path,
    slave_manifest_path: Path,
    crop_request: dict,
    backend_used: str,
    input_h5: str,
) -> None:
    stage = "p5"
    record = {
        "stage": stage,
        "input_manifests": {
            "master": str(master_manifest_path),
            "slave": str(slave_manifest_path),
        },
        "effective_crop": crop_request,
        "backend_used": backend_used,
        "upstream_stage_dependencies": ["p4"],
        "output_files": {"interferogram_h5": input_h5},
        "start_time": utc_now_iso(),
        "end_time": utc_now_iso(),
        "success": True,
        "fallback_reason": None,
    }
    write_stage_record(output_dir, stage, record)
    mark_stage_success(output_dir, stage)


def _load_stage_output_path(output_dir: Path, stage: str, key: str) -> str:
    record = load_stage_record(output_dir, stage) or {}
    path = record.get("output_files", {}).get(key)
    if not path:
        raise RuntimeError(f"Missing cached output '{key}' for stage '{stage}'")
    return str(path)


def _load_cached_array(output_dir: Path, stage: str, key: str) -> np.ndarray:
    return np.load(_load_stage_output_path(output_dir, stage, key))


def _run_hdf_stage_from_cache(
    *,
    output_dir: Path,
    master_manifest_path: Path,
    slave_manifest_path: Path,
    wavelength: float,
    crop_request: dict,
    backend_used: str,
    block_rows: int,
    unwrap_method: str,
    extra_hdf_attrs: dict | None,
    master_manifest: dict,
    master_orbit_data: dict,
    master_acq_data: dict,
    master_rg_data: dict,
    resolved_dem: str,
    orbit_interp: str,
) -> dict:
    stage = "p5"
    stage_path = stage_dir(output_dir, stage)
    stage_path.mkdir(parents=True, exist_ok=True)
    slave_manifest = load_manifest(slave_manifest_path)
    master_slc = resolve_manifest_data_path(master_manifest_path, master_manifest["slc"]["path"])
    slave_slc = resolve_manifest_data_path(slave_manifest_path, slave_manifest["slc"]["path"])
    raw_interferogram = _load_cached_array(output_dir, "p2", "interferogram")
    try:
        filtered_interferogram = _load_cached_array(output_dir, "p2", "filtered_interferogram")
    except RuntimeError:
        filtered_interferogram = raw_interferogram
    coherence = _load_cached_array(output_dir, "p2", "coherence")
    unwrapped_phase = _load_cached_array(output_dir, "p3", "unwrapped_phase")
    los_displacement = _load_cached_array(output_dir, "p4", "los_displacement")
    cropped_rg, cropped_acq = _apply_crop_to_metadata(
        master_rg_data, master_acq_data, crop_request
    )
    radar_grid = construct_radar_grid(cropped_rg, cropped_acq, master_orbit_data)
    output_h5 = build_output_paths(output_dir)["interferogram_h5"]
    write_insar_hdf(
        master_slc,
        slave_slc,
        raw_interferogram,
        coherence,
        unwrapped_phase,
        los_displacement,
        wavelength,
        unwrap_method,
        output_h5,
        radar_grid,
        block_rows=block_rows,
        extra_attrs=extra_hdf_attrs,
        filtered_interferogram=filtered_interferogram,
        crop_request=crop_request,
    )
    if _is_full_crop(crop_request):
        append_topo_coordinates_hdf(
            master_manifest_path,
            resolved_dem,
            output_h5,
            block_rows=block_rows,
            orbit_interp=orbit_interp,
            use_gpu=False,
        )
        append_utm_coordinates_hdf(
            output_h5,
            master_manifest_path,
            block_rows=min(block_rows, 64),
        )
    else:
        master_topo = _load_stage_output_path(output_dir, "p0", "master_topo")
        _append_cropped_coordinates_hdf(
            output_h5,
            master_topo,
            crop_request or {},
            block_rows=min(block_rows, 64),
        )
    record = {
        "stage": stage,
        "input_manifests": {
            "master": str(master_manifest_path),
            "slave": str(slave_manifest_path),
        },
        "effective_crop": crop_request,
        "backend_used": backend_used,
        "upstream_stage_dependencies": ["p4"],
        "output_files": {"interferogram_h5": output_h5},
        "start_time": utc_now_iso(),
        "end_time": utc_now_iso(),
        "success": True,
        "fallback_reason": None,
    }
    write_stage_record(output_dir, stage, record)
    mark_stage_success(output_dir, stage)
    return {"interferogram_h5": output_h5}


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
        if require_existing and not Path(candidate).is_file():
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

    p2_dir = stage_dir(output_dir, "p2")
    p2_dir.mkdir(parents=True, exist_ok=True)
    flatten_path = p2_dir / "range_flatten.off.tif"
    flatten_mask_path = p2_dir / "range_flatten_valid_mask.tif"
    flatten_model = str(os.environ.get("D2SAR_FLATTEN_OFFSET_MODEL", "full")).strip().lower()
    if flatten_model not in {"full", "plane"}:
        flatten_model = "full"

    def split_geo2rdr_offsets(block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        clean = np.asarray(block, dtype=np.float64)
        valid = np.isfinite(clean)
        valid &= clean != GEO2RDR_OFFSET_NODATA
        valid &= clean >= GEO2RDR_OFFSET_INVALID_LOW
        if not np.all(valid):
            clean = clean.copy()
            clean[~valid] = 0.0
        return clean, valid

    try:
        src_ds = gdal.Open(str(source_path), gdal.GA_ReadOnly)
    except Exception:
        src_ds = None
    driver = gdal.GetDriverByName("GTiff")
    dst_ds = driver.Create(
        str(flatten_path),
        cols,
        rows,
        1,
        gdal.GDT_Float64,
        options=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER"],
    )
    if dst_ds is None:
        raise RuntimeError(f"failed to create flatten range offset raster: {flatten_path}")
    flatten_band = dst_ds.GetRasterBand(1)
    flatten_mask_ds = None
    flatten_mask_ds = driver.Create(
        str(flatten_mask_path),
        cols,
        rows,
        1,
        gdal.GDT_Byte,
        options=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER"],
    )
    if flatten_mask_ds is None:
        dst_ds = None
        raise RuntimeError(f"failed to create flatten valid mask raster: {flatten_mask_path}")
    mask_band = flatten_mask_ds.GetRasterBand(1)
    mask_band.SetNoDataValue(0.0)
    block_rows = 512

    plane_coeffs: tuple[float, float, float] | None = None
    if flatten_model == "plane":
        sample_rows: list[np.ndarray] = []
        sample_cols: list[np.ndarray] = []
        sample_vals: list[np.ndarray] = []
        row_stride = 32
        col_stride = 32
        try:
            if src_ds is not None:
                src_band = src_ds.GetRasterBand(1)
                for row0 in range(0, rows, block_rows):
                    nrows = min(block_rows, rows - row0)
                    block, valid = split_geo2rdr_offsets(
                        _read_band_array(src_band, 0, row0, cols, nrows, dtype=np.float64)
                    )
                    rr = np.arange(row0, row0 + nrows, row_stride, dtype=np.float64)
                    cc = np.arange(0, cols, col_stride, dtype=np.float64)
                    if rr.size == 0 or cc.size == 0:
                        continue
                    row_sub = (rr - row0).astype(np.int32)
                    col_sub = cc.astype(np.int32)
                    sub = block[row_sub[:, None], col_sub[None, :]]
                    sub_valid = valid[row_sub[:, None], col_sub[None, :]]
                    if not np.any(sub_valid):
                        continue
                    yy, xx = np.meshgrid(rr, cc, indexing="ij")
                    sample_rows.append(yy[sub_valid])
                    sample_cols.append(xx[sub_valid])
                    sample_vals.append(sub[sub_valid])
            else:
                src = np.memmap(source_path, dtype=np.float64, mode="r", shape=(rows, cols))
                for row0 in range(0, rows, block_rows):
                    nrows = min(block_rows, rows - row0)
                    block, valid = split_geo2rdr_offsets(src[row0 : row0 + nrows, :])
                    rr = np.arange(row0, row0 + nrows, row_stride, dtype=np.float64)
                    cc = np.arange(0, cols, col_stride, dtype=np.float64)
                    if rr.size == 0 or cc.size == 0:
                        continue
                    row_sub = (rr - row0).astype(np.int32)
                    col_sub = cc.astype(np.int32)
                    sub = block[row_sub[:, None], col_sub[None, :]]
                    sub_valid = valid[row_sub[:, None], col_sub[None, :]]
                    if not np.any(sub_valid):
                        continue
                    yy, xx = np.meshgrid(rr, cc, indexing="ij")
                    sample_rows.append(yy[sub_valid])
                    sample_cols.append(xx[sub_valid])
                    sample_vals.append(sub[sub_valid])
                del src
            if sample_vals:
                rows_vec = np.concatenate(sample_rows).astype(np.float64)
                cols_vec = np.concatenate(sample_cols).astype(np.float64)
                vals_vec = np.concatenate(sample_vals).astype(np.float64)
                A = np.column_stack(
                    [
                        np.ones_like(rows_vec, dtype=np.float64),
                        rows_vec,
                        cols_vec,
                    ]
                )
                coeffs, *_ = np.linalg.lstsq(A, vals_vec, rcond=None)
                plane_coeffs = (float(coeffs[0]), float(coeffs[1]), float(coeffs[2]))
        except Exception:
            plane_coeffs = None
        if plane_coeffs is None:
            flatten_model = "full"

    try:
        if src_ds is not None:
            if src_ds.RasterXSize != cols or src_ds.RasterYSize != rows:
                raise RuntimeError(
                    "coarse geo2rdr range offset dimensions do not match radar grid: "
                    f"{source_path}={src_ds.RasterYSize}x{src_ds.RasterXSize}, "
                    f"radargrid={rows}x{cols}"
                )
            src_band = src_ds.GetRasterBand(1)
            for row0 in range(0, rows, block_rows):
                nrows = min(block_rows, rows - row0)
                block, valid = split_geo2rdr_offsets(
                    _read_band_array(src_band, 0, row0, cols, nrows, dtype=np.float64)
                )
                if flatten_model == "plane" and plane_coeffs is not None:
                    a0, a1, a2 = plane_coeffs
                    rr = np.arange(row0, row0 + nrows, dtype=np.float64)[:, None]
                    cc = np.arange(cols, dtype=np.float64)[None, :]
                    model_block = a0 + a1 * rr + a2 * cc
                    _write_band_array(flatten_band, model_block, 0, row0)
                else:
                    # D2SAR geo2rdr offsets are slave-minus-master (secondary
                    # minus reference) range offsets. For interferogram phase
                    # ref * conj(sec), the flat-earth term is +4*pi*(sec-ref)/lambda,
                    # so crossmul must subtract that same sign.
                    _write_band_array(flatten_band, block, 0, row0)
                _write_band_array(mask_band, valid.astype(np.uint8), 0, row0)
        else:
            expected_size = rows * cols * np.dtype(np.float64).itemsize
            if source_path.stat().st_size != expected_size:
                raise RuntimeError(
                    f"cannot interpret coarse geo2rdr range offsets as raster or raw float64 grid: {source_path}"
                )
            src = np.memmap(source_path, dtype=np.float64, mode="r", shape=(rows, cols))
            for row0 in range(0, rows, block_rows):
                nrows = min(block_rows, rows - row0)
                block, valid = split_geo2rdr_offsets(src[row0 : row0 + nrows, :])
                if flatten_model == "plane" and plane_coeffs is not None:
                    a0, a1, a2 = plane_coeffs
                    rr = np.arange(row0, row0 + nrows, dtype=np.float64)[:, None]
                    cc = np.arange(cols, dtype=np.float64)[None, :]
                    model_block = a0 + a1 * rr + a2 * cc
                    _write_band_array(flatten_band, model_block, 0, row0)
                else:
                    _write_band_array(flatten_band, block, 0, row0)
                _write_band_array(mask_band, valid.astype(np.uint8), 0, row0)
            del src
        flatten_band.FlushCache()
        mask_band.FlushCache()
        dst_ds.FlushCache()
        flatten_mask_ds.FlushCache()
    finally:
        src_ds = None
        dst_ds = None
        flatten_mask_ds = None
    return str(flatten_path), str(flatten_mask_path), flatten_model


def _isce2_flatten_policy_record(
    *,
    source: str,
    model: str | None = None,
    source_mode: str | None = None,
    mask_raster: str | None = None,
    ref_sec_starting_range_shift_m: float | None = None,
) -> dict:
    source_note = (
        "Use final range offsets consistent with the resampled secondary SLC "
        "(preferred when fine_coreg_slave is used)."
        if source == "p1_final_range_offsets"
        else "Use coarse geo2rdr range offsets."
    )
    model_note = (
        "Use full-resolution range offsets directly."
        if str(model or "").strip().lower() == "full"
        else "Use a robust fitted plane from selected range offsets."
    )
    return {
        "method": "isce2-explicit-rangeoff-flatten",
        "source_mode": source_mode,
        "source": source,
        "model": model,
        "ref_sec_starting_range_shift_m": ref_sec_starting_range_shift_m,
        "sign_convention": "slave_minus_master_pixel_index",
        "formula": (
            "ifg *= exp(-1j * 4pi * "
            "(range_pixel_spacing * range_offset + ref_sec_starting_range_shift_m) / wavelength)"
        ),
        "invalid_values": {
            "nodata": GEO2RDR_OFFSET_NODATA,
            "invalid_low": GEO2RDR_OFFSET_INVALID_LOW,
            "handling": "mask",
            "mask_raster": mask_raster,
        },
        "note": (
            "ISCE2-style explicit flat-earth/topographic phase removal. "
            f"{source_note} "
            f"{model_note} "
            "Invalid offset pixels are masked out during flattening "
            "(no flatten phase applied). "
            "Use the same geometry range offset convention as resampling."
        ),
    }


def _add_flatten_outputs(output_files: dict, flatten_options: dict) -> None:
    flatten_raster = flatten_options.get("flatten_raster")
    if not flatten_raster:
        return
    flatten_source = str(flatten_options.get("flatten_source") or "unknown")
    flatten_model = str(flatten_options.get("flatten_model") or "unknown")
    flatten_source_mode = str(flatten_options.get("flatten_source_mode") or "unknown")
    flatten_mask = flatten_options.get("flatten_mask_raster")
    flatten_shift_m = flatten_options.get("flatten_starting_range_shift_m")
    output_files["flatten_range_offsets"] = flatten_raster
    output_files["flatten_range_offsets_mask"] = flatten_mask
    output_files["flatten_method"] = "isce2-explicit-rangeoff-flatten"
    output_files["flatten_offset_model"] = flatten_model
    output_files["flatten_offset_source_mode"] = flatten_source_mode
    output_files["flatten_ref_sec_starting_range_shift_m"] = (
        None if flatten_shift_m is None else float(flatten_shift_m)
    )
    output_files["flatten_offset_policy"] = _isce2_flatten_policy_record(
        source=flatten_source,
        model=flatten_model,
        source_mode=flatten_source_mode,
        mask_raster=None if flatten_mask is None else str(flatten_mask),
        ref_sec_starting_range_shift_m=(
            None if flatten_shift_m is None else float(flatten_shift_m)
        ),
    )


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

    selected_source: str | None = None
    selected_range_offset: str | None = None
    coarse_range_offset: str | None = None
    final_range_offset: str | None = None
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
                same_registered_slave = (
                    Path(str(fine_coreg_slave)).resolve()
                    == Path(str(registered_slave_slc)).resolve()
                )
            except Exception:
                same_registered_slave = str(fine_coreg_slave) == str(registered_slave_slc)
            final_consistent = bool(same_registered_slave)

    if flatten_source_mode == "none":
        selected_source = None
        selected_range_offset = None
    elif flatten_source_mode == "coarse":
        if coarse_range_offset:
            selected_source = "coarse_geo2rdr_range_offsets"
            selected_range_offset = coarse_range_offset
        elif final_range_offset and final_consistent:
            selected_source = "p1_final_range_offsets"
            selected_range_offset = final_range_offset
    elif flatten_source_mode == "final":
        if final_range_offset and final_consistent:
            selected_source = "p1_final_range_offsets"
            selected_range_offset = final_range_offset
        elif coarse_range_offset:
            selected_source = "coarse_geo2rdr_range_offsets"
            selected_range_offset = coarse_range_offset
    else:
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
    if flatten_raster is None or flatten_model is None:
        return {
            "flatten_raster": None,
            "flatten_mask_raster": None,
            "range_pixel_spacing": None,
            "wavelength": None,
            "flatten_source": None,
            "flatten_model": None,
            "flatten_source_mode": flatten_source_mode,
            "flatten_starting_range_shift_m": None,
        }
    ref_sec_starting_range_shift_m = _compute_ref_sec_starting_range_shift_m(
        radargrid_data,
        slave_radargrid_data,
    )
    return {
        "flatten_raster": flatten_raster,
        "flatten_mask_raster": flatten_mask_raster,
        "range_pixel_spacing": float(radargrid_data["columnSpacing"]),
        "wavelength": get_wavelength(acquisition_data),
        "flatten_source": selected_source,
        "flatten_model": flatten_model,
        "flatten_source_mode": flatten_source_mode,
        "flatten_starting_range_shift_m": ref_sec_starting_range_shift_m,
    }


def _run_p2_stage_from_cache(
    *,
    output_dir: Path,
    master_manifest_path: Path,
    slave_manifest_path: Path,
    master_manifest: dict,
    slave_manifest: dict,
    crop_request: dict,
    backend_used: str,
    block_rows: int,
    use_gpu: bool,
    gpu_id: int,
    master_acq_data: dict | None = None,
    master_rg_data: dict | None = None,
    slave_rg_data: dict | None = None,
) -> dict:
    master_slc = resolve_manifest_data_path(master_manifest_path, master_manifest["slc"]["path"])
    slave_slc = resolve_manifest_data_path(slave_manifest_path, slave_manifest["slc"]["path"])
    registered_slave_slc = slave_slc
    p1_outputs = None
    try:
        p1_record = load_stage_record(output_dir, "p1")
        if p1_record:
            p1_outputs = p1_record.get("output_files")
            registered_slave_slc = _select_registered_slave_slc(
                p1_outputs,
                slave_slc,
                require_existing=True,
            )
    except Exception:
        pass
    flatten_options = _build_crossmul_flatten_options(
        output_dir=output_dir,
        p1_outputs=p1_outputs,
        registered_slave_slc=registered_slave_slc,
        radargrid_data=master_rg_data or {},
        slave_radargrid_data=slave_rg_data or {},
        acquisition_data=master_acq_data or {},
    )
    crossmul_result = _run_crossmul(
        master_slc,
        registered_slave_slc,
        use_gpu=use_gpu,
        gpu_id=gpu_id,
        output_dir=output_dir,
        block_rows=block_rows,
        **flatten_options,
    )
    if len(crossmul_result) == 2:
        interferogram, coherence = crossmul_result
        crossmul_backend = backend_used
        crossmul_fallback_reason = None
    else:
        interferogram, coherence, crossmul_backend, crossmul_fallback_reason = crossmul_result
    interferogram = _apply_crop_to_array(interferogram, crop_request)
    coherence = _apply_crop_to_array(coherence, crop_request)
    filtered_interferogram = goldstein_filter(interferogram)
    output_files = {
        "interferogram": _save_stage_array(output_dir, "p2", "interferogram", interferogram),
        "filtered_interferogram": _save_stage_array(output_dir, "p2", "filtered_interferogram", filtered_interferogram),
        "coherence": _save_stage_array(output_dir, "p2", "coherence", coherence),
    }
    _add_flatten_outputs(output_files, flatten_options)
    output_files["wrapped_phase_radar_png"] = _write_radar_wrapped_phase_png(
        interferogram,
        output_dir / "wrapped_phase_radar.png",
    )
    _write_stage_outputs_record(
        output_dir=output_dir,
        stage="p2",
        master_manifest_path=master_manifest_path,
        slave_manifest_path=slave_manifest_path,
        crop_request=crop_request,
        backend_used=crossmul_backend,
        output_files=output_files,
        fallback_reason=crossmul_fallback_reason,
    )
    return output_files


def _run_p0_stage(
    *,
    output_dir: Path,
    master_manifest_path: Path,
    slave_manifest_path: Path,
    resolved_dem: str,
    orbit_interp: str,
    crop_request: dict,
    backend_used: str,
    block_rows: int,
    gpu_id: int,
) -> dict:
    master_topo, slave_topo = _run_geo2rdr(
        str(master_manifest_path),
        str(slave_manifest_path),
        resolved_dem,
        orbit_interp,
        use_gpu=backend_used == "gpu",
        gpu_id=gpu_id,
        output_dir=output_dir,
        block_rows=block_rows,
    )
    output_files = {"master_topo": master_topo, "slave_topo": slave_topo}
    master_topo_vrt = output_dir / "geo2rdr_master" / "topo.vrt"
    slave_topo_vrt = output_dir / "geo2rdr_slave" / "topo.vrt"
    if master_topo_vrt.is_file():
        output_files["master_topo_vrt"] = str(master_topo_vrt)
    if slave_topo_vrt.is_file():
        output_files["slave_topo_vrt"] = str(slave_topo_vrt)
    _write_stage_outputs_record(
        output_dir=output_dir,
        stage="p0",
        master_manifest_path=master_manifest_path,
        slave_manifest_path=slave_manifest_path,
        crop_request=crop_request,
        backend_used=backend_used,
        output_files=output_files,
    )
    return output_files


def _has_radar_grid_inputs(
    orbit_data: dict | None,
    acquisition_data: dict | None,
    radargrid_data: dict | None,
) -> bool:
    if not orbit_data or not acquisition_data or not radargrid_data:
        return False
    orbit_header = orbit_data.get("header", {})
    return all(
        key in acquisition_data
        for key in ("startGPSTime", "centerFrequency", "prf")
    ) and all(
        key in radargrid_data
        for key in (
            "numberOfRows",
            "numberOfColumns",
            "rangeTimeFirstPixel",
            "columnSpacing",
        )
    ) and "firstStateTimeUTC" in orbit_header


def _construct_radar_grid_if_possible(
    orbit_data: dict | None,
    acquisition_data: dict | None,
    radargrid_data: dict | None,
):
    if not _has_radar_grid_inputs(orbit_data, acquisition_data, radargrid_data):
        return None
    return construct_radar_grid(radargrid_data, acquisition_data, orbit_data)


def _construct_doppler_if_possible(
    doppler_data: dict | None,
    *,
    orbit_data: dict | None = None,
    acquisition_data: dict | None = None,
    radargrid_data: dict | None = None,
):
    if not doppler_data or "combinedDoppler" not in doppler_data:
        return None
    try:
        if orbit_data is None or acquisition_data is None or radargrid_data is None:
            return None
        return construct_doppler_lut2d(
            doppler_data,
            radargrid_json=radargrid_data,
            acquisition_json=acquisition_data,
            orbit_json=orbit_data,
        )
    except Exception:
        return None


def _run_p1_stage_from_cache(
    *,
    output_dir: Path,
    master_manifest_path: Path,
    slave_manifest_path: Path,
    master_manifest: dict,
    slave_manifest: dict,
    crop_request: dict,
    backend_used: str,
    gpu_id: int,
    master_orbit_data: dict | None = None,
    master_acq_data: dict | None = None,
    master_rg_data: dict | None = None,
    slave_orbit_data: dict | None = None,
    slave_acq_data: dict | None = None,
    slave_rg_data: dict | None = None,
    slave_dop_data: dict | None = None,
) -> dict:
    """Run p1: coarse SLC resampling + optional PyCuAmpcor dense matching.

    Follows ISCE3 standard order:
      1. ResampSlc (zero offsets) → coarse_coreg_slave.tif  (radar domain)
      2. PyCuAmpcor (GPU only) on master × coarse_coreg_slave → dense offsets

    p2 crossmul will prefer fine_coreg_slave.tif when dense offsets are staged,
    and fall back to coarse_coreg_slave.tif otherwise.
    """
    master_slc = resolve_manifest_data_path(master_manifest_path, master_manifest["slc"]["path"])
    slave_slc = resolve_manifest_data_path(slave_manifest_path, slave_manifest["slc"]["path"])
    p1_stage_path = stage_dir(output_dir, "p1")
    p1_stage_path.mkdir(parents=True, exist_ok=True)
    coarse_coreg_slave_path = str(p1_stage_path / "coarse_coreg_slave.tif")

    ref_radar_grid = _construct_radar_grid_if_possible(
        master_orbit_data,
        master_acq_data,
        master_rg_data,
    )
    slave_radar_grid = _construct_radar_grid_if_possible(
        slave_orbit_data,
        slave_acq_data,
        slave_rg_data,
    )
    slave_doppler = _construct_doppler_if_possible(
        slave_dop_data,
        orbit_data=slave_orbit_data,
        acquisition_data=slave_acq_data,
        radargrid_data=slave_rg_data,
    )

    use_gpu = (backend_used == "gpu")

    range_res = float(master_rg_data.get("groundRangeResolution", 0.0) or 0.0)
    azimuth_res = float(master_rg_data.get("azimuthResolution", 0.0) or 0.0)
    effective_resolution = max(range_res, azimuth_res)
    ok = False
    coarse_rg_offset_path = None
    coarse_az_offset_path = None
    if slave_radar_grid is not None and ref_radar_grid is not None:
        p0_record = load_stage_record(output_dir, "p0") or {}
        p0_outputs = p0_record.get("output_files", {})
        master_topo_vrt_path = p0_outputs.get("master_topo_vrt")
        fallback_master_topo_vrt = output_dir / "geo2rdr_master" / "topo.vrt"
        if master_topo_vrt_path and not Path(str(master_topo_vrt_path)).is_file():
            master_topo_vrt_path = None
        if not master_topo_vrt_path and fallback_master_topo_vrt.is_file():
            master_topo_vrt_path = str(fallback_master_topo_vrt)
        if master_topo_vrt_path:
            try:
                coarse_rg_offset_path, coarse_az_offset_path = _run_slave_geo2rdr_from_master_topo(
                    master_topo_vrt_path=master_topo_vrt_path,
                    slave_orbit_data=slave_orbit_data,
                    slave_acq_data=slave_acq_data,
                    slave_rg_data=slave_rg_data,
                    slave_dop_data=slave_dop_data,
                    output_dir=p1_stage_path,
                    use_gpu=use_gpu,
                    gpu_id=gpu_id,
                )
            except Exception as exc:
                print(f"[P1] coarse geo2rdr offsets unavailable: {exc}", flush=True)
                coarse_rg_offset_path = None
                coarse_az_offset_path = None
        ok = run_coarse_resamp_isce3_v2(
            slave_slc_path=slave_slc,
            coarse_coreg_slave_path=coarse_coreg_slave_path,
            radar_grid=slave_radar_grid,
            doppler=slave_doppler,
            ref_radar_grid=ref_radar_grid,
            rg_offset_path=coarse_rg_offset_path,
            az_offset_path=coarse_az_offset_path,
            use_gpu=use_gpu,
        )

    if not ok:
        from insar_registration import _copy_raster
        _copy_raster(slave_slc, Path(coarse_coreg_slave_path))

    row_offset = None
    col_offset = None
    dense_match_details = None
    dense_source = "pycuampcor" if use_gpu else "cpu-dense-match"
    try:
        master_topo = _load_stage_output_path(output_dir, "p0", "master_topo") if use_gpu else ""
        slave_topo = _load_stage_output_path(output_dir, "p0", "slave_topo") if use_gpu else ""
        gross_offset = (0.0, 0.0)
        if coarse_az_offset_path is not None and coarse_rg_offset_path is not None:
            try:
                gross_offset = (
                    _estimate_offset_mean_from_raster(coarse_az_offset_path),
                    _estimate_offset_mean_from_raster(coarse_rg_offset_path),
                )
            except Exception:
                gross_offset = (0.0, 0.0)
        dense_plan = _select_cpu_dense_match_plan(effective_resolution)

        def _run_dense_match(
            stage_path_override: Path,
        ) -> tuple[np.ndarray | None, np.ndarray | None, dict | None]:
            if use_gpu:
                first_candidate = dense_plan["candidates"][0]
                window_size = tuple(first_candidate["window_size"])
                search_range = tuple(first_candidate["search_range"])
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
                        full_shape = _raster_shape(master_slc)
                        if full_shape is not None:
                            varying_cfg = _write_varying_gross_offset_file(
                                range_offset_path=coarse_rg_offset_path,
                                azimuth_offset_path=coarse_az_offset_path,
                                output_path=(
                                    stage_path_override / "pycuampcor_varying_gross_offsets.bin"
                                ),
                                full_shape=full_shape,
                                window_size=window_size,
                                search_range=search_range,
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
                                "skip": (
                                    int(varying_cfg.get("skip_down", 32)),
                                    int(varying_cfg.get("skip_across", 32)),
                                ),
                            }
                    except Exception as exc:
                        print(
                            f"[P1] varying gross offsets unavailable for pycuampcor: {exc}",
                            flush=True,
                        )
                elif coarse_rg_offset_path is not None and coarse_az_offset_path is not None:
                    print(
                        "[P1] skip varying gross offsets for pycuampcor "
                        "(slave input is coarse_coreg_slave); "
                        "set D2SAR_DENSE_GROSS_ON_COARSE_COREG=1 to enable",
                        flush=True,
                    )
                return _run_pycuampcor(
                    master_slc,
                    coarse_coreg_slave_path,
                    master_topo,
                    slave_topo,
                    output_dir=stage_path_override,
                    gpu_id=gpu_id,
                    return_details=True,
                    window_size=window_size,
                    search_range=search_range,
                    **pycuampcor_kwargs,
                )
            return run_cpu_dense_offsets(
                master_slc_path=master_slc,
                slave_slc_path=coarse_coreg_slave_path,
                output_dir=stage_path_override,
                return_details=True,
                gross_offset=gross_offset,
                window_candidates=dense_plan["candidates"],
            )

        row_offset, col_offset, dense_match_details = _run_dense_match(p1_stage_path)
    except NotImplementedError:
        row_offset = None
        col_offset = None
        dense_match_details = None
    except Exception:
        row_offset = None
        col_offset = None
        dense_match_details = None

    registration_outputs = write_registration_outputs(
        stage_path=p1_stage_path,
        slave_slc_path=slave_slc,
        coarse_coreg_slave_path=coarse_coreg_slave_path,
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
    )
    def _try_write_stage_preview(slc_path: str | None, png_path: Path) -> str | None:
        if not slc_path:
            return None
        try:
            return _write_radar_amplitude_png(slc_path, png_path)
        except Exception:
            return None

    coarse_coreg_png = _try_write_stage_preview(
        registration_outputs.get("coarse_coreg_slave"),
        output_dir / "slave_coarse_coreg_fullres.png",
    )
    fine_coreg_png = _try_write_stage_preview(
        registration_outputs.get("fine_coreg_slave"),
        output_dir / "slave_fine_coreg_fullres.png",
    )
    if coarse_coreg_png:
        registration_outputs["coarse_coreg_slave_png"] = coarse_coreg_png
    if fine_coreg_png:
        registration_outputs["fine_coreg_slave_png"] = fine_coreg_png

    offsets_path = p1_stage_path / "offsets.json"
    offsets_path.write_text(
        json.dumps(
            {
                "row_offset": None if row_offset is None else str(p1_stage_path / "azimuth.off.tif"),
                "col_offset": None if col_offset is None else str(p1_stage_path / "range.off.tif"),
                "row_residual_offset": None if row_offset is None else str(p1_stage_path / "azimuth_residual.off.tif"),
                "col_residual_offset": None if col_offset is None else str(p1_stage_path / "range_residual.off.tif"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    output_files = {"offsets": str(offsets_path), **registration_outputs}
    if coarse_rg_offset_path:
        output_files["coarse_geo2rdr_range_offsets"] = coarse_rg_offset_path
    if coarse_az_offset_path:
        output_files["coarse_geo2rdr_azimuth_offsets"] = coarse_az_offset_path
    coarse_model_path = p1_stage_path / "coarse_geo2rdr_model.json"
    if coarse_model_path.is_file():
        output_files["coarse_geo2rdr_model"] = str(coarse_model_path)
    _write_stage_outputs_record(
        output_dir=output_dir,
        stage="p1",
        master_manifest_path=master_manifest_path,
        slave_manifest_path=slave_manifest_path,
        crop_request=crop_request,
        backend_used=backend_used,
        output_files=output_files,
    )
    return output_files


def _run_p3_stage_from_cache(
    *,
    output_dir: Path,
    master_manifest_path: Path,
    slave_manifest_path: Path,
    resolved_dem: str,
    orbit_interp: str,
    unwrapper: PhaseUnwrapper,
    crop_request: dict,
    block_rows: int,
    backend_used: str,
    master_orbit_data: dict,
    master_acq_data: dict,
    master_rg_data: dict,
) -> dict:
    try:
        interferogram = _load_cached_array(output_dir, "p2", "filtered_interferogram")
    except RuntimeError:
        interferogram = _load_cached_array(output_dir, "p2", "interferogram")
    coherence = _load_cached_array(output_dir, "p2", "coherence")
    cropped_rg, cropped_acq = _apply_crop_to_metadata(
        master_rg_data, master_acq_data, crop_request
    )
    radar_grid = construct_radar_grid(cropped_rg, cropped_acq, master_orbit_data)
    orbit = construct_orbit(master_orbit_data, orbit_interp)
    with tempfile.TemporaryDirectory(prefix="insar_unwrap_", dir=str(output_dir)) as tmpdir:
        unwrapped_phase = unwrapper.unwrap(
            interferogram,
            coherence,
            radar_grid,
            orbit,
            resolved_dem,
            tmpdir,
            block_rows,
        )
    output_files = {
        "unwrapped_phase": _save_stage_array(output_dir, "p3", "unwrapped_phase", unwrapped_phase)
    }
    _write_stage_outputs_record(
        output_dir=output_dir,
        stage="p3",
        master_manifest_path=master_manifest_path,
        slave_manifest_path=slave_manifest_path,
        crop_request=crop_request,
        backend_used=backend_used,
        output_files=output_files,
    )
    return output_files


def _run_p4_stage_from_cache(
    *,
    output_dir: Path,
    master_manifest_path: Path,
    slave_manifest_path: Path,
    wavelength: float,
    crop_request: dict,
    backend_used: str,
) -> dict:
    unwrapped_phase = _load_cached_array(output_dir, "p3", "unwrapped_phase")
    los_displacement = compute_los_displacement(unwrapped_phase, wavelength)
    output_files = {
        "los_displacement": _save_stage_array(output_dir, "p4", "los_displacement", los_displacement)
    }
    _write_stage_outputs_record(
        output_dir=output_dir,
        stage="p4",
        master_manifest_path=master_manifest_path,
        slave_manifest_path=slave_manifest_path,
        crop_request=crop_request,
        backend_used=backend_used,
        output_files=output_files,
    )
    return output_files


def _run_publish_stage_from_hdf(
    *,
    output_dir: Path,
    master_manifest_path: Path,
    slave_manifest_path: Path,
    input_h5: str,
    resolution_meters: float,
    crop_request: dict,
    backend_used: str,
) -> dict:
    stage = "p6"
    stage_path = stage_dir(output_dir, stage)
    stage_path.mkdir(parents=True, exist_ok=True)
    started_at = utc_now_iso()
    output_paths = build_output_paths(output_dir)
    target_width, target_height = compute_utm_output_shape(
        input_h5,
        resolution_meters,
        block_rows=64,
    )
    geocode_product(
        input_h5,
        output_paths["coherence_tif"],
        "coherence",
        target_width,
        target_height,
        block_rows=64,
    )
    geocode_product(
        input_h5,
        output_paths["unwrapped_phase_tif"],
        "unwrapped_phase",
        target_width,
        target_height,
        block_rows=64,
    )
    geocode_product(
        input_h5,
        output_paths["los_displacement_tif"],
        "los_displacement",
        target_width,
        target_height,
        block_rows=64,
    )
    write_wrapped_phase_geotiff(
        input_h5,
        output_paths["interferogram_tif"],
        dataset_name="interferogram",
        target_width=target_width,
        target_height=target_height,
        block_rows=64,
    )
    write_wrapped_phase_png(
        input_h5,
        output_paths["interferogram_png"],
        dataset_name="interferogram",
        target_width=target_width,
        target_height=target_height,
        block_rows=64,
    )
    write_wrapped_phase_png(
        input_h5,
        output_paths["filtered_interferogram_png"],
        dataset_name="filtered_interferogram",
        target_width=target_width,
        target_height=target_height,
        block_rows=64,
    )
    record = {
        "stage": stage,
        "input_manifests": {
            "master": str(master_manifest_path),
            "slave": str(slave_manifest_path),
        },
        "effective_crop": crop_request,
        "backend_used": backend_used,
        "upstream_stage_dependencies": ["p5"],
        "output_files": output_paths,
        "start_time": started_at,
        "end_time": utc_now_iso(),
        "success": True,
        "fallback_reason": None,
    }
    write_stage_record(output_dir, stage, record)
    mark_stage_success(output_dir, stage)
    return output_paths


def _save_stage_array(output_dir: Path, stage: str, name: str, arr: np.ndarray) -> str:
    path = stage_dir(output_dir, stage)
    path.mkdir(parents=True, exist_ok=True)
    array_path = path / f"{name}.npy"
    np.save(array_path, arr)
    return str(array_path)


def _crop_window(crop_request: dict | None) -> tuple[int, int, int, int] | None:
    if not crop_request:
        return None
    window = crop_request.get("master_window")
    if not window:
        return None
    row0 = int(window.get("row0", 0))
    col0 = int(window.get("col0", 0))
    rows = int(window.get("rows", 0))
    cols = int(window.get("cols", 0))
    if row0 == 0 and col0 == 0 and rows == 0 and cols == 0:
        return None
    return row0, col0, rows, cols


def _apply_crop_to_array(arr: np.ndarray, crop_request: dict | None) -> np.ndarray:
    window = _crop_window(crop_request)
    if window is None:
        return arr
    row0, col0, rows, cols = window
    if rows <= 0 or cols <= 0:
        return arr
    return arr[row0 : row0 + rows, col0 : col0 + cols]


def _apply_crop_to_metadata(
    radargrid_data: dict,
    acquisition_data: dict,
    crop_request: dict | None,
) -> tuple[dict, dict]:
    window = _crop_window(crop_request)
    if window is None:
        return dict(radargrid_data), dict(acquisition_data)

    row0, col0, rows, cols = window
    cropped_rg = dict(radargrid_data)
    cropped_acq = dict(acquisition_data)
    cropped_rg["numberOfRows"] = rows
    cropped_rg["numberOfColumns"] = cols

    column_spacing = float(radargrid_data.get("columnSpacing", 0.0))
    range_time_first = float(radargrid_data.get("rangeTimeFirstPixel", 0.0))
    cropped_rg["rangeTimeFirstPixel"] = range_time_first + (
        2.0 * col0 * column_spacing / 299792458.0
    )

    prf = float(acquisition_data.get("prf", 0.0))
    if prf > 0:
        cropped_acq["startGPSTime"] = float(acquisition_data.get("startGPSTime", 0.0)) + (
            row0 / prf
        )
    return cropped_rg, cropped_acq


def _is_full_crop(crop_request: dict | None) -> bool:
    if not crop_request:
        return True
    return crop_request.get("mode") in {None, "full"}


def _write_stage_outputs_record(
    *,
    output_dir: Path,
    stage: str,
    master_manifest_path: str | Path,
    slave_manifest_path: str | Path,
    crop_request: dict,
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
        "effective_crop": crop_request,
        "backend_used": backend_used,
        "upstream_stage_dependencies": upstream[stage],
        "output_files": output_files,
        "start_time": utc_now_iso(),
        "end_time": utc_now_iso(),
        "success": True,
        "fallback_reason": fallback_reason,
    }
    write_stage_record(output_dir, stage, record)
    mark_stage_success(output_dir, stage)


def _compute_slc_amplitude(slc_path: str) -> np.ndarray:
    """Compute linear amplitude from SLC TIFF."""
    from osgeo import gdal

    ds = gdal.Open(slc_path)
    if ds is None:
        raise RuntimeError(f"failed to open SLC: {slc_path}")
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    print(
        f"[DEBUG] _compute_slc_amplitude: path={slc_path}, cols={cols}, rows={rows}, bands={ds.RasterCount}"
    )

    if cols == 0 or rows == 0:
        raise RuntimeError(f"GDAL opened 0-size raster: {slc_path} ({cols}x{rows})")

    band1 = ds.GetRasterBand(1)
    if ds.RasterCount >= 2:
        band2 = ds.GetRasterBand(2)
        block1 = _read_band_array(band1, 0, 0, cols, rows)
        block2 = _read_band_array(band2, 0, 0, cols, rows)
        amp = np.sqrt(block1.astype(np.float32) ** 2 + block2.astype(np.float32) ** 2)
    else:
        block = _read_band_array(band1, 0, 0, cols, rows)
        amp = np.sqrt(
            block.real.astype(np.float32) ** 2 + block.imag.astype(np.float32) ** 2
        )
    ds = None
    return amp


def _write_radar_amplitude_png(
    slc_path: str,
    output_png: str | Path,
    *,
    block_rows: int = 512,
) -> str:
    """Write a full-resolution radar-domain amplitude PNG from an SLC raster."""
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    ds = gdal.Open(slc_path)
    if ds is None:
        raise RuntimeError(f"failed to open SLC for PNG preview: {slc_path}")

    rows = ds.RasterYSize
    cols = ds.RasterXSize
    if rows <= 0 or cols <= 0:
        ds = None
        raise RuntimeError(f"invalid raster size for PNG preview: {slc_path}")

    band1 = ds.GetRasterBand(1)
    band2 = ds.GetRasterBand(2) if ds.RasterCount >= 2 else None

    stride = max(1, int(np.ceil(max(rows, cols) / 2048)))
    samples: list[np.ndarray] = []
    for row0 in range(0, rows, block_rows):
        nrows = min(block_rows, rows - row0)
        block1 = _read_band_array(band1, 0, row0, cols, nrows)
        if band2 is not None:
            block2 = _read_band_array(band2, 0, row0, cols, nrows)
            amp = np.sqrt(block1.astype(np.float32) ** 2 + block2.astype(np.float32) ** 2)
        else:
            block = block1.astype(np.complex64, copy=False)
            amp = np.abs(block).astype(np.float32, copy=False)
        sampled = amp[::stride, ::stride]
        valid = np.isfinite(sampled) & (sampled > 0)
        if np.any(valid):
            samples.append(sampled[valid].astype(np.float32, copy=False))

    if samples:
        sampled_vals = np.concatenate(samples)
        db_vals = 10.0 * np.log10(sampled_vals)
        p2 = float(np.percentile(db_vals, 2))
        p98 = float(np.percentile(db_vals, 98))
    else:
        p2 = 0.0
        p98 = 1.0

    img = np.zeros((rows, cols), dtype=np.uint8)
    for row0 in range(0, rows, block_rows):
        nrows = min(block_rows, rows - row0)
        block1 = _read_band_array(band1, 0, row0, cols, nrows)
        if band2 is not None:
            block2 = _read_band_array(band2, 0, row0, cols, nrows)
            amp = np.sqrt(block1.astype(np.float32) ** 2 + block2.astype(np.float32) ** 2)
        else:
            block = block1.astype(np.complex64, copy=False)
            amp = np.abs(block).astype(np.float32, copy=False)
        valid = np.isfinite(amp) & (amp > 0)
        block_img = np.zeros((nrows, cols), dtype=np.uint8)
        if np.any(valid):
            vals = 10.0 * np.log10(amp[valid])
            scaled = np.clip((vals - p2) / (p98 - p2 + 1e-9), 0, 1)
            block_img[valid] = (scaled * 255).astype(np.uint8)
        img[row0 : row0 + nrows, :] = block_img

    ds = None
    Image.fromarray(img, mode="L").save(output_png)
    return str(output_png)


def _write_radar_wrapped_phase_png(
    interferogram: np.ndarray,
    output_png: str | Path,
) -> str:
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    rows, cols = interferogram.shape
    stride = max(1, int(np.ceil(max(rows, cols) / 2048)))
    sampled = interferogram[::stride, ::stride]
    phase = np.angle(sampled).astype(np.float32, copy=False)
    valid = np.isfinite(phase)

    hsv = np.zeros((*phase.shape, 3), dtype=np.uint8)
    if np.any(valid):
        hue = np.mod((phase[valid] + np.pi) / (2.0 * np.pi), 1.0)
        hsv[..., 0][valid] = np.rint(hue * 255.0).astype(np.uint8)
        hsv[..., 1][valid] = 255
        hsv[..., 2][valid] = 255

    Image.fromarray(hsv, mode="HSV").convert("RGB").save(output_png)
    return str(output_png)


# ---------------------------------------------------------------------------
# Geocoding helpers (adapted from common_processing)
# ---------------------------------------------------------------------------


def geocode_product(
    input_h5: str,
    output_tif: str,
    dataset_name: str,
    target_width: int,
    target_height: int,
    block_rows: int = 64,
) -> str:
    """Geocode a dataset from HDF5 to GeoTIFF on UTM grid."""
    from osgeo import gdal, osr

    with h5py.File(input_h5, "r") as f:
        x_ds = f["utm_x"]
        y_ds = f["utm_y"]
        prod_ds = f[dataset_name]
        length, width = prod_ds.shape
        utm_epsg = int(f.attrs.get("utm_epsg", 4326))

    # Compute accumulation grid
    acc_out, meta = accumulate_utm_grid(
        input_h5,
        dataset_name=dataset_name,
        target_width=target_width,
        target_height=target_height,
        block_rows=block_rows,
    )

    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(
        output_tif,
        meta["target_width"],
        meta["target_height"],
        1,
        gdal.GDT_Float32,
        options=["COMPRESS=LZW", "TILED=YES"],
    )
    ds.SetGeoTransform(
        [meta["x_min"], meta["x_res"], 0.0, meta["y_max"], 0.0, -meta["y_res"]]
    )
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(meta["utm_epsg"])
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(np.nan)
    _write_band_array(band, acc_out.astype(np.float32))
    band.FlushCache()
    ds.FlushCache()
    ds = None
    return output_tif


# ---------------------------------------------------------------------------
# Processing stages
# ---------------------------------------------------------------------------


def _load_master_metadata(manifest_path: Path) -> tuple[dict, dict, dict, dict, dict]:
    """Load all metadata for master SLC."""
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    with open(
        resolve_manifest_metadata_path(manifest_path, manifest, "orbit"),
    ) as f:
        orbit_data = json.load(f)
    with open(
        resolve_manifest_metadata_path(manifest_path, manifest, "acquisition"),
    ) as f:
        acquisition_data = json.load(f)
    with open(
        resolve_manifest_metadata_path(manifest_path, manifest, "radargrid"),
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
        resolved = resolve_manifest_data_path(manifest_path, manifest_dem)
        if resolved is not None:
            return resolved
    if dem_cache_dir is None:
        dem_cache_dir = str(manifest_path.parent / "dem")
    resolved, _ = resolve_dem_for_scene(
        corners,
        dem_path=dem_path,
        output_dir=dem_cache_dir,
        margin_deg=dem_margin_deg,
    )
    return resolved


def _ceil_to_half_or_int(x: float) -> float:
    import math

    return math.ceil(x * 2) / 2


# ---------------------------------------------------------------------------
# P0: Geo2Rdr coarse coregistration
# ---------------------------------------------------------------------------


def _run_geo2rdr(
    master_manifest_path: str,
    slave_manifest_path: str,
    dem_path: str,
    orbit_interp: str,
    use_gpu: bool,
    gpu_id: int,
    output_dir: Path,
    block_rows: int = 256,
) -> tuple[str, str]:
    """Run Geo2Rdr coarse coregistration for both SLCs.

    Returns paths to (master_topo, slave_topo) TIFF rasters containing
    longitude/latitude/height for each pixel in radar coordinates.
    """
    master_path = Path(master_manifest_path)
    slave_path = Path(slave_manifest_path)

    _, master_orbit_data, master_acq_data, master_rg_data, master_dop_data = (
        _load_master_metadata(master_path)
    )
    _, slave_orbit_data, slave_acq_data, slave_rg_data, slave_dop_data = (
        _load_master_metadata(slave_path)
    )

    import isce3.io

    master_orbit = construct_orbit(master_orbit_data, orbit_interp)
    slave_orbit = construct_orbit(slave_orbit_data, orbit_interp)

    master_grid = construct_radar_grid(
        master_rg_data, master_acq_data, master_orbit_data
    )
    slave_grid = construct_radar_grid(slave_rg_data, slave_acq_data, slave_orbit_data)

    if master_grid.width <= 0 or master_grid.length <= 0:
        raise RuntimeError(
            f"invalid master radar grid dimensions: {master_grid.width}x{master_grid.length}"
        )
    if slave_grid.width <= 0 or slave_grid.length <= 0:
        raise RuntimeError(
            f"invalid slave radar grid dimensions: {slave_grid.width}x{slave_grid.length}"
        )

    print(
        f"[DEBUG] _run_geo2rdr: master_grid={master_grid.width}x{master_grid.length}, "
        f"slave_grid={slave_grid.width}x{slave_grid.length}, dem={dem_path}",
        flush=True,
    )

    dem_raster = isce3.io.Raster(str(dem_path))

    def set_epsg_4326(path: str | Path) -> None:
        ds = gdal.Open(str(path), gdal.GA_Update)
        if ds is None:
            raise RuntimeError(f"failed to reopen topo raster for EPSG metadata: {path}")
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        ds.SetProjection(srs.ExportToWkt())
        ds.FlushCache()
        ds = None

    def run_geo2rdr_single(radar_grid, orbit, doppler_lut, output_tif, is_master: bool):
        if use_gpu:
            import isce3.core
            import isce3.cuda.core
            import isce3.cuda.geometry

            device = isce3.cuda.core.Device(gpu_id)
            isce3.cuda.core.set_device(device)
            topo_cls = isce3.cuda.geometry.Rdr2Geo
        else:
            import isce3.core
            import isce3.geometry

            topo_cls = isce3.geometry.Rdr2Geo

        topo = topo_cls(
            radar_grid,
            orbit,
            isce3.core.Ellipsoid(),
            doppler_lut,
            epsg_out=4326,
            lines_per_block=block_rows,
        )

        tmp_dir = output_dir / f"geo2rdr_{'master' if is_master else 'slave'}"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # Explicit raster outputs are more stable here than the workdir overload.
        lon_raster = _make_raster(
            str(tmp_dir / "lon.tif"),
            gdal.GDT_Float64,
            radar_grid.width,
            radar_grid.length,
        )
        lat_raster = _make_raster(
            str(tmp_dir / "lat.tif"),
            gdal.GDT_Float64,
            radar_grid.width,
            radar_grid.length,
        )
        hgt_raster = _make_raster(
            str(tmp_dir / "hgt.tif"),
            gdal.GDT_Float64,
            radar_grid.width,
            radar_grid.length,
        )

        topo.topo(
            dem_raster,
            lon_raster,
            lat_raster,
            hgt_raster,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

        lon_raster.close_dataset()
        lat_raster.close_dataset()
        hgt_raster.close_dataset()
        for name in ("lon.tif", "lat.tif", "hgt.tif"):
            set_epsg_4326(tmp_dir / name)

        vrt_ds = gdal.BuildVRT(
            str(tmp_dir / "topo.vrt"),
            [str(tmp_dir / "lon.tif"), str(tmp_dir / "lat.tif"), str(tmp_dir / "hgt.tif")],
            separate=True,
        )
        if vrt_ds is not None:
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(4326)
            srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            vrt_ds.SetProjection(srs.ExportToWkt())
        vrt_ds = None

        # Merge into single output TIFF
        out_tif = str(output_dir / f"geo2rdr_{'master' if is_master else 'slave'}.tif")
        _merge_tiffs(
            [str(tmp_dir / p) for p in ["lon.tif", "lat.tif", "hgt.tif"]], out_tif
        )
        set_epsg_4326(out_tif)
        return out_tif

    import isce3.core

    master_doppler = _construct_doppler_if_possible(
        master_dop_data,
        orbit_data=master_orbit_data,
        acquisition_data=master_acq_data,
        radargrid_data=master_rg_data,
    ) or isce3.core.LUT2d()
    slave_doppler = _construct_doppler_if_possible(
        slave_dop_data,
        orbit_data=slave_orbit_data,
        acquisition_data=slave_acq_data,
        radargrid_data=slave_rg_data,
    ) or isce3.core.LUT2d()

    master_out = run_geo2rdr_single(master_grid, master_orbit, master_doppler, "", True)
    slave_out = run_geo2rdr_single(slave_grid, slave_orbit, slave_doppler, "", False)
    dem_raster.close_dataset()
    return master_out, slave_out


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
    if not _isce_driver_write_supported(output_dir):
        raise RuntimeError(
            "coarse geo2rdr sampled-scalar fitting fallback is disabled; "
            "ISCE driver write support is required"
        )

    import isce3.core
    import isce3.io

    slave_orbit = construct_orbit(slave_orbit_data, choose_orbit_interp(slave_orbit_data, slave_acq_data))
    slave_grid = construct_radar_grid(slave_rg_data, slave_acq_data, slave_orbit_data)
    slave_doppler = _construct_doppler_if_possible(
        slave_dop_data,
        orbit_data=slave_orbit_data,
        acquisition_data=slave_acq_data,
        radargrid_data=slave_rg_data,
    )

    # Use ISCE3 default zero-doppler LUT2d. This is more robust across
    # ISCE3 builds than manually constructed LUTs.
    zero_doppler = isce3.core.LUT2d()

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

    artifacts = _p1_coarse_geo2rdr_artifact_paths(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clear stale artifacts from previous runs to avoid confusing reuse of
    # outdated coarse offsets when current run fails.
    for stale in (
        artifacts["range"],
        artifacts["azimuth"],
        artifacts["model"],
        artifacts["error"],
        output_dir / "range.off",
        output_dir / "azimuth.off",
    ):
        try:
            stale.unlink(missing_ok=True)
        except Exception:
            pass

    doppler_mode = str(os.environ.get("D2SAR_GEO2RDR_DOPPLER_MODE", "auto")).strip().lower()
    if doppler_mode not in {"auto", "slave_only", "zero_only"}:
        doppler_mode = "auto"

    doppler_candidates: list[tuple[str, object]] = []
    if doppler_mode == "zero_only":
        doppler_candidates.append(("zero_doppler", zero_doppler))
    elif doppler_mode == "slave_only":
        if slave_doppler is None:
            raise RuntimeError(
                "D2SAR_GEO2RDR_DOPPLER_MODE=slave_only but slave doppler is unavailable"
            )
        doppler_candidates.append(("slave_doppler", slave_doppler))
    else:
        if slave_doppler is not None:
            doppler_candidates.append(("slave_doppler", slave_doppler))
        # Fallback candidate for problematic LUT cases.
        doppler_candidates.append(("zero_doppler", zero_doppler))

    errors: list[str] = []
    for doppler_label, doppler_to_use in doppler_candidates:
        try:
            geo2rdr = geo2rdr_cls(
                slave_grid,
                slave_orbit,
                isce3.core.Ellipsoid(),
                doppler_to_use,
                threshold=1.0e-8,
                numiter=50,
                lines_per_block=block_rows,
            )
            geo2rdr.geo2rdr(topo_raster, str(output_dir))
            _move_geo2rdr_output_if_needed(output_dir / "range.off", artifacts["range"])
            _move_geo2rdr_output_if_needed(output_dir / "azimuth.off", artifacts["azimuth"])

            # Record a read-only interpretation diagnostic for the Geo2Rdr
            # outputs, but keep the files unchanged to match upstream ISCE3
            # semantics.
            conversion_decisions = _convert_geo2rdr_abs_to_relative_offsets(
                range_path=artifacts["range"],
                azimuth_path=artifacts["azimuth"],
                rows=slave_grid.length,
                cols=slave_grid.width,
            )

            artifacts["model"].write_text(
                json.dumps(
                    {
                        "method": "isce3-full-resolution-geo2rdr",
                        "sign_convention": "slave_minus_master_pixel_index",
                        "topographic_phase_removal": "full_resolution_range_offset",
                        "doppler_source": doppler_label,
                        "doppler_retry_count": len(errors),
                        "geo2rdr_output_interpretation": conversion_decisions,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            try:
                artifacts["error"].unlink(missing_ok=True)
            except Exception:
                pass
            return str(artifacts["range"]), str(artifacts["azimuth"])
        except Exception as exc:
            errors.append(f"{doppler_label}: {type(exc).__name__}: {exc}")

    artifacts["error"].write_text("\n".join(errors) + "\n", encoding="utf-8")
    raise RuntimeError(
        "coarse geo2rdr failed across doppler candidates; sampled-scalar fitting fallback is disabled"
    )


def _isce_driver_write_supported(output_dir: Path) -> bool:
    probe_dir = output_dir / ".isce_driver_probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    probe_path = probe_dir / "probe.off"
    script = (
        "from osgeo import gdal; import isce3.io; "
        "r=isce3.io.Raster(r'%s',1,1,1,gdal.GDT_Float64,'ISCE'); "
        "print('ok')"
    ) % str(probe_path)
    result = subprocess.run(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.returncode == 0


def _p1_coarse_geo2rdr_artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "range": output_dir / "coarse_geo2rdr_range.off",
        "azimuth": output_dir / "coarse_geo2rdr_azimuth.off",
        "model": output_dir / "coarse_geo2rdr_model.json",
        "error": output_dir / "coarse_geo2rdr_error.txt",
    }


def _sample_geo2rdr_offset_values(
    path: Path,
    rows: int,
    cols: int,
    *,
    sample_rows: int = 64,
    sample_cols: int = 96,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_idx = np.linspace(0, max(0, rows - 1), num=max(1, min(rows, sample_rows)), dtype=np.int64)
    col_idx = np.linspace(0, max(0, cols - 1), num=max(1, min(cols, sample_cols)), dtype=np.int64)
    expected_bytes = rows * cols * np.dtype(np.float64).itemsize
    if path.is_file() and path.stat().st_size == expected_bytes:
        arr = np.memmap(str(path), dtype=np.float64, mode="r", shape=(rows, cols))
        try:
            sample = np.asarray(arr[np.ix_(row_idx, col_idx)], dtype=np.float64)
        finally:
            del arr
        return sample, row_idx, col_idx

    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"unable to open geo2rdr offset for sampling: {path}")
    try:
        band = ds.GetRasterBand(1)
        sample = np.full((len(row_idx), len(col_idx)), np.nan, dtype=np.float64)
        for i, row in enumerate(row_idx.tolist()):
            row_block = _read_band_array(band, 0, int(row), cols, 1, dtype=np.float64)
            sample[i, :] = np.asarray(row_block, dtype=np.float64)[0, col_idx]
    finally:
        ds = None
    return sample, row_idx, col_idx


def _detect_geo2rdr_offset_mode(
    path: Path,
    rows: int,
    cols: int,
    *,
    subtract_cols: bool,
) -> dict:
    """Infer whether a Geo2Rdr output file stores absolute or relative coordinates."""
    try:
        sample, row_idx, col_idx = _sample_geo2rdr_offset_values(path, rows, cols)
    except Exception as exc:
        return {
            "mode": "unknown",
            "should_subtract": True,
            "reason": f"sampling_failed: {exc}",
            "confidence": 0.0,
        }

    if subtract_cols:
        idx_grid = np.broadcast_to(col_idx[None, :], sample.shape)
        axis_name = "range"
    else:
        idx_grid = np.broadcast_to(row_idx[:, None], sample.shape)
        axis_name = "azimuth"

    valid = np.isfinite(sample)
    valid &= sample != GEO2RDR_OFFSET_NODATA
    valid &= sample >= GEO2RDR_OFFSET_INVALID_LOW
    valid &= sample <= 1.0e5

    valid_count = int(np.count_nonzero(valid))
    # For tiny diagnostic rasters, require enough valid samples to make a
    # decision but do not force the full-sized threshold used for real scenes.
    min_valid_samples = min(64, max(6, sample.size))
    if valid_count < min_valid_samples:
        return {
            "mode": "unknown",
            "should_subtract": True,
            "reason": "insufficient_valid_samples",
            "valid_samples": valid_count,
            "required_valid_samples": int(min_valid_samples),
            "confidence": 0.0,
        }

    vals = sample[valid]
    idx = idx_grid[valid]
    residual = vals - idx
    mean_abs_vals = float(np.mean(np.abs(vals)))
    mean_abs_residual = float(np.mean(np.abs(residual)))
    std_vals = float(np.std(vals))
    std_residual = float(np.std(residual))
    centered_idx = idx - float(np.mean(idx))
    centered_vals = vals - float(np.mean(vals))
    denom = float(np.dot(centered_idx, centered_idx))
    slope = float(np.dot(centered_idx, centered_vals) / denom) if denom > 0.0 else float("nan")

    # Small diagnostic rasters can contain nearly constant relative offsets,
    # especially for azimuth. In that case subtracting the master index
    # injects most of the observed variation, so prefer "relative".
    if (
        np.isfinite(slope)
        and abs(slope) < 0.2
        and mean_abs_vals < 500.0
        and std_vals < max(1.0, std_residual * 0.25)
    ):
        return {
            "axis": axis_name,
            "mode": "relative",
            "should_subtract": False,
            "confidence": 1.0,
            "valid_samples": valid_count,
            "slope_vs_master_index": slope,
            "mean_abs_value": mean_abs_vals,
            "mean_abs_value_minus_master_index": mean_abs_residual,
            "std_value": std_vals,
            "std_value_minus_master_index": std_residual,
            "vote_absolute": 0,
            "vote_relative": 3,
            "reason": "low_variance_relative_offsets",
        }

    absolute_votes = 0
    relative_votes = 0
    if np.isfinite(slope) and abs(slope - 1.0) < 0.35:
        absolute_votes += 2
    if np.isfinite(slope) and abs(slope) < 0.2:
        relative_votes += 2
    if mean_abs_residual < max(10.0, mean_abs_vals * 0.35):
        absolute_votes += 2
    if mean_abs_vals < 500.0 and mean_abs_residual > max(200.0, mean_abs_vals * 3.0):
        relative_votes += 2
    if std_residual < max(5.0, std_vals * 0.35):
        absolute_votes += 1
    if std_vals < 500.0 and std_residual > max(50.0, std_vals * 3.0):
        relative_votes += 1

    if absolute_votes > relative_votes:
        mode = "absolute"
        should_subtract = True
    elif relative_votes > absolute_votes:
        mode = "relative"
        should_subtract = False
    else:
        # Tie-breaker: whichever representation has smaller magnitude is likely true.
        should_subtract = mean_abs_residual <= mean_abs_vals
        mode = "absolute" if should_subtract else "relative"

    confidence = float(abs(absolute_votes - relative_votes) / 5.0)
    confidence = max(0.0, min(1.0, confidence))
    return {
        "axis": axis_name,
        "mode": mode,
        "should_subtract": should_subtract,
        "confidence": confidence,
        "valid_samples": valid_count,
        "slope_vs_master_index": slope,
        "mean_abs_value": mean_abs_vals,
        "mean_abs_value_minus_master_index": mean_abs_residual,
        "std_value": std_vals,
        "std_value_minus_master_index": std_residual,
        "vote_absolute": absolute_votes,
        "vote_relative": relative_votes,
    }


def _convert_geo2rdr_abs_to_relative_offsets(
    range_path: Path,
    azimuth_path: Path,
    rows: int,
    cols: int,
) -> dict:
    """Diagnose whether Geo2Rdr outputs look absolute or relative.

    This function is intentionally read-only. Upstream ISCE3 Geo2Rdr already
    writes offsets suitable for downstream resample/flatten use, so D2SAR only
    records the interpretation for logging and troubleshooting.
    """
    decisions: dict[str, dict] = {}
    for axis_name, path, subtract_cols in (
        ("range", range_path, True),
        ("azimuth", azimuth_path, False),
    ):
        if not path.is_file():
            decisions[axis_name] = {
                "axis": axis_name,
                "mode": "missing",
                "should_subtract": False,
                "confidence": 0.0,
            }
            continue
        decision = _detect_geo2rdr_offset_mode(
            path,
            rows,
            cols,
            subtract_cols=subtract_cols,
        )
        decisions[axis_name] = decision
    return decisions


def _move_geo2rdr_output_if_needed(source: Path, target: Path) -> None:
    if source == target or not source.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    source.replace(target)
    for suffix in (".xml", ".vrt", ".hdr"):
        sidecar = Path(f"{source}{suffix}")
        if sidecar.exists():
            sidecar.replace(Path(f"{target}{suffix}"))


def _sample_indices(length: int, target_count: int = 48) -> np.ndarray:
    if length <= 1:
        return np.array([0], dtype=np.int32)
    step = max(1, int(np.ceil(length / max(2, target_count))))
    indices = np.arange(0, length, step, dtype=np.int32)
    if indices[-1] != length - 1:
        indices = np.concatenate([indices, np.array([length - 1], dtype=np.int32)])
    return np.unique(indices)


def _fit_offset_surface(
    sample_rows: np.ndarray,
    sample_cols: np.ndarray,
    sample_offsets: np.ndarray,
) -> tuple[np.ndarray, dict]:
    yy, xx = np.meshgrid(sample_rows, sample_cols, indexing="ij")
    valid = np.isfinite(sample_offsets)
    if int(valid.sum()) < 6:
        raise RuntimeError("insufficient valid geo2rdr samples for coarse offset model")

    y = yy[valid].astype(np.float64)
    x = xx[valid].astype(np.float64)
    z = sample_offsets[valid].astype(np.float64)
    design = np.column_stack(
        [
            np.ones_like(x),
            x,
            y,
            x * y,
            x * x,
            y * y,
        ]
    )
    coeffs, *_ = np.linalg.lstsq(design, z, rcond=None)
    residual = z - (design @ coeffs)
    diagnostics = {
        "sample_count": int(sample_offsets.size),
        "valid_count": int(valid.sum()),
        "rmse_pixels": float(np.sqrt(np.mean(residual * residual))) if residual.size else 0.0,
        "max_abs_pixels": float(np.max(np.abs(residual))) if residual.size else 0.0,
        "coefficients": [float(v) for v in coeffs],
    }
    return coeffs, diagnostics


def _evaluate_offset_surface(
    coeffs: np.ndarray,
    row_values: np.ndarray,
    col_values: np.ndarray,
) -> np.ndarray:
    rr = row_values[:, None].astype(np.float64)
    cc = col_values[None, :].astype(np.float64)
    return (
        coeffs[0]
        + coeffs[1] * cc
        + coeffs[2] * rr
        + coeffs[3] * rr * cc
        + coeffs[4] * cc * cc
        + coeffs[5] * rr * rr
    )


def _geo2rdr_pixel_offsets(
    *,
    master_col: float,
    master_row: float,
    slave_col: float,
    slave_row: float,
) -> tuple[float, float]:
    return float(slave_col) - float(master_col), float(slave_row) - float(master_row)


def _write_sampled_geo2rdr_offsets(
    *,
    master_topo_path: str,
    slave_orbit_data: dict,
    slave_acq_data: dict,
    slave_rg_data: dict,
    slave_dop_data: dict | None,
    output_dir: Path,
    sample_target_count: int = 48,
    block_rows: int = 256,
) -> tuple[str, str]:
    import isce3.core
    import isce3.geometry

    topo_ds = gdal.Open(str(master_topo_path), gdal.GA_ReadOnly)
    if topo_ds is None or topo_ds.RasterCount < 3:
        raise RuntimeError(f"unable to open master topo raster: {master_topo_path}")
    try:
        rows = topo_ds.RasterYSize
        cols = topo_ds.RasterXSize
        lon_band = topo_ds.GetRasterBand(1)
        lat_band = topo_ds.GetRasterBand(2)
        hgt_band = topo_ds.GetRasterBand(3)

        sample_rows = _sample_indices(rows, sample_target_count)
        sample_cols = _sample_indices(cols, sample_target_count)
        rg_samples = np.full((len(sample_rows), len(sample_cols)), np.nan, dtype=np.float64)
        az_samples = np.full((len(sample_rows), len(sample_cols)), np.nan, dtype=np.float64)

        slave_orbit = construct_orbit(
            slave_orbit_data,
            choose_orbit_interp(slave_orbit_data, slave_acq_data),
        )
        slave_grid = construct_radar_grid(slave_rg_data, slave_acq_data, slave_orbit_data)
        slave_doppler = _construct_doppler_if_possible(
            slave_dop_data,
            orbit_data=slave_orbit_data,
            acquisition_data=slave_acq_data,
            radargrid_data=slave_rg_data,
        ) or isce3.core.LUT2d()
        ellipsoid = isce3.core.Ellipsoid()
        sensing_start = float(slave_grid.sensing_start)
        sensing_end = sensing_start + (float(slave_grid.length) - 1.0) / float(slave_grid.prf)
        range_start = float(slave_grid.starting_range)
        range_end = range_start + (float(slave_grid.width) - 1.0) * float(slave_grid.range_pixel_spacing)

        for row_idx, row in enumerate(sample_rows):
            lon_row = _read_band_array(lon_band, 0, int(row), cols, 1, dtype=np.float64).astype(np.float64)[0, sample_cols]
            lat_row = _read_band_array(lat_band, 0, int(row), cols, 1, dtype=np.float64).astype(np.float64)[0, sample_cols]
            hgt_row = _read_band_array(hgt_band, 0, int(row), cols, 1, dtype=np.float64).astype(np.float64)[0, sample_cols]
            for col_idx, col in enumerate(sample_cols):
                try:
                    llh = np.array(
                        [
                            np.deg2rad(lon_row[col_idx]),
                            np.deg2rad(lat_row[col_idx]),
                            hgt_row[col_idx],
                        ],
                        dtype=np.float64,
                    )
                    aztime, slant_range = isce3.geometry.geo2rdr(
                        llh,
                        ellipsoid,
                        slave_orbit,
                        slave_doppler,
                        slave_grid.wavelength,
                        slave_grid.lookside,
                        threshold=1.0e-8,
                        maxiter=50,
                        delta_range=10.0,
                    )
                except Exception:
                    continue

                if not (sensing_start <= aztime <= sensing_end):
                    continue
                if not (range_start <= slant_range <= range_end):
                    continue

                slave_col = (
                    (slant_range - range_start) / float(slave_grid.range_pixel_spacing)
                )
                slave_row = (aztime - sensing_start) * float(slave_grid.prf)
                rg_samples[row_idx, col_idx], az_samples[row_idx, col_idx] = _geo2rdr_pixel_offsets(
                    master_col=float(col),
                    master_row=float(row),
                    slave_col=slave_col,
                    slave_row=slave_row,
                )

        rg_coeffs, rg_diag = _fit_offset_surface(sample_rows, sample_cols, rg_samples)
        az_coeffs, az_diag = _fit_offset_surface(sample_rows, sample_cols, az_samples)

        output_dir.mkdir(parents=True, exist_ok=True)
        artifacts = _p1_coarse_geo2rdr_artifact_paths(output_dir)
        rg_path = artifacts["range"]
        az_path = artifacts["azimuth"]
        rg_memmap = np.memmap(rg_path, dtype=np.float64, mode="w+", shape=(rows, cols))
        az_memmap = np.memmap(az_path, dtype=np.float64, mode="w+", shape=(rows, cols))
        col_values = np.arange(cols, dtype=np.float64)
        for row_start in range(0, rows, block_rows):
            row_stop = min(rows, row_start + block_rows)
            row_values = np.arange(row_start, row_stop, dtype=np.float64)
            rg_memmap[row_start:row_stop, :] = _evaluate_offset_surface(
                rg_coeffs,
                row_values,
                col_values,
            )
            az_memmap[row_start:row_stop, :] = _evaluate_offset_surface(
                az_coeffs,
                row_values,
                col_values,
            )
        rg_memmap.flush()
        az_memmap.flush()
        del rg_memmap
        del az_memmap

        diagnostics = {
            "method": "sampled-scalar-geo2rdr",
            "sign_convention": "slave_minus_master_pixel_index",
            "sample_grid": {
                "rows": sample_rows.tolist(),
                "cols": sample_cols.tolist(),
            },
            "range_model": rg_diag,
            "azimuth_model": az_diag,
        }
        artifacts["model"].write_text(
            json.dumps(diagnostics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return str(rg_path), str(az_path)
    finally:
        topo_ds = None


def _make_raster(
    path: str, dtype: int = gdal.GDT_Float32, width: int = 0, length: int = 0
):
    import isce3.io

    return isce3.io.Raster(path, width, length, 1, dtype, "GTiff")


def _merge_tiffs(input_tifs: list[str], output_tif: str):
    """Merge multiple single-band TIFFs into a multi-band TIFF."""
    from osgeo import gdal

    if not input_tifs:
        raise ValueError("input_tifs must not be empty")

    src_datasets = []
    try:
        for tif in input_tifs:
            ds = gdal.Open(tif, gdal.GA_ReadOnly)
            if ds is None:
                raise RuntimeError(f"failed to open input TIFF for merge: {tif}")
            src_datasets.append(ds)

        width = src_datasets[0].RasterXSize
        height = src_datasets[0].RasterYSize
        for tif, ds in zip(input_tifs, src_datasets):
            if ds.RasterXSize != width or ds.RasterYSize != height:
                raise RuntimeError(
                    "input TIFF dimensions do not match for merge: "
                    f"{tif}={ds.RasterYSize}x{ds.RasterXSize}, "
                    f"expected={height}x{width}"
                )

        drv = gdal.GetDriverByName("GTiff")
        out_ds = drv.Create(
            output_tif,
            width,
            height,
            len(src_datasets),
            gdal.GDT_Float32,
            options=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER"],
        )
        if out_ds is None:
            raise RuntimeError(f"failed to create merged TIFF: {output_tif}")

        block_rows = 512
        for band_index, src_ds in enumerate(src_datasets, start=1):
            src_band = src_ds.GetRasterBand(1)
            dst_band = out_ds.GetRasterBand(band_index)
            for row0 in range(0, height, block_rows):
                nrows = min(block_rows, height - row0)
                chunk = src_band.ReadRaster(
                    0,
                    row0,
                    width,
                    nrows,
                    buf_xsize=width,
                    buf_ysize=nrows,
                    buf_type=gdal.GDT_Float32,
                )
                if chunk is None:
                    raise RuntimeError(f"failed to read block from input TIFF: {input_tifs[band_index - 1]}")
                dst_band.WriteRaster(
                    0,
                    row0,
                    width,
                    nrows,
                    chunk,
                    buf_xsize=width,
                    buf_ysize=nrows,
                    buf_type=gdal.GDT_Float32,
                )
            dst_band.FlushCache()
        out_ds.FlushCache()
        out_ds = None
    finally:
        for ds in src_datasets:
            ds = None


# ---------------------------------------------------------------------------
# P1: PyCuAmpcor dense matching
# ---------------------------------------------------------------------------


def _run_pycuampcor(
    master_slc_path: str,
    slave_slc_path: str,
    master_topo_path: str,
    slave_topo_path: str,
    output_dir: Path,
    gpu_id: int = 0,
    return_details: bool = False,
    window_size: tuple[int, int] = (64, 64),
    search_range: tuple[int, int] = (20, 20),
    skip: tuple[int, int] = (32, 32),
    gross_offset_filepath: str | Path | None = None,
    reference_start_pixel_down: int | None = None,
    reference_start_pixel_across: int | None = None,
    number_window_down: int | None = None,
    number_window_across: int | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    return run_pycuampcor_dense_offsets(
        master_slc_path=master_slc_path,
        slave_slc_path=slave_slc_path,
        output_dir=output_dir,
        gpu_id=gpu_id,
        return_details=return_details,
        window_size=window_size,
        search_range=search_range,
        skip=skip,
        gross_offset_filepath=gross_offset_filepath,
        reference_start_pixel_down=reference_start_pixel_down,
        reference_start_pixel_across=reference_start_pixel_across,
        number_window_down=number_window_down,
        number_window_across=number_window_across,
    )


def _compute_gross_offset(
    master_topo_path: str, slave_topo_path: str
) -> tuple[int, int]:
    """Compute approximate gross offset between master and slave from topo geometry.

    This is a simple approximation using corner positions.
    """
    from osgeo import gdal

    def read_corners(topo_path):
        ds = gdal.Open(topo_path)
        if ds is None:
            return None
        # Assume multi-band: band1=lon, band2=lat, band3=hgt
        lon = _read_band_array(ds.GetRasterBand(1), dtype=np.float32)
        lat = _read_band_array(ds.GetRasterBand(2), dtype=np.float32)
        ds = None
        # Use center pixel as reference
        cy, cx = lon.shape[0] // 2, lon.shape[1] // 2
        return float(lon[cy, cx]), float(lat[cy, cx])

    mc = read_corners(master_topo_path)
    sc = read_corners(slave_topo_path)
    if mc is None or sc is None:
        return 0, 0

    # Very rough estimate: ignore for now
    return 0, 0


# ---------------------------------------------------------------------------
# P2: Crossmul interferogram + coherence
# ---------------------------------------------------------------------------


def _run_crossmul(
    master_slc_path: str,
    slave_slc_path: str,
    use_gpu: bool,
    gpu_id: int,
    output_dir: Path,
    block_rows: int = 256,
    *,
    flatten_raster: str | None = None,
    flatten_mask_raster: str | None = None,
    range_pixel_spacing: float | None = None,
    wavelength: float | None = None,
    flatten_source: str | None = None,
    flatten_model: str | None = None,
    flatten_source_mode: str | None = None,
    flatten_starting_range_shift_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray, str, str | None]:
    fallback_reason = None
    if use_gpu:
        if flatten_raster is not None and flatten_mask_raster is not None:
            fallback_reason = (
                "flatten mask mode requires masked phase application; "
                "CUDA Crossmul path does not support mask input, using CPU crossmul"
            )
            print(f"[STRIP_INSAR] {fallback_reason}")
            try:
                fallback_path = stage_dir(output_dir, "p2") / "gpu_fallback_reason.txt"
                fallback_path.parent.mkdir(parents=True, exist_ok=True)
                fallback_path.write_text(fallback_reason + "\n", encoding="utf-8")
            except Exception:
                pass
        elif not _is_experimental_gpu_crossmul_enabled():
            fallback_reason = (
                "ISCE3 CUDA Crossmul is experimental and disabled by default; "
                f"set {EXPERIMENTAL_GPU_CROSSMUL_ENV}=1 to enable it"
            )
            print(f"[STRIP_INSAR] {fallback_reason}; using pure Python crossmul")
            try:
                fallback_path = stage_dir(output_dir, "p2") / "gpu_fallback_reason.txt"
                fallback_path.parent.mkdir(parents=True, exist_ok=True)
                fallback_path.write_text(fallback_reason + "\n", encoding="utf-8")
            except Exception:
                pass
        else:
            try:
                import isce3.cuda.core
                import isce3.cuda.signal

                device = isce3.cuda.core.Device(gpu_id)
                isce3.cuda.core.set_device(device)

                int_gpu, coh_gpu = _crossmul_isce3_gpu(
                    master_slc_path,
                    slave_slc_path,
                    output_dir,
                    gpu_id,
                    block_rows,
                    flatten_raster=flatten_raster,
                    range_pixel_spacing=range_pixel_spacing,
                    wavelength=wavelength,
                    flatten_starting_range_shift_m=flatten_starting_range_shift_m,
                )
                coh_gpu = _estimate_coherence_from_slcs(
                    master_slc_path,
                    slave_slc_path,
                    block_rows=block_rows,
                )
                try:
                    (stage_dir(output_dir, "p2") / "gpu_fallback_reason.txt").unlink(missing_ok=True)
                except Exception:
                    pass
                return int_gpu, coh_gpu, "gpu", None
            except Exception as exc:
                fallback_reason = str(exc)
                print(
                    f"[STRIP_INSAR] ISCE3 CUDA Crossmul unavailable ({exc}); "
                    "using pure Python crossmul"
                )
                try:
                    fallback_path = stage_dir(output_dir, "p2") / "gpu_fallback_reason.txt"
                    fallback_path.parent.mkdir(parents=True, exist_ok=True)
                    fallback_path.write_text(fallback_reason + "\n", encoding="utf-8")
                except Exception:
                    pass

    int_cpu, coh_cpu = _crossmul_numpy(
        master_slc_path,
        slave_slc_path,
        output_dir,
        block_rows,
        flatten_raster=flatten_raster,
        flatten_mask_raster=flatten_mask_raster,
        range_pixel_spacing=range_pixel_spacing,
        wavelength=wavelength,
        flatten_starting_range_shift_m=flatten_starting_range_shift_m,
    )
    return int_cpu, coh_cpu, "cpu", fallback_reason


def _crossmul_numpy(
    master_slc_path: str,
    slave_slc_path: str,
    output_dir: Path,
    block_rows: int = 256,
    *,
    flatten_raster: str | None = None,
    flatten_mask_raster: str | None = None,
    range_pixel_spacing: float | None = None,
    wavelength: float | None = None,
    flatten_starting_range_shift_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    master_local, master_cleanup = _ensure_local_tiff(master_slc_path)
    slave_local, slave_cleanup = _ensure_local_tiff(slave_slc_path)
    if master_local is None or slave_local is None:
        raise RuntimeError(
            f"failed to extract SLC files for crossmul: master={master_slc_path}, slave={slave_slc_path}"
        )

    master_ds = gdal.Open(master_local)
    slave_ds = gdal.Open(slave_local)
    if master_ds is None or slave_ds is None:
        if master_cleanup:
            master_cleanup()
        if slave_cleanup:
            slave_cleanup()
        raise RuntimeError(
            f"failed to open SLC files for crossmul: master={master_slc_path}, slave={slave_slc_path}"
        )
    width = master_ds.RasterXSize
    length = master_ds.RasterYSize
    print(
        f"[DEBUG] _crossmul_numpy: master={master_slc_path}, slave={slave_slc_path}, width={width}, length={length}"
    )

    interferogram = np.zeros((length, width), dtype=np.complex64)

    flatten_ds = None
    flatten_band = None
    flatten_mask_ds = None
    flatten_mask_band = None
    if flatten_raster is not None:
        if range_pixel_spacing is None or wavelength is None:
            raise ValueError("range_pixel_spacing and wavelength are required for crossmul flattening")
        flatten_ds = gdal.Open(str(flatten_raster), gdal.GA_ReadOnly)
        if flatten_ds is None:
            raise RuntimeError(f"failed to open flatten range offset raster: {flatten_raster}")
        if flatten_ds.RasterXSize != width or flatten_ds.RasterYSize != length:
            raise RuntimeError(
                "flatten range offset dimensions do not match crossmul inputs: "
                f"{flatten_raster}={flatten_ds.RasterYSize}x{flatten_ds.RasterXSize}, "
                f"slc={length}x{width}"
            )
        flatten_band = flatten_ds.GetRasterBand(1)
        if flatten_mask_raster is not None:
            flatten_mask_ds = gdal.Open(str(flatten_mask_raster), gdal.GA_ReadOnly)
            if flatten_mask_ds is None:
                raise RuntimeError(f"failed to open flatten valid-mask raster: {flatten_mask_raster}")
            if flatten_mask_ds.RasterXSize != width or flatten_mask_ds.RasterYSize != length:
                raise RuntimeError(
                    "flatten valid-mask dimensions do not match crossmul inputs: "
                    f"{flatten_mask_raster}={flatten_mask_ds.RasterYSize}x{flatten_mask_ds.RasterXSize}, "
                    f"slc={length}x{width}"
                )
            flatten_mask_band = flatten_mask_ds.GetRasterBand(1)

    try:
        for row0 in range(0, length, block_rows):
            rows = min(block_rows, length - row0)

            m_band1 = _read_band_array(master_ds.GetRasterBand(1), 0, row0, width, rows)
            s_band1 = _read_band_array(slave_ds.GetRasterBand(1), 0, row0, width, rows)

            if master_ds.RasterCount >= 2:
                m_band2 = _read_band_array(master_ds.GetRasterBand(2), 0, row0, width, rows)
                s_band2 = _read_band_array(slave_ds.GetRasterBand(2), 0, row0, width, rows)
                m_complex = (m_band1 + 1j * m_band2).astype(np.complex64)
                s_complex = (s_band1 + 1j * s_band2).astype(np.complex64)
            else:
                m_complex = m_band1.astype(np.complex64)
                s_complex = s_band1.astype(np.complex64)

            block_int = (m_complex * np.conj(s_complex)).astype(np.complex64)
            if flatten_band is not None:
                range_offset = _read_band_array(
                    flatten_band, 0, row0, width, rows, dtype=np.float32
                ).astype(np.float32)
                valid_mask = None
                if flatten_mask_band is not None:
                    valid_mask = (
                        _read_band_array(
                            flatten_mask_band,
                            0,
                            row0,
                            width,
                            rows,
                            dtype=np.uint8,
                        ).astype(np.uint8)
                        > 0
                    )
                start_shift_m = float(flatten_starting_range_shift_m or 0.0)
                flat_phase = (
                    4.0
                    * np.pi
                    * (float(range_pixel_spacing) * range_offset + start_shift_m)
                    / float(wavelength)
                )
                flat_factor = np.exp(-1j * flat_phase).astype(np.complex64)
                if valid_mask is None:
                    block_int *= flat_factor
                elif np.any(valid_mask):
                    block_int[valid_mask] *= flat_factor[valid_mask]
            interferogram[row0 : row0 + rows] = block_int
        coherence = _estimate_coherence_from_datasets(
            master_ds,
            slave_ds,
            block_rows=block_rows,
        )
    finally:
        flatten_ds = None
        flatten_mask_ds = None

    master_ds = None
    slave_ds = None
    if master_cleanup:
        master_cleanup()
    if slave_cleanup:
        slave_cleanup()
    return interferogram, coherence


def _read_slc_block_as_complex(
    dataset,
    row0: int,
    rows: int,
    width: int,
) -> np.ndarray:
    band1 = _read_band_array(dataset.GetRasterBand(1), 0, row0, width, rows)
    if dataset.RasterCount >= 2:
        band2 = _read_band_array(dataset.GetRasterBand(2), 0, row0, width, rows)
        return (band1 + 1j * band2).astype(np.complex64)
    return band1.astype(np.complex64)


def _estimate_coherence_from_complex_slcs(
    master_complex: np.ndarray,
    slave_complex: np.ndarray,
    window_size: int = 5,
) -> np.ndarray:
    from scipy.ndimage import uniform_filter

    if master_complex.shape != slave_complex.shape:
        raise ValueError("master_complex and slave_complex must have the same shape")
    if master_complex.ndim != 2 or slave_complex.ndim != 2:
        raise ValueError("coherence estimation expects 2D complex arrays")
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")

    interferogram = master_complex * np.conj(slave_complex)
    master_power = np.abs(master_complex) ** 2
    slave_power = np.abs(slave_complex) ** 2
    pad = window_size // 2

    if pad == 0:
        denom = np.sqrt(master_power * slave_power)
        coherence = np.zeros(master_complex.shape, dtype=np.float32)
        valid = denom > 0
        coherence[valid] = np.abs(interferogram[valid]) / denom[valid]
        return np.clip(coherence, 0.0, 1.0).astype(np.float32)

    real_pad = np.pad(interferogram.real, pad, mode="edge")
    imag_pad = np.pad(interferogram.imag, pad, mode="edge")
    master_power_pad = np.pad(master_power, pad, mode="edge")
    slave_power_pad = np.pad(slave_power, pad, mode="edge")

    mean_real = uniform_filter(real_pad, size=window_size, mode="constant")[pad:-pad, pad:-pad]
    mean_imag = uniform_filter(imag_pad, size=window_size, mode="constant")[pad:-pad, pad:-pad]
    mean_master_power = uniform_filter(master_power_pad, size=window_size, mode="constant")[
        pad:-pad, pad:-pad
    ]
    mean_slave_power = uniform_filter(slave_power_pad, size=window_size, mode="constant")[
        pad:-pad, pad:-pad
    ]

    numerator = np.sqrt(mean_real**2 + mean_imag**2)
    denominator = np.sqrt(mean_master_power * mean_slave_power)
    coherence = np.zeros(master_complex.shape, dtype=np.float32)
    valid = denominator > 0
    coherence[valid] = numerator[valid] / denominator[valid]
    return np.clip(coherence, 0.0, 1.0).astype(np.float32)


def _estimate_coherence_from_datasets(
    master_ds,
    slave_ds,
    *,
    block_rows: int = 256,
    window_size: int = 5,
) -> np.ndarray:
    if master_ds is None or slave_ds is None:
        raise ValueError("master_ds and slave_ds are required")
    width = int(master_ds.RasterXSize)
    length = int(master_ds.RasterYSize)
    if int(slave_ds.RasterXSize) != width or int(slave_ds.RasterYSize) != length:
        raise RuntimeError(
            f"coherence input dimensions differ: master={length}x{width}, slave={slave_ds.RasterYSize}x{slave_ds.RasterXSize}"
        )

    coherence = np.zeros((length, width), dtype=np.float32)
    pad = window_size // 2
    for row0 in range(0, length, block_rows):
        rows = min(block_rows, length - row0)
        read_row0 = max(0, row0 - pad)
        read_row1 = min(length, row0 + rows + pad)
        read_rows = read_row1 - read_row0
        master_complex = _read_slc_block_as_complex(master_ds, read_row0, read_rows, width)
        slave_complex = _read_slc_block_as_complex(slave_ds, read_row0, read_rows, width)
        coherence_ext = _estimate_coherence_from_complex_slcs(
            master_complex,
            slave_complex,
            window_size=window_size,
        )
        local_row0 = row0 - read_row0
        coherence[row0 : row0 + rows] = coherence_ext[local_row0 : local_row0 + rows]
    return coherence


def _estimate_coherence_from_slcs(
    master_slc_path: str,
    slave_slc_path: str,
    *,
    block_rows: int = 256,
    window_size: int = 5,
) -> np.ndarray:
    master_local, master_cleanup = _ensure_local_tiff(master_slc_path)
    slave_local, slave_cleanup = _ensure_local_tiff(slave_slc_path)
    if master_local is None or slave_local is None:
        raise RuntimeError(
            f"failed to extract SLC files for coherence estimation: master={master_slc_path}, slave={slave_slc_path}"
        )

    master_ds = gdal.Open(master_local)
    slave_ds = gdal.Open(slave_local)
    if master_ds is None or slave_ds is None:
        if master_cleanup:
            master_cleanup()
        if slave_cleanup:
            slave_cleanup()
        raise RuntimeError(
            f"failed to open SLC files for coherence estimation: master={master_slc_path}, slave={slave_slc_path}"
        )

    try:
        return _estimate_coherence_from_datasets(
            master_ds,
            slave_ds,
            block_rows=block_rows,
            window_size=window_size,
        )
    finally:
        master_ds = None
        slave_ds = None
        if master_cleanup:
            master_cleanup()
        if slave_cleanup:
            slave_cleanup()


def _crossmul_isce3_gpu(
    master_slc_path: str,
    slave_slc_path: str,
    output_dir: Path,
    gpu_id: int,
    block_rows: int,
    *,
    flatten_raster: str | None = None,
    range_pixel_spacing: float | None = None,
    wavelength: float | None = None,
    flatten_starting_range_shift_m: float | None = None,
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
) -> tuple[np.ndarray, np.ndarray]:
    output_dir = Path(output_dir)
    p2_dir = stage_dir(output_dir, "p2")
    p2_dir.mkdir(parents=True, exist_ok=True)
    ifg_path = p2_dir / "cuda_interferogram.int"
    coh_path = p2_dir / "cuda_coherence.bin"
    helper = p2_dir / "cuda_crossmul_helper.py"
    helper.write_text(
        """
import sys
from osgeo import gdal
import isce3.core
import isce3.cuda.core
import isce3.cuda.signal
import isce3.io
from pathlib import Path
from insar_registration import _copy_raster_to_envi_complex64

master_slc_path, slave_slc_path, gpu_id, block_rows, ifg_path, coh_path, flatten_raster, range_pixel_spacing, wavelength, flatten_starting_range_shift_m = sys.argv[1:]
gpu_id = int(gpu_id)
block_rows = int(block_rows)
flatten_raster = None if flatten_raster == "__NONE__" else flatten_raster
range_pixel_spacing = None if range_pixel_spacing == "__NONE__" else float(range_pixel_spacing)
wavelength = None if wavelength == "__NONE__" else float(wavelength)
flatten_starting_range_shift_m = None if flatten_starting_range_shift_m == "__NONE__" else float(flatten_starting_range_shift_m)

input_dir = Path(ifg_path).parent / "cuda_inputs"
input_dir.mkdir(parents=True, exist_ok=True)
master_local = _copy_raster_to_envi_complex64(master_slc_path, input_dir / "master.slc")
slave_local = _copy_raster_to_envi_complex64(slave_slc_path, input_dir / "slave.slc")

try:
    device = isce3.cuda.core.Device(gpu_id)
    isce3.cuda.core.set_device(device)
    master_raster = isce3.io.Raster(str(master_local))
    slave_raster = isce3.io.Raster(str(slave_local))
    width = int(master_raster.width)
    length = int(master_raster.length)
    if int(slave_raster.width) != width or int(slave_raster.length) != length:
        raise RuntimeError(f"CUDA crossmul input dimensions differ: master={length}x{width}, slave={slave_raster.length}x{slave_raster.width}")

    ifg_raster = isce3.io.Raster(str(ifg_path), width, length, 1, gdal.GDT_CFloat32, "ENVI")
    coh_raster = isce3.io.Raster(str(coh_path), width, length, 1, gdal.GDT_Float32, "ENVI")
    crossmul = isce3.cuda.signal.Crossmul()
    crossmul.range_looks = 1
    crossmul.az_looks = 1
    crossmul.lines_per_block = block_rows
    try:
        crossmul.set_dopplers(isce3.core.LUT1d(), isce3.core.LUT1d())
    except Exception:
        pass
    flatten_isce_raster = None
    if flatten_raster is not None:
        if range_pixel_spacing is None or wavelength is None:
            raise ValueError("range_pixel_spacing and wavelength are required for CUDA crossmul flattening")
        crossmul.range_pixel_spacing = range_pixel_spacing
        crossmul.wavelength = wavelength
        crossmul.ref_sec_offset_starting_range_shift = float(flatten_starting_range_shift_m or 0.0)
        flatten_isce_raster = isce3.io.Raster(str(flatten_raster))
    print(f"[STRIP_INSAR] Launching CUDA Crossmul: master={master_local}, slave={slave_local}, flatten={flatten_raster}", flush=True)
    crossmul.crossmul(master_raster, slave_raster, ifg_raster, coh_raster, flatten_isce_raster)
    print("[STRIP_INSAR] CUDA Crossmul finished", flush=True)
except Exception as exc:
    print(f"[STRIP_INSAR] CUDA Crossmul helper failed: {exc}", file=sys.stderr, flush=True)
    raise
        """.strip()
        + "\n",
        encoding="utf-8",
    )
    py_compile.compile(str(helper), doraise=True)
    cmd = [
        sys.executable,
        "-X",
        "faulthandler",
        str(helper),
        master_slc_path,
        slave_slc_path,
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
    result = subprocess.run(
        cmd,
        cwd=str(Path(__file__).resolve().parent),
        env={
            **dict(__import__("os").environ),
            "PYTHONPATH": str(Path(__file__).resolve().parent)
            + (
                ":" + __import__("os").environ["PYTHONPATH"]
                if __import__("os").environ.get("PYTHONPATH")
                else ""
            ),
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
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
    interferogram = _read_band_array(ifg_ds.GetRasterBand(1), dtype=np.complex64).astype(np.complex64)
    coherence = _read_band_array(coh_ds.GetRasterBand(1), dtype=np.float32).astype(np.float32)
    ifg_ds = None
    coh_ds = None
    return interferogram, coherence


def _copy_raster(src: str, dst: str):
    from osgeo import gdal

    ds = gdal.Open(src)
    if ds is None:
        raise RuntimeError(f"cannot open {src}")
    proj = ds.GetProjection()
    gt = ds.GetGeoTransform()
    driver = gdal.GetDriverByName("GTiff")
    out = driver.Create(
        dst,
        ds.RasterXSize,
        ds.RasterYSize,
        ds.RasterCount,
        ds.GetRasterBand(1).DataType,
        options=["COMPRESS=LZW", "TILED=YES"],
    )
    out.SetProjection(proj)
    out.SetGeoTransform(gt)
    for i in range(ds.RasterCount):
        src_band = ds.GetRasterBand(i + 1)
        dst_band = out.GetRasterBand(i + 1)
        for row0 in range(0, ds.RasterYSize, 512):
            rows = min(512, ds.RasterYSize - row0)
            block = _read_band_array(src_band, 0, row0, ds.RasterXSize, rows)
            _write_band_array(dst_band, block, 0, row0)
    out.FlushCache()
    ds = None
    out = None


def _read_raster_as_array(path: str) -> np.ndarray:
    from osgeo import gdal

    ds = gdal.Open(path)
    if ds is None:
        raise RuntimeError(f"cannot open {path}")
    arr = _read_band_array(ds.GetRasterBand(1))
    ds = None
    return arr


# ---------------------------------------------------------------------------
# Main processing dispatcher
# ---------------------------------------------------------------------------


def process_strip_insar(
    master_manifest_path: str,
    slave_manifest_path: str,
    output_dir: str,
    dem_path: str | None = None,
    dem_cache_dir: str | None = None,
    dem_margin_deg: float = 0.05,
    unwrap_method: str = "icu",
    block_rows: int = 256,
    resolution_meters: float | None = None,
    gpu_mode: str = "auto",
    gpu_id: int = 0,
    step: str | None = None,
    start_step: str | None = None,
    end_step: str | None = None,
    resume: bool = False,
    bbox: tuple[float, float, float, float] | None = None,
    window: tuple[int, int, int, int] | None = None,
    dc_policy: str = "auto",
    prf_policy: str = "auto",
    skip_precheck: bool = False,
) -> dict:
    """Main InSAR processing dispatcher."""
    # Validate unwrap_method early, before any file I/O
    unwrapper = _create_unwrapper(unwrap_method)

    master_path = Path(master_manifest_path)
    slave_path = Path(slave_manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load manifests
    master_manifest = load_manifest(master_path)
    slave_manifest = load_manifest(slave_path)

    master_sensor = detect_sensor_from_manifest(master_path)
    slave_sensor = detect_sensor_from_manifest(slave_path)

    # Load metadata
    (master_mf, master_orbit_data, master_acq_data, master_rg_data, master_dop_data) = (
        _load_master_metadata(master_path)
    )
    (_, slave_orbit_data, slave_acq_data, slave_rg_data, slave_dop_data) = _load_master_metadata(
        slave_path
    )

    # Orbit interpolation
    orbit_interp = choose_orbit_interp(master_orbit_data, master_acq_data)

    # Wavelength
    wavelength = get_wavelength(master_acq_data)

    requested_stages = resolve_requested_stages(
        output_dir,
        step=step,
        start_step=start_step,
        end_step=end_step,
        resume=resume,
    )
    if not requested_stages:
        return {
            "sensor": master_sensor,
            "slave_sensor": slave_sensor,
            "master_manifest": str(master_path),
            "slave_manifest": str(slave_path),
            "completed_stages": [],
            "execution_requested": {"step": step, "start_step": start_step, "end_step": end_step, "resume": resume},
            "status": "already-complete",
        }

    if requested_stages == ["p6"] and stage_succeeded(output_dir, "p5"):
        p5_record = load_stage_record(output_dir, "p5") or {}
        crop_request = p5_record.get(
            "effective_crop",
            {
                "mode": "full",
                "master_window": {
                    "row0": 0,
                    "col0": 0,
                    "rows": int(master_rg_data.get("numberOfRows", 0)),
                    "cols": int(master_rg_data.get("numberOfColumns", 0)),
                },
                "bbox": None,
            },
        )
        if resolution_meters is None:
            range_res = master_rg_data.get("groundRangeResolution", 0)
            azimuth_res = master_rg_data.get("azimuthResolution", 0)
            resolution_meters = _ceil_to_half_or_int(max(range_res, azimuth_res) * 2.0)
        input_h5 = p5_record.get("output_files", {}).get("interferogram_h5") or build_output_paths(output_dir)["interferogram_h5"]
        outputs = _run_publish_stage_from_hdf(
            output_dir=output_dir,
            master_manifest_path=master_path,
            slave_manifest_path=slave_path,
            input_h5=input_h5,
            resolution_meters=resolution_meters,
            crop_request=crop_request,
            backend_used="cpu",
        )
        return {
            "sensor": master_sensor,
            "slave_sensor": slave_sensor,
            "master_manifest": str(master_path),
            "slave_manifest": str(slave_path),
            "completed_stages": ["p6"],
            "execution_requested": {"step": step, "start_step": start_step, "end_step": end_step, "resume": resume},
            "backend_requested": "cpu",
            "backend_reason": "cached p5 HDF reused for publish stage",
            "backend_used": "cpu",
            "pipeline_mode": "cpu",
            "stage_backends": {"publish": "cpu"},
            "resolution_meters": resolution_meters,
            "crop_request": crop_request,
            **outputs,
        }

    if requested_stages[0] == "p5" and stage_succeeded(output_dir, "p4"):
        p4_record = load_stage_record(output_dir, "p4") or {}
        crop_request = p4_record.get(
            "effective_crop",
            {
                "mode": "full",
                "master_window": {
                    "row0": 0,
                    "col0": 0,
                    "rows": int(master_rg_data.get("numberOfRows", 0)),
                    "cols": int(master_rg_data.get("numberOfColumns", 0)),
                },
                "bbox": None,
            },
        )
        if resolution_meters is None:
            range_res = master_rg_data.get("groundRangeResolution", 0)
            azimuth_res = master_rg_data.get("azimuthResolution", 0)
            resolution_meters = _ceil_to_half_or_int(max(range_res, azimuth_res) * 2.0)
        extra_hdf_attrs = {
            "execution_requested": {"step": step, "start_step": start_step, "end_step": end_step, "resume": resume},
            "crop_request": crop_request,
            "effective_master_window": crop_request.get("master_window"),
            "prf_dc_policy": {"dc_policy": dc_policy, "prf_policy": prf_policy},
        }
        dem_entry = master_manifest.get("dem")
        resolved_dem_for_p5 = None
        if dem_path:
            resolved_dem_for_p5 = dem_path
        elif isinstance(dem_entry, dict):
            resolved_dem_for_p5 = resolve_manifest_data_path(
                master_path,
                dem_entry.get("path"),
            )
        elif dem_entry:
            resolved_dem_for_p5 = resolve_manifest_data_path(master_path, dem_entry)
        if resolved_dem_for_p5 is None:
            resolved_dem_for_p5 = ""
        outputs = _run_hdf_stage_from_cache(
            output_dir=output_dir,
            master_manifest_path=master_path,
            slave_manifest_path=slave_path,
            wavelength=wavelength,
            crop_request=crop_request,
            backend_used="cpu",
            block_rows=block_rows,
            unwrap_method=unwrap_method,
            extra_hdf_attrs=extra_hdf_attrs,
            master_manifest=master_manifest,
            master_orbit_data=master_orbit_data,
            master_acq_data=master_acq_data,
            master_rg_data=master_rg_data,
            resolved_dem=resolved_dem_for_p5,
            orbit_interp=orbit_interp,
        )
        completed_stages = ["p5"]
        if requested_stages == ["p5", "p6"]:
            publish_outputs = _run_publish_stage_from_hdf(
                output_dir=output_dir,
                master_manifest_path=master_path,
                slave_manifest_path=slave_path,
                input_h5=outputs["interferogram_h5"],
                resolution_meters=resolution_meters,
                crop_request=crop_request,
                backend_used="cpu",
            )
            outputs = {**outputs, **publish_outputs}
            completed_stages.append("p6")
        return {
            "sensor": master_sensor,
            "slave_sensor": slave_sensor,
            "master_manifest": str(master_path),
            "slave_manifest": str(slave_path),
            "completed_stages": completed_stages,
            "execution_requested": {"step": step, "start_step": start_step, "end_step": end_step, "resume": resume},
            "backend_requested": "cpu",
            "backend_reason": "cached p2-p4 products reused for HDF/publish stages",
            "backend_used": "cpu",
            "pipeline_mode": "cpu",
            "stage_backends": {"hdf": "cpu", "publish": "cpu" if "p6" in completed_stages else None},
            "resolution_meters": resolution_meters,
            "crop_request": crop_request,
            **outputs,
        }

    corners = load_scene_corners_with_fallback(master_path, master_mf)
    crop_request = normalize_crop_request(
        bbox=bbox,
        window=window,
        radargrid_data=master_rg_data,
        scene_corners=corners,
    )

    # DEM
    resolved_dem = _resolve_dem_path(
        master_path,
        master_mf,
        corners,
        dem_path,
        dem_cache_dir,
        dem_margin_deg,
    )

    # Resolution
    if resolution_meters is None:
        range_res = master_rg_data.get("groundRangeResolution", 0)
        azimuth_res = master_rg_data.get("azimuthResolution", 0)
        resolution_meters = max(range_res, azimuth_res) * 2.0
        resolution_meters = _ceil_to_half_or_int(resolution_meters)

    precheck = _write_check_stage(
        output_dir=output_dir,
        master_manifest_path=master_path,
        slave_manifest_path=slave_path,
        master_manifest=master_manifest,
        slave_manifest=slave_manifest,
        master_acq_data=master_acq_data,
        slave_acq_data=slave_acq_data,
        master_rg_data=master_rg_data,
        slave_rg_data=slave_rg_data,
        master_dop_data=master_dop_data,
        slave_dop_data=slave_dop_data,
        crop_request=crop_request,
        dc_policy=dc_policy,
        prf_policy=prf_policy,
        skip_precheck=skip_precheck,
    )
    normalized_slave_manifest_path = _write_prep_stage(
        output_dir=output_dir,
        slave_manifest_path=slave_path,
        crop_request=crop_request,
        precheck=precheck,
        master_acq_data=master_acq_data,
        master_rg_data=master_rg_data,
        slave_acq_data=slave_acq_data,
        slave_rg_data=slave_rg_data,
        slave_dop_data=slave_dop_data,
    )
    if requested_stages[-1] in {"check", "prep", "crop"}:
        completed_stages = [stage_name for stage_name in requested_stages if stage_name in {"check", "prep", "crop"}]
        crop_outputs = None
        if "crop" in completed_stages:
            crop_outputs = _write_crop_stage(
                output_dir=output_dir,
                master_manifest_path=master_path,
                normalized_slave_manifest_path=Path(normalized_slave_manifest_path),
                crop_request=crop_request,
            )
        return {
            "sensor": master_sensor,
            "slave_sensor": slave_sensor,
            "master_manifest": str(master_path),
            "slave_manifest": str(slave_path),
            "normalized_slave_manifest": normalized_slave_manifest_path,
            "cropped_master_manifest": None if crop_outputs is None else crop_outputs["master_manifest"],
            "cropped_slave_manifest": None if crop_outputs is None else crop_outputs["slave_manifest"],
            "completed_stages": completed_stages,
            "execution_requested": {"step": step, "start_step": start_step, "end_step": end_step, "resume": resume},
            "crop_request": crop_request,
            "precheck": precheck,
            "backend_requested": "cpu",
            "backend_used": "cpu",
    }

    crop_outputs = _write_crop_stage(
        output_dir=output_dir,
        master_manifest_path=master_path,
        normalized_slave_manifest_path=Path(normalized_slave_manifest_path),
        crop_request=crop_request,
    )

    if requested_stages[0] == "p4" and stage_succeeded(output_dir, "p3"):
        cropped_master_path = Path(crop_outputs["master_manifest"])
        cropped_slave_path = Path(crop_outputs["slave_manifest"])
        runtime_rg, runtime_acq = _apply_crop_to_metadata(master_rg_data, master_acq_data, crop_request)
        runtime_crop_request = {
            "mode": "full",
            "master_window": {
                "row0": 0,
                "col0": 0,
                "rows": int(runtime_rg.get("numberOfRows", 0)),
                "cols": int(runtime_rg.get("numberOfColumns", 0)),
            },
            "bbox": None,
        }
        if resolution_meters is None:
            range_res = master_rg_data.get("groundRangeResolution", 0)
            azimuth_res = master_rg_data.get("azimuthResolution", 0)
            resolution_meters = _ceil_to_half_or_int(max(range_res, azimuth_res) * 2.0)
        extra_hdf_attrs = {
            "execution_requested": {"step": step, "start_step": start_step, "end_step": end_step, "resume": resume},
            "crop_request": crop_request,
            "effective_master_window": crop_request.get("master_window"),
            "prf_dc_policy": {"dc_policy": dc_policy, "prf_policy": prf_policy},
        }
        dem_entry = master_manifest.get("dem")
        resolved_dem_for_p5 = None
        if dem_path:
            resolved_dem_for_p5 = dem_path
        elif isinstance(dem_entry, dict):
            resolved_dem_for_p5 = resolve_manifest_data_path(
                master_path,
                dem_entry.get("path"),
            )
        elif dem_entry:
            resolved_dem_for_p5 = resolve_manifest_data_path(master_path, dem_entry)
        if resolved_dem_for_p5 is None:
            resolved_dem_for_p5 = ""

        outputs: dict[str, str] = {}
        completed_stages: list[str] = []
        outputs.update(
            _run_p4_stage_from_cache(
                output_dir=output_dir,
                master_manifest_path=cropped_master_path,
                slave_manifest_path=cropped_slave_path,
                wavelength=wavelength,
                crop_request=runtime_crop_request,
                backend_used="cpu",
            )
        )
        completed_stages.append("p4")
        if "p5" in requested_stages:
            outputs.update(
                _run_hdf_stage_from_cache(
                    output_dir=output_dir,
                    master_manifest_path=cropped_master_path,
                    slave_manifest_path=cropped_slave_path,
                    wavelength=wavelength,
                    crop_request=runtime_crop_request,
                    backend_used="cpu",
                    block_rows=block_rows,
                    unwrap_method=unwrap_method,
                    extra_hdf_attrs=extra_hdf_attrs,
                    master_manifest=master_manifest,
                    master_orbit_data=master_orbit_data,
                    master_acq_data=master_acq_data,
                    master_rg_data=master_rg_data,
                    resolved_dem=resolved_dem_for_p5,
                    orbit_interp=orbit_interp,
                )
            )
            completed_stages.append("p5")
        if "p6" in requested_stages:
            input_h5 = outputs.get("interferogram_h5") or build_output_paths(output_dir)["interferogram_h5"]
            outputs.update(
                _run_publish_stage_from_hdf(
                    output_dir=output_dir,
                    master_manifest_path=master_path,
                    slave_manifest_path=slave_path,
                    input_h5=input_h5,
                    resolution_meters=resolution_meters,
                    crop_request=runtime_crop_request,
                    backend_used="cpu",
                )
            )
            completed_stages.append("p6")
        return {
            "sensor": master_sensor,
            "slave_sensor": slave_sensor,
            "master_manifest": str(master_path),
            "slave_manifest": str(slave_path),
            "normalized_slave_manifest": normalized_slave_manifest_path,
            "cropped_master_manifest": str(cropped_master_path),
            "cropped_slave_manifest": str(cropped_slave_path),
            "completed_stages": completed_stages,
            "execution_requested": {"step": step, "start_step": start_step, "end_step": end_step, "resume": resume},
            "backend_requested": "cpu",
            "backend_reason": "cached p3 products reused for downstream stages",
            "backend_used": "cpu",
            "pipeline_mode": "cpu",
            "stage_backends": {
                "p4": "cpu" if "p4" in completed_stages else None,
                "p5": "cpu" if "p5" in completed_stages else None,
                "p6": "cpu" if "p6" in completed_stages else None,
            },
            "resolution_meters": resolution_meters,
            "crop_request": runtime_crop_request,
            "precheck": precheck,
            **outputs,
        }

    if requested_stages[-1] in {"p0", "p1", "p2", "p3", "p4"}:
        cropped_master_path = Path(crop_outputs["master_manifest"])
        cropped_slave_path = Path(crop_outputs["slave_manifest"])
        pipeline_slave_manifest = load_manifest(cropped_slave_path)
        pipeline_master_manifest = load_manifest(cropped_master_path)
        _, runtime_slave_orbit, runtime_slave_acq, runtime_slave_rg, runtime_slave_dop = (
            _load_master_metadata(str(cropped_slave_path))
        )
        runtime_rg, runtime_acq = _apply_crop_to_metadata(master_rg_data, master_acq_data, crop_request)
        runtime_crop_request = {
            "mode": "full",
            "master_window": {
                "row0": 0,
                "col0": 0,
                "rows": int(runtime_rg.get("numberOfRows", 0)),
                "cols": int(runtime_rg.get("numberOfColumns", 0)),
            },
            "bbox": None,
        }
        requested_backend, backend_reason = select_processing_backend(gpu_mode, gpu_id)
        backend_used = "gpu" if requested_backend == "gpu" else "cpu"
        outputs: dict[str, str] = {}
        completed_stages: list[str] = [
            stage_name for stage_name in requested_stages if stage_name in {"check", "prep", "crop"}
        ]
        if "p0" in requested_stages:
            outputs.update(
                _run_p0_stage(
                    output_dir=output_dir,
                    master_manifest_path=cropped_master_path,
                    slave_manifest_path=cropped_slave_path,
                    resolved_dem=resolved_dem,
                    orbit_interp=orbit_interp,
                    crop_request=runtime_crop_request,
                    backend_used=backend_used,
                    block_rows=block_rows,
                    gpu_id=gpu_id,
                )
            )
            completed_stages.append("p0")
        if "p1" in requested_stages:
            outputs.update(
                _run_p1_stage_from_cache(
                    output_dir=output_dir,
                    master_manifest_path=cropped_master_path,
                    slave_manifest_path=cropped_slave_path,
                    master_manifest=pipeline_master_manifest,
                    slave_manifest=pipeline_slave_manifest,
                    crop_request=runtime_crop_request,
                    backend_used=backend_used,
                    gpu_id=gpu_id,
                    master_orbit_data=master_orbit_data,
                    master_acq_data=runtime_acq,
                    master_rg_data=runtime_rg,
                    slave_orbit_data=runtime_slave_orbit,
                    slave_acq_data=runtime_slave_acq,
                    slave_rg_data=runtime_slave_rg,
                    slave_dop_data=runtime_slave_dop,
                )
            )
            completed_stages.append("p1")
        if "p2" in requested_stages:
            outputs.update(
                _run_p2_stage_from_cache(
                    output_dir=output_dir,
                    master_manifest_path=cropped_master_path,
                    slave_manifest_path=cropped_slave_path,
                    master_manifest=pipeline_master_manifest,
                    slave_manifest=pipeline_slave_manifest,
                    crop_request=runtime_crop_request,
                    backend_used=backend_used,
                    block_rows=block_rows,
                    use_gpu=backend_used == "gpu",
                    gpu_id=gpu_id,
                    master_acq_data=runtime_acq,
                    master_rg_data=runtime_rg,
                    slave_rg_data=runtime_slave_rg,
                )
            )
            completed_stages.append("p2")
        if "p3" in requested_stages:
            outputs.update(
                _run_p3_stage_from_cache(
                    output_dir=output_dir,
                    master_manifest_path=cropped_master_path,
                    slave_manifest_path=cropped_slave_path,
                    resolved_dem=resolved_dem,
                    orbit_interp=orbit_interp,
                    unwrapper=unwrapper,
                    crop_request=runtime_crop_request,
                    block_rows=block_rows,
                    backend_used="cpu",
                    master_orbit_data=master_orbit_data,
                    master_acq_data=runtime_acq,
                    master_rg_data=runtime_rg,
                )
            )
            completed_stages.append("p3")
        if "p4" in requested_stages:
            outputs.update(
                _run_p4_stage_from_cache(
                    output_dir=output_dir,
                    master_manifest_path=cropped_master_path,
                    slave_manifest_path=cropped_slave_path,
                    wavelength=wavelength,
                    crop_request=runtime_crop_request,
                    backend_used="cpu",
                )
            )
            completed_stages.append("p4")
        return {
            "sensor": master_sensor,
            "slave_sensor": slave_sensor,
            "master_manifest": str(master_path),
            "slave_manifest": str(slave_path),
            "normalized_slave_manifest": normalized_slave_manifest_path,
            "cropped_master_manifest": str(cropped_master_path),
            "cropped_slave_manifest": str(cropped_slave_path),
            "completed_stages": completed_stages,
            "execution_requested": {"step": step, "start_step": start_step, "end_step": end_step, "resume": resume},
            "backend_requested": requested_backend,
            "backend_reason": backend_reason,
            "backend_used": backend_used,
            "crop_request": runtime_crop_request,
            "precheck": precheck,
            **outputs,
        }

    if requested_stages != [
        "check",
        "prep",
        "crop",
        "p0",
        "p1",
        "p2",
        "p3",
        "p4",
        "p5",
        "p6",
    ]:
        raise NotImplementedError(
            "Stage-local execution for p0-p6 is not wired yet; current implementation supports check/prep or the full pipeline."
        )

    # Backend selection
    requested_backend, backend_reason = select_processing_backend(gpu_mode, gpu_id)
    backend_used = requested_backend
    fallback_reason = None
    stage_backends = CPU_STAGE_BACKENDS.copy()
    gpu_block_rows = None
    gpu_block_rows_reason = None
    cropped_master_path = Path(crop_outputs["master_manifest"])
    cropped_slave_path = Path(crop_outputs["slave_manifest"])
    pipeline_slave_manifest = load_manifest(cropped_slave_path)
    pipeline_master_manifest = load_manifest(cropped_master_path)
    _, runtime_slave_orbit, runtime_slave_acq, runtime_slave_rg, runtime_slave_dop = (
        _load_master_metadata(str(cropped_slave_path))
    )
    runtime_rg, runtime_acq = _apply_crop_to_metadata(master_rg_data, master_acq_data, crop_request)
    runtime_crop_request = {
        "mode": "full",
        "master_window": {"row0": 0, "col0": 0, "rows": int(runtime_rg.get("numberOfRows", 0)), "cols": int(runtime_rg.get("numberOfColumns", 0))},
        "bbox": None,
    }
    extra_hdf_attrs = {
        "execution_requested": {"step": step, "start_step": start_step, "end_step": end_step, "resume": resume},
        "crop_request": crop_request,
        "effective_master_window": crop_request.get("master_window"),
        "precheck_severity": precheck.get("overall_severity"),
        "preprocessing_decisions": {
            "requires_prep": precheck.get("requires_prep"),
            "recommended_geometry_mode": precheck.get("recommended_geometry_mode"),
        },
        "prf_dc_policy": {"dc_policy": dc_policy, "prf_policy": prf_policy},
    }
    try:
        if requested_backend == "gpu":
            gpu_block_rows, gpu_block_rows_reason = choose_gpu_topo_block_rows(
                width=int(master_rg_data.get("numberOfColumns", 0)),
                default_block_rows=block_rows,
                memory_info=query_gpu_memory_info(gpu_id),
            )
            print(f"[STRIP_INSAR] Using GPU mode")
            print(f"[STRIP_INSAR] {gpu_block_rows_reason}")
            outputs = _process_insar_gpu(
                master_manifest_path=str(cropped_master_path),
                slave_manifest_path=str(cropped_slave_path),
                master_manifest=pipeline_master_manifest,
                slave_manifest=pipeline_slave_manifest,
                orbit_interp=orbit_interp,
                wavelength=wavelength,
                resolved_dem=resolved_dem,
                unwrapper=unwrapper,
                block_rows=block_rows,
                gpu_block_rows=gpu_block_rows,
                resolution_meters=resolution_meters,
                gpu_id=gpu_id,
                output_dir=output_dir,
                extra_hdf_attrs=extra_hdf_attrs,
                crop_request=runtime_crop_request,
                master_orbit_data=master_orbit_data,
                master_acq_data=runtime_acq,
                master_rg_data=runtime_rg,
                slave_orbit_data=runtime_slave_orbit,
                slave_acq_data=runtime_slave_acq,
                slave_rg_data=runtime_slave_rg,
                slave_dop_data=runtime_slave_dop,
            )
            stage_backends = outputs.pop("stage_backends", HYBRID_GPU_STAGE_BACKENDS.copy())
            nested_fallback_reasons = outputs.pop("fallback_reasons", None)
        else:
            print(f"[STRIP_INSAR] Using CPU mode")
            outputs = _process_insar_cpu(
                master_manifest_path=str(cropped_master_path),
                slave_manifest_path=str(cropped_slave_path),
                master_manifest=pipeline_master_manifest,
                slave_manifest=pipeline_slave_manifest,
                orbit_interp=orbit_interp,
                wavelength=wavelength,
                resolved_dem=resolved_dem,
                unwrapper=unwrapper,
                block_rows=block_rows,
                resolution_meters=resolution_meters,
                output_dir=output_dir,
                extra_hdf_attrs=extra_hdf_attrs,
                crop_request=runtime_crop_request,
                master_orbit_data=master_orbit_data,
                master_acq_data=runtime_acq,
                master_rg_data=runtime_rg,
                slave_orbit_data=runtime_slave_orbit,
                slave_acq_data=runtime_slave_acq,
                slave_rg_data=runtime_slave_rg,
                slave_dop_data=runtime_slave_dop,
            )
    except Exception as exc:
        if requested_backend != "gpu":
            raise
        backend_used = "cpu"
        fallback_reason = str(exc)
        print(f"[STRIP_INSAR] GPU path failed: {exc}")
        print(f"[STRIP_INSAR] Falling back to CPU mode...")
        outputs = _process_insar_cpu(
            master_manifest_path=str(cropped_master_path),
            slave_manifest_path=str(cropped_slave_path),
            master_manifest=pipeline_master_manifest,
            slave_manifest=pipeline_slave_manifest,
            orbit_interp=orbit_interp,
            wavelength=wavelength,
            resolved_dem=resolved_dem,
            unwrapper=unwrapper,
            block_rows=block_rows,
            resolution_meters=resolution_meters,
            output_dir=output_dir,
            extra_hdf_attrs=extra_hdf_attrs,
            crop_request=runtime_crop_request,
            master_orbit_data=master_orbit_data,
            master_acq_data=runtime_acq,
            master_rg_data=runtime_rg,
            slave_orbit_data=slave_orbit_data,
            slave_acq_data=slave_acq_data,
            slave_rg_data=slave_rg_data,
            slave_dop_data=slave_dop_data,
        )
        stage_backends = CPU_STAGE_BACKENDS.copy()

    input_h5 = outputs.get("interferogram_h5") or build_output_paths(output_dir)[
        "interferogram_h5"
    ]
    _write_p5_stage_record(
        output_dir=output_dir,
        master_manifest_path=master_path,
        slave_manifest_path=slave_path,
        crop_request=runtime_crop_request,
        backend_used=backend_used,
        input_h5=input_h5,
    )
    need_publish = any(
        outputs.get(key) is None
        for key in (
            "interferogram_tif",
            "coherence_tif",
            "unwrapped_phase_tif",
            "los_displacement_tif",
            "interferogram_png",
            "filtered_interferogram_png",
        )
    )
    if need_publish:
        publish_outputs = _run_publish_stage_from_hdf(
        output_dir=output_dir,
        master_manifest_path=master_path,
        slave_manifest_path=slave_path,
        input_h5=input_h5,
        resolution_meters=resolution_meters,
        crop_request=runtime_crop_request,
        backend_used=backend_used,
    )
        outputs = {**outputs, **publish_outputs}

    result = {
        "sensor": master_sensor,
        "slave_sensor": slave_sensor,
        "master_manifest": str(master_path),
        "slave_manifest": str(slave_path),
        "normalized_slave_manifest": normalized_slave_manifest_path,
        "completed_stages": requested_stages,
        "execution_requested": {"step": step, "start_step": start_step, "end_step": end_step, "resume": resume},
        "backend_requested": requested_backend,
        "backend_reason": backend_reason,
        "backend_used": backend_used,
        "pipeline_mode": (
            "cpu-fallback"
            if fallback_reason is not None
            else _describe_pipeline_mode(stage_backends)
        ),
        "stage_backends": stage_backends,
        "unwrap_method": unwrap_method,
        "dem": resolved_dem,
        "orbit_interp": orbit_interp,
        "resolution_meters": resolution_meters,
        "gpu_block_rows": gpu_block_rows,
        "gpu_block_rows_reason": gpu_block_rows_reason,
        "wavelength": wavelength,
        "crop_request": crop_request,
        "precheck": precheck,
        "dc_policy": dc_policy,
        "prf_policy": prf_policy,
        **outputs,
    }
    fallback_reasons: dict[str, str] = {}
    if requested_backend == "gpu":
        nested = locals().get("nested_fallback_reasons")
        if nested:
            fallback_reasons.update(nested)
    if fallback_reason is not None:
        fallback_reasons["pipeline"] = fallback_reason
    if fallback_reasons:
        result["fallback_reasons"] = fallback_reasons
    return result


def _describe_pipeline_mode(stage_backends: dict) -> str:
    unique = set(stage_backends.values())
    if unique == {"cpu"}:
        return "cpu"
    if unique == {"gpu"}:
        return "gpu"
    return "hybrid"


# ---------------------------------------------------------------------------
# CPU processing pipeline
# ---------------------------------------------------------------------------


def _process_insar_cpu(
    master_manifest_path: str,
    slave_manifest_path: str,
    master_manifest: dict,
    slave_manifest: dict,
    orbit_interp: str,
    wavelength: float,
    resolved_dem: str,
    unwrapper: PhaseUnwrapper,
    block_rows: int,
    resolution_meters: float,
    output_dir: Path,
    extra_hdf_attrs: dict | None = None,
    crop_request: dict | None = None,
    master_orbit_data: dict | None = None,
    master_acq_data: dict | None = None,
    master_rg_data: dict | None = None,
    slave_orbit_data: dict | None = None,
    slave_acq_data: dict | None = None,
    slave_rg_data: dict | None = None,
    slave_dop_data: dict | None = None,
) -> dict:
    """Full CPU InSAR pipeline."""
    output_paths = build_output_paths(output_dir)

    # DEBUG: trace manifest and resolved paths
    print(f"[DEBUG] master_manifest_path={master_manifest_path}")
    print(f"[DEBUG] slave_manifest_path={slave_manifest_path}")
    print(
        f"[DEBUG] master slc path entry: {master_manifest.get('slc', {}).get('path')}"
    )
    print(f"[DEBUG] slave slc path entry: {slave_manifest.get('slc', {}).get('path')}")
    print(f"[DEBUG] master dem: {master_manifest.get('dem', {})}")
    print(f"[DEBUG] resolved_dem={resolved_dem}")

    master_slc = resolve_manifest_data_path(
        master_manifest_path, master_manifest["slc"]["path"]
    )
    slave_slc = resolve_manifest_data_path(
        slave_manifest_path, slave_manifest["slc"]["path"]
    )
    print(f"[DEBUG] resolved master_slc={master_slc}")
    print(f"[DEBUG] resolved slave_slc={slave_slc}")

    print("[P0/6] Geo2Rdr coarse coregistration (CPU)...")
    master_topo, slave_topo = _run_geo2rdr(
        master_manifest_path,
        slave_manifest_path,
        resolved_dem,
        orbit_interp,
        use_gpu=False,
        gpu_id=0,
        output_dir=output_dir,
        block_rows=block_rows,
    )
    _write_stage_outputs_record(
        output_dir=output_dir,
        stage="p0",
        master_manifest_path=master_manifest_path,
        slave_manifest_path=slave_manifest_path,
        crop_request=crop_request or {},
        backend_used="cpu",
        output_files={
            "master_topo": master_topo,
            "slave_topo": slave_topo,
            **(
                {"master_topo_vrt": str(output_dir / "geo2rdr_master" / "topo.vrt")}
                if (output_dir / "geo2rdr_master" / "topo.vrt").is_file()
                else {}
            ),
            **(
                {"slave_topo_vrt": str(output_dir / "geo2rdr_slave" / "topo.vrt")}
                if (output_dir / "geo2rdr_slave" / "topo.vrt").is_file()
                else {}
            ),
        },
    )

    print("[P1/6] Coarse SLC resampling (CPU) — PyCuAmpcor GPU-only, skipped...")
    p1_outputs = _run_p1_stage_from_cache(
        output_dir=output_dir,
        master_manifest_path=master_manifest_path,
        slave_manifest_path=slave_manifest_path,
        master_manifest=master_manifest,
        slave_manifest=slave_manifest,
        crop_request=crop_request or {},
        backend_used="cpu",
        gpu_id=0,
        master_orbit_data=master_orbit_data,
        master_acq_data=master_acq_data,
        master_rg_data=master_rg_data,
        slave_orbit_data=slave_orbit_data,
        slave_acq_data=slave_acq_data,
        slave_rg_data=slave_rg_data,
        slave_dop_data=slave_dop_data,
    )
    registered_slave_slc = _select_registered_slave_slc(p1_outputs, slave_slc)
    flatten_options = _build_crossmul_flatten_options(
        output_dir=output_dir,
        p1_outputs=p1_outputs,
        registered_slave_slc=registered_slave_slc,
        radargrid_data=master_rg_data or {},
        slave_radargrid_data=slave_rg_data or {},
        acquisition_data=master_acq_data or {},
    )

    print("[P2/6] Crossmul interferogram + coherence (CPU)...")
    crossmul_result = _run_crossmul(
        master_slc,
        registered_slave_slc,
        use_gpu=False,
        gpu_id=0,
        output_dir=output_dir,
        block_rows=block_rows,
        **flatten_options,
    )
    if len(crossmul_result) == 2:
        interferogram, coherence = crossmul_result
        crossmul_backend = "cpu"
        crossmul_fallback_reason = None
    else:
        interferogram, coherence, crossmul_backend, crossmul_fallback_reason = crossmul_result
    interferogram = _apply_crop_to_array(interferogram, crop_request)
    coherence = _apply_crop_to_array(coherence, crop_request)
    filtered_interferogram = goldstein_filter(interferogram)
    p2_output_files = {
        "interferogram": _save_stage_array(output_dir, "p2", "interferogram", interferogram),
        "filtered_interferogram": _save_stage_array(output_dir, "p2", "filtered_interferogram", filtered_interferogram),
        "coherence": _save_stage_array(output_dir, "p2", "coherence", coherence),
    }
    _add_flatten_outputs(p2_output_files, flatten_options)
    _write_stage_outputs_record(
        output_dir=output_dir,
        stage="p2",
        master_manifest_path=master_manifest_path,
        slave_manifest_path=slave_manifest_path,
        crop_request=crop_request or {},
        backend_used=crossmul_backend,
        output_files=p2_output_files,
        fallback_reason=crossmul_fallback_reason,
    )

    print("[P3/6] Phase unwrapping...")
    cropped_rg, cropped_acq = _apply_crop_to_metadata(
        master_rg_data, master_acq_data, crop_request
    )
    radar_grid = construct_radar_grid(cropped_rg, cropped_acq, master_orbit_data)
    orbit = construct_orbit(master_orbit_data, orbit_interp)
    with tempfile.TemporaryDirectory(
        prefix="insar_unwrap_", dir=str(output_dir)
    ) as tmpdir:
        unwrapped_phase = unwrapper.unwrap(
            filtered_interferogram,
            coherence,
            radar_grid,
            orbit,
            resolved_dem,
            tmpdir,
            block_rows,
        )
    _write_stage_outputs_record(
        output_dir=output_dir,
        stage="p3",
        master_manifest_path=master_manifest_path,
        slave_manifest_path=slave_manifest_path,
        crop_request=crop_request or {},
        backend_used="cpu",
        output_files={
            "unwrapped_phase": _save_stage_array(output_dir, "p3", "unwrapped_phase", unwrapped_phase)
        },
    )

    print("[P4/6] Geocoding products...")
    los_displacement = compute_los_displacement(unwrapped_phase, wavelength)
    _write_stage_outputs_record(
        output_dir=output_dir,
        stage="p4",
        master_manifest_path=master_manifest_path,
        slave_manifest_path=slave_manifest_path,
        crop_request=crop_request or {},
        backend_used="cpu",
        output_files={
            "los_displacement": _save_stage_array(output_dir, "p4", "los_displacement", los_displacement)
        },
    )

    # Write HDF5
    print("[P5/6] Writing HDF5...")
    write_insar_hdf(
        master_slc,
        slave_slc,
        interferogram,
        coherence,
        unwrapped_phase,
        los_displacement,
        wavelength,
        unwrapper.__class__.__name__.replace("Unwrapper", "").lower(),
        output_paths["interferogram_h5"],
        radar_grid,
        block_rows=block_rows,
        extra_attrs=extra_hdf_attrs,
        filtered_interferogram=filtered_interferogram,
        crop_request=crop_request,
    )

    # Append coordinates to HDF5
    if _is_full_crop(crop_request):
        append_topo_coordinates_hdf(
            master_manifest_path,
            resolved_dem,
            output_paths["interferogram_h5"],
            block_rows=block_rows,
            orbit_interp=orbit_interp,
            use_gpu=False,
        )
        append_utm_coordinates_hdf(
            output_paths["interferogram_h5"],
            master_manifest_path,
            block_rows=min(block_rows, 64),
        )
    else:
        _append_cropped_coordinates_hdf(
            output_paths["interferogram_h5"],
            master_topo,
            crop_request or {},
            block_rows=min(block_rows, 64),
        )

    return {"interferogram_h5": output_paths["interferogram_h5"]}


# ---------------------------------------------------------------------------
# GPU processing pipeline
# ---------------------------------------------------------------------------


def _process_insar_gpu(
    master_manifest_path: str,
    slave_manifest_path: str,
    master_manifest: dict,
    slave_manifest: dict,
    orbit_interp: str,
    wavelength: float,
    resolved_dem: str,
    unwrapper: PhaseUnwrapper,
    block_rows: int,
    gpu_block_rows: int,
    resolution_meters: float,
    gpu_id: int,
    output_dir: Path,
    extra_hdf_attrs: dict | None = None,
    crop_request: dict | None = None,
    master_orbit_data: dict | None = None,
    master_acq_data: dict | None = None,
    master_rg_data: dict | None = None,
    slave_orbit_data: dict | None = None,
    slave_acq_data: dict | None = None,
    slave_rg_data: dict | None = None,
    slave_dop_data: dict | None = None,
) -> dict:
    """Hybrid GPU InSAR pipeline."""
    output_paths = build_output_paths(output_dir)
    master_slc = resolve_manifest_data_path(
        master_manifest_path, master_manifest["slc"]["path"]
    )
    slave_slc = resolve_manifest_data_path(
        slave_manifest_path, slave_manifest["slc"]["path"]
    )
    stage_backends = HYBRID_GPU_STAGE_BACKENDS.copy()
    fallback_reasons: dict[str, str] = {}

    current_gpu_block_rows = gpu_block_rows

    while True:
        try:
            print(
                f"[P0/6] Geo2Rdr coarse coregistration (GPU, {current_gpu_block_rows} rows/block)..."
            )
            master_topo, slave_topo = _run_geo2rdr(
                master_manifest_path,
                slave_manifest_path,
                resolved_dem,
                orbit_interp,
                use_gpu=True,
                gpu_id=gpu_id,
                output_dir=output_dir,
                block_rows=current_gpu_block_rows,
            )
            break
        except Exception as exc:
            if not _is_gpu_memory_error(exc) or current_gpu_block_rows <= 64:
                raise
            next_br = _halve_block_rows(current_gpu_block_rows)
            if next_br >= current_gpu_block_rows:
                raise
            print(
                f"[STRIP_INSAR] GPU Geo2Rdr OOM at {current_gpu_block_rows}; "
                f"retrying with {next_br}"
            )
            current_gpu_block_rows = next_br
    _write_stage_outputs_record(
        output_dir=output_dir,
        stage="p0",
        master_manifest_path=master_manifest_path,
        slave_manifest_path=slave_manifest_path,
        crop_request=crop_request or {},
        backend_used="gpu",
        output_files={
            "master_topo": master_topo,
            "slave_topo": slave_topo,
            **(
                {"master_topo_vrt": str(output_dir / "geo2rdr_master" / "topo.vrt")}
                if (output_dir / "geo2rdr_master" / "topo.vrt").is_file()
                else {}
            ),
            **(
                {"slave_topo_vrt": str(output_dir / "geo2rdr_slave" / "topo.vrt")}
                if (output_dir / "geo2rdr_slave" / "topo.vrt").is_file()
                else {}
            ),
        },
    )

    print("[P1/6] Coarse SLC resampling (GPU) + PyCuAmpcor dense matching...")
    p1_outputs = _run_p1_stage_from_cache(
        output_dir=output_dir,
        master_manifest_path=master_manifest_path,
        slave_manifest_path=slave_manifest_path,
        master_manifest=master_manifest,
        slave_manifest=slave_manifest,
        crop_request=crop_request or {},
        backend_used="gpu",
        gpu_id=gpu_id,
        master_orbit_data=master_orbit_data,
        master_acq_data=master_acq_data,
        master_rg_data=master_rg_data,
        slave_orbit_data=slave_orbit_data,
        slave_acq_data=slave_acq_data,
        slave_rg_data=slave_rg_data,
        slave_dop_data=slave_dop_data,
    )
    registered_slave_slc = _select_registered_slave_slc(p1_outputs, slave_slc)
    flatten_options = _build_crossmul_flatten_options(
        output_dir=output_dir,
        p1_outputs=p1_outputs,
        registered_slave_slc=registered_slave_slc,
        radargrid_data=master_rg_data or {},
        slave_radargrid_data=slave_rg_data or {},
        acquisition_data=master_acq_data or {},
    )

    print(
        f"[P2/6] Crossmul interferogram + coherence (GPU, {current_gpu_block_rows} rows/block)..."
    )
    while True:
        try:
            crossmul_result = _run_crossmul(
                master_slc,
                registered_slave_slc,
                use_gpu=True,
                gpu_id=gpu_id,
                output_dir=output_dir,
                block_rows=current_gpu_block_rows,
                **flatten_options,
            )
            if len(crossmul_result) == 2:
                interferogram, coherence = crossmul_result
                crossmul_backend = "gpu"
                crossmul_fallback_reason = None
            else:
                interferogram, coherence, crossmul_backend, crossmul_fallback_reason = crossmul_result
            stage_backends["crossmul"] = crossmul_backend
            if crossmul_fallback_reason is not None:
                fallback_reasons["crossmul"] = crossmul_fallback_reason
            interferogram = _apply_crop_to_array(interferogram, crop_request)
            coherence = _apply_crop_to_array(coherence, crop_request)
            filtered_interferogram = goldstein_filter(interferogram)
            break
        except Exception as exc:
            if not _is_gpu_memory_error(exc) or current_gpu_block_rows <= 64:
                raise
            next_br = _halve_block_rows(current_gpu_block_rows)
            if next_br >= current_gpu_block_rows:
                raise
            print(
                f"[STRIP_INSAR] GPU Crossmul OOM at {current_gpu_block_rows}; "
                f"retrying with {next_br}"
            )
            current_gpu_block_rows = next_br
    p2_output_files = {
        "interferogram": _save_stage_array(output_dir, "p2", "interferogram", interferogram),
        "filtered_interferogram": _save_stage_array(output_dir, "p2", "filtered_interferogram", filtered_interferogram),
        "coherence": _save_stage_array(output_dir, "p2", "coherence", coherence),
    }
    _add_flatten_outputs(p2_output_files, flatten_options)
    _write_stage_outputs_record(
        output_dir=output_dir,
        stage="p2",
        master_manifest_path=master_manifest_path,
        slave_manifest_path=slave_manifest_path,
        crop_request=crop_request or {},
        backend_used=crossmul_backend,
        output_files=p2_output_files,
        fallback_reason=crossmul_fallback_reason,
    )

    print("[P3/6] Phase unwrapping (CPU)...")
    cropped_rg, cropped_acq = _apply_crop_to_metadata(
        master_rg_data, master_acq_data, crop_request
    )
    radar_grid = construct_radar_grid(cropped_rg, cropped_acq, master_orbit_data)
    orbit = construct_orbit(master_orbit_data, orbit_interp)
    with tempfile.TemporaryDirectory(
        prefix="insar_unwrap_", dir=str(output_dir)
    ) as tmpdir:
        unwrapped_phase = unwrapper.unwrap(
            filtered_interferogram,
            coherence,
            radar_grid,
            orbit,
            resolved_dem,
            tmpdir,
            block_rows,
        )
    _write_stage_outputs_record(
        output_dir=output_dir,
        stage="p3",
        master_manifest_path=master_manifest_path,
        slave_manifest_path=slave_manifest_path,
        crop_request=crop_request or {},
        backend_used="cpu",
        output_files={
            "unwrapped_phase": _save_stage_array(output_dir, "p3", "unwrapped_phase", unwrapped_phase)
        },
    )

    print("[P4/6] Geocoding products (GPU)...")
    los_displacement = compute_los_displacement(unwrapped_phase, wavelength)
    _write_stage_outputs_record(
        output_dir=output_dir,
        stage="p4",
        master_manifest_path=master_manifest_path,
        slave_manifest_path=slave_manifest_path,
        crop_request=crop_request or {},
        backend_used="gpu",
        output_files={
            "los_displacement": _save_stage_array(output_dir, "p4", "los_displacement", los_displacement)
        },
    )

    print("[P5/6] Writing HDF5...")
    write_insar_hdf(
        master_slc,
        slave_slc,
        interferogram,
        coherence,
        unwrapped_phase,
        los_displacement,
        wavelength,
        unwrapper.__class__.__name__.replace("Unwrapper", "").lower(),
        output_paths["interferogram_h5"],
        radar_grid,
        block_rows=block_rows,
        extra_attrs=extra_hdf_attrs,
        filtered_interferogram=filtered_interferogram,
        crop_request=crop_request,
    )

    if _is_full_crop(crop_request):
        append_topo_coordinates_hdf(
            master_manifest_path,
            resolved_dem,
            output_paths["interferogram_h5"],
            block_rows=block_rows,
            orbit_interp=orbit_interp,
            use_gpu=True,
            gpu_id=gpu_id,
        )
        append_utm_coordinates_hdf(
            output_paths["interferogram_h5"],
            master_manifest_path,
            block_rows=min(block_rows, 64),
        )
    else:
        _append_cropped_coordinates_hdf(
            output_paths["interferogram_h5"],
            master_topo,
            crop_request or {},
            block_rows=min(block_rows, 64),
        )

    result = {"interferogram_h5": output_paths["interferogram_h5"], "stage_backends": stage_backends}
    if fallback_reasons:
        result["fallback_reasons"] = fallback_reasons
    return result


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Unified Tianyi/Lutan strip InSAR processor with GPU-first fallback"
    )
    parser.add_argument("master_manifest", help="Path to master SLC manifest.json")
    parser.add_argument("slave_manifest", help="Path to slave SLC manifest.json")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--dem", help="Override DEM path")
    parser.add_argument("--dem-cache-dir", help="DEM cache/search directory")
    parser.add_argument("--dem-margin-deg", type=float, default=0.05)
    parser.add_argument(
        "--unwrap-method",
        choices=["icu", "snaphu"],
        default="icu",
        help="Phase unwrapping method",
    )
    parser.add_argument("--block-rows", type=int, default=256)
    parser.add_argument(
        "--resolution",
        type=float,
        default=None,
        help="Output ground resolution in meters",
    )
    parser.add_argument(
        "--gpu-mode",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="GPU preference",
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--step", choices=["check", "prep", "crop", "p0", "p1", "p2", "p3", "p4", "p5", "p6"])
    parser.add_argument("--start-step", choices=["check", "prep", "crop", "p0", "p1", "p2", "p3", "p4", "p5", "p6"])
    parser.add_argument("--end-step", choices=["check", "prep", "crop", "p0", "p1", "p2", "p3", "p4", "p5", "p6"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    parser.add_argument("--window", nargs=4, type=int, metavar=("ROW0", "COL0", "ROWS", "COLS"))
    parser.add_argument("--dc-policy", choices=["auto", "zero", "strict"], default="auto")
    parser.add_argument("--prf-policy", choices=["auto", "strict"], default="auto")
    parser.add_argument("--skip-precheck", action="store_true")
    args = parser.parse_args()

    result = process_strip_insar(
        args.master_manifest,
        args.slave_manifest,
        args.output_dir,
        dem_path=args.dem,
        dem_cache_dir=args.dem_cache_dir,
        dem_margin_deg=args.dem_margin_deg,
        unwrap_method=args.unwrap_method,
        block_rows=args.block_rows,
        resolution_meters=args.resolution,
        gpu_mode=args.gpu_mode,
        gpu_id=args.gpu_id,
        step=args.step,
        start_step=args.start_step,
        end_step=args.end_step,
        resume=args.resume,
        bbox=tuple(args.bbox) if args.bbox is not None else None,
        window=tuple(args.window) if args.window is not None else None,
        dc_policy=args.dc_policy,
        prf_policy=args.prf_policy,
        skip_precheck=args.skip_precheck,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[STRIP_INSAR] Done. Outputs:")
    print(f"  HDF5: {result.get('interferogram_h5', 'N/A')}")
    print(f"  Interferogram TIFF: {result.get('interferogram_tif', 'N/A')}")
    print(f"  Coherence TIFF: {result.get('coherence_tif', 'N/A')}")
    print(f"  Unwrapped Phase TIFF: {result.get('unwrapped_phase_tif', 'N/A')}")
    print(f"  LOS Displacement TIFF: {result.get('los_displacement_tif', 'N/A')}")
    print(f"  PNG: {result.get('interferogram_png', 'N/A')}")


if __name__ == "__main__":
    main()
