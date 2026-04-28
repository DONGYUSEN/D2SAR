import json
import subprocess
import tempfile
from pathlib import Path

from common_processing import (
    append_topo_coordinates_hdf,
    append_utm_coordinates_hdf,
    choose_orbit_interp,
    compute_utm_output_shape,
    compute_rtc_factor,
    load_scene_corners_with_fallback,
    resolve_dem_for_scene,
    resolve_manifest_data_path,
    resolve_manifest_metadata_path,
    write_geocoded_geotiff,
    write_geocoded_png,
    write_rtc_hdf,
)


SUPPORTED_SENSORS = {"tianyi", "lutan"}
CPU_STAGE_BACKENDS = {
    "rtc_factor": "cpu",
    "amplitude_hdf": "cpu",
    "topo_lonlatheight": "cpu",
    "utm_transform": "cpu",
    "utm_rasterize": "cpu",
    "preview_png": "cpu",
}
HYBRID_GPU_STAGE_BACKENDS = {
    **CPU_STAGE_BACKENDS,
    "topo_lonlatheight": "gpu",
}


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
        "amplitude_h5": str(output_dir / "amplitude_fullres.h5"),
        "amplitude_utm_tif": str(output_dir / "amplitude_utm_geocoded.tif"),
        "amplitude_utm_png": str(output_dir / "amplitude_utm_geocoded.png"),
        "dem_validation": str(output_dir / "dem_validation.json"),
    }


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

    if dem_cache_dir is None:
        dem_cache_dir = str(manifest_path.parent / "dem")
    resolved_dem, _ = resolve_dem_for_scene(
        corners,
        dem_path=dem_path,
        output_dir=dem_cache_dir,
        margin_deg=dem_margin_deg,
    )
    return resolved_dem


def _load_processing_metadata(manifest_path: Path, manifest: dict) -> tuple[dict, dict]:
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
    return orbit_data, acquisition_data


def _load_radargrid_metadata(manifest_path: Path, manifest: dict) -> dict:
    with open(
        resolve_manifest_metadata_path(manifest_path, manifest, "radargrid"),
        encoding="utf-8",
    ) as f:
        return json.load(f)


def _ceil_to_half_or_int(x: float) -> float:
    import math

    return math.ceil(x * 2) / 2


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
        total_mib_str, free_mib_str = [part.strip() for part in line.split(",", 1)]
        mib = 1024 * 1024
        return {
            "total_bytes": int(float(total_mib_str) * mib),
            "free_bytes": int(float(free_mib_str) * mib),
        }
    except Exception:
        return None


def choose_gpu_topo_block_rows(
    width: int,
    default_block_rows: int,
    memory_info: dict[str, int] | None,
    min_block_rows: int = 64,
    max_block_rows: int = 1024,
) -> tuple[int, str]:
    if memory_info is None:
        return (
            default_block_rows,
            "Default topo block_rows used; GPU memory info unavailable",
        )
    total_bytes = int(memory_info.get("total_bytes", 0))
    free_bytes = int(memory_info.get("free_bytes", 0))
    if width <= 0 or total_bytes <= 0 or free_bytes <= 0:
        return (
            default_block_rows,
            "Default topo block_rows used; GPU memory info invalid",
        )

    reserve_bytes = max(1024**3, int(0.20 * total_bytes))
    budget_bytes = min(
        int(0.40 * total_bytes),
        int(0.65 * free_bytes),
        free_bytes - reserve_bytes,
    )
    if budget_bytes <= 0:
        return (
            default_block_rows,
            "Default topo block_rows used; GPU memory budget unavailable",
        )

    bytes_per_row_lower_bound = width * 41
    safety_factor = 4
    estimated_rows = budget_bytes // (bytes_per_row_lower_bound * safety_factor)
    estimated_rows = max(32, (estimated_rows // 32) * 32)
    topo_block_rows = max(min_block_rows, min(max_block_rows, int(estimated_rows)))
    return (
        topo_block_rows,
        f"Adaptive GPU topo block_rows={topo_block_rows} from VRAM budget {budget_bytes} bytes",
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


def _halve_topo_block_rows(block_rows: int, min_block_rows: int = 64) -> int:
    halved = max(min_block_rows, block_rows // 2)
    halved = max(min_block_rows, (halved // 32) * 32)
    return halved


def _process_strip_cpu(
    manifest_path: Path,
    manifest: dict,
    output_dir: Path,
    resolved_dem: str,
    orbit_interp: str,
    block_rows: int,
    resolution_meters: float,
) -> dict:
    output_paths = build_output_paths(output_dir)
    slc_path = resolve_manifest_data_path(manifest_path, manifest["slc"]["path"])

    print("[1/6] Computing RTC factor (DEM-assisted terrain correction)...")
    with tempfile.TemporaryDirectory(
        prefix="strip_rtc_", dir=str(output_dir)
    ) as tmpdir:
        rtc_factor_path = Path(tmpdir) / "rtc_factor.tif"
        compute_rtc_factor(
            str(manifest_path),
            resolved_dem,
            str(rtc_factor_path),
            orbit_interp=orbit_interp,
        )
        print("[2/6] Writing amplitude HDF5 (full-resolution SLC + RTC)...")
        write_rtc_hdf(
            slc_path,
            str(rtc_factor_path),
            output_paths["amplitude_h5"],
            block_rows=block_rows,
        )

    print("[3/6] Computing topo coordinates (lat/lon/height)...")
    append_topo_coordinates_hdf(
        str(manifest_path),
        resolved_dem,
        output_paths["amplitude_h5"],
        block_rows=block_rows,
        orbit_interp=orbit_interp,
    )
    print("[4/6] Computing UTM coordinates...")
    append_utm_coordinates_hdf(
        output_paths["amplitude_h5"],
        str(manifest_path),
        block_rows=min(block_rows, 64),
    )
    target_width, target_height = compute_utm_output_shape(
        output_paths["amplitude_h5"],
        resolution_meters,
        block_rows=min(block_rows, 64),
    )
    print(
        f"[5/6] Writing GeoTIFF ({target_width}x{target_height} px, {resolution_meters:.1f}m/pix)..."
    )
    write_geocoded_geotiff(
        output_paths["amplitude_h5"],
        output_paths["amplitude_utm_tif"],
        target_width=target_width,
        target_height=target_height,
        block_rows=min(block_rows, 64),
    )
    print("[6/6] Writing PNG preview...")
    write_geocoded_png(
        output_paths["amplitude_h5"],
        output_paths["amplitude_utm_png"],
        target_width=target_width,
        target_height=target_height,
        block_rows=min(block_rows, 64),
    )
    return output_paths


def _process_strip_gpu(
    manifest_path: Path,
    manifest: dict,
    output_dir: Path,
    resolved_dem: str,
    orbit_interp: str,
    block_rows: int,
    topo_block_rows: int,
    resolution_meters: float,
    gpu_id: int = 0,
) -> dict:
    output_paths = build_output_paths(output_dir)
    slc_path = resolve_manifest_data_path(manifest_path, manifest["slc"]["path"])

    print("[1/6] Computing RTC factor (DEM-assisted terrain correction)...")
    with tempfile.TemporaryDirectory(
        prefix="strip_rtc_", dir=str(output_dir)
    ) as tmpdir:
        rtc_factor_path = Path(tmpdir) / "rtc_factor.tif"
        compute_rtc_factor(
            str(manifest_path),
            resolved_dem,
            str(rtc_factor_path),
            orbit_interp=orbit_interp,
        )
        print("[2/6] Writing amplitude HDF5 (full-resolution SLC + RTC)...")
        write_rtc_hdf(
            slc_path,
            str(rtc_factor_path),
            output_paths["amplitude_h5"],
            block_rows=block_rows,
        )

    print(
        f"[3/6] Computing topo coordinates on GPU (lat/lon/height, {topo_block_rows} rows/block)..."
    )
    append_topo_coordinates_hdf(
        str(manifest_path),
        resolved_dem,
        output_paths["amplitude_h5"],
        block_rows=topo_block_rows,
        orbit_interp=orbit_interp,
        use_gpu=True,
        gpu_id=gpu_id,
    )
    print("[4/6] Computing UTM coordinates...")
    append_utm_coordinates_hdf(
        output_paths["amplitude_h5"],
        str(manifest_path),
        block_rows=min(block_rows, 64),
    )
    target_width, target_height = compute_utm_output_shape(
        output_paths["amplitude_h5"],
        resolution_meters,
        block_rows=min(block_rows, 64),
    )
    print(
        f"[5/6] Writing GeoTIFF ({target_width}x{target_height} px, {resolution_meters:.1f}m/pix)..."
    )
    write_geocoded_geotiff(
        output_paths["amplitude_h5"],
        output_paths["amplitude_utm_tif"],
        target_width=target_width,
        target_height=target_height,
        block_rows=min(block_rows, 64),
    )
    print("[6/6] Writing PNG preview...")
    write_geocoded_png(
        output_paths["amplitude_h5"],
        output_paths["amplitude_utm_png"],
        target_width=target_width,
        target_height=target_height,
        block_rows=min(block_rows, 64),
    )
    return output_paths


def describe_pipeline_mode(stage_backends: dict[str, str]) -> str:
    unique_backends = set(stage_backends.values())
    if unique_backends == {"cpu"}:
        return "cpu"
    if unique_backends == {"gpu"}:
        return "gpu"
    return "hybrid"


def process_strip_rtc(
    manifest_path: str,
    output_dir: str,
    dem_path: str | None = None,
    dem_cache_dir: str | None = None,
    dem_margin_deg: float = 0.05,
    block_rows: int = 256,
    resolution_meters: float | None = None,
    gpu_mode: str = "auto",
    gpu_id: int = 0,
) -> dict:
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(manifest_path)
    sensor = detect_sensor_from_manifest(manifest_path)
    corners = load_scene_corners_with_fallback(manifest_path, manifest)
    orbit_data, acquisition_data = _load_processing_metadata(manifest_path, manifest)
    radargrid_data = _load_radargrid_metadata(manifest_path, manifest)
    orbit_interp = choose_orbit_interp(orbit_data, acquisition_data)
    resolved_dem = _resolve_dem_path(
        manifest_path,
        manifest,
        corners,
        dem_path,
        dem_cache_dir,
        dem_margin_deg,
    )

    if resolution_meters is None:
        range_res = radargrid_data.get("groundRangeResolution", 0)
        azimuth_res = radargrid_data.get("azimuthResolution", 0)
        resolution_meters = max(range_res, azimuth_res) * 2.0
        resolution_meters = _ceil_to_half_or_int(resolution_meters)

    requested_backend, backend_reason = select_processing_backend(gpu_mode, gpu_id)
    backend_used = requested_backend
    fallback_reason = None
    stage_backends = CPU_STAGE_BACKENDS.copy()
    topo_block_rows = None
    topo_block_rows_reason = None

    try:
        if requested_backend == "gpu":
            topo_block_rows, topo_block_rows_reason = choose_gpu_topo_block_rows(
                width=int(radargrid_data.get("numberOfColumns", 0)),
                default_block_rows=block_rows,
                memory_info=query_gpu_memory_info(gpu_id),
            )
            print(f"[STRIP_RTC] Using GPU mode (topo stage on GPU, others on CPU)")
            print(f"[STRIP_RTC] {topo_block_rows_reason}")
            current_topo_block_rows = topo_block_rows
            while True:
                try:
                    outputs = _process_strip_gpu(
                        manifest_path=manifest_path,
                        manifest=manifest,
                        output_dir=output_dir,
                        resolved_dem=resolved_dem,
                        orbit_interp=orbit_interp,
                        block_rows=block_rows,
                        topo_block_rows=current_topo_block_rows,
                        resolution_meters=resolution_meters,
                        gpu_id=gpu_id,
                    )
                    topo_block_rows = current_topo_block_rows
                    break
                except Exception as exc:
                    if not _is_gpu_memory_error(exc) or current_topo_block_rows <= 64:
                        raise
                    next_topo_block_rows = _halve_topo_block_rows(
                        current_topo_block_rows
                    )
                    if next_topo_block_rows >= current_topo_block_rows:
                        raise
                    print(
                        f"[STRIP_RTC] GPU topo OOM at {current_topo_block_rows} rows/block; retrying with {next_topo_block_rows}"
                    )
                    current_topo_block_rows = next_topo_block_rows
            stage_backends = HYBRID_GPU_STAGE_BACKENDS.copy()
        else:
            print(f"[STRIP_RTC] Using CPU mode")
            outputs = _process_strip_cpu(
                manifest_path=manifest_path,
                manifest=manifest,
                output_dir=output_dir,
                resolved_dem=resolved_dem,
                orbit_interp=orbit_interp,
                block_rows=block_rows,
                resolution_meters=resolution_meters,
            )
    except Exception as exc:
        if requested_backend != "gpu":
            raise
        backend_used = "cpu"
        fallback_reason = str(exc)
        print(f"[STRIP_RTC] GPU path failed: {exc}")
        print(f"[STRIP_RTC] Falling back to CPU mode...")
        outputs = _process_strip_cpu(
            manifest_path=manifest_path,
            manifest=manifest,
            output_dir=output_dir,
            resolved_dem=resolved_dem,
            orbit_interp=orbit_interp,
            block_rows=block_rows,
            resolution_meters=resolution_meters,
        )
        stage_backends = CPU_STAGE_BACKENDS.copy()

    result = {
        "sensor": sensor,
        "backend_requested": requested_backend,
        "backend_reason": backend_reason,
        "backend_used": backend_used,
        "pipeline_mode": (
            "cpu-fallback"
            if fallback_reason is not None
            else describe_pipeline_mode(stage_backends)
        ),
        "stage_backends": stage_backends,
        "dem": resolved_dem,
        "orbit_interp": orbit_interp,
        "resolution_meters": resolution_meters,
        "topo_block_rows": topo_block_rows,
        "topo_block_rows_reason": topo_block_rows_reason,
        **outputs,
    }
    if fallback_reason is not None:
        result["fallback_reasons"] = {"pipeline": fallback_reason}
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Unified Tianyi/Lutan strip RTC processor with GPU-first fallback"
    )
    parser.add_argument("manifest", help="Path to manifest.json")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--dem", help="Override DEM path")
    parser.add_argument(
        "--dem-cache-dir", help="DEM cache/search directory when manifest has no DEM"
    )
    parser.add_argument("--dem-margin-deg", type=float, default=0.05)
    parser.add_argument("--block-rows", type=int, default=256)
    parser.add_argument(
        "--resolution",
        type=float,
        default=None,
        help="Output ground resolution in meters (default: 2x max of range/azimuth resolution)",
    )
    parser.add_argument(
        "--gpu-mode",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="Prefer GPU when available, or force CPU/GPU",
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()

    result = process_strip_rtc(
        args.manifest,
        args.output_dir,
        dem_path=args.dem,
        dem_cache_dir=args.dem_cache_dir,
        dem_margin_deg=args.dem_margin_deg,
        block_rows=args.block_rows,
        resolution_meters=args.resolution,
        gpu_mode=args.gpu_mode,
        gpu_id=args.gpu_id,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[STRIP_RTC] Done. Outputs:")
    print(f"  HDF5: {result.get('amplitude_h5', 'N/A')}")
    print(f"  GeoTIFF: {result.get('amplitude_utm_tif', 'N/A')}")
    print(f"  PNG: {result.get('amplitude_utm_png', 'N/A')}")


if __name__ == "__main__":
    main()
