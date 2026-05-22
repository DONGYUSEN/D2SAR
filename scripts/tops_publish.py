"""tops_publish — Unwrap, geocode, and package merged InSAR products.

Final-stage module consumed by tops_insar2.py after merge_bursts().
This module does not import any strip-backend or tops_insar modules.
"""

from __future__ import annotations

__all__ = [
    "geocode_ifg",
    "unwrap_ifg",
    "write_hdf5_product",
    "write_product",
    "write_tiff_array",
    "read_tiff_array",
]

import colorsys
import json
import logging
import shutil
import subprocess
import tempfile
from datetime import timedelta
from pathlib import Path

import numpy as np

from scripts.tops_model import BurstRadarGrid, BurstWindow

log = logging.getLogger(__name__)


def write_tiff_array(path: Path, data: np.ndarray) -> Path:
    try:
        from osgeo import gdal
    except ImportError as exc:
        raise NotImplementedError("GDAL required for TIFF I/O") from exc
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    driver = gdal.GetDriverByName("GTiff")
    if np.iscomplexobj(data):
        ds = driver.Create(str(path), data.shape[1], data.shape[0], 2, gdal.GDT_Float32, options=["TILED=YES", "COMPRESS=DEFLATE"])
        ds.GetRasterBand(1).WriteArray(np.real(data).astype(np.float32))
        ds.GetRasterBand(2).WriteArray(np.imag(data).astype(np.float32))
    else:
        ds = driver.Create(str(path), data.shape[1], data.shape[0], 1, gdal.GDT_Float32, options=["TILED=YES", "COMPRESS=DEFLATE"])
        ds.GetRasterBand(1).WriteArray(np.asarray(data, dtype=np.float32))
    ds = None
    return path


def read_tiff_array(path: Path) -> np.ndarray:
    try:
        from osgeo import gdal
    except ImportError as exc:
        raise NotImplementedError("GDAL required for TIFF I/O") from exc
    ds = gdal.Open(str(path))
    if ds is None:
        raise FileNotFoundError(path)
    if ds.RasterCount == 2:
        return ds.GetRasterBand(1).ReadAsArray().astype(np.float32) + 1j * ds.GetRasterBand(2).ReadAsArray().astype(np.float32)
    return ds.ReadAsArray().astype(np.float32)


def _phase_to_color_rgba(merged_ifg: np.ndarray, merged_coh: np.ndarray | None = None) -> np.ndarray:
    """Convert wrapped interferogram to RGBA PNG data.

    Invalid pixels (NaN phase or non-positive/invalid coherence) become fully
    transparent. Valid phase pixels are colorized in HSV space.
    """
    phase = np.angle(merged_ifg).astype(np.float64)
    rgba = np.zeros((*phase.shape, 4), dtype=np.uint8)

    valid = np.isfinite(phase)
    if merged_coh is not None:
        valid &= np.isfinite(merged_coh) & (merged_coh > 0.0)

    if np.any(valid):
        hue = (phase[valid] + np.pi) / (2.0 * np.pi)
        sat = np.ones_like(hue)
        val = np.full_like(hue, 1.0)
        rgb = np.array(
            [colorsys.hsv_to_rgb(float(h), float(s), float(v)) for h, s, v in zip(hue, sat, val, strict=False)],
            dtype=np.float64,
        )
        rgba[valid, :3] = (rgb * 255.0).astype(np.uint8)
        rgba[valid, 3] = 255
    return rgba


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _write_raster(
    driver,
    path: str,
    data: np.ndarray,
    geo_transform: tuple,
    projection: str,
) -> None:
    """Write a 2-D float32 array as a GDAL raster (noData=0.0)."""
    from osgeo import gdal

    h, w = data.shape
    ds = driver.Create(
        path, w, h, 1, gdal.GDT_Float32,
        options=["TILED=YES", "COMPRESS=DEFLATE"],
    )
    ds.SetGeoTransform(geo_transform)
    ds.SetProjection(projection)
    band = ds.GetRasterBand(1)
    band.WriteArray(data)
    band.SetNoDataValue(0.0)
    band.FlushCache()
    ds = None


# ---------------------------------------------------------------------------
# Geocoding
# ---------------------------------------------------------------------------

def geocode_ifg(
    merged_ifg: np.ndarray,
    merged_coh: np.ndarray,
    burst: BurstRadarGrid,
    dem_path: Path,
    work_dir: Path,
    *,
    res_meters: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Geocode wrapped interferogram and coherence to geographic coordinates.

    Parameters
    ----------
    merged_ifg : np.ndarray
        2-D complex array of the merged (burst-mosaicked) interferogram.
    merged_coh : np.ndarray
        2-D float32 array of the merged coherence (same shape as merged_ifg).
    burst : BurstRadarGrid
        Burst geometry used to derive geotransform parameters.
        The first burst in the selection is used as the reference for
        geotransform computation.
    dem_path : Path
        Path to the DEM GeoTIFF used for geocoding.
    work_dir : Path
        Working directory for temporary files.
    res_meters : float | None
        Target ground resolution in metres.  If None, uses DEM resolution.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(geocoded_ifg, geocoded_coh)`` — both 2-D float32 arrays in
        geographic coordinates.

    Raises
    ------
    NotImplementedError
        When GDAL is not available in the environment.
    """
    # Lazy import so the module can still be imported when GDAL is absent
    # (for dependency-checking purposes).
    try:
        from osgeo import gdal, gdalconst, osr
    except ImportError as exc:
        raise NotImplementedError(
            "geocode_ifg requires GDAL (osgeo). Install with: pip install GDAL"
        ) from exc

    # Load DEM to get ground CRS and pixel spacing
    dem_ds = gdal.Open(str(dem_path), gdalconst.GA_ReadOnly)
    if dem_ds is None:
        raise FileNotFoundError(f"Cannot open DEM: {dem_path}")

    dem_proj = dem_ds.GetProjection()
    dem_gt = dem_ds.GetGeoTransform()
    dem_res = abs(dem_gt[1])  # pixel width in map units

    if res_meters is not None:
        target_res = float(res_meters)
    else:
        target_res = dem_res

    dem_ds = None  # close

    # Build geotransform for the radar raster.
    # Convention: (ul_x, w_e, rot_1, ul_y, rot_2, n_s)
    # We use the burst's starting_range and sensing_start as origin.
    ul_x = float(burst.starting_range)          # slant range → map x
    w_e = float(burst.range_pixel_spacing)      # metres per sample
    ul_y_ts = burst.identity.sensing_start      # first azimuth time

    # n_s: negative because azimuth increases downward in raster but
    # time/decreasing-y in map coordinates
    n_s = -float(burst.azimuth_time_interval)   # seconds per line → -seconds

    geo_transform = (ul_x, w_e, 0.0, ul_y_ts.timestamp(), 0.0, n_s)

    # Use GDAL Warp to geocode.
    import tempfile as _tmp, os as _os

    with _tmp.TemporaryDirectory() as td:
        td = _os.path.join(td)

        src_ifg_path = _os.path.join(td, "src_ifg.tif")
        dst_ifg_path = _os.path.join(td, "dst_ifg.tif")
        src_coh_path = _os.path.join(td, "src_coh.tif")
        dst_coh_path = _os.path.join(td, "dst_coh.tif")

        # Write source IFG raster
        driver = gdal.GetDriverByName("GTiff")
        _write_raster(
            driver, src_ifg_path, merged_ifg.real.astype(np.float32),
            geo_transform, dem_proj,
        )
        # Write source coherence raster
        _write_raster(
            driver, src_coh_path, merged_coh.astype(np.float32),
            geo_transform, dem_proj,
        )

        # Warp to geocoded coordinates
        warp_ds = gdal.Warp(
            dst_ifg_path, src_ifg_path,
            dstSRS=dem_proj,
            xRes=target_res, yRes=target_res,
            resampleAlg=gdalconst.GRA_Bilinear,
        )
        geo_ifg = warp_ds.ReadAsArray()
        warp_ds = None

        warp_ds = gdal.Warp(
            dst_coh_path, src_coh_path,
            dstSRS=dem_proj,
            xRes=target_res, yRes=target_res,
            resampleAlg=gdalconst.GRA_Bilinear,
        )
        geo_coh = warp_ds.ReadAsArray()
        warp_ds = None

    return geo_ifg, geo_coh

    return geo_ifg, geo_coh


# ---------------------------------------------------------------------------
# Phase unwrapping
# ---------------------------------------------------------------------------

# ICU defaults matching strip_insar2.py ICU_DEFAULTS
ICU_DEFAULTS = {
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

# Relaxed ICU params for retry on low valid fraction
ICU_RELAXED_OVERRIDES = {
    "overlap_lines": 400,
    "use_phase_gradient_neutron": True,
    "phase_gradient_window_size": 7,
    "neutron_phase_gradient_threshold": 1.5,
    "trees_number": 12,
    "max_branch_length": 128,
    "initial_correlation_threshold": 0.015,
    "max_correlation_threshold": 0.55,
    "correlation_threshold_increments": 0.03,
    "min_tile_area": 0.0005,
    "bootstrap_lines": 12,
    "min_overlap_area": 8,
    "phase_variance_threshold": 15.0,
}

ICU_MIN_VALID_FRACTION_FOR_RELAXED_RETRY = 0.01


def _build_icu(config: dict | None = None):
    """Build isce3.unwrap.ICU with config overrides."""
    import isce3.unwrap

    cfg = dict(ICU_DEFAULTS)
    if config:
        cfg.update(config)
    unw = isce3.unwrap.ICU()
    unw.corr_incr_thr = cfg["correlation_threshold_increments"]
    unw.buffer_lines = cfg["buffer_lines"]
    unw.overlap_lines = cfg["overlap_lines"]
    unw.use_phase_grad_neut = cfg["use_phase_gradient_neutron"]
    unw.use_intensity_neut = cfg["use_intensity_neutron"]
    unw.phase_grad_win_size = cfg["phase_gradient_window_size"]
    unw.neut_phase_grad_thr = cfg["neutron_phase_gradient_threshold"]
    unw.neut_intensity_thr = cfg["neutron_intensity_threshold"]
    unw.neut_correlation_thr = cfg["max_intensity_correlation_threshold"]
    unw.trees_number = cfg["trees_number"]
    unw.max_branch_length = cfg["max_branch_length"]
    unw.ratio_dxdy = cfg["pixel_spacing_ratio"]
    unw.init_corr_thr = cfg["initial_correlation_threshold"]
    unw.max_corr_thr = cfg["max_correlation_threshold"]
    unw.min_cc_area = cfg["min_tile_area"]
    unw.num_bs_lines = cfg["bootstrap_lines"]
    unw.min_overlap_area = cfg["min_overlap_area"]
    unw.phase_var_thr = cfg["phase_variance_threshold"]
    return unw


def _make_mem_raster(data: np.ndarray, name: str = "mem"):
    """Create a GDAL memory raster from a numpy array."""
    from osgeo import gdal
    h, w = data.shape
    ds = gdal.GetDriverByName("MEM").Create(name, w, h, 1, gdal.GDT_Float32)
    ds.GetRasterBand(1).WriteArray(data)
    return ds


def _unwrap_with_icu(
    phase: np.ndarray,
    coherence: np.ndarray,
    scratch_dir: Path,
    config: dict | None = None,
) -> np.ndarray:
    """Unwrap phase with ICU; NaN where mask rejected."""
    import isce3
    from osgeo import gdal

    scratch_dir.mkdir(parents=True, exist_ok=True)
    h, w = phase.shape

    driver = gdal.GetDriverByName("GTiff")
    unw_ds = driver.Create(str(scratch_dir / "unwrapped.tif"), w, h, 1, gdal.GDT_Float32)
    cc_ds = driver.Create(str(scratch_dir / "conncomp.tif"), w, h, 1, gdal.GDT_Byte)
    if unw_ds is None or cc_ds is None:
        raise RuntimeError("failed to create ICU scratch rasters")
    unw_ds = None
    cc_ds = None

    icu = _build_icu(config)
    icu.unwrap(
        isce3.io.Raster(str(scratch_dir / "unwrapped.tif"), update=True),
        isce3.io.Raster(str(scratch_dir / "conncomp.tif"), update=True),
        _make_mem_raster(phase),
        _make_mem_raster(coherence),
        seed=0,
    )

    ds = gdal.Open(str(scratch_dir / "unwrapped.tif"), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError("ICU did not produce unwrapped phase raster")
    result = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    ds = None

    cc_ds = gdal.Open(str(scratch_dir / "conncomp.tif"), gdal.GA_ReadOnly)
    if cc_ds is None:
        raise RuntimeError("ICU did not produce connected component raster")
    cc = cc_ds.GetRasterBand(1).ReadAsArray()
    cc_ds = None

    result[cc == 0] = np.nan
    if not np.any(np.isfinite(result)):
        raise RuntimeError("ICU produced no finite pixels")
    return result


def _unwrap_with_icu_profiles(
    phase: np.ndarray,
    coherence: np.ndarray,
    scratch_dir: Path,
) -> tuple[np.ndarray, str | None]:
    """Try ICU default, retry relaxed if coverage low."""
    attempts = (
        ("default", None),
        ("relaxed", ICU_RELAXED_OVERRIDES),
    )
    best_result: np.ndarray | None = None
    best_profile: str | None = None
    best_valid_fraction = -1.0

    for profile_name, config in attempts:
        try:
            result = _unwrap_with_icu(
                phase, coherence,
                scratch_dir / profile_name,
                config=config,
            )
            vf = float(np.isfinite(result).mean())
            if vf > best_valid_fraction:
                best_result = result
                best_profile = profile_name
                best_valid_fraction = vf
            if profile_name == "default" and vf >= ICU_MIN_VALID_FRACTION_FOR_RELAXED_RETRY:
                break
        except Exception as exc:
            log.warning("ICU profile %s failed: %s", profile_name, exc)
            continue

    if best_result is None:
        raise RuntimeError("All ICU profiles failed")

    log.info("ICU unwrap: profile=%s valid_fraction=%.4f", best_profile, best_valid_fraction)
    return best_result, best_profile


def _unwrap_with_snaphu(
    phase: np.ndarray,
    coherence: np.ndarray,
    scratch_dir: Path,
    nlooks: float = 1.0,
    range_pixel_spacing: float | None = None,
    azimuth_pixel_spacing: float | None = None,
    wavelength: float | None = None,
    config_overrides: dict | None = None,
) -> np.ndarray:
    """Unwrap phase with Python snaphu module."""
    import snaphu

    scratch_dir.mkdir(parents=True, exist_ok=True)

    cfg = dict(
        nlooks=nlooks,
        cost_mode="defo",
        init_method="mst",
        ntiles_row=1,
        ntiles_col=1,
        row_overlap=0,
        col_overlap=0,
        corr_thresh=0.12 if nlooks <= 1 else 0.05,
        min_conncomp_frac=0.01,
        min_region_size=300,
        phase_grad_window=(5, 5),
        single_tile_reoptimize=False,
        regrow_conncomps=True,
        defomax_cycle=4.0,
        nproc=4,
        tile_cost_thresh=300,
        tile_overlap=(512, 512),
        auto_tile_max_pixels=4_000_000,
    )
    if range_pixel_spacing is not None:
        cfg["range_pixel_spacing"] = range_pixel_spacing
    if azimuth_pixel_spacing is not None:
        cfg["azimuth_pixel_spacing"] = azimuth_pixel_spacing
    if wavelength is not None:
        cfg["wavelength"] = wavelength
    if config_overrides:
        cfg.update(config_overrides)

    # Auto-tile for large images
    rows, cols = phase.shape
    total_pixels = rows * cols
    if total_pixels > cfg["auto_tile_max_pixels"]:
        target = min(cfg["auto_tile_max_pixels"], total_pixels // max(1, cfg["nproc"]))
        tile_side = max(512, (int(np.sqrt(target)) // 128) * 128)
        cfg["ntiles_row"] = max(1, (rows + tile_side - 1) // tile_side)
        cfg["ntiles_col"] = max(1, (cols + tile_side - 1) // tile_side)

    unw = np.zeros(phase.shape, dtype=np.float32)
    conncomp = np.zeros(phase.shape, dtype=np.uint32)

    ntiles_dict = {"n_tiles_row": cfg["ntiles_row"], "n_tiles_col": cfg["ntiles_col"]}

    snaphu.unwrap(
        (coherence * np.exp(1j * phase)).astype(np.complex64),
        coherence.astype(np.float32),
        cfg["nlooks"],
        unw=unw,
        conncomp=conncomp,
        cost=cfg["cost_mode"],
        init=cfg["init_method"],
        min_conncomp_frac=cfg["min_conncomp_frac"],
        phase_grad_window=cfg["phase_grad_window"],
        ntiles=ntiles_dict,
        tile_overlap=cfg["tile_overlap"],
        nproc=cfg["nproc"],
        tile_cost_thresh=cfg["tile_cost_thresh"],
        min_region_size=cfg["min_region_size"],
        single_tile_reoptimize=cfg["single_tile_reoptimize"],
        regrow_conncomps=cfg["regrow_conncomps"],
        row_overlap=cfg["row_overlap"],
        col_overlap=cfg["col_overlap"],
        scratchdir=scratch_dir,
        delete_scratch=True,
    )

    if not np.any(np.isfinite(unw)):
        raise RuntimeError("SNAPHU produced no finite pixels")
    return unw.astype(np.float32)


SNAPHU_RETRY_PROFILES = (
    ("default", {}),
    (
        "relaxed",
        {
            "min_conncomp_frac": 0.001,
            "min_region_size": 100,
            "phase_grad_window": (5, 5),
        },
    ),
)


def unwrap_ifg(
    phase: np.ndarray,
    coherence: np.ndarray,
    method: str,
    *,
    work_dir: Path | None = None,
    nlooks: float = 1.0,
    range_pixel_spacing: float | None = None,
    azimuth_pixel_spacing: float | None = None,
    wavelength: float | None = None,
    use_fallback: bool = True,
) -> np.ndarray:
    """Unwrap 2-D wrapped phase via ICU or SNAPHU with profile retry.

    Parameters
    ----------
    phase : np.ndarray
        2-D wrapped phase in radians (float32 or float64).
    coherence : np.ndarray
        2-D coherence (float32) used as quality mask.
    method : str
        Engine: "icu" or "snaphu".
    work_dir : Path | None
        Scratch directory. System temp if None.
    nlooks : float, default 1.0
        Effective looks (range * azimuth) for SNAPHU cost calibration.
    range_pixel_spacing : float | None
        Slant range pixel spacing (m). Passed to SNAPHU geometry model.
    azimuth_pixel_spacing : float | None
        Azimuth pixel spacing (m). Passed to SNAPHU geometry model.
    wavelength : float | None
        Radar wavelength (m). Passed to SNAPHU deformation cost mode.
    use_fallback : bool, default True
        If True and ICU/SNAPHU fail, fall back to unwrap_phase_2d.

    Returns
    -------
    np.ndarray
        2-D unwrapped phase in radians (float32).
    """
    method_lower = method.lower()
    if method_lower not in ("icu", "snaphu"):
        raise NotImplementedError(
            f"unwrap_ifg only supports 'icu' or 'snaphu', got: {method!r}"
        )

    if work_dir is None:
        work_dir = Path(tempfile.gettempdir())
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    phase = np.asarray(phase, dtype=np.float32)
    coherence = np.asarray(coherence, dtype=np.float32)

    if method_lower == "icu":
        try:
            unwrapped, profile = _unwrap_with_icu_profiles(
                phase, coherence, work_dir / "icu_profiles",
            )
            log.info("ICU unwrap: profile=%s shape=%s", profile, phase.shape)
            return unwrapped
        except Exception as exc:
            log.warning("ICU unwrap failed: %s", exc)
            if not use_fallback:
                raise
            from scripts.tops_utils import unwrap_phase_2d
            log.warning("Falling back to simple 2D unwrap")
            return unwrap_phase_2d(phase)

    # SNAPHU
    for profile_name, overrides in SNAPHU_RETRY_PROFILES:
        try:
            unwrapped = _unwrap_with_snaphu(
                phase, coherence,
                work_dir / f"snaphu_{profile_name}",
                nlooks=nlooks,
                range_pixel_spacing=range_pixel_spacing,
                azimuth_pixel_spacing=azimuth_pixel_spacing,
                wavelength=wavelength,
                config_overrides=overrides,
            )
            log.info(
                "SNAPHU %s: shape=%s nlooks=%.1f",
                profile_name, phase.shape, nlooks,
            )
            return unwrapped
        except Exception as exc:
            log.warning("SNAPHU profile %s failed: %s", profile_name, exc)
            continue

    if use_fallback:
        from scripts.tops_utils import unwrap_phase_2d
        log.warning("SNAPHU failed; using simple 2D unwrap fallback")
        return unwrap_phase_2d(phase)

    raise RuntimeError("SNAPHU unwrap failed -- all retry profiles exhausted")


# ---------------------------------------------------------------------------
# HDF5 product writer
# ---------------------------------------------------------------------------

def write_hdf5_product(
    merged_ifg: np.ndarray,
    merged_coh: np.ndarray,
    unwrapped: np.ndarray | None,
    geo_transform: tuple,
    projection: str,
    output_path: Path,
    metadata: dict,
) -> None:
    """Write a NISAR-style HDF5 InSAR product.

    Creates ``/science/SENTINEL1/interferogram/`` datasets and metadata
    groups as specified in the D2SAR product convention.

    Required datasets
    ----------------
    /science/SENTINEL1/interferogram/phase          (float32)
    /science/SENTINEL1/interferogram/coherence       (float32)
    /science/SENTINEL1/interferogram/unwrappedPhase (float32, optional)
    /science/SENTINEL1/metadata/productType
    /science/SENTINEL1/metadata/burstBoundaries
    /science/SENTINEL1/metadata/lookSide

    Parameters
    ----------
    merged_ifg : np.ndarray
        Complex wrapped interferogram (written as phase in radians).
    merged_coh : np.ndarray
        Coherence (float32, values 0–1).
    unwrapped : np.ndarray | None
        Unwrapped phase (float32, radians).  If None the unwrappedPhase
        dataset is skipped.
    geo_transform : tuple
        GDAL geotransform ``(ul_x, w_e, rot_1, ul_y, rot_2, n_s)``.
    projection : str
        CRS projection string (e.g. EPSG code or WKT).
    output_path : Path
        Destination path for the HDF5 file.
    metadata : dict
        Arbitrary metadata dict written to the root ``/`` group.

    Raises
    ------
    ImportError
        When h5py is not installed.
    """
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "write_hdf5_product requires h5py. Install with: pip install h5py"
        ) from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure phase is wrapped in [-π, π]
    phase_wrapped = np.angle(merged_ifg).astype(np.float32)

    with h5py.File(output_path, "w") as f:
        # Root-level metadata
        for key, value in metadata.items():
            f.attrs[key] = (
                json.dumps(value)
                if not isinstance(value, (str, int, float))
                else value
            )

        # Interferogram group
        grp_ifg = f.create_group("science/SENTINEL1/interferogram")
        grp_ifg.create_dataset(
            "phase", data=phase_wrapped, dtype=np.float32, compression="gzip"
        )
        grp_ifg.create_dataset(
            "coherence", data=merged_coh.astype(np.float32),
            dtype=np.float32, compression="gzip"
        )

        if unwrapped is not None:
            grp_ifg.create_dataset(
                "unwrappedPhase",
                data=unwrapped.astype(np.float32),
                dtype=np.float32,
                compression="gzip",
            )

        # Metadata group
        grp_meta = f.create_group("science/SENTINEL1/metadata")
        grp_meta.create_dataset("productType", data=b"TOPSAR_INSAR")
        grp_meta.create_dataset("lookSide", data=b"right")

        # Burst boundaries: stored as JSON string
        burst_bounds = metadata.get("burstBoundaries", {})
        grp_meta.create_dataset(
            "burstBoundaries",
            data=json.dumps(burst_bounds),
        )

        # Coordinate reference group
        grp_crs = f.create_group("science/SENTINEL1/coordinates")
        grp_crs.attrs["geo_transform"] = json.dumps(list(geo_transform))
        grp_crs.attrs["projection"] = projection

        # Longitude / latitude coordinate datasets (placeholder)
        # These would be computed by geocoding; we write empty stubs
        # sized to match the geocoded output shape.
        if "geocoded_lines" in metadata and "geocoded_samples" in metadata:
            gl = int(metadata["geocoded_lines"])
            gs = int(metadata["geocoded_samples"])
            grp_crs.create_dataset("longitude", shape=(gl, gs), dtype=np.float32)
            grp_crs.create_dataset("latitude", shape=(gl, gs), dtype=np.float32)

    log.info("Wrote HDF5 product: %s", output_path)


# ---------------------------------------------------------------------------
# Product file writer
# ---------------------------------------------------------------------------

def write_product(
    merged_ifg: np.ndarray,
    merged_coh: np.ndarray,
    unwrapped: np.ndarray | None,
    geo_transform: tuple,
    projection: str,
    output_dir: Path,
    product_name: str,
) -> list[Path]:
    """Write geocoded TIFFs and HDF5 product for one swath/product.

    Output files
    -----------
    ``{output_dir}/{product_name}.unw.geo.tif`` — unwrapped + geocoded phase
    ``{output_dir}/{product_name}.int.geo.tif`` — wrapped + geocoded phase
    ``{output_dir}/{product_name}.coh.geo.tif`` — coherence + geocoded
    ``{output_dir}/{product_name}.h5``         — HDF5 product

    The ``.unw.geo.tif`` is skipped when ``unwrapped`` is None.

    Parameters
    ----------
    merged_ifg : np.ndarray
        Complex merged interferogram.
    merged_coh : np.ndarray
        Float32 merged coherence.
    unwrapped : np.ndarray | None
        Unwrapped phase, or None to skip.
    geo_transform : tuple
        GDAL geotransform for the geocoded rasters.
    projection : str
        CRS string for the geocoded rasters.
    output_dir : Path
        Output directory (created if missing).
    product_name : str
        Base filename without extension.

    Returns
    -------
    list[Path]
        Paths of all files written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []

    # Lazy GDAL import
    try:
        from osgeo import gdal, gdalconst
    except ImportError:
        log.warning("GDAL not available — skipping TIFF output")
        return written

    def _write_geo_tiff(
        data: np.ndarray,
        path: Path,
        geo_transform: tuple,
        projection: str,
    ) -> Path:
        """Write a 2-D float32 array as a georeferenced GeoTIFF."""
        driver = gdal.GetDriverByName("GTiff")
        h, w = data.shape
        ds = driver.Create(
            str(path), w, h, 1, gdal.GDT_Float32,
            options=["TILED=YES", "COMPRESS=DEFLATE"],
        )
        ds.SetGeoTransform(geo_transform)
        ds.SetProjection(projection)
        band = ds.GetRasterBand(1)
        band.WriteArray(data)
        band.SetNoDataValue(0.0)
        band.FlushCache()
        ds = None
        log.info("Wrote GeoTIFF: %s", path)
        return path

    # Wrapped phase (color PNG) → .int.geo.png
    int_png_path = output_dir / f"{product_name}.int.geo.png"
    int_rgba = _phase_to_color_rgba(merged_ifg, merged_coh)
    try:
        from PIL import Image

        Image.fromarray(int_rgba, mode="RGBA").save(int_png_path)
        written.append(int_png_path)
        log.info("Wrote PNG: %s", int_png_path)
    except Exception as exc:
        log.warning("Failed to write wrapped-phase PNG %s: %s", int_png_path, exc)

    # Wrapped phase (real part) → .int.geo.tif
    int_data = np.angle(merged_ifg).astype(np.float32)
    int_path = output_dir / f"{product_name}.int.geo.tif"
    _write_geo_tiff(int_data, int_path, geo_transform, projection)
    written.append(int_path)

    # Coherence → .coh.geo.tif
    coh_path = output_dir / f"{product_name}.coh.geo.tif"
    _write_geo_tiff(merged_coh.astype(np.float32), coh_path, geo_transform, projection)
    written.append(coh_path)

    # Unwrapped phase → .unw.geo.tif (only if available)
    if unwrapped is not None:
        unw_path = output_dir / f"{product_name}.unw.geo.tif"
        _write_geo_tiff(unwrapped.astype(np.float32), unw_path, geo_transform, projection)
        written.append(unw_path)

    # HDF5 product
    h5_path = output_dir / f"{product_name}.h5"
    write_hdf5_product(
        merged_ifg=merged_ifg,
        merged_coh=merged_coh,
        unwrapped=unwrapped,
        geo_transform=geo_transform,
        projection=projection,
        output_path=h5_path,
        metadata={
            "productType": "TOPSAR_INSAR",
            "lookSide": "right",
            "geo_transform": list(geo_transform),
            "projection": projection,
        },
    )
    written.append(h5_path)

    return written
