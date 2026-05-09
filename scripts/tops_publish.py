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
]

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

def unwrap_ifg(
    phase: np.ndarray,
    coherence: np.ndarray,
    method: str,
    *,
    work_dir: Path | None = None,
) -> np.ndarray:
    """Unwrap 2-D wrapped phase via ICU or SNAPHU.

    Parameters
    ----------
    phase : np.ndarray
        2-D wrapped phase in radians (float32 or float64).
    coherence : np.ndarray
        2-D coherence (float32) used as a quality mask.
    method : str
        Unwrapping engine: ``"icu"`` or ``"snaphu"``.
    work_dir : Path | None
        Directory for temporary files.  If None, a system temp dir is used.

    Returns
    -------
    np.ndarray
        2-D unwrapped phase in radians (float32).

    Raises
    ------
    NotImplementedError
        When ``method`` is not ``"icu"`` or ``"snaphu"``, or when the
        requested tool is not installed.
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

    # Write input rasters
    phase_path = work_dir / "phase_input.bin"
    coh_path = work_dir / "coh_input.bin"
    unw_path = work_dir / "unw_output.bin"

    phase.astype(np.float32).tofile(str(phase_path))
    coherence.astype(np.float32).tofile(str(coh_path))

    h, w = phase.shape

    if method_lower == "icu":
        # ICU: GPU-accelerated unwrapper (part of ISCE2/ISCE3).
        # Invocation: icu -i phase.cor -o unwrapped.int
        # We construct a minimal ICU command.
        icu_exe = shutil.which("icu")
        if icu_exe is None:
            raise NotImplementedError(
                "ICU executable not found in PATH. "
                "Install ISCE2/ISCE3 and ensure 'icu' is on PATH."
            )

        cmd = [
            icu_exe,
            "-i", str(coh_path),
            "-o", str(unw_path),
            "-m", str(w),
            "-n", str(h),
        ]
        log.info("Running ICU: %s", " ".join(cmd))
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True,
        )
        if result.returncode != 0:
            log.error("ICU stderr: %s", result.stderr)
            raise RuntimeError(f"ICU failed: {result.stderr}")

    elif method_lower == "snaphu":
        # SNAPHU: CPU unwrapper (https://github.com/isce-framework/snaphu).
        snaphu_exe = shutil.which("snaphu")
        if snaphu_exe is None:
            raise NotImplementedError(
                "SNAPHU executable not found in PATH. "
                "Install SNAPHU: https://github.com/isce-framework/snaphu"
            )

        # SNAPHU config: tiled processing, correlation threshold 0.1
        cmd = [
            snaphu_exe,
            "-f", str(coh_path),   # using coh as cost source file
            "-o", str(unw_path),
            "--nrows", str(h),
            "--ncols", str(w),
            "-t", "0.1",
        ]
        log.info("Running SNAPHU: %s", " ".join(cmd))
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True,
        )
        if result.returncode != 0:
            log.error("SNAPHU stderr: %s", result.stderr)
            raise RuntimeError(f"SNAPHU failed: {result.stderr}")

    if not unw_path.exists():
        raise FileNotFoundError(
            f"Unwrap output not produced at {unw_path}. "
            "Check unwrapper logs above."
        )

    unwrapped = np.fromfile(str(unw_path), dtype=np.float32).reshape(h, w)
    return unwrapped


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
