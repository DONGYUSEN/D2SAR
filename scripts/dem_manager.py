import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from osgeo import gdal


DEFAULT_DEM_CACHE_DIR = os.environ.get("D2SAR_DEM_CACHE_DIR", "/tmp/d2sar_dem_cache")
_default_share_base = os.environ.get("HOME") or tempfile.gettempdir()
D2SAR_SHARE_DIR = os.environ.get(
    "D2SAR_SHARE_DIR", os.path.join(_default_share_base, ".d2sar", "share")
)
STEP_SRTMGL1_URL = "https://step.esa.int/auxdata/dem/SRTMGL1"
EGM96_BINARY_URL = (
    "https://download.osgeo.org/proj/vdatum/egm96_15/outdated/WW15MGH.DAC"
)


def _read_band_array(
    band,
    xoff: int = 0,
    yoff: int = 0,
    xsize: int | None = None,
    ysize: int | None = None,
    *,
    dtype=np.float32,
) -> np.ndarray:
    if xsize is None:
        xsize = band.XSize
    if ysize is None:
        ysize = band.YSize
    np_dtype = np.dtype(dtype)
    buf_type = gdal.GDT_Float64 if np_dtype == np.dtype(np.float64) else gdal.GDT_Float32
    if buf_type == gdal.GDT_Float32:
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
    buf_type = gdal.GDT_Byte if arr.dtype == np.dtype(np.uint8) else gdal.GDT_Float32
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


def _ensure_geoid() -> str:
    """Ensure EGM96 geoid GeoTIFF exists locally; download from OSGeo and convert if needed.

    Returns path to geoid GeoTIFF.
    """
    os.makedirs(D2SAR_SHARE_DIR, exist_ok=True)
    geoid_tif = os.path.join(D2SAR_SHARE_DIR, "geoid_egm96_icgem.tif")

    if os.path.isfile(geoid_tif):
        print(f"[DEM] Geoid found: {geoid_tif}")
        return geoid_tif

    dac_path = os.path.join(D2SAR_SHARE_DIR, "WW15MGH.DAC")
    if not os.path.isfile(dac_path):
        print(f"[DEM] Downloading EGM96 15-min geoid grid from OSGeo...")
        r = subprocess.run(
            ["wget", "-q", "-O", dac_path, EGM96_BINARY_URL],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"[DEM] Failed to download EGM96 from {EGM96_BINARY_URL}\n"
                f"  stderr: {r.stderr}"
            )
        print(f"[DEM] Downloaded EGM96 to {dac_path}")

    print(f"[DEM] Converting EGM96 binary to GeoTIFF...")
    _convert_ww15mgh_to_geotiff(dac_path, geoid_tif)
    print(f"[DEM] Geoid ready: {geoid_tif}")
    return geoid_tif


def _convert_ww15mgh_to_geotiff(dac_path: str, out_path: str):
    """Convert EGM96 WW15MGH.DAC binary grid to GeoTIFF using numpy + GDAL.

    WW15MGH.DAC format:
    - 721 rows x 1440 columns, big-endian unsigned shorts (centimeters)
    - Rows: 90N to 90S (15-min step)
    - Columns: 0E to 360E (15-min step)
    - 32767 = nodata (ocean areas with no data)
    """
    import numpy as np
    from osgeo import gdal

    gdal.UseExceptions()

    # WW15MGH.DAC is big-endian SIGNED 2-byte integers (centimeters)
    data = np.fromfile(dac_path, dtype=">i2").reshape(721, 1440)

    nodata_raw = -32768  # sentinel for missing ocean areas
    geoid_m = data.astype(np.float32) / 100.0
    geoid_m[data == nodata_raw] = np.nan

    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(
        out_path,
        1440,
        721,
        1,
        gdal.GDT_Float32,
        options=["COMPRESS=LZW", "TILED=YES"],
    )
    ds.SetGeoTransform(
        [
            0.0,
            0.25,
            0.0,
            90.0,
            0.0,
            -0.25,
        ]
    )
    ds.SetProjection(
        'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
        'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'
    )
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(np.nan)
    _write_band_array(band, geoid_m)
    band.ComputeStatistics(False)
    ds.FlushCache()
    ds = None


def _convert_srtm_hgt_to_geotiff(hgt_path: str, out_path: str, tile_name: str):
    """Convert an SRTM .hgt tile to GeoTIFF without relying on GDAL's HGT driver."""
    data = np.fromfile(hgt_path, dtype=">i2")
    n = int(round(np.sqrt(data.size)))
    if n * n != data.size:
        raise RuntimeError(
            f"[DEM] Unexpected HGT sample count in {hgt_path}: {data.size}"
        )
    arr = data.reshape(n, n).astype(np.float32)
    arr[arr <= -32768] = np.nan

    if tile_name[0] == "N":
        lat0 = int(tile_name[1:3])
    else:
        lat0 = -int(tile_name[1:3])
    if tile_name[3] == "E":
        lon0 = int(tile_name[4:7])
    else:
        lon0 = -int(tile_name[4:7])

    pixel_size = 1.0 / (n - 1)

    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(
        out_path,
        n,
        n,
        1,
        gdal.GDT_Float32,
        options=["COMPRESS=LZW", "TILED=YES"],
    )
    ds.SetGeoTransform([lon0, pixel_size, 0.0, lat0 + 1.0, 0.0, -pixel_size])
    ds.SetProjection(
        'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
        'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'
    )
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(np.nan)
    _write_band_array(band, arr)
    ds.FlushCache()
    ds = None


def _gdal(cmd_args, check=True):
    result = subprocess.run(cmd_args, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(
            f"[DEM] GDAL command failed: {' '.join(cmd_args)}\n{result.stderr}"
        )
    return result


def _normalize_lon_360(lon):
    lon = float(lon) % 360.0
    return lon if lon >= 0 else lon + 360.0


def _canonicalize_unwrapped_west(west, east):
    while west < -180.0:
        west += 360.0
        east += 360.0
    while west >= 180.0:
        west -= 360.0
        east -= 360.0
    return west, east


def _scene_lon_interval(lons, margin_deg=0.0):
    normalized = sorted(_normalize_lon_360(lon) for lon in lons)
    if not normalized:
        raise ValueError("No longitudes provided")
    if any(not np.isfinite(lon) for lon in normalized):
        raise ValueError("Scene contains non-finite longitudes")

    if len(normalized) == 1:
        start = normalized[0]
        span = 0.0
    else:
        gaps = []
        for idx, lon in enumerate(normalized):
            next_lon = normalized[(idx + 1) % len(normalized)]
            if idx == len(normalized) - 1:
                next_lon += 360.0
            gaps.append(next_lon - lon)
        max_gap_idx = int(np.argmax(gaps))
        max_gap = gaps[max_gap_idx]
        span = 360.0 - max_gap
        start = normalized[(max_gap_idx + 1) % len(normalized)]

    west = start
    east = start + span
    west, east = _canonicalize_unwrapped_west(west, east)
    west -= margin_deg
    east += margin_deg
    west, east = _canonicalize_unwrapped_west(west, east)

    if east <= west or east - west >= 180.0:
        raise ValueError(f"Scene longitude span ({east - west:.1f} deg) is unsupported")
    return west, east


def _bbox_snwe_to_tile_names(south, north, west, east):
    def floor(v):
        iv = int(v)
        return iv if (v >= 0 or v == iv) else iv - 1

    def ceil(v):
        iv = int(v)
        return iv if (v <= 0 or v == iv) else iv + 1

    lat_s, lat_e = floor(south), ceil(north)
    if lat_s >= lat_e:
        raise ValueError(f"Invalid bbox: south={south} north={north}")

    def _tile_lon(lon):
        while lon < -180:
            lon += 360
        while lon >= 180:
            lon -= 360
        if lon == -180:
            return "W180"
        return f"E{lon:03d}" if lon >= 0 else f"W{-lon:03d}"

    def _tile_slice(lon_start, lon_end):
        tiles = []
        lon = lon_start
        while lon < lon_end:
            tiles.append(_tile_lon(lon))
            lon += 1
        return tiles

    lon_s = floor(west)
    lon_e = ceil(east)
    lat = lat_s
    tiles = []
    while lat < lat_e:
        lat_tag = f"N{lat:02d}" if lat >= 0 else f"S{-lat:02d}"
        tiles.extend([f"{lat_tag}{t}" for t in _tile_slice(lon_s, lon_e)])
        lat += 1
    return tiles


def _shift_dem_source_to_unwrapped_branch(src_path, dst_path, shift_deg):
    ds = gdal.Open(src_path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"[DEM] Failed to open source DEM tile: {src_path}")
    gt = ds.GetGeoTransform()
    xmin = gt[0]
    ymax = gt[3]
    xmax = xmin + ds.RasterXSize * gt[1]
    ymin = ymax + ds.RasterYSize * gt[5]
    xres = abs(gt[1])
    yres = abs(gt[5])
    ds = None
    shifted = gdal.Warp(
        dst_path,
        src_path,
        outputBounds=(xmin + shift_deg, ymin, xmax + shift_deg, ymax),
        outputBoundsSRS="EPSG:4326",
        dstSRS="EPSG:4326",
        xRes=xres,
        yRes=yres,
        resampleAlg="near",
        dstNodata=-32768,
    )
    if shifted is None:
        raise RuntimeError(f"[DEM] Failed to shift source DEM tile: {src_path}")
    shifted = None
    return dst_path


def fetch_dem(bbox, output_dir=None, source=1, correct_geoid=True):
    """Download, mosaic, and geoid-correct SRTMGL1 DEM using GDAL only.

    Parameters
    ----------
    bbox : list [west, east, south, north]
    output_dir : str, optional
        Output directory. Defaults to D2SAR_DEM_CACHE_DIR.
    source : int
        1=SRTMGL1 (1-arcsec). Only source=1 downloads from ESA.
    correct_geoid : bool
        Subtract EGM96 geoid (WGS84 ellipsoid correction).

    Returns
    -------
    str
        Path to output DEM GeoTIFF.
    """
    west, east, south, north = bbox

    if output_dir is None:
        output_dir = DEFAULT_DEM_CACHE_DIR
    os.makedirs(output_dir, exist_ok=True)

    tiles = _bbox_snwe_to_tile_names(south, north, west, east)
    print(f"[DEM] {len(tiles)} tiles needed for bbox={bbox}")

    tile_vrt_sources = []
    missing_tiles = []

    def ensure_extracted_hgt(zip_path: str, tile_name: str) -> str:
        hgt_path = os.path.join(output_dir, f"{tile_name}.hgt")
        tif_path = os.path.join(output_dir, f"{tile_name}.tif")
        if os.path.isfile(tif_path):
            return tif_path
        if os.path.isfile(hgt_path):
            _convert_srtm_hgt_to_geotiff(hgt_path, tif_path, tile_name)
            return tif_path
        with zipfile.ZipFile(zip_path) as zf:
            member = next(
                (n for n in zf.namelist() if n.lower().endswith(".hgt")), None
            )
            if member is None:
                raise RuntimeError(f"[DEM] No .hgt member found in {zip_path}")
            print(f"[DEM] Extracting {member} from {zip_path} ...")
            zf.extract(member, output_dir)
            extracted = os.path.join(output_dir, member)
            if extracted != hgt_path:
                shutil.move(extracted, hgt_path)
        _convert_srtm_hgt_to_geotiff(hgt_path, tif_path, tile_name)
        return tif_path

    for tile_name in tiles:
        for variant in (
            os.path.join(output_dir, f"{tile_name}.tif"),
            os.path.join(output_dir, f"{tile_name}.hgt"),
            os.path.join(output_dir, f"{tile_name}.hgt.zip"),
            os.path.join(output_dir, f"{tile_name}.SRTMGL1.hgt.zip"),
        ):
            if os.path.isfile(variant):
                if variant.endswith(".tif"):
                    tile_vrt_sources.append(variant)
                elif variant.endswith(".hgt"):
                    tif_path = os.path.join(output_dir, f"{tile_name}.tif")
                    _convert_srtm_hgt_to_geotiff(variant, tif_path, tile_name)
                    tile_vrt_sources.append(tif_path)
                else:
                    try:
                        tile_vrt_sources.append(
                            ensure_extracted_hgt(variant, tile_name)
                        )
                    except zipfile.BadZipFile:
                        print(
                            f"[DEM] Invalid ZIP detected, removing and re-downloading: {variant}"
                        )
                        os.remove(variant)
                        continue
                break
        else:
            if source == 1:
                zip_name = f"{tile_name}.SRTMGL1.hgt.zip"
                url = f"{STEP_SRTMGL1_URL}/{zip_name}"
                zip_path = os.path.join(output_dir, zip_name)
                if not os.path.isfile(zip_path):
                    print(f"[DEM] Downloading {tile_name} from ESA: {url}")
                    r = subprocess.run(
                        ["wget", "-q", "-O", zip_path, url],
                        capture_output=True,
                        text=True,
                    )
                    if r.returncode != 0:
                        missing_tiles.append(tile_name)
                        continue
                try:
                    tile_vrt_sources.append(ensure_extracted_hgt(zip_path, tile_name))
                except zipfile.BadZipFile:
                    print(f"[DEM] Re-downloading corrupted ZIP: {zip_path}")
                    if os.path.exists(zip_path):
                        os.remove(zip_path)
                    r = subprocess.run(
                        ["wget", "-q", "-O", zip_path, url],
                        capture_output=True,
                        text=True,
                    )
                    if r.returncode != 0:
                        missing_tiles.append(tile_name)
                        continue
                    tile_vrt_sources.append(ensure_extracted_hgt(zip_path, tile_name))

    if missing_tiles:
        raise RuntimeError(f"[DEM] Failed to download required tiles: {missing_tiles}")

    if not tile_vrt_sources:
        raise RuntimeError(f"[DEM] No tiles available for bbox={bbox}")

    tmp_dir = tempfile.mkdtemp(prefix="d2sar_dem_")
    try:
        if east > 180.0:
            shifted_sources = []
            for src in tile_vrt_sources:
                ds, xmin, xmax, ymin, ymax = _dem_bbox(src)
                ds = None
                if xmax <= 0.0:
                    shifted_sources.append(
                        _shift_dem_source_to_unwrapped_branch(
                            src,
                            os.path.join(tmp_dir, f"shifted_{os.path.basename(src)}"),
                            360.0,
                        )
                    )
                else:
                    shifted_sources.append(src)
            tile_vrt_sources = shifted_sources

        vrt_path = os.path.join(tmp_dir, "mosaic.vrt")
        print(f"[DEM] Building VRT mosaic from {len(tile_vrt_sources)} tiles...")
        vrt_ds = gdal.BuildVRT(vrt_path, tile_vrt_sources)
        if vrt_ds is None:
            raise RuntimeError("[DEM] gdal.BuildVRT failed")
        vrt_ds = None

        clipped_path = os.path.join(tmp_dir, "dem_clip.tif")
        print(f"[DEM] Clipping to bbox {bbox}...")
        warp_ds = gdal.Warp(
            clipped_path,
            vrt_path,
            outputBounds=(west, south, east, north),
            outputBoundsSRS="EPSG:4326",
            dstSRS="EPSG:4326",
            resampleAlg="bilinear",
            dstNodata=-32768,
        )
        if warp_ds is None:
            raise RuntimeError("[DEM] gdal.Warp clipping failed")
        warp_ds = None

        dem_path = os.path.join(
            output_dir,
            f"dem_{south:.4f}_{north:.4f}_{west:.4f}_{east:.4f}.tif",
        )

        print(f"[DEM] Filling nodata holes...")
        filled_path = _fill_nodata_gdal(clipped_path, tmp_dir)

        if correct_geoid:
            print(f"[DEM] Applying EGM96 geoid correction...")
            dem_path = _subtract_geoid_gdal(filled_path, tmp_dir)
            final_path = os.path.join(output_dir, os.path.basename(dem_path))
            if os.path.exists(final_path):
                os.remove(final_path)
            shutil.move(dem_path, final_path)
            dem_path = final_path
        else:
            shutil.copy2(filled_path, dem_path)

        print(f"[DEM] DEM ready: {dem_path}")
        return dem_path

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _fill_nodata_gdal(dem_path, work_dir):
    """Fill nodata holes using GDAL FillNodata (IDW)."""
    from osgeo import gdal

    gdal.UseExceptions()

    ds = gdal.Open(dem_path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"[DEM] Cannot open {dem_path}")

    band = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    if nodata is None:
        nodata = -32768

    arr = _read_band_array(band)

    missing = int((arr == nodata).sum())
    print(f"[DEM] Nodata pixels before filling: {missing}")
    ds = None

    if missing == 0:
        return dem_path

    src_ds = gdal.Open(dem_path, gdal.GA_ReadOnly)
    out_path = os.path.join(work_dir, "dem_filled.tif")
    dst_ds = gdal.GetDriverByName("GTiff").Create(
        out_path,
        src_ds.RasterXSize,
        src_ds.RasterYSize,
        1,
        gdal.GDT_Float64,
    )
    dst_ds.SetGeoTransform(src_ds.GetGeoTransform())
    dst_ds.SetProjection(src_ds.GetProjection())
    dst_ds.GetRasterBand(1).SetNoDataValue(nodata)

    band_in = src_ds.GetRasterBand(1)
    mask = gdal.GetDriverByName("MEM").Create(
        "",
        src_ds.RasterXSize,
        src_ds.RasterYSize,
        1,
        gdal.GDT_Byte,
    )
    mask_band = mask.GetRasterBand(1)
    valid = (arr != nodata).astype("u1") * 255
    _write_band_array(mask_band, valid)

    gdal.FillNodata(
        targetBand=dst_ds.GetRasterBand(1),
        maskBand=mask_band,
        maxSearchDist=100,
        smoothingIterations=0,
    )
    src_ds = None
    mask = None
    dst_ds.FlushCache()
    dst_ds = None

    print(f"[DEM] Nodata filled: {out_path}")
    return out_path


def _subtract_geoid_gdal(dem_path, work_dir):
    """Subtract EGM96 geoid from DEM using pure GDAL.

    Clips geoid to DEM extent, resamples to DEM resolution,
    then subtracts using gdal_calc.py.
    """
    from osgeo import gdal

    gdal.UseExceptions()

    geoid_path = _ensure_geoid()

    dem_ds = gdal.Open(dem_path, gdal.GA_ReadOnly)
    if dem_ds is None:
        raise RuntimeError(f"[DEM] Cannot open DEM: {dem_path}")
    gt = dem_ds.GetGeoTransform()
    cols = dem_ds.RasterXSize
    rows = dem_ds.RasterYSize
    dem_proj = dem_ds.GetProjection()
    west, north = gt[0], gt[3]
    east = west + gt[1] * cols
    south = north + gt[5] * rows
    dem_ds = None

    clipped_geoid = os.path.join(work_dir, "geoid_clip.tif")
    print(f"[DEM] Clipping geoid to DEM extent ({west},{south},{east},{north})...")
    warp_ds = gdal.Warp(
        clipped_geoid,
        geoid_path,
        outputBounds=(west, south, east, north),
        outputBoundsSRS="EPSG:4326",
        dstSRS="EPSG:4326",
        resampleAlg="bilinear",
        dstNodata=0,
    )
    if warp_ds is None:
        raise RuntimeError("[DEM] gdal.Warp geoid clip failed")
    warp_ds = None

    geoid_ds = gdal.Open(clipped_geoid, gdal.GA_ReadOnly)
    if geoid_ds is None:
        raise RuntimeError(f"[DEM] Cannot open clipped geoid: {clipped_geoid}")
    g_gt = geoid_ds.GetGeoTransform()
    g_cols = geoid_ds.RasterXSize
    g_rows = geoid_ds.RasterYSize
    geoid_ds = None

    print(f"[DEM] Geoid: {g_cols}x{g_rows}, DEM: {cols}x{rows}")

    resampled_geoid = os.path.join(work_dir, "geoid_rs.tif")
    if g_cols != cols or g_rows != rows:
        print(f"[DEM] Resampling geoid to DEM resolution {cols}x{rows}...")
        warp_ds = gdal.Warp(
            resampled_geoid,
            clipped_geoid,
            width=cols,
            height=rows,
            resampleAlg="bilinear",
            dstSRS=dem_proj,
        )
        if warp_ds is None:
            raise RuntimeError("[DEM] gdal.Warp geoid resample failed")
        warp_ds = None
    else:
        shutil.copy2(clipped_geoid, resampled_geoid)

    out_path = dem_path.replace(".tif", "_wgs84.tif")
    print(f"[DEM] Subtracting geoid...")
    dem_ds = gdal.Open(dem_path, gdal.GA_ReadOnly)
    geoid_ds = gdal.Open(resampled_geoid, gdal.GA_ReadOnly)
    dem_arr = _read_band_array(dem_ds.GetRasterBand(1)).astype(np.float32)
    geoid_arr = _read_band_array(geoid_ds.GetRasterBand(1)).astype(np.float32)
    out_arr = dem_arr - geoid_arr
    drv = gdal.GetDriverByName("GTiff")
    out_ds = drv.Create(
        out_path, cols, rows, 1, gdal.GDT_Float32, options=["COMPRESS=LZW", "TILED=YES"]
    )
    out_ds.SetGeoTransform(dem_ds.GetGeoTransform())
    out_ds.SetProjection(dem_ds.GetProjection())
    out_band = out_ds.GetRasterBand(1)
    out_band.SetNoDataValue(-32768)
    _write_band_array(out_band, out_arr)
    out_ds.FlushCache()
    out_ds = None
    dem_ds = None
    geoid_ds = None

    print(f"[DEM] Geoid-corrected DEM: {out_path}")
    return out_path


def dem_from_scene_corners(corners, output_dir=None, **kwargs):
    """Fetch DEM given scene corner coordinates from meta.xml."""
    corners = normalize_scene_corners(corners)
    lats = [c["lat"] for c in corners]
    lons = [c["lon"] for c in corners]
    south, north = min(lats), max(lats)
    west, east = min(lons), max(lons)
    bbox = [west, east, south, north]
    print(f"[DEM] Bbox from corners (W,E,S,N): {bbox}")
    return fetch_dem(bbox, output_dir=output_dir, **kwargs)


def find_dem_in_directory_for_scene(corners, dem_dir, margin_deg=0.05):
    """Find an existing DEM file in a directory that covers the scene corners."""
    corners = normalize_scene_corners(corners)
    dem_dir = Path(dem_dir)
    if not dem_dir.exists():
        raise FileNotFoundError(f"[DEM] DEM directory does not exist: {dem_dir}")
    if not dem_dir.is_dir():
        raise NotADirectoryError(f"[DEM] DEM directory is not a directory: {dem_dir}")

    candidates = []
    for p in sorted(dem_dir.rglob("*")):
        if not p.is_file():
            continue
        suffix = p.suffix.lower()
        if suffix in {".tif", ".tiff", ".hgt"}:
            candidates.append(p)
        elif suffix == ".zip" and ".hgt" in p.name.lower():
            try:
                with zipfile.ZipFile(p) as zf:
                    member = next(
                        (
                            name
                            for name in zf.namelist()
                            if name.lower().endswith(".hgt")
                        ),
                        None,
                    )
                    if member is None:
                        continue
                    extracted = dem_dir / Path(member).name
                    if not extracted.exists():
                        print(f"[DEM] Extracting {member} from {p} ...")
                        zf.extract(member, dem_dir)
                        extracted_src = dem_dir / member
                        if extracted_src != extracted:
                            extracted.parent.mkdir(parents=True, exist_ok=True)
                            shutil.move(str(extracted_src), str(extracted))
                    candidates.append(extracted)
            except zipfile.BadZipFile:
                continue

    failures = []
    for candidate in candidates:
        try:
            check = dem_covers_scene_corners(
                str(candidate), corners, margin_deg=margin_deg
            )
        except Exception as exc:
            failures.append({"path": str(candidate), "error": str(exc)})
            continue
        if check["ok"]:
            print(f"[DEM] Existing DEM from directory validated: {candidate}")
            return str(candidate), check
        failures.append({"path": str(candidate), "check": check})

    raise FileNotFoundError(
        f"[DEM] No DEM in directory covers the scene: {dem_dir}. "
        f"Checked {len(candidates)} candidate(s). Failures: {failures[:5]}"
    )


def normalize_scene_corners(corners):
    if isinstance(corners, dict):
        return list(corners.values())
    if isinstance(corners, list):
        return corners
    raise TypeError(f"Unsupported corners type: {type(corners)!r}")


def scene_bbox_from_corners(corners, margin_deg=0.0):
    corners = normalize_scene_corners(corners)
    lats = [c["lat"] for c in corners]
    lons = [c["lon"] for c in corners]
    south = min(lats) - margin_deg
    north = max(lats) + margin_deg
    west, east = _scene_lon_interval(lons, margin_deg=margin_deg)
    return [round(west, 10), round(east, 10), round(south, 10), round(north, 10)]


def _dem_bbox(dem_path):
    ds = gdal.Open(dem_path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"[DEM] Failed to open DEM: {dem_path}")
    gt = ds.GetGeoTransform()
    width = ds.RasterXSize
    length = ds.RasterYSize
    xmin = gt[0]
    ymax = gt[3]
    xmax = xmin + width * gt[1]
    ymin = ymax + length * gt[5]
    xmin, xmax = sorted((xmin, xmax))
    ymin, ymax = sorted((ymin, ymax))
    return ds, xmin, xmax, ymin, ymax


def _shift_lon_into_range(lon, xmin, xmax):
    candidates = [lon - 360.0, lon, lon + 360.0]
    for candidate in candidates:
        if xmin <= candidate <= xmax:
            return candidate
    return None


def _shift_interval_into_range(west, east, xmin, xmax):
    for shift in (-360.0, 0.0, 360.0):
        shifted_west = west + shift
        shifted_east = east + shift
        if shifted_west >= xmin and shifted_east <= xmax:
            return shifted_west, shifted_east
    return None


def dem_covers_scene_corners(dem_path, corners, margin_deg=0.0):
    corners = normalize_scene_corners(corners)
    ds, xmin, xmax, ymin, ymax = _dem_bbox(dem_path)
    west, east, south, north = scene_bbox_from_corners(corners, margin_deg)
    shifted_bbox = _shift_interval_into_range(west, east, xmin, xmax)
    if shifted_bbox is None or south < ymin or north > ymax:
        return {
            "ok": False,
            "reason": "bbox_outside_dem",
            "scene_bbox": [west, east, south, north],
            "dem_bbox": [xmin, xmax, ymin, ymax],
        }
    shifted_west, shifted_east = shifted_bbox

    band = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    gt = ds.GetGeoTransform()
    failures = []
    samples = corners + [
        {
            "line": "center",
            "pixel": "center",
            "lon": 0.5 * (shifted_west + shifted_east),
            "lat": 0.5 * (south + north),
        }
    ]
    for pt in samples:
        shifted_lon = _shift_lon_into_range(pt["lon"], xmin, xmax)
        if shifted_lon is None:
            failures.append({"point": pt, "reason": "outside_longitude_range"})
            continue
        x = int((shifted_lon - gt[0]) / gt[1])
        y = int((pt["lat"] - gt[3]) / gt[5])
        inside = 0 <= x < ds.RasterXSize and 0 <= y < ds.RasterYSize
        if not inside:
            failures.append({"point": pt, "reason": "outside_pixel_window"})
            continue
        val = float(_read_band_array(band, x, y, 1, 1)[0, 0])
        if not np.isfinite(val) or (nodata is not None and np.isclose(val, nodata)):
            failures.append({"point": pt, "reason": "nodata", "value": val})

    if failures:
        return {
            "ok": False,
            "reason": "corner_or_center_nodata",
            "scene_bbox": [west, east, south, north],
            "dem_bbox": [xmin, xmax, ymin, ymax],
            "failures": failures,
        }

    return {
        "ok": True,
        "scene_bbox": [west, east, south, north],
        "dem_bbox": [xmin, xmax, ymin, ymax],
    }


def resolve_dem_for_scene(
    corners,
    dem_path=None,
    output_dir=None,
    margin_deg=0.05,
    source=1,
    correct_geoid=True,
):
    corners = normalize_scene_corners(corners)
    if dem_path:
        check = dem_covers_scene_corners(dem_path, corners, margin_deg=margin_deg)
        if check["ok"]:
            print(f"[DEM] Existing DEM validated: {dem_path}")
            return dem_path, check
        print(f"[DEM] Existing DEM rejected: {check}")

    dem_path = fetch_dem(
        scene_bbox_from_corners(corners, margin_deg=margin_deg),
        output_dir=output_dir,
        source=source,
        correct_geoid=correct_geoid,
    )
    check = dem_covers_scene_corners(dem_path, corners, margin_deg=0.0)
    if not check["ok"]:
        raise RuntimeError(f"[DEM] Auto-downloaded DEM still invalid: {check}")
    print(f"[DEM] Scene DEM ready: {dem_path}")
    return dem_path, check


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Auto-download SRTMGL1 DEM (pure GDAL)"
    )
    parser.add_argument(
        "-b",
        "--bbox",
        type=float,
        nargs=4,
        help="Bounding box: west east south north",
        required=True,
    )
    parser.add_argument("-o", "--output", type=str, default=DEFAULT_DEM_CACHE_DIR)
    parser.add_argument("--no-geoid", action="store_true", help="Skip geoid correction")
    args = parser.parse_args()

    result = fetch_dem(
        args.bbox, output_dir=args.output, correct_geoid=not args.no_geoid
    )
    print(f"Output: {result}")
