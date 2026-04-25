from __future__ import annotations

import argparse
import importlib
import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from osgeo import gdal, osr
from PIL import Image

gdal.UseExceptions()

GEO2RDR_OFFSET_NODATA = -999999.0
GEO2RDR_OFFSET_INVALID_LOW = -1.0e5
NISAR_OFFSET_INVALID_VALUE = -1.0e6
_ISCE3_READY = False
_ISCE3_SELECTED_ROOT: Path | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _clear_isce3_modules() -> None:
    for name in list(sys.modules):
        if name == "isce3" or name.startswith("isce3."):
            sys.modules.pop(name, None)


def _strip_repo_isce3_paths() -> None:
    repo_root = _repo_root().resolve()
    preferred_opt_root = Path("/opt/isce3/packages").resolve()
    conflicting_opt_root = Path("/opt/isce3/local/lib/python3.10/dist-packages").resolve()
    filtered: list[str] = []
    for entry in sys.path:
        try:
            entry_path = Path(entry or ".").resolve()
        except Exception:
            filtered.append(entry)
            continue
        if entry_path == repo_root:
            continue
        if entry_path == (repo_root / "isce3"):
            continue
        if str(entry_path).startswith(str((repo_root / "isce3").resolve())):
            continue
        if preferred_opt_root.is_dir() and entry_path == conflicting_opt_root:
            continue
        filtered.append(entry)
    sys.path[:] = filtered


def _candidate_isce3_roots() -> list[Path]:
    repo_root = _repo_root()
    candidates: list[Path] = []
    for path in (
        Path("/opt/isce3/packages"),
    ):
        if path.is_dir():
            candidates.append(path)
    conda_pkg_roots = sorted(
        Path.home().glob("miniconda3/pkgs/isce3-*/lib/python*/site-packages")
    )
    candidates.extend(conda_pkg_roots)
    repo_packages = repo_root / "isce3" / "python" / "packages"
    if repo_packages.is_dir():
        candidates.append(repo_packages)
    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _ensure_nisar_python_packages_on_path() -> Path | None:
    global _ISCE3_READY, _ISCE3_SELECTED_ROOT
    if _ISCE3_READY:
        return _ISCE3_SELECTED_ROOT

    _strip_repo_isce3_paths()
    try:
        mod = importlib.import_module("isce3")
        core_mod = importlib.import_module("isce3.core")
        mod_file = str(Path(getattr(mod, "__file__", "")).resolve()) if getattr(mod, "__file__", None) else ""
        if mod_file and not mod_file.startswith(str(_repo_root().resolve())):
            _ISCE3_READY = True
            _ISCE3_SELECTED_ROOT = None
            return None
        _ = core_mod
    except Exception:
        _clear_isce3_modules()
        pass

    for candidate in _candidate_isce3_roots():
        candidate_str = str(candidate)
        if candidate_str in sys.path:
            sys.path.remove(candidate_str)
        sys.path.insert(0, candidate_str)
        _strip_repo_isce3_paths()
        _clear_isce3_modules()
        try:
            mod = importlib.import_module("isce3")
            importlib.import_module("isce3.core")
            mod_file = str(Path(getattr(mod, "__file__", "")).resolve()) if getattr(mod, "__file__", None) else ""
            if mod_file and mod_file.startswith(str(_repo_root().resolve())):
                continue
            _ISCE3_READY = True
            _ISCE3_SELECTED_ROOT = candidate
            return candidate
        except Exception:
            continue
    return None


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
    except Exception:
        return "Legendre"
    return "Legendre"


def construct_orbit(orbit_json: dict, interp_method: str = "Hermite"):
    _ensure_nisar_python_packages_on_path()
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


def construct_radar_grid(
    radargrid_json: dict,
    acquisition_json: dict,
    orbit_json: dict,
):
    _ensure_nisar_python_packages_on_path()
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


def _read_band_array(
    band,
    *,
    dtype: np.dtype | type | None = None,
) -> np.ndarray:
    arr = band.ReadAsArray()
    if arr is None:
        raise RuntimeError("failed to read raster band")
    if dtype is None:
        return np.asarray(arr)
    return np.asarray(arr, dtype=dtype)


def _write_float_gtiff(path: str | Path, data: np.ndarray, *, nodata=None) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows, cols = data.shape
    ds = gdal.GetDriverByName("GTiff").Create(
        str(path),
        int(cols),
        int(rows),
        1,
        gdal.GDT_Float32,
        options=["COMPRESS=LZW", "TILED=YES"],
    )
    if ds is None:
        raise RuntimeError(f"failed to create raster: {path}")
    band = ds.GetRasterBand(1)
    if nodata is not None:
        band.SetNoDataValue(float(nodata))
    band.WriteArray(np.asarray(data, dtype=np.float32))
    band.FlushCache()
    ds.FlushCache()
    ds = None
    return str(path)


def _build_topo_vrt(target_dir: Path, *, epsg: int) -> str:
    _ensure_nisar_python_packages_on_path()
    import isce3.io

    vrt_path = target_dir / "topo.vrt"
    x_path = target_dir / "x.tif"
    y_path = target_dir / "y.tif"
    z_path = target_dir / "z.tif"
    try:
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
    output_dir: Path,
    block_rows: int,
) -> str:
    _ensure_nisar_python_packages_on_path()
    import isce3.core
    import isce3.geometry
    import isce3.io

    orbit = construct_orbit(orbit_data, orbit_interp)
    radar_grid = construct_radar_grid(radargrid_data, acquisition_data, orbit_data)
    dem_raster = isce3.io.Raster(str(dem_path))

    output_dir.mkdir(parents=True, exist_ok=True)
    topo = isce3.geometry.Rdr2Geo(
        radar_grid,
        orbit,
        isce3.core.Ellipsoid(),
        isce3.core.LUT2d(),
        epsg_out=4326,
        compute_mask=False,
        lines_per_block=int(block_rows),
    )

    def make_raster(name: str, dtype: int = gdal.GDT_Float32):
        return isce3.io.Raster(str(output_dir / name), radar_grid.width, radar_grid.length, 1, dtype, "GTiff")

    x_raster = make_raster("x.tif", gdal.GDT_Float64)
    y_raster = make_raster("y.tif", gdal.GDT_Float64)
    z_raster = make_raster("z.tif", gdal.GDT_Float64)
    inc_raster = make_raster("inc.tif")
    hdg_raster = make_raster("hdg.tif")
    local_inc_raster = make_raster("localInc.tif")
    local_psi_raster = make_raster("localPsi.tif")
    simamp_raster = make_raster("simamp.tif")
    layover_raster = make_raster("layoverShadowMask.tif", gdal.GDT_Byte)
    los_e_raster = make_raster("los_east.tif")
    los_n_raster = make_raster("los_north.tif")

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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Use master topo/rdr2geo results plus master/slave metadata to simulate "
            "flat-earth, topographic, and combined geometric phase without range.off."
        )
    )
    parser.add_argument("--master-acquisition-json", required=True)
    parser.add_argument("--master-radargrid-json", required=True)
    parser.add_argument("--master-orbit-json", required=True)
    parser.add_argument("--slave-acquisition-json", required=True)
    parser.add_argument("--slave-radargrid-json", required=True)
    parser.add_argument("--slave-orbit-json", required=True)
    parser.add_argument("--master-topo-vrt", required=True)
    parser.add_argument("--flat-height", required=True, type=float)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--block-rows", type=int, default=256)
    parser.add_argument("--flat-dem-width", type=int, default=512)
    parser.add_argument("--flat-dem-height", type=int, default=512)
    parser.add_argument("--flat-dem-margin-deg", type=float, default=0.02)
    return parser.parse_args(argv)


def load_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def wrap_phase(phase_rad: np.ndarray) -> np.ndarray:
    phase_rad = np.asarray(phase_rad, dtype=np.float64)
    out = np.angle(np.exp(1j * phase_rad)).astype(np.float32)
    out[~np.isfinite(phase_rad)] = np.nan
    return out


def normalize_geo2rdr_offsets(offsets: np.ndarray) -> np.ndarray:
    arr = np.asarray(offsets, dtype=np.float64).copy()
    invalid = ~np.isfinite(arr)
    invalid |= arr == NISAR_OFFSET_INVALID_VALUE
    invalid |= arr == GEO2RDR_OFFSET_NODATA
    invalid |= arr <= GEO2RDR_OFFSET_INVALID_LOW
    arr[invalid] = np.nan
    return arr


def compute_master_range_grid(
    *,
    rows: int,
    cols: int,
    master_starting_range_m: float,
    master_range_spacing_m: float,
) -> np.ndarray:
    col_index = np.arange(cols, dtype=np.float64)[None, :]
    return np.broadcast_to(
        master_starting_range_m + col_index * master_range_spacing_m,
        (rows, cols),
    ).astype(np.float64)


def compute_slave_range_from_offsets(
    range_offsets: np.ndarray,
    *,
    slave_starting_range_m: float,
    slave_range_spacing_m: float,
) -> np.ndarray:
    offsets = np.asarray(range_offsets, dtype=np.float64)
    cols = offsets.shape[1]
    col_index = np.arange(cols, dtype=np.float64)[None, :]
    slave_pixel = col_index + offsets
    out = slave_starting_range_m + slave_pixel * slave_range_spacing_m
    out[~np.isfinite(offsets)] = np.nan
    return out.astype(np.float64)


def compute_phase_products(
    *,
    master_range_m: np.ndarray,
    slave_true_range_m: np.ndarray,
    slave_flat_range_m: np.ndarray,
    wavelength_m: float,
) -> dict[str, np.ndarray]:
    master_range_m = np.asarray(master_range_m, dtype=np.float64)
    slave_true_range_m = np.asarray(slave_true_range_m, dtype=np.float64)
    slave_flat_range_m = np.asarray(slave_flat_range_m, dtype=np.float64)
    factor = 4.0 * np.pi / float(wavelength_m)

    phi_geom_raw = factor * (slave_true_range_m - master_range_m)
    phi_flat_raw = factor * (slave_flat_range_m - master_range_m)
    phi_topo_raw = phi_geom_raw - phi_flat_raw
    valid = np.isfinite(master_range_m) & np.isfinite(slave_true_range_m) & np.isfinite(slave_flat_range_m)

    phi_geom = wrap_phase(phi_geom_raw)
    phi_flat = wrap_phase(phi_flat_raw)
    phi_topo = wrap_phase(phi_topo_raw)
    phi_geom[~valid] = np.nan
    phi_flat[~valid] = np.nan
    phi_topo[~valid] = np.nan
    return {
        "phi_geom": phi_geom,
        "phi_flat": phi_flat,
        "phi_topo": phi_topo,
    }


def _write_phase_png(path: str | Path, phase: np.ndarray) -> str:
    phase = np.asarray(phase, dtype=np.float32)
    rgb = np.zeros((*phase.shape, 3), dtype=np.uint8)
    valid = np.isfinite(phase)
    if np.any(valid):
        hue = np.mod((phase[valid] + np.pi) / (2.0 * np.pi), 1.0)
        hsv = np.zeros((hue.size, 3), dtype=np.uint8)
        hsv[:, 0] = np.rint(hue * 255.0).astype(np.uint8)
        hsv[:, 1] = 255
        hsv[:, 2] = 255
        rgb[valid] = np.asarray(Image.fromarray(hsv.reshape(-1, 1, 3), mode="HSV").convert("RGB")).reshape(-1, 3)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(path)
    return str(path)


def _write_npy(path: str | Path, array: np.ndarray) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(array))
    return str(path)


def write_phase_products(
    out_dir: str | Path,
    phase_products: dict[str, np.ndarray],
    *,
    summary: dict | None = None,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    product_paths: dict[str, dict[str, str]] = {}
    for name, array in phase_products.items():
        npy_path = out_dir / f"{name}.npy"
        tif_path = out_dir / f"{name}.tif"
        png_path = out_dir / f"{name}.png"
        _write_npy(npy_path, array.astype(np.float32))
        _write_float_gtiff(tif_path, np.asarray(array, dtype=np.float32), nodata=np.nan)
        _write_phase_png(png_path, array)
        product_paths[name] = {
            "npy": str(npy_path),
            "tif": str(tif_path),
            "png": str(png_path),
        }
    payload = dict(summary or {})
    payload["products"] = product_paths
    payload.setdefault("combined_phase_name", "phi_geom")
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["summary_json"] = str(summary_path)
    return payload


def _read_topo_lon_lat(master_topo_vrt_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    ds = gdal.Open(str(master_topo_vrt_path), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"failed to open topo vrt: {master_topo_vrt_path}")
    try:
        lon = _read_band_array(ds.GetRasterBand(1), dtype=np.float64).astype(np.float64)
        lat = _read_band_array(ds.GetRasterBand(2), dtype=np.float64).astype(np.float64)
    finally:
        ds = None
    return lon, lat


def build_flat_dem_from_topo(
    *,
    master_topo_vrt_path: str | Path,
    flat_height_m: float,
    out_path: str | Path,
    width: int,
    height: int,
    margin_deg: float,
) -> str:
    lon, lat = _read_topo_lon_lat(master_topo_vrt_path)
    valid = np.isfinite(lon) & np.isfinite(lat)
    if not np.any(valid):
        raise RuntimeError("master topo vrt has no finite lon/lat samples")

    lon_min = float(np.nanmin(lon[valid]))
    lon_max = float(np.nanmax(lon[valid]))
    lat_min = float(np.nanmin(lat[valid]))
    lat_max = float(np.nanmax(lat[valid]))
    lon_pad = max(float(margin_deg), 0.02 * max(lon_max - lon_min, 1.0e-6))
    lat_pad = max(float(margin_deg), 0.02 * max(lat_max - lat_min, 1.0e-6))
    west = lon_min - lon_pad
    east = lon_max + lon_pad
    south = lat_min - lat_pad
    north = lat_max + lat_pad

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds = gdal.GetDriverByName("GTiff").Create(
        str(out_path),
        int(width),
        int(height),
        1,
        gdal.GDT_Float32,
        options=["COMPRESS=LZW", "TILED=YES"],
    )
    if ds is None:
        raise RuntimeError(f"failed to create flat DEM: {out_path}")
    xres = (east - west) / max(int(width), 1)
    yres = (north - south) / max(int(height), 1)
    ds.SetGeoTransform([west, xres, 0.0, north, 0.0, -yres])
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1)
    band.WriteArray(np.full((int(height), int(width)), float(flat_height_m), dtype=np.float32))
    band.SetNoDataValue(np.nan)
    band.FlushCache()
    ds.FlushCache()
    ds = None
    return str(out_path)


def _read_geo2rdr_offset_file(path: str | Path, *, rows: int, cols: int) -> np.ndarray:
    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    if ds is not None:
        try:
            return _read_band_array(ds.GetRasterBand(1), dtype=np.float64).astype(np.float64)
        finally:
            ds = None
    arr = np.memmap(str(path), dtype=np.float64, mode="r", shape=(rows, cols))
    return np.asarray(arr, dtype=np.float64)


def run_geo2rdr_from_topo(
    *,
    topo_vrt_path: str | Path,
    slave_orbit_data: dict,
    slave_acq_data: dict,
    slave_radargrid_data: dict,
    output_dir: str | Path,
    block_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    _ensure_nisar_python_packages_on_path()
    import isce3.core
    import isce3.geometry
    import isce3.io

    slave_orbit = construct_orbit(
        slave_orbit_data,
        choose_orbit_interp(slave_orbit_data, slave_acq_data),
    )
    slave_grid = construct_radar_grid(slave_radargrid_data, slave_acq_data, slave_orbit_data)
    topo_raster = isce3.io.Raster(str(topo_vrt_path))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = int(getattr(topo_raster, "length", slave_grid.length))
    cols = int(getattr(topo_raster, "width", slave_grid.width))
    try:
        with tempfile.TemporaryDirectory(prefix="simulate_geo2rdr_", dir=str(output_dir)) as tmpdir:
            geo2rdr = isce3.geometry.Geo2Rdr(
                slave_grid,
                slave_orbit,
                isce3.core.Ellipsoid(),
                isce3.core.LUT2d(),
                threshold=1.0e-8,
                numiter=50,
                lines_per_block=int(block_rows),
            )
            geo2rdr.geo2rdr(topo_raster, tmpdir)
            range_offsets = _read_geo2rdr_offset_file(Path(tmpdir) / "range.off", rows=rows, cols=cols)
            azimuth_offsets = _read_geo2rdr_offset_file(Path(tmpdir) / "azimuth.off", rows=rows, cols=cols)
    finally:
        topo_raster.close_dataset()

    range_offsets = normalize_geo2rdr_offsets(range_offsets)
    azimuth_offsets = normalize_geo2rdr_offsets(azimuth_offsets)
    _write_npy(output_dir / "range_offsets.npy", range_offsets.astype(np.float32))
    _write_npy(output_dir / "azimuth_offsets.npy", azimuth_offsets.astype(np.float32))
    _write_float_gtiff(output_dir / "range_offsets.tif", range_offsets.astype(np.float32), nodata=np.nan)
    _write_float_gtiff(output_dir / "azimuth_offsets.tif", azimuth_offsets.astype(np.float32), nodata=np.nan)
    return range_offsets, azimuth_offsets


def simulate_geom_phase(
    *,
    master_acquisition_json: str | Path,
    master_radargrid_json: str | Path,
    master_orbit_json: str | Path,
    slave_acquisition_json: str | Path,
    slave_radargrid_json: str | Path,
    slave_orbit_json: str | Path,
    master_topo_vrt: str | Path,
    flat_height: float,
    out_dir: str | Path,
    block_rows: int = 256,
    flat_dem_width: int = 512,
    flat_dem_height: int = 512,
    flat_dem_margin_deg: float = 0.02,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    master_acq = load_json(master_acquisition_json)
    master_rg = load_json(master_radargrid_json)
    master_orbit = load_json(master_orbit_json)
    slave_acq = load_json(slave_acquisition_json)
    slave_rg = load_json(slave_radargrid_json)
    slave_orbit = load_json(slave_orbit_json)

    wavelength = float(get_wavelength(master_acq))
    slave_wavelength = float(get_wavelength(slave_acq))
    if not np.isclose(wavelength, slave_wavelength, rtol=0.0, atol=1.0e-9):
        raise ValueError(
            f"master/slave centerFrequency mismatch: wavelength {wavelength} vs {slave_wavelength}"
        )

    master_start = _starting_range_from_radargrid(master_rg)
    slave_start = _starting_range_from_radargrid(slave_rg)
    if master_start is None or slave_start is None:
        raise ValueError("failed to determine starting range from master/slave radargrid json")
    master_dr = float(master_rg["columnSpacing"])
    slave_dr = float(slave_rg["columnSpacing"])

    topo_root = out_dir / "topo_products"
    flat_dem_path = build_flat_dem_from_topo(
        master_topo_vrt_path=master_topo_vrt,
        flat_height_m=float(flat_height),
        out_path=topo_root / "flat_dem.tif",
        width=int(flat_dem_width),
        height=int(flat_dem_height),
        margin_deg=float(flat_dem_margin_deg),
    )
    flat_topo_vrt = _run_rdr2geo_topo(
        orbit_data=master_orbit,
        acquisition_data=master_acq,
        radargrid_data=master_rg,
        dem_path=flat_dem_path,
        orbit_interp=choose_orbit_interp(master_orbit, master_acq),
        output_dir=topo_root / "master_flat_topo",
        block_rows=int(block_rows),
    )

    true_rg_offsets, true_az_offsets = run_geo2rdr_from_topo(
        topo_vrt_path=master_topo_vrt,
        slave_orbit_data=slave_orbit,
        slave_acq_data=slave_acq,
        slave_radargrid_data=slave_rg,
        output_dir=out_dir / "geo2rdr_true",
        block_rows=int(block_rows),
    )
    flat_rg_offsets, flat_az_offsets = run_geo2rdr_from_topo(
        topo_vrt_path=flat_topo_vrt,
        slave_orbit_data=slave_orbit,
        slave_acq_data=slave_acq,
        slave_radargrid_data=slave_rg,
        output_dir=out_dir / "geo2rdr_flat",
        block_rows=int(block_rows),
    )

    rows, cols = true_rg_offsets.shape
    master_range = compute_master_range_grid(
        rows=rows,
        cols=cols,
        master_starting_range_m=float(master_start),
        master_range_spacing_m=master_dr,
    )
    slave_range_true = compute_slave_range_from_offsets(
        true_rg_offsets,
        slave_starting_range_m=float(slave_start),
        slave_range_spacing_m=slave_dr,
    )
    slave_range_flat = compute_slave_range_from_offsets(
        flat_rg_offsets,
        slave_starting_range_m=float(slave_start),
        slave_range_spacing_m=slave_dr,
    )

    aux_dir = out_dir / "ranges"
    aux_dir.mkdir(parents=True, exist_ok=True)
    for name, array in (
        ("master_range_m", master_range),
        ("slave_range_true_m", slave_range_true),
        ("slave_range_flat_m", slave_range_flat),
        ("true_azimuth_offsets", true_az_offsets),
        ("flat_azimuth_offsets", flat_az_offsets),
    ):
        _write_npy(aux_dir / f"{name}.npy", array.astype(np.float32))
        _write_float_gtiff(aux_dir / f"{name}.tif", array.astype(np.float32), nodata=np.nan)

    phase_products = compute_phase_products(
        master_range_m=master_range,
        slave_true_range_m=slave_range_true,
        slave_flat_range_m=slave_range_flat,
        wavelength_m=wavelength,
    )
    summary = write_phase_products(
        out_dir,
        phase_products,
        summary={
            "combined_phase_name": "phi_geom",
            "wavelength_m": wavelength,
            "flat_height_m": float(flat_height),
            "master_starting_range_m": float(master_start),
            "slave_starting_range_m": float(slave_start),
            "master_range_spacing_m": master_dr,
            "slave_range_spacing_m": slave_dr,
            "inputs": {
                "master_acquisition_json": str(master_acquisition_json),
                "master_radargrid_json": str(master_radargrid_json),
                "master_orbit_json": str(master_orbit_json),
                "slave_acquisition_json": str(slave_acquisition_json),
                "slave_radargrid_json": str(slave_radargrid_json),
                "slave_orbit_json": str(slave_orbit_json),
                "master_topo_vrt": str(master_topo_vrt),
                "flat_dem_tif": str(flat_dem_path),
                "flat_topo_vrt": str(flat_topo_vrt),
            },
            "auxiliary_outputs": {
                "geo2rdr_true_dir": str(out_dir / "geo2rdr_true"),
                "geo2rdr_flat_dir": str(out_dir / "geo2rdr_flat"),
                "ranges_dir": str(aux_dir),
            },
            "phase_convention": {
                "interferogram": "master * conj(slave)",
                "phi_geom": "wrap(4*pi*(R_slave_true - R_master)/lambda)",
                "phi_flat": "wrap(4*pi*(R_slave_flat - R_master)/lambda)",
                "phi_topo": "wrap(phi_geom_raw - phi_flat_raw)",
                "removal_hint": "interferogram * exp(-1j * phi_geom)",
            },
        },
    )
    return summary


def main(argv=None):
    args = parse_args(argv)
    summary = simulate_geom_phase(
        master_acquisition_json=args.master_acquisition_json,
        master_radargrid_json=args.master_radargrid_json,
        master_orbit_json=args.master_orbit_json,
        slave_acquisition_json=args.slave_acquisition_json,
        slave_radargrid_json=args.slave_radargrid_json,
        slave_orbit_json=args.slave_orbit_json,
        master_topo_vrt=args.master_topo_vrt,
        flat_height=args.flat_height,
        out_dir=args.out_dir,
        block_rows=args.block_rows,
        flat_dem_width=args.flat_dem_width,
        flat_dem_height=args.flat_dem_height,
        flat_dem_margin_deg=args.flat_dem_margin_deg,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
