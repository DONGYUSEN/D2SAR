import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from osgeo import gdal

from common_processing import resolve_manifest_data_path, choose_orbit_interp


def _gps_to_datetime(ts: str) -> datetime:
    """Parse ISO timestamp string to UTC datetime."""
    ts = ts.strip()
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    elif not (ts[-6:-5] in ("+", "-") and ts[-6:].count(":") == 3):
        ts = ts + "+00:00"
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _construct_orbit(
    orbit_json: dict, interp_method: str = "Hermite"
) -> "isce3.core.Orbit":
    import isce3.core

    raw_datetimes = [
        _gps_to_datetime(sv["timeUTC"]) for sv in orbit_json["stateVectors"]
    ]
    if len(raw_datetimes) >= 3:
        raw_seconds = np.array(
            [dt.timestamp() for dt in raw_datetimes], dtype=np.float64
        )
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

    ref_dt = isce3.core.DateTime(
        _gps_to_datetime(orbit_json["header"]["firstStateTimeUTC"])
    )
    method_map = {
        "Hermite": isce3.core.OrbitInterpMethod.HERMITE,
        "Legendre": isce3.core.OrbitInterpMethod.LEGENDRE,
    }
    orbit = isce3.core.Orbit(state_vectors, ref_dt, method_map[interp_method])
    return orbit


def _construct_doppler_lut2d(
    doppler_json: dict,
    radargrid_json: dict,
    acquisition_json: dict,
    orbit_json: dict,
) -> "isce3.core.LUT2d":
    import isce3.core

    combined = doppler_json["combinedDoppler"]
    degree = combined["polynomialDegree"]
    coeffs = combined["coefficients"]

    starting_range = (
        isce3.core.speed_of_light * float(radargrid_json["rangeTimeFirstPixel"]) / 2.0
    )
    range_pixel_spacing = float(radargrid_json["columnSpacing"])
    width = int(radargrid_json["numberOfColumns"])
    x_coord = starting_range + range_pixel_spacing * np.arange(width + 1, dtype=np.float64)

    orbit_ref_dt = _gps_to_datetime(orbit_json["header"]["firstStateTimeUTC"])
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
        row += float(c) * np.power(range_times - float(combined["referencePoint"]), j)
    data[:] = row[None, :]

    lut = isce3.core.LUT2d(
        xstart=float(x_coord[0]),
        ystart=float(y_coord[0]),
        dx=(float(x_coord[-1]) - float(x_coord[0])) / max(len(x_coord) - 1, 1),
        dy=(float(y_coord[-1]) - float(y_coord[0])) / max(len(y_coord) - 1, 1),
        data=data,
        method="bilinear",
        b_error=True,
    )
    return lut


def _construct_radar_grid(
    radargrid_json: dict, acquisition_json: dict, orbit_json: dict
) -> "isce3.product.RadarGridParameters":
    import isce3.core
    import isce3.product

    sensing_start_abs_gps = acquisition_json["startGPSTime"]
    orbit_ref_dt = _gps_to_datetime(orbit_json["header"]["firstStateTimeUTC"])

    gps_epoch = datetime(1980, 1, 6, tzinfo=timezone.utc)
    orbit_ref_gps = (orbit_ref_dt - gps_epoch).total_seconds()
    sensing_start_rel = sensing_start_abs_gps - orbit_ref_gps

    wavelength = isce3.core.speed_of_light / acquisition_json["centerFrequency"]
    prf = acquisition_json["prf"]

    r0 = isce3.core.speed_of_light * radargrid_json["rangeTimeFirstPixel"] / 2.0
    range_pixel_spacing = radargrid_json["columnSpacing"]

    look_raw = acquisition_json.get("lookDirection", "RIGHT").strip().upper()
    look_side = (
        isce3.core.LookSide.Left if look_raw == "LEFT" else isce3.core.LookSide.Right
    )

    length = radargrid_json["numberOfRows"]
    width = radargrid_json["numberOfColumns"]

    ref_epoch = isce3.core.DateTime(orbit_ref_dt)

    radar_grid = isce3.product.RadarGridParameters(
        sensing_start=sensing_start_rel,
        wavelength=wavelength,
        prf=prf,
        starting_range=r0,
        range_pixel_spacing=range_pixel_spacing,
        lookside=look_side,
        length=length,
        width=width,
        ref_epoch=ref_epoch,
    )
    return radar_grid


def run_rtc(manifest_path: str, dem_path: str, output_dir: str) -> str:
    import isce3.geometry
    import isce3.io

    manifest_path = Path(manifest_path)
    dem_path = Path(dem_path)
    output_dir = Path(output_dir)

    if not dem_path.exists():
        raise FileNotFoundError(
            f"[RTC] DEM not found at '{dem_path}'. "
            "DEM must be provided externally. Aborting."
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    metadata_dir = manifest_path.parent / "metadata"

    with open(metadata_dir / "orbit.json", "r", encoding="utf-8") as f:
        orbit_data = json.load(f)
    with open(metadata_dir / "radargrid.json", "r", encoding="utf-8") as f:
        radargrid_data = json.load(f)
    with open(metadata_dir / "doppler.json", "r", encoding="utf-8") as f:
        doppler_data = json.load(f)
    with open(metadata_dir / "acquisition.json", "r", encoding="utf-8") as f:
        acquisition_data = json.load(f)

    slc_path = resolve_manifest_data_path(manifest_path, manifest["slc"]["path"])

    print(f"[RTC] Loading SLC: {slc_path}")
    gdal_ds = gdal.Open(slc_path)
    slc_ptr = int(gdal_ds.this)
    slc_raster = isce3.io.Raster(slc_ptr)

    print(f"[RTC] Loading DEM: {dem_path}")
    dem_raster = isce3.io.Raster(str(dem_path))

    orbit_interp = choose_orbit_interp(orbit_data, acquisition_data)

    print(f"[RTC] Constructing Orbit ({orbit_interp})...")
    orbit = _construct_orbit(orbit_data, orbit_interp)

    print("[RTC] Constructing Doppler LUT2d...")
    doppler_lut = _construct_doppler_lut2d(
        doppler_data,
        radargrid_data,
        acquisition_data,
        orbit_data,
    )

    print("[RTC] Constructing RadarGridParameters...")
    radar_grid = _construct_radar_grid(radargrid_data, acquisition_data, orbit_data)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_tiff = output_dir / "rtc.tif"
    output_raster = isce3.io.Raster(
        str(output_tiff),
        radar_grid.width,
        radar_grid.length,
        1,
        6,  # GDT_Float32
        "GTiff",
    )

    print(f"[RTC] Running apply_rtc -> {output_tiff}")
    isce3.geometry.apply_rtc(
        radar_grid,
        orbit,
        doppler_lut,
        slc_raster,
        dem_raster,
        output_raster,
    )

    slc_raster.close_dataset()
    dem_raster.close_dataset()
    output_raster.close_dataset()

    print(f"[RTC] Done. Output: {output_tiff}")
    return str(output_tiff)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python rtc_pipeline.py <manifest.json> <dem_path> <output_dir>")
        sys.exit(1)
    result = run_rtc(sys.argv[1], sys.argv[2], sys.argv[3])
    print(f"RTC output: {result}")
