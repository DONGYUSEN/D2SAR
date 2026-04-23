"""
手动几何 RTC Pipeline - 替代 isce3 geo2rdr 的不依赖 geo2rdr 收敛问题。

算法流程:
  1. 建立 GeoGrid (输出地理网格)
  2. 对每个 GeoGrid 像素，用手动 Newton-Raphson 求 azimuth time
  3. 计算 incident angle → area normalization factor (ANF)
  4. 将 SLC 重采样到 GeoGrid
  5. 应用 RTC 校正
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from osgeo import gdal

from common_processing import (
    construct_orbit,
    gps_to_datetime,
    resolve_manifest_data_path,
)
from geo2rdr_compat import geo2rdr_compat


def manual_geo2rdr(
    target_llh,
    orbit,
    sensing_start_rel,
    prf,
    starting_range,
    range_pixel_spacing,
    wavelength,
    threshold=1e-9,
    max_iter=50,
    initial_guess_rel=None,
):
    """
    手动 Newton-Raphson geo2rdr，对任意 LLH 目标求 azimuth time 和 slant range。
    target_llh: [lon, lat, height] degrees/degrees/meters
    返回: (az_time_relative, slant_range_m)
    """
    a, e2 = 6378137.0, 0.00669437999014
    lon_r, lat_r, h = (
        np.radians(target_llh[0]),
        np.radians(target_llh[1]),
        target_llh[2],
    )
    n = a / np.sqrt(1 - e2 * np.sin(lat_r) ** 2)
    x_t = (n + h) * np.cos(lat_r) * np.cos(lon_r)
    y_t = (n + h) * np.cos(lat_r) * np.sin(lon_r)
    z_t = (n * (1 - e2) + h) * np.sin(lat_r)
    target = np.array([x_t, y_t, z_t])

    t = sensing_start_rel if initial_guess_rel is None else initial_guess_rel

    for i in range(max_iter):
        pos_i, vel_i = orbit.interpolate(t)
        rel = target - pos_i
        f = np.dot(vel_i, rel)
        eps_t = 0.01
        _, vel1 = orbit.interpolate(t + eps_t)
        acc = (vel1 - vel_i) / eps_t
        df = np.dot(acc, rel) - np.dot(vel_i, vel_i)
        dt = np.clip(f / df, -1.0, 1.0)
        if abs(dt) < threshold:
            r = np.linalg.norm(target - pos_i)
            return t, r, i + 1
        t = t - dt
    return t, np.linalg.norm(target - pos_i), max_iter


def compute_area_factor(incidence_deg):
    """根据入射角计算 RTC area normalization factor 辅助量。"""
    inc_rad = np.radians(incidence_deg)
    return np.sin(inc_rad)


def run_manual_rtc(
    manifest_path,
    dem_path,
    output_path,
    x_start,
    y_start,
    x_spacing,
    y_spacing,
    width,
    length,
    epsg=4326,
    block_rows=500,
):
    """
    手动 RTC main loop - 分块处理避免内存溢出。
    """
    import isce3.core
    import isce3.product
    import isce3.io

    print(f"[MANUAL-RTC] Loading manifest: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    metadata_dir = Path(manifest_path).parent / "metadata"

    with open(metadata_dir / "orbit.json", encoding="utf-8") as f:
        orbit_data = json.load(f)
    with open(metadata_dir / "radargrid.json", encoding="utf-8") as f:
        radargrid_data = json.load(f)
    with open(metadata_dir / "acquisition.json", encoding="utf-8") as f:
        acquisition_data = json.load(f)

    slc_path = resolve_manifest_data_path(manifest_path, manifest["slc"]["path"])
    print(f"[MANUAL-RTC] SLC: {slc_path}")

    print("[MANUAL-RTC] Constructing orbit (LEGENDRE)...")
    orbit = construct_orbit(orbit_data, "Legendre")

    c = isce3.core.speed_of_light
    prf = acquisition_data["prf"]
    wavelength = c / acquisition_data["centerFrequency"]
    range_pixel_spacing = radargrid_data["columnSpacing"]
    starting_range = c * radargrid_data["rangeTimeFirstPixel"] / 2.0

    ref_dt = gps_to_datetime(orbit_data["header"]["firstStateTimeUTC"])
    gps_epoch = datetime(1980, 1, 6, tzinfo=timezone.utc)
    orbit_ref_gps = (ref_dt - gps_epoch).total_seconds()
    sensing_start_rel = acquisition_data["startGPSTime"] - orbit_ref_gps
    sensing_mid_rel = (
        0.5 * (acquisition_data["startGPSTime"] + acquisition_data["stopGPSTime"])
        - orbit_ref_gps
    )

    look_raw = acquisition_data.get("lookDirection", "RIGHT").strip().upper()
    look_side = (
        isce3.core.LookSide.Left if look_raw == "LEFT" else isce3.core.LookSide.Right
    )

    radar_grid = isce3.product.RadarGridParameters(
        sensing_start=sensing_start_rel,
        wavelength=wavelength,
        prf=prf,
        starting_range=starting_range,
        range_pixel_spacing=range_pixel_spacing,
        lookside=look_side,
        length=radargrid_data["numberOfRows"],
        width=radargrid_data["numberOfColumns"],
        ref_epoch=isce3.core.DateTime(ref_dt),
    )

    print(
        f"[MANUAL-RTC] RadarGrid: {radargrid_data['numberOfColumns']} x {radargrid_data['numberOfRows']}"
    )
    print(
        f"[MANUAL-RTC] Starting range: {starting_range / 1000:.3f} km, look: {look_raw}"
    )
    print(
        f"[MANUAL-RTC] GeoGrid: {width} x {length}, spacing={x_spacing:.6f} x {y_spacing:.6f}"
    )

    isce3.product.GeoGridParameters(
        x_start, y_start, x_spacing, y_spacing, width, length, epsg
    )

    dem_ds = gdal.Open(dem_path)
    dem_transform = dem_ds.GetGeoTransform()
    dem_cols = dem_ds.RasterXSize
    dem_rows = dem_ds.RasterYSize

    def get_dem_height(lon, lat):
        x = int((lon - dem_transform[0]) / dem_transform[1])
        y = int((lat - dem_transform[3]) / dem_transform[5])
        if 0 <= x < dem_cols and 0 <= y < dem_rows:
            band = dem_ds.GetRasterBand(1)
            return band.ReadAsArray(x, y, 1, 1)[0, 0]
        return 0.0

    gdal_slc = gdal.Open(slc_path)
    slc_cols = gdal_slc.RasterXSize
    slc_rows = gdal_slc.RasterYSize
    slc_real = gdal_slc.GetRasterBand(1)
    slc_imag = gdal_slc.GetRasterBand(2)
    incidence_deg = float(acquisition_data["incidenceAngleCenter"])

    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        output_path, width, length, 1, gdal.GDT_Float32, ["COMPRESS=LZW", "TILED=YES"]
    )
    out_ds.SetGeoTransform([x_start, x_spacing, 0, y_start, 0, y_spacing])
    out_ds.SetProjection("EPSG:4326")
    out_band = out_ds.GetRasterBand(1)

    t0 = time.time()
    n_valid = 0
    n_total = 0

    for row_block in range(0, length, block_rows):
        block_h = min(block_rows, length - row_block)
        for col_block in range(0, width, block_rows):
            block_w = min(block_rows, width - col_block)
            out_block = np.zeros((block_h, block_w), dtype=np.float32)

            for r in range(block_h):
                geo_r = row_block + r
                for c_col in range(block_w):
                    geo_c = col_block + c_col
                    lon = x_start + geo_c * x_spacing
                    lat = y_start + geo_r * y_spacing
                    h_dem = get_dem_height(lon, lat)

                    azt_rel, slant_range, _ = geo2rdr_compat(
                        lon,
                        lat,
                        h_dem,
                        orbit,
                        wavelength,
                        look_raw.lower(),
                        sensing_mid_rel,
                    )

                    col = (slant_range - starting_range) / range_pixel_spacing
                    row = (azt_rel - sensing_start_rel) * prf

                    c0, r0 = int(col), int(row)
                    if 1 <= c0 < slc_cols - 1 and 1 <= r0 < slc_rows - 1:
                        fr, fc = row - r0, col - c0

                        def amp(c, r):
                            re = float(slc_real.ReadAsArray(c, r, 1, 1)[0, 0])
                            im = float(slc_imag.ReadAsArray(c, r, 1, 1)[0, 0])
                            if np.isnan(re) or np.isnan(im):
                                return np.nan
                            v = re * re + im * im
                            return np.sqrt(v) if v > 0 else 0.0

                        s00 = amp(c0, r0)
                        s01 = amp(c0 + 1, r0)
                        s10 = amp(c0, r0 + 1)
                        s11 = amp(c0 + 1, r0 + 1)
                        slc_val = (
                            s00 * (1 - fc) * (1 - fr)
                            + s01 * fc * (1 - fr)
                            + s10 * (1 - fc) * fr
                            + s11 * fc * fr
                        )
                        if np.isfinite(slc_val):
                            area_factor = compute_area_factor(incidence_deg)
                            rtc_val = slc_val / area_factor if area_factor > 0 else 0.0
                            out_block[r, c_col] = rtc_val
                            n_valid += 1
                    n_total += 1

            out_band.WriteArray(out_block, col_block, row_block)

        elapsed = time.time() - t0
        pct = (row_block + block_h) / length * 100
        print(
            f"[MANUAL-RTC] {pct:.1f}% done, {elapsed:.0f}s elapsed, valid={n_valid}/{n_total}"
        )

    out_band.FlushCache()
    out_ds.FlushCache()
    out_band = None
    out_ds = None
    dem_ds = None
    gdal_slc = None
    print(
        f"[MANUAL-RTC] Done! Total: {n_valid}/{n_total} valid pixels, {time.time() - t0:.0f}s"
    )
    return output_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Manual RTC without isce3 geo2rdr")
    parser.add_argument("manifest")
    parser.add_argument("dem")
    parser.add_argument("output")
    parser.add_argument("--x0", type=float, required=True, help="GeoGrid x start (lon)")
    parser.add_argument("--y0", type=float, required=True, help="GeoGrid y start (lat)")
    parser.add_argument("--xs", type=float, required=True, help="x spacing (deg)")
    parser.add_argument("--ys", type=float, required=True, help="y spacing (deg)")
    parser.add_argument("--w", type=int, required=True, help="GeoGrid width")
    parser.add_argument("--l", type=int, required=True, help="GeoGrid length")
    parser.add_argument("--epsg", type=int, default=4326, help="GeoGrid EPSG")
    parser.add_argument("--block", type=int, default=500, help="Block size")
    args = parser.parse_args()

    run_manual_rtc(
        args.manifest,
        args.dem,
        args.output,
        args.x0,
        args.y0,
        args.xs,
        args.ys,
        args.w,
        args.l,
        epsg=args.epsg,
        block_rows=args.block,
    )


if __name__ == "__main__":
    main()
