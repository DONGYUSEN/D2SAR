import json
from pathlib import Path

import numpy as np
from osgeo import gdal

from common_processing import resolve_manifest_data_path
from rtc_pipeline import (
    _construct_doppler_lut2d,
    _construct_orbit,
    _construct_radar_grid,
)


def compute_tianyi_rtc_fallback(
    manifest_path: str,
    dem_path: str,
    output_dir: str,
    block_rows: int = 512,
) -> str:
    import isce3.core
    import isce3.geometry
    import isce3.io

    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    metadata_dir = manifest_path.parent / "metadata"
    with open(metadata_dir / "orbit.json", encoding="utf-8") as f:
        orbit_data = json.load(f)
    with open(metadata_dir / "radargrid.json", encoding="utf-8") as f:
        radargrid_data = json.load(f)
    with open(metadata_dir / "doppler.json", encoding="utf-8") as f:
        doppler_data = json.load(f)
    with open(metadata_dir / "acquisition.json", encoding="utf-8") as f:
        acquisition_data = json.load(f)

    slc_path = resolve_manifest_data_path(manifest_path, manifest["slc"]["path"])
    rtc_factor_path = output_dir / "rtc_factor.tif"
    output_path = output_dir / "amplitude_fallback.tif"

    orbit = _construct_orbit(orbit_data, "Hermite")
    doppler_lut = isce3.core.LUT2d()
    radar_grid = _construct_radar_grid(radargrid_data, acquisition_data, orbit_data)
    dem_raster = isce3.io.Raster(str(dem_path))

    rtc_factor_raster = isce3.io.Raster(
        str(rtc_factor_path),
        radar_grid.width,
        radar_grid.length,
        1,
        6,
        "GTiff",
    )
    isce3.geometry.compute_rtc(
        radar_grid,
        orbit,
        doppler_lut,
        dem_raster,
        rtc_factor_raster,
    )
    dem_raster.close_dataset()
    rtc_factor_raster.close_dataset()

    slc_ds = gdal.Open(slc_path)
    if slc_ds is None:
        raise RuntimeError(f"failed to open SLC: {slc_path}")
    rtc_ds = gdal.Open(str(rtc_factor_path))
    if rtc_ds is None:
        raise RuntimeError(f"failed to open RTC factor: {rtc_factor_path}")

    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        str(output_path),
        slc_ds.RasterXSize,
        slc_ds.RasterYSize,
        1,
        gdal.GDT_Float32,
    )
    out_band = out_ds.GetRasterBand(1)
    slc_band = slc_ds.GetRasterBand(1)
    rtc_band = rtc_ds.GetRasterBand(1)

    width = slc_ds.RasterXSize
    length = slc_ds.RasterYSize
    for row0 in range(0, length, block_rows):
        rows = min(block_rows, length - row0)
        slc_block = slc_band.ReadAsArray(0, row0, width, rows)
        rtc_band.ReadAsArray(0, row0, width, rows).astype(np.float32)

        amplitude = np.sqrt(
            slc_block.real.astype(np.float32) ** 2
            + slc_block.imag.astype(np.float32) ** 2
        ).astype(np.float32)
        out_band.WriteArray(amplitude, 0, row0)

    out_band.FlushCache()
    out_ds.FlushCache()
    out_band = None
    out_ds = None
    rtc_ds = None
    slc_ds = None
    return str(output_path)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Tianyi amplitude fallback using compute_rtc + manual SLC amplitude export"
    )
    parser.add_argument("manifest")
    parser.add_argument("dem")
    parser.add_argument("output_dir")
    parser.add_argument("--block-rows", type=int, default=512)
    args = parser.parse_args()

    out = compute_tianyi_rtc_fallback(
        args.manifest,
        args.dem,
        args.output_dir,
        block_rows=args.block_rows,
    )
    print(out)


if __name__ == "__main__":
    main()
