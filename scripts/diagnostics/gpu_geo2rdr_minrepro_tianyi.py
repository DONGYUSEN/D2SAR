import json
from pathlib import Path

from osgeo import gdal, osr

from common_processing import construct_orbit, construct_radar_grid


def run_minrepro(
    metadata_dir: str,
    dem_path: str,
    outdir: str,
    lines_per_block: int = 128,
) -> None:
    import isce3.core
    import isce3.cuda.geometry
    import isce3.io

    metadata_dir = Path(metadata_dir)
    outdir = Path(outdir)

    with open(metadata_dir / "orbit.json", encoding="utf-8") as f:
        orbit_data = json.load(f)
    with open(metadata_dir / "radargrid.json", encoding="utf-8") as f:
        radargrid_data = json.load(f)
    with open(metadata_dir / "acquisition.json", encoding="utf-8") as f:
        acquisition_data = json.load(f)

    orbit = construct_orbit(orbit_data, "Hermite")
    radar_grid = construct_radar_grid(
        radargrid_data, acquisition_data, orbit_data
    ).offset_and_resize(0, 0, 256, 256)
    dem = isce3.io.Raster(str(dem_path))
    doppler = isce3.core.LUT2d()
    ellipsoid = isce3.core.Ellipsoid()

    outdir.mkdir(parents=True, exist_ok=True)

    rasters = {
        name: isce3.io.Raster(
            str(outdir / f"{name}.tif"),
            radar_grid.width,
            radar_grid.length,
            1,
            6,
            "GTiff",
        )
        for name in [
            "x",
            "y",
            "height",
            "inc",
            "hdg",
            "localInc",
            "localPsi",
            "simamp",
            "los_east",
            "los_north",
        ]
    }

    topo = isce3.cuda.geometry.Rdr2Geo(
        radar_grid, orbit, ellipsoid, doppler, epsg_out=4326, lines_per_block=128
    )
    print("before topo", flush=True)
    topo.topo(
        dem,
        rasters["x"],
        rasters["y"],
        rasters["height"],
        rasters["inc"],
        rasters["hdg"],
        rasters["localInc"],
        rasters["localPsi"],
        rasters["simamp"],
        None,
        rasters["los_east"],
        rasters["los_north"],
    )
    print("after topo", flush=True)

    stack = outdir / "topo_xyz.tif"
    x = gdal.Open(str(outdir / "x.tif")).ReadAsArray()
    y = gdal.Open(str(outdir / "y.tif")).ReadAsArray()
    z = gdal.Open(str(outdir / "height.tif")).ReadAsArray()

    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(
        str(stack), radar_grid.width, radar_grid.length, 3, gdal.GDT_Float64
    )
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())
    for i, arr in enumerate([x, y, z], start=1):
        ds.GetRasterBand(i).WriteArray(arr)
    ds.FlushCache()
    ds = None
    print("after topo stack", flush=True)

    topo_raster = isce3.io.Raster(str(stack))
    print("before gpu geo2rdr", flush=True)
    geo2rdr = isce3.cuda.geometry.Geo2Rdr(
        radar_grid, orbit, ellipsoid, doppler, lines_per_block=lines_per_block
    )
    geo2rdr.geo2rdr(topo_raster, str(outdir))
    print("after gpu geo2rdr", flush=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="GPU geo2rdr Tianyi minimal repro")
    parser.add_argument(
        "metadata_dir", help="Directory containing orbit/radargrid/acquisition JSON"
    )
    parser.add_argument("dem", help="DEM raster path")
    parser.add_argument("output_dir", help="Output directory for repro rasters")
    parser.add_argument("--lines-per-block", type=int, default=128)
    args = parser.parse_args()

    run_minrepro(
        args.metadata_dir,
        args.dem,
        args.output_dir,
        lines_per_block=args.lines_per_block,
    )


if __name__ == "__main__":
    main()
