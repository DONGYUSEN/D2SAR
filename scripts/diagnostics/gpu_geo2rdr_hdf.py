import json
import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np

from common_processing import construct_orbit, construct_radar_grid


def _read_raster_array(path: str) -> np.ndarray:
    import isce3.io

    raster = isce3.io.Raster(path)
    arr = np.zeros((raster.length, raster.width), dtype=np.float64)
    raster.get_block(arr, 0, 0, raster.width, raster.length, 1)
    return arr


def run_gpu_geo2rdr_hdf(
    manifest_path: str,
    dem_path: str,
    output_h5: str,
    yoff: int = 0,
    xoff: int = 0,
    ysize: int | None = None,
    xsize: int | None = None,
    epsg_out: int = 4326,
    doppler_mode: str = "zero",
    lines_per_block: int = 512,
) -> str:
    import isce3.core
    import isce3.cuda.geometry
    import isce3.io

    manifest_path = Path(manifest_path)
    output_h5 = Path(output_h5)
    output_h5.parent.mkdir(parents=True, exist_ok=True)

    metadata_dir = manifest_path.parent / "metadata"
    with open(metadata_dir / "orbit.json", encoding="utf-8") as f:
        orbit_data = json.load(f)
    with open(metadata_dir / "radargrid.json", encoding="utf-8") as f:
        radargrid_data = json.load(f)
    with open(metadata_dir / "acquisition.json", encoding="utf-8") as f:
        acquisition_data = json.load(f)

    orbit_interp = (
        "Legendre" if acquisition_data.get("source") == "lutan" else "Hermite"
    )
    orbit = construct_orbit(orbit_data, orbit_interp)
    radar_grid = construct_radar_grid(radargrid_data, acquisition_data, orbit_data)

    if ysize is None:
        ysize = radar_grid.length - yoff
    if xsize is None:
        xsize = radar_grid.width - xoff
    radar_grid = radar_grid.offset_and_resize(yoff, xoff, ysize, xsize)

    if doppler_mode == "zero":
        doppler = isce3.core.LUT2d()
    else:
        raise ValueError(f"unsupported doppler_mode: {doppler_mode}")

    ellipsoid = isce3.core.Ellipsoid()
    dem_raster = isce3.io.Raster(str(dem_path))

    workdir = Path(tempfile.mkdtemp(prefix="gpu_geo2rdr_", dir="/tmp"))
    try:
        topo = isce3.cuda.geometry.Rdr2Geo(
            radar_grid,
            orbit,
            ellipsoid,
            doppler,
            epsg_out=epsg_out,
            lines_per_block=lines_per_block,
        )
        topo.topo(dem_raster, str(workdir))

        topo_raster = isce3.io.Raster(str(workdir / "topo.vrt"))
        geo2rdr = isce3.cuda.geometry.Geo2Rdr(
            radar_grid, orbit, ellipsoid, doppler, lines_per_block=lines_per_block
        )
        geo2rdr.geo2rdr(topo_raster, str(workdir))

        datasets = {
            "x": _read_raster_array(str(workdir / "x.rdr")),
            "y": _read_raster_array(str(workdir / "y.rdr")),
            "z": _read_raster_array(str(workdir / "z.rdr")),
            "incidence": _read_raster_array(str(workdir / "inc.rdr")),
            "heading": _read_raster_array(str(workdir / "hdg.rdr")),
            "range_offset": _read_raster_array(str(workdir / "range.off")),
            "azimuth_offset": _read_raster_array(str(workdir / "azimuth.off")),
        }

        with h5py.File(output_h5, "w") as f:
            meta = f.create_group("metadata")
            meta.attrs["sensor"] = acquisition_data.get("source", "unknown")
            meta.attrs["lookDirection"] = acquisition_data.get("lookDirection", "")
            meta.attrs["orbitInterp"] = orbit_interp
            meta.attrs["dopplerMode"] = doppler_mode
            meta.attrs["epsg_out"] = epsg_out
            meta.attrs["yoff"] = yoff
            meta.attrs["xoff"] = xoff
            meta.attrs["ysize"] = ysize
            meta.attrs["xsize"] = xsize
            meta.attrs["sensing_start"] = radar_grid.sensing_start
            meta.attrs["starting_range"] = radar_grid.starting_range
            meta.attrs["prf"] = radar_grid.prf
            meta.attrs["range_pixel_spacing"] = radar_grid.range_pixel_spacing

            geom = f.create_group("geometry")
            for name, arr in datasets.items():
                geom.create_dataset(name, data=arr, compression="gzip")
        return str(output_h5)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GPU Geo2Rdr + HDF5 output")
    parser.add_argument("manifest")
    parser.add_argument("dem")
    parser.add_argument("output_h5")
    parser.add_argument("--yoff", type=int, default=0)
    parser.add_argument("--xoff", type=int, default=0)
    parser.add_argument("--ysize", type=int)
    parser.add_argument("--xsize", type=int)
    parser.add_argument("--epsg-out", type=int, default=4326)
    parser.add_argument("--doppler-mode", choices=["zero"], default="zero")
    parser.add_argument("--lines-per-block", type=int, default=512)
    args = parser.parse_args()

    out = run_gpu_geo2rdr_hdf(
        args.manifest,
        args.dem,
        args.output_h5,
        yoff=args.yoff,
        xoff=args.xoff,
        ysize=args.ysize,
        xsize=args.xsize,
        epsg_out=args.epsg_out,
        doppler_mode=args.doppler_mode,
        lines_per_block=args.lines_per_block,
    )
    print(out)
