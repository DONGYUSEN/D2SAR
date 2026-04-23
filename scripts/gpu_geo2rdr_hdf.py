def run_gpu_geo2rdr_hdf(*args, **kwargs):
    from diagnostics.gpu_geo2rdr_hdf import run_gpu_geo2rdr_hdf as _run_gpu_geo2rdr_hdf

    return _run_gpu_geo2rdr_hdf(*args, **kwargs)


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
