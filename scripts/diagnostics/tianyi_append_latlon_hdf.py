from common_processing import append_topo_coordinates_hdf as append_tianyi_latlon_hdf


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Append per-pixel lat/lon/height to Tianyi HDF"
    )
    parser.add_argument("manifest")
    parser.add_argument("dem")
    parser.add_argument("output_h5")
    parser.add_argument("--block-rows", type=int, default=256)
    args = parser.parse_args()

    out = append_tianyi_latlon_hdf(
        args.manifest,
        args.dem,
        args.output_h5,
        block_rows=args.block_rows,
    )
    print(out)
