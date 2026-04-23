from common_processing import (
    append_utm_coordinates_hdf as append_tianyi_utm_hdf,
    point2epsg,
)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Append full-resolution UTM coordinate layers to Tianyi amplitude HDF"
    )
    parser.add_argument("output_h5")
    parser.add_argument("manifest")
    parser.add_argument("--block-rows", type=int, default=32)
    args = parser.parse_args()

    out = append_tianyi_utm_hdf(
        args.output_h5,
        args.manifest,
        block_rows=args.block_rows,
    )
    print(out)
