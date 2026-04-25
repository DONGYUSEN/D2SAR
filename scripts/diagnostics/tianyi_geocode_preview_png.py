from common_processing import (
    accumulate_utm_grid as _accumulate_utm_grid,
    prepare_display_grid as _prepare_display_grid,
    write_geocoded_geotiff as write_tianyi_utm_geotiff,
    write_geocoded_png as write_tianyi_utm_preview_png,
)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate UTM-grid amplitude GeoTIFF and PNG preview from Tianyi HDF"
    )
    parser.add_argument("input_h5")
    parser.add_argument("output_tif")
    parser.add_argument("output_png")
    parser.add_argument("--target-width", type=int, default=2048)
    parser.add_argument("--block-rows", type=int, default=64)
    parser.add_argument("--confidence-interval-pct", type=float, default=98.0)
    args = parser.parse_args()

    tif = write_tianyi_utm_geotiff(
        args.input_h5,
        args.output_tif,
        target_width=args.target_width,
        block_rows=args.block_rows,
        confidence_interval_pct=args.confidence_interval_pct,
    )
    png = write_tianyi_utm_preview_png(
        args.input_h5,
        args.output_png,
        target_width=args.target_width,
        block_rows=args.block_rows,
        confidence_interval_pct=args.confidence_interval_pct,
    )
    print(tif)
    print(png)
