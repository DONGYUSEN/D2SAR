from common_processing import write_rtc_hdf as write_tianyi_rtc_hdf


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Write Tianyi full-resolution amplitude result to HDF5"
    )
    parser.add_argument("slc")
    parser.add_argument("rtc_factor")
    parser.add_argument("output_h5")
    parser.add_argument("--block-rows", type=int, default=256)
    args = parser.parse_args()

    out = write_tianyi_rtc_hdf(
        args.slc,
        args.rtc_factor,
        args.output_h5,
        block_rows=args.block_rows,
    )
    print(out)
