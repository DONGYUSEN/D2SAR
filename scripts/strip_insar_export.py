from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py

from common_processing import (
    compute_utm_output_shape,
    write_geocoded_geotiff,
    write_geocoded_png,
    write_wrapped_phase_geotiff,
    write_wrapped_phase_png,
)


SUPPORTED_DATASETS = {
    "avg_amplitude",
    "interferogram",
    "coherence",
    "unwrapped_phase",
    "los_displacement",
}


def _resolve_target_size(
    input_h5: str,
    resolution_meters: float | None,
    width: int | None,
    height: int | None,
) -> tuple[int, int | None]:
    if resolution_meters is not None and (width is not None or height is not None):
        raise ValueError("--resolution and --width/--height are mutually exclusive")
    if (width is None) != (height is None):
        raise ValueError("--width and --height must be provided together")
    if resolution_meters is not None:
        return compute_utm_output_shape(input_h5, resolution_meters)
    if width is not None and height is not None:
        return int(width), int(height)
    return 2048, None


def _validate_input(input_h5: str, dataset_name: str) -> None:
    input_path = Path(input_h5)
    if not input_path.exists():
        raise FileNotFoundError(f"Input HDF5 not found: {input_h5}")
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset '{dataset_name}'")
    with h5py.File(input_path, "r") as f:
        for required in ("utm_x", "utm_y"):
            if required not in f:
                raise KeyError(f"Required dataset missing: {required}")
        if dataset_name not in f:
            raise KeyError(f"Required dataset missing: {dataset_name}")


def _output_paths(output_dir: str | Path, dataset_name: str) -> tuple[str, str]:
    output_dir = Path(output_dir)
    tif_name = f"{dataset_name}_utm_geocoded.tif"
    png_name = f"{dataset_name}_utm_geocoded.png"
    if dataset_name == "interferogram":
        png_name = "interferogram_wrapped_phase_utm_geocoded.png"
    return str(output_dir / tif_name), str(output_dir / png_name)


def export_dataset(
    input_h5: str,
    output_dir: str,
    dataset_name: str,
    output_format: str = "both",
    resolution_meters: float | None = None,
    width: int | None = None,
    height: int | None = None,
    block_rows: int = 64,
) -> dict:
    if output_format not in {"tif", "png", "both"}:
        raise ValueError(f"Unsupported format '{output_format}'")

    _validate_input(input_h5, dataset_name)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_width, target_height = _resolve_target_size(
        input_h5, resolution_meters, width, height
    )
    output_tif, output_png = _output_paths(output_dir, dataset_name)

    result = {
        "input_h5": str(Path(input_h5)),
        "dataset": dataset_name,
        "format": output_format,
        "target_width": target_width,
        "target_height": target_height,
    }

    if output_format in {"tif", "both"}:
        if dataset_name == "interferogram":
            write_wrapped_phase_geotiff(
                input_h5,
                output_tif,
                dataset_name=dataset_name,
                target_width=target_width,
                target_height=target_height,
                block_rows=block_rows,
            )
        else:
            write_geocoded_geotiff(
                input_h5,
                output_tif,
                dataset_name=dataset_name,
                target_width=target_width,
                target_height=target_height,
                block_rows=block_rows,
            )
        result["output_tif"] = output_tif

    if output_format in {"png", "both"}:
        if dataset_name == "interferogram":
            write_wrapped_phase_png(
                input_h5,
                output_png,
                dataset_name=dataset_name,
                target_width=target_width,
                target_height=target_height,
                block_rows=block_rows,
            )
        else:
            write_geocoded_png(
                input_h5,
                output_png,
                dataset_name=dataset_name,
                target_width=target_width,
                target_height=target_height,
                block_rows=block_rows,
            )
        result["output_png"] = output_png

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export GeoTIFF/PNG products from strip_insar HDF5"
    )
    parser.add_argument("input_h5", help="Input interferogram_fullres.h5")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--dataset", required=True, choices=sorted(SUPPORTED_DATASETS))
    parser.add_argument("--resolution", type=float, help="Output resolution in meters")
    parser.add_argument("--width", type=int, help="Output width in pixels")
    parser.add_argument("--height", type=int, help="Output height in pixels")
    parser.add_argument(
        "--format",
        choices=["tif", "png", "both"],
        default="both",
        help="Export format",
    )
    parser.add_argument("--block-rows", type=int, default=64)
    args = parser.parse_args()

    result = export_dataset(
        args.input_h5,
        args.output_dir,
        dataset_name=args.dataset,
        output_format=args.format,
        resolution_meters=args.resolution,
        width=args.width,
        height=args.height,
        block_rows=args.block_rows,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
