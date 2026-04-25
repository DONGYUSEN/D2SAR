from __future__ import annotations

import argparse
import colorsys
from pathlib import Path

import numpy as np
from osgeo import gdal
from PIL import Image


def _read_complex(path: Path, stride: int) -> np.ndarray:
    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"failed to open raster: {path}")
    try:
        arr = ds.GetRasterBand(1).ReadAsArray().astype(np.complex64)
    finally:
        ds = None
    return arr[::stride, ::stride]


def _scale_value_from_avg_amplitude(avg_amplitude: np.ndarray) -> np.ndarray:
    value = np.full(avg_amplitude.shape, 0.15, dtype=np.float64)
    valid = np.isfinite(avg_amplitude) & (avg_amplitude > 0)
    if not np.any(valid):
        return value
    amp_db = 20.0 * np.log10(avg_amplitude[valid])
    p2 = float(np.percentile(amp_db, 2))
    p98 = float(np.percentile(amp_db, 98))
    scaled = np.clip((amp_db - p2) / (p98 - p2 + 1.0e-9), 0.0, 1.0)
    value[valid] = np.clip(scaled, 0.15, 1.0)
    return value


def export_preview(pair_dir: Path, output_path: Path, stride: int) -> Path:
    ifg = np.load(pair_dir / "work" / "p2_crossmul" / "interferogram.npy").astype(np.complex64)
    ifg = ifg[::stride, ::stride]
    master = _read_complex(pair_dir / "work" / "p2_crossmul" / "cuda_inputs" / "master.slc", stride)
    slave = _read_complex(pair_dir / "work" / "p2_crossmul" / "cuda_inputs" / "slave.slc", stride)

    phase = np.angle(ifg)
    avg_amplitude = 0.5 * (
        np.abs(master).astype(np.float32) + np.abs(slave).astype(np.float32)
    )
    value = _scale_value_from_avg_amplitude(avg_amplitude)

    rgb = np.zeros((*phase.shape, 3), dtype=np.uint8)
    valid = np.isfinite(phase)
    if np.any(valid):
        hue = np.mod((phase[valid] + np.pi) / (2.0 * np.pi), 1.0)
        colors = np.array(
            [
                colorsys.hsv_to_rgb(float(h), 1.0, float(v))
                for h, v in zip(hue, value[valid], strict=False)
            ]
        )
        rgb[valid] = np.clip(colors * 255.0, 0.0, 255.0).astype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export radar wrapped-phase preview with avg amplitude as value")
    parser.add_argument("pair_dir")
    parser.add_argument("--output")
    parser.add_argument("--stride", type=int, default=4)
    args = parser.parse_args()

    pair_dir = Path(args.pair_dir)
    output = Path(args.output) if args.output else pair_dir / f"wrapped_phase_radar_avgamp_stride{args.stride}.png"
    result = export_preview(pair_dir=pair_dir, output_path=output, stride=max(int(args.stride), 1))
    print(result)


if __name__ == "__main__":
    main()
