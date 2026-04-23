import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
from PIL import Image
from osgeo import gdal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common_processing import write_wrapped_phase_geotiff, write_wrapped_phase_png
from strip_insar_export import export_dataset


class StripInsarExportTests(unittest.TestCase):
    def _make_h5(self, path: Path) -> Path:
        with h5py.File(path, "w") as f:
            f.attrs["utm_epsg"] = 32648
            x = np.array([[500000.0, 500010.0], [500000.0, 500010.0]], dtype=np.float32)
            y = np.array(
                [[3300000.0, 3300000.0], [3299990.0, 3299990.0]], dtype=np.float32
            )
            f.create_dataset("utm_x", data=x)
            f.create_dataset("utm_y", data=y)
            f.create_dataset(
                "avg_amplitude",
                data=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            )
            f.create_dataset(
                "coherence", data=np.array([[0.1, 0.5], [0.8, 1.0]], dtype=np.float32)
            )
            f.create_dataset(
                "unwrapped_phase",
                data=np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32),
            )
            f.create_dataset(
                "los_displacement",
                data=np.array([[0.0, 0.2], [0.4, 0.6]], dtype=np.float32),
            )
            f.create_dataset(
                "interferogram",
                data=np.array(
                    [
                        [1.0 + 0.0j, 0.0 + 1.0j],
                        [-1.0 + 0.0j, 0.0 - 1.0j],
                    ],
                    dtype=np.complex64,
                ),
            )
        return path

    def test_write_wrapped_phase_png_creates_rgb_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = self._make_h5(Path(tmpdir) / "insar.h5")
            png_path = Path(tmpdir) / "wrapped.png"

            write_wrapped_phase_png(
                str(h5_path),
                str(png_path),
                dataset_name="interferogram",
                target_width=2,
                target_height=2,
            )

            self.assertTrue(png_path.exists())
            img = Image.open(png_path)
            self.assertEqual(img.mode, "RGB")
            self.assertEqual(img.size, (2, 2))
            pixels = [tuple(px) for px in np.array(img).reshape(-1, 3)]
            self.assertGreater(len(set(pixels)), 1)

    def test_write_wrapped_phase_geotiff_prefers_strongest_sample_per_cell(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = Path(tmpdir) / "insar_overlap.h5"
            with h5py.File(h5_path, "w") as f:
                f.attrs["utm_epsg"] = 32648
                f.create_dataset(
                    "utm_x",
                    data=np.array([[500000.0, 500000.0]], dtype=np.float32),
                )
                f.create_dataset(
                    "utm_y",
                    data=np.array([[3300000.0, 3300000.0]], dtype=np.float32),
                )
                f.create_dataset(
                    "interferogram",
                    data=np.array([[10.0 + 0.0j, -9.0 + 0.0j]], dtype=np.complex64),
                )

            tif_path = Path(tmpdir) / "wrapped_overlap.tif"
            write_wrapped_phase_geotiff(
                str(h5_path),
                str(tif_path),
                dataset_name="interferogram",
                target_width=1,
                target_height=1,
            )

            ds = gdal.Open(str(tif_path))
            self.assertIsNotNone(ds)
            phase = float(ds.GetRasterBand(1).ReadAsArray()[0, 0])
            ds = None
            self.assertAlmostEqual(phase, 0.0, places=5)

    def test_export_dataset_raises_for_missing_h5_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = self._make_h5(Path(tmpdir) / "insar.h5")
            with h5py.File(h5_path, "a") as f:
                del f["avg_amplitude"]
            with self.assertRaises(KeyError):
                export_dataset(
                    str(h5_path),
                    str(Path(tmpdir) / "out"),
                    dataset_name="avg_amplitude",
                    output_format="both",
                    resolution_meters=10.0,
                )

    def test_export_dataset_uses_resolution_to_compute_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = self._make_h5(Path(tmpdir) / "insar.h5")
            out_dir = Path(tmpdir) / "out"
            out_dir.mkdir()

            with (
                mock.patch(
                    "strip_insar_export.compute_utm_output_shape",
                    return_value=(123, 45),
                ) as mock_shape,
                mock.patch("strip_insar_export.write_geocoded_geotiff") as mock_tif,
                mock.patch("strip_insar_export.write_geocoded_png") as mock_png,
            ):
                export_dataset(
                    str(h5_path),
                    str(out_dir),
                    dataset_name="avg_amplitude",
                    output_format="both",
                    resolution_meters=20.0,
                )

            mock_shape.assert_called_once_with(str(h5_path), 20.0)
            mock_tif.assert_called_once()
            mock_png.assert_called_once()
            self.assertEqual(mock_tif.call_args.kwargs["target_width"], 123)
            self.assertEqual(mock_tif.call_args.kwargs["target_height"], 45)
            self.assertEqual(mock_png.call_args.kwargs["dataset_name"], "avg_amplitude")


if __name__ == "__main__":
    unittest.main()
