import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


class StripInsar2AvgAmplitudePngTests(unittest.TestCase):
    def _write_input_h5(self, input_h5: Path) -> None:
        values = np.concatenate(
            [
                np.zeros(50, dtype=np.float32),
                np.arange(1, 101, dtype=np.float32),
            ]
        ).reshape(1, 150)
        with h5py.File(input_h5, "w") as f:
            f.create_dataset("avg_amplitude", data=values)
            f.create_dataset(
                "utm_x",
                data=np.linspace(0.0, 149.0, 150, dtype=np.float32).reshape(1, 150),
            )
            f.create_dataset("utm_y", data=np.zeros((1, 150), dtype=np.float32))
            f.attrs["utm_epsg"] = 32646

    def _assert_mid_gray(self, output_png: Path) -> None:
        img = np.asarray(Image.open(output_png))
        self.assertEqual(img.shape, (1, 1))
        self.assertGreaterEqual(int(img[0, 0]), 120)
        self.assertLessEqual(int(img[0, 0]), 135)

    def test_avg_amplitude_png_stretches_before_utm_accumulation(self) -> None:
        import strip_insar2

        with tempfile.TemporaryDirectory() as tmp:
            input_h5 = Path(tmp) / "product.h5"
            output_png = Path(tmp) / "avg.png"
            self._write_input_h5(input_h5)

            strip_insar2.write_geocoded_png(
                str(input_h5),
                str(output_png),
                "avg_amplitude",
                target_width=1,
                target_height=1,
                block_rows=1,
            )

            self._assert_mid_gray(output_png)

    def test_common_processing_avg_amplitude_png_uses_same_pre_utm_stretch(self) -> None:
        import common_processing

        with tempfile.TemporaryDirectory() as tmp:
            input_h5 = Path(tmp) / "product.h5"
            output_png = Path(tmp) / "avg.png"
            self._write_input_h5(input_h5)

            common_processing.write_geocoded_png(
                str(input_h5),
                str(output_png),
                dataset_name="avg_amplitude",
                target_width=1,
                target_height=1,
                block_rows=1,
            )

            self._assert_mid_gray(output_png)


if __name__ == "__main__":
    unittest.main()
