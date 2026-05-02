import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from osgeo import gdal


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


class InSARRegistrationVsiTests(unittest.TestCase):
    def test_resample_input_path_preserves_vsitar_absolute_archive_separator(self):
        import insar_registration

        path = (
            "/vsitar//temp/LT1B_MONO_KSC_STRIP1_018638_E95.2_N29.4_20250802_"
            "SLC_HH_L1A_0000825861.tar.gz/"
            "LT1B_MONO_KSC_STRIP1_018638_E95.2_N29.4_20250802_SLC_HH_L1A_0000825861.tiff"
        )

        self.assertEqual(insar_registration._resample_reader_path(path), path)

    def test_resample_input_path_normalizes_regular_files(self):
        import insar_registration

        self.assertEqual(
            insar_registration._resample_reader_path("relative/file.tif"),
            str(Path("relative/file.tif")),
        )

    def test_ampcor_materialize_combines_two_band_iq_into_one_complex_band(self):
        import insar_registration

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "iq.tif"
            dst = Path(tmp) / "ampcor.slc"
            driver = gdal.GetDriverByName("GTiff")
            ds = driver.Create(str(src), 3, 2, 2, gdal.GDT_Int16)
            ds.GetRasterBand(1).WriteArray(
                np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16)
            )
            ds.GetRasterBand(2).WriteArray(
                np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int16)
            )
            ds = None

            out = insar_registration._materialize_ampcor_input(str(src), dst)
            out_ds = gdal.Open(out)
            self.assertIsNotNone(out_ds)
            self.assertEqual(out_ds.RasterCount, 1)
            self.assertEqual(out_ds.GetRasterBand(1).DataType, gdal.GDT_CFloat32)
            arr = out_ds.GetRasterBand(1).ReadAsArray()
            np.testing.assert_array_equal(
                arr,
                np.array(
                    [[1 + 10j, 2 + 20j, 3 + 30j], [4 + 40j, 5 + 50j, 6 + 60j]],
                    dtype=np.complex64,
                ),
            )


if __name__ == "__main__":
    unittest.main()
