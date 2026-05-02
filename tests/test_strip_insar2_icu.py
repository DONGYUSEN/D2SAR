import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from osgeo import gdal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


class StripInsar2IcuTests(unittest.TestCase):
    def test_icu_default_profile_is_conservative(self) -> None:
        import strip_insar2

        defaults = strip_insar2.ICU_DEFAULTS

        self.assertFalse(defaults["use_phase_gradient_neutron"])
        self.assertEqual(defaults["trees_number"], 7)
        self.assertEqual(defaults["max_branch_length"], 64)
        self.assertAlmostEqual(defaults["initial_correlation_threshold"], 0.1)
        self.assertAlmostEqual(defaults["max_correlation_threshold"], 0.9)
        self.assertAlmostEqual(defaults["correlation_threshold_increments"], 0.1)
        self.assertAlmostEqual(defaults["min_tile_area"], 0.003125)
        self.assertEqual(defaults["bootstrap_lines"], 16)
        self.assertEqual(defaults["min_overlap_area"], 16)
        self.assertAlmostEqual(defaults["phase_variance_threshold"], 8.0)

    def test_icu_output_masks_zero_connected_component_labels(self) -> None:
        import strip_insar2

        class FakeRaster:
            def __init__(self, path, *args, **kwargs):
                self.path = str(path)

        class FakeICU:
            def unwrap(self, unw_raster, ccl_raster, *args, **kwargs):
                unw_ds = gdal.Open(unw_raster.path, gdal.GA_Update)
                cc_ds = gdal.Open(ccl_raster.path, gdal.GA_Update)
                unw_ds.GetRasterBand(1).WriteArray(
                    np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
                )
                cc_ds.GetRasterBand(1).WriteArray(
                    np.array([[1, 0], [2, 0]], dtype=np.uint8)
                )
                unw_ds = None
                cc_ds = None

        fake_isce3 = types.SimpleNamespace(
            unwrap=types.SimpleNamespace(ICU=FakeICU),
            io=types.SimpleNamespace(Raster=FakeRaster),
        )

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(sys.modules, {"isce3": fake_isce3}):
                result = strip_insar2._unwrap_with_icu(
                    np.ones((2, 2), dtype=np.complex64),
                    np.ones((2, 2), dtype=np.float32),
                    Path(tmp),
                )

        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(float(result[0, 0]), 10.0)
        self.assertEqual(float(result[1, 0]), 30.0)
        self.assertTrue(np.isnan(result[0, 1]))
        self.assertTrue(np.isnan(result[1, 1]))

    def test_run_unwrap_stage_uses_dolphin_without_logger_name_error(self) -> None:
        import strip_insar2

        with tempfile.TemporaryDirectory() as tmp:
            pair_dir = Path(tmp)
            context = strip_insar2.PairContext(
                master_manifest_path=pair_dir / "master.json",
                slave_manifest_path=pair_dir / "slave.json",
                master_manifest={},
                slave_manifest={},
                master_orbit_data={},
                slave_orbit_data={},
                master_acq_data={},
                slave_acq_data={},
                master_rg_data={},
                slave_rg_data={},
                master_dop_data={},
                slave_dop_data={},
                output_root=pair_dir,
                pair_name="pair",
                pair_dir=pair_dir,
                output_paths={},
                resolved_dem=None,
                orbit_interp="Hermite",
                wavelength=0.24,
            )
            p2_dir = pair_dir / "p2"
            p2_dir.mkdir()
            interferogram_path = p2_dir / "interferogram.npy"
            coherence_path = p2_dir / "coherence.npy"
            np.save(interferogram_path, np.ones((2, 2), dtype=np.complex64))
            np.save(coherence_path, np.ones((2, 2), dtype=np.float32))

            def fake_load_stage_record(output_dir, stage):
                if stage == "p2":
                    return {
                        "success": True,
                        "output_files": {
                            "interferogram": str(interferogram_path),
                            "coherence": str(coherence_path),
                        },
                    }
                return None

            with mock.patch.object(strip_insar2, "load_stage_record", side_effect=fake_load_stage_record), \
                mock.patch.object(strip_insar2, "success_marker_path", return_value=pair_dir / "missing.ok"), \
                mock.patch.object(strip_insar2, "mark_stage_success"), \
                mock.patch.object(strip_insar2, "write_stage_record"), \
                mock.patch.object(strip_insar2, "_unwrap_with_dolphin", return_value=(np.zeros((2, 2), dtype=np.float32), None)):
                outputs, backend, fallback = strip_insar2.run_unwrap_stage(
                    context,
                    unwrap_method="snaphu",
                    block_rows=10,
                    range_looks=1,
                    azimuth_looks=1,
                )

        self.assertEqual(backend, "cpu")
        self.assertIsNone(fallback)
        self.assertIn("unwrapped_phase", outputs)


if __name__ == "__main__":
    unittest.main()
