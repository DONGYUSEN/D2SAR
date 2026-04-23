import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


from insar_preprocess import build_preprocess_plan


class InSARPreprocessTests(unittest.TestCase):
    def test_build_preprocess_plan_materializes_normalized_slave_inputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stage_dir = root / "prep"
            stage_dir.mkdir()
            slc_path = root / "slave.tif"
            slc_path.write_bytes(b"fake-slc")
            acq_path = root / "metadata" / "acquisition.json"
            rg_path = root / "metadata" / "radargrid.json"
            dop_path = root / "metadata" / "doppler.json"
            acq_path.parent.mkdir()
            acq_path.write_text(json.dumps({"prf": 1020.0, "startGPSTime": 100.0}), encoding="utf-8")
            rg_path.write_text(
                json.dumps(
                    {
                        "numberOfRows": 120,
                        "numberOfColumns": 220,
                        "columnSpacing": 2.5,
                        "rangeTimeFirstPixel": 0.002,
                    }
                ),
                encoding="utf-8",
            )
            dop_path.write_text(json.dumps({"combinedDoppler": {"coefficients": [1.0]}}), encoding="utf-8")
            slave_manifest_path = root / "slave_manifest.json"
            slave_manifest_path.write_text(
                json.dumps(
                    {
                        "slc": {"path": "slave.tif"},
                        "metadata": {
                            "acquisition": "metadata/acquisition.json",
                            "radargrid": "metadata/radargrid.json",
                            "doppler": "metadata/doppler.json",
                        },
                    }
                ),
                encoding="utf-8",
            )

            precheck = {
                "requires_prep": True,
                "recommended_geometry_mode": "zero-doppler",
                "checks": {
                    "prf": {"severity": "warn"},
                    "doppler": {"severity": "warn"},
                    "radar_grid": {"severity": "warn"},
                },
            }

            with mock.patch("insar_preprocess._resample_slave_slc", side_effect=lambda src, dst, **kwargs: Path(dst).write_bytes(Path(src).read_bytes())) as mock_resample:
                plan, normalized_manifest_path = build_preprocess_plan(
                    precheck=precheck,
                    slave_manifest_path=slave_manifest_path,
                    stage_dir=stage_dir,
                    master_acquisition={"prf": 1000.0, "startGPSTime": 95.0},
                    master_radargrid={
                        "numberOfRows": 100,
                        "numberOfColumns": 200,
                        "columnSpacing": 2.0,
                        "rangeTimeFirstPixel": 0.001,
                    },
                )

            self.assertIn("normalize-slave-prf", plan["actions"])
            self.assertIn("resample-slave-to-master-grid", plan["actions"])
            self.assertTrue(mock_resample.called)
            self.assertEqual(mock_resample.call_args.kwargs["source_prf"], 1020.0)
            self.assertEqual(mock_resample.call_args.kwargs["target_prf"], 1000.0)
            self.assertEqual(mock_resample.call_args.kwargs["geometry_mode"], "zero-doppler")
            self.assertIn("doppler_coefficients", mock_resample.call_args.kwargs)

            normalized_manifest = json.loads(Path(normalized_manifest_path).read_text(encoding="utf-8"))
            self.assertNotEqual(normalized_manifest["slc"]["path"], "slave.tif")
            acq = json.loads((stage_dir / "normalized_acquisition.json").read_text(encoding="utf-8"))
            rg = json.loads((stage_dir / "normalized_radargrid.json").read_text(encoding="utf-8"))
            dop = json.loads((stage_dir / "normalized_doppler.json").read_text(encoding="utf-8"))
            self.assertEqual(acq["prf"], 1000.0)
            self.assertEqual(rg["numberOfRows"], 100)
            self.assertEqual(rg["numberOfColumns"], 200)
            self.assertEqual(dop["geometryMode"], "zero-doppler")
            prep_meta = normalized_manifest["processing"]["insar_preprocess"]
            self.assertIn("resamp_slc", prep_meta)
            self.assertEqual(prep_meta["resamp_slc"]["source_prf"], 1020.0)
            self.assertEqual(prep_meta["resamp_slc"]["target_prf"], 1000.0)
            self.assertEqual(prep_meta["resamp_slc"]["target_rows"], 100)
            self.assertEqual(prep_meta["resamp_slc"]["target_cols"], 200)

    def test_build_preprocess_plan_prefers_isce3_resamp_wrapper(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stage_dir = root / "prep"
            stage_dir.mkdir()
            slc_path = root / "slave.tif"
            slc_path.write_bytes(b"fake-slc")
            slave_manifest_path = root / "slave_manifest.json"
            slave_manifest_path.write_text(
                json.dumps(
                    {
                        "slc": {"path": "slave.tif"},
                        "metadata": {},
                    }
                ),
                encoding="utf-8",
            )
            precheck = {
                "requires_prep": True,
                "recommended_geometry_mode": "zero-doppler",
                "checks": {
                    "prf": {"severity": "warn"},
                    "doppler": {"severity": "warn"},
                    "radar_grid": {"severity": "warn"},
                },
            }

            with mock.patch(
                "insar_preprocess._resample_slave_slc_with_isce3",
                side_effect=lambda src, dst, **kwargs: Path(dst).write_bytes(Path(src).read_bytes()),
            ) as mock_isce3:
                build_preprocess_plan(
                    precheck=precheck,
                    slave_manifest_path=slave_manifest_path,
                    stage_dir=stage_dir,
                    master_acquisition={"prf": 1000.0, "startGPSTime": 95.0, "centerFrequency": 5405e6},
                    master_radargrid={
                        "numberOfRows": 100,
                        "numberOfColumns": 200,
                        "columnSpacing": 2.0,
                        "rangeTimeFirstPixel": 0.001,
                    },
                    slave_acquisition={"prf": 1020.0, "startGPSTime": 100.0, "centerFrequency": 5405e6},
                    slave_radargrid={"numberOfRows": 120, "numberOfColumns": 220, "columnSpacing": 2.5, "rangeTimeFirstPixel": 0.002},
                    slave_doppler={"combinedDoppler": {"coefficients": [1.0]}},
                )

            mock_isce3.assert_called_once()
            self.assertEqual(mock_isce3.call_args.kwargs["target_rows"], 100)
            self.assertEqual(mock_isce3.call_args.kwargs["target_cols"], 200)


if __name__ == "__main__":
    unittest.main()
