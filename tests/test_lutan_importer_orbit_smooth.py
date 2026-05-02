import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


from lutan_importer import LutanImporter


class LutanImporterOrbitSmoothingTests(unittest.TestCase):
    def _minimal_orbit(self) -> dict:
        return {
            "header": {
                "firstStateTimeUTC": "2025-01-01T00:00:00.000000",
                "lastStateTimeUTC": "2025-01-01T00:00:11.000000",
                "numStateVectors": 12,
            },
            "stateVectors": [
                {
                    "timeUTC": f"2025-01-01T00:00:{idx:02d}.000000",
                    "gpsTime": 1000.0 + idx,
                    "posX": 7000000.0 + idx,
                    "posY": 100.0 + idx,
                    "posZ": 200.0 + idx,
                    "velX": 1.0,
                    "velY": 2.0,
                    "velZ": 3.0,
                }
                for idx in range(12)
            ],
        }

    def test_run_writes_smoothed_orbit_and_raw_orbit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            product_dir = Path(tmp) / "product"
            output_dir = Path(tmp) / "out"
            product_dir.mkdir()
            (product_dir / "scene_SLC.tiff").write_bytes(b"")
            (product_dir / "scene.meta.xml").write_text("<root/>", encoding="utf-8")
            (product_dir / "scene.rpc").write_text("LINE_OFF=0\n", encoding="utf-8")

            orbit = self._minimal_orbit()
            smoothed = {
                **orbit,
                "stateVectors": [
                    {**sv, "posX": sv["posX"] + 0.25}
                    for sv in orbit["stateVectors"]
                ],
                "smoothed": True,
                "smoothing": {
                    "algorithm": "isce2-lutan1-orbit-filter",
                    "status": "applied",
                    "n_outliers": 1,
                    "ignore_start": 1,
                    "ignore_end": 1,
                },
            }

            importer = LutanImporter(str(product_dir))
            fake_common_processing = types.SimpleNamespace(
                verify_and_correct_look_direction=lambda *args, **kwargs: ("RIGHT", False)
            )
            with (
                mock.patch.dict(sys.modules, {"common_processing": fake_common_processing}),
                mock.patch.object(importer, "parse_meta_xml", return_value=object()),
                mock.patch.object(importer, "extract_general_header", return_value={"mission": "LT1", "sensorMode": "STRIP"}),
                mock.patch.object(
                    importer,
                    "extract_acquisition_info",
                    return_value={
                        "sensor": "LUTAN1",
                        "imagingMode": "STRIP",
                        "polarisationMode": "HH",
                        "lookDirection": "RIGHT",
                    },
                ),
                mock.patch.object(
                    importer,
                    "extract_image_data_info",
                    return_value={
                        "numberOfRows": 10,
                        "numberOfColumns": 20,
                        "rowSpacing": 1.0,
                        "columnSpacing": 1.0,
                        "groundRangeResolution": 1.0,
                        "azimuthResolution": 1.0,
                    },
                ),
                mock.patch.object(
                    importer,
                    "extract_scene_info",
                    return_value={
                        "sceneID": "scene",
                        "startTimeUTC": "2025-01-01T00:00:00.000000",
                        "stopTimeUTC": "2025-01-01T00:00:10.000000",
                        "rangeTimeFirstPixel": 0.1,
                        "rangeTimeLastPixel": 0.2,
                        "sceneCenterCoord": {"lat": 0.0, "lon": 0.0, "incidenceAngle": 30.0},
                        "sceneCorners": {"a": {"lat": 0.0, "lon": 0.0}},
                        "sceneAverageHeight": 0.0,
                        "headingAngle": 0.0,
                    },
                ),
                mock.patch.object(importer, "extract_orbit", return_value=orbit),
                mock.patch.object(importer, "extract_doppler", return_value={}),
                mock.patch.object(
                    importer,
                    "extract_radar_params",
                    return_value={"centerFrequency": 1.0, "prf": 2.0},
                ),
                mock.patch.object(
                    importer,
                    "extract_processing_info",
                    return_value={"rangeLooks": 1, "azimuthLooks": 1, "geocodedFlag": False},
                ),
                mock.patch.object(importer, "resolve_dem_option", return_value=(None, None)),
                mock.patch.object(importer, "_smooth_orbit_for_import", return_value=smoothed) as smooth_mock,
            ):
                manifest_path = importer.run(str(output_dir), download_dem=False)

            metadata_dir = output_dir / "metadata"
            written_orbit = json.loads((metadata_dir / "orbit.json").read_text(encoding="utf-8"))
            raw_orbit = json.loads((metadata_dir / "orbit.raw.json").read_text(encoding="utf-8"))
            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))

            smooth_mock.assert_called_once_with(orbit)
            self.assertTrue(written_orbit["smoothed"])
            self.assertEqual(written_orbit["header"], orbit["header"])
            self.assertEqual(raw_orbit, orbit)
            self.assertEqual(manifest["slc"]["sample_format"], "iq_int16")
            self.assertEqual(manifest["slc"]["storage_layout"], "two_band_iq")
            self.assertEqual(manifest["slc"]["complex_band_count"], 1)
            self.assertEqual(manifest["metadata"]["orbit"], str((metadata_dir / "orbit.json").resolve()))
            self.assertEqual(
                manifest["metadata"]["orbit_raw"],
                str((metadata_dir / "orbit.raw.json").resolve()),
            )
            self.assertNotIn("rpc", manifest["ancillary"])


if __name__ == "__main__":
    unittest.main()
