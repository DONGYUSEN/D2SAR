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


from tianyi_importer import TianyiImporter


class TianyiImporterSlcFormatTests(unittest.TestCase):
    def test_run_marks_tianyi_measurement_as_single_band_cint16_complex(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            product_dir = Path(tmp) / "product"
            output_dir = Path(tmp) / "out"
            (product_dir / "annotation").mkdir(parents=True)
            (product_dir / "measurement").mkdir()
            (product_dir / "annotation" / "scene.xml").write_text(
                "<root/>", encoding="utf-8"
            )
            (product_dir / "measurement" / "scene.tiff").write_bytes(b"")

            importer = TianyiImporter(str(product_dir))
            fake_common_processing = types.SimpleNamespace(
                verify_and_correct_look_direction=lambda *args, **kwargs: (
                    "RIGHT",
                    False,
                )
            )

            with (
                mock.patch.dict(
                    sys.modules, {"common_processing": fake_common_processing}
                ),
                mock.patch.object(importer, "parse_xml_root", return_value=object()),
                mock.patch.object(
                    importer,
                    "extract_acquisition",
                    return_value={
                        "polarisation": "VV",
                        "startTimeUTC": "2023-11-10T04:39:48.000000",
                        "stopTimeUTC": "2023-11-10T04:40:00.000000",
                        "lookDirection": "RIGHT",
                    },
                ),
                mock.patch.object(
                    importer,
                    "extract_scene_info",
                    return_value={"sceneCorners": [{"lat": 0.0, "lon": 0.0}]},
                ),
                mock.patch.object(importer, "extract_orbit", return_value={}),
                mock.patch.object(
                    importer,
                    "extract_radar_grid",
                    return_value={"numberOfRows": 14580, "numberOfColumns": 12544},
                ),
                mock.patch.object(importer, "extract_doppler", return_value={}),
                mock.patch.object(
                    importer, "resolve_dem_option", return_value=(None, None)
                ),
            ):
                manifest_path = importer.run(str(output_dir), download_dem=False)

            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            self.assertEqual(manifest["slc"]["sample_format"], "cint16")
            self.assertEqual(manifest["slc"]["storage_layout"], "single_band_complex")
            self.assertEqual(manifest["slc"]["complex_band_count"], 1)
            self.assertNotIn("band_mapping", manifest["slc"])
            self.assertEqual(
                manifest["slc"]["processing_format"], "single_band_cfloat32"
            )


if __name__ == "__main__":
    unittest.main()
