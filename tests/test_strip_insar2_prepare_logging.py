import sys
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


class StripInsar2PrepareLoggingTests(unittest.TestCase):
    def test_normalize_start_message_mentions_prf_difference_before_normal(self):
        import strip_insar2

        message = strip_insar2._format_normalize_start_message(
            {
                "checks": {
                    "prf": {"severity": "warn"},
                    "doppler": {"severity": "ok"},
                    "radar_grid": {"severity": "ok"},
                }
            },
            master_acquisition={"prf": 3292.181152},
            slave_acquisition={"prf": 3325.942382},
        )

        self.assertIn("find PRF 差异大", message)
        self.assertIn("开始 normal 处理", message)
        self.assertIn("slave PRF 3325.942382", message)
        self.assertIn("master PRF 3292.181152", message)
        self.assertIn("同步修改 acquisition/radargrid/doppler JSON", message)

    def test_runtime_prepare_runs_for_full_scene_without_window(self):
        import strip_insar2

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dem = root / "dem.tif"
            dem.write_bytes(b"dem")

            def write_scene(name, prf):
                from osgeo import gdal
                import numpy as np

                scene = root / name
                metadata = scene / "metadata"
                metadata.mkdir(parents=True)
                ds = gdal.GetDriverByName("GTiff").Create(
                    str(scene / "slc.tif"),
                    8,
                    10,
                    1,
                    gdal.GDT_CFloat32,
                )
                ds.GetRasterBand(1).WriteArray(np.ones((10, 8), dtype=np.complex64))
                ds = None
                (metadata / "acquisition.json").write_text(
                    json.dumps(
                        {
                            "prf": prf,
                            "startGPSTime": 100.0,
                            "centerFrequency": 1.0,
                        }
                    ),
                    encoding="utf-8",
                )
                (metadata / "radargrid.json").write_text(
                    json.dumps(
                        {
                            "numberOfRows": 10,
                            "numberOfColumns": 8,
                            "columnSpacing": 1.0,
                            "rangeTimeFirstPixel": 0.001,
                        }
                    ),
                    encoding="utf-8",
                )
                (metadata / "doppler.json").write_text(
                    json.dumps({"combinedDoppler": {"referencePoint": 0.001, "coefficients": [0.0]}}),
                    encoding="utf-8",
                )
                (metadata / "orbit.json").write_text(json.dumps({}), encoding="utf-8")
                (metadata / "scene.json").write_text(
                    json.dumps({"sceneCorners": []}),
                    encoding="utf-8",
                )
                manifest = scene / "manifest.json"
                manifest.write_text(
                    json.dumps(
                        {
                            "slc": {"path": "slc.tif"},
                            "metadata": {
                                "acquisition": "metadata/acquisition.json",
                                "radargrid": "metadata/radargrid.json",
                                "doppler": "metadata/doppler.json",
                                "orbit": "metadata/orbit.json",
                                "scene": "metadata/scene.json",
                            },
                            "dem": {"path": str(dem)},
                        }
                    ),
                    encoding="utf-8",
                )
                return manifest

            master = write_scene("master", 10.0)
            slave = write_scene("slave", 11.0)
            pair_dir = root / "pair"

            import insar_preprocess

            original_resample = insar_preprocess._resample_slave_slc

            def fake_resample(src_path, dst_path, **kwargs):
                insar_preprocess._copy_raster_with_gdal(src_path, dst_path)
                return (
                    str(dst_path),
                    int(kwargs["source_rows"]),
                    int(kwargs["source_cols"]),
                    1.0,
                    1.0,
                )

            insar_preprocess._resample_slave_slc = fake_resample
            try:
                prepared = strip_insar2._prepare_runtime_inputs(
                    master_manifest_path=master,
                    slave_manifest_path=slave,
                    pair_dir=pair_dir,
                    window=None,
                    dem_path=str(dem),
                    dem_cache_dir=None,
                    dem_margin_deg=0.2,
                )
            finally:
                insar_preprocess._resample_slave_slc = original_resample

            summary = json.loads((pair_dir / "prepared" / "prepare_summary.json").read_text())
            self.assertTrue(summary["normalization"]["requested"])
            self.assertIn("normalize-slave-prf", summary["normalization"]["actions"])
            self.assertEqual(summary["crop"]["window"], None)
            self.assertTrue(Path(prepared["prepared_master_manifest"]).exists())
            self.assertTrue(Path(prepared["prepared_slave_manifest"]).exists())
            prepared_slave = json.loads(Path(prepared["prepared_slave_manifest"]).read_text())
            self.assertIn("insar_preprocess", prepared_slave["processing"])

    def test_cropped_slave_manifest_preserves_normalization_provenance(self):
        import insar_subset

        with tempfile.TemporaryDirectory() as tmp:
            from osgeo import gdal
            import numpy as np

            root = Path(tmp)
            metadata = root / "metadata"
            metadata.mkdir()
            ds = gdal.GetDriverByName("GTiff").Create(
                str(root / "slave.tif"),
                10,
                10,
                1,
                gdal.GDT_CFloat32,
            )
            ds.GetRasterBand(1).WriteArray(np.ones((10, 10), dtype=np.complex64))
            ds = None
            (metadata / "acquisition.json").write_text(
                json.dumps({"prf": 10.0, "startGPSTime": 1.0}),
                encoding="utf-8",
            )
            (metadata / "radargrid.json").write_text(
                json.dumps(
                    {
                        "numberOfRows": 10,
                        "numberOfColumns": 10,
                        "columnSpacing": 1.0,
                        "rangeTimeFirstPixel": 0.001,
                    }
                ),
                encoding="utf-8",
            )
            for name in ("doppler", "orbit", "scene"):
                (metadata / f"{name}.json").write_text(json.dumps({}), encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "slc": {"path": "slave.tif"},
                        "metadata": {
                            "acquisition": "metadata/acquisition.json",
                            "radargrid": "metadata/radargrid.json",
                            "doppler": "metadata/doppler.json",
                            "orbit": "metadata/orbit.json",
                            "scene": "metadata/scene.json",
                        },
                        "processing": {
                            "insar_preprocess": {
                                "actions": ["normalize-slave-prf"],
                                "source_manifest": "original.json",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            out_manifest = insar_subset.build_cropped_manifest(
                manifest_path=manifest,
                output_dir=root / "prepared",
                output_name="slave",
                window=(0, 0, 1, 1),
            )
            cropped = json.loads(Path(out_manifest).read_text(encoding="utf-8"))
            self.assertIn("insar_preprocess", cropped["processing"])
            self.assertIn("insar_crop", cropped["processing"])


if __name__ == "__main__":
    unittest.main()
