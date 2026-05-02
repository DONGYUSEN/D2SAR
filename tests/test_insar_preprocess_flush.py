import sys
import tempfile
import types
import unittest
import json
import contextlib
import io
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


class ResampleFlushTests(unittest.TestCase):
    def test_resample_closes_isce3_rasters_before_return(self):
        import insar_preprocess

        created_rasters = []

        class FakeRaster:
            def __init__(self, *args):
                self.args = args
                self.closed = False
                self.width = int(args[1]) if len(args) > 2 and isinstance(args[1], int) else 5
                self.length = int(args[2]) if len(args) > 2 and isinstance(args[2], int) else 4
                self.dataset = self
                created_rasters.append(self)

            def close_dataset(self):
                self.closed = True

            def GetRasterBand(self, _idx):
                return self

            def WriteArray(self, _data):
                pass

            def FlushCache(self):
                pass

        class FakeResampSlc:
            def __init__(self, *args):
                pass

            def resamp(self, *args):
                pass

        class FakePoly2d:
            def initPoly(self, *, rangeOrder, azimuthOrder, coeffs):
                self.coeffs = coeffs

            def eval(self, col, row):
                if len(self.coeffs) > 1:
                    return self.coeffs[1][0] * row
                if self.coeffs and len(self.coeffs[0]) > 1:
                    return self.coeffs[0][1] * col
                return 0.0

        fake_isce3 = types.ModuleType("isce3")
        fake_isce3.image = types.SimpleNamespace(ResampSlc=FakeResampSlc)
        fake_core = types.ModuleType("isce3.core")
        fake_core.LUT2d = lambda *args, **kwargs: ("lut", args, kwargs)
        fake_core.Poly2d = FakePoly2d
        fake_io = types.ModuleType("isce3.io")
        fake_io.Raster = FakeRaster

        original_modules = {
            name: sys.modules.get(name)
            for name in ("isce3", "isce3.core", "isce3.io")
        }
        sys.modules["isce3"] = fake_isce3
        sys.modules["isce3.core"] = fake_core
        sys.modules["isce3.io"] = fake_io
        try:
            with tempfile.TemporaryDirectory() as tmp:
                dst = Path(tmp) / "normalized.tif"
                insar_preprocess._resample_slave_slc_with_isce3(
                    "source.tif",
                    dst,
                    source_rows=4,
                    source_cols=5,
                    source_prf=1.0,
                    target_prf=1.0,
                    source_rsr=1.0,
                    target_rsr=1.0,
                    geometry_mode="zero-doppler",
                    doppler_coefficients=[0.0],
                )
        finally:
            for name, module in original_modules.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

        self.assertGreaterEqual(len(created_rasters), 4)
        self.assertTrue(all(raster.closed for raster in created_rasters))
        offset_rasters = [
            raster for raster in created_rasters
            if raster.args and str(raster.args[0]).endswith(("prep_range_offsets.tif", "prep_azimuth_offsets.tif"))
        ]
        self.assertTrue(offset_rasters)
        self.assertTrue(all(raster.width == 5 and raster.length == 4 for raster in offset_rasters))

    def test_resample_failure_raises_instead_of_copying_original_slc(self):
        import insar_preprocess
        from osgeo import gdal
        import numpy as np

        class FakeRaster:
            def __init__(self, *args):
                self.args = args
                self.width = int(args[1]) if len(args) > 2 and isinstance(args[1], int) else 5
                self.length = int(args[2]) if len(args) > 2 and isinstance(args[2], int) else 4
                self.dataset = self

            def close_dataset(self):
                pass

            def GetRasterBand(self, _idx):
                return self

            def WriteArray(self, _data):
                pass

            def FlushCache(self):
                pass

        class FailingResampSlc:
            def __init__(self, *args):
                pass

            def resamp(self, *args):
                raise RuntimeError("synthetic resample failure")

        class FakePoly2d:
            def initPoly(self, *, rangeOrder, azimuthOrder, coeffs):
                self.coeffs = coeffs

            def eval(self, col, row):
                return 0.0

        fake_isce3 = types.ModuleType("isce3")
        fake_isce3.image = types.SimpleNamespace(ResampSlc=FailingResampSlc)
        fake_core = types.ModuleType("isce3.core")
        fake_core.LUT2d = lambda *args, **kwargs: ("lut", args, kwargs)
        fake_core.Poly2d = FakePoly2d
        fake_io = types.ModuleType("isce3.io")
        fake_io.Raster = FakeRaster

        original_modules = {
            name: sys.modules.get(name)
            for name in ("isce3", "isce3.core", "isce3.io")
        }
        sys.modules["isce3"] = fake_isce3
        sys.modules["isce3.core"] = fake_core
        sys.modules["isce3.io"] = fake_io
        try:
            with tempfile.TemporaryDirectory() as tmp:
                src = Path(tmp) / "source.tif"
                dst = Path(tmp) / "normalized.tif"
                ds = gdal.GetDriverByName("GTiff").Create(
                    str(src),
                    5,
                    4,
                    1,
                    gdal.GDT_CFloat32,
                )
                ds.GetRasterBand(1).WriteArray(np.ones((4, 5), dtype=np.complex64))
                ds = None

                with self.assertRaisesRegex(RuntimeError, "synthetic resample failure"):
                    insar_preprocess._resample_slave_slc_with_isce3(
                        src,
                        dst,
                        source_rows=4,
                        source_cols=5,
                        source_prf=2.0,
                        target_prf=1.0,
                        source_rsr=1.0,
                        target_rsr=1.0,
                        geometry_mode="zero-doppler",
                        doppler_coefficients=[0.0],
                    )
        finally:
            for name, module in original_modules.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

    def test_resample_progress_output_is_suppressed(self):
        import insar_preprocess

        class FakeRaster:
            def __init__(self, *args):
                self.width = int(args[1]) if len(args) > 2 and isinstance(args[1], int) else 5
                self.length = int(args[2]) if len(args) > 2 and isinstance(args[2], int) else 4
                self.dataset = self

            def close_dataset(self):
                pass

            def GetRasterBand(self, _idx):
                return self

            def WriteArray(self, _data):
                pass

            def FlushCache(self):
                pass

        class FakeResampSlc:
            def __init__(self, *args):
                pass

            def resamp(self, *args):
                print("Resampling using 29 tiles of 1000 lines per tile")
                print("Reading in image data for tile 0")
                print("Interpolating tile 0")

        class FakePoly2d:
            def initPoly(self, *, rangeOrder, azimuthOrder, coeffs):
                self.coeffs = coeffs

            def eval(self, col, row):
                if len(self.coeffs) > 1:
                    return self.coeffs[1][0] * row
                if self.coeffs and len(self.coeffs[0]) > 1:
                    return self.coeffs[0][1] * col
                return 0.0

        fake_isce3 = types.ModuleType("isce3")
        fake_isce3.image = types.SimpleNamespace(ResampSlc=FakeResampSlc)
        fake_core = types.ModuleType("isce3.core")
        fake_core.LUT2d = lambda *args, **kwargs: ("lut", args, kwargs)
        fake_core.Poly2d = FakePoly2d
        fake_io = types.ModuleType("isce3.io")
        fake_io.Raster = FakeRaster

        original_modules = {
            name: sys.modules.get(name)
            for name in ("isce3", "isce3.core", "isce3.io")
        }
        sys.modules["isce3"] = fake_isce3
        sys.modules["isce3.core"] = fake_core
        sys.modules["isce3.io"] = fake_io
        try:
            with tempfile.TemporaryDirectory() as tmp:
                dst = Path(tmp) / "normalized.tif"
                captured = io.StringIO()
                with contextlib.redirect_stdout(captured):
                    insar_preprocess._resample_slave_slc_with_isce3(
                        "source.tif",
                        dst,
                        source_rows=4,
                        source_cols=5,
                        source_prf=1.0,
                        target_prf=1.0,
                        source_rsr=1.0,
                        target_rsr=1.0,
                        geometry_mode="zero-doppler",
                        doppler_coefficients=[0.0],
                    )
        finally:
            for name, module in original_modules.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

        self.assertNotIn("Resampling using", captured.getvalue())
        self.assertNotIn("Reading in image data", captured.getvalue())
        self.assertNotIn("Interpolating tile", captured.getvalue())


class PreprocessManifestPathTests(unittest.TestCase):
    def test_normalized_manifest_rewrites_orbit_and_scene_to_resolved_paths(self):
        import insar_preprocess

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata_dir = root / "metadata"
            metadata_dir.mkdir()
            (metadata_dir / "acquisition.json").write_text(
                json.dumps({"prf": 10.0, "startGPSTime": 1.0}),
                encoding="utf-8",
            )
            (metadata_dir / "radargrid.json").write_text(
                json.dumps({"numberOfRows": 4, "numberOfColumns": 5}),
                encoding="utf-8",
            )
            (metadata_dir / "doppler.json").write_text(json.dumps({}), encoding="utf-8")
            (metadata_dir / "orbit.json").write_text(
                json.dumps({"stateVectors": [{"timeUTC": "2025-01-01T00:00:00"}]}),
                encoding="utf-8",
            )
            (metadata_dir / "scene.json").write_text(
                json.dumps({"sceneCorners": []}),
                encoding="utf-8",
            )
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "slc": {"path": "missing.tif"},
                        "metadata": {
                            "acquisition": "/missing/acquisition.json",
                            "radargrid": "/missing/radargrid.json",
                            "doppler": "/missing/doppler.json",
                            "orbit": "/missing/orbit.json",
                            "scene": "/missing/scene.json",
                        },
                    }
                ),
                encoding="utf-8",
            )
            stage_dir = root / "normalize"
            stage_dir.mkdir()

            _, normalized_manifest_path = insar_preprocess.build_preprocess_plan(
                precheck={
                    "requires_prep": True,
                    "recommended_geometry_mode": "zero-doppler",
                    "checks": {"doppler": {"severity": "warn"}},
                },
                slave_manifest_path=manifest_path,
                stage_dir=stage_dir,
            )

            normalized = json.loads(Path(normalized_manifest_path).read_text(encoding="utf-8"))
            self.assertEqual(
                normalized["metadata"]["orbit"],
                str((metadata_dir / "orbit.json").resolve()),
            )
            self.assertEqual(
                normalized["metadata"]["scene"],
                str((metadata_dir / "scene.json").resolve()),
            )

    def test_dc_policy_records_signal_processing_actions(self):
        import insar_preprocess

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata_dir = root / "metadata"
            metadata_dir.mkdir()
            (metadata_dir / "acquisition.json").write_text(
                json.dumps({"prf": 1000.0, "startGPSTime": 20.0}),
                encoding="utf-8",
            )
            (metadata_dir / "radargrid.json").write_text(
                json.dumps(
                    {
                        "numberOfRows": 4,
                        "numberOfColumns": 5,
                        "columnSpacing": 1.0,
                        "rangeTimeFirstPixel": 0.001,
                    }
                ),
                encoding="utf-8",
            )
            (metadata_dir / "doppler.json").write_text(
                json.dumps(
                    {
                        "combinedDoppler": {
                            "referencePoint": 0.001,
                            "coefficients": [50.0],
                        }
                    }
                ),
                encoding="utf-8",
            )
            (metadata_dir / "orbit.json").write_text(json.dumps({}), encoding="utf-8")
            (metadata_dir / "scene.json").write_text(json.dumps({}), encoding="utf-8")
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "slc": {"path": "missing.tif"},
                        "metadata": {
                            "acquisition": "metadata/acquisition.json",
                            "radargrid": "metadata/radargrid.json",
                            "doppler": "metadata/doppler.json",
                            "orbit": "metadata/orbit.json",
                            "scene": "metadata/scene.json",
                        },
                    }
                ),
                encoding="utf-8",
            )

            _, normalized_manifest_path = insar_preprocess.build_preprocess_plan(
                precheck={
                    "requires_prep": True,
                    "recommended_geometry_mode": "zero-doppler",
                    "checks": {"doppler": {"severity": "warn"}},
                },
                slave_manifest_path=manifest_path,
                stage_dir=root / "normalize",
                master_acquisition={"prf": 1000.0, "startGPSTime": 10.0},
                master_radargrid={
                    "numberOfRows": 4,
                    "numberOfColumns": 5,
                    "columnSpacing": 1.0,
                    "rangeTimeFirstPixel": 0.001,
                },
                master_doppler={
                    "combinedDoppler": {
                        "referencePoint": 0.001,
                        "coefficients": [0.0],
                    }
                },
            )

            normalized = json.loads(Path(normalized_manifest_path).read_text(encoding="utf-8"))
            processing = normalized["processing"]["insar_preprocess"]
            self.assertEqual(processing["dc_policy"]["regime"], "medium")
            self.assertIn("dc-deramp-reramp", processing["actions"])
            self.assertIn("harmonize-doppler-to-master", processing["actions"])

    def test_normalized_manifest_slc_dimensions_and_radargrid_prf_match_resample_output(self):
        import insar_preprocess

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata_dir = root / "metadata"
            metadata_dir.mkdir()
            (root / "slave.tif").write_bytes(b"placeholder")
            (metadata_dir / "acquisition.json").write_text(
                json.dumps({"prf": 20.0, "startGPSTime": 1.0}),
                encoding="utf-8",
            )
            (metadata_dir / "radargrid.json").write_text(
                json.dumps(
                    {
                        "prf": 20.0,
                        "numberOfRows": 11,
                        "numberOfColumns": 5,
                        "columnSpacing": 1.0,
                        "rangeTimeFirstPixel": 0.001,
                    }
                ),
                encoding="utf-8",
            )
            for name in ("doppler", "orbit", "scene"):
                (metadata_dir / f"{name}.json").write_text(json.dumps({}), encoding="utf-8")
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "slc": {"path": "slave.tif", "rows": 11, "columns": 5},
                        "metadata": {
                            "acquisition": "metadata/acquisition.json",
                            "radargrid": "metadata/radargrid.json",
                            "doppler": "metadata/doppler.json",
                            "orbit": "metadata/orbit.json",
                            "scene": "metadata/scene.json",
                        },
                    }
                ),
                encoding="utf-8",
            )

            original_resample = insar_preprocess._resample_slave_slc
            try:
                insar_preprocess._resample_slave_slc = (
                    lambda src, dst, **kwargs: (str(dst), 7, 5, 2.0, 1.0)
                )
                _, normalized_manifest_path = insar_preprocess.build_preprocess_plan(
                    precheck={
                        "requires_prep": True,
                        "recommended_geometry_mode": "zero-doppler",
                        "checks": {"prf": {"severity": "warn"}},
                    },
                    slave_manifest_path=manifest_path,
                    stage_dir=root / "normalize",
                    master_acquisition={"prf": 10.0, "startGPSTime": 1.0},
                    master_radargrid={
                        "prf": 10.0,
                        "numberOfRows": 11,
                        "numberOfColumns": 5,
                        "columnSpacing": 1.0,
                        "rangeTimeFirstPixel": 0.001,
                    },
                )
            finally:
                insar_preprocess._resample_slave_slc = original_resample

            normalized = json.loads(Path(normalized_manifest_path).read_text(encoding="utf-8"))
            radargrid = json.loads(Path(normalized["metadata"]["radargrid"]).read_text(encoding="utf-8"))
            self.assertEqual(normalized["slc"]["rows"], 7)
            self.assertEqual(normalized["slc"]["columns"], 5)
            self.assertEqual(radargrid["numberOfRows"], 7)
            self.assertEqual(radargrid["numberOfColumns"], 5)
            self.assertEqual(radargrid["prf"], 10.0)

if __name__ == "__main__":
    unittest.main()
