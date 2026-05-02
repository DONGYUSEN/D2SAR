import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
from osgeo import gdal, osr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(PROJECT_ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from test_sentinel_importer import SentinelImporterTests, make_sentinel_safe_dir


class TopsRtcTests(unittest.TestCase):
    def _write_test_slc(self, path: Path) -> None:
        driver = gdal.GetDriverByName("GTiff")
        ds = driver.Create(str(path), 4, 8, 1, gdal.GDT_CFloat32)
        if ds is None:
            raise RuntimeError("failed to create test SLC")
        arr = np.arange(32, dtype=np.float32).reshape(8, 4) + 1j
        ds.GetRasterBand(1).WriteArray(arr.astype(np.complex64))
        ds = None

    def _make_manifest(self, root: Path) -> Path:
        from sentinel_importer import SentinelImporter

        safe = make_sentinel_safe_dir(root, SentinelImporterTests.MANIFEST_XML)
        return Path(SentinelImporter(str(safe)).run(str(root / "imported"), download_dem=False))

    def test_prepare_tops_rtc_writes_per_burst_plan(self) -> None:
        from tops_rtc import prepare_tops_rtc

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = self._make_manifest(root)

            result = prepare_tops_rtc(manifest_path, root / "rtc")
            plan_path = Path(result["plan_path"])
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            self.assertTrue(plan_path.is_file())
            self.assertEqual(plan["sensor"], "sentinel-1")
            self.assertEqual(plan["swath"], "IW2")
            self.assertEqual(plan["polarisation"], "VV")
            self.assertEqual(plan["burst_count"], 2)
            self.assertEqual(len(plan["bursts"]), 2)
            self.assertEqual(plan["bursts"][0]["burstIndex"], 1)
            self.assertEqual(plan["bursts"][0]["slcWindow"]["xoff"], 0)
            self.assertEqual(plan["bursts"][0]["slcWindow"]["yoff"], 0)
            self.assertEqual(plan["bursts"][0]["slcWindow"]["xsize"], 4)
            self.assertEqual(plan["bursts"][0]["slcWindow"]["ysize"], 4)
            self.assertEqual(plan["bursts"][1]["slcWindow"]["yoff"], 4)
            self.assertEqual(plan["bursts"][0]["slcWindow"]["validWindow"]["xoff"], 1)
            self.assertEqual(plan["bursts"][0]["slcWindow"]["validWindow"]["ysize"], 4)
            self.assertTrue(plan["bursts"][0]["outputs"]["amplitude_h5"].endswith("burst_001/amplitude_fullres.h5"))
            self.assertEqual(plan["bursts"][0]["doppler"]["coefficients"], [1.0, 2.0, 3.0])

    def test_prepare_tops_rtc_rejects_non_sentinel_manifest(self) -> None:
        from tops_rtc import prepare_tops_rtc

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps({"sensor": "tianyi"}), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "sentinel-1"):
                prepare_tops_rtc(manifest_path, root / "rtc")

    def test_prepare_tops_rtc_keeps_vsi_zip_slc_path_normalized(self) -> None:
        from tops_rtc import prepare_tops_rtc

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = self._make_manifest(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["slc"]["path"] = {
                "path": "/vsizip//tmp/sentinel.zip/product.SAFE/measurement/s1.tiff",
                "storage": "zip",
                "member": "product.SAFE/measurement/s1.tiff",
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            result = prepare_tops_rtc(manifest_path, root / "rtc")
            plan = json.loads(Path(result["plan_path"]).read_text(encoding="utf-8"))

        self.assertEqual(plan["slc_path"], "/vsizip//tmp/sentinel.zip/product.SAFE/measurement/s1.tiff")

    def test_materialize_tops_rtc_plan_writes_burst_amplitude_hdf(self) -> None:
        from tops_rtc import materialize_tops_rtc_plan, prepare_tops_rtc

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = self._make_manifest(root)
            slc_path = root / "test_slc.tif"
            self._write_test_slc(slc_path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["slc"]["path"] = str(slc_path)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            result = prepare_tops_rtc(manifest_path, root / "rtc")
            materialized = materialize_tops_rtc_plan(result["plan_path"], burst_limit=1, block_rows=2)
            output_h5 = Path(materialized["bursts"][0]["amplitude_h5"])

            with h5py.File(output_h5, "r") as f:
                amplitude = f["slc_amplitude"][:]
                valid_mask = f["valid_mask"][:]
                product_type = f.attrs["product_type"]
                burst_index = int(f.attrs["burst_index"])

            self.assertEqual(materialized["burst_count"], 1)
            self.assertTrue(output_h5.is_file())

        self.assertEqual(product_type, "sentinel_tops_burst_amplitude")
        self.assertEqual(burst_index, 1)
        self.assertEqual(amplitude.shape, (4, 4))
        self.assertAlmostEqual(float(amplitude[0, 0]), 1.0)
        self.assertAlmostEqual(float(amplitude[3, 3]), float(np.sqrt(15.0**2 + 1.0)), places=6)
        self.assertEqual(int(valid_mask.sum()), 4)
        self.assertEqual(int(valid_mask[:, 1].sum()), 4)
        self.assertEqual(int(valid_mask[:, 0].sum()), 0)

    def test_compute_burst_rtc_factor_writes_burst_manifest_and_calls_compute(self) -> None:
        from tops_rtc import compute_burst_rtc_factor, prepare_tops_rtc

        calls = []

        def fake_compute(manifest_path, dem_path, output_path, orbit_interp="Legendre"):
            calls.append(
                {
                    "manifest_path": manifest_path,
                    "dem_path": dem_path,
                    "output_path": output_path,
                    "orbit_interp": orbit_interp,
                }
            )
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text("rtc", encoding="utf-8")
            return output_path

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = self._make_manifest(root)
            result = prepare_tops_rtc(manifest_path, root / "rtc")

            outputs = compute_burst_rtc_factor(
                result["plan_path"],
                root / "dem.tif",
                burst_limit=1,
                compute_func=fake_compute,
            )
            burst_manifest = Path(outputs["bursts"][0]["burst_manifest"])
            burst_manifest_data = json.loads(burst_manifest.read_text(encoding="utf-8"))
            radargrid = json.loads(Path(burst_manifest_data["metadata"]["radargrid"]).read_text(encoding="utf-8"))
            doppler = json.loads(Path(burst_manifest_data["metadata"]["doppler"]).read_text(encoding="utf-8"))

            self.assertEqual(outputs["burst_count"], 1)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["dem_path"], str(root / "dem.tif"))
            self.assertTrue(calls[0]["output_path"].endswith("burst_001/rtc_factor.tif"))
            self.assertEqual(burst_manifest_data["sensor"], "sentinel-1")
            self.assertEqual(burst_manifest_data["tops"]["burst_index"], 1)
            self.assertEqual(radargrid["numberOfRows"], 4)
            self.assertEqual(radargrid["numberOfColumns"], 4)
            self.assertEqual(doppler["combinedDoppler"]["coefficients"], [1.0, 2.0, 3.0])

    def test_compute_rtc_factor_multi_burst_uses_merged_radar_manifest(self) -> None:
        from tops_rtc import compute_burst_rtc_factor, prepare_tops_rtc

        calls = []

        def fake_compute(manifest_path, dem_path, output_path, orbit_interp="Legendre"):
            calls.append(
                {
                    "manifest_path": manifest_path,
                    "dem_path": dem_path,
                    "output_path": output_path,
                    "orbit_interp": orbit_interp,
                }
            )
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text("merged rtc", encoding="utf-8")
            return output_path

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = self._make_manifest(root)
            result = prepare_tops_rtc(manifest_path, root / "rtc")

            outputs = compute_burst_rtc_factor(
                result["plan_path"],
                root / "dem.tif",
                burst_limit=2,
                compute_func=fake_compute,
            )
            merged_manifest = Path(outputs["mosaic"]["manifest"])
            merged_manifest_data = json.loads(merged_manifest.read_text(encoding="utf-8"))
            radargrid = json.loads(Path(merged_manifest_data["metadata"]["radargrid"]).read_text(encoding="utf-8"))

        self.assertEqual(outputs["processing_mode"], "radar_mosaic")
        self.assertEqual(outputs["burst_count"], 2)
        self.assertEqual(len(calls), 1)
        self.assertEqual(outputs["plan_path"], str(Path(result["plan_path"])))
        self.assertTrue(calls[0]["output_path"].endswith("mosaic_rtc_factor.tif"))
        self.assertTrue(calls[0]["output_path"].startswith(str(root / "rtc")))
        self.assertEqual(merged_manifest_data["tops"]["burst_count"], 2)
        self.assertGreaterEqual(radargrid["numberOfRows"], 4)
        self.assertEqual(radargrid["numberOfColumns"], 4)

    def test_compute_burst_topo_materializes_amplitude_and_calls_topo(self) -> None:
        from tops_rtc import compute_burst_topo, prepare_tops_rtc

        calls = []

        def fake_topo(manifest_path, dem_path, output_h5, block_rows=256, orbit_interp=None, use_gpu=False, gpu_id=0):
            calls.append(
                {
                    "manifest_path": manifest_path,
                    "dem_path": dem_path,
                    "output_h5": output_h5,
                    "block_rows": block_rows,
                    "orbit_interp": orbit_interp,
                    "use_gpu": use_gpu,
                    "gpu_id": gpu_id,
                }
            )
            with h5py.File(output_h5, "a") as f:
                shape = f["slc_amplitude"].shape
                f.create_dataset("longitude", data=np.zeros(shape, dtype=np.float32))
                f.create_dataset("latitude", data=np.ones(shape, dtype=np.float32))
                f.create_dataset("height", data=np.full(shape, 2, dtype=np.float32))
            return output_h5

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = self._make_manifest(root)
            slc_path = root / "test_slc.tif"
            self._write_test_slc(slc_path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["slc"]["path"] = str(slc_path)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            result = prepare_tops_rtc(manifest_path, root / "rtc")

            topo = compute_burst_topo(
                result["plan_path"],
                root / "dem.tif",
                burst_limit=1,
                block_rows=2,
                use_gpu=True,
                gpu_id=3,
                topo_func=fake_topo,
            )
            output_h5 = Path(topo["bursts"][0]["amplitude_h5"])
            burst_manifest = Path(topo["bursts"][0]["burst_manifest"])
            with h5py.File(output_h5, "r") as f:
                has_topo = all(name in f for name in ("longitude", "latitude", "height"))

            self.assertEqual(topo["burst_count"], 1)
            self.assertTrue(output_h5.is_file())
            self.assertTrue(burst_manifest.is_file())
            self.assertTrue(has_topo)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["dem_path"], str(root / "dem.tif"))
            self.assertEqual(calls[0]["block_rows"], 2)
            self.assertTrue(calls[0]["use_gpu"])
            self.assertEqual(calls[0]["gpu_id"], 3)

    def test_apply_burst_rtc_factor_writes_rtc_amplitude_hdf(self) -> None:
        from tops_rtc import apply_burst_rtc_factor

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            amp_path = root / "amplitude_fullres.h5"
            factor_path = root / "rtc_factor.tif"
            rtc_path = root / "amplitude_rtc.h5"

            amp_data = np.array([[2.0, 4.0], [8.0, 18.0]], dtype=np.float32)
            valid = np.array([[1, 1], [1, 1]], dtype=np.uint8)
            lon = np.array([[94.0, 95.0], [94.0, 95.0]], dtype=np.float32)
            lat = np.array([[28.0, 28.0], [29.0, 29.0]], dtype=np.float32)
            height = np.array([[100.0, 200.0], [100.0, 200.0]], dtype=np.float32)
            factor_data = np.array([[4.0, 1.0], [4.0, 1.0]], dtype=np.float32)

            with h5py.File(amp_path, "w") as f:
                f.attrs["product_type"] = "sentinel_tops_burst_amplitude"
                f.attrs["burst_index"] = 1
                f.create_dataset("slc_amplitude", data=amp_data)
                f.create_dataset("valid_mask", data=valid)
                f.create_dataset("longitude", data=lon)
                f.create_dataset("latitude", data=lat)
                f.create_dataset("height", data=height)

            driver = gdal.GetDriverByName("GTiff")
            ds = driver.Create(str(factor_path), 2, 2, 1, gdal.GDT_Float32)
            ds.GetRasterBand(1).WriteArray(factor_data)
            ds = None

            result_path = apply_burst_rtc_factor(str(amp_path), str(factor_path), str(rtc_path))

            with h5py.File(result_path, "r") as f:
                rtc_amp = f["rtc_amplitude"][:]
                stored_valid = f["valid_mask"][:]
                stored_lon = f["longitude"][:]
                stored_lat = f["latitude"][:]
                stored_height = f["height"][:]
                product_type = f.attrs["product_type"]

            np.testing.assert_almost_equal(rtc_amp[0, 0], 1.0)
            np.testing.assert_almost_equal(rtc_amp[0, 1], 4.0)
            np.testing.assert_almost_equal(rtc_amp[1, 0], 4.0)
            np.testing.assert_almost_equal(rtc_amp[1, 1], 18.0)
            self.assertEqual(product_type, "sentinel_tops_burst_rtc_amplitude")
            np.testing.assert_array_equal(stored_valid, valid)
            np.testing.assert_array_equal(stored_lon, lon)
            np.testing.assert_array_equal(stored_lat, lat)
            np.testing.assert_array_equal(stored_height, height)

    def test_apply_burst_rtc_factor_zeros_invalid_where_factor_is_zero_or_nan(self) -> None:
        from tops_rtc import apply_burst_rtc_factor

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            amp_path = root / "amplitude_fullres.h5"
            factor_path = root / "rtc_factor.tif"
            rtc_path = root / "amplitude_rtc.h5"

            amp_data = np.array([[1.0, 2.0]], dtype=np.float32)
            valid = np.array([[1, 0]], dtype=np.uint8)
            lon = np.array([[95.0, 96.0]], dtype=np.float32)
            lat = np.array([[29.0, 30.0]], dtype=np.float32)
            height = np.array([[300.0, 400.0]], dtype=np.float32)
            factor_data = np.array([[1.0, 0.0]], dtype=np.float32)

            with h5py.File(amp_path, "w") as f:
                f.attrs["product_type"] = "sentinel_tops_burst_amplitude"
                f.attrs["burst_index"] = 2
                f.create_dataset("slc_amplitude", data=amp_data)
                f.create_dataset("valid_mask", data=valid)
                f.create_dataset("longitude", data=lon)
                f.create_dataset("latitude", data=lat)
                f.create_dataset("height", data=height)

            driver = gdal.GetDriverByName("GTiff")
            ds = driver.Create(str(factor_path), 2, 1, 1, gdal.GDT_Float32)
            ds.GetRasterBand(1).WriteArray(factor_data)
            ds = None

            result_path = apply_burst_rtc_factor(str(amp_path), str(factor_path), str(rtc_path))

            with h5py.File(result_path, "r") as f:
                rtc_amp = f["rtc_amplitude"][:]
                stored_valid = f["valid_mask"][:]

            np.testing.assert_almost_equal(rtc_amp[0, 0], 1.0)
            self.assertEqual(rtc_amp[0, 1], 0.0)
            self.assertEqual(stored_valid[0, 1], 0)

    def test_apply_rtc_multi_burst_merges_radar_grid_before_rtc(self) -> None:
        from tops_rtc import apply_burst_rtc

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bursts = []
            for idx, (start, value) in enumerate(((1000.0, 10.0), (1002.0, 20.0)), start=1):
                burst_dir = root / f"burst_{idx:03d}"
                burst_dir.mkdir()
                h5_path = burst_dir / "amplitude_fullres.h5"
                with h5py.File(h5_path, "w") as f:
                    f.create_dataset("slc_amplitude", data=np.full((3, 3), value, dtype=np.float32))
                    f.create_dataset("valid_mask", data=np.ones((3, 3), dtype=np.uint8))
                    f.create_dataset("longitude", data=np.full((3, 3), 95.0, dtype=np.float32))
                    f.create_dataset("latitude", data=np.full((3, 3), 29.0, dtype=np.float32))
                    f.create_dataset("height", data=np.full((3, 3), 100.0, dtype=np.float32))
                bursts.append(
                    {
                        "burstIndex": idx,
                        "radargrid": {
                            "numberOfRows": 3,
                            "numberOfColumns": 3,
                            "sensingStartGPSTime": start,
                            "rowSpacing": 1.0,
                            "rangeTimeFirstPixel": 0.0,
                            "columnSpacing": 10.0,
                            "firstValidLine": 0,
                            "numValidLines": 3,
                            "firstValidSample": 0,
                            "numValidSamples": 3,
                        },
                        "outputs": {
                            "directory": str(burst_dir),
                            "amplitude_h5": str(h5_path),
                            "rtc_factor_tif": str(burst_dir / "rtc_factor.tif"),
                        },
                    }
                )
            factor_path = root / "rtc" / "mosaic_rtc_factor.tif"
            factor_path.parent.mkdir()
            driver = gdal.GetDriverByName("GTiff")
            ds = driver.Create(str(factor_path), 3, 5, 1, gdal.GDT_Float32)
            ds.GetRasterBand(1).WriteArray(np.ones((5, 3), dtype=np.float32))
            ds = None
            plan_path = root / "rtc" / "tops_rtc_plan.json"
            plan_path.write_text(json.dumps({"bursts": bursts}), encoding="utf-8")

            result = apply_burst_rtc(plan_path, burst_limit=2)
            with h5py.File(result["mosaic"]["rtc_h5"], "r") as f:
                rtc_amp = f["rtc_amplitude"][:]

        self.assertEqual(result["processing_mode"], "radar_mosaic")
        self.assertEqual(result["burst_count"], 2)
        self.assertEqual(rtc_amp.shape, (5, 3))
        self.assertTrue(np.all(rtc_amp[:3, :] == 10.0))
        self.assertTrue(np.all(rtc_amp[3:, :] == 20.0))

    def test_apply_rtc_multi_burst_can_run_topo_on_merged_grid(self) -> None:
        import tops_rtc
        from tops_rtc import apply_burst_rtc

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bursts = []
            for idx, (start, value) in enumerate(((1000.0, 10.0), (1002.0, 20.0)), start=1):
                burst_dir = root / f"burst_{idx:03d}"
                burst_dir.mkdir()
                h5_path = burst_dir / "amplitude_fullres.h5"
                with h5py.File(h5_path, "w") as f:
                    f.create_dataset("slc_amplitude", data=np.full((3, 3), value, dtype=np.float32))
                    f.create_dataset("valid_mask", data=np.ones((3, 3), dtype=np.uint8))
                bursts.append(
                    {
                        "burstIndex": idx,
                        "doppler": {"coefficients": [0.0], "t0": 0.0},
                        "radargrid": {
                            "numberOfRows": 3,
                            "numberOfColumns": 3,
                            "sensingStartGPSTime": start,
                            "rowSpacing": 1.0,
                            "rangeTimeFirstPixel": 0.0,
                            "columnSpacing": 10.0,
                            "firstValidLine": 0,
                            "numValidLines": 3,
                            "firstValidSample": 0,
                            "numValidSamples": 3,
                        },
                        "outputs": {
                            "directory": str(burst_dir),
                            "amplitude_h5": str(h5_path),
                            "rtc_factor_tif": str(burst_dir / "rtc_factor.tif"),
                        },
                    }
                )
            factor_path = root / "rtc" / "mosaic_rtc_factor.tif"
            factor_path.parent.mkdir()
            ds = gdal.GetDriverByName("GTiff").Create(str(factor_path), 3, 5, 1, gdal.GDT_Float32)
            ds.GetRasterBand(1).WriteArray(np.ones((5, 3), dtype=np.float32))
            ds = None

            source_manifest = root / "source_manifest.json"
            metadata_dir = root / "source_metadata"
            metadata_dir.mkdir()
            for name in ("acquisition", "orbit", "scene"):
                (metadata_dir / f"{name}.json").write_text(json.dumps({name: True}), encoding="utf-8")
            source_manifest.write_text(
                json.dumps(
                    {
                        "sensor": "sentinel-1",
                        "metadata": {
                            "acquisition": str(metadata_dir / "acquisition.json"),
                            "orbit": str(metadata_dir / "orbit.json"),
                            "scene": str(metadata_dir / "scene.json"),
                        },
                    }
                ),
                encoding="utf-8",
            )
            plan_path = root / "rtc" / "tops_rtc_plan.json"
            plan_path.write_text(
                json.dumps({"input_manifest": str(source_manifest), "bursts": bursts}),
                encoding="utf-8",
            )
            topo_calls = []

            def fake_topo(manifest_path, dem_path, output_h5, block_rows=256, orbit_interp=None):
                topo_calls.append((manifest_path, dem_path, output_h5, block_rows, orbit_interp))
                with h5py.File(output_h5, "a") as f:
                    shape = f["slc_amplitude"].shape
                    f.create_dataset("longitude", data=np.full(shape, 95.0, dtype=np.float32))
                    f.create_dataset("latitude", data=np.full(shape, 29.0, dtype=np.float32))
                    f.create_dataset("height", data=np.full(shape, 100.0, dtype=np.float32))
                return output_h5

            with mock.patch.object(tops_rtc, "append_topo_coordinates_hdf", side_effect=fake_topo):
                result = apply_burst_rtc(
                    plan_path,
                    burst_limit=2,
                    dem_path=root / "dem.tif",
                    block_rows=7,
                    orbit_interp="Legendre",
                )

            with h5py.File(result["mosaic"]["rtc_h5"], "r") as f:
                has_geometry = all(name in f for name in ("longitude", "latitude", "height"))

        self.assertEqual(result["processing_mode"], "radar_mosaic")
        self.assertEqual(len(topo_calls), 1)
        self.assertTrue(result["mosaic"]["topo_manifest"].endswith("burst_manifests/mosaic/manifest.json"))
        self.assertTrue(has_geometry)

    def test_apply_rtc_multi_burst_exports_merged_geocoded_products(self) -> None:
        import tops_rtc
        from tops_rtc import apply_burst_rtc

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bursts = []
            for idx, (start, value) in enumerate(((1000.0, 10.0), (1002.0, 20.0)), start=1):
                burst_dir = root / f"burst_{idx:03d}"
                burst_dir.mkdir()
                h5_path = burst_dir / "amplitude_fullres.h5"
                with h5py.File(h5_path, "w") as f:
                    f.create_dataset("slc_amplitude", data=np.full((3, 3), value, dtype=np.float32))
                    f.create_dataset("valid_mask", data=np.ones((3, 3), dtype=np.uint8))
                bursts.append(
                    {
                        "burstIndex": idx,
                        "doppler": {"coefficients": [0.0], "t0": 0.0},
                        "radargrid": {
                            "numberOfRows": 3,
                            "numberOfColumns": 3,
                            "sensingStartGPSTime": start,
                            "rowSpacing": 1.0,
                            "rangeTimeFirstPixel": 0.0,
                            "columnSpacing": 10.0,
                            "firstValidLine": 0,
                            "numValidLines": 3,
                            "firstValidSample": 0,
                            "numValidSamples": 3,
                        },
                        "outputs": {
                            "directory": str(burst_dir),
                            "amplitude_h5": str(h5_path),
                            "rtc_factor_tif": str(burst_dir / "rtc_factor.tif"),
                        },
                    }
                )
            factor_path = root / "rtc" / "mosaic_rtc_factor.tif"
            factor_path.parent.mkdir()
            ds = gdal.GetDriverByName("GTiff").Create(str(factor_path), 3, 5, 1, gdal.GDT_Float32)
            ds.GetRasterBand(1).WriteArray(np.ones((5, 3), dtype=np.float32))
            ds = None

            source_manifest = root / "source_manifest.json"
            metadata_dir = root / "source_metadata"
            metadata_dir.mkdir()
            for name in ("acquisition", "orbit", "scene"):
                (metadata_dir / f"{name}.json").write_text(json.dumps({name: True}), encoding="utf-8")
            source_manifest.write_text(
                json.dumps(
                    {
                        "sensor": "sentinel-1",
                        "metadata": {
                            "acquisition": str(metadata_dir / "acquisition.json"),
                            "orbit": str(metadata_dir / "orbit.json"),
                            "scene": str(metadata_dir / "scene.json"),
                        },
                    }
                ),
                encoding="utf-8",
            )
            plan_path = root / "rtc" / "tops_rtc_plan.json"
            plan_path.write_text(json.dumps({"input_manifest": str(source_manifest), "bursts": bursts}), encoding="utf-8")

            def fake_topo(manifest_path, dem_path, output_h5, block_rows=256, orbit_interp=None):
                with h5py.File(output_h5, "a") as f:
                    shape = f["slc_amplitude"].shape
                    f.create_dataset("longitude", data=np.full(shape, 95.0, dtype=np.float32))
                    f.create_dataset("latitude", data=np.full(shape, 29.0, dtype=np.float32))
                    f.create_dataset("height", data=np.full(shape, 100.0, dtype=np.float32))
                return output_h5

            export_calls = []

            def fake_export(h5_path, dataset_name, output_prefix, resolution_meters=20.0, block_rows=64):
                export_calls.append((str(h5_path), dataset_name, str(output_prefix), resolution_meters, block_rows))
                return {
                    "geotiff": f"{output_prefix}.tif",
                    "png": f"{output_prefix}.png",
                    "target_width": 4,
                    "target_height": 5,
                    "resolution_meters": resolution_meters,
                }

            with mock.patch.object(tops_rtc, "append_topo_coordinates_hdf", side_effect=fake_topo):
                with mock.patch.object(tops_rtc, "export_radar_hdf_geocoded", side_effect=fake_export):
                    result = apply_burst_rtc(
                        plan_path,
                        burst_limit=2,
                        dem_path=root / "dem.tif",
                        block_rows=7,
                        orbit_interp="Legendre",
                    )

        self.assertEqual([call[1] for call in export_calls], ["slc_amplitude", "rtc_amplitude"])
        self.assertIn("slc_geocoded", result["mosaic"])
        self.assertIn("rtc_geocoded", result["mosaic"])
        self.assertTrue(result["mosaic"]["slc_geocoded"]["geotiff"].endswith("mosaic_slc_amplitude_geocoded.tif"))
        self.assertTrue(result["mosaic"]["rtc_geocoded"]["png"].endswith("mosaic_rtc_amplitude_geocoded.png"))


    def test_isce2_style_radar_merge_uses_valid_windows_and_top_overlap(self):
        from tops_rtc import merge_bursts_radar_grid

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bursts = []
            for idx, (start, value) in enumerate(((1000.0, 10.0), (1002.0, 20.0)), start=1):
                burst_dir = root / f"burst_{idx:03d}"
                burst_dir.mkdir()
                h5_path = burst_dir / "amplitude_fullres.h5"
                data = np.full((3, 3), value, dtype=np.float32)
                data[:, 0] = value + 1.0
                valid = np.zeros((3, 3), dtype=np.uint8)
                valid[:, 1:] = 1
                lon = np.full((3, 3), 95.0 + idx, dtype=np.float32)
                lat = np.full((3, 3), 29.0 + idx, dtype=np.float32)
                height = np.full((3, 3), 100.0 + idx, dtype=np.float32)
                with h5py.File(h5_path, "w") as f:
                    f.create_dataset("slc_amplitude", data=data)
                    f.create_dataset("valid_mask", data=valid)
                    f.create_dataset("longitude", data=lon)
                    f.create_dataset("latitude", data=lat)
                    f.create_dataset("height", data=height)
                bursts.append(
                    {
                        "burstIndex": idx,
                        "radargrid": {
                            "numberOfRows": 3,
                            "numberOfColumns": 3,
                            "sensingStartGPSTime": start,
                            "rowSpacing": 1.0,
                            "rangeTimeFirstPixel": 0.0,
                            "columnSpacing": 10.0,
                            "firstValidLine": 0,
                            "numValidLines": 3,
                            "firstValidSample": 1,
                            "numValidSamples": 2,
                        },
                        "outputs": {
                            "directory": str(burst_dir),
                            "amplitude_h5": str(h5_path),
                        },
                    }
                )

            merged_path = root / "merged.h5"
            result = merge_bursts_radar_grid(bursts, "slc_amplitude", merged_path)
            with h5py.File(merged_path, "r") as f:
                merged = f["slc_amplitude"][:]
                mask = f["valid_mask"][:]

        self.assertEqual(result["mosaic_source"], "radar")
        self.assertEqual(result["mosaic_shape"], (5, 3))
        np.testing.assert_array_equal(mask[:, 0], np.zeros(5, dtype=np.uint8))
        np.testing.assert_array_equal(mask[:, 1:], np.ones((5, 2), dtype=np.uint8))
        self.assertTrue(np.all(merged[:3, 1:] == 10.0))
        self.assertTrue(np.all(merged[3:, 1:] == 20.0))

    def test_radar_merge_can_skip_per_burst_geometry_for_merged_topo(self):
        from tops_rtc import merge_bursts_radar_grid

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bursts = []
            for idx, (start, value) in enumerate(((1000.0, 10.0), (1002.0, 20.0)), start=1):
                burst_dir = root / f"burst_{idx:03d}"
                burst_dir.mkdir()
                h5_path = burst_dir / "amplitude_fullres.h5"
                with h5py.File(h5_path, "w") as f:
                    f.create_dataset("slc_amplitude", data=np.full((3, 3), value, dtype=np.float32))
                    f.create_dataset("valid_mask", data=np.ones((3, 3), dtype=np.uint8))
                bursts.append(
                    {
                        "burstIndex": idx,
                        "radargrid": {
                            "numberOfRows": 3,
                            "numberOfColumns": 3,
                            "sensingStartGPSTime": start,
                            "rowSpacing": 1.0,
                            "rangeTimeFirstPixel": 0.0,
                            "columnSpacing": 10.0,
                            "firstValidLine": 0,
                            "numValidLines": 3,
                            "firstValidSample": 0,
                            "numValidSamples": 3,
                        },
                        "outputs": {"directory": str(burst_dir), "amplitude_h5": str(h5_path)},
                    }
                )

            merged_path = root / "merged.h5"
            result = merge_bursts_radar_grid(
                bursts,
                "slc_amplitude",
                merged_path,
                include_geometry=False,
            )
            with h5py.File(merged_path, "r") as f:
                datasets = set(f.keys())

        self.assertEqual(result["mosaic_source"], "radar")
        self.assertIn("slc_amplitude", datasets)
        self.assertIn("valid_mask", datasets)
        self.assertNotIn("longitude", datasets)
        self.assertNotIn("latitude", datasets)
        self.assertNotIn("height", datasets)

    def test_radar_merge_converts_range_time_to_meters_for_offsets(self):
        from tops_rtc import merge_bursts_radar_grid

        c = 299792458.0
        dr = 10.0
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bursts = []
            for idx, (range_time, value) in enumerate(((0.004, 10.0), (0.004 + 2.0 * dr / c, 20.0)), start=1):
                burst_dir = root / f"burst_{idx:03d}"
                burst_dir.mkdir()
                h5_path = burst_dir / "amplitude_fullres.h5"
                with h5py.File(h5_path, "w") as f:
                    f.create_dataset("slc_amplitude", data=np.full((2, 2), value, dtype=np.float32))
                    f.create_dataset("valid_mask", data=np.ones((2, 2), dtype=np.uint8))
                    f.create_dataset("longitude", data=np.full((2, 2), 95.0, dtype=np.float32))
                    f.create_dataset("latitude", data=np.full((2, 2), 29.0, dtype=np.float32))
                    f.create_dataset("height", data=np.full((2, 2), 100.0, dtype=np.float32))
                bursts.append(
                    {
                        "burstIndex": idx,
                        "radargrid": {
                            "numberOfRows": 2,
                            "numberOfColumns": 2,
                            "sensingStartGPSTime": 1000.0,
                            "rowSpacing": 1.0,
                            "rangeTimeFirstPixel": range_time,
                            "columnSpacing": dr,
                            "firstValidLine": 0,
                            "numValidLines": 2,
                            "firstValidSample": 0,
                            "numValidSamples": 2,
                        },
                        "outputs": {"directory": str(burst_dir), "amplitude_h5": str(h5_path)},
                    }
                )

            merged_path = root / "merged.h5"
            result = merge_bursts_radar_grid(bursts, "slc_amplitude", merged_path)
            with h5py.File(merged_path, "r") as f:
                merged = f["slc_amplitude"][:]
                mask = f["valid_mask"][:]

        self.assertEqual(result["mosaic_shape"], (2, 3))
        np.testing.assert_array_equal(mask, np.ones((2, 3), dtype=np.uint8))
        self.assertTrue(np.all(merged[:, :2] == 10.0))
        self.assertTrue(np.all(merged[:, 2] == 20.0))


if __name__ == "__main__":
    unittest.main()
