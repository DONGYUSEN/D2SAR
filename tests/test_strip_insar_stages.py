import json
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import strip_insar
from insar_crop import normalize_crop_request
from insar_precheck import run_compatibility_precheck
from insar_stage_cache import resolve_requested_stages, stage_dir
from strip_insar import process_strip_insar, _write_crop_stage


class StripInsarStageHelperTests(unittest.TestCase):
    def test_resolve_requested_stages_supports_step_range_and_resume(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertEqual(resolve_requested_stages(tmpdir, step="prep"), ["prep"])
            self.assertEqual(
                resolve_requested_stages(tmpdir, start_step="check", end_step="prep"),
                ["check", "prep"],
            )
            self.assertEqual(
                resolve_requested_stages(tmpdir, start_step="crop", end_step="p0"),
                ["crop", "p0"],
            )

            check_dir = stage_dir(tmpdir, "check")
            check_dir.mkdir(parents=True, exist_ok=True)
            (check_dir / "stage.json").write_text(
                json.dumps({"stage": "check", "success": True}), encoding="utf-8"
            )
            (check_dir / "SUCCESS").write_text("success\n", encoding="utf-8")
            self.assertEqual(
                resolve_requested_stages(tmpdir, resume=True),
                ["prep", "crop", "p0", "p1", "p2", "p3", "p4", "p5", "p6"],
            )

    def test_normalize_crop_request_validates_window_and_bbox(self):
        radargrid = {"numberOfRows": 100, "numberOfColumns": 200}
        crop = normalize_crop_request(
            bbox=None,
            window=(10, 20, 30, 40),
            radargrid_data=radargrid,
            scene_corners=None,
        )
        self.assertEqual(crop["mode"], "window")
        self.assertEqual(crop["master_window"]["rows"], 30)

        bbox_crop = normalize_crop_request(
            bbox=(100.0, 29.0, 101.0, 30.0),
            window=None,
            radargrid_data=radargrid,
            scene_corners=[
                {"lon": 99.0, "lat": 31.0},
                {"lon": 102.0, "lat": 31.0},
                {"lon": 102.0, "lat": 28.0},
                {"lon": 99.0, "lat": 28.0},
            ],
        )
        self.assertEqual(bbox_crop["mode"], "bbox")
        self.assertGreater(bbox_crop["master_window"]["rows"], 0)
        self.assertGreater(bbox_crop["master_window"]["cols"], 0)

    def test_run_compatibility_precheck_warns_for_prf_mismatch(self):
        precheck = run_compatibility_precheck(
            master_acquisition={
                "centerFrequency": 5405e6,
                "prf": 1000.0,
                "lookDirection": "RIGHT",
                "polarisation": "VV",
                "startGPSTime": 100.0,
            },
            slave_acquisition={
                "centerFrequency": 5405e6,
                "prf": 1020.0,
                "lookDirection": "RIGHT",
                "polarisation": "VV",
                "startGPSTime": 101.0,
            },
            master_radargrid={"numberOfRows": 100, "numberOfColumns": 100, "columnSpacing": 2.0},
            slave_radargrid={"numberOfRows": 100, "numberOfColumns": 100, "columnSpacing": 2.0},
            master_doppler={},
            slave_doppler={},
            dc_policy="auto",
            prf_policy="auto",
            skip_precheck=False,
        )
        self.assertEqual(precheck["checks"]["prf"]["severity"], "warn")
        self.assertEqual(precheck["checks"]["doppler"]["severity"], "warn")
        self.assertTrue(precheck["requires_prep"])

    def test_query_gpu_memory_info_parses_nvidia_smi_csv(self):
        fake_result = mock.Mock()
        fake_result.stdout = "6144, 4096\n"
        with mock.patch("strip_insar.subprocess.run", return_value=fake_result) as mock_run:
            info = strip_insar.query_gpu_memory_info(1)

        mock_run.assert_called_once()
        self.assertEqual(info["total_bytes"], 6144 * 1024 * 1024)
        self.assertEqual(info["free_bytes"], 4096 * 1024 * 1024)


class StripInsarStageProcessTests(unittest.TestCase):
    def _write_manifest(self, tmpdir: str, filename: str, payload: dict) -> Path:
        path = Path(tmpdir) / filename
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_process_strip_insar_supports_check_and_prep_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            master_manifest = self._write_manifest(
                tmpdir,
                "master.json",
                {"sensor": "tianyi", "slc": {"path": "master.tif"}},
            )
            slave_manifest = self._write_manifest(
                tmpdir,
                "slave.json",
                {"sensor": "tianyi", "slc": {"path": "slave.tif"}},
            )
            output_dir = Path(tmpdir) / "out"

            meta = (
                {},
                {"header": {"firstStateTimeUTC": "2024-01-01T00:00:00Z"}},
                {
                    "centerFrequency": 5405e6,
                    "prf": 1000.0,
                    "lookDirection": "RIGHT",
                    "polarisation": "VV",
                    "startGPSTime": 100.0,
                },
                {
                    "numberOfRows": 100,
                    "numberOfColumns": 200,
                    "groundRangeResolution": 2.0,
                    "azimuthResolution": 3.0,
                },
                {},
            )

            with (
                mock.patch("strip_insar.load_scene_corners_with_fallback", return_value=[]),
                mock.patch("strip_insar._load_master_metadata", side_effect=[meta, meta, meta]),
                mock.patch("strip_insar.choose_orbit_interp", return_value="Legendre"),
                mock.patch("strip_insar._resolve_dem_path", return_value="/tmp/dem.tif"),
                mock.patch("strip_insar._write_radar_amplitude_png", side_effect=lambda slc_path, output_png: str(output_png)) as mock_png,
            ):
                result = process_strip_insar(
                    str(master_manifest),
                    str(slave_manifest),
                    str(output_dir),
                    start_step="check",
                    end_step="prep",
                )

            self.assertEqual(result["completed_stages"], ["check", "prep"])
            self.assertEqual(result["precheck"]["overall_severity"], "warn")
            self.assertTrue((output_dir / "work" / "check" / "SUCCESS").exists())
            self.assertTrue((output_dir / "work" / "prep" / "SUCCESS").exists())
            check_record = json.loads((output_dir / "work" / "check" / "stage.json").read_text(encoding="utf-8"))
            self.assertIsNone(check_record["effective_crop"])
            self.assertNotIn("crop", check_record["output_files"])
            self.assertNotIn("master_fullres_png", check_record["output_files"])
            self.assertNotIn("slave_fullres_png", check_record["output_files"])
            self.assertEqual(mock_png.call_count, 0)
            self.assertTrue(
                Path(result["normalized_slave_manifest"]).exists(),
                "prep stage should write normalized slave manifest",
            )

    def test_write_crop_stage_writes_crop_named_manifests_and_previews(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "out"
            master_manifest = self._write_manifest(
                tmpdir,
                "master.json",
                {
                    "sensor": "tianyi",
                    "slc": {"path": "master.tif", "rows": 100, "columns": 200},
                    "metadata": {
                        "acquisition": "metadata/master_acquisition.json",
                        "radargrid": "metadata/master_radargrid.json",
                        "doppler": "metadata/master_doppler.json",
                        "scene": "metadata/master_scene.json",
                        "orbit": "metadata/master_orbit.json",
                    },
                },
            )
            slave_manifest = self._write_manifest(
                tmpdir,
                "normalized_slave_manifest.json",
                {
                    "sensor": "tianyi",
                    "slc": {"path": "slave.tif", "rows": 100, "columns": 200},
                    "metadata": {
                        "acquisition": "metadata/slave_acquisition.json",
                        "radargrid": "metadata/slave_radargrid.json",
                        "doppler": "metadata/slave_doppler.json",
                        "scene": "metadata/slave_scene.json",
                        "orbit": "metadata/slave_orbit.json",
                    },
                    "processing": {
                        "insar_preprocess": {
                            "source_manifest": str(root / "slave.json"),
                            "actions": ["pass-through"],
                            "geometry_mode": "zero-doppler",
                            "normalized_slc": "slave.tif",
                            "resamp_slc": {
                                "source_prf": 1000.0,
                                "target_prf": 1000.0,
                                "target_rows": 100,
                                "target_cols": 200,
                                "geometry_mode": "zero-doppler",
                            },
                        }
                    },
                },
            )

            metadata_dir = root / "metadata"
            metadata_dir.mkdir(parents=True, exist_ok=True)
            for name in (
                "master_acquisition",
                "slave_acquisition",
                "master_doppler",
                "slave_doppler",
                "master_scene",
                "slave_scene",
                "master_orbit",
                "slave_orbit",
            ):
                (metadata_dir / f"{name}.json").write_text("{}", encoding="utf-8")
            (metadata_dir / "master_radargrid.json").write_text(
                json.dumps(
                    {
                        "numberOfRows": 100,
                        "numberOfColumns": 200,
                        "columnSpacing": 2.0,
                        "rangeTimeFirstPixel": 0.001,
                    }
                ),
                encoding="utf-8",
            )
            (metadata_dir / "slave_radargrid.json").write_text(
                json.dumps(
                    {
                        "numberOfRows": 100,
                        "numberOfColumns": 200,
                        "columnSpacing": 2.0,
                        "rangeTimeFirstPixel": 0.001,
                    }
                ),
                encoding="utf-8",
            )
            (root / "master.tif").write_bytes(b"")
            (root / "slave.tif").write_bytes(b"")

            with mock.patch(
                "strip_insar._write_radar_amplitude_png",
                side_effect=lambda slc_path, output_png: str(output_png),
            ) as mock_png:
                crop_outputs = _write_crop_stage(
                    output_dir=output_dir,
                    master_manifest_path=master_manifest,
                    normalized_slave_manifest_path=slave_manifest,
                    crop_request={
                        "mode": "window",
                        "master_window": {"row0": 10, "col0": 20, "rows": 30, "cols": 40},
                        "bbox": None,
                    },
                )

            self.assertEqual(mock_png.call_count, 4)
            self.assertTrue(crop_outputs["master_manifest"].endswith("master.crop.json"))
            self.assertTrue(crop_outputs["slave_manifest"].endswith("slave.crop.json"))

            crop_record = json.loads(
                (output_dir / "work" / "crop" / "stage.json").read_text(encoding="utf-8")
            )
            self.assertIn("master_normal_fullres_png", crop_record["output_files"])
            self.assertIn("slave_normal_fullres_png", crop_record["output_files"])
            self.assertIn("master_crop_fullres_png", crop_record["output_files"])
            self.assertIn("slave_crop_fullres_png", crop_record["output_files"])
            self.assertIn("master_crop_manifest", crop_record["output_files"])
            self.assertIn("slave_crop_manifest", crop_record["output_files"])

            cropped_master_manifest = json.loads(
                Path(crop_outputs["master_manifest"]).read_text(encoding="utf-8")
            )
            cropped_slave_manifest = json.loads(
                Path(crop_outputs["slave_manifest"]).read_text(encoding="utf-8")
            )

            self.assertEqual(cropped_master_manifest["slc"]["rows"], 30)
            self.assertEqual(cropped_master_manifest["slc"]["columns"], 40)
            self.assertEqual(cropped_slave_manifest["slc"]["rows"], 30)
            self.assertEqual(cropped_slave_manifest["slc"]["columns"], 40)
            self.assertEqual(cropped_master_manifest["slc"]["path"], "master.crop.tif")
            self.assertEqual(cropped_slave_manifest["slc"]["path"], "slave.crop.tif")
            self.assertEqual(
                cropped_master_manifest["metadata"]["radargrid"],
                "master.crop.radargrid.json",
            )
            self.assertEqual(
                cropped_slave_manifest["metadata"]["radargrid"],
                "slave.crop.radargrid.json",
            )
            self.assertIn("insar_crop", cropped_master_manifest["processing"])
            self.assertIn("insar_crop", cropped_slave_manifest["processing"])
            self.assertEqual(
                cropped_slave_manifest["processing"]["insar_crop"]["master_window"],
                {"row0": 10, "col0": 20, "rows": 30, "cols": 40},
            )

    def test_process_strip_insar_supports_crop_only_after_prep(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            master_manifest = self._write_manifest(
                tmpdir,
                "master.json",
                {"sensor": "tianyi", "slc": {"path": "master.tif"}},
            )
            slave_manifest = self._write_manifest(
                tmpdir,
                "slave.json",
                {"sensor": "tianyi", "slc": {"path": "slave.tif"}},
            )
            output_dir = Path(tmpdir) / "out"
            meta = (
                {},
                {"header": {"firstStateTimeUTC": "2024-01-01T00:00:00Z"}},
                {
                    "centerFrequency": 5405e6,
                    "prf": 1000.0,
                    "lookDirection": "RIGHT",
                    "polarisation": "VV",
                    "startGPSTime": 100.0,
                },
                {
                    "numberOfRows": 100,
                    "numberOfColumns": 200,
                    "groundRangeResolution": 2.0,
                    "azimuthResolution": 3.0,
                    "columnSpacing": 2.0,
                    "rangeTimeFirstPixel": 0.001,
                },
                {},
            )

            with (
                mock.patch("strip_insar.load_scene_corners_with_fallback", return_value=[]),
                mock.patch("strip_insar._load_master_metadata", side_effect=[meta, meta, meta]),
                mock.patch("strip_insar.choose_orbit_interp", return_value="Legendre"),
                mock.patch("strip_insar._resolve_dem_path", return_value="/tmp/dem.tif"),
                mock.patch("strip_insar._write_crop_stage") as mock_crop,
            ):
                mock_crop.return_value = {
                    "master_manifest": str(output_dir / "work" / "crop" / "master.crop.json"),
                    "slave_manifest": str(output_dir / "work" / "crop" / "slave.crop.json"),
                    "crop_request": {
                        "mode": "window",
                        "master_window": {"row0": 10, "col0": 20, "rows": 30, "cols": 40},
                        "bbox": None,
                    },
                }
                result = process_strip_insar(
                    str(master_manifest),
                    str(slave_manifest),
                    str(output_dir),
                    step="crop",
                    window=(10, 20, 30, 40),
                )

            mock_crop.assert_called_once()
            self.assertEqual(result["completed_stages"], ["crop"])
            self.assertIn("cropped_master_manifest", result)
            self.assertIn("cropped_slave_manifest", result)

    def test_process_strip_insar_step_p6_reuses_cached_p5_hdf(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            master_manifest = self._write_manifest(
                tmpdir,
                "master.json",
                {"sensor": "tianyi", "slc": {"path": "master.tif"}},
            )
            slave_manifest = self._write_manifest(
                tmpdir,
                "slave.json",
                {"sensor": "tianyi", "slc": {"path": "slave.tif"}},
            )
            output_dir = Path(tmpdir) / "out"
            p5_dir = output_dir / "work" / "p5_hdf"
            p5_dir.mkdir(parents=True, exist_ok=True)
            h5_path = output_dir / "interferogram_fullres.h5"
            h5_path.write_text("stub", encoding="utf-8")
            (p5_dir / "stage.json").write_text(
                json.dumps(
                    {
                        "stage": "p5",
                        "success": True,
                        "effective_crop": {"mode": "full", "master_window": {"row0": 0, "col0": 0, "rows": 100, "cols": 200}, "bbox": None},
                        "output_files": {"interferogram_h5": str(h5_path)},
                    }
                ),
                encoding="utf-8",
            )
            (p5_dir / "SUCCESS").write_text("success\n", encoding="utf-8")

            meta = (
                {},
                {"header": {"firstStateTimeUTC": "2024-01-01T00:00:00Z"}},
                {
                    "centerFrequency": 5405e6,
                    "prf": 1000.0,
                    "lookDirection": "RIGHT",
                    "polarisation": "VV",
                    "startGPSTime": 100.0,
                },
                {
                    "numberOfRows": 100,
                    "numberOfColumns": 200,
                    "groundRangeResolution": 2.0,
                    "azimuthResolution": 3.0,
                },
                {},
            )

            with (
                mock.patch("strip_insar._load_master_metadata", side_effect=[meta, meta, meta]),
                mock.patch("strip_insar.choose_orbit_interp", return_value="Legendre"),
                mock.patch("strip_insar.compute_utm_output_shape", return_value=(32, 16)),
                mock.patch("strip_insar.geocode_product") as mock_geocode,
                mock.patch("strip_insar.write_wrapped_phase_geotiff") as mock_ifg_tif,
                mock.patch("strip_insar.write_wrapped_phase_png") as mock_png,
                mock.patch("strip_insar._process_insar_cpu") as mock_cpu,
                mock.patch("strip_insar._process_insar_gpu") as mock_gpu,
            ):
                result = process_strip_insar(
                    str(master_manifest),
                    str(slave_manifest),
                    str(output_dir),
                    step="p6",
                )

            self.assertEqual(mock_geocode.call_count, 3)
            mock_ifg_tif.assert_called_once()
            self.assertEqual(mock_png.call_count, 2)
            mock_cpu.assert_not_called()
            mock_gpu.assert_not_called()
            self.assertEqual(result["completed_stages"], ["p6"])
            self.assertEqual(result["backend_used"], "cpu")
            self.assertTrue((output_dir / "work" / "p6_publish" / "SUCCESS").exists())

    def test_process_strip_insar_resume_from_p6_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "out"
            for stage_name, dir_name in [
                ("check", "check"),
                ("prep", "prep"),
                ("crop", "crop"),
                ("p0", "p0_geo2rdr"),
                ("p1", "p1_dense_match"),
                ("p2", "p2_crossmul"),
                ("p3", "p3_unwrap"),
                ("p4", "p4_geocode"),
                ("p5", "p5_hdf"),
            ]:
                stage_path = output_dir / "work" / dir_name
                stage_path.mkdir(parents=True, exist_ok=True)
                record = {"stage": stage_name, "success": True, "output_files": {}}
                if stage_name == "p5":
                    h5_path = output_dir / "interferogram_fullres.h5"
                    h5_path.write_text("stub", encoding="utf-8")
                    record["output_files"] = {"interferogram_h5": str(h5_path)}
                (stage_path / "stage.json").write_text(json.dumps(record), encoding="utf-8")
                (stage_path / "SUCCESS").write_text("success\n", encoding="utf-8")

            master_manifest = self._write_manifest(
                tmpdir,
                "master.json",
                {"sensor": "tianyi", "slc": {"path": "master.tif"}},
            )
            slave_manifest = self._write_manifest(
                tmpdir,
                "slave.json",
                {"sensor": "tianyi", "slc": {"path": "slave.tif"}},
            )

            meta = (
                {},
                {"header": {"firstStateTimeUTC": "2024-01-01T00:00:00Z"}},
                {
                    "centerFrequency": 5405e6,
                    "prf": 1000.0,
                    "lookDirection": "RIGHT",
                    "polarisation": "VV",
                    "startGPSTime": 100.0,
                },
                {
                    "numberOfRows": 100,
                    "numberOfColumns": 200,
                    "groundRangeResolution": 2.0,
                    "azimuthResolution": 3.0,
                },
                {},
            )

            with (
                mock.patch("strip_insar._load_master_metadata", side_effect=[meta, meta, meta]),
                mock.patch("strip_insar.choose_orbit_interp", return_value="Legendre"),
                mock.patch("strip_insar.compute_utm_output_shape", return_value=(32, 16)),
                mock.patch("strip_insar.geocode_product") as mock_geocode,
                mock.patch("strip_insar.write_wrapped_phase_geotiff") as mock_ifg_tif,
                mock.patch("strip_insar.write_wrapped_phase_png") as mock_png,
            ):
                result = process_strip_insar(
                    str(master_manifest),
                    str(slave_manifest),
                    str(output_dir),
                    resume=True,
                )

            self.assertEqual(result["completed_stages"], ["p6"])
            self.assertEqual(mock_geocode.call_count, 3)
            mock_ifg_tif.assert_called_once()
            self.assertEqual(mock_png.call_count, 2)

    def test_process_strip_insar_runs_publish_when_filtered_png_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            master_manifest = self._write_manifest(
                tmpdir,
                "master.json",
                {"sensor": "tianyi", "slc": {"path": "master.tif"}},
            )
            slave_manifest = self._write_manifest(
                tmpdir,
                "slave.json",
                {"sensor": "tianyi", "slc": {"path": "slave.tif"}},
            )
            output_dir = Path(tmpdir) / "out"
            h5_path = output_dir / "interferogram_fullres.h5"

            meta = (
                {},
                {"header": {"firstStateTimeUTC": "2024-01-01T00:00:00Z"}},
                {
                    "centerFrequency": 5405e6,
                    "prf": 1000.0,
                    "lookDirection": "RIGHT",
                    "polarisation": "VV",
                    "startGPSTime": 100.0,
                },
                {
                    "numberOfRows": 100,
                    "numberOfColumns": 200,
                    "groundRangeResolution": 2.0,
                    "azimuthResolution": 3.0,
                },
                {},
            )

            gpu_outputs = {
                "interferogram_h5": str(h5_path),
                "interferogram_tif": str(output_dir / "interferogram_utm_geocoded.tif"),
                "coherence_tif": str(output_dir / "coherence_utm_geocoded.tif"),
                "unwrapped_phase_tif": str(output_dir / "unwrapped_phase_utm_geocoded.tif"),
                "los_displacement_tif": str(output_dir / "los_displacement_utm_geocoded.tif"),
                "interferogram_png": str(output_dir / "interferogram_utm_geocoded.png"),
                "filtered_interferogram_png": None,
            }

            with (
                mock.patch("strip_insar.load_scene_corners_with_fallback", return_value=[]),
                mock.patch("strip_insar._load_master_metadata", side_effect=[meta, meta, meta]),
                mock.patch("strip_insar.choose_orbit_interp", return_value="Legendre"),
                mock.patch("strip_insar._resolve_dem_path", return_value="/tmp/dem.tif"),
                mock.patch(
                    "strip_insar.select_processing_backend",
                    return_value=("gpu", "GPU available"),
                ),
                mock.patch("strip_insar.query_gpu_memory_info", return_value=None),
                mock.patch("strip_insar._process_insar_gpu", return_value=gpu_outputs),
                mock.patch("strip_insar.compute_utm_output_shape", return_value=(32, 16)),
                mock.patch("strip_insar.geocode_product") as mock_geocode,
                mock.patch("strip_insar.write_wrapped_phase_geotiff") as mock_ifg_tif,
                mock.patch("strip_insar.write_wrapped_phase_png") as mock_png,
            ):
                result = process_strip_insar(
                    str(master_manifest),
                    str(slave_manifest),
                    str(output_dir),
                    gpu_mode="gpu",
                )

            self.assertEqual(mock_geocode.call_count, 3)
            mock_ifg_tif.assert_called_once()
            self.assertEqual(mock_png.call_count, 2)
            self.assertTrue((output_dir / "work" / "p6_publish" / "SUCCESS").exists())
            self.assertEqual(
                result["filtered_interferogram_png"],
                str(output_dir / "filtered_interferogram_utm_geocoded.png"),
            )

    def test_process_strip_insar_step_p5_reuses_cached_arrays(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            import numpy as np

            output_dir = Path(tmpdir) / "out"
            for stage_name, dir_name in [
                ("check", "check"),
                ("prep", "prep"),
                ("crop", "crop"),
                ("p0", "p0_geo2rdr"),
                ("p1", "p1_dense_match"),
                ("p2", "p2_crossmul"),
                ("p3", "p3_unwrap"),
                ("p4", "p4_geocode"),
            ]:
                stage_path = output_dir / "work" / dir_name
                stage_path.mkdir(parents=True, exist_ok=True)
                record = {"stage": stage_name, "success": True, "output_files": {}}
                if stage_name == "p2":
                    igram = stage_path / "interferogram.npy"
                    coh = stage_path / "coherence.npy"
                    np.save(igram, np.ones((2, 2), dtype=np.complex64))
                    np.save(coh, np.ones((2, 2), dtype=np.float32))
                    record["output_files"] = {
                        "interferogram": str(igram),
                        "coherence": str(coh),
                    }
                elif stage_name == "p3":
                    unw = stage_path / "unwrapped_phase.npy"
                    np.save(unw, np.ones((2, 2), dtype=np.float32))
                    record["output_files"] = {"unwrapped_phase": str(unw)}
                elif stage_name == "p4":
                    los = stage_path / "los_displacement.npy"
                    np.save(los, np.ones((2, 2), dtype=np.float32))
                    record["output_files"] = {"los_displacement": str(los)}
                (stage_path / "stage.json").write_text(json.dumps(record), encoding="utf-8")
                (stage_path / "SUCCESS").write_text("success\n", encoding="utf-8")

            master_manifest = self._write_manifest(
                tmpdir,
                "master.json",
                {"sensor": "tianyi", "slc": {"path": "master.tif"}},
            )
            slave_manifest = self._write_manifest(
                tmpdir,
                "slave.json",
                {"sensor": "tianyi", "slc": {"path": "slave.tif"}},
            )
            meta = (
                {},
                {"header": {"firstStateTimeUTC": "2024-01-01T00:00:00Z"}},
                {
                    "centerFrequency": 5405e6,
                    "prf": 1000.0,
                    "lookDirection": "RIGHT",
                    "polarisation": "VV",
                    "startGPSTime": 100.0,
                },
                {
                    "numberOfRows": 100,
                    "numberOfColumns": 200,
                    "groundRangeResolution": 2.0,
                    "azimuthResolution": 3.0,
                },
                {},
            )

            with (
                mock.patch("strip_insar._load_master_metadata", side_effect=[meta, meta, meta]),
                mock.patch("strip_insar.choose_orbit_interp", return_value="Legendre"),
                mock.patch("strip_insar.resolve_manifest_data_path", side_effect=lambda manifest_path, rel: rel),
                mock.patch("strip_insar.construct_radar_grid", return_value=object()),
                mock.patch("strip_insar.write_insar_hdf") as mock_hdf,
                mock.patch("strip_insar.append_topo_coordinates_hdf") as mock_append_topo,
                mock.patch("strip_insar.append_utm_coordinates_hdf") as mock_append_utm,
                mock.patch("strip_insar._process_insar_cpu") as mock_cpu,
                mock.patch("strip_insar._process_insar_gpu") as mock_gpu,
            ):
                result = process_strip_insar(
                    str(master_manifest),
                    str(slave_manifest),
                    str(output_dir),
                    step="p5",
                )

            mock_hdf.assert_called_once()
            mock_append_topo.assert_called_once()
            mock_append_utm.assert_called_once()
            mock_cpu.assert_not_called()
            mock_gpu.assert_not_called()
            self.assertEqual(result["completed_stages"], ["p5"])
            self.assertTrue((output_dir / "work" / "p5_hdf" / "SUCCESS").exists())

    def test_process_strip_insar_step_p2_reuses_cached_p1(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            import numpy as np

            output_dir = Path(tmpdir) / "out"
            for stage_name, dir_name in [
                ("check", "check"),
                ("prep", "prep"),
                ("crop", "crop"),
                ("p0", "p0_geo2rdr"),
                ("p1", "p1_dense_match"),
            ]:
                stage_path = output_dir / "work" / dir_name
                stage_path.mkdir(parents=True, exist_ok=True)
                record = {"stage": stage_name, "success": True, "output_files": {}}
                if stage_name == "p1":
                    offsets = stage_path / "offsets.json"
                    offsets.write_text(json.dumps({"row_offset": None, "col_offset": None}), encoding="utf-8")
                    record["output_files"] = {"offsets": str(offsets)}
                (stage_path / "stage.json").write_text(json.dumps(record), encoding="utf-8")
                (stage_path / "SUCCESS").write_text("success\n", encoding="utf-8")

            master_manifest = self._write_manifest(tmpdir, "master.json", {"sensor": "tianyi", "slc": {"path": "master.tif"}})
            slave_manifest = self._write_manifest(tmpdir, "slave.json", {"sensor": "tianyi", "slc": {"path": "slave.tif"}})
            meta = (
                {},
                {"header": {"firstStateTimeUTC": "2024-01-01T00:00:00Z"}},
                {"centerFrequency": 5405e6, "prf": 1000.0, "lookDirection": "RIGHT", "polarisation": "VV", "startGPSTime": 100.0},
                {"numberOfRows": 100, "numberOfColumns": 200, "groundRangeResolution": 2.0, "azimuthResolution": 3.0},
                {},
            )

            with (
                mock.patch("strip_insar.load_scene_corners_with_fallback", return_value=[]),
                mock.patch("strip_insar._load_master_metadata", side_effect=[meta, meta, meta]),
                mock.patch("strip_insar.choose_orbit_interp", return_value="Legendre"),
                mock.patch("strip_insar._resolve_dem_path", return_value="/tmp/dem.tif"),
                mock.patch("strip_insar._run_crossmul", return_value=(np.ones((100, 200), dtype=np.complex64), np.ones((100, 200), dtype=np.float32))) as mock_crossmul,
                mock.patch("strip_insar._process_insar_cpu") as mock_cpu,
                mock.patch("strip_insar._process_insar_gpu") as mock_gpu,
            ):
                result = process_strip_insar(
                    str(master_manifest),
                    str(slave_manifest),
                    str(output_dir),
                    step="p2",
                    window=(10, 20, 30, 40),
                )

            mock_crossmul.assert_called_once()
            mock_cpu.assert_not_called()
            mock_gpu.assert_not_called()
            self.assertEqual(result["completed_stages"], ["p2"])
            self.assertTrue((output_dir / "work" / "p2_crossmul" / "SUCCESS").exists())
            cropped_ifg = np.load(output_dir / "work" / "p2_crossmul" / "interferogram.npy")
            self.assertEqual(cropped_ifg.shape, (30, 40))

    def test_process_strip_insar_step_p0_runs_geo2rdr_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "out"
            master_manifest = self._write_manifest(tmpdir, "master.json", {"sensor": "tianyi", "slc": {"path": "master.tif"}})
            slave_manifest = self._write_manifest(tmpdir, "slave.json", {"sensor": "tianyi", "slc": {"path": "slave.tif"}})
            meta = (
                {},
                {"header": {"firstStateTimeUTC": "2024-01-01T00:00:00Z"}},
                {"centerFrequency": 5405e6, "prf": 1000.0, "lookDirection": "RIGHT", "polarisation": "VV", "startGPSTime": 100.0},
                {"numberOfRows": 100, "numberOfColumns": 200, "groundRangeResolution": 2.0, "azimuthResolution": 3.0},
                {},
            )

            with (
                mock.patch("strip_insar.load_scene_corners_with_fallback", return_value=[]),
                mock.patch("strip_insar._load_master_metadata", side_effect=[meta, meta, meta]),
                mock.patch("strip_insar.choose_orbit_interp", return_value="Legendre"),
                mock.patch("strip_insar._resolve_dem_path", return_value="/tmp/dem.tif"),
                mock.patch("strip_insar._run_geo2rdr", return_value=("/tmp/master_topo.tif", "/tmp/slave_topo.tif")) as mock_geo2rdr,
                mock.patch("strip_insar._process_insar_cpu") as mock_cpu,
            ):
                result = process_strip_insar(str(master_manifest), str(slave_manifest), str(output_dir), step="p0")

            mock_geo2rdr.assert_called_once()
            mock_cpu.assert_not_called()
            self.assertEqual(result["completed_stages"], ["p0"])
            self.assertTrue((output_dir / "work" / "p0_geo2rdr" / "SUCCESS").exists())

    def test_process_strip_insar_range_p0_to_p1_runs_front_stages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "out"
            master_manifest = self._write_manifest(tmpdir, "master.json", {"sensor": "tianyi", "slc": {"path": "master.tif"}})
            slave_manifest = self._write_manifest(tmpdir, "slave.json", {"sensor": "tianyi", "slc": {"path": "slave.tif"}})
            meta = (
                {},
                {"header": {"firstStateTimeUTC": "2024-01-01T00:00:00Z"}},
                {"centerFrequency": 5405e6, "prf": 1000.0, "lookDirection": "RIGHT", "polarisation": "VV", "startGPSTime": 100.0},
                {"numberOfRows": 100, "numberOfColumns": 200, "groundRangeResolution": 2.0, "azimuthResolution": 3.0},
                {},
            )

            with (
                mock.patch("strip_insar.load_scene_corners_with_fallback", return_value=[]),
                mock.patch("strip_insar._load_master_metadata", side_effect=[meta, meta, meta]),
                mock.patch("strip_insar.choose_orbit_interp", return_value="Legendre"),
                mock.patch("strip_insar._resolve_dem_path", return_value="/tmp/dem.tif"),
                mock.patch("strip_insar.select_processing_backend", return_value=("gpu", "GPU available")),
                mock.patch("strip_insar._run_geo2rdr", return_value=("/tmp/master_topo.tif", "/tmp/slave_topo.tif")) as mock_geo2rdr,
                mock.patch("strip_insar._run_pycuampcor", side_effect=NotImplementedError("skip")) as mock_ampcor,
                mock.patch("strip_insar._write_radar_amplitude_png", side_effect=lambda slc_path, output_png: str(output_png)) as mock_png,
                mock.patch(
                    "strip_insar.write_registration_outputs",
                    return_value={
                        "coarse_coreg_slave": str(output_dir / "work" / "p1_dense_match" / "coarse_coreg_slave.tif"),
                        "fine_coreg_slave": str(output_dir / "work" / "p1_dense_match" / "fine_coreg_slave.tif"),
                        "registration_model": str(output_dir / "work" / "p1_dense_match" / "registration_model.json"),
                        "range_offsets": str(output_dir / "work" / "p1_dense_match" / "range.off.tif"),
                        "azimuth_offsets": str(output_dir / "work" / "p1_dense_match" / "azimuth.off.tif"),
                    },
                ) as mock_write_reg,
            ):
                result = process_strip_insar(
                    str(master_manifest),
                    str(slave_manifest),
                    str(output_dir),
                    start_step="p0",
                    end_step="p1",
                    gpu_mode="gpu",
                )

            mock_geo2rdr.assert_called_once()
            mock_ampcor.assert_called_once()
            mock_write_reg.assert_called_once()
            self.assertEqual(result["completed_stages"], ["p0", "p1"])
            self.assertTrue((output_dir / "work" / "p1_dense_match" / "SUCCESS").exists())
            p1_record = json.loads((output_dir / "work" / "p1_dense_match" / "stage.json").read_text(encoding="utf-8"))
            self.assertIn("coarse_coreg_slave", p1_record["output_files"])
            self.assertIn("fine_coreg_slave", p1_record["output_files"])
            self.assertIn("registration_model", p1_record["output_files"])
            self.assertIn("range_offsets", p1_record["output_files"])
            self.assertIn("azimuth_offsets", p1_record["output_files"])
            self.assertIn("coarse_coreg_slave_png", p1_record["output_files"])
            self.assertIn("fine_coreg_slave_png", p1_record["output_files"])
            self.assertGreaterEqual(mock_png.call_count, 4)

    def test_process_strip_insar_range_check_to_p2_runs_front_pipeline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            import numpy as np

            output_dir = Path(tmpdir) / "out"
            master_manifest = self._write_manifest(tmpdir, "master.json", {"sensor": "tianyi", "slc": {"path": "master.tif"}})
            slave_manifest = self._write_manifest(tmpdir, "slave.json", {"sensor": "tianyi", "slc": {"path": "slave.tif"}})
            meta = (
                {},
                {"header": {"firstStateTimeUTC": "2024-01-01T00:00:00Z"}},
                {"centerFrequency": 5405e6, "prf": 1000.0, "lookDirection": "RIGHT", "polarisation": "VV", "startGPSTime": 100.0},
                {"numberOfRows": 100, "numberOfColumns": 200, "groundRangeResolution": 2.0, "azimuthResolution": 3.0},
                {},
            )

            with (
                mock.patch("strip_insar.load_scene_corners_with_fallback", return_value=[]),
                mock.patch("strip_insar._load_master_metadata", side_effect=[meta, meta, meta, meta]),
                mock.patch("strip_insar.choose_orbit_interp", return_value="Legendre"),
                mock.patch("strip_insar._resolve_dem_path", return_value="/tmp/dem.tif"),
                mock.patch("strip_insar.select_processing_backend", return_value=("gpu", "GPU available")),
                mock.patch("strip_insar._run_geo2rdr", return_value=("/tmp/master_topo.tif", "/tmp/slave_topo.tif")) as mock_geo2rdr,
                mock.patch("strip_insar._run_pycuampcor", side_effect=NotImplementedError("skip")) as mock_ampcor,
                mock.patch(
                    "strip_insar.write_registration_outputs",
                    return_value={
                        "coarse_coreg_slave": str(output_dir / "work" / "p1_dense_match" / "coarse_coreg_slave.tif"),
                        "fine_coreg_slave": str(output_dir / "work" / "p1_dense_match" / "fine_coreg_slave.tif"),
                        "registration_model": str(output_dir / "work" / "p1_dense_match" / "registration_model.json"),
                        "range_offsets": str(output_dir / "work" / "p1_dense_match" / "range.off.tif"),
                        "azimuth_offsets": str(output_dir / "work" / "p1_dense_match" / "azimuth.off.tif"),
                    },
                ),
                mock.patch(
                    "strip_insar._run_crossmul",
                    return_value=(np.ones((100, 200), dtype=np.complex64), np.ones((100, 200), dtype=np.float32)),
                ) as mock_crossmul,
                mock.patch("strip_insar._write_radar_amplitude_png", side_effect=lambda slc_path, output_png: str(output_png)) as mock_png,
                mock.patch("strip_insar._write_radar_wrapped_phase_png", return_value=str(output_dir / "wrapped_phase_radar.png")) as mock_phase_png,
            ):
                result = process_strip_insar(
                    str(master_manifest),
                    str(slave_manifest),
                    str(output_dir),
                    start_step="check",
                    end_step="p2",
                    gpu_mode="gpu",
                )

            mock_geo2rdr.assert_called_once()
            mock_ampcor.assert_called_once()
            mock_crossmul.assert_called_once()
            mock_phase_png.assert_called_once()
            self.assertEqual(result["completed_stages"], ["check", "prep", "crop", "p0", "p1", "p2"])

    def test_process_strip_insar_step_p2_prefers_fine_coreg_slave_from_p1(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            import numpy as np

            output_dir = Path(tmpdir) / "out"
            for stage_name, dir_name in [
                ("check", "check"),
                ("prep", "prep"),
                ("crop", "crop"),
                ("p0", "p0_geo2rdr"),
                ("p1", "p1_dense_match"),
            ]:
                stage_path = output_dir / "work" / dir_name
                stage_path.mkdir(parents=True, exist_ok=True)
                record = {"stage": stage_name, "success": True, "output_files": {}}
                if stage_name == "p1":
                    fine_coreg = stage_path / "fine_coreg_slave.tif"
                    fine_coreg.write_text("stub", encoding="utf-8")
                    coarse_coreg = stage_path / "coarse_coreg_slave.tif"
                    coarse_coreg.write_text("stub", encoding="utf-8")
                    record["output_files"] = {
                        "fine_coreg_slave": str(fine_coreg),
                        "coarse_coreg_slave": str(coarse_coreg),
                    }
                (stage_path / "stage.json").write_text(json.dumps(record), encoding="utf-8")
                (stage_path / "SUCCESS").write_text("success\n", encoding="utf-8")

            master_manifest = self._write_manifest(tmpdir, "master.json", {"sensor": "tianyi", "slc": {"path": "master.tif"}})
            slave_manifest = self._write_manifest(tmpdir, "slave.json", {"sensor": "tianyi", "slc": {"path": "slave.tif"}})
            meta = (
                {},
                {"header": {"firstStateTimeUTC": "2024-01-01T00:00:00Z"}},
                {"centerFrequency": 5405e6, "prf": 1000.0, "lookDirection": "RIGHT", "polarisation": "VV", "startGPSTime": 100.0},
                {"numberOfRows": 100, "numberOfColumns": 200, "groundRangeResolution": 2.0, "azimuthResolution": 3.0},
                {},
            )

            with (
                mock.patch("strip_insar.load_scene_corners_with_fallback", return_value=[]),
                mock.patch("strip_insar._load_master_metadata", side_effect=[meta, meta, meta]),
                mock.patch("strip_insar.choose_orbit_interp", return_value="Legendre"),
                mock.patch("strip_insar._resolve_dem_path", return_value="/tmp/dem.tif"),
                mock.patch("strip_insar._run_crossmul", return_value=(np.ones((100, 200), dtype=np.complex64), np.ones((100, 200), dtype=np.float32))) as mock_crossmul,
                mock.patch("strip_insar._write_radar_amplitude_png", side_effect=lambda slc_path, output_png: str(output_png)),
            ):
                process_strip_insar(
                    str(master_manifest),
                    str(slave_manifest),
                    str(output_dir),
                    step="p2",
                )

            self.assertEqual(mock_crossmul.call_args.args[1], str(output_dir / "work" / "p1_dense_match" / "fine_coreg_slave.tif"))

    def test_run_p3_stage_from_cache_uses_filtered_interferogram(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "out"
            p2_dir = output_dir / "work" / "p2_crossmul"
            p2_dir.mkdir(parents=True, exist_ok=True)

            interferogram = np.full((4, 5), 1 + 2j, dtype=np.complex64)
            filtered = np.full((4, 5), 3 + 4j, dtype=np.complex64)
            coherence = np.full((4, 5), 0.6, dtype=np.float32)
            np.save(p2_dir / "interferogram.npy", interferogram)
            np.save(p2_dir / "filtered_interferogram.npy", filtered)
            np.save(p2_dir / "coherence.npy", coherence)
            (p2_dir / "stage.json").write_text(
                json.dumps(
                    {
                        "stage": "p2",
                        "success": True,
                        "output_files": {
                            "interferogram": str(p2_dir / "interferogram.npy"),
                            "filtered_interferogram": str(p2_dir / "filtered_interferogram.npy"),
                            "coherence": str(p2_dir / "coherence.npy"),
                        },
                    }
                ),
                encoding="utf-8",
            )
            (p2_dir / "SUCCESS").write_text("success\n", encoding="utf-8")

            fake_unwrapper = mock.Mock()
            fake_unwrapper.unwrap.return_value = np.ones((4, 5), dtype=np.float32)

            with (
                mock.patch("strip_insar.construct_radar_grid", return_value=object()),
                mock.patch("strip_insar.construct_orbit", return_value=object()),
            ):
                result = strip_insar._run_p3_stage_from_cache(
                    output_dir=output_dir,
                    master_manifest_path=Path("master.json"),
                    slave_manifest_path=Path("slave.json"),
                    resolved_dem="/tmp/dem.tif",
                    orbit_interp="Legendre",
                    unwrapper=fake_unwrapper,
                    crop_request={
                        "mode": "full",
                        "master_window": {"row0": 0, "col0": 0, "rows": 4, "cols": 5},
                        "bbox": None,
                    },
                    block_rows=128,
                    backend_used="cpu",
                    master_orbit_data={},
                    master_acq_data={},
                    master_rg_data={},
                )

            unwrap_ifg = fake_unwrapper.unwrap.call_args.args[0]
            np.testing.assert_allclose(unwrap_ifg, filtered)
            self.assertEqual(
                result["unwrapped_phase"],
                str(output_dir / "work" / "p3_unwrap" / "unwrapped_phase.npy"),
            )

    def test_process_strip_insar_full_pipeline_prefers_fine_coreg_slave_from_p1(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            import numpy as np

            output_dir = Path(tmpdir) / "out"
            master_manifest = self._write_manifest(tmpdir, "master.json", {"sensor": "tianyi", "slc": {"path": "master.tif"}})
            slave_manifest = self._write_manifest(tmpdir, "slave.json", {"sensor": "tianyi", "slc": {"path": "slave.tif"}})
            fine_coreg = output_dir / "work" / "p1_dense_match" / "fine_coreg_slave.tif"
            coarse_coreg = output_dir / "work" / "p1_dense_match" / "coarse_coreg_slave.tif"

            meta = (
                {},
                {"header": {"firstStateTimeUTC": "2024-01-01T00:00:00Z"}},
                {"centerFrequency": 5405e6, "prf": 1000.0, "lookDirection": "RIGHT", "polarisation": "VV", "startGPSTime": 100.0},
                {"numberOfRows": 100, "numberOfColumns": 200, "groundRangeResolution": 2.0, "azimuthResolution": 3.0},
                {},
            )

            with ExitStack() as stack:
                stack.enter_context(mock.patch("strip_insar.load_scene_corners_with_fallback", return_value=[]))
                stack.enter_context(mock.patch("strip_insar._load_master_metadata", side_effect=[meta, meta, meta]))
                stack.enter_context(mock.patch("strip_insar.choose_orbit_interp", return_value="Legendre"))
                stack.enter_context(mock.patch("strip_insar._resolve_dem_path", return_value="/tmp/dem.tif"))
                stack.enter_context(mock.patch("strip_insar.select_processing_backend", return_value=("cpu", "CPU forced")))
                stack.enter_context(mock.patch("strip_insar._run_geo2rdr", return_value=("/tmp/master_topo.tif", "/tmp/slave_topo.tif")))
                stack.enter_context(
                    mock.patch(
                        "strip_insar._run_p1_stage_from_cache",
                        return_value={
                            "coarse_coreg_slave": str(coarse_coreg),
                            "fine_coreg_slave": str(fine_coreg),
                        },
                    )
                )
                mock_crossmul = stack.enter_context(
                    mock.patch(
                        "strip_insar._run_crossmul",
                        return_value=(np.ones((100, 200), dtype=np.complex64), np.ones((100, 200), dtype=np.float32)),
                    )
                )
                stack.enter_context(mock.patch("strip_insar._write_radar_amplitude_png", side_effect=lambda slc_path, output_png: str(output_png)))
                stack.enter_context(mock.patch("strip_insar.construct_radar_grid", return_value=object()))
                stack.enter_context(mock.patch("strip_insar.construct_orbit", return_value=object()))
                mock_tmpdir = stack.enter_context(mock.patch("tempfile.TemporaryDirectory"))
                stack.enter_context(mock.patch("strip_insar.write_insar_hdf", return_value=str(output_dir / "interferogram_fullres.h5")))
                stack.enter_context(mock.patch("strip_insar.append_topo_coordinates_hdf"))
                stack.enter_context(mock.patch("strip_insar.append_utm_coordinates_hdf"))
                stack.enter_context(mock.patch("strip_insar.compute_utm_output_shape", return_value=(32, 16)))
                stack.enter_context(mock.patch("strip_insar.geocode_product"))
                stack.enter_context(mock.patch("strip_insar.write_wrapped_phase_geotiff"))
                stack.enter_context(mock.patch("strip_insar.write_wrapped_phase_png"))
                mock_tmpdir.return_value.__enter__.return_value = str(output_dir / "tmp_unwrap")
                mock_tmpdir.return_value.__exit__.return_value = False
                fake_unwrapper = mock.Mock()
                fake_unwrapper.unwrap.return_value = np.ones((100, 200), dtype=np.float32)
                stack.enter_context(mock.patch("strip_insar._create_unwrapper", return_value=fake_unwrapper))
                process_strip_insar(
                    str(master_manifest),
                    str(slave_manifest),
                    str(output_dir),
                    gpu_mode="cpu",
                )

            self.assertEqual(mock_crossmul.call_args.args[1], str(fine_coreg))

    def test_process_strip_insar_full_pipeline_marks_crossmul_cpu_when_gpu_crossmul_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            import numpy as np

            output_dir = Path(tmpdir) / "out"
            master_manifest = self._write_manifest(tmpdir, "master.json", {"sensor": "tianyi", "slc": {"path": "master.tif"}})
            slave_manifest = self._write_manifest(tmpdir, "slave.json", {"sensor": "tianyi", "slc": {"path": "slave.tif"}})

            meta = (
                {},
                {"header": {"firstStateTimeUTC": "2024-01-01T00:00:00Z"}},
                {"centerFrequency": 5405e6, "prf": 1000.0, "lookDirection": "RIGHT", "polarisation": "VV", "startGPSTime": 100.0},
                {"numberOfRows": 100, "numberOfColumns": 200, "groundRangeResolution": 2.0, "azimuthResolution": 3.0},
                {},
            )

            with ExitStack() as stack:
                stack.enter_context(mock.patch("strip_insar.load_scene_corners_with_fallback", return_value=[]))
                stack.enter_context(mock.patch("strip_insar._load_master_metadata", side_effect=[meta, meta, meta, meta]))
                stack.enter_context(mock.patch("strip_insar.choose_orbit_interp", return_value="Legendre"))
                stack.enter_context(mock.patch("strip_insar._resolve_dem_path", return_value="/tmp/dem.tif"))
                stack.enter_context(mock.patch("strip_insar.select_processing_backend", return_value=("gpu", "GPU available")))
                stack.enter_context(mock.patch("strip_insar.query_gpu_memory_info", return_value={"total": 1024, "free": 1024}))
                stack.enter_context(mock.patch("strip_insar.choose_gpu_topo_block_rows", return_value=(256, "fixed for test")))
                stack.enter_context(mock.patch("strip_insar._run_geo2rdr", return_value=("/tmp/master_topo.tif", "/tmp/slave_topo.tif")))
                stack.enter_context(
                    mock.patch(
                        "strip_insar._run_p1_stage_from_cache",
                        return_value={
                            "coarse_coreg_slave": str(output_dir / "work" / "p1_dense_match" / "coarse_coreg_slave.tif"),
                            "fine_coreg_slave": str(output_dir / "work" / "p1_dense_match" / "fine_coreg_slave.tif"),
                        },
                    )
                )
                stack.enter_context(
                    mock.patch(
                        "strip_insar._run_crossmul",
                        return_value=(
                            np.ones((100, 200), dtype=np.complex64),
                            np.ones((100, 200), dtype=np.float32),
                            "cpu",
                            "ISCE3 CUDA Crossmul is experimental and disabled by default",
                        ),
                    )
                )
                stack.enter_context(mock.patch("strip_insar._write_radar_amplitude_png", side_effect=lambda slc_path, output_png: str(output_png)))
                stack.enter_context(mock.patch("strip_insar.construct_radar_grid", return_value=object()))
                stack.enter_context(mock.patch("strip_insar.construct_orbit", return_value=object()))
                mock_tmpdir = stack.enter_context(mock.patch("tempfile.TemporaryDirectory"))
                stack.enter_context(mock.patch("strip_insar.write_insar_hdf", return_value=str(output_dir / "interferogram_fullres.h5")))
                stack.enter_context(mock.patch("strip_insar.append_topo_coordinates_hdf"))
                stack.enter_context(mock.patch("strip_insar.append_utm_coordinates_hdf"))
                stack.enter_context(mock.patch("strip_insar.compute_utm_output_shape", return_value=(32, 16)))
                stack.enter_context(mock.patch("strip_insar.geocode_product"))
                stack.enter_context(mock.patch("strip_insar.write_wrapped_phase_geotiff"))
                stack.enter_context(mock.patch("strip_insar.write_wrapped_phase_png"))
                mock_tmpdir.return_value.__enter__.return_value = str(output_dir / "tmp_unwrap")
                mock_tmpdir.return_value.__exit__.return_value = False
                fake_unwrapper = mock.Mock()
                fake_unwrapper.unwrap.return_value = np.ones((100, 200), dtype=np.float32)
                stack.enter_context(mock.patch("strip_insar._create_unwrapper", return_value=fake_unwrapper))

                result = process_strip_insar(
                    str(master_manifest),
                    str(slave_manifest),
                    str(output_dir),
                    gpu_mode="gpu",
                )

            self.assertEqual(result["backend_requested"], "gpu")
            self.assertEqual(result["backend_used"], "gpu")
            self.assertEqual(result["pipeline_mode"], "hybrid")
            self.assertEqual(result["stage_backends"]["crossmul"], "cpu")
            self.assertEqual(
                result["fallback_reasons"]["crossmul"],
                "ISCE3 CUDA Crossmul is experimental and disabled by default",
            )

    def test_process_strip_insar_step_p2_writes_radar_phase_png(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "out"
            for stage_name, dir_name in [
                ("check", "check"),
                ("prep", "prep"),
                ("crop", "crop"),
                ("p0", "p0_geo2rdr"),
                ("p1", "p1_dense_match"),
            ]:
                stage_path = output_dir / "work" / dir_name
                stage_path.mkdir(parents=True, exist_ok=True)
                record = {"stage": stage_name, "success": True, "output_files": {}}
                if stage_name == "p1":
                    fine_coreg = stage_path / "fine_coreg_slave.tif"
                    fine_coreg.write_text("stub", encoding="utf-8")
                    record["output_files"] = {"fine_coreg_slave": str(fine_coreg)}
                (stage_path / "stage.json").write_text(json.dumps(record), encoding="utf-8")
                (stage_path / "SUCCESS").write_text("success\n", encoding="utf-8")

            master_manifest = self._write_manifest(tmpdir, "master.json", {"sensor": "tianyi", "slc": {"path": "master.tif"}})
            slave_manifest = self._write_manifest(tmpdir, "slave.json", {"sensor": "tianyi", "slc": {"path": "slave.tif"}})
            meta = (
                {},
                {"header": {"firstStateTimeUTC": "2024-01-01T00:00:00Z"}},
                {"centerFrequency": 5405e6, "prf": 1000.0, "lookDirection": "RIGHT", "polarisation": "VV", "startGPSTime": 100.0},
                {"numberOfRows": 100, "numberOfColumns": 200, "groundRangeResolution": 2.0, "azimuthResolution": 3.0},
                {},
            )

            with (
                mock.patch("strip_insar.load_scene_corners_with_fallback", return_value=[]),
                mock.patch("strip_insar._load_master_metadata", side_effect=[meta, meta, meta]),
                mock.patch("strip_insar.choose_orbit_interp", return_value="Legendre"),
                mock.patch("strip_insar._resolve_dem_path", return_value="/tmp/dem.tif"),
                mock.patch(
                    "strip_insar._run_crossmul",
                    return_value=(np.ones((16, 16), dtype=np.complex64), np.ones((16, 16), dtype=np.float32)),
                ),
                mock.patch("strip_insar._write_radar_amplitude_png", side_effect=lambda slc_path, output_png: str(output_png)),
                mock.patch("strip_insar._write_radar_wrapped_phase_png", return_value=str(output_dir / "wrapped_phase_radar.png")) as mock_phase_png,
            ):
                process_strip_insar(
                    str(master_manifest),
                    str(slave_manifest),
                    str(output_dir),
                    step="p2",
                )

            mock_phase_png.assert_called_once()
            p2_record = json.loads((output_dir / "work" / "p2_crossmul" / "stage.json").read_text(encoding="utf-8"))
            self.assertIn("wrapped_phase_radar_png", p2_record["output_files"])

    def test_process_strip_insar_step_p2_records_isce2_flatten_policy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "out"
            p1_dir = output_dir / "work" / "p1_dense_match"
            p1_dir.mkdir(parents=True, exist_ok=True)
            fine_coreg = p1_dir / "fine_coreg_slave.tif"
            fine_coreg.write_text("stub", encoding="utf-8")
            range_off = output_dir / "p1_geo2rdr_offsets" / "range.off"
            range_off.parent.mkdir(parents=True, exist_ok=True)
            np.full((100, 200), 3.0, dtype=np.float64).tofile(range_off)
            for stage_name, dir_name in [
                ("check", "check"),
                ("prep", "prep"),
                ("crop", "crop"),
                ("p0", "p0_geo2rdr"),
            ]:
                stage_path = output_dir / "work" / dir_name
                stage_path.mkdir(parents=True, exist_ok=True)
                (stage_path / "stage.json").write_text(
                    json.dumps({"stage": stage_name, "success": True, "output_files": {}}),
                    encoding="utf-8",
                )
                (stage_path / "SUCCESS").write_text("success\n", encoding="utf-8")
            (p1_dir / "stage.json").write_text(
                json.dumps(
                    {
                        "stage": "p1",
                        "success": True,
                        "output_files": {
                            "fine_coreg_slave": str(fine_coreg),
                            "coarse_geo2rdr_range_offsets": str(range_off),
                        },
                    }
                ),
                encoding="utf-8",
            )
            (p1_dir / "SUCCESS").write_text("success\n", encoding="utf-8")

            master_manifest = self._write_manifest(tmpdir, "master.json", {"sensor": "tianyi", "slc": {"path": "master.tif"}})
            slave_manifest = self._write_manifest(tmpdir, "slave.json", {"sensor": "tianyi", "slc": {"path": "slave.tif"}})
            meta = (
                {},
                {"header": {"firstStateTimeUTC": "2024-01-01T00:00:00Z"}},
                {"centerFrequency": 5405e6, "prf": 1000.0, "lookDirection": "RIGHT", "polarisation": "VV", "startGPSTime": 100.0},
                {"numberOfRows": 100, "numberOfColumns": 200, "groundRangeResolution": 2.0, "azimuthResolution": 3.0, "columnSpacing": 2.0, "rangeTimeFirstPixel": 0.001},
                {},
            )

            with (
                mock.patch("strip_insar.load_scene_corners_with_fallback", return_value=[]),
                mock.patch("strip_insar._load_master_metadata", side_effect=[meta, meta, meta]),
                mock.patch("strip_insar.choose_orbit_interp", return_value="Legendre"),
                mock.patch("strip_insar._resolve_dem_path", return_value="/tmp/dem.tif"),
                mock.patch(
                    "strip_insar._run_crossmul",
                    return_value=(np.ones((16, 16), dtype=np.complex64), np.ones((16, 16), dtype=np.float32)),
                ),
                mock.patch("strip_insar._write_radar_amplitude_png", side_effect=lambda slc_path, output_png: str(output_png)),
                mock.patch("strip_insar._write_radar_wrapped_phase_png", return_value=str(output_dir / "wrapped_phase_radar.png")),
            ):
                process_strip_insar(
                    str(master_manifest),
                    str(slave_manifest),
                    str(output_dir),
                    step="p2",
                )

            p2_record = json.loads((output_dir / "work" / "p2_crossmul" / "stage.json").read_text(encoding="utf-8"))
            self.assertEqual(p2_record["output_files"]["flatten_method"], "isce2-explicit-rangeoff-flatten")
            self.assertEqual(p2_record["output_files"]["flatten_offset_policy"]["source"], "coarse_geo2rdr_range_offsets")
            self.assertEqual(
                p2_record["output_files"]["flatten_offset_policy"]["invalid_values"]["replacement"],
                0.0,
            )

    def test_process_strip_insar_step_p2_keeps_flatten_for_fine_coreg_slave(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "out"
            p1_dir = output_dir / "work" / "p1_dense_match"
            p1_dir.mkdir(parents=True, exist_ok=True)
            fine_coreg = p1_dir / "fine_coreg_slave.tif"
            fine_coreg.write_text("stub", encoding="utf-8")
            range_off = output_dir / "p1_geo2rdr_offsets" / "range.off"
            range_off.parent.mkdir(parents=True, exist_ok=True)
            np.full((100, 200), 3.0, dtype=np.float64).tofile(range_off)
            for stage_name, dir_name in [
                ("check", "check"),
                ("prep", "prep"),
                ("crop", "crop"),
                ("p0", "p0_geo2rdr"),
            ]:
                stage_path = output_dir / "work" / dir_name
                stage_path.mkdir(parents=True, exist_ok=True)
                (stage_path / "stage.json").write_text(
                    json.dumps({"stage": stage_name, "success": True, "output_files": {}}),
                    encoding="utf-8",
                )
                (stage_path / "SUCCESS").write_text("success\n", encoding="utf-8")
            (p1_dir / "stage.json").write_text(
                json.dumps(
                    {
                        "stage": "p1",
                        "success": True,
                        "output_files": {
                            "fine_coreg_slave": str(fine_coreg),
                            "coarse_geo2rdr_range_offsets": str(range_off),
                        },
                    }
                ),
                encoding="utf-8",
            )
            (p1_dir / "SUCCESS").write_text("success\n", encoding="utf-8")

            master_manifest = self._write_manifest(tmpdir, "master.json", {"sensor": "tianyi", "slc": {"path": "master.tif"}})
            slave_manifest = self._write_manifest(tmpdir, "slave.json", {"sensor": "tianyi", "slc": {"path": "slave.tif"}})
            meta = (
                {},
                {"header": {"firstStateTimeUTC": "2024-01-01T00:00:00Z"}},
                {"centerFrequency": 5405e6, "prf": 1000.0, "lookDirection": "RIGHT", "polarisation": "VV", "startGPSTime": 100.0},
                {"numberOfRows": 100, "numberOfColumns": 200, "groundRangeResolution": 2.0, "azimuthResolution": 3.0, "columnSpacing": 2.0, "rangeTimeFirstPixel": 0.001},
                {},
            )

            with (
                mock.patch("strip_insar.load_scene_corners_with_fallback", return_value=[]),
                mock.patch("strip_insar._load_master_metadata", side_effect=[meta, meta, meta]),
                mock.patch("strip_insar.choose_orbit_interp", return_value="Legendre"),
                mock.patch("strip_insar._resolve_dem_path", return_value="/tmp/dem.tif"),
                mock.patch(
                    "strip_insar._run_crossmul",
                    return_value=(np.ones((16, 16), dtype=np.complex64), np.ones((16, 16), dtype=np.float32)),
                ) as mock_crossmul,
                mock.patch("strip_insar._write_radar_amplitude_png", side_effect=lambda slc_path, output_png: str(output_png)),
                mock.patch("strip_insar._write_radar_wrapped_phase_png", return_value=str(output_dir / "wrapped_phase_radar.png")),
            ):
                process_strip_insar(
                    str(master_manifest),
                    str(slave_manifest),
                    str(output_dir),
                    step="p2",
                )

            flatten_raster = Path(mock_crossmul.call_args.kwargs["flatten_raster"])
            self.assertEqual(flatten_raster.name, "range_flatten.off.tif")
            self.assertTrue(flatten_raster.is_file())
            self.assertEqual(mock_crossmul.call_args.kwargs["range_pixel_spacing"], 2.0)
            self.assertAlmostEqual(mock_crossmul.call_args.kwargs["wavelength"], 299792458.0 / 5405e6)

    def test_process_strip_insar_step_p2_passes_flatten_inputs_for_raw_slave(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "out"
            p1_dir = output_dir / "work" / "p1_dense_match"
            p1_dir.mkdir(parents=True, exist_ok=True)
            range_off = output_dir / "p1_geo2rdr_offsets" / "range.off"
            range_off.parent.mkdir(parents=True, exist_ok=True)
            np.full((100, 200), 3.0, dtype=np.float64).tofile(range_off)
            for stage_name, dir_name in [
                ("check", "check"),
                ("prep", "prep"),
                ("crop", "crop"),
                ("p0", "p0_geo2rdr"),
            ]:
                stage_path = output_dir / "work" / dir_name
                stage_path.mkdir(parents=True, exist_ok=True)
                (stage_path / "stage.json").write_text(
                    json.dumps({"stage": stage_name, "success": True, "output_files": {}}),
                    encoding="utf-8",
                )
                (stage_path / "SUCCESS").write_text("success\n", encoding="utf-8")
            (p1_dir / "stage.json").write_text(
                json.dumps(
                    {
                        "stage": "p1",
                        "success": True,
                        "output_files": {
                            "coarse_geo2rdr_range_offsets": str(range_off),
                        },
                    }
                ),
                encoding="utf-8",
            )
            (p1_dir / "SUCCESS").write_text("success\n", encoding="utf-8")

            master_manifest = self._write_manifest(tmpdir, "master.json", {"sensor": "tianyi", "slc": {"path": "master.tif"}})
            slave_manifest = self._write_manifest(tmpdir, "slave.json", {"sensor": "tianyi", "slc": {"path": "slave.tif"}})
            meta = (
                {},
                {"header": {"firstStateTimeUTC": "2024-01-01T00:00:00Z"}},
                {"centerFrequency": 5405e6, "prf": 1000.0, "lookDirection": "RIGHT", "polarisation": "VV", "startGPSTime": 100.0},
                {"numberOfRows": 100, "numberOfColumns": 200, "groundRangeResolution": 2.0, "azimuthResolution": 3.0, "columnSpacing": 2.0, "rangeTimeFirstPixel": 0.001},
                {},
            )

            with (
                mock.patch("strip_insar.load_scene_corners_with_fallback", return_value=[]),
                mock.patch("strip_insar._load_master_metadata", side_effect=[meta, meta, meta]),
                mock.patch("strip_insar.choose_orbit_interp", return_value="Legendre"),
                mock.patch("strip_insar._resolve_dem_path", return_value="/tmp/dem.tif"),
                mock.patch(
                    "strip_insar._run_crossmul",
                    return_value=(np.ones((16, 16), dtype=np.complex64), np.ones((16, 16), dtype=np.float32)),
                ) as mock_crossmul,
                mock.patch("strip_insar._write_radar_amplitude_png", side_effect=lambda slc_path, output_png: str(output_png)),
                mock.patch("strip_insar._write_radar_wrapped_phase_png", return_value=str(output_dir / "wrapped_phase_radar.png")),
            ):
                process_strip_insar(
                    str(master_manifest),
                    str(slave_manifest),
                    str(output_dir),
                    step="p2",
                )

            flatten_raster = Path(mock_crossmul.call_args.kwargs["flatten_raster"])
            self.assertEqual(flatten_raster.name, "range_flatten.off.tif")
            self.assertTrue(flatten_raster.is_file())
            self.assertEqual(mock_crossmul.call_args.kwargs["range_pixel_spacing"], 2.0)
            self.assertAlmostEqual(mock_crossmul.call_args.kwargs["wavelength"], 299792458.0 / 5405e6)

    def test_prepare_crossmul_flatten_raster_writes_float64_geotiff(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "out"
            range_off = output_dir / "p1_geo2rdr_offsets" / "range.off"
            range_off.parent.mkdir(parents=True, exist_ok=True)
            np.full((4, 5), 2.0, dtype=np.float64).tofile(range_off)

            flatten_path = strip_insar._prepare_crossmul_flatten_raster(
                output_dir=output_dir,
                p1_outputs={"coarse_geo2rdr_range_offsets": str(range_off)},
                radargrid_data={"numberOfRows": 4, "numberOfColumns": 5},
            )

            from osgeo import gdal

            ds = gdal.Open(flatten_path)
            self.assertIsNotNone(ds)
            band = ds.GetRasterBand(1)
            self.assertEqual(gdal.GetDataTypeName(band.DataType), "Float64")
            arr = band.ReadAsArray()
            ds = None
            np.testing.assert_allclose(arr, 2.0)

    def test_prepare_crossmul_flatten_raster_sanitizes_geo2rdr_invalid_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "out"
            range_off = output_dir / "p1_geo2rdr_offsets" / "range.off"
            range_off.parent.mkdir(parents=True, exist_ok=True)
            np.array(
                [
                    [2.0, -1000000.0, np.nan],
                    [3.0, -999999.0, -100001.0],
                ],
                dtype=np.float64,
            ).tofile(range_off)

            flatten_path = strip_insar._prepare_crossmul_flatten_raster(
                output_dir=output_dir,
                p1_outputs={"coarse_geo2rdr_range_offsets": str(range_off)},
                radargrid_data={"numberOfRows": 2, "numberOfColumns": 3},
            )

            from osgeo import gdal

            ds = gdal.Open(flatten_path)
            self.assertIsNotNone(ds)
            arr = ds.GetRasterBand(1).ReadAsArray()
            ds = None
            np.testing.assert_allclose(
                arr,
                np.array(
                    [
                        [2.0, 0.0, 0.0],
                        [3.0, 0.0, 0.0],
                    ],
                    dtype=np.float64,
                ),
            )

    def test_run_crossmul_gpu_uses_isce3_cuda_crossmul(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            fake_ifg = np.array([[1 + 0j]], dtype=np.complex64)
            fake_coh = np.array([[0.5]], dtype=np.float32)
            fake_cuda_core = mock.Mock()
            fake_cuda_core.Device.return_value = "gpu0"
            fake_cuda_signal = mock.Mock()

            with (
                mock.patch("isce3.cuda.core.Device", fake_cuda_core.Device),
                mock.patch("isce3.cuda.core.set_device", fake_cuda_core.set_device),
                mock.patch("isce3.cuda.signal.Crossmul", fake_cuda_signal.Crossmul),
                mock.patch(
                    "strip_insar._crossmul_numpy",
                    return_value=(np.zeros((1, 1), dtype=np.complex64), np.zeros((1, 1), dtype=np.float32)),
                ),
                mock.patch(
                    "strip_insar._crossmul_isce3_gpu",
                    return_value=(fake_ifg, fake_coh),
                ) as mock_gpu_impl,
                mock.patch(
                    "strip_insar._estimate_coherence_from_slcs",
                    return_value=fake_coh,
                ) as mock_estimate_coh,
                mock.patch.dict("os.environ", {"D2SAR_ENABLE_EXPERIMENTAL_GPU_CROSSMUL": "1"}),
            ):
                ifg, coh, backend, fallback_reason = strip_insar._run_crossmul(
                    "master.tif",
                    "slave.tif",
                    use_gpu=True,
                    gpu_id=2,
                    output_dir=output_dir,
                    flatten_raster="range.off",
                    range_pixel_spacing=2.5,
                    wavelength=0.23,
                    block_rows=128,
                )

            np.testing.assert_allclose(ifg, fake_ifg)
            np.testing.assert_allclose(coh, fake_coh)
            self.assertEqual(backend, "gpu")
            self.assertIsNone(fallback_reason)
            fake_cuda_core.Device.assert_called_once_with(2)
            fake_cuda_core.set_device.assert_called_once_with("gpu0")
            mock_gpu_impl.assert_called_once_with(
                "master.tif",
                "slave.tif",
                output_dir,
                2,
                128,
                flatten_raster="range.off",
                range_pixel_spacing=2.5,
                wavelength=0.23,
            )
            mock_estimate_coh.assert_called_once_with(
                "master.tif",
                "slave.tif",
                block_rows=128,
            )

    def test_run_crossmul_gpu_success_removes_stale_fallback_reason(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            fallback_path = output_dir / "work" / "p2_crossmul" / "gpu_fallback_reason.txt"
            fallback_path.parent.mkdir(parents=True, exist_ok=True)
            fallback_path.write_text("old failure\n", encoding="utf-8")
            fake_ifg = np.array([[1 + 0j]], dtype=np.complex64)
            fake_coh = np.array([[0.5]], dtype=np.float32)

            with (
                mock.patch("isce3.cuda.core.Device", return_value="gpu0"),
                mock.patch("isce3.cuda.core.set_device"),
                mock.patch(
                    "strip_insar._crossmul_isce3_gpu",
                    return_value=(fake_ifg, fake_coh),
                ),
                mock.patch(
                    "strip_insar._estimate_coherence_from_slcs",
                    return_value=fake_coh,
                ),
                mock.patch.dict("os.environ", {"D2SAR_ENABLE_EXPERIMENTAL_GPU_CROSSMUL": "1"}),
            ):
                _, _, backend, fallback_reason = strip_insar._run_crossmul(
                    "master.tif",
                    "slave.tif",
                    use_gpu=True,
                    gpu_id=0,
                    output_dir=output_dir,
                )

            self.assertEqual(backend, "gpu")
            self.assertIsNone(fallback_reason)
            self.assertFalse(fallback_path.exists())

    def test_run_crossmul_gpu_disabled_by_default_uses_cpu_with_reason(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            fake_ifg = np.array([[2 + 0j]], dtype=np.complex64)
            fake_coh = np.array([[0.8]], dtype=np.float32)

            with (
                mock.patch(
                    "strip_insar._crossmul_numpy",
                    return_value=(fake_ifg, fake_coh),
                ) as mock_cpu,
                mock.patch("strip_insar._crossmul_isce3_gpu") as mock_gpu_impl,
                mock.patch.dict("os.environ", {}, clear=False),
            ):
                ifg, coh, backend, fallback_reason = strip_insar._run_crossmul(
                    "master.tif",
                    "slave.tif",
                    use_gpu=True,
                    gpu_id=0,
                    output_dir=output_dir,
                )

            np.testing.assert_allclose(ifg, fake_ifg)
            np.testing.assert_allclose(coh, fake_coh)
            self.assertEqual(backend, "cpu")
            self.assertIn("experimental", fallback_reason)
            self.assertIn("disabled", fallback_reason)
            mock_cpu.assert_called_once()
            mock_gpu_impl.assert_not_called()

    def test_run_crossmul_gpu_subprocess_failure_raises_runtime_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch(
                "strip_insar._crossmul_isce3_gpu_subprocess",
                side_effect=RuntimeError("gpu crossmul subprocess failed with exit code 139: stderr tail"),
            ):
                with self.assertRaisesRegex(RuntimeError, "exit code 139"):
                    strip_insar._crossmul_isce3_gpu(
                        "master.tif",
                        "slave.tif",
                        Path(tmpdir),
                        0,
                        128,
                        flatten_raster="range.off",
                        range_pixel_spacing=2.0,
                        wavelength=0.23,
                    )

    def test_crossmul_gpu_helper_materializes_inputs_as_envi(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            with mock.patch("strip_insar.subprocess.run") as mock_run:
                mock_run.return_value.returncode = 99
                mock_run.return_value.stdout = ""
                mock_run.return_value.stderr = "forced failure"

                with self.assertRaisesRegex(RuntimeError, "exit code 99"):
                    strip_insar._crossmul_isce3_gpu_subprocess(
                        master_slc_path="master.tif",
                        slave_slc_path="slave.tif",
                        output_dir=output_dir,
                        gpu_id=0,
                        block_rows=128,
                    )

            helper = output_dir / "work" / "p2_crossmul" / "cuda_crossmul_helper.py"
            helper_text = helper.read_text(encoding="utf-8")
            self.assertIn("_copy_raster_to_envi_complex64", helper_text)
            self.assertIn("cuda_inputs", helper_text)
            self.assertNotIn("_ensure_local_tiff", helper_text)

    def test_crossmul_gpu_helper_is_valid_python(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            with mock.patch("strip_insar.subprocess.run") as mock_run:
                mock_run.return_value.returncode = 99
                mock_run.return_value.stdout = ""
                mock_run.return_value.stderr = "forced failure"

                with self.assertRaisesRegex(RuntimeError, "exit code 99"):
                    strip_insar._crossmul_isce3_gpu_subprocess(
                        master_slc_path="master.tif",
                        slave_slc_path="slave.tif",
                        output_dir=output_dir,
                        gpu_id=0,
                        block_rows=128,
                    )

            helper = output_dir / "work" / "p2_crossmul" / "cuda_crossmul_helper.py"
            import py_compile

            py_compile.compile(str(helper), doraise=True)

    def test_process_strip_insar_range_p2_to_p4_uses_cached_dependencies(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "out"
            for stage_name, dir_name in [("check", "check"), ("prep", "prep"), ("crop", "crop"), ("p0", "p0_geo2rdr"), ("p1", "p1_dense_match")]:
                stage_path = output_dir / "work" / dir_name
                stage_path.mkdir(parents=True, exist_ok=True)
                record = {"stage": stage_name, "success": True, "output_files": {}}
                if stage_name == "p1":
                    offsets = stage_path / "offsets.json"
                    offsets.write_text(json.dumps({"row_offset": None, "col_offset": None}), encoding="utf-8")
                    record["output_files"] = {"offsets": str(offsets)}
                (stage_path / "stage.json").write_text(json.dumps(record), encoding="utf-8")
                (stage_path / "SUCCESS").write_text("success\n", encoding="utf-8")

            master_manifest = self._write_manifest(tmpdir, "master.json", {"sensor": "tianyi", "slc": {"path": "master.tif"}})
            slave_manifest = self._write_manifest(tmpdir, "slave.json", {"sensor": "tianyi", "slc": {"path": "slave.tif"}})
            meta = (
                {},
                {"header": {"firstStateTimeUTC": "2024-01-01T00:00:00Z"}},
                {"centerFrequency": 5405e6, "prf": 1000.0, "lookDirection": "RIGHT", "polarisation": "VV", "startGPSTime": 100.0},
                {"numberOfRows": 100, "numberOfColumns": 200, "groundRangeResolution": 2.0, "azimuthResolution": 3.0},
                {},
            )

            fake_unwrapper = mock.Mock()
            fake_unwrapper.unwrap.return_value = np.full((2, 2), 3.0, dtype=np.float32)

            with (
                mock.patch("strip_insar.load_scene_corners_with_fallback", return_value=[]),
                mock.patch("strip_insar._load_master_metadata", side_effect=[meta, meta, meta]),
                mock.patch("strip_insar.choose_orbit_interp", return_value="Legendre"),
                mock.patch("strip_insar._resolve_dem_path", return_value="/tmp/dem.tif"),
                mock.patch("strip_insar._create_unwrapper", return_value=fake_unwrapper),
                mock.patch("strip_insar._run_crossmul", return_value=(np.ones((100, 200), dtype=np.complex64), np.ones((100, 200), dtype=np.float32))) as mock_crossmul,
                mock.patch("strip_insar.construct_radar_grid", return_value=object()),
                mock.patch("strip_insar.construct_orbit", return_value=object()),
                mock.patch("tempfile.TemporaryDirectory") as mock_tmpdir,
            ):
                mock_tmpdir.return_value.__enter__.return_value = str(output_dir / "tmp_unwrap")
                mock_tmpdir.return_value.__exit__.return_value = False
                result = process_strip_insar(
                    str(master_manifest),
                    str(slave_manifest),
                    str(output_dir),
                    start_step="p2",
                    end_step="p4",
                    window=(10, 20, 30, 40),
                )

            mock_crossmul.assert_called_once()
            fake_unwrapper.unwrap.assert_called_once()
            unwrap_ifg = fake_unwrapper.unwrap.call_args.args[0]
            self.assertEqual(unwrap_ifg.shape, (30, 40))
            p2_record = json.loads((output_dir / "work" / "p2_crossmul" / "stage.json").read_text(encoding="utf-8"))
            filtered_path = p2_record["output_files"]["filtered_interferogram"]
            np.testing.assert_allclose(unwrap_ifg, np.load(filtered_path))
            self.assertEqual(result["completed_stages"], ["p2", "p3", "p4"])
            self.assertTrue((output_dir / "work" / "p3_unwrap" / "SUCCESS").exists())
            self.assertTrue((output_dir / "work" / "p4_geocode" / "SUCCESS").exists())

    def test_process_strip_insar_crop_pipeline_writes_cropped_coordinates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            master_manifest = self._write_manifest(
                tmpdir,
                "master.json",
                {"sensor": "tianyi", "slc": {"path": "master.tif"}},
            )
            slave_manifest = self._write_manifest(
                tmpdir,
                "slave.json",
                {"sensor": "tianyi", "slc": {"path": "slave.tif"}},
            )
            output_dir = Path(tmpdir) / "out"
            meta = (
                {},
                {"header": {"firstStateTimeUTC": "2024-01-01T00:00:00Z"}},
                {
                    "centerFrequency": 5405e6,
                    "prf": 1000.0,
                    "lookDirection": "RIGHT",
                    "polarisation": "VV",
                    "startGPSTime": 100.0,
                },
                {
                    "numberOfRows": 100,
                    "numberOfColumns": 200,
                    "groundRangeResolution": 2.0,
                    "azimuthResolution": 3.0,
                },
                {},
            )

            with (
                mock.patch("strip_insar.load_scene_corners_with_fallback", return_value=[]),
                mock.patch("strip_insar._load_master_metadata", side_effect=[meta, meta, meta]),
                mock.patch("strip_insar.choose_orbit_interp", return_value="Legendre"),
                mock.patch("strip_insar._resolve_dem_path", return_value="/tmp/dem.tif"),
                mock.patch("strip_insar.select_processing_backend", return_value=("cpu", "CPU forced")),
                mock.patch("strip_insar._run_geo2rdr", return_value=("/tmp/master_topo.tif", "/tmp/slave_topo.tif")),
                mock.patch("strip_insar._run_crossmul", return_value=(__import__("numpy").ones((100, 200), dtype=__import__("numpy").complex64), __import__("numpy").ones((100, 200), dtype=__import__("numpy").float32))),
                mock.patch("strip_insar.construct_radar_grid", return_value=object()),
                mock.patch("strip_insar.construct_orbit", return_value=object()),
                mock.patch("tempfile.TemporaryDirectory") as mock_tmpdir,
                mock.patch("strip_insar.write_insar_hdf", return_value=str(output_dir / "interferogram_fullres.h5")),
                mock.patch("strip_insar.append_topo_coordinates_hdf") as mock_append_topo,
                mock.patch("strip_insar.append_utm_coordinates_hdf") as mock_append_utm,
                mock.patch("strip_insar.compute_utm_output_shape", return_value=(32, 16)),
                mock.patch("strip_insar.geocode_product"),
                mock.patch("strip_insar.write_wrapped_phase_geotiff"),
                mock.patch("strip_insar.write_wrapped_phase_png"),
            ):
                mock_tmpdir.return_value.__enter__.return_value = str(output_dir / "tmp_unwrap")
                mock_tmpdir.return_value.__exit__.return_value = False
                fake_unwrapper = mock.Mock()
                fake_unwrapper.unwrap.return_value = __import__("numpy").ones((30, 40), dtype=__import__("numpy").float32)
                with mock.patch("strip_insar._create_unwrapper", return_value=fake_unwrapper):
                    process_strip_insar(
                        str(master_manifest),
                        str(slave_manifest),
                        str(output_dir),
                        window=(10, 20, 30, 40),
                        gpu_mode="cpu",
                    )

            mock_append_topo.assert_called()
            mock_append_utm.assert_called()

    def test_process_strip_insar_resume_from_p5_runs_p5_then_p6(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            import numpy as np

            output_dir = Path(tmpdir) / "out"
            for stage_name, dir_name in [
                ("check", "check"),
                ("prep", "prep"),
                ("crop", "crop"),
                ("p0", "p0_geo2rdr"),
                ("p1", "p1_dense_match"),
                ("p2", "p2_crossmul"),
                ("p3", "p3_unwrap"),
                ("p4", "p4_geocode"),
            ]:
                stage_path = output_dir / "work" / dir_name
                stage_path.mkdir(parents=True, exist_ok=True)
                record = {"stage": stage_name, "success": True, "output_files": {}}
                if stage_name == "p2":
                    igram = stage_path / "interferogram.npy"
                    coh = stage_path / "coherence.npy"
                    np.save(igram, np.ones((2, 2), dtype=np.complex64))
                    np.save(coh, np.ones((2, 2), dtype=np.float32))
                    record["output_files"] = {"interferogram": str(igram), "coherence": str(coh)}
                elif stage_name == "p3":
                    unw = stage_path / "unwrapped_phase.npy"
                    np.save(unw, np.ones((2, 2), dtype=np.float32))
                    record["output_files"] = {"unwrapped_phase": str(unw)}
                elif stage_name == "p4":
                    los = stage_path / "los_displacement.npy"
                    np.save(los, np.ones((2, 2), dtype=np.float32))
                    record["output_files"] = {"los_displacement": str(los)}
                (stage_path / "stage.json").write_text(json.dumps(record), encoding="utf-8")
                (stage_path / "SUCCESS").write_text("success\n", encoding="utf-8")

            master_manifest = self._write_manifest(
                tmpdir,
                "master.json",
                {"sensor": "tianyi", "slc": {"path": "master.tif"}},
            )
            slave_manifest = self._write_manifest(
                tmpdir,
                "slave.json",
                {"sensor": "tianyi", "slc": {"path": "slave.tif"}},
            )
            meta = (
                {},
                {"header": {"firstStateTimeUTC": "2024-01-01T00:00:00Z"}},
                {
                    "centerFrequency": 5405e6,
                    "prf": 1000.0,
                    "lookDirection": "RIGHT",
                    "polarisation": "VV",
                    "startGPSTime": 100.0,
                },
                {
                    "numberOfRows": 100,
                    "numberOfColumns": 200,
                    "groundRangeResolution": 2.0,
                    "azimuthResolution": 3.0,
                },
                {},
            )

            with (
                mock.patch("strip_insar._load_master_metadata", side_effect=[meta, meta]),
                mock.patch("strip_insar.choose_orbit_interp", return_value="Legendre"),
                mock.patch("strip_insar.resolve_manifest_data_path", side_effect=lambda manifest_path, rel: rel),
                mock.patch("strip_insar.construct_radar_grid", return_value=object()),
                mock.patch("strip_insar.write_insar_hdf") as mock_hdf,
                mock.patch("strip_insar.append_topo_coordinates_hdf"),
                mock.patch("strip_insar.append_utm_coordinates_hdf"),
                mock.patch("strip_insar.compute_utm_output_shape", return_value=(32, 16)),
                mock.patch("strip_insar.geocode_product") as mock_geocode,
                mock.patch("strip_insar.write_wrapped_phase_geotiff") as mock_ifg_tif,
                mock.patch("strip_insar.write_wrapped_phase_png") as mock_png,
            ):
                result = process_strip_insar(
                    str(master_manifest),
                    str(slave_manifest),
                    str(output_dir),
                    resume=True,
                )

            mock_hdf.assert_called_once()
            self.assertEqual(mock_geocode.call_count, 3)
            mock_ifg_tif.assert_called_once()
            self.assertEqual(mock_png.call_count, 2)
            self.assertEqual(result["completed_stages"], ["p5", "p6"])

    def test_process_strip_insar_does_not_fallback_full_pipeline_on_p6_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            master_manifest = self._write_manifest(
                tmpdir,
                "master.json",
                {"sensor": "tianyi", "slc": {"path": "master.tif"}},
            )
            slave_manifest = self._write_manifest(
                tmpdir,
                "slave.json",
                {"sensor": "tianyi", "slc": {"path": "slave.tif"}},
            )
            output_dir = Path(tmpdir) / "out"
            h5_path = output_dir / "interferogram_fullres.h5"

            meta = (
                {},
                {"header": {"firstStateTimeUTC": "2024-01-01T00:00:00Z"}},
                {
                    "centerFrequency": 5405e6,
                    "prf": 1000.0,
                    "lookDirection": "RIGHT",
                    "polarisation": "VV",
                    "startGPSTime": 100.0,
                },
                {
                    "numberOfRows": 100,
                    "numberOfColumns": 200,
                    "groundRangeResolution": 2.0,
                    "azimuthResolution": 3.0,
                },
                {},
            )

            with (
                mock.patch("strip_insar.load_scene_corners_with_fallback", return_value=[]),
                mock.patch("strip_insar._load_master_metadata", side_effect=[meta, meta, meta]),
                mock.patch("strip_insar.choose_orbit_interp", return_value="Legendre"),
                mock.patch("strip_insar._resolve_dem_path", return_value="/tmp/dem.tif"),
                mock.patch("strip_insar.select_processing_backend", return_value=("gpu", "GPU available")),
                mock.patch("strip_insar.query_gpu_memory_info", return_value=None),
                mock.patch("strip_insar._process_insar_gpu", return_value={"interferogram_h5": str(h5_path)}),
                mock.patch("strip_insar._process_insar_cpu") as mock_cpu,
                mock.patch("strip_insar.compute_utm_output_shape", return_value=(32, 16)),
                mock.patch("strip_insar.geocode_product", side_effect=RuntimeError("p6 publish failed")),
            ):
                with self.assertRaisesRegex(RuntimeError, "p6 publish failed"):
                    process_strip_insar(
                        str(master_manifest),
                        str(slave_manifest),
                        str(output_dir),
                        gpu_mode="gpu",
                    )

            mock_cpu.assert_not_called()
            self.assertTrue((output_dir / "work" / "p5_hdf" / "SUCCESS").exists())
            self.assertFalse((output_dir / "work" / "p6_publish" / "SUCCESS").exists())


if __name__ == "__main__":
    unittest.main()
