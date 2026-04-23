import json
import sys
import tempfile
import types
import unittest
from datetime import datetime, timezone
from unittest import mock
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


from strip_rtc import (  # noqa: E402
    choose_gpu_topo_block_rows,
    build_output_paths,
    detect_sensor_from_manifest,
    process_strip_rtc,
    select_processing_backend,
)
from common_processing import construct_doppler_lut2d, resolve_manifest_metadata_path  # noqa: E402


class StripRtcTests(unittest.TestCase):
    def _write_manifest(self, tmpdir: str, payload: dict) -> Path:
        manifest_path = Path(tmpdir) / "manifest.json"
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        return manifest_path

    def test_construct_doppler_lut2d_uses_azimuth_time_and_slant_range_axes(self):
        class FakeLUT2d:
            def __init__(self, xstart, ystart, dx, dy, data, method, b_error):
                self.xstart = xstart
                self.ystart = ystart
                self.dx = dx
                self.dy = dy
                self.data = data
                self.method = method
                self.b_error = b_error

        fake_core = types.SimpleNamespace(
            LUT2d=FakeLUT2d,
            speed_of_light=299792458.0,
        )
        fake_isce3 = types.SimpleNamespace(core=fake_core)
        doppler = {
            "combinedDoppler": {
                "validityRangeMin": 0.004,
                "validityRangeMax": 0.00400002,
                "referencePoint": 0.004,
                "polynomialDegree": 1,
                "coefficients": [0.01, 2.0],
            }
        }
        radargrid = {
            "numberOfRows": 4,
            "numberOfColumns": 3,
            "rangeTimeFirstPixel": 0.004,
            "columnSpacing": 5.0,
        }
        acquisition = {
            "startGPSTime": 1383626388.881889,
            "prf": 4.0,
        }
        orbit = {
            "header": {
                "firstStateTimeUTC": "2023-11-10T04:38:00.000000Z",
            }
        }

        with mock.patch.dict(sys.modules, {"isce3": fake_isce3, "isce3.core": fake_core}):
            lut = construct_doppler_lut2d(
                doppler,
                radargrid_json=radargrid,
                acquisition_json=acquisition,
                orbit_json=orbit,
            )

        c0 = 299792458.0 * radargrid["rangeTimeFirstPixel"] / 2.0
        self.assertAlmostEqual(lut.xstart, c0)
        self.assertAlmostEqual(lut.dx, radargrid["columnSpacing"])
        gps_epoch = datetime(1980, 1, 6, tzinfo=timezone.utc)
        orbit_ref = datetime.fromisoformat("2023-11-10T04:38:00+00:00")
        expected_ystart = acquisition["startGPSTime"] - (orbit_ref - gps_epoch).total_seconds()
        self.assertAlmostEqual(lut.ystart, expected_ystart, places=6)
        self.assertAlmostEqual(lut.dy, 1.0)
        self.assertEqual(lut.data.shape, (2, 4))
        np.testing.assert_allclose(lut.data[0], lut.data[1])
        tau0 = 2.0 * c0 / 299792458.0
        expected0 = 0.01 + 2.0 * (tau0 - doppler["combinedDoppler"]["referencePoint"])
        self.assertAlmostEqual(float(lut.data[0, 0]), expected0)

    def test_detect_sensor_from_manifest_accepts_tianyi(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = self._write_manifest(tmpdir, {"sensor": "tianyi"})
            self.assertEqual(detect_sensor_from_manifest(manifest_path), "tianyi")

    def test_detect_sensor_from_manifest_accepts_lutan(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = self._write_manifest(tmpdir, {"sensor": "LUTAN"})
            self.assertEqual(detect_sensor_from_manifest(manifest_path), "lutan")

    def test_detect_sensor_from_manifest_rejects_unknown_sensor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = self._write_manifest(tmpdir, {"sensor": "sentinel"})
            with self.assertRaisesRegex(ValueError, "Unsupported sensor"):
                detect_sensor_from_manifest(manifest_path)

    def test_build_output_paths_uses_unified_names(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = build_output_paths(tmpdir)
            self.assertEqual(
                paths["amplitude_h5"], str(Path(tmpdir) / "amplitude_fullres.h5")
            )
            self.assertEqual(
                paths["amplitude_utm_tif"],
                str(Path(tmpdir) / "amplitude_utm_geocoded.tif"),
            )
            self.assertEqual(
                paths["amplitude_utm_png"],
                str(Path(tmpdir) / "amplitude_utm_geocoded.png"),
            )
            self.assertEqual(
                paths["dem_validation"], str(Path(tmpdir) / "dem_validation.json")
            )

    def test_select_processing_backend_prefers_gpu_when_available(self):
        backend, reason = select_processing_backend(
            gpu_mode="auto",
            gpu_id=0,
            gpu_check=lambda gpu_requested, gpu_id: True,
        )
        self.assertEqual(backend, "gpu")
        self.assertIn("available", reason.lower())

    def test_select_processing_backend_falls_back_to_cpu_when_gpu_unavailable(self):
        backend, reason = select_processing_backend(
            gpu_mode="auto",
            gpu_id=0,
            gpu_check=lambda gpu_requested, gpu_id: False,
        )
        self.assertEqual(backend, "cpu")
        self.assertIn("fallback", reason.lower())

    def test_select_processing_backend_obeys_explicit_cpu_mode(self):
        backend, reason = select_processing_backend(
            gpu_mode="cpu",
            gpu_id=0,
            gpu_check=lambda gpu_requested, gpu_id: True,
        )
        self.assertEqual(backend, "cpu")
        self.assertIn("forced", reason.lower())

    def test_choose_gpu_topo_block_rows_uses_conservative_vram_budget(self):
        topo_rows, reason = choose_gpu_topo_block_rows(
            width=20000,
            default_block_rows=256,
            memory_info={
                "total_bytes": 24 * 1024**3,
                "free_bytes": 20 * 1024**3,
            },
        )
        self.assertEqual(topo_rows, 1024)
        self.assertIn("adaptive", reason.lower())

    def test_choose_gpu_topo_block_rows_falls_back_when_memory_unknown(self):
        topo_rows, reason = choose_gpu_topo_block_rows(
            width=20000,
            default_block_rows=256,
            memory_info=None,
        )
        self.assertEqual(topo_rows, 256)
        self.assertIn("default", reason.lower())

    def test_process_strip_rtc_retries_gpu_topo_with_smaller_blocks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = self._write_manifest(
                tmpdir, {"sensor": "tianyi", "slc": {"path": "dummy.tif"}}
            )
            output_dir = Path(tmpdir) / "out"
            gpu_calls = []

            def fake_process_strip_gpu(**kwargs):
                gpu_calls.append(kwargs["topo_block_rows"])
                if kwargs["topo_block_rows"] in (512, 1024):
                    raise RuntimeError("CUDA out of memory during topo")
                return build_output_paths(output_dir)

            with (
                mock.patch(
                    "strip_rtc.load_scene_corners_with_fallback", return_value=[]
                ),
                mock.patch(
                    "strip_rtc._load_processing_metadata", return_value=({}, {})
                ),
                mock.patch(
                    "strip_rtc._load_radargrid_metadata",
                    return_value={
                        "groundRangeResolution": 1.0,
                        "azimuthResolution": 1.0,
                        "numberOfColumns": 20000,
                    },
                ),
                mock.patch("strip_rtc.choose_orbit_interp", return_value="Legendre"),
                mock.patch("strip_rtc._resolve_dem_path", return_value="/tmp/dem.tif"),
                mock.patch(
                    "strip_rtc.select_processing_backend",
                    return_value=("gpu", "GPU available"),
                ),
                mock.patch(
                    "strip_rtc.query_gpu_memory_info",
                    return_value={
                        "total_bytes": 24 * 1024**3,
                        "free_bytes": 20 * 1024**3,
                    },
                ),
                mock.patch(
                    "strip_rtc._process_strip_gpu",
                    side_effect=fake_process_strip_gpu,
                ),
            ):
                result = process_strip_rtc(str(manifest_path), str(output_dir))

            self.assertEqual(gpu_calls, [1024, 512, 256])
            self.assertEqual(result["backend_used"], "gpu")
            self.assertEqual(result["topo_block_rows"], 256)

    def test_process_strip_rtc_reports_hybrid_stage_backends(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = self._write_manifest(
                tmpdir, {"sensor": "tianyi", "slc": {"path": "dummy.tif"}}
            )
            output_dir = Path(tmpdir) / "out"

            with (
                mock.patch(
                    "strip_rtc.load_scene_corners_with_fallback", return_value=[]
                ),
                mock.patch(
                    "strip_rtc._load_processing_metadata", return_value=({}, {})
                ),
                mock.patch(
                    "strip_rtc._load_radargrid_metadata",
                    return_value={
                        "groundRangeResolution": 1.0,
                        "azimuthResolution": 1.0,
                    },
                ),
                mock.patch("strip_rtc.choose_orbit_interp", return_value="Legendre"),
                mock.patch("strip_rtc._resolve_dem_path", return_value="/tmp/dem.tif"),
                mock.patch(
                    "strip_rtc.select_processing_backend",
                    return_value=("gpu", "GPU available"),
                ),
                mock.patch(
                    "strip_rtc._process_strip_gpu",
                    return_value=build_output_paths(output_dir),
                ),
            ):
                result = process_strip_rtc(str(manifest_path), str(output_dir))

            self.assertEqual(result["backend_used"], "gpu")
            self.assertEqual(result["pipeline_mode"], "hybrid")
            self.assertEqual(result["stage_backends"]["rtc_factor"], "cpu")
            self.assertEqual(result["stage_backends"]["topo_lonlatheight"], "gpu")

    def test_process_strip_rtc_reports_gpu_fallback_reason(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = self._write_manifest(
                tmpdir, {"sensor": "lutan", "slc": {"path": "dummy.tif"}}
            )
            output_dir = Path(tmpdir) / "out"

            with (
                mock.patch(
                    "strip_rtc.load_scene_corners_with_fallback", return_value=[]
                ),
                mock.patch(
                    "strip_rtc._load_processing_metadata", return_value=({}, {})
                ),
                mock.patch(
                    "strip_rtc._load_radargrid_metadata",
                    return_value={
                        "groundRangeResolution": 1.0,
                        "azimuthResolution": 1.0,
                    },
                ),
                mock.patch("strip_rtc.choose_orbit_interp", return_value="Hermite"),
                mock.patch("strip_rtc._resolve_dem_path", return_value="/tmp/dem.tif"),
                mock.patch(
                    "strip_rtc.select_processing_backend",
                    return_value=("gpu", "GPU available"),
                ),
                mock.patch(
                    "strip_rtc._process_strip_gpu",
                    side_effect=RuntimeError("gpu topo failed"),
                ),
                mock.patch(
                    "strip_rtc._process_strip_cpu",
                    return_value=build_output_paths(output_dir),
                ),
            ):
                result = process_strip_rtc(str(manifest_path), str(output_dir))

            self.assertEqual(result["backend_used"], "cpu")
            self.assertEqual(result["pipeline_mode"], "cpu-fallback")
            self.assertIn("gpu topo failed", result["fallback_reasons"]["pipeline"])

    def test_resolve_manifest_metadata_path_falls_back_to_local_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_dir = Path(tmpdir)
            metadata_dir = manifest_dir / "metadata"
            metadata_dir.mkdir()
            local_scene = metadata_dir / "scene.json"
            local_scene.write_text("{}", encoding="utf-8")
            manifest_path = manifest_dir / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {"metadata": {"scene": "/work/results/example/metadata/scene.json"}}
                ),
                encoding="utf-8",
            )

            resolved = resolve_manifest_metadata_path(
                manifest_path,
                json.loads(manifest_path.read_text(encoding="utf-8")),
                "scene",
            )

            self.assertEqual(resolved, local_scene)


if __name__ == "__main__":
    unittest.main()
