import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import ast
import numpy as np

from osgeo import gdal, osr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

if "h5py" not in sys.modules:
    sys.modules["h5py"] = mock.MagicMock()

import strip_insar2
from strip_insar2_export import write_ground_overlay_kml_from_geotiff


class StripInsar2IndependenceTests(unittest.TestCase):
    def test_strip_insar2_does_not_import_local_processing_modules(self):
        source_path = SCRIPTS_DIR / "strip_insar2.py"
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(source_path))

        imported_modules = set()
        imported_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported_modules.add(node.module)
                imported_names.update(alias.name for alias in node.names)

        forbidden_modules = {
            "strip_insar",
            "common_processing",
            "insar_registration",
            "insar_filtering",
            "insar_stage_cache",
            "strip_insar2_export",
        }
        self.assertTrue(
            forbidden_modules.isdisjoint(imported_modules),
            msg=f"unexpected local imports: {sorted(forbidden_modules & imported_modules)}",
        )
        for legacy_name in (
            "_run_p1_stage_from_cache",
            "_run_p2_stage_from_cache",
            "_run_p3_stage_from_cache",
            "_run_p4_stage_from_cache",
            "_create_unwrapper",
        ):
            self.assertNotIn(legacy_name, imported_names)


class StripInsar2NamingTests(unittest.TestCase):
    def test_pair_name_and_output_paths_use_master_slave_dates(self):
        master_date = strip_insar2.extract_scene_date(
            {"startTimeUTC": "2023-11-10T04:39:48.881889"},
            {"header": {"firstStateTimeUTC": "2023-11-10T04:39:48Z"}},
        )
        slave_date = strip_insar2.extract_scene_date(
            {"startTimeUTC": "2023-11-21T04:39:48.881889"},
            {"header": {"firstStateTimeUTC": "2023-11-21T04:39:48Z"}},
        )

        pair_name = strip_insar2.build_pair_name(master_date, slave_date)
        self.assertEqual(pair_name, "20231110_20231121")

        output_paths = strip_insar2.build_output_paths("/tmp/out", pair_name)
        self.assertTrue(output_paths["interferogram_h5"].endswith("20231110_20231121_insar.h5"))
        self.assertTrue(
            output_paths["interferogram_png"].endswith(
                "20231110_20231121_interferogram_wrapped_phase_utm_geocoded.png"
            )
        )
        self.assertTrue(
            output_paths["coherence_tif"].endswith(
                "20231110_20231121_coherence_utm_geocoded.tif"
            )
        )

    def test_resolve_manifest_data_path_builds_vsizip_member_path(self):
        manifest_path = Path("/tmp/example/manifest.json")
        entry = {
            "path": "/temp/example.zip",
            "storage": "zip",
            "member": "folder/data.tiff",
        }

        resolved = strip_insar2.resolve_manifest_data_path(manifest_path, entry)

        self.assertEqual(resolved, "/vsizip//temp/example.zip/folder/data.tiff")


class StripInsar2KmlTests(unittest.TestCase):
    def test_write_ground_overlay_kml_from_geotiff_uses_geographic_bounds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tif_path = root / "coherence.tif"
            png_path = root / "coherence.png"
            kml_path = root / "coherence.kml"
            png_path.write_bytes(b"png")

            drv = gdal.GetDriverByName("GTiff")
            ds = drv.Create(str(tif_path), 2, 2, 1, gdal.GDT_Float32)
            ds.SetGeoTransform([100.0, 1.0, 0.0, 30.0, 0.0, -1.0])
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(4326)
            srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            ds.SetProjection(srs.ExportToWkt())
            ds.GetRasterBand(1).Fill(1.0)
            ds.FlushCache()
            ds = None

            write_ground_overlay_kml_from_geotiff(
                tif_path=str(tif_path),
                image_path=str(png_path),
                output_kml=str(kml_path),
                overlay_name="coherence",
            )

            self.assertTrue(kml_path.exists())
            text = kml_path.read_text(encoding="utf-8")
            self.assertIn("<GroundOverlay>", text)
            self.assertIn("<href>coherence.png</href>", text)
            self.assertIn("<west>100.0</west>", text)
            self.assertIn("<east>102.0</east>", text)
            self.assertIn("<north>30.0</north>", text)
            self.assertIn("<south>28.0</south>", text)


class StripInsar2TopoTests(unittest.TestCase):
    def test_build_topo_vrt_sets_projection_from_epsg(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            driver = gdal.GetDriverByName("GTiff")
            for name in ("x.tif", "y.tif", "z.tif"):
                ds = driver.Create(str(root / name), 2, 2, 1, gdal.GDT_Float32)
                ds.GetRasterBand(1).Fill(1.0)
                ds.FlushCache()
                ds = None

            vrt_path = strip_insar2._build_topo_vrt(root, epsg=4326)
            ds = gdal.Open(vrt_path, gdal.GA_ReadOnly)
            self.assertIsNotNone(ds)
            projection = ds.GetProjectionRef()
            ds = None

            self.assertTrue(projection)
            self.assertIn("4326", projection)


class StripInsar2FallbackTests(unittest.TestCase):
    def test_run_stage_with_fallback_uses_cpu_when_gpu_stage_fails(self):
        calls: list[str] = []

        def _gpu():
            calls.append("gpu")
            raise RuntimeError("cuda crossmul failed")

        def _cpu():
            calls.append("cpu")
            return {"ok": True}

        result, backend_used, fallback_reason = strip_insar2.run_stage_with_fallback(
            stage_name="crossmul",
            gpu_mode="auto",
            gpu_id=0,
            gpu_runner=_gpu,
            cpu_runner=_cpu,
            gpu_check=lambda _requested, _gpu_id: True,
        )

        self.assertEqual(result, {"ok": True})
        self.assertEqual(backend_used, "cpu")
        self.assertIn("cuda crossmul failed", fallback_reason)
        self.assertEqual(calls, ["gpu", "cpu"])


class StripInsar2StageCacheTests(unittest.TestCase):
    def test_run_geo2rdr_stage_reuses_cached_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pair_dir = root / "20231110_20231121"
            p0_dir = pair_dir / "work" / "p0_geo2rdr"
            p0_dir.mkdir(parents=True, exist_ok=True)

            master_topo = p0_dir / "master_topo.vrt"
            slave_topo = p0_dir / "slave_topo.vrt"
            master_topo.write_text("master", encoding="utf-8")
            slave_topo.write_text("slave", encoding="utf-8")

            strip_insar2.write_stage_record(
                pair_dir,
                "p0",
                {
                    "stage": "p0",
                    "backend_used": "gpu",
                    "success": True,
                    "output_files": {
                        "master_topo": str(master_topo),
                        "slave_topo": str(slave_topo),
                        "master_topo_vrt": str(master_topo),
                        "slave_topo_vrt": str(slave_topo),
                    },
                },
            )
            strip_insar2.mark_stage_success(pair_dir, "p0")

            context = mock.Mock()
            context.pair_dir = pair_dir
            context.master_manifest_path = root / "master.json"
            context.slave_manifest_path = root / "slave.json"

            with mock.patch(
                "strip_insar2._run_rdr2geo_topo",
                side_effect=AssertionError("p0 should have been reused from cache"),
            ):
                output_files, backend_used, fallback_reason = strip_insar2.run_geo2rdr_stage(
                    context,
                    gpu_mode="auto",
                    gpu_id=0,
                    block_rows=256,
                )

            self.assertEqual(output_files["master_topo"], str(master_topo))
            self.assertEqual(output_files["slave_topo"], str(slave_topo))
            self.assertEqual(backend_used, "cache")
            self.assertIsNone(fallback_reason)


class StripInsar2ImportPathTests(unittest.TestCase):
    def test_ensure_nisar_python_packages_on_path_prefers_installed_packages(self):
        original_path = list(strip_insar2.sys.path)
        try:
            strip_insar2.sys.path[:] = ["/installed/a", "/installed/b"]

            with mock.patch(
                "strip_insar2.importlib.import_module",
                side_effect=lambda name: object() if name in {"isce3", "nisar"} else None,
            ):
                packages_dir = strip_insar2._ensure_nisar_python_packages_on_path()

            self.assertIsNone(packages_dir)
            self.assertEqual(strip_insar2.sys.path, ["/installed/a", "/installed/b"])
        finally:
            strip_insar2.sys.path[:] = original_path


class StripInsar2RasterTranslateTests(unittest.TestCase):
    def test_translate_raster_omits_outsize_when_width_height_not_set(self):
        fake_dataset = object()
        translated = mock.Mock()

        with (
            mock.patch("strip_insar2.gdal.Open", return_value=fake_dataset),
            mock.patch("strip_insar2.gdal.Translate", return_value=translated) as translate,
        ):
            result = strip_insar2._translate_raster(
                "/tmp/src.tif",
                Path("/tmp/out.bin"),
                driver="ENVI",
            )

        self.assertEqual(result, "/tmp/out.bin")
        kwargs = translate.call_args.kwargs
        self.assertEqual(kwargs["format"], "ENVI")
        self.assertEqual(kwargs["resampleAlg"], "nearest")
        self.assertNotIn("width", kwargs)
        self.assertNotIn("height", kwargs)

    def test_translate_raster_passes_output_type_when_requested(self):
        fake_dataset = object()
        translated = mock.Mock()

        with (
            mock.patch("strip_insar2.gdal.Open", return_value=fake_dataset),
            mock.patch("strip_insar2.gdal.Translate", return_value=translated) as translate,
        ):
            strip_insar2._translate_raster(
                "/tmp/src.tif",
                Path("/tmp/out.bin"),
                driver="ENVI",
                output_type=gdal.GDT_CFloat32,
            )

        kwargs = translate.call_args.kwargs
        self.assertEqual(kwargs["outputType"], gdal.GDT_CFloat32)


class StripInsar2RegistrationTests(unittest.TestCase):
    def test_run_nisar_registration_chain_uses_coregistration_zero_doppler(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            p1_stage_path = root / "p1"
            p1_stage_path.mkdir(parents=True, exist_ok=True)
            topo_vrt = root / "topo.vrt"
            topo_vrt.write_text("vrt", encoding="utf-8")

            context = mock.Mock()
            context.master_manifest_path = root / "master.json"
            context.slave_manifest_path = root / "slave.json"
            context.master_manifest = {"slc": {"path": "master.tif"}}
            context.slave_manifest = {"slc": {"path": "slave.tif"}}
            context.master_rg_data = {}
            context.master_acq_data = {}
            context.master_orbit_data = {}
            context.slave_rg_data = {}
            context.slave_acq_data = {}
            context.slave_orbit_data = {}
            context.slave_dop_data = {}

            ref_grid = mock.Mock(length=2, width=3)
            sec_grid = mock.Mock(length=2, width=3)
            zero_doppler = object()

            with (
                mock.patch(
                    "strip_insar2.resolve_manifest_data_path",
                    side_effect=["/tmp/slave.tif", "/tmp/master.tif"],
                ),
                mock.patch(
                    "strip_insar2.construct_radar_grid",
                    side_effect=[ref_grid, sec_grid],
                ),
                mock.patch(
                    "strip_insar2._build_coregistration_doppler_lut",
                    return_value=zero_doppler,
                ),
                mock.patch(
                    "strip_insar2.load_stage_record",
                    return_value={"output_files": {"master_topo_vrt": str(topo_vrt)}},
                ),
                mock.patch(
                    "strip_insar2._run_slave_geo2rdr_from_master_topo",
                    return_value=("/tmp/coarse_rg.tif", "/tmp/coarse_az.tif"),
                ),
                mock.patch(
                    "strip_insar2._prepare_nisar_geo2rdr_offsets",
                    return_value=("/tmp/coarse_rg.off", "/tmp/coarse_az.off"),
                ),
                mock.patch(
                    "strip_insar2.run_coarse_resamp_isce3_v2",
                    return_value=True,
                ) as coarse_resamp,
                mock.patch(
                    "strip_insar2._run_nisar_dense_offsets",
                    return_value={"dense_offsets": "/tmp/dense_offsets"},
                ),
                mock.patch(
                    "strip_insar2._run_nisar_rubbersheet",
                    return_value={
                        "range_offsets": "/tmp/range.off",
                        "azimuth_offsets": "/tmp/azimuth.off",
                        "resampled_range_offsets": "/tmp/resampled_range.off",
                        "resampled_azimuth_offsets": "/tmp/resampled_azimuth.off",
                    },
                ),
                mock.patch(
                    "strip_insar2._read_raster_array",
                    return_value=np.zeros((2, 3), dtype=np.float64),
                ),
                mock.patch(
                    "strip_insar2._write_offset_raster",
                    side_effect=lambda path, _arr: str(path),
                ),
            ):
                strip_insar2._run_nisar_registration_chain(
                    context=context,
                    use_gpu=False,
                    gpu_id=0,
                    p1_stage_path=p1_stage_path,
                )

            self.assertEqual(coarse_resamp.call_count, 2)
            for call in coarse_resamp.call_args_list:
                self.assertIs(call.kwargs["doppler"], zero_doppler)


class StripInsar2DenseOffsetsTests(unittest.TestCase):
    def test_run_nisar_dense_offsets_falls_back_to_cpu_when_gpu_subprocess_fails(self):
        cpu_outputs = {"dense_offsets": "/tmp/dense_offsets"}

        with (
            mock.patch(
                "strip_insar2._run_nisar_dense_offsets_gpu_subprocess",
                side_effect=RuntimeError("gpu child failed"),
            ) as gpu_subprocess,
            mock.patch(
                "strip_insar2._run_nisar_dense_offsets_impl",
                return_value=cpu_outputs,
            ) as impl,
        ):
            outputs = strip_insar2._run_nisar_dense_offsets(
                reference_slc_path="/tmp/ref.slc",
                secondary_slc_path="/tmp/sec.slc",
                output_dir=Path("/tmp/out"),
                use_gpu=True,
                gpu_id=0,
            )

        self.assertEqual(outputs, cpu_outputs)
        gpu_subprocess.assert_called_once()
        impl.assert_called_once_with(
            reference_slc_path="/tmp/ref.slc",
            secondary_slc_path="/tmp/sec.slc",
            output_dir=Path("/tmp/out"),
            use_gpu=False,
            gpu_id=0,
        )

    def test_run_nisar_dense_offsets_returns_gpu_subprocess_result_on_success(self):
        gpu_outputs = {"dense_offsets": "/tmp/gpu_dense_offsets"}

        with (
            mock.patch(
                "strip_insar2._run_nisar_dense_offsets_gpu_subprocess",
                return_value=gpu_outputs,
            ) as gpu_subprocess,
            mock.patch("strip_insar2._run_nisar_dense_offsets_impl") as impl,
        ):
            outputs = strip_insar2._run_nisar_dense_offsets(
                reference_slc_path="/tmp/ref.slc",
                secondary_slc_path="/tmp/sec.slc",
                output_dir=Path("/tmp/out"),
                use_gpu=True,
                gpu_id=0,
            )

        self.assertEqual(outputs, gpu_outputs)
        gpu_subprocess.assert_called_once()
        impl.assert_not_called()


class StripInsar2ProcessTests(unittest.TestCase):
    def _write_manifest(self, root: Path, name: str, start_time: str) -> Path:
        metadata_dir = root / f"{name}_metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        (metadata_dir / "acquisition.json").write_text(
            json.dumps(
                {
                    "startTimeUTC": start_time,
                    "centerFrequency": 5400000100.0,
                    "prf": 4105.0903,
                    "startGPSTime": 1383626388.881889,
                }
            ),
            encoding="utf-8",
        )
        (metadata_dir / "orbit.json").write_text(
            json.dumps({"header": {"firstStateTimeUTC": start_time}}),
            encoding="utf-8",
        )
        (metadata_dir / "radargrid.json").write_text(
            json.dumps(
                {
                    "numberOfRows": 4,
                    "numberOfColumns": 4,
                    "rangeTimeFirstPixel": 0.004179906307,
                    "columnSpacing": 1.249135241667,
                }
            ),
            encoding="utf-8",
        )
        (metadata_dir / "doppler.json").write_text(
            json.dumps({"combinedDoppler": {"x": [0.0], "y": [0.0], "values": [[0.0]]}}),
            encoding="utf-8",
        )
        manifest_path = root / f"{name}.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "sensor": "tianyi",
                    "slc": {"path": f"{name}.tif"},
                    "metadata": {
                        "acquisition": str(metadata_dir / "acquisition.json"),
                        "orbit": str(metadata_dir / "orbit.json"),
                        "radargrid": str(metadata_dir / "radargrid.json"),
                        "doppler": str(metadata_dir / "doppler.json"),
                    },
                }
            ),
            encoding="utf-8",
        )
        return manifest_path

    def test_process_strip_insar2_builds_pair_output_directory_and_named_h5(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            master_manifest = self._write_manifest(root, "master", "2023-11-10T04:39:48.881889")
            slave_manifest = self._write_manifest(root, "slave", "2023-11-21T04:39:48.881889")
            output_root = root / "results"

            with (
                mock.patch("strip_insar2.run_geo2rdr_stage", return_value=({"master_topo": "a", "slave_topo": "b"}, "cpu", None)),
                mock.patch("strip_insar2.run_resample_stage", return_value=({"fine_coreg_slave": "fine.tif"}, "cpu", None)),
                mock.patch("strip_insar2.run_crossmul_stage", return_value=({"interferogram": "ifg.npy"}, "cpu", None)),
                mock.patch("strip_insar2.run_unwrap_stage", return_value=({"unwrapped_phase": "unw.npy"}, "cpu", None)),
                mock.patch("strip_insar2.run_los_stage", return_value=({"los_displacement": "los.npy"}, "cpu", None)),
                mock.patch("strip_insar2.write_primary_product", return_value=("/tmp/fake.h5", "cpu", None)),
                mock.patch("strip_insar2.export_insar_products", return_value={"interferogram_png": "ifg.png"}),
            ):
                result = strip_insar2.process_strip_insar2(
                    str(master_manifest),
                    str(slave_manifest),
                    output_root=str(output_root),
                    gpu_mode="cpu",
                    dem_path=str(root / "dem.tif"),
                )

            self.assertEqual(result["pair_name"], "20231110_20231121")
            self.assertEqual(
                Path(result["pair_dir"]),
                output_root / "20231110_20231121",
            )
            self.assertTrue(
                result["output_paths"]["interferogram_h5"].endswith(
                    "20231110_20231121_insar.h5"
                )
            )

    def test_run_resample_stage_delegates_to_nisar_registration_chain(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            master_manifest = self._write_manifest(root, "master", "2023-11-10T04:39:48.881889")
            slave_manifest = self._write_manifest(root, "slave", "2023-11-21T04:39:48.881889")
            output_root = root / "results"
            context = strip_insar2.load_pair_context(
                str(master_manifest),
                str(slave_manifest),
                output_root=str(output_root),
                dem_path=str(root / "dem.tif"),
            )

            expected_outputs = {
                "coarse_coreg_slave": str(output_root / "20231110_20231121" / "coarse_coreg_slave.tif"),
                "fine_coreg_slave": str(output_root / "20231110_20231121" / "fine_coreg_slave.tif"),
                "range_offsets": str(output_root / "20231110_20231121" / "range.off.tif"),
                "azimuth_offsets": str(output_root / "20231110_20231121" / "azimuth.off.tif"),
                "registration_model": str(output_root / "20231110_20231121" / "registration_model.json"),
            }

            with mock.patch(
                "strip_insar2._run_nisar_registration_chain",
                return_value=expected_outputs,
            ) as registration_chain:
                outputs, backend_used, fallback_reason = strip_insar2.run_resample_stage(
                    context,
                    gpu_mode="cpu",
                    gpu_id=0,
                )

            self.assertEqual(outputs, expected_outputs)
            self.assertEqual(backend_used, "cpu")
            self.assertIsNone(fallback_reason)
            registration_chain.assert_called_once()


if __name__ == "__main__":
    unittest.main()
