import json
import sys
import tempfile
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock
import ast
import numpy as np
import types

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

    def test_convert_geo2rdr_output_to_gtiff_preserves_float64(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "range.off"
            output = root / "range.off.tif"
            data = np.array([[10.125, 20.25], [30.5, 40.75]], dtype=np.float64)
            data.tofile(source)

            strip_insar2._convert_geo2rdr_output_to_gtiff(
                source_path=source,
                rows=2,
                cols=2,
                output_path=output,
            )

            ds = gdal.Open(str(output), gdal.GA_ReadOnly)
            self.assertIsNotNone(ds)
            self.assertEqual(ds.GetRasterBand(1).DataType, gdal.GDT_Float64)
            out = ds.GetRasterBand(1).ReadAsArray().astype(np.float64)
            ds = None

        np.testing.assert_allclose(out, data)

    def test_append_topo_coordinates_hdf_uses_float64_xyz(self):
        class FakeRaster:
            created = []

            def __init__(self, *args, **kwargs):
                if len(args) >= 6:
                    FakeRaster.created.append((Path(args[0]).name, args[4]))

            def close_dataset(self):
                pass

        class FakeTopo:
            def __init__(self, *args, **kwargs):
                pass

            def topo(self, *args, **kwargs):
                raise RuntimeError("stop_after_raster_creation")

        class FakeGrid:
            width = 16
            length = 8

        fake_isce3 = types.SimpleNamespace(
            io=types.SimpleNamespace(Raster=FakeRaster),
            geometry=types.SimpleNamespace(Rdr2Geo=FakeTopo),
            core=types.SimpleNamespace(
                Ellipsoid=mock.Mock(return_value=object()),
                LUT2d=mock.Mock(return_value=object()),
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps({"metadata": {}}), encoding="utf-8")
            output_h5 = root / "out.h5"
            radargrid_path = root / "radargrid.json"
            orbit_path = root / "orbit.json"
            acquisition_path = root / "acquisition.json"
            radargrid_path.write_text(json.dumps({"numberOfColumns": 16, "numberOfRows": 8}), encoding="utf-8")
            orbit_path.write_text(json.dumps({}), encoding="utf-8")
            acquisition_path.write_text(json.dumps({}), encoding="utf-8")

            def fake_resolve(_manifest_path, _manifest, key):
                return str(
                    {
                        "radargrid": radargrid_path,
                        "orbit": orbit_path,
                        "acquisition": acquisition_path,
                    }[key]
                )

            FakeRaster.created = []
            with (
                mock.patch("strip_insar2.resolve_manifest_metadata_path", side_effect=fake_resolve),
                mock.patch("strip_insar2.choose_orbit_interp", return_value="Legendre"),
                mock.patch("strip_insar2.construct_orbit", return_value=object()),
                mock.patch("strip_insar2.construct_radar_grid", return_value=FakeGrid()),
                mock.patch.dict(
                    sys.modules,
                    {
                        "isce3": fake_isce3,
                        "isce3.io": fake_isce3.io,
                        "isce3.core": fake_isce3.core,
                        "isce3.geometry": fake_isce3.geometry,
                    },
                ),
                self.assertRaisesRegex(RuntimeError, "stop_after_raster_creation"),
            ):
                strip_insar2.append_topo_coordinates_hdf(
                    manifest_path=str(manifest_path),
                    dem_path="/tmp/dem.tif",
                    output_h5=str(output_h5),
                    block_rows=8,
                    use_gpu=False,
                )

        topo_rasters = [
            (name, dtype)
            for name, dtype in FakeRaster.created
            if name in {"x.tif", "y.tif", "z.tif"}
        ]
        self.assertEqual(len(topo_rasters), 3)
        self.assertTrue(
            all(dtype == gdal.GDT_Float64 for _name, dtype in topo_rasters),
            topo_rasters,
        )

    def test_run_rdr2geo_topo_silences_isce3_journal(self):
        class FakeRaster:
            def __init__(self, *args, **kwargs):
                pass

            def close_dataset(self):
                pass

        class FakeTopo:
            epsg_out = 4326

            def __init__(self, *args, **kwargs):
                pass

            def topo(self, *args, **kwargs):
                return None

        fake_isce3 = types.SimpleNamespace(
            io=types.SimpleNamespace(Raster=FakeRaster),
            geometry=types.SimpleNamespace(Rdr2Geo=FakeTopo),
            core=types.SimpleNamespace(
                Ellipsoid=mock.Mock(return_value=object()),
                LUT2d=mock.Mock(return_value=object()),
            ),
        )

        called_paths = []

        @contextmanager
        def fake_silence(path):
            called_paths.append(Path(path))
            yield str(path)

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir) / "master_topo"
            with (
                mock.patch("strip_insar2.construct_orbit", return_value=object()),
                mock.patch("strip_insar2.construct_radar_grid", return_value=mock.Mock(width=4, length=3)),
                mock.patch("strip_insar2._build_topo_vrt", return_value=str(outdir / "topo.vrt")),
                mock.patch("strip_insar2._silence_isce3_journal", side_effect=fake_silence),
                mock.patch.dict(
                    sys.modules,
                    {
                        "isce3": fake_isce3,
                        "isce3.io": fake_isce3.io,
                        "isce3.core": fake_isce3.core,
                        "isce3.geometry": fake_isce3.geometry,
                    },
                ),
            ):
                strip_insar2._run_rdr2geo_topo(
                    orbit_data={},
                    acquisition_data={},
                    radargrid_data={},
                    dem_path="/tmp/dem.tif",
                    orbit_interp="Legendre",
                    use_gpu=False,
                    gpu_id=0,
                    output_dir=outdir,
                    block_rows=64,
                )

        self.assertEqual(called_paths, [outdir / "isce3_journal.log"])

    def test_append_topo_coordinates_hdf_writes_lon_lat_hgt_as_float64(self):
        class FakeProgress:
            def __init__(self):
                self.events = []

            def block(self, *, backend, current, total, detail):
                self.events.append((backend, current, total, detail))

        class FakeRaster:
            def __init__(self, *args, **kwargs):
                pass

            def close_dataset(self):
                pass

        class FakeTopo:
            def __init__(self, *args, **kwargs):
                pass

            def topo(self, *args, **kwargs):
                return None

        class FakeGrid:
            width = 4
            length = 3

        class FakeBand:
            pass

        class FakeGdalDataset:
            def GetRasterBand(self, _index):
                return FakeBand()

        class FakeWritableDataset:
            def __init__(self, dtype):
                self.dtype = dtype
                self.data = None

            def __setitem__(self, key, value):
                self.data = np.asarray(value)

        class FakeH5File:
            last = None

            def __init__(self, *args, **kwargs):
                self.datasets = {}
                self.attrs = {}
                FakeH5File.last = self

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def __contains__(self, key):
                return key in self.datasets

            def __delitem__(self, key):
                self.datasets.pop(key, None)

            def create_dataset(self, name, **kwargs):
                ds = FakeWritableDataset(kwargs.get("dtype"))
                self.datasets[name] = ds
                return ds

        fake_isce3 = types.SimpleNamespace(
            io=types.SimpleNamespace(Raster=FakeRaster),
            geometry=types.SimpleNamespace(Rdr2Geo=FakeTopo),
            core=types.SimpleNamespace(
                Ellipsoid=mock.Mock(return_value=object()),
                LUT2d=mock.Mock(return_value=object()),
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            progress = FakeProgress()
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps({"metadata": {}}), encoding="utf-8")
            output_h5 = root / "out.h5"
            radargrid_path = root / "radargrid.json"
            orbit_path = root / "orbit.json"
            acquisition_path = root / "acquisition.json"
            radargrid_path.write_text(json.dumps({"numberOfColumns": 4, "numberOfRows": 3}), encoding="utf-8")
            orbit_path.write_text(json.dumps({}), encoding="utf-8")
            acquisition_path.write_text(json.dumps({}), encoding="utf-8")

            def fake_resolve(_manifest_path, _manifest, key):
                return str(
                    {
                        "radargrid": radargrid_path,
                        "orbit": orbit_path,
                        "acquisition": acquisition_path,
                    }[key]
                )

            def fake_read_band_array(_band, _xoff=0, _yoff=0, xsize=None, ysize=None, **_kwargs):
                return np.ones((int(ysize), int(xsize)), dtype=np.float64)

            with (
                mock.patch("strip_insar2.resolve_manifest_metadata_path", side_effect=fake_resolve),
                mock.patch("strip_insar2.choose_orbit_interp", return_value="Legendre"),
                mock.patch("strip_insar2.construct_orbit", return_value=object()),
                mock.patch("strip_insar2.construct_radar_grid", return_value=FakeGrid()),
                mock.patch("strip_insar2.gdal.Open", return_value=FakeGdalDataset()),
                mock.patch("strip_insar2._read_band_array", side_effect=fake_read_band_array),
                mock.patch("strip_insar2.h5py.File", side_effect=FakeH5File),
                mock.patch.dict(
                    sys.modules,
                    {
                        "isce3": fake_isce3,
                        "isce3.io": fake_isce3.io,
                        "isce3.core": fake_isce3.core,
                        "isce3.geometry": fake_isce3.geometry,
                    },
                ),
            ):
                strip_insar2.append_topo_coordinates_hdf(
                    manifest_path=str(manifest_path),
                    dem_path="/tmp/dem.tif",
                    output_h5=str(output_h5),
                    block_rows=2,
                    use_gpu=False,
                    progress_reporter=progress,
                )

        self.assertIsNotNone(FakeH5File.last)
        for name in ("longitude", "latitude", "height"):
            self.assertEqual(FakeH5File.last.datasets[name].dtype, "f8")
            self.assertEqual(FakeH5File.last.datasets[name].data.dtype, np.float64)
        self.assertEqual(
            progress.events,
            [
                ("cpu", 1, 2, "write_hdf_coordinates"),
                ("cpu", 2, 2, "write_hdf_coordinates"),
            ],
        )

    def test_append_topo_coordinates_hdf_silences_isce3_journal(self):
        class FakeRaster:
            def __init__(self, *args, **kwargs):
                pass

            def close_dataset(self):
                pass

        class FakeTopo:
            def __init__(self, *args, **kwargs):
                pass

            def topo(self, *args, **kwargs):
                return None

        class FakeGrid:
            width = 4
            length = 3

        class FakeBand:
            pass

        class FakeGdalDataset:
            def GetRasterBand(self, _index):
                return FakeBand()

        class FakeWritableDataset:
            def __init__(self, dtype):
                self.dtype = dtype
                self.data = None

            def __setitem__(self, key, value):
                self.data = np.asarray(value)

        class FakeH5File:
            def __init__(self, *args, **kwargs):
                self.datasets = {}
                self.attrs = {}

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def __contains__(self, key):
                return key in self.datasets

            def __delitem__(self, key):
                self.datasets.pop(key, None)

            def create_dataset(self, name, **kwargs):
                ds = FakeWritableDataset(kwargs.get("dtype"))
                self.datasets[name] = ds
                return ds

        fake_isce3 = types.SimpleNamespace(
            io=types.SimpleNamespace(Raster=FakeRaster),
            geometry=types.SimpleNamespace(Rdr2Geo=FakeTopo),
            core=types.SimpleNamespace(
                Ellipsoid=mock.Mock(return_value=object()),
                LUT2d=mock.Mock(return_value=object()),
            ),
        )

        called_paths = []

        @contextmanager
        def fake_silence(path):
            called_paths.append(Path(path))
            yield str(path)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps({"metadata": {}}), encoding="utf-8")
            output_h5 = root / "out.h5"
            radargrid_path = root / "radargrid.json"
            orbit_path = root / "orbit.json"
            acquisition_path = root / "acquisition.json"
            radargrid_path.write_text(json.dumps({"numberOfColumns": 4, "numberOfRows": 3}), encoding="utf-8")
            orbit_path.write_text(json.dumps({}), encoding="utf-8")
            acquisition_path.write_text(json.dumps({}), encoding="utf-8")

            def fake_resolve(_manifest_path, _manifest, key):
                return str(
                    {
                        "radargrid": radargrid_path,
                        "orbit": orbit_path,
                        "acquisition": acquisition_path,
                    }[key]
                )

            def fake_read_band_array(_band, _xoff=0, _yoff=0, xsize=None, ysize=None, **_kwargs):
                return np.ones((int(ysize), int(xsize)), dtype=np.float64)

            with (
                mock.patch("strip_insar2.resolve_manifest_metadata_path", side_effect=fake_resolve),
                mock.patch("strip_insar2.choose_orbit_interp", return_value="Legendre"),
                mock.patch("strip_insar2.construct_orbit", return_value=object()),
                mock.patch("strip_insar2.construct_radar_grid", return_value=FakeGrid()),
                mock.patch("strip_insar2.gdal.Open", return_value=FakeGdalDataset()),
                mock.patch("strip_insar2._read_band_array", side_effect=fake_read_band_array),
                mock.patch("strip_insar2.h5py.File", side_effect=FakeH5File),
                mock.patch("strip_insar2._silence_isce3_journal", side_effect=fake_silence),
                mock.patch.dict(
                    sys.modules,
                    {
                        "isce3": fake_isce3,
                        "isce3.io": fake_isce3.io,
                        "isce3.core": fake_isce3.core,
                        "isce3.geometry": fake_isce3.geometry,
                    },
                ),
            ):
                strip_insar2.append_topo_coordinates_hdf(
                    manifest_path=str(manifest_path),
                    dem_path="/tmp/dem.tif",
                    output_h5=str(output_h5),
                    block_rows=2,
                    use_gpu=False,
                )

        self.assertEqual(
            called_paths,
            [root / "work" / "p5_hdf" / "isce3_journal.log"],
        )


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


class StripInsar2ProgressTests(unittest.TestCase):
    def test_run_with_running_progress_reports_elapsed(self):
        class FakeProgress:
            def __init__(self):
                self.events = []

            def running(self, *, backend, detail, elapsed, force=False):
                self.events.append((backend, detail, force, elapsed))

        progress = FakeProgress()
        result = strip_insar2._run_with_running_progress(
            progress_reporter=progress,
            backend="cpu",
            detail="demo",
            func=lambda: time.sleep(0.03) or "ok",
            interval_s=0.01,
        )

        self.assertEqual(result, "ok")
        self.assertTrue(progress.events)
        self.assertEqual(progress.events[0][0], "cpu")
        self.assertEqual(progress.events[0][1], "demo")
        self.assertTrue(progress.events[0][2])


class StripInsar2CrossmulTests(unittest.TestCase):
    def test_run_crossmul_cpu_reports_block_progress(self):
        class FakeProgress:
            def __init__(self):
                self.events = []

            def block(self, *, backend, current, total, detail):
                self.events.append((backend, current, total, detail))

        master = np.ones((5, 3), dtype=np.complex64)
        slave = np.full((5, 3), 2 + 0j, dtype=np.complex64)
        progress = FakeProgress()

        with (
            mock.patch("strip_insar2._open_slc_as_complex", side_effect=[master, slave]),
            mock.patch("strip_insar2._estimate_coherence", return_value=np.ones((5, 3), dtype=np.float32)),
        ):
            ifg, coh = strip_insar2._run_crossmul_cpu(
                master_slc_path="master.tif",
                slave_slc_path="slave.tif",
                flatten_raster=None,
                flatten_mask_raster=None,
                range_pixel_spacing=None,
                wavelength=None,
                flatten_starting_range_shift_m=None,
                block_rows=2,
                progress_reporter=progress,
            )

        self.assertEqual(ifg.shape, (5, 3))
        self.assertEqual(coh.shape, (5, 3))
        self.assertEqual(
            progress.events,
            [
                ("cpu", 1, 3, "crossmul"),
                ("cpu", 2, 3, "crossmul"),
                ("cpu", 3, 3, "crossmul"),
            ],
        )

    def test_run_crossmul_prefers_gpu_when_available(self):
        fake_ifg = np.array([[1 + 2j]], dtype=np.complex64)
        fake_coh = np.array([[0.75]], dtype=np.float32)

        with (
            mock.patch(
                "strip_insar2._crossmul_isce3_gpu",
                return_value=(fake_ifg, fake_coh),
            ) as mock_gpu,
            mock.patch("strip_insar2._run_crossmul_cpu") as mock_cpu,
        ):
            ifg, coh, backend, fallback_reason = strip_insar2._run_crossmul(
                master_slc_path="master.tif",
                slave_slc_path="slave.tif",
                use_gpu=True,
                gpu_id=2,
                output_dir=Path("/tmp/out"),
                block_rows=128,
                flatten_raster="range.off.tif",
                flatten_mask_raster="range.mask.tif",
                range_pixel_spacing=2.5,
                wavelength=0.23,
                flatten_starting_range_shift_m=10.0,
            )

        np.testing.assert_allclose(ifg, fake_ifg)
        np.testing.assert_allclose(coh, fake_coh)
        self.assertEqual(backend, "gpu")
        self.assertIsNone(fallback_reason)
        mock_gpu.assert_called_once_with(
            master_slc_path="master.tif",
            slave_slc_path="slave.tif",
            output_dir=Path("/tmp/out"),
            gpu_id=2,
            block_rows=128,
            flatten_raster="range.off.tif",
            range_pixel_spacing=2.5,
            wavelength=0.23,
            flatten_starting_range_shift_m=10.0,
            progress_reporter=None,
        )
        mock_cpu.assert_not_called()

    def test_run_crossmul_falls_back_to_cpu_when_gpu_fails(self):
        fake_ifg = np.array([[3 + 0j]], dtype=np.complex64)
        fake_coh = np.array([[0.25]], dtype=np.float32)

        with (
            mock.patch(
                "strip_insar2._crossmul_isce3_gpu",
                side_effect=RuntimeError("gpu crossmul failed"),
            ) as mock_gpu,
            mock.patch(
                "strip_insar2._run_crossmul_cpu",
                return_value=(fake_ifg, fake_coh),
            ) as mock_cpu,
        ):
            ifg, coh, backend, fallback_reason = strip_insar2._run_crossmul(
                master_slc_path="master.tif",
                slave_slc_path="slave.tif",
                use_gpu=True,
                gpu_id=0,
                output_dir=Path("/tmp/out"),
                block_rows=64,
                flatten_raster="range.off.tif",
                flatten_mask_raster="range.mask.tif",
                range_pixel_spacing=2.0,
                wavelength=0.24,
            )

        np.testing.assert_allclose(ifg, fake_ifg)
        np.testing.assert_allclose(coh, fake_coh)
        self.assertEqual(backend, "cpu")
        self.assertIn("gpu crossmul failed", fallback_reason)
        mock_gpu.assert_called_once()
        mock_cpu.assert_called_once()

    def test_run_crossmul_uses_cpu_directly_when_gpu_not_requested(self):
        fake_ifg = np.array([[5 + 0j]], dtype=np.complex64)
        fake_coh = np.array([[0.9]], dtype=np.float32)

        with (
            mock.patch("strip_insar2._crossmul_isce3_gpu") as mock_gpu,
            mock.patch(
                "strip_insar2._run_crossmul_cpu",
                return_value=(fake_ifg, fake_coh),
            ) as mock_cpu,
        ):
            ifg, coh, backend, fallback_reason = strip_insar2._run_crossmul(
                master_slc_path="master.tif",
                slave_slc_path="slave.tif",
                use_gpu=False,
                gpu_id=0,
                output_dir=Path("/tmp/out"),
                block_rows=64,
            )

        np.testing.assert_allclose(ifg, fake_ifg)
        np.testing.assert_allclose(coh, fake_coh)
        self.assertEqual(backend, "cpu")
        self.assertIsNone(fallback_reason)
        mock_gpu.assert_not_called()
        mock_cpu.assert_called_once()

    def test_run_goldstein_filter_prefers_gpu_when_available(self):
        fake_filtered = np.array([[1 + 0j]], dtype=np.complex64)

        with (
            mock.patch(
                "strip_insar2._goldstein_filter_gpu",
                return_value=fake_filtered,
            ) as gpu_filter,
            mock.patch("strip_insar2.goldstein_filter") as cpu_filter,
        ):
            filtered, backend, fallback_reason = strip_insar2._run_goldstein_filter(
                interferogram=np.array([[2 + 3j]], dtype=np.complex64),
                use_gpu=True,
                gpu_id=0,
                progress_reporter=None,
            )

        np.testing.assert_allclose(filtered, fake_filtered)
        self.assertEqual(backend, "gpu")
        self.assertIsNone(fallback_reason)
        gpu_filter.assert_called_once()
        cpu_filter.assert_not_called()

    def test_run_goldstein_filter_falls_back_to_cpu_when_gpu_fails(self):
        fake_filtered = np.array([[4 + 0j]], dtype=np.complex64)

        with (
            mock.patch(
                "strip_insar2._goldstein_filter_gpu",
                side_effect=RuntimeError("gpu filter failed"),
            ) as gpu_filter,
            mock.patch(
                "strip_insar2.goldstein_filter",
                return_value=fake_filtered,
            ) as cpu_filter,
        ):
            filtered, backend, fallback_reason = strip_insar2._run_goldstein_filter(
                interferogram=np.array([[2 + 3j]], dtype=np.complex64),
                use_gpu=True,
                gpu_id=0,
                progress_reporter=None,
            )

        np.testing.assert_allclose(filtered, fake_filtered)
        self.assertEqual(backend, "cpu")
        self.assertIn("gpu filter failed", fallback_reason)
        gpu_filter.assert_called_once()
        cpu_filter.assert_called_once()

    def test_run_crossmul_and_filter_prefers_combined_gpu_pipeline(self):
        fake_ifg = np.array([[1 + 2j]], dtype=np.complex64)
        fake_coh = np.array([[0.8]], dtype=np.float32)
        fake_filtered = np.array([[3 + 4j]], dtype=np.complex64)

        with (
            mock.patch(
                "strip_insar2._run_crossmul_filter_gpu_pipeline",
                return_value=(fake_ifg, fake_coh, fake_filtered, "gpu", None),
            ) as combined_gpu,
            mock.patch("strip_insar2._run_crossmul") as split_crossmul,
            mock.patch("strip_insar2._run_goldstein_filter") as split_filter,
        ):
            ifg, coh, filtered, backends, fallback_reason = strip_insar2._run_crossmul_and_filter(
                master_slc_path="master.tif",
                slave_slc_path="slave.tif",
                use_gpu=True,
                gpu_id=0,
                output_dir=Path("/tmp/out"),
                block_rows=64,
                flatten_raster="range.off.tif",
                flatten_mask_raster="range.mask.tif",
                range_pixel_spacing=2.0,
                wavelength=0.24,
                flatten_starting_range_shift_m=1.0,
                progress_reporter=None,
            )

        np.testing.assert_allclose(ifg, fake_ifg)
        np.testing.assert_allclose(coh, fake_coh)
        np.testing.assert_allclose(filtered, fake_filtered)
        self.assertEqual(backends, {"crossmul": "gpu", "goldstein_filter": "gpu"})
        self.assertIsNone(fallback_reason)
        combined_gpu.assert_called_once()
        split_crossmul.assert_not_called()
        split_filter.assert_not_called()

    def test_run_crossmul_and_filter_keeps_gpu_crossmul_when_helper_filter_falls_back_to_cpu(self):
        fake_ifg = np.array([[1 + 2j]], dtype=np.complex64)
        fake_coh = np.array([[0.8]], dtype=np.float32)
        fake_filtered = np.array([[3 + 4j]], dtype=np.complex64)

        with (
            mock.patch(
                "strip_insar2._run_crossmul_filter_gpu_pipeline",
                return_value=(
                    fake_ifg,
                    fake_coh,
                    fake_filtered,
                    "cpu",
                    "No module named 'cupy'",
                ),
            ) as combined_gpu,
            mock.patch("strip_insar2._run_crossmul") as split_crossmul,
            mock.patch("strip_insar2._run_goldstein_filter") as split_filter,
        ):
            ifg, coh, filtered, backends, fallback_reason = strip_insar2._run_crossmul_and_filter(
                master_slc_path="master.tif",
                slave_slc_path="slave.tif",
                use_gpu=True,
                gpu_id=0,
                output_dir=Path("/tmp/out"),
                block_rows=64,
                progress_reporter=None,
            )

        np.testing.assert_allclose(ifg, fake_ifg)
        np.testing.assert_allclose(coh, fake_coh)
        np.testing.assert_allclose(filtered, fake_filtered)
        self.assertEqual(backends, {"crossmul": "gpu", "goldstein_filter": "cpu"})
        self.assertIn("combined_gpu_pipeline goldstein_filter->cpu", fallback_reason)
        self.assertIn("No module named 'cupy'", fallback_reason)
        combined_gpu.assert_called_once()
        split_crossmul.assert_not_called()
        split_filter.assert_not_called()

    def test_run_crossmul_and_filter_falls_back_to_split_pipeline_when_combined_fails(self):
        fake_ifg = np.array([[5 + 0j]], dtype=np.complex64)
        fake_coh = np.array([[0.6]], dtype=np.float32)
        fake_filtered = np.array([[7 + 0j]], dtype=np.complex64)

        with (
            mock.patch(
                "strip_insar2._run_crossmul_filter_gpu_pipeline",
                side_effect=RuntimeError("combined failed"),
            ) as combined_gpu,
            mock.patch(
                "strip_insar2._run_crossmul",
                return_value=(fake_ifg, fake_coh, "gpu", None),
            ) as split_crossmul,
            mock.patch(
                "strip_insar2._run_goldstein_filter",
                return_value=(fake_filtered, "gpu", None),
            ) as split_filter,
        ):
            ifg, coh, filtered, backends, fallback_reason = strip_insar2._run_crossmul_and_filter(
                master_slc_path="master.tif",
                slave_slc_path="slave.tif",
                use_gpu=True,
                gpu_id=0,
                output_dir=Path("/tmp/out"),
                block_rows=64,
                progress_reporter=None,
            )

        np.testing.assert_allclose(ifg, fake_ifg)
        np.testing.assert_allclose(coh, fake_coh)
        np.testing.assert_allclose(filtered, fake_filtered)
        self.assertEqual(backends, {"crossmul": "gpu", "goldstein_filter": "gpu"})
        self.assertIn("combined_gpu_pipeline: combined failed", fallback_reason)
        combined_gpu.assert_called_once()
        split_crossmul.assert_called_once()
        split_filter.assert_called_once()

    def test_combined_gpu_helper_releases_output_rasters_before_reopen(self):
        fake_ifg = np.array([[1 + 0j]], dtype=np.complex64)
        fake_coh = np.array([[0.5]], dtype=np.float32)
        fake_filtered = np.array([[2 + 0j]], dtype=np.complex64)

        class FakeProcess:
            def __init__(self, *args, **kwargs):
                self.returncode = 0

            def communicate(self, timeout=None):
                return ("", "")

        class FakeDataset:
            def __init__(self, array):
                self.array = array

            def GetRasterBand(self, index):
                return self

            def ReadAsArray(self):
                return self.array

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "pair"

            def fake_open(path, mode):
                path = str(path)
                if path.endswith("cuda_interferogram.int"):
                    return FakeDataset(fake_ifg)
                if path.endswith("cuda_coherence.bin"):
                    return FakeDataset(fake_coh)
                if path.endswith("cuda_filtered_interferogram.int"):
                    return FakeDataset(fake_filtered)
                raise AssertionError(path)

            with (
                mock.patch("strip_insar2.subprocess.Popen", side_effect=FakeProcess),
                mock.patch("strip_insar2.py_compile.compile"),
                mock.patch("strip_insar2.gdal.Open", side_effect=fake_open),
                mock.patch(
                    "strip_insar2._read_band_array",
                    side_effect=lambda band, dtype=None: band.ReadAsArray(),
                ),
            ):
                strip_insar2._crossmul_filter_isce3_gpu_subprocess(
                    master_slc_path="master.tif",
                    slave_slc_path="slave.tif",
                    output_dir=output_dir,
                    gpu_id=0,
                    block_rows=128,
                    progress_reporter=None,
                )

            helper_text = (
                output_dir / "work" / "p2_crossmul" / "cuda_crossmul_filter_helper.py"
            ).read_text(encoding="utf-8")

        self.assertIn("ifg_raster = None", helper_text)
        self.assertIn("coh_raster = None", helper_text)
        self.assertIn("master_raster = None", helper_text)
        self.assertIn("slave_raster = None", helper_text)
        self.assertIn("_wait_for_min_file_size(ifg_path", helper_text)
        self.assertIn("def _goldstein_filter_cpu(", helper_text)
        self.assertIn("except Exception as exc:", helper_text)
        self.assertIn("filtered = _goldstein_filter_cpu(interferogram)", helper_text)
        self.assertIn("D2SAR_FILTER_INFO=", helper_text)
        self.assertLess(
            helper_text.index("ifg_raster = None"),
            helper_text.index("ifg_ds = gdal.Open(str(ifg_path), gdal.GA_ReadOnly)"),
        )


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

    def test_run_geo2rdr_stage_uses_running_progress_wrapper(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pair_dir = root / "20231110_20231121"

            context = mock.Mock()
            context.pair_dir = pair_dir
            context.master_manifest_path = root / "master.json"
            context.slave_manifest_path = root / "slave.json"
            context.master_orbit_data = {}
            context.slave_orbit_data = {}
            context.master_acq_data = {}
            context.slave_acq_data = {}
            context.master_rg_data = {}
            context.slave_rg_data = {}
            context.resolved_dem = "/tmp/dem.tif"
            context.orbit_interp = "Legendre"

            with (
                mock.patch("strip_insar2._run_rdr2geo_topo", side_effect=["master.vrt", "slave.vrt"]),
                mock.patch(
                    "strip_insar2._run_with_running_progress",
                    side_effect=lambda **kwargs: kwargs["func"](),
                ) as wrapped,
            ):
                output_files, backend_used, fallback_reason = strip_insar2.run_geo2rdr_stage(
                    context,
                    gpu_mode="cpu",
                    gpu_id=0,
                    block_rows=64,
                )

            self.assertEqual(output_files["master_topo"], "master.vrt")
            self.assertEqual(output_files["slave_topo"], "slave.vrt")
            self.assertEqual(backend_used, "cpu")
            self.assertIsNone(fallback_reason)
            wrapped.assert_called_once()
            self.assertEqual(wrapped.call_args.kwargs["backend"], "cpu")
            self.assertEqual(wrapped.call_args.kwargs["detail"], "rdr2geo/topo")


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
    def test_run_coarse_resamp_uses_zero_fill_value_for_invalid_offsets(self):
        class FakeBand:
            def __init__(self, array):
                self.array = array

        class FakeDataset:
            def __init__(self, array):
                self.array = array

            def GetRasterBand(self, _index):
                return FakeBand(self.array)

        class FakeGrid:
            length = 2
            width = 2

        captured = {}
        called_paths = []
        rg_offsets = np.array(
            [
                [0.0, strip_insar2.NISAR_OFFSET_INVALID_VALUE],
                [0.0, 0.0],
            ],
            dtype=np.float64,
        )
        az_offsets = np.array(
            [
                [0.0, 0.0],
                [strip_insar2.GEO2RDR_OFFSET_NODATA, 0.0],
            ],
            dtype=np.float64,
        )

        def fake_resample_slc_blocks(**kwargs):
            captured["fill_value"] = kwargs.get("fill_value")
            output_slc = kwargs["output_resampled_slcs"][0]
            output_slc[0, 0] = np.complex64(1.0 + 2.0j)

        @contextmanager
        def fake_silence(path):
            called_paths.append(Path(path))
            yield str(path)

        fake_resample_module = types.SimpleNamespace(
            resample_slc_blocks=fake_resample_slc_blocks
        )
        fake_isce3_core = types.SimpleNamespace(LUT2d=mock.Mock(return_value=object()))
        fake_isce3 = types.SimpleNamespace(core=fake_isce3_core)
        datasets = {
            "/tmp/rg.off": FakeDataset(rg_offsets),
            "/tmp/az.off": FakeDataset(az_offsets),
        }

        with (
            mock.patch("strip_insar2._open_slc_as_complex", return_value=np.ones((2, 2), dtype=np.complex64)),
            mock.patch("strip_insar2.gdal.Open", side_effect=lambda path, _mode: datasets[str(path)]),
            mock.patch("strip_insar2._read_band_array", side_effect=lambda band, dtype=None: band.array.astype(dtype)),
            mock.patch("strip_insar2._write_complex_envi") as write_envi,
            mock.patch("strip_insar2._write_complex_gtiff") as write_gtiff,
            mock.patch("strip_insar2._silence_isce3_journal", side_effect=fake_silence),
            mock.patch.dict(
                sys.modules,
                {
                    "isce3": fake_isce3,
                    "isce3.core": fake_isce3_core,
                    "isce3.image": types.SimpleNamespace(v2=types.SimpleNamespace(resample_slc=fake_resample_module)),
                    "isce3.image.v2": types.SimpleNamespace(resample_slc=fake_resample_module),
                    "isce3.image.v2.resample_slc": fake_resample_module,
                },
            ),
        ):
            ok = strip_insar2.run_coarse_resamp_isce3_v2(
                slave_slc_path="/tmp/slave.slc",
                coarse_coreg_slave_path="/tmp/out.slc",
                radar_grid=FakeGrid(),
                doppler=None,
                ref_radar_grid=FakeGrid(),
                rg_offset_path="/tmp/rg.off",
                az_offset_path="/tmp/az.off",
                use_gpu=False,
                block_size_az=2,
                block_size_rg=2,
                coarse_coreg_slave_gtiff_path="/tmp/out.tif",
            )

        self.assertTrue(ok)
        self.assertEqual(captured["fill_value"], 0.0 + 0.0j)
        write_envi.assert_called_once()
        write_gtiff.assert_called_once()
        written = write_envi.call_args.args[1]
        self.assertTrue(np.isfinite(written.real).all())
        self.assertTrue(np.isfinite(written.imag).all())
        self.assertEqual(written.dtype, np.complex64)
        self.assertEqual(written[0, 0], np.complex64(1.0 + 2.0j))
        self.assertEqual(written[0, 1], np.complex64(0.0 + 0.0j))
        self.assertEqual(called_paths, [Path("/tmp/isce3_resample_journal.log")])
        self.assertEqual(written[1, 0], np.complex64(0.0 + 0.0j))

    def test_run_slave_geo2rdr_from_master_topo_silences_isce3_journal(self):
        class FakeRaster:
            length = 2
            width = 3

            def __init__(self, *args, **kwargs):
                pass

        class FakeGeo2Rdr:
            def __init__(self, *args, **kwargs):
                pass

            def geo2rdr(self, *args, **kwargs):
                return None

        class FakeGrid:
            length = 2
            width = 3

        fake_isce3 = types.SimpleNamespace(
            io=types.SimpleNamespace(Raster=FakeRaster),
            geometry=types.SimpleNamespace(Geo2Rdr=FakeGeo2Rdr),
            core=types.SimpleNamespace(
                Ellipsoid=mock.Mock(return_value=object()),
            ),
        )

        called_paths = []

        @contextmanager
        def fake_silence(path):
            called_paths.append(Path(path))
            yield str(path)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            outdir = root / "p1"
            with (
                mock.patch("strip_insar2.construct_orbit", return_value=object()),
                mock.patch("strip_insar2.choose_orbit_interp", return_value="Legendre"),
                mock.patch("strip_insar2.construct_radar_grid", return_value=FakeGrid()),
                mock.patch("strip_insar2._build_coregistration_doppler_lut", return_value=object()),
                mock.patch("strip_insar2._convert_geo2rdr_output_to_gtiff", side_effect=["range.tif", "az.tif"]),
                mock.patch("strip_insar2._silence_isce3_journal", side_effect=fake_silence),
                mock.patch.dict(
                    sys.modules,
                    {
                        "isce3": fake_isce3,
                        "isce3.io": fake_isce3.io,
                        "isce3.core": fake_isce3.core,
                        "isce3.geometry": fake_isce3.geometry,
                    },
                ),
            ):
                range_path, az_path = strip_insar2._run_slave_geo2rdr_from_master_topo(
                    master_topo_vrt_path="/tmp/topo.vrt",
                    slave_orbit_data={},
                    slave_acq_data={},
                    slave_rg_data={},
                    slave_dop_data=None,
                    output_dir=outdir,
                    use_gpu=False,
                    gpu_id=0,
                    block_rows=64,
                )

        self.assertEqual((range_path, az_path), ("range.tif", "az.tif"))
        self.assertEqual(called_paths, [outdir / "isce3_journal.log"])

    def test_run_nisar_registration_chain_uses_resolution_aware_cpu_dense_offsets(self):
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
            context.master_rg_data = {
                "groundRangeResolution": 7.5,
                "azimuthResolution": 5.2,
            }
            context.master_acq_data = {}
            context.master_orbit_data = {}
            context.slave_rg_data = {}
            context.slave_acq_data = {}
            context.slave_orbit_data = {}
            context.slave_dop_data = {}
            context.pair_dir = root

            ref_grid = mock.Mock(length=2, width=3)
            sec_grid = mock.Mock(length=2, width=3)
            zero_doppler = object()
            dense_plan = {
                "gross_search_range": (3, 3),
                "candidates": [
                    {"window_size": (256, 256), "search_range": (64, 64)},
                    {"window_size": (128, 128), "search_range": (64, 64)},
                ],
            }
            registration_outputs = {
                "coarse_coreg_slave": str(p1_stage_path / "coarse_coreg_slave.tif"),
                "fine_coreg_slave": str(p1_stage_path / "fine_coreg_slave.tif"),
                "registration_model": str(p1_stage_path / "registration_model.json"),
                "range_offsets": str(p1_stage_path / "range.off.tif"),
                "azimuth_offsets": str(p1_stage_path / "azimuth.off.tif"),
                "range_residual_offsets": str(p1_stage_path / "range_residual.off.tif"),
                "azimuth_residual_offsets": str(p1_stage_path / "azimuth_residual.off.tif"),
            }

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
                ) as prepare_offsets,
                mock.patch(
                    "strip_insar2.run_coarse_resamp_isce3_v2",
                    return_value=True,
                ) as coarse_resamp,
                mock.patch(
                    "strip_insar2._select_strip_dense_match_plan",
                    return_value=dense_plan,
                ) as select_plan,
                mock.patch(
                    "strip_insar2._run_strip_cpu_dense_offsets",
                    return_value=(
                        np.zeros((2, 3), dtype=np.float32),
                        np.zeros((2, 3), dtype=np.float32),
                        {"diagnostics": {"status": "ok", "engine": "cpu-template-search"}},
                    ),
                ) as cpu_dense,
                mock.patch(
                    "strip_insar2._write_strip_registration_outputs",
                    return_value=registration_outputs,
                ) as write_outputs,
            ):
                outputs = strip_insar2._run_nisar_registration_chain(
                    context=context,
                    use_gpu=False,
                    gpu_id=0,
                    p1_stage_path=p1_stage_path,
                )

            self.assertEqual(coarse_resamp.call_count, 1)
            self.assertIs(coarse_resamp.call_args.kwargs["doppler"], zero_doppler)
            prepare_offsets.assert_not_called()
            select_plan.assert_called_once_with(7.5)
            cpu_dense.assert_called_once()
            self.assertEqual(
                cpu_dense.call_args.kwargs["window_candidates"],
                dense_plan["candidates"],
            )
            self.assertEqual(cpu_dense.call_args.kwargs["gross_offset"], (0.0, 0.0))
            write_outputs.assert_called_once()
            self.assertIs(write_outputs.call_args.kwargs["doppler"], zero_doppler)
            self.assertEqual(
                outputs["fine_coreg_slave"],
                registration_outputs["fine_coreg_slave"],
            )
            self.assertEqual(outputs["fine_coreg_slave_tif"], registration_outputs["fine_coreg_slave"])

    def test_run_nisar_registration_chain_uses_first_resolution_candidate_for_gpu_dense_offsets(self):
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
            context.master_rg_data = {
                "groundRangeResolution": 3.0,
                "azimuthResolution": 2.8,
            }
            context.master_acq_data = {}
            context.master_orbit_data = {}
            context.slave_rg_data = {}
            context.slave_acq_data = {}
            context.slave_orbit_data = {}
            context.slave_dop_data = {}
            context.pair_dir = root

            ref_grid = mock.Mock(length=2, width=3)
            sec_grid = mock.Mock(length=2, width=3)
            dense_plan = {
                "gross_search_range": (3, 3),
                "candidates": [
                    {"window_size": (512, 512), "search_range": (128, 128)},
                    {"window_size": (256, 256), "search_range": (64, 64)},
                ],
            }
            registration_outputs = {
                "coarse_coreg_slave": str(p1_stage_path / "coarse_coreg_slave.tif"),
                "fine_coreg_slave": str(p1_stage_path / "fine_coreg_slave.tif"),
                "registration_model": str(p1_stage_path / "registration_model.json"),
                "range_offsets": str(p1_stage_path / "range.off.tif"),
                "azimuth_offsets": str(p1_stage_path / "azimuth.off.tif"),
                "range_residual_offsets": str(p1_stage_path / "range_residual.off.tif"),
                "azimuth_residual_offsets": str(p1_stage_path / "azimuth_residual.off.tif"),
            }

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
                    return_value=object(),
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
                ) as prepare_offsets,
                mock.patch(
                    "strip_insar2.run_coarse_resamp_isce3_v2",
                    return_value=True,
                ),
                mock.patch(
                    "strip_insar2._select_strip_dense_match_plan",
                    return_value=dense_plan,
                ),
                mock.patch(
                    "strip_insar2._strip_raster_shape",
                    return_value=(2, 3),
                ),
                mock.patch(
                    "strip_insar2._run_strip_pycuampcor_dense_offsets",
                    return_value=(
                        np.zeros((2, 3), dtype=np.float32),
                        np.zeros((2, 3), dtype=np.float32),
                        {"diagnostics": {"status": "ok", "engine": "pycuampcor"}},
                    ),
                ) as gpu_dense,
                mock.patch(
                    "strip_insar2._write_strip_registration_outputs",
                    return_value=registration_outputs,
                ),
            ):
                strip_insar2._run_nisar_registration_chain(
                    context=context,
                    use_gpu=True,
                    gpu_id=0,
                    p1_stage_path=p1_stage_path,
                )

            prepare_offsets.assert_not_called()
            gpu_dense.assert_called_once()
            self.assertEqual(gpu_dense.call_args.kwargs["window_size"], (512, 512))
            self.assertEqual(gpu_dense.call_args.kwargs["search_range"], (128, 128))


class StripInsar2DenseOffsetsTests(unittest.TestCase):
    def test_default_dense_offsets_cfg_matches_conservative_gpu_profile(self):
        cfg = strip_insar2._default_dense_offsets_cfg()

        self.assertEqual(cfg["window_range"], 64)
        self.assertEqual(cfg["window_azimuth"], 64)
        self.assertEqual(cfg["half_search_range"], 20)
        self.assertEqual(cfg["half_search_azimuth"], 20)
        self.assertEqual(cfg["skip_range"], 32)
        self.assertEqual(cfg["skip_azimuth"], 32)
        self.assertEqual(cfg["cross_correlation_domain"], "frequency")
        self.assertEqual(cfg["deramping_method"], "complex")
        self.assertEqual(cfg["deramping_axis"], "azimuth")
        self.assertEqual(cfg["correlation_statistics_zoom"], 21)
        self.assertEqual(cfg["correlation_surface_zoom"], 8)
        self.assertEqual(cfg["correlation_surface_oversampling_factor"], 64)
        self.assertEqual(cfg["correlation_surface_oversampling_method"], "fft")
        self.assertEqual(cfg["windows_batch_range"], 10)
        self.assertEqual(cfg["windows_batch_azimuth"], 1)
        self.assertEqual(cfg["cuda_streams"], 2)

    def test_run_nisar_dense_offsets_impl_materializes_ampcor_inputs_as_envi_bip(self):
        class FakeAmpcor:
            def __init__(self):
                self.numberWindowAcross = 2
                self.numberWindowDown = 3

            def setupParams(self):
                return None

            def checkPixelInImageRange(self):
                return None

            def runAmpcor(self):
                return None

        fake_isce3 = types.SimpleNamespace(
            matchtemplate=types.SimpleNamespace(PyCPUAmpcor=FakeAmpcor),
        )

        with (
            mock.patch("strip_insar2._copy_raster_to_envi_complex64", side_effect=lambda src, dst: str(dst)) as copy_envi,
            mock.patch("strip_insar2._raster_shape", return_value=(32, 48)),
            mock.patch("strip_insar2._configure_ampcor_from_cfg", return_value={}),
            mock.patch("strip_insar2._create_empty_dataset"),
            mock.patch.dict(
                sys.modules,
                {
                    "isce3": fake_isce3,
                    "isce3.matchtemplate": fake_isce3.matchtemplate,
                },
            ),
        ):
            with tempfile.TemporaryDirectory() as tmpdir:
                outdir = Path(tmpdir)
                outputs = strip_insar2._run_nisar_dense_offsets_impl(
                    reference_slc_path="/tmp/master.tif",
                    secondary_slc_path="/tmp/slave.tif",
                    output_dir=outdir,
                    use_gpu=False,
                    gpu_id=0,
                )

        self.assertEqual(outputs["reference_slc"], str(Path(tmpdir) / "reference.slc"))
        self.assertEqual(outputs["secondary_slc"], str(Path(tmpdir) / "secondary.slc"))
        self.assertEqual(copy_envi.call_count, 2)
        self.assertEqual(copy_envi.call_args_list[0].args[0], "/tmp/master.tif")
        self.assertEqual(copy_envi.call_args_list[1].args[0], "/tmp/slave.tif")

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

    def test_process_strip_insar2_prints_stage_start_and_end_messages(self):
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
                mock.patch("builtins.print") as print_mock,
            ):
                strip_insar2.process_strip_insar2(
                    str(master_manifest),
                    str(slave_manifest),
                    output_root=str(output_root),
                    gpu_mode="cpu",
                    dem_path=str(root / "dem.tif"),
                )

            printed_lines = [
                " ".join(str(arg) for arg in call.args)
                for call in print_mock.call_args_list
            ]

            for stage in ("p0", "p1", "p2", "p3", "p4", "p5", "p6"):
                self.assertTrue(
                    any(f"[{stage}] START" in line for line in printed_lines),
                    msg=f"missing START message for {stage}: {printed_lines}",
                )
                self.assertTrue(
                    any(f"[{stage}] DONE" in line for line in printed_lines),
                    msg=f"missing DONE message for {stage}: {printed_lines}",
                )

    def test_process_strip_insar2_defaults_to_icu_unwrap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            master_manifest = self._write_manifest(root, "master", "2023-11-10T04:39:48.881889")
            slave_manifest = self._write_manifest(root, "slave", "2023-11-21T04:39:48.881889")
            output_root = root / "results"

            with (
                mock.patch("strip_insar2.run_geo2rdr_stage", return_value=({"master_topo": "a", "slave_topo": "b"}, "cpu", None)),
                mock.patch("strip_insar2.run_resample_stage", return_value=({"fine_coreg_slave": "fine.tif"}, "cpu", None)),
                mock.patch("strip_insar2.run_crossmul_stage", return_value=({"interferogram": "ifg.npy"}, "cpu", None)),
                mock.patch("strip_insar2.run_unwrap_stage", return_value=({"unwrapped_phase": "unw.npy"}, "cpu", None)) as unwrap_stage,
                mock.patch("strip_insar2.run_los_stage", return_value=({"los_displacement": "los.npy"}, "cpu", None)),
                mock.patch("strip_insar2.write_primary_product", return_value=("/tmp/fake.h5", "cpu", None)),
                mock.patch("strip_insar2.export_insar_products", return_value={"interferogram_png": "ifg.png"}),
            ):
                strip_insar2.process_strip_insar2(
                    str(master_manifest),
                    str(slave_manifest),
                    output_root=str(output_root),
                    gpu_mode="cpu",
                    dem_path=str(root / "dem.tif"),
                )

            self.assertEqual(unwrap_stage.call_args.kwargs["unwrap_method"], "icu")

    def test_process_strip_insar2_uses_isce3_default_block_rows_by_stage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            master_manifest = self._write_manifest(root, "master", "2023-11-10T04:39:48.881889")
            slave_manifest = self._write_manifest(root, "slave", "2023-11-21T04:39:48.881889")
            output_root = root / "results"

            with (
                mock.patch("strip_insar2.run_geo2rdr_stage", return_value=({"master_topo": "a", "slave_topo": "b"}, "cpu", None)) as geo2rdr_stage,
                mock.patch("strip_insar2.run_resample_stage", return_value=({"fine_coreg_slave": "fine.tif"}, "cpu", None)),
                mock.patch("strip_insar2.run_crossmul_stage", return_value=({"interferogram": "ifg.npy"}, "cpu", None)) as crossmul_stage,
                mock.patch("strip_insar2.run_unwrap_stage", return_value=({"unwrapped_phase": "unw.npy"}, "cpu", None)) as unwrap_stage,
                mock.patch("strip_insar2.run_los_stage", return_value=({"los_displacement": "los.npy"}, "cpu", None)),
                mock.patch("strip_insar2.write_primary_product", return_value=("/tmp/fake.h5", "cpu", None)) as write_product,
                mock.patch("strip_insar2.export_insar_products", return_value={"interferogram_png": "ifg.png"}),
            ):
                strip_insar2.process_strip_insar2(
                    str(master_manifest),
                    str(slave_manifest),
                    output_root=str(output_root),
                    gpu_mode="cpu",
                    dem_path=str(root / "dem.tif"),
                )

            self.assertEqual(
                geo2rdr_stage.call_args.kwargs["block_rows"],
                strip_insar2.ISCE3_GEOMETRY_LINES_PER_BLOCK_DEFAULT,
            )
            self.assertEqual(
                crossmul_stage.call_args.kwargs["block_rows"],
                strip_insar2.ISCE3_CROSSMUL_LINES_PER_BLOCK_DEFAULT,
            )
            self.assertEqual(
                unwrap_stage.call_args.kwargs["block_rows"],
                strip_insar2.ISCE3_GEOMETRY_LINES_PER_BLOCK_DEFAULT,
            )
            self.assertEqual(
                write_product.call_args.kwargs["block_rows"],
                strip_insar2.ISCE3_GEOMETRY_LINES_PER_BLOCK_DEFAULT,
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
            ) as registration_chain, mock.patch(
                "strip_insar2._run_with_running_progress",
                side_effect=lambda **kwargs: kwargs["func"](),
            ) as wrapped:
                outputs, backend_used, fallback_reason = strip_insar2.run_resample_stage(
                    context,
                    gpu_mode="cpu",
                    gpu_id=0,
                )

            self.assertEqual(outputs, expected_outputs)
            self.assertEqual(backend_used, "cpu")
            self.assertIsNone(fallback_reason)
            registration_chain.assert_called_once()
            wrapped.assert_called_once()
            self.assertEqual(wrapped.call_args.kwargs["backend"], "cpu")
            self.assertEqual(wrapped.call_args.kwargs["detail"], "resample/registration")


class StripInsar2UnwrapTests(unittest.TestCase):
    def test_run_unwrap_stage_rejects_snaphu_as_primary_method(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pair_dir = root / "20231110_20231121"
            pair_dir.mkdir(parents=True, exist_ok=True)

            interferogram = np.ones((2, 2), dtype=np.complex64)
            coherence = np.ones((2, 2), dtype=np.float32)
            expected_unwrapped = np.full((2, 2), 4.0, dtype=np.float32)

            filtered_path = strip_insar2._save_stage_array(pair_dir, "p2", "filtered_interferogram", interferogram)
            coherence_path = strip_insar2._save_stage_array(pair_dir, "p2", "coherence", coherence)
            strip_insar2.write_stage_record(
                pair_dir,
                "p2",
                {
                    "stage": "p2",
                    "output_files": {
                        "filtered_interferogram": filtered_path,
                        "coherence": coherence_path,
                    },
                    "success": True,
                },
            )

            context = mock.Mock()
            context.pair_dir = pair_dir
            context.master_manifest_path = root / "master.json"
            context.slave_manifest_path = root / "slave.json"

            with self.assertRaisesRegex(ValueError, "unsupported unwrap method: snaphu"):
                strip_insar2.run_unwrap_stage(
                    context,
                    unwrap_method="snaphu",
                    block_rows=256,
                )

    def test_run_unwrap_stage_falls_back_to_snaphu_when_icu_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pair_dir = root / "20231110_20231121"
            pair_dir.mkdir(parents=True, exist_ok=True)

            interferogram = np.ones((2, 2), dtype=np.complex64)
            coherence = np.ones((2, 2), dtype=np.float32)
            expected_unwrapped = np.full((2, 2), 3.0, dtype=np.float32)

            filtered_path = strip_insar2._save_stage_array(pair_dir, "p2", "filtered_interferogram", interferogram)
            coherence_path = strip_insar2._save_stage_array(pair_dir, "p2", "coherence", coherence)
            strip_insar2.write_stage_record(
                pair_dir,
                "p2",
                {
                    "stage": "p2",
                    "output_files": {
                        "filtered_interferogram": filtered_path,
                        "coherence": coherence_path,
                    },
                    "success": True,
                },
            )

            context = mock.Mock()
            context.pair_dir = pair_dir
            context.master_manifest_path = root / "master.json"
            context.slave_manifest_path = root / "slave.json"

            with (
                mock.patch(
                    "strip_insar2._unwrap_with_icu",
                    side_effect=RuntimeError("failed to unwrap tile at max correlation threshold"),
                ) as icu_unwrap,
                mock.patch(
                    "strip_insar2._unwrap_with_snaphu_profiles",
                    return_value=(expected_unwrapped, "SNAPHU profile=default"),
                ) as snaphu_unwrap,
            ):
                outputs, backend_used, fallback_reason = strip_insar2.run_unwrap_stage(
                    context,
                    unwrap_method="icu",
                    block_rows=256,
                )

            self.assertEqual(backend_used, "cpu")
            self.assertIn("ICU failed", fallback_reason)
            self.assertIn("max correlation threshold", fallback_reason)
            self.assertIn("SNAPHU profile=default", fallback_reason)
            icu_unwrap.assert_called_once()
            snaphu_unwrap.assert_called_once()
            loaded = np.load(outputs["unwrapped_phase"])
            np.testing.assert_allclose(loaded, expected_unwrapped)

    def test_run_unwrap_stage_uses_relaxed_snaphu_profile_after_icu_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pair_dir = root / "20231110_20231121"
            pair_dir.mkdir(parents=True, exist_ok=True)

            interferogram = np.ones((2, 2), dtype=np.complex64)
            coherence = np.ones((2, 2), dtype=np.float32)
            expected_unwrapped = np.full((2, 2), 5.0, dtype=np.float32)

            filtered_path = strip_insar2._save_stage_array(pair_dir, "p2", "filtered_interferogram", interferogram)
            coherence_path = strip_insar2._save_stage_array(pair_dir, "p2", "coherence", coherence)
            strip_insar2.write_stage_record(
                pair_dir,
                "p2",
                {
                    "stage": "p2",
                    "output_files": {
                        "filtered_interferogram": filtered_path,
                        "coherence": coherence_path,
                    },
                    "success": True,
                },
            )

            context = mock.Mock()
            context.pair_dir = pair_dir
            context.master_manifest_path = root / "master.json"
            context.slave_manifest_path = root / "slave.json"

            with (
                mock.patch(
                    "strip_insar2._unwrap_with_icu",
                    side_effect=RuntimeError("icu failed"),
                ) as icu_unwrap,
                mock.patch(
                    "strip_insar2._unwrap_with_snaphu",
                    side_effect=[
                        RuntimeError("default profile failed"),
                        expected_unwrapped,
                    ],
                ) as snaphu_unwrap,
            ):
                outputs, backend_used, fallback_reason = strip_insar2.run_unwrap_stage(
                    context,
                    unwrap_method="icu",
                    block_rows=256,
                )

            self.assertEqual(backend_used, "cpu")
            self.assertIn("ICU failed", fallback_reason)
            self.assertIn("SNAPHU profile=relaxed", fallback_reason)
            icu_unwrap.assert_called_once()
            self.assertEqual(snaphu_unwrap.call_count, 2)
            loaded = np.load(outputs["unwrapped_phase"])
            np.testing.assert_allclose(loaded, expected_unwrapped)


if __name__ == "__main__":
    unittest.main()
