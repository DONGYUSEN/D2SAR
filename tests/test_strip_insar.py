import json
import numpy as np
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import strip_insar

from strip_insar import (
    detect_sensor_from_manifest,
    build_output_paths,
    select_processing_backend,
    choose_gpu_topo_block_rows,
    process_strip_insar,
    compute_los_displacement,
    write_insar_hdf,
    PhaseUnwrapper,
    ICUUnwrapper,
    SNAPHUUnwrapper,
    _create_unwrapper,
    _run_p1_stage_from_cache,
    _run_pycuampcor,
    _run_slave_geo2rdr_from_master_topo,
    _geo2rdr_pixel_offsets,
    _convert_geo2rdr_abs_to_relative_offsets,
    _write_radar_wrapped_phase_png,
    _estimate_coherence_from_complex_slcs,
    _run_geo2rdr,
)
from insar_filtering import goldstein_filter
from insar_registration import (
    write_registration_outputs,
    _estimate_offset_mean_from_raster,
    _load_offset_dataset_for_resample,
    _sanitize_geo2rdr_offset_array,
    _plan_matching_grid,
    _select_cpu_dense_match_plan,
    _write_varying_gross_offset_file,
    _prepare_sparse_offsets_for_dense_model,
    _rubbersheet_dense_offsets,
    run_pycuampcor_dense_offsets,
    run_cpu_dense_offsets,
)
from common_processing import resolve_manifest_metadata_path


class StripInSARTests(unittest.TestCase):
    def _write_float_tiff(self, path: Path, data: np.ndarray) -> None:
        from osgeo import gdal

        arr = np.asarray(data, dtype=np.float32)
        ds = gdal.GetDriverByName("GTiff").Create(
            str(path),
            int(arr.shape[1]),
            int(arr.shape[0]),
            1,
            gdal.GDT_Float32,
        )
        ds.GetRasterBand(1).WriteArray(arr)
        ds = None

    def _write_manifest(self, tmpdir: str, payload: dict) -> Path:
        manifest_path = Path(tmpdir) / "manifest.json"
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        return manifest_path

    def test_detect_sensor_accepts_tianyi(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = self._write_manifest(tmpdir, {"sensor": "tianyi"})
            self.assertEqual(detect_sensor_from_manifest(p), "tianyi")

    def test_detect_sensor_accepts_lutan(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = self._write_manifest(tmpdir, {"sensor": "lutan"})
            self.assertEqual(detect_sensor_from_manifest(p), "lutan")

    def test_estimate_offset_mean_from_raster_ignores_geo2rdr_invalid_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "offset.tif"
            self._write_float_tiff(
                path,
                np.array(
                    [
                        [10.0, -1000000.0, -999999.0],
                        [14.0, np.nan, -100001.0],
                    ],
                    dtype=np.float32,
                ),
            )
            self.assertAlmostEqual(_estimate_offset_mean_from_raster(path), 4.0)

    def test_sanitize_geo2rdr_offset_array_replaces_invalid_values_with_zero(self):
        arr = np.array([[1.0, -999999.0], [np.nan, -100001.0]], dtype=np.float32)
        np.testing.assert_allclose(
            _sanitize_geo2rdr_offset_array(arr),
            np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32),
        )

    def test_run_geo2rdr_writes_topo_lon_lat_hgt_as_float64(self):
        class FakeRaster:
            def close_dataset(self):
                pass

        class FakeTopo:
            def __init__(self, *args, **kwargs):
                pass

            def topo(self, *args, **kwargs):
                return None

        class FakeGrid:
            def __init__(self, width=16, length=8):
                self.width = width
                self.length = length

        class FakeSRS:
            def ImportFromEPSG(self, _epsg):
                return None

            def SetAxisMappingStrategy(self, _strategy):
                return None

            def ExportToWkt(self):
                return "EPSG:4326"

        class FakeDataset:
            def SetProjection(self, _wkt):
                return None

            def FlushCache(self):
                return None

        created = []

        def fake_make_raster(path, dtype, width, length):
            created.append((Path(path).name, dtype, width, length))
            return FakeRaster()

        fake_isce3 = types.SimpleNamespace(
            io=types.SimpleNamespace(Raster=mock.Mock(return_value=FakeRaster())),
            geometry=types.SimpleNamespace(Rdr2Geo=FakeTopo),
            core=types.SimpleNamespace(
                Ellipsoid=mock.Mock(return_value=object()),
                LUT2d=mock.Mock(return_value=object()),
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "out"
            with (
                mock.patch("strip_insar._load_master_metadata", side_effect=[
                    ({}, {}, {}, {}, {}),
                    ({}, {}, {}, {}, {}),
                ]),
                mock.patch("strip_insar.construct_orbit", return_value=object()),
                mock.patch("strip_insar.construct_radar_grid", side_effect=[
                    FakeGrid(),
                    FakeGrid(),
                ]),
                mock.patch("strip_insar._construct_doppler_if_possible", return_value=None),
                mock.patch("strip_insar._make_raster", side_effect=fake_make_raster),
                mock.patch("strip_insar._merge_tiffs"),
                mock.patch("strip_insar.gdal.BuildVRT", return_value=FakeDataset()),
                mock.patch("strip_insar.gdal.Open", return_value=FakeDataset()),
                mock.patch("strip_insar.osr.SpatialReference", return_value=FakeSRS()),
                mock.patch.dict(
                    sys.modules,
                    {
                        "isce3": fake_isce3,
                        "isce3.io": fake_isce3.io,
                        "isce3.core": fake_isce3.core,
                        "isce3.geometry": fake_isce3.geometry,
                    },
                ),
                mock.patch.object(strip_insar, "isce3", fake_isce3, create=True),
            ):
                _run_geo2rdr(
                    master_manifest_path="/tmp/master.json",
                    slave_manifest_path="/tmp/slave.json",
                    dem_path="/tmp/dem.tif",
                    orbit_interp="Legendre",
                    use_gpu=False,
                    gpu_id=0,
                    output_dir=out_dir,
                )

        topo_rasters = [
            (name, dtype)
            for name, dtype, _width, _length in created
            if name in {"lon.tif", "lat.tif", "hgt.tif"}
        ]
        self.assertEqual(len(topo_rasters), 6)
        self.assertTrue(
            all(dtype == strip_insar.gdal.GDT_Float64 for _name, dtype in topo_rasters),
            topo_rasters,
        )

    def test_detect_sensor_rejects_unknown(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = self._write_manifest(tmpdir, {"sensor": "unknown"})
            with self.assertRaises(ValueError):
                detect_sensor_from_manifest(p)

    def test_build_output_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = build_output_paths(tmpdir)
            self.assertEqual(
                out["interferogram_h5"], str(Path(tmpdir) / "interferogram_fullres.h5")
            )
            self.assertEqual(
                out["interferogram_tif"],
                str(Path(tmpdir) / "interferogram_utm_geocoded.tif"),
            )
            self.assertEqual(
                out["coherence_tif"], str(Path(tmpdir) / "coherence_utm_geocoded.tif")
            )
            self.assertEqual(
                out["unwrapped_phase_tif"],
                str(Path(tmpdir) / "unwrapped_phase_utm_geocoded.tif"),
            )
            self.assertEqual(
                out["los_displacement_tif"],
                str(Path(tmpdir) / "los_displacement_utm_geocoded.tif"),
            )
            self.assertEqual(
                out["interferogram_png"],
                str(Path(tmpdir) / "interferogram_utm_geocoded.png"),
            )

    def test_select_processing_backend_cpu_forced(self):
        backend, reason = select_processing_backend("cpu", 0)
        self.assertEqual(backend, "cpu")
        self.assertIn("forced", reason.lower())

    def test_select_processing_backend_auto_prefers_gpu(self):
        with mock.patch("strip_insar._default_gpu_check", return_value=True):
            backend, reason = select_processing_backend("auto", 0)
            self.assertEqual(backend, "gpu")
            self.assertIn("available", reason.lower())

    def test_select_processing_backend_auto_falls_back(self):
        with mock.patch("strip_insar._default_gpu_check", return_value=False):
            backend, reason = select_processing_backend("auto", 0)
            self.assertEqual(backend, "cpu")

    def test_choose_gpu_block_rows_unknown_memory(self):
        rows, reason = choose_gpu_topo_block_rows(20000, 256, None)
        self.assertEqual(rows, 256)
        self.assertIn("default", reason.lower())

    def test_choose_gpu_block_rows_adaptive(self):
        rows, reason = choose_gpu_topo_block_rows(
            width=20000,
            default_block_rows=256,
            memory_info={"total_bytes": 24 * 1024**3, "free_bytes": 20 * 1024**3},
        )
        self.assertGreater(rows, 256)
        self.assertIn("adaptive", reason.lower())

    def test_compute_los_displacement(self):
        wavelength = 0.055465
        phase = 2 * np.pi
        los = compute_los_displacement(np.array([[phase]]), wavelength)
        self.assertAlmostEqual(
            float(los[0, 0]), wavelength / (4 * np.pi) * phase, places=5
        )

    def test_compute_los_displacement_zero(self):
        los = compute_los_displacement(np.array([[0.0]]), 0.055)
        self.assertEqual(float(los[0, 0]), 0.0)

    def test_phase_unwrapper_interface_is_abstract(self):
        with self.assertRaises(TypeError):
            PhaseUnwrapper()

    def test_icu_unwrapper_is_phase_unwrapper(self):
        self.assertIsInstance(ICUUnwrapper(), PhaseUnwrapper)

    def test_snaphu_unwrapper_is_phase_unwrapper(self):
        self.assertIsInstance(SNAPHUUnwrapper(), PhaseUnwrapper)

    def test_icu_unwrapper_rejects_empty_connected_components(self):
        class FakeBand:
            def __init__(self, data=None):
                self.data = data

            def WriteArray(self, data):
                self.data = data

            def ReadAsArray(self):
                return self.data

        class FakeDataset:
            def __init__(self, data=None):
                self.band = FakeBand(data)

            def GetRasterBand(self, _):
                return self.band

        class FakeDriver:
            def Create(self, *args, **kwargs):
                return FakeDataset()

        class FakeRaster:
            def __init__(self, *args, **kwargs):
                pass

            def close_dataset(self):
                pass

        fake_unw = np.zeros((4, 5), dtype=np.float32)
        fake_ccl = np.zeros((4, 5), dtype=np.uint8)
        open_calls = []

        def fake_open(path):
            open_calls.append(str(path))
            if str(path).endswith("unw.int"):
                return FakeDataset(fake_unw)
            if str(path).endswith("ccl.cc"):
                return FakeDataset(fake_ccl)
            return FakeDataset()

        fake_gdal = mock.Mock()
        fake_gdal.GDT_CFloat32 = object()
        fake_gdal.GDT_Float32 = object()
        fake_gdal.GDT_Byte = object()
        fake_gdal.GetDriverByName.return_value = FakeDriver()
        fake_gdal.Open.side_effect = fake_open

        with (
            mock.patch("osgeo.gdal.GetDriverByName", fake_gdal.GetDriverByName),
            mock.patch("osgeo.gdal.Open", fake_gdal.Open),
            mock.patch("osgeo.gdal.GDT_CFloat32", fake_gdal.GDT_CFloat32, create=True),
            mock.patch("osgeo.gdal.GDT_Float32", fake_gdal.GDT_Float32, create=True),
            mock.patch("osgeo.gdal.GDT_Byte", fake_gdal.GDT_Byte, create=True),
            mock.patch("isce3.io.Raster", FakeRaster),
            mock.patch("isce3.unwrap.ICU") as mock_icu,
            self.assertRaisesRegex(RuntimeError, "valid pixels|connected component"),
            ):
            mock_icu.return_value.unwrap.return_value = None
            ICUUnwrapper().unwrap(
                np.ones((4, 5), dtype=np.complex64),
                np.ones((4, 5), dtype=np.float32),
                object(),
                object(),
                "/tmp/dem.tif",
                tempfile.mkdtemp(),
            )

    def test_icu_unwrapper_accepts_zero_phase_when_connected_components_exist(self):
        class FakeBand:
            def __init__(self, data=None):
                self.data = data

            def WriteArray(self, data):
                self.data = data

            def ReadAsArray(self):
                return self.data

        class FakeDataset:
            def __init__(self, data=None):
                self.band = FakeBand(data)

            def GetRasterBand(self, _):
                return self.band

        class FakeDriver:
            def Create(self, *args, **kwargs):
                return FakeDataset()

        class FakeRaster:
            def __init__(self, *args, **kwargs):
                pass

            def close_dataset(self):
                pass

        fake_unw = np.zeros((4, 5), dtype=np.float32)
        fake_ccl = np.ones((4, 5), dtype=np.uint8)

        def fake_open(path):
            if str(path).endswith("unw.int"):
                return FakeDataset(fake_unw)
            if str(path).endswith("ccl.cc"):
                return FakeDataset(fake_ccl)
            return FakeDataset()

        fake_gdal = mock.Mock()
        fake_gdal.GDT_CFloat32 = object()
        fake_gdal.GDT_Float32 = object()
        fake_gdal.GDT_Byte = object()
        fake_gdal.GetDriverByName.return_value = FakeDriver()
        fake_gdal.Open.side_effect = fake_open

        fake_icu = mock.Mock()
        fake_icu.unwrap.return_value = None
        fake_io_module = types.SimpleNamespace(Raster=FakeRaster)
        fake_unwrap_module = types.SimpleNamespace(ICU=mock.Mock(return_value=fake_icu))
        import isce3

        with (
            mock.patch("osgeo.gdal.GetDriverByName", fake_gdal.GetDriverByName),
            mock.patch("osgeo.gdal.Open", fake_gdal.Open),
            mock.patch("osgeo.gdal.GDT_CFloat32", fake_gdal.GDT_CFloat32, create=True),
            mock.patch("osgeo.gdal.GDT_Float32", fake_gdal.GDT_Float32, create=True),
            mock.patch("osgeo.gdal.GDT_Byte", fake_gdal.GDT_Byte, create=True),
            mock.patch.dict(sys.modules, {"isce3.io": fake_io_module, "isce3.unwrap": fake_unwrap_module}),
            mock.patch.object(isce3, "io", fake_io_module, create=True),
            mock.patch.object(isce3, "unwrap", fake_unwrap_module, create=True),
            mock.patch("strip_insar._write_band_array", side_effect=lambda band, data, *a, **k: setattr(band, "data", data)),
            mock.patch("strip_insar._read_band_array", side_effect=lambda band, *a, **k: np.array(band.data, copy=True)),
        ):
            result = ICUUnwrapper().unwrap(
                np.ones((4, 5), dtype=np.complex64),
                np.ones((4, 5), dtype=np.float32),
                object(),
                object(),
                "/tmp/dem.tif",
                tempfile.mkdtemp(),
            )

        np.testing.assert_allclose(result, fake_unw)

    def test_icu_unwrapper_tiled_path_accepts_zero_phase_tiles(self):
        unwrapper = ICUUnwrapper()

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch.object(
                unwrapper,
                "_unwrap_once",
                side_effect=lambda interferogram, coherence, output_dir: np.zeros(
                    interferogram.shape, dtype=np.float32
                ),
            ),
            mock.patch.dict("os.environ", {"D2SAR_ICU_TILE_SIZE": "256", "D2SAR_ICU_TILE_OVERLAP": "0"}),
        ):
            result = unwrapper.unwrap(
                np.ones((300, 300), dtype=np.complex64),
                np.ones((300, 300), dtype=np.float32),
                object(),
                object(),
                "/tmp/dem.tif",
                tmpdir,
            )

        self.assertEqual(result.shape, (300, 300))
        self.assertTrue(np.all(result == 0.0))

    def test_icu_unwrapper_falls_back_to_tiled_unwrap(self):
        unwrapper = ICUUnwrapper()
        calls = []

        def fake_unwrap_once(interferogram, coherence, output_dir):
            calls.append((interferogram.shape, Path(output_dir).name))
            if Path(output_dir).name == "icu_full":
                raise RuntimeError("ICU unwrapping produced no connected component labels")
            return np.ones(interferogram.shape, dtype=np.float32)

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch.object(unwrapper, "_unwrap_once", side_effect=fake_unwrap_once),
            mock.patch.dict("os.environ", {"D2SAR_ICU_TILE_SIZE": "512", "D2SAR_ICU_TILE_OVERLAP": "64"}),
        ):
            result = unwrapper.unwrap(
                np.ones((300, 300), dtype=np.complex64),
                np.ones((300, 300), dtype=np.float32),
                object(),
                object(),
                "/tmp/dem.tif",
                tmpdir,
            )

        self.assertTrue(np.any(result != 0.0))
        self.assertGreater(len(calls), 1)
        self.assertEqual(calls[0], ((300, 300), "icu_full"))

    def test_icu_unwrapper_large_raster_uses_tiled_unwrap_directly(self):
        unwrapper = ICUUnwrapper()

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch.object(
                unwrapper,
                "_unwrap_tiled",
                return_value=np.ones((300, 300), dtype=np.float32),
            ) as mock_tiled,
            mock.patch.object(unwrapper, "_unwrap_once") as mock_once,
            mock.patch.dict("os.environ", {"D2SAR_ICU_TILE_SIZE": "256"}),
        ):
            result = unwrapper.unwrap(
                np.ones((300, 300), dtype=np.complex64),
                np.ones((300, 300), dtype=np.float32),
                object(),
                object(),
                "/tmp/dem.tif",
                tmpdir,
            )

        self.assertTrue(np.any(result != 0.0))
        mock_once.assert_not_called()
        mock_tiled.assert_called_once()

    def test_process_strip_insar_rejects_unknown_unwrap_method(self):
        with mock.patch(
            "strip_insar._create_unwrapper",
            side_effect=ValueError("Unknown unwrap_method: nonexistent"),
        ):
            with self.assertRaises(ValueError) as ctx:
                process_strip_insar(
                    "/dev/null",
                    "/dev/null",
                    "/tmp/out",
                    unwrap_method="nonexistent",
                    gpu_mode="cpu",
                )
            self.assertIn("nonexistent", str(ctx.exception))

    def test_process_strip_insar_cpu_pipeline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            master_mf = self._write_manifest(
                tmpdir,
                {
                    "sensor": "tianyi",
                    "slc": {"path": "dummy.tif"},
                },
            )
            slave_mf = self._write_manifest(
                tmpdir,
                {
                    "sensor": "tianyi",
                    "slc": {"path": "dummy2.tif"},
                },
            )
            out = Path(tmpdir) / "out"

            mock_acq = {"centerFrequency": 5405e6}
            with (
                mock.patch("strip_insar.load_manifest") as mock_load_m,
                mock.patch(
                    "strip_insar.detect_sensor_from_manifest", return_value="tianyi"
                ),
                mock.patch(
                    "strip_insar._load_master_metadata",
                    return_value=({}, {}, mock_acq, {}, {}),
                ),
                mock.patch("strip_insar.choose_orbit_interp", return_value="Legendre"),
                mock.patch(
                    "strip_insar.load_scene_corners_with_fallback", return_value=[]
                ),
                mock.patch(
                    "strip_insar._resolve_dem_path", return_value="/tmp/dem.tif"
                ),
                mock.patch(
                    "strip_insar.select_processing_backend", return_value=("gpu", "GPU")
                ),
                mock.patch(
                    "strip_insar.query_gpu_memory_info",
                    return_value={
                        "total_bytes": 6 * 1024**3,
                        "free_bytes": 5 * 1024**3,
                    },
                ),
                mock.patch("strip_insar._process_insar_gpu") as mock_gpu,
            ):
                mock_gpu.return_value = build_output_paths(out)
                result = process_strip_insar(
                    str(master_mf),
                    str(slave_mf),
                    str(out),
                    unwrap_method="icu",
                    gpu_mode="gpu",
                )
                self.assertEqual(result["backend_used"], "gpu")
                self.assertEqual(result["unwrap_method"], "icu")
                mock_gpu.assert_called_once()

    def test_write_insar_hdf_applies_crop_to_amplitude(self):
        import h5py

        with tempfile.TemporaryDirectory() as tmpdir:
            output_h5 = str(Path(tmpdir) / "out.h5")
            with (
                mock.patch(
                    "strip_insar._compute_slc_amplitude",
                    side_effect=[
                        np.ones((100, 200), dtype=np.float32),
                        np.ones((100, 200), dtype=np.float32),
                    ],
                )
            ):
                write_insar_hdf(
                    "master.tif",
                    "slave.tif",
                    np.ones((30, 40), dtype=np.complex64),
                    np.ones((30, 40), dtype=np.float32),
                    np.ones((30, 40), dtype=np.float32),
                    np.ones((30, 40), dtype=np.float32),
                    0.055,
                    "icu",
                    output_h5,
                    object(),
                    filtered_interferogram=np.ones((30, 40), dtype=np.complex64),
                    crop_request={
                        "mode": "window",
                        "master_window": {"row0": 10, "col0": 20, "rows": 30, "cols": 40},
                        "bbox": None,
                    },
                )

            with h5py.File(output_h5, "r") as f:
                self.assertEqual(f["avg_amplitude"].shape, (30, 40))
                self.assertEqual(f["interferogram"].shape, (30, 40))
                self.assertEqual(f["filtered_interferogram"].shape, (30, 40))

    def test_goldstein_filter_preserves_shape(self):
        ifg = np.ones((16, 16), dtype=np.complex64) + 1j * np.ones((16, 16), dtype=np.complex64)
        filtered = goldstein_filter(ifg, alpha=0.5)
        self.assertEqual(filtered.shape, ifg.shape)
        self.assertTrue(np.iscomplexobj(filtered))

    def test_estimate_coherence_from_complex_slcs_single_look_is_one(self):
        master = np.array([[1 + 0j, 1 + 0j]], dtype=np.complex64)
        slave = np.array([[1 + 0j, -1 + 0j]], dtype=np.complex64)

        coherence = _estimate_coherence_from_complex_slcs(master, slave, window_size=1)

        np.testing.assert_allclose(coherence, 1.0, atol=1e-6)

    def test_estimate_coherence_from_complex_slcs_windowed_captures_phase_cancellation(self):
        master = np.ones((9, 9), dtype=np.complex64)
        slave = np.ones((9, 9), dtype=np.complex64)
        slave[:, 1::2] *= -1

        coherence = _estimate_coherence_from_complex_slcs(master, slave, window_size=5)

        center = coherence[3:6, 3:6]
        self.assertLess(float(center.mean()), 0.3)

    def test_run_cpu_dense_offsets_recovers_known_shift(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rng = np.random.default_rng(1234)
            master = rng.normal(0.0, 1.0, size=(96, 96)).astype(np.float32)
            slave = np.zeros_like(master)
            slave[3:, 2:] = master[:-3, :-2]

            master_path = Path(tmpdir) / "master.tif"
            slave_path = Path(tmpdir) / "slave.tif"
            self._write_float_tiff(master_path, master)
            self._write_float_tiff(slave_path, slave)

            row_offset, col_offset, details = run_cpu_dense_offsets(
                master_slc_path=str(master_path),
                slave_slc_path=str(slave_path),
                output_dir=Path(tmpdir) / "dense",
                window_size=(32, 32),
                search_range=(6, 6),
                skip=(32, 32),
                max_windows=16,
                return_details=True,
            )

            self.assertIsNotNone(row_offset)
            self.assertIsNotNone(col_offset)
            self.assertIsNotNone(details)
            self.assertEqual(details["diagnostics"]["status"], "ok")
            self.assertAlmostEqual(float(np.median(details["row_sparse"])), 3.0, places=1)
            self.assertAlmostEqual(float(np.median(details["col_sparse"])), 2.0, places=1)

    def test_run_cpu_dense_offsets_uses_gross_offset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rng = np.random.default_rng(9012)
            master = rng.normal(0.0, 1.0, size=(128, 128)).astype(np.float32)
            slave = np.zeros_like(master)
            slave[8:, 9:] = master[:-8, :-9]

            master_path = Path(tmpdir) / "master.tif"
            slave_path = Path(tmpdir) / "slave.tif"
            self._write_float_tiff(master_path, master)
            self._write_float_tiff(slave_path, slave)

            row_offset, col_offset, details = run_cpu_dense_offsets(
                master_slc_path=str(master_path),
                slave_slc_path=str(slave_path),
                output_dir=Path(tmpdir) / "dense",
                window_size=(32, 32),
                search_range=(3, 3),
                skip=(32, 32),
                max_windows=16,
                gross_offset=(8.0, 9.0),
                return_details=True,
            )

            self.assertIsNotNone(row_offset)
            self.assertIsNotNone(col_offset)
            self.assertEqual(details["diagnostics"]["gross_offset"], {"azimuth": 8.0, "range": 9.0})
            self.assertAlmostEqual(float(np.median(details["row_sparse"])), 0.0, places=1)
            self.assertAlmostEqual(float(np.median(details["col_sparse"])), 0.0, places=1)

    def test_run_cpu_dense_offsets_rejects_boundary_hits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rng = np.random.default_rng(5678)
            master = rng.normal(0.0, 1.0, size=(64, 64)).astype(np.float32)
            slave = np.zeros_like(master)
            slave[4:, 4:] = master[:-4, :-4]

            master_path = Path(tmpdir) / "master.tif"
            slave_path = Path(tmpdir) / "slave.tif"
            self._write_float_tiff(master_path, master)
            self._write_float_tiff(slave_path, slave)

            row_offset, col_offset, details = run_cpu_dense_offsets(
                master_slc_path=str(master_path),
                slave_slc_path=str(slave_path),
                output_dir=Path(tmpdir) / "dense",
                window_size=(32, 32),
                search_range=(4, 4),
                skip=(32, 32),
                max_windows=4,
                return_details=True,
            )

            self.assertIsNone(row_offset)
            self.assertIsNone(col_offset)
            self.assertIsNotNone(details)
            self.assertEqual(details["diagnostics"]["status"], "failed")
            self.assertEqual(details["diagnostics"]["valid_points"], 0)

    def test_run_cpu_dense_offsets_fails_on_textureless_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            master = np.ones((64, 64), dtype=np.float32)
            slave = np.ones((64, 64), dtype=np.float32)

            master_path = Path(tmpdir) / "master.tif"
            slave_path = Path(tmpdir) / "slave.tif"
            self._write_float_tiff(master_path, master)
            self._write_float_tiff(slave_path, slave)

            row_offset, col_offset, details = run_cpu_dense_offsets(
                master_slc_path=str(master_path),
                slave_slc_path=str(slave_path),
                output_dir=Path(tmpdir) / "dense",
                window_size=(32, 32),
                search_range=(4, 4),
                skip=(32, 32),
                max_windows=4,
                return_details=True,
            )

            self.assertIsNone(row_offset)
            self.assertIsNone(col_offset)
            self.assertIsNotNone(details)
            self.assertEqual(details["diagnostics"]["status"], "failed")
            self.assertEqual(details["diagnostics"]["valid_points"], 0)

    def test_run_cpu_dense_offsets_limits_sampling_grid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rng = np.random.default_rng(2468)
            master = rng.normal(0.0, 1.0, size=(4096, 4096)).astype(np.float32)
            slave = master.copy()

            master_path = Path(tmpdir) / "master.tif"
            slave_path = Path(tmpdir) / "slave.tif"
            self._write_float_tiff(master_path, master)
            self._write_float_tiff(slave_path, slave)

            with mock.patch(
                "insar_registration._phase_correlate_search_window",
                return_value=(0.0, 0.0, 35.0),
            ):
                row_offset, col_offset, details = run_cpu_dense_offsets(
                    master_slc_path=str(master_path),
                    slave_slc_path=str(slave_path),
                    output_dir=Path(tmpdir) / "dense",
                    window_size=(64, 64),
                    search_range=(20, 20),
                    skip=(8, 8),
                    max_windows=100000,
                    return_details=True,
                )

            self.assertIsNotNone(row_offset)
            self.assertIsNotNone(col_offset)
            self.assertIsNotNone(details)
            self.assertEqual(details["diagnostics"]["status"], "ok")
            self.assertLessEqual(details["diagnostics"]["number_window_down"], 60)
            self.assertLessEqual(details["diagnostics"]["number_window_across"], 40)
            self.assertLessEqual(details["diagnostics"]["candidate_points"], 2400)

    def test_prepare_sparse_offsets_for_dense_model_filters_quality_and_rejects_outlier(self):
        row_coords = np.array([0.0, 40.0, 80.0, 120.0], dtype=np.float64)
        col_coords = np.array([0.0, 30.0, 60.0, 90.0], dtype=np.float64)
        rr, cc = np.meshgrid(row_coords, col_coords, indexing="ij")
        row_sparse = (
            1.0
            + 0.01 * rr
            - 0.02 * cc
            + 0.0001 * rr * cc
            + 0.00005 * rr * rr
            - 0.00008 * cc * cc
        ).astype(np.float32)
        col_sparse = (
            -2.0
            + 0.015 * rr
            + 0.01 * cc
            - 0.00007 * rr * cc
            + 0.00003 * rr * rr
            + 0.00002 * cc * cc
        ).astype(np.float32)
        snr = np.full((4, 4), 8.0, dtype=np.float32)
        correlation = np.full((4, 4), 0.95, dtype=np.float32)
        row_sparse[1, 1] += 7.0
        col_sparse[1, 1] -= 8.0
        snr[0, 3] = 1.5
        correlation[3, 0] = 0.2

        prepared, diagnostics = _prepare_sparse_offsets_for_dense_model(
            {
                "row_sparse": row_sparse,
                "col_sparse": col_sparse,
                "row_coords": row_coords,
                "col_coords": col_coords,
                "snr": snr,
                "correlation": correlation,
                "diagnostics": {"engine": "unit-test", "status": "ok"},
            },
            out_shape=(121, 91),
        )

        self.assertTrue(diagnostics["success"])
        self.assertEqual(diagnostics["quality_filter"]["input_points"], 16)
        self.assertEqual(diagnostics["quality_filter"]["kept_points"], 14)
        self.assertEqual(diagnostics["fit"]["iterations_requested"], 5)
        self.assertLess(diagnostics["fit"]["final_inliers"], diagnostics["quality_filter"]["kept_points"])
        self.assertTrue(np.isnan(prepared["row_sparse"][1, 1]))
        self.assertTrue(np.isnan(prepared["col_sparse"][1, 1]))
        self.assertTrue(np.isnan(prepared["snr"][0, 3]))
        self.assertTrue(np.isnan(prepared["correlation"][3, 0]))
        self.assertEqual(prepared["row_offset"].shape, (121, 91))
        self.assertEqual(prepared["col_offset"].shape, (121, 91))
        expected_center = (
            1.0
            + 0.01 * 40.0
            - 0.02 * 30.0
            + 0.0001 * 40.0 * 30.0
            + 0.00005 * 40.0 * 40.0
            - 0.00008 * 30.0 * 30.0
        )
        self.assertAlmostEqual(float(prepared["row_offset"][40, 30]), expected_center, places=2)

    def test_prepare_sparse_offsets_for_dense_model_fails_when_quality_filter_keeps_no_points(self):
        row_sparse = np.full((3, 3), np.nan, dtype=np.float32)
        col_sparse = np.full((3, 3), np.nan, dtype=np.float32)
        snr = np.full((3, 3), np.nan, dtype=np.float32)
        correlation = np.full((3, 3), np.nan, dtype=np.float32)
        row_sparse[0, 0] = 2.0
        col_sparse[0, 0] = -1.0
        snr[0, 0] = 1.0
        correlation[0, 0] = 0.2

        prepared, diagnostics = _prepare_sparse_offsets_for_dense_model(
            {
                "row_sparse": row_sparse,
                "col_sparse": col_sparse,
                "row_coords": np.arange(3, dtype=np.float64),
                "col_coords": np.arange(3, dtype=np.float64),
                "snr": snr,
                "correlation": correlation,
                "diagnostics": {"engine": "unit-test", "status": "ok"},
            },
            out_shape=(3, 3),
        )

        self.assertFalse(diagnostics["success"])
        self.assertEqual(diagnostics["reason"], "insufficient_points_after_quality_filter")
        self.assertTrue(np.isnan(prepared["row_sparse"][0, 0]))
        self.assertTrue(np.isnan(prepared["col_sparse"][0, 0]))

    def test_estimate_offset_mean_from_raster_uses_finite_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            offset = np.full((4, 5), -7.0, dtype=np.float32)
            offset[0, 0] = np.nan
            offset_path = Path(tmpdir) / "azimuth.tif"
            self._write_float_tiff(offset_path, offset)
            self.assertAlmostEqual(_estimate_offset_mean_from_raster(offset_path), -7.0)

    def test_select_cpu_dense_match_plan_prefers_staged_large_window_for_high_res(self):
        plan = _select_cpu_dense_match_plan(3.0)
        self.assertEqual(plan["gross_search_range"], (3, 3))
        self.assertGreaterEqual(len(plan["candidates"]), 2)
        self.assertEqual(plan["candidates"][0]["window_size"], (512, 512))
        self.assertEqual(plan["candidates"][0]["search_range"], (128, 128))
        self.assertEqual(plan["candidates"][1]["window_size"], (256, 256))
        self.assertEqual(plan["candidates"][1]["search_range"], (64, 64))
        self.assertEqual(plan["candidates"][2]["window_size"], (128, 128))
        self.assertEqual(plan["candidates"][2]["search_range"], (64, 64))

    def test_run_cpu_dense_offsets_records_staged_candidate_selection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rng = np.random.default_rng(1357)
            master = rng.normal(0.0, 1.0, size=(512, 512)).astype(np.float32)
            slave = master.copy()

            master_path = Path(tmpdir) / "master.tif"
            slave_path = Path(tmpdir) / "slave.tif"
            self._write_float_tiff(master_path, master)
            self._write_float_tiff(slave_path, slave)

            def fake_match(reference_patch, secondary_patch, search_range):
                if tuple(search_range) == (6, 6):
                    return 0.0, 0.0, 45.0
                return 0.0, 0.0, 25.0

            with mock.patch(
                "insar_registration._phase_correlate_search_window",
                side_effect=fake_match,
            ):
                row_offset, col_offset, details = run_cpu_dense_offsets(
                    master_slc_path=str(master_path),
                    slave_slc_path=str(slave_path),
                    output_dir=Path(tmpdir) / "dense",
                    skip=(16, 16),
                    max_windows=400,
                    window_candidates=[
                        {"window_size": (32, 32), "search_range": (4, 4)},
                        {"window_size": (32, 32), "search_range": (6, 6)},
                    ],
                    return_details=True,
                )

            self.assertIsNotNone(row_offset)
            self.assertIsNotNone(col_offset)
            self.assertEqual(details["diagnostics"]["search_range"], [6, 6])
            selection = details["diagnostics"]["candidate_selection"]
            self.assertEqual(selection["strategy"], "staged-preview")
            self.assertEqual(selection["selected_index"], 1)
            self.assertEqual(selection["selected_search_range"], [6, 6])

    def test_plan_matching_grid_limits_axis_counts_and_total(self):
        grid = _plan_matching_grid(
            rows=14580,
            cols=12544,
            window_size=(512, 512),
            search_range=(128, 128),
            skip=(32, 32),
            gross_offset=(-7.0, -41.0),
            max_windows=2400,
            max_window_down=60,
            max_window_across=40,
        )
        self.assertLessEqual(grid["number_window_down"], 60)
        self.assertLessEqual(grid["number_window_across"], 40)
        self.assertLessEqual(grid["number_window_down"] * grid["number_window_across"], 2400)

    def test_write_radar_wrapped_phase_png_downsamples_large_preview(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmpdir:
            interferogram = np.ones((5000, 3000), dtype=np.complex64) * np.exp(1j * 0.5)
            output_png = Path(tmpdir) / "wrapped_phase_radar.png"

            result = _write_radar_wrapped_phase_png(interferogram, output_png)

            self.assertEqual(result, str(output_png))
            self.assertTrue(output_png.exists())
            with Image.open(output_png) as image:
                self.assertEqual(image.mode, "RGB")
                self.assertLessEqual(max(image.size), 2048)

    def test_run_pycuampcor_returns_none_when_backend_unavailable(self):
        from osgeo import gdal

        with tempfile.TemporaryDirectory() as tmpdir:
            master = Path(tmpdir) / "master.tif"
            slave = Path(tmpdir) / "slave.tif"
            topo = Path(tmpdir) / "topo.tif"
            for path in (master, slave, topo):
                ds = gdal.GetDriverByName("GTiff").Create(str(path), 20, 10, 1, gdal.GDT_Float32)
                ds.GetRasterBand(1).WriteArray(np.ones((10, 20), dtype=np.float32))
                ds = None

            failed = subprocess.CompletedProcess(
                args=["python3", "-c", "..."],
                returncode=1,
                stdout="",
                stderr="backend unavailable",
            )
            with mock.patch("insar_registration.subprocess.run", return_value=failed):
                row_off, col_off = _run_pycuampcor(
                    str(master),
                    str(slave),
                    str(topo),
                    str(topo),
                    output_dir=Path(tmpdir),
                    gpu_id=0,
                )

            self.assertIsNone(row_off)
            self.assertIsNone(col_off)

    def test_run_pycuampcor_dense_offsets_returns_none_on_subprocess_failure(self):
        from osgeo import gdal

        with tempfile.TemporaryDirectory() as tmpdir:
            master = Path(tmpdir) / "master.tif"
            slave = Path(tmpdir) / "slave.tif"
            for path in (master, slave):
                ds = gdal.GetDriverByName("GTiff").Create(str(path), 20, 10, 1, gdal.GDT_Float32)
                ds.GetRasterBand(1).WriteArray(np.ones((10, 20), dtype=np.float32))
                ds = None

            failed = subprocess.CompletedProcess(
                args=["python3", "-c", "..."],
                returncode=1,
                stdout="",
                stderr="PyCuAmpcor failed",
            )
            with mock.patch("insar_registration.subprocess.run", return_value=failed):
                row_off, col_off = run_pycuampcor_dense_offsets(
                    master_slc_path=str(master),
                    slave_slc_path=str(slave),
                    output_dir=Path(tmpdir) / "ampcor",
                    gpu_id=0,
                )

            self.assertIsNone(row_off)
            self.assertIsNone(col_off)

    def test_run_pycuampcor_dense_offsets_writes_raw_ampcor_logs(self):
        from osgeo import gdal

        with tempfile.TemporaryDirectory() as tmpdir:
            master = Path(tmpdir) / "master.tif"
            slave = Path(tmpdir) / "slave.tif"
            for path in (master, slave):
                ds = gdal.GetDriverByName("GTiff").Create(str(path), 20, 10, 1, gdal.GDT_Float32)
                ds.GetRasterBand(1).WriteArray(np.ones((10, 20), dtype=np.float32))
                ds = None

            ampcor_dir = Path(tmpdir) / "ampcor"
            failed = subprocess.CompletedProcess(
                args=["python3", "-c", "..."],
                returncode=1,
                stdout="stdout line\n",
                stderr="stderr line\n",
            )
            with mock.patch("insar_registration.subprocess.run", return_value=failed):
                run_pycuampcor_dense_offsets(
                    master_slc_path=str(master),
                    slave_slc_path=str(slave),
                    output_dir=ampcor_dir,
                    gpu_id=0,
                )

            self.assertEqual(
                (ampcor_dir / "pycuampcor.stdout.log").read_text(encoding="utf-8"),
                "stdout line\n",
            )
            self.assertEqual(
                (ampcor_dir / "pycuampcor.stderr.log").read_text(encoding="utf-8"),
                "stderr line\n",
            )

    def test_run_pycuampcor_dense_offsets_materializes_local_inputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            master = Path(tmpdir) / "master.tif"
            slave = Path(tmpdir) / "slave.tif"
            master.write_bytes(b"master")
            slave.write_bytes(b"slave")

            ampcor_dir = Path(tmpdir) / "ampcor"
            failed = subprocess.CompletedProcess(
                args=["python3", "-c", "..."],
                returncode=1,
                stdout="",
                stderr="stderr line\n",
            )

            def fake_copy(src, dst):
                dst = Path(dst)
                dst.write_bytes(b"envi")
                dst.with_suffix(dst.suffix + ".hdr").write_text("ENVI", encoding="utf-8")
                return str(dst)

            with (
                mock.patch("insar_registration._raster_shape", return_value=(10, 20)),
                mock.patch("insar_registration._materialize_ampcor_input", side_effect=fake_copy) as mock_copy,
                mock.patch("insar_registration.subprocess.run", return_value=failed) as mock_run,
            ):
                run_pycuampcor_dense_offsets(
                    master_slc_path=str(master),
                    slave_slc_path=str(slave),
                    output_dir=ampcor_dir,
                    gpu_id=0,
                )

            self.assertEqual(mock_copy.call_count, 2)
            payload = json.loads(mock_run.call_args.args[0][3])
            self.assertIn("reference", payload["master_slc_path"])
            self.assertIn("secondary", payload["slave_slc_path"])

    def test_run_pycuampcor_dense_offsets_passes_isce3_like_dense_defaults(self):
        from osgeo import gdal

        with tempfile.TemporaryDirectory() as tmpdir:
            master = Path(tmpdir) / "master.tif"
            slave = Path(tmpdir) / "slave.tif"
            for path in (master, slave):
                ds = gdal.GetDriverByName("GTiff").Create(str(path), 128, 64, 1, gdal.GDT_Float32)
                ds.GetRasterBand(1).WriteArray(np.ones((64, 128), dtype=np.float32))
                ds = None

            failed = subprocess.CompletedProcess(
                args=["python3", "-c", "..."],
                returncode=1,
                stdout="",
                stderr="PyCuAmpcor failed",
            )
            with mock.patch("insar_registration.subprocess.run", return_value=failed) as mock_run:
                run_pycuampcor_dense_offsets(
                    master_slc_path=str(master),
                    slave_slc_path=str(slave),
                    output_dir=Path(tmpdir) / "ampcor",
                    gpu_id=3,
                )

            payload = json.loads(mock_run.call_args.args[0][3])
            self.assertEqual(payload["gpu_id"], 3)
            self.assertEqual(payload["window_size"], [64, 64])
            self.assertEqual(payload["search_range"], [20, 20])
            self.assertEqual(payload["skip"], [32, 32])
            self.assertEqual(payload["windows_batch_range"], 10)
            self.assertEqual(payload["windows_batch_azimuth"], 1)
            self.assertEqual(payload["cuda_streams"], 2)
            self.assertEqual(payload["raw_data_oversampling_factor"], 2)
            self.assertEqual(payload["correlation_statistics_zoom"], 21)
            self.assertEqual(payload["correlation_surface_zoom"], 8)
            self.assertEqual(payload["correlation_surface_oversampling_factor"], 64)
            self.assertEqual(payload["correlation_surface_oversampling_method"], "fft")

    def test_run_pycuampcor_dense_offsets_passes_gross_offset_filepath(self):
        from osgeo import gdal

        with tempfile.TemporaryDirectory() as tmpdir:
            master = Path(tmpdir) / "master.tif"
            slave = Path(tmpdir) / "slave.tif"
            gross = Path(tmpdir) / "gross_offsets.bin"
            gross.write_bytes(b"\x00" * 16)
            for path in (master, slave):
                ds = gdal.GetDriverByName("GTiff").Create(str(path), 128, 64, 1, gdal.GDT_Float32)
                ds.GetRasterBand(1).WriteArray(np.ones((64, 128), dtype=np.float32))
                ds = None

            failed = subprocess.CompletedProcess(
                args=["python3", "-c", "..."],
                returncode=1,
                stdout="",
                stderr="PyCuAmpcor failed",
            )
            with mock.patch("insar_registration.subprocess.run", return_value=failed) as mock_run:
                run_pycuampcor_dense_offsets(
                    master_slc_path=str(master),
                    slave_slc_path=str(slave),
                    output_dir=Path(tmpdir) / "ampcor",
                    gpu_id=0,
                    gross_offset_filepath=gross,
                )

            payload = json.loads(mock_run.call_args.args[0][3])
            self.assertEqual(payload["gross_offset_filepath"], str(gross))

    def test_run_pycuampcor_dense_offsets_passes_explicit_window_layout(self):
        from osgeo import gdal

        with tempfile.TemporaryDirectory() as tmpdir:
            master = Path(tmpdir) / "master.tif"
            slave = Path(tmpdir) / "slave.tif"
            for path in (master, slave):
                ds = gdal.GetDriverByName("GTiff").Create(str(path), 128, 64, 1, gdal.GDT_Float32)
                ds.GetRasterBand(1).WriteArray(np.ones((64, 128), dtype=np.float32))
                ds = None

            failed = subprocess.CompletedProcess(
                args=["python3", "-c", "..."],
                returncode=1,
                stdout="",
                stderr="PyCuAmpcor failed",
            )
            with mock.patch("insar_registration.subprocess.run", return_value=failed) as mock_run:
                run_pycuampcor_dense_offsets(
                    master_slc_path=str(master),
                    slave_slc_path=str(slave),
                    output_dir=Path(tmpdir) / "ampcor",
                    gpu_id=0,
                    reference_start_pixel_down=33,
                    reference_start_pixel_across=44,
                    number_window_down=5,
                    number_window_across=6,
                )

            payload = json.loads(mock_run.call_args.args[0][3])
            self.assertEqual(payload["reference_start_pixel_down"], 33)
            self.assertEqual(payload["reference_start_pixel_across"], 44)
            self.assertEqual(payload["number_window_down"], 5)
            self.assertEqual(payload["number_window_across"], 6)

    def test_run_pycuampcor_dense_offsets_accepts_resolution_tuned_window(self):
        from osgeo import gdal

        with tempfile.TemporaryDirectory() as tmpdir:
            master = Path(tmpdir) / "master.tif"
            slave = Path(tmpdir) / "slave.tif"
            for path in (master, slave):
                ds = gdal.GetDriverByName("GTiff").Create(str(path), 2048, 512, 1, gdal.GDT_Float32)
                ds.GetRasterBand(1).WriteArray(np.ones((512, 2048), dtype=np.float32))
                ds = None

            failed = subprocess.CompletedProcess(
                args=["python3", "-c", "..."],
                returncode=1,
                stdout="",
                stderr="PyCuAmpcor failed",
            )
            with mock.patch("insar_registration.subprocess.run", return_value=failed) as mock_run:
                run_pycuampcor_dense_offsets(
                    master_slc_path=str(master),
                    slave_slc_path=str(slave),
                    output_dir=Path(tmpdir) / "ampcor",
                    gpu_id=0,
                    window_size=(12, 512),
                )

            payload = json.loads(mock_run.call_args.args[0][3])
            self.assertEqual(payload["window_size"], [12, 512])

    def test_write_registration_outputs_marks_retry_when_fit_is_poor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            slave_slc = Path(tmpdir) / "slave.tif"
            coarse_slc = Path(tmpdir) / "coarse.tif"
            slave_slc.write_bytes(b"slc")
            coarse_slc.write_bytes(b"coarse")
            row = np.full((4, 5), 25.0, dtype=np.float32)
            col = np.full((4, 5), -30.0, dtype=np.float32)

            with mock.patch("insar_registration._copy_raster"):
                outputs = write_registration_outputs(
                    stage_path=Path(tmpdir) / "p1",
                    slave_slc_path=str(slave_slc),
                    coarse_coreg_slave_path=str(coarse_slc),
                    row_offset=row,
                    col_offset=col,
                    use_gpu=False,
                )

            model = json.loads(Path(outputs["registration_model"]).read_text(encoding="utf-8"))
            self.assertIn("fit_quality", model)
            self.assertIn("retry_recommended", model["fit_quality"])

    def test_write_registration_outputs_marks_dense_match_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            slave_slc = Path(tmpdir) / "slave.tif"
            coarse_slc = Path(tmpdir) / "coarse.tif"
            slave_slc.write_bytes(b"slc")
            coarse_slc.write_bytes(b"coarse")

            dense_details = {
                "row_sparse": np.zeros((0, 0), dtype=np.float32),
                "col_sparse": np.zeros((0, 0), dtype=np.float32),
                "row_coords": np.zeros((0,), dtype=np.float64),
                "col_coords": np.zeros((0,), dtype=np.float64),
                "snr": np.zeros((0, 0), dtype=np.float32),
                "covariance_az": np.zeros((0, 0), dtype=np.float32),
                "covariance_rg": np.zeros((0, 0), dtype=np.float32),
                "diagnostics": {
                    "engine": "cpu-template-search",
                    "status": "failed",
                    "reason": "insufficient_valid_points",
                    "valid_points": 0,
                },
            }

            with mock.patch("insar_registration._copy_raster"):
                outputs = write_registration_outputs(
                    stage_path=Path(tmpdir) / "p1",
                    slave_slc_path=str(slave_slc),
                    coarse_coreg_slave_path=str(coarse_slc),
                    row_offset=None,
                    col_offset=None,
                    dense_match_details=dense_details,
                    source="cpu-dense-match",
                    use_gpu=False,
                )

            model = json.loads(Path(outputs["registration_model"]).read_text(encoding="utf-8"))
            self.assertEqual(model["source"], "fine_match_failed")
            self.assertFalse(model["dense_match"]["success"])
            self.assertEqual(model["dense_match"]["reason"], "insufficient_valid_points")
            self.assertEqual(model["fit_quality"]["azimuth_rms"], 0.0)
            self.assertEqual(model["fit_quality"]["range_rms"], 0.0)

    def test_write_registration_outputs_applies_same_sparse_fit_path_for_cpu_and_gpu(self):
        for engine in ("cpu-template-search", "pycuampcor"):
            with self.subTest(engine=engine):
                with tempfile.TemporaryDirectory() as tmpdir:
                    slave_slc = Path(tmpdir) / "slave.tif"
                    coarse_slc = Path(tmpdir) / "coarse.tif"
                    slave_slc.write_bytes(b"slc")
                    coarse_slc.write_bytes(b"coarse")

                    row_sparse = np.full((3, 3), 0.5, dtype=np.float32)
                    col_sparse = np.full((3, 3), -0.25, dtype=np.float32)
                    snr = np.full((3, 3), 20.0, dtype=np.float32)
                    correlation = np.full((3, 3), 0.8, dtype=np.float32)
                    prepared = {
                        "row_sparse": np.full((3, 3), 0.5, dtype=np.float32),
                        "col_sparse": np.full((3, 3), -0.25, dtype=np.float32),
                        "row_coords": np.array([10.0, 20.0, 30.0], dtype=np.float64),
                        "col_coords": np.array([30.0, 40.0, 50.0], dtype=np.float64),
                        "snr": np.full((3, 3), 20.0, dtype=np.float32),
                        "correlation": np.full((3, 3), 0.8, dtype=np.float32),
                        "row_offset": np.zeros((2, 2), dtype=np.float32),
                        "col_offset": np.zeros((2, 2), dtype=np.float32),
                        "diagnostics": {
                            "engine": engine,
                            "status": "ok",
                            "reason": None,
                            "valid_points": 9,
                            "common_sparse_fit": {"success": True, "fit": {"final_inliers": 9}},
                        },
                    }

                    dense_details = {
                        "row_sparse": row_sparse,
                        "col_sparse": col_sparse,
                        "row_coords": np.array([10.0, 20.0, 30.0], dtype=np.float64),
                        "col_coords": np.array([30.0, 40.0, 50.0], dtype=np.float64),
                        "snr": snr,
                        "correlation": correlation,
                        "covariance_az": np.full((3, 3), 0.1, dtype=np.float32),
                        "covariance_rg": np.full((3, 3), 0.1, dtype=np.float32),
                        "diagnostics": {
                            "engine": engine,
                            "status": "ok",
                            "reason": None,
                            "valid_points": 9,
                        },
                    }

                    with (
                        mock.patch(
                            "insar_registration._prepare_sparse_offsets_for_dense_model",
                            return_value=(prepared, prepared["diagnostics"]["common_sparse_fit"]),
                        ) as mock_prepare,
                        mock.patch("insar_registration._copy_raster"),
                    ):
                        write_registration_outputs(
                            stage_path=Path(tmpdir) / "p1",
                            slave_slc_path=str(slave_slc),
                            coarse_coreg_slave_path=str(coarse_slc),
                            row_offset=np.zeros((2, 2), dtype=np.float32),
                            col_offset=np.zeros((2, 2), dtype=np.float32),
                            dense_match_details=dense_details,
                            source=engine,
                            use_gpu=(engine == "pycuampcor"),
                        )

                    mock_prepare.assert_called_once()
                    self.assertEqual(mock_prepare.call_args.kwargs["out_shape"], (2, 2))

    def test_run_pycuampcor_dense_offsets_writes_exception_log_on_timeout(self):
        from osgeo import gdal

        with tempfile.TemporaryDirectory() as tmpdir:
            master = Path(tmpdir) / "master.tif"
            slave = Path(tmpdir) / "slave.tif"
            for path in (master, slave):
                ds = gdal.GetDriverByName("GTiff").Create(str(path), 20, 10, 1, gdal.GDT_Float32)
                ds.GetRasterBand(1).WriteArray(np.ones((10, 20), dtype=np.float32))
                ds = None

            ampcor_dir = Path(tmpdir) / "ampcor"
            with mock.patch(
                "insar_registration.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd=["python3"], timeout=900),
            ):
                row_off, col_off = run_pycuampcor_dense_offsets(
                    master_slc_path=str(master),
                    slave_slc_path=str(slave),
                    output_dir=ampcor_dir,
                    gpu_id=0,
                )

            self.assertIsNone(row_off)
            self.assertIsNone(col_off)
            exc_text = (ampcor_dir / "pycuampcor.exception.log").read_text(encoding="utf-8")
            self.assertIn("TimeoutExpired", exc_text)

    def test_run_pycuampcor_dense_offsets_accepts_explicit_timeout(self):
        from osgeo import gdal

        with tempfile.TemporaryDirectory() as tmpdir:
            master = Path(tmpdir) / "master.tif"
            slave = Path(tmpdir) / "slave.tif"
            for path in (master, slave):
                ds = gdal.GetDriverByName("GTiff").Create(str(path), 20, 10, 1, gdal.GDT_Float32)
                ds.GetRasterBand(1).WriteArray(np.ones((10, 20), dtype=np.float32))
                ds = None

            failed = subprocess.CompletedProcess(
                args=["python3", "-c", "..."],
                returncode=1,
                stdout="",
                stderr="PyCuAmpcor failed",
            )
            with mock.patch("insar_registration.subprocess.run", return_value=failed) as mock_run:
                run_pycuampcor_dense_offsets(
                    master_slc_path=str(master),
                    slave_slc_path=str(slave),
                    output_dir=Path(tmpdir) / "ampcor",
                    gpu_id=0,
                    timeout_seconds=1800,
                )

            self.assertEqual(mock_run.call_args.kwargs["timeout"], 1800)

    def test_run_pycuampcor_dense_offsets_limits_sampling_grid(self):
        from osgeo import gdal

        with tempfile.TemporaryDirectory() as tmpdir:
            master = Path(tmpdir) / "master.tif"
            slave = Path(tmpdir) / "slave.tif"
            for path in (master, slave):
                ds = gdal.GetDriverByName("GTiff").Create(str(path), 4096, 4096, 1, gdal.GDT_Float32)
                ds.GetRasterBand(1).WriteArray(np.ones((4096, 4096), dtype=np.float32))
                ds = None

            payload_holder = {}

            def fake_run(cmd, stdout, stderr, text, env, timeout):
                payload_holder["cfg"] = json.loads(cmd[3])
                out_dir = Path(payload_holder["cfg"]["output_dir"])
                out_dir.mkdir(parents=True, exist_ok=True)
                np.zeros((60, 40, 2), dtype=np.float32).tofile(out_dir / "dense_offsets")
                np.full((60, 40), 5.0, dtype=np.float32).tofile(out_dir / "snr")
                np.zeros((60, 40, 3), dtype=np.float32).tofile(out_dir / "covariance")
                np.full((60, 40), 0.8, dtype=np.float32).tofile(out_dir / "correlation_peak")
                return subprocess.CompletedProcess(
                    args=cmd,
                    returncode=0,
                    stdout=json.dumps(
                        {
                            "number_window_down": 60,
                            "number_window_across": 40,
                            "reference_start_pixel_down": 20,
                            "reference_start_pixel_across": 20,
                            "skip_down": payload_holder["cfg"]["skip"][0],
                            "skip_across": payload_holder["cfg"]["skip"][1],
                            "window_size_height": 64,
                            "window_size_width": 64,
                        }
                    ),
                    stderr="",
                )

            with mock.patch("insar_registration.subprocess.run", side_effect=fake_run):
                row_offset, col_offset, details = run_pycuampcor_dense_offsets(
                    master_slc_path=str(master),
                    slave_slc_path=str(slave),
                    output_dir=Path(tmpdir) / "ampcor",
                    gpu_id=0,
                    window_size=(64, 64),
                    search_range=(20, 20),
                    skip=(8, 8),
                    return_details=True,
                )

            self.assertIsNotNone(row_offset)
            self.assertIsNotNone(col_offset)
            self.assertIsNotNone(details)
            cfg = payload_holder["cfg"]
            self.assertLessEqual(cfg["number_window_down"], 60)
            self.assertLessEqual(cfg["number_window_across"], 40)
            self.assertLessEqual(cfg["number_window_down"] * cfg["number_window_across"], 2400)

    def test_run_pycuampcor_dense_offsets_loads_correlation_peak(self):
        from osgeo import gdal

        with tempfile.TemporaryDirectory() as tmpdir:
            master = Path(tmpdir) / "master.tif"
            slave = Path(tmpdir) / "slave.tif"
            for path in (master, slave):
                ds = gdal.GetDriverByName("GTiff").Create(str(path), 128, 128, 1, gdal.GDT_Float32)
                ds.GetRasterBand(1).WriteArray(np.ones((128, 128), dtype=np.float32))
                ds = None

            def fake_run(cmd, stdout, stderr, text, env, timeout):
                cfg = json.loads(cmd[3])
                out_dir = Path(cfg["output_dir"])
                out_dir.mkdir(parents=True, exist_ok=True)
                shape = (3, 4)
                np.zeros((shape[0], shape[1], 2), dtype=np.float32).tofile(out_dir / "dense_offsets")
                np.full(shape, 5.0, dtype=np.float32).tofile(out_dir / "snr")
                np.zeros((shape[0], shape[1], 3), dtype=np.float32).tofile(out_dir / "covariance")
                np.full(shape, 0.75, dtype=np.float32).tofile(out_dir / "correlation_peak")
                return subprocess.CompletedProcess(
                    args=cmd,
                    returncode=0,
                    stdout=json.dumps(
                        {
                            "number_window_down": shape[0],
                            "number_window_across": shape[1],
                            "reference_start_pixel_down": 10,
                            "reference_start_pixel_across": 20,
                            "skip_down": 16,
                            "skip_across": 16,
                            "window_size_height": 64,
                            "window_size_width": 64,
                        }
                    ),
                    stderr="",
                )

            with mock.patch("insar_registration.subprocess.run", side_effect=fake_run):
                _, _, details = run_pycuampcor_dense_offsets(
                    master_slc_path=str(master),
                    slave_slc_path=str(slave),
                    output_dir=Path(tmpdir) / "ampcor",
                    gpu_id=0,
                    return_details=True,
                )

            self.assertIsNotNone(details)
            self.assertIn("correlation", details)
            np.testing.assert_allclose(details["correlation"], 0.75)

    def test_write_varying_gross_offset_file_samples_window_grid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rows, cols = 256, 256
            rg_path = Path(tmpdir) / "range.off"
            az_path = Path(tmpdir) / "azimuth.off"
            rg = np.full((rows, cols), -41.0, dtype=np.float64)
            az = np.full((rows, cols), -7.0, dtype=np.float64)
            rg_mm = np.memmap(rg_path, dtype=np.float64, mode="w+", shape=(rows, cols))
            az_mm = np.memmap(az_path, dtype=np.float64, mode="w+", shape=(rows, cols))
            rg_mm[:] = rg
            az_mm[:] = az
            rg_mm.flush()
            az_mm.flush()
            del rg_mm
            del az_mm

            gross_path = Path(tmpdir) / "gross_offsets.bin"
            meta = _write_varying_gross_offset_file(
                range_offset_path=rg_path,
                azimuth_offset_path=az_path,
                output_path=gross_path,
                full_shape=(rows, cols),
                window_size=(32, 32),
                search_range=(8, 8),
                skip=(16, 16),
            )

            raw = np.fromfile(gross_path, dtype=np.int32).reshape(-1, 2)
            self.assertEqual(meta["number_window_down"] * meta["number_window_across"], raw.shape[0])
            self.assertTrue(np.all(raw[:, 0] == -7))
            self.assertTrue(np.all(raw[:, 1] == -41))
            self.assertEqual(meta["reference_start_pixel_down"], 49)
            self.assertEqual(meta["reference_start_pixel_across"], 49)

    def test_run_pycuampcor_uses_given_stage_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stage_dir = Path(tmpdir) / "work" / "p1_dense_match"

            with mock.patch(
                "strip_insar.run_pycuampcor_dense_offsets",
                return_value=(None, None),
            ) as mock_dense:
                _run_pycuampcor(
                    "master.tif",
                    "slave.tif",
                    "master_topo.tif",
                    "slave_topo.tif",
                    output_dir=stage_dir,
                    gpu_id=0,
                )

            self.assertEqual(mock_dense.call_args.kwargs["output_dir"], stage_dir)

    def test_run_pycuampcor_passes_search_range(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stage_dir = Path(tmpdir) / "work" / "p1_dense_match"

            with mock.patch(
                "strip_insar.run_pycuampcor_dense_offsets",
                return_value=(None, None),
            ) as mock_dense:
                _run_pycuampcor(
                    "master.tif",
                    "slave.tif",
                    "master_topo.tif",
                    "slave_topo.tif",
                    output_dir=stage_dir,
                    gpu_id=0,
                    window_size=(512, 512),
                    search_range=(128, 128),
                )

            self.assertEqual(mock_dense.call_args.kwargs["window_size"], (512, 512))
            self.assertEqual(mock_dense.call_args.kwargs["search_range"], (128, 128))

    def test_write_registration_outputs_writes_offset_rasters_and_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            slave_slc = Path(tmpdir) / "slave.tif"
            slave_slc.write_bytes(b"slc")
            row = np.full((4, 5), 1.5, dtype=np.float32)
            col = np.full((4, 5), -0.5, dtype=np.float32)
            outputs = write_registration_outputs(
                stage_path=Path(tmpdir) / "p1",
                slave_slc_path=str(slave_slc),
                row_offset=row,
                col_offset=col,
                use_gpu=False,
            )
            self.assertTrue(Path(outputs["range_offsets"]).exists())
            self.assertTrue(Path(outputs["azimuth_offsets"]).exists())
            self.assertTrue(Path(outputs["registration_model"]).exists())
            model = json.loads(Path(outputs["registration_model"]).read_text(encoding="utf-8"))
            self.assertIn("fine_fit", model)
            self.assertIn("range_poly", model["fine_fit"])
            self.assertIn("azimuth_poly", model["fine_fit"])

    def test_write_registration_outputs_falls_back_to_cpu_for_fine_resample(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            slave_slc = Path(tmpdir) / "slave.tif"
            coarse_slc = Path(tmpdir) / "coarse.tif"
            slave_slc.write_bytes(b"slc")
            coarse_slc.write_bytes(b"coarse")
            row = np.full((4, 5), 1.5, dtype=np.float32)
            col = np.full((4, 5), -0.5, dtype=np.float32)

            with (
                mock.patch(
                    "insar_registration.run_resamp_isce3_v2",
                    side_effect=[
                        (False, "isce3-v2-resample_slc_blocks-gpu unavailable: cudaErrorMemoryAllocation"),
                        (True, "isce3-v2-resample_slc_blocks-cpu"),
                    ],
                ) as mock_resamp,
                mock.patch("insar_registration._copy_raster") as mock_copy,
            ):
                outputs = write_registration_outputs(
                    stage_path=Path(tmpdir) / "p1",
                    slave_slc_path=str(slave_slc),
                    coarse_coreg_slave_path=str(coarse_slc),
                    row_offset=row,
                    col_offset=col,
                    use_gpu=True,
                    radar_grid=object(),
                    ref_radar_grid=object(),
                )

            self.assertEqual(
                [call.kwargs["use_gpu"] for call in mock_resamp.call_args_list],
                [True, False],
            )
            first_call = mock_resamp.call_args_list[0]
            np.testing.assert_array_equal(first_call.kwargs["az_offset_dataset"], row)
            np.testing.assert_array_equal(first_call.kwargs["rg_offset_dataset"], col)
            mock_copy.assert_not_called()

            model = json.loads(Path(outputs["registration_model"]).read_text(encoding="utf-8"))
            self.assertIn("isce3-v2-resample_slc_blocks-cpu", model["fine_resamp_method"])
            self.assertIn("gpu failed", model["fine_resamp_method"])

    def test_write_registration_outputs_uses_original_slave_and_summed_offsets_for_fine_resample(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            slave_slc = Path(tmpdir) / "slave.tif"
            coarse_slc = Path(tmpdir) / "coarse.tif"
            coarse_range = Path(tmpdir) / "coarse_range.off"
            coarse_az = Path(tmpdir) / "coarse_az.off"
            slave_slc.write_bytes(b"slc")
            coarse_slc.write_bytes(b"coarse")
            coarse_range.write_bytes(b"rg")
            coarse_az.write_bytes(b"az")

            row_residual = np.full((4, 5), 1.5, dtype=np.float32)
            col_residual = np.full((4, 5), -0.5, dtype=np.float32)
            coarse_row = np.full((4, 5), -7.0, dtype=np.float32)
            coarse_col = np.full((4, 5), -41.0, dtype=np.float32)
            slave_grid = object()
            ref_grid = object()

            def fake_load(dataset, out_shape):
                if str(dataset) == str(coarse_range):
                    return coarse_col
                if str(dataset) == str(coarse_az):
                    return coarse_row
                raise AssertionError(f"unexpected offset dataset {dataset}")

            with (
                mock.patch("insar_registration._load_offset_dataset_for_resample", side_effect=fake_load),
                mock.patch(
                    "insar_registration.run_resamp_isce3_v2",
                    return_value=(True, "isce3-v2-resample_slc_blocks-cpu"),
                ) as mock_resamp,
                mock.patch("insar_registration._copy_raster") as mock_copy,
            ):
                outputs = write_registration_outputs(
                    stage_path=Path(tmpdir) / "p1",
                    slave_slc_path=str(slave_slc),
                    coarse_coreg_slave_path=str(coarse_slc),
                    row_offset=row_residual,
                    col_offset=col_residual,
                    coarse_az_offset_path=str(coarse_az),
                    coarse_rg_offset_path=str(coarse_range),
                    use_gpu=False,
                    radar_grid=slave_grid,
                    ref_radar_grid=ref_grid,
                )

            mock_copy.assert_not_called()
            mock_resamp.assert_called_once()
            call = mock_resamp.call_args
            self.assertEqual(call.kwargs["input_slc_path"], str(slave_slc))
            self.assertIs(call.kwargs["radar_grid"], slave_grid)
            self.assertIs(call.kwargs["ref_radar_grid"], ref_grid)
            np.testing.assert_array_equal(
                call.kwargs["az_offset_dataset"],
                coarse_row + row_residual,
            )
            np.testing.assert_array_equal(
                call.kwargs["rg_offset_dataset"],
                coarse_col + col_residual,
            )

            model = json.loads(Path(outputs["registration_model"]).read_text(encoding="utf-8"))
            self.assertEqual(model["fine_resample_input"], "original_slave_slc")
            self.assertAlmostEqual(model["offset_magnitude"]["azimuth_max_abs"], 5.5)
            self.assertAlmostEqual(model["offset_magnitude"]["range_max_abs"], 41.5)

    def test_write_registration_outputs_fit_quality_uses_residual_offsets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            slave_slc = Path(tmpdir) / "slave.tif"
            coarse_slc = Path(tmpdir) / "coarse.tif"
            coarse_range = Path(tmpdir) / "coarse_range.off"
            coarse_az = Path(tmpdir) / "coarse_az.off"
            slave_slc.write_bytes(b"slc")
            coarse_slc.write_bytes(b"coarse")
            coarse_range.write_bytes(b"rg")
            coarse_az.write_bytes(b"az")

            row_residual = np.full((4, 5), 0.5, dtype=np.float32)
            col_residual = np.full((4, 5), -0.25, dtype=np.float32)
            coarse_row = np.full((4, 5), -7.0, dtype=np.float32)
            coarse_col = np.full((4, 5), -41.0, dtype=np.float32)

            def fake_load(dataset, out_shape):
                if str(dataset) == str(coarse_range):
                    return coarse_col
                if str(dataset) == str(coarse_az):
                    return coarse_row
                raise AssertionError(f"unexpected offset dataset {dataset}")

            with (
                mock.patch("insar_registration._load_offset_dataset_for_resample", side_effect=fake_load),
                mock.patch(
                    "insar_registration.run_resamp_isce3_v2",
                    return_value=(True, "isce3-v2-resample_slc_blocks-cpu"),
                ),
                mock.patch("insar_registration._copy_raster"),
            ):
                outputs = write_registration_outputs(
                    stage_path=Path(tmpdir) / "p1",
                    slave_slc_path=str(slave_slc),
                    coarse_coreg_slave_path=str(coarse_slc),
                    row_offset=row_residual,
                    col_offset=col_residual,
                    coarse_az_offset_path=str(coarse_az),
                    coarse_rg_offset_path=str(coarse_range),
                    use_gpu=False,
                    radar_grid=object(),
                    ref_radar_grid=object(),
                )

            model = json.loads(Path(outputs["registration_model"]).read_text(encoding="utf-8"))
            self.assertAlmostEqual(model["fit_quality"]["azimuth_rms"], 0.5)
            self.assertAlmostEqual(model["fit_quality"]["range_rms"], 0.25)
            self.assertFalse(model["fit_quality"]["retry_recommended"])

    def test_write_registration_outputs_resamples_vsi_original_slave_when_coarse_offsets_exist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            coarse_slc = Path(tmpdir) / "coarse.tif"
            coarse_range = Path(tmpdir) / "coarse_range.off"
            coarse_az = Path(tmpdir) / "coarse_az.off"
            coarse_slc.write_bytes(b"coarse")
            coarse_range.write_bytes(b"rg")
            coarse_az.write_bytes(b"az")

            row_residual = np.full((4, 5), 1.0, dtype=np.float32)
            col_residual = np.full((4, 5), -2.0, dtype=np.float32)
            coarse_row = np.full((4, 5), -7.0, dtype=np.float32)
            coarse_col = np.full((4, 5), -41.0, dtype=np.float32)
            slave_slc = "/vsizip//tmp/input.zip/slave.tif"

            def fake_load(dataset, out_shape):
                if str(dataset) == str(coarse_range):
                    return coarse_col
                if str(dataset) == str(coarse_az):
                    return coarse_row
                raise AssertionError(f"unexpected offset dataset {dataset}")

            with (
                mock.patch("insar_registration._load_offset_dataset_for_resample", side_effect=fake_load),
                mock.patch(
                    "insar_registration.run_resamp_isce3_v2",
                    return_value=(True, "isce3-v2-resample_slc_blocks-gpu"),
                ) as mock_resamp,
                mock.patch("insar_registration._copy_raster") as mock_copy,
            ):
                write_registration_outputs(
                    stage_path=Path(tmpdir) / "p1",
                    slave_slc_path=slave_slc,
                    coarse_coreg_slave_path=str(coarse_slc),
                    row_offset=row_residual,
                    col_offset=col_residual,
                    coarse_az_offset_path=str(coarse_az),
                    coarse_rg_offset_path=str(coarse_range),
                    use_gpu=True,
                    radar_grid=object(),
                    ref_radar_grid=object(),
                )

            mock_resamp.assert_called()
            self.assertEqual(mock_resamp.call_args.kwargs["input_slc_path"], slave_slc)
            mock_copy.assert_not_called()

    def test_rubbersheet_dense_offsets_culls_outlier_and_fills_hole(self):
        row_sparse = np.ones((5, 5), dtype=np.float32)
        col_sparse = np.ones((5, 5), dtype=np.float32) * -2.0
        row_sparse[2, 2] = 10.0
        col_sparse[2, 2] = -20.0
        snr = np.full((5, 5), 10.0, dtype=np.float32)
        row_coords = np.arange(5, dtype=np.float64)
        col_coords = np.arange(5, dtype=np.float64)

        row_dense, col_dense, diagnostics = _rubbersheet_dense_offsets(
            row_sparse=row_sparse,
            col_sparse=col_sparse,
            row_coords=row_coords,
            col_coords=col_coords,
            out_shape=(5, 5),
            snr=snr,
        )

        self.assertEqual(row_dense.shape, (5, 5))
        self.assertEqual(col_dense.shape, (5, 5))
        self.assertTrue(np.isfinite(row_dense).all())
        self.assertTrue(np.isfinite(col_dense).all())
        self.assertLess(abs(float(row_dense[2, 2]) - 1.0), 0.5)
        self.assertLess(abs(float(col_dense[2, 2]) + 2.0), 0.5)
        self.assertGreater(diagnostics["masked_fraction_initial"], 0.0)
        self.assertEqual(diagnostics["interpolation_method"], "linear")

    def test_rubbersheet_dense_offsets_keeps_uniform_field_under_low_snr_threshold(self):
        row_sparse = np.full((5, 5), -1.25, dtype=np.float32)
        col_sparse = np.full((5, 5), 2.5, dtype=np.float32)
        snr = np.full((5, 5), 5.0, dtype=np.float32)

        row_dense, col_dense, diagnostics = _rubbersheet_dense_offsets(
            row_sparse=row_sparse,
            col_sparse=col_sparse,
            row_coords=np.arange(5, dtype=np.float64),
            col_coords=np.arange(5, dtype=np.float64),
            out_shape=(5, 5),
            snr=snr,
        )

        np.testing.assert_allclose(row_dense, -1.25, atol=1e-4)
        np.testing.assert_allclose(col_dense, 2.5, atol=1e-4)
        self.assertEqual(diagnostics["masked_fraction_initial"], 0.0)

    def test_run_p1_stage_uses_geo2rdr_offsets_for_coarse_resample(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "out"
            output_dir.mkdir(parents=True, exist_ok=True)

            master_manifest = Path(tmpdir) / "master.json"
            slave_manifest = Path(tmpdir) / "slave.json"
            master_manifest.write_text(
                json.dumps({"sensor": "tianyi", "slc": {"path": "master.tif"}}),
                encoding="utf-8",
            )
            slave_manifest.write_text(
                json.dumps({"sensor": "tianyi", "slc": {"path": "slave.tif"}}),
                encoding="utf-8",
            )

            master_orbit = {"header": {"firstStateTimeUTC": "2024-01-01T00:00:00Z"}}
            slave_orbit = {"header": {"firstStateTimeUTC": "2024-01-01T00:00:00Z"}}
            acq = {
                "centerFrequency": 5405e6,
                "prf": 1000.0,
                "lookDirection": "RIGHT",
                "polarisation": "VV",
                "startGPSTime": 100.0,
            }
            rg = {
                "numberOfRows": 100,
                "numberOfColumns": 200,
                "rangeTimeFirstPixel": 0.004,
                "columnSpacing": 1.2,
            }
            dop = {"combinedDoppler": {"coefficients": [0.0]}}

            coarse_range = output_dir / "work" / "p0_geo2rdr" / "coarse_range.off"
            coarse_az = output_dir / "work" / "p0_geo2rdr" / "coarse_azimuth.off"
            master_topo_vrt = output_dir / "geo2rdr_master" / "topo.vrt"
            coarse_range.parent.mkdir(parents=True, exist_ok=True)
            master_topo_vrt.parent.mkdir(parents=True, exist_ok=True)
            self._write_float_tiff(coarse_range, np.full((100, 200), -41.0, dtype=np.float32))
            self._write_float_tiff(coarse_az, np.full((100, 200), -7.0, dtype=np.float32))
            master_topo_vrt.write_text("vrt", encoding="utf-8")

            ref_grid = object()
            slave_grid = object()

            with (
                mock.patch(
                    "strip_insar.resolve_manifest_data_path",
                    side_effect=lambda manifest_path, rel: str(Path(manifest_path).with_name(rel)),
                ),
                mock.patch("strip_insar.construct_radar_grid", side_effect=[ref_grid, slave_grid]),
                mock.patch("strip_insar.construct_doppler_lut2d", return_value=object()),
                mock.patch(
                    "strip_insar._run_slave_geo2rdr_from_master_topo",
                    return_value=(str(coarse_range), str(coarse_az)),
                ) as mock_geo2rdr_offsets,
                mock.patch("strip_insar.run_coarse_resamp_isce3_v2", return_value=True) as mock_coarse_resamp,
                mock.patch("strip_insar._run_pycuampcor", side_effect=NotImplementedError("skip")),
                mock.patch(
                    "strip_insar.write_registration_outputs",
                    return_value={
                        "coarse_coreg_slave": str(output_dir / "work" / "p1_dense_match" / "coarse_coreg_slave.tif"),
                        "fine_coreg_slave": str(output_dir / "work" / "p1_dense_match" / "fine_coreg_slave.tif"),
                        "registration_model": str(output_dir / "work" / "p1_dense_match" / "registration_model.json"),
                        "range_offsets": str(output_dir / "work" / "p1_dense_match" / "range.off.tif"),
                        "azimuth_offsets": str(output_dir / "work" / "p1_dense_match" / "azimuth.off.tif"),
                    },
                ) as mock_write_registration,
                mock.patch("strip_insar._write_radar_amplitude_png", return_value=str(output_dir / "preview.png")),
            ):
                _run_p1_stage_from_cache(
                    output_dir=output_dir,
                    master_manifest_path=master_manifest,
                    slave_manifest_path=slave_manifest,
                    master_manifest={"slc": {"path": "master.tif"}},
                    slave_manifest={"slc": {"path": "slave.tif"}},
                    crop_request={"mode": "full", "master_window": {"row0": 0, "col0": 0, "rows": 100, "cols": 200}, "bbox": None},
                    backend_used="gpu",
                    gpu_id=0,
                    master_orbit_data=master_orbit,
                    master_acq_data=acq,
                    master_rg_data=rg,
                    slave_orbit_data=slave_orbit,
                    slave_acq_data=acq,
                    slave_rg_data=rg,
                    slave_dop_data=dop,
                )

            mock_geo2rdr_offsets.assert_called_once()
            self.assertEqual(mock_coarse_resamp.call_args.kwargs["rg_offset_path"], str(coarse_range))
            self.assertEqual(mock_coarse_resamp.call_args.kwargs["az_offset_path"], str(coarse_az))
            self.assertEqual(mock_write_registration.call_args.kwargs["coarse_rg_offset_path"], str(coarse_range))
            self.assertEqual(mock_write_registration.call_args.kwargs["coarse_az_offset_path"], str(coarse_az))
            self.assertIs(mock_write_registration.call_args.kwargs["radar_grid"], slave_grid)
            self.assertIs(mock_write_registration.call_args.kwargs["ref_radar_grid"], ref_grid)

    def test_run_slave_geo2rdr_from_master_topo_raises_when_isce_driver_unavailable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "out" / "work" / "p1_dense_match"
            output_dir.mkdir(parents=True, exist_ok=True)

            with mock.patch("strip_insar._isce_driver_write_supported", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "sampled-scalar fitting fallback is disabled"):
                    _run_slave_geo2rdr_from_master_topo(
                        master_topo_vrt_path=str(output_dir / "geo2rdr_master" / "topo.vrt"),
                        slave_orbit_data={"header": {"firstStateTimeUTC": "2024-01-01T00:00:00Z"}},
                        slave_acq_data={
                            "centerFrequency": 5405e6,
                            "prf": 1000.0,
                            "lookDirection": "RIGHT",
                            "polarisation": "VV",
                            "startGPSTime": 100.0,
                        },
                        slave_rg_data={
                            "numberOfRows": 100,
                            "numberOfColumns": 200,
                            "rangeTimeFirstPixel": 0.004,
                            "columnSpacing": 1.2,
                        },
                        slave_dop_data={"combinedDoppler": {"coefficients": [0.0]}},
                        output_dir=output_dir,
                        use_gpu=True,
                        gpu_id=0,
                    )

    def test_geo2rdr_pixel_offsets_use_slave_minus_master_convention(self):
        rg_offset, az_offset = _geo2rdr_pixel_offsets(
            master_col=100.0,
            master_row=20.0,
            slave_col=141.0,
            slave_row=27.0,
        )

        self.assertEqual(rg_offset, 41.0)
        self.assertEqual(az_offset, 7.0)

    def test_convert_geo2rdr_offsets_is_read_only_diagnostic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = 4
            cols = 5
            range_path = Path(tmpdir) / "range.off"
            azimuth_path = Path(tmpdir) / "azimuth.off"

            range_data = np.full((rows, cols), 41.5, dtype=np.float64)
            azimuth_data = np.full((rows, cols), 6.75, dtype=np.float64)
            range_data.tofile(range_path)
            azimuth_data.tofile(azimuth_path)

            decisions = _convert_geo2rdr_abs_to_relative_offsets(
                range_path=range_path,
                azimuth_path=azimuth_path,
                rows=rows,
                cols=cols,
            )

            self.assertEqual(decisions["range"]["mode"], "relative")
            self.assertFalse(decisions["range"]["should_subtract"])
            self.assertEqual(decisions["azimuth"]["mode"], "relative")
            self.assertFalse(decisions["azimuth"]["should_subtract"])
            np.testing.assert_allclose(
                np.fromfile(range_path, dtype=np.float64).reshape(rows, cols),
                range_data,
            )
            np.testing.assert_allclose(
                np.fromfile(azimuth_path, dtype=np.float64).reshape(rows, cols),
                azimuth_data,
            )

    def test_run_p1_stage_from_cache_uses_cpu_dense_match_when_gpu_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "out"
            output_dir.mkdir(parents=True, exist_ok=True)
            master_manifest = Path(tmpdir) / "master.json"
            slave_manifest = Path(tmpdir) / "slave.json"
            master_manifest.write_text(json.dumps({"slc": {"path": "master.tif"}}), encoding="utf-8")
            slave_manifest.write_text(json.dumps({"slc": {"path": "slave.tif"}}), encoding="utf-8")

            acq = {
                "centerFrequency": 5405e6,
                "prf": 1000.0,
                "lookDirection": "RIGHT",
                "polarisation": "VV",
                "startGPSTime": 100.0,
            }
            master_orbit = {"header": {"firstStateTimeUTC": "2024-01-01T00:00:00Z"}}
            slave_orbit = {"header": {"firstStateTimeUTC": "2024-01-01T00:00:00Z"}}
            rg = {
                "numberOfRows": 100,
                "numberOfColumns": 200,
                "rangeTimeFirstPixel": 0.004,
                "columnSpacing": 1.2,
                "groundRangeResolution": 3.0,
                "azimuthResolution": 3.0,
            }
            dop = {"combinedDoppler": {"coefficients": [0.0]}}

            coarse_range = output_dir / "work" / "p0_geo2rdr" / "coarse_range.off"
            coarse_az = output_dir / "work" / "p0_geo2rdr" / "coarse_azimuth.off"
            master_topo_vrt = output_dir / "geo2rdr_master" / "topo.vrt"
            coarse_range.parent.mkdir(parents=True, exist_ok=True)
            master_topo_vrt.parent.mkdir(parents=True, exist_ok=True)
            self._write_float_tiff(coarse_range, np.full((100, 200), -41.0, dtype=np.float32))
            self._write_float_tiff(coarse_az, np.full((100, 200), -7.0, dtype=np.float32))
            master_topo_vrt.write_text("vrt", encoding="utf-8")

            ref_grid = object()
            slave_grid = object()
            row_offset = np.full((100, 200), 0.25, dtype=np.float32)
            col_offset = np.full((100, 200), -0.5, dtype=np.float32)
            dense_details = {
                "row_sparse": np.full((2, 2), 0.25, dtype=np.float32),
                "col_sparse": np.full((2, 2), -0.5, dtype=np.float32),
                "row_coords": np.array([10.0, 60.0], dtype=np.float64),
                "col_coords": np.array([20.0, 120.0], dtype=np.float64),
                "snr": np.full((2, 2), 9.0, dtype=np.float32),
                "covariance_az": np.full((2, 2), 0.1, dtype=np.float32),
                "covariance_rg": np.full((2, 2), 0.1, dtype=np.float32),
                "diagnostics": {"engine": "cpu"},
            }

            with (
                mock.patch(
                    "strip_insar.resolve_manifest_data_path",
                    side_effect=lambda manifest_path, rel: str(Path(manifest_path).with_name(rel)),
                ),
                mock.patch("strip_insar.construct_radar_grid", side_effect=[ref_grid, slave_grid]),
                mock.patch("strip_insar.construct_doppler_lut2d", return_value=object()),
                mock.patch(
                    "strip_insar._run_slave_geo2rdr_from_master_topo",
                    return_value=(str(coarse_range), str(coarse_az)),
                ),
                mock.patch("strip_insar.run_coarse_resamp_isce3_v2", return_value=True),
                mock.patch(
                    "strip_insar.run_cpu_dense_offsets",
                    return_value=(row_offset, col_offset, dense_details),
                    create=True,
                ) as mock_cpu_dense,
                mock.patch("strip_insar.write_registration_outputs") as mock_write_registration,
                mock.patch("strip_insar._write_radar_amplitude_png", return_value=str(output_dir / "preview.png")),
            ):
                def fake_write_registration_outputs(**kwargs):
                    stage_path = Path(kwargs["stage_path"])
                    stage_path.mkdir(parents=True, exist_ok=True)
                    registration_model = stage_path / "registration_model.json"
                    registration_model.write_text(
                        json.dumps(
                            {
                                "fit_quality": {"retry_recommended": False},
                            }
                        ),
                        encoding="utf-8",
                    )
                    return {
                        "coarse_coreg_slave": str(stage_path / "coarse_coreg_slave.tif"),
                        "fine_coreg_slave": str(stage_path / "fine_coreg_slave.tif"),
                        "registration_model": str(registration_model),
                        "range_offsets": str(stage_path / "range.off.tif"),
                        "azimuth_offsets": str(stage_path / "azimuth.off.tif"),
                    }

                mock_write_registration.side_effect = fake_write_registration_outputs
                _run_p1_stage_from_cache(
                    output_dir=output_dir,
                    master_manifest_path=master_manifest,
                    slave_manifest_path=slave_manifest,
                    master_manifest={"slc": {"path": "master.tif"}},
                    slave_manifest={"slc": {"path": "slave.tif"}},
                    crop_request={"mode": "full", "master_window": {"row0": 0, "col0": 0, "rows": 100, "cols": 200}, "bbox": None},
                    backend_used="cpu",
                    gpu_id=0,
                    master_orbit_data=master_orbit,
                    master_acq_data=acq,
                    master_rg_data=rg,
                    slave_orbit_data=slave_orbit,
                    slave_acq_data=acq,
                    slave_rg_data=rg,
                    slave_dop_data=dop,
                )

            mock_cpu_dense.assert_called_once()
            self.assertEqual(mock_cpu_dense.call_args.kwargs["gross_offset"], (-7.0, -41.0))
            self.assertIn("window_candidates", mock_cpu_dense.call_args.kwargs)
            self.assertGreaterEqual(len(mock_cpu_dense.call_args.kwargs["window_candidates"]), 2)
            self.assertIs(mock_write_registration.call_args.kwargs["row_offset"], row_offset)
            self.assertIs(mock_write_registration.call_args.kwargs["col_offset"], col_offset)
            self.assertEqual(mock_write_registration.call_args.kwargs["source"], "cpu-dense-match")

    def test_run_p1_stage_from_cache_records_coarse_geo2rdr_outputs_under_p1_dense_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "out"
            output_dir.mkdir(parents=True, exist_ok=True)
            master_manifest = Path(tmpdir) / "master.json"
            slave_manifest = Path(tmpdir) / "slave.json"
            master_manifest.write_text(json.dumps({"slc": {"path": "master.tif"}}), encoding="utf-8")
            slave_manifest.write_text(json.dumps({"slc": {"path": "slave.tif"}}), encoding="utf-8")

            acq = {
                "centerFrequency": 5405e6,
                "prf": 1000.0,
                "lookDirection": "RIGHT",
                "polarisation": "VV",
                "startGPSTime": 100.0,
            }
            master_orbit = {"header": {"firstStateTimeUTC": "2024-01-01T00:00:00Z"}}
            slave_orbit = {"header": {"firstStateTimeUTC": "2024-01-01T00:00:00Z"}}
            rg = {
                "numberOfRows": 100,
                "numberOfColumns": 200,
                "rangeTimeFirstPixel": 0.004,
                "columnSpacing": 1.2,
                "groundRangeResolution": 3.0,
                "azimuthResolution": 3.0,
            }
            dop = {"combinedDoppler": {"coefficients": [0.0]}}

            p0_stage = output_dir / "work" / "p0_geo2rdr"
            p0_stage.mkdir(parents=True, exist_ok=True)
            (p0_stage / "stage.json").write_text(
                json.dumps(
                    {
                        "stage": "p0",
                        "success": True,
                        "output_files": {
                            "master_topo_vrt": str(output_dir / "geo2rdr_master" / "topo.vrt"),
                        },
                    }
                ),
                encoding="utf-8",
            )
            (p0_stage / "SUCCESS").write_text("success\n", encoding="utf-8")
            (output_dir / "geo2rdr_master").mkdir(parents=True, exist_ok=True)
            (output_dir / "geo2rdr_master" / "topo.vrt").write_text("vrt", encoding="utf-8")

            p1_stage = output_dir / "work" / "p1_dense_match"
            p1_stage.mkdir(parents=True, exist_ok=True)
            coarse_range = p1_stage / "coarse_geo2rdr_range.off"
            coarse_az = p1_stage / "coarse_geo2rdr_azimuth.off"
            coarse_model = p1_stage / "coarse_geo2rdr_model.json"
            coarse_range.write_bytes(b"rg")
            coarse_az.write_bytes(b"az")
            coarse_model.write_text("{}", encoding="utf-8")

            with (
                mock.patch(
                    "strip_insar.resolve_manifest_data_path",
                    side_effect=lambda manifest_path, rel: str(Path(manifest_path).with_name(rel)),
                ),
                mock.patch("strip_insar.construct_radar_grid", side_effect=[object(), object()]),
                mock.patch("strip_insar.construct_doppler_lut2d", return_value=object()),
                mock.patch(
                    "strip_insar._run_slave_geo2rdr_from_master_topo",
                    return_value=(str(coarse_range), str(coarse_az)),
                ),
                mock.patch("strip_insar.run_coarse_resamp_isce3_v2", return_value=True),
                mock.patch("strip_insar.run_cpu_dense_offsets", return_value=(None, None, None), create=True),
                mock.patch(
                    "strip_insar.write_registration_outputs",
                    return_value={
                        "coarse_coreg_slave": str(p1_stage / "coarse_coreg_slave.tif"),
                        "fine_coreg_slave": str(p1_stage / "fine_coreg_slave.tif"),
                        "registration_model": str(p1_stage / "registration_model.json"),
                        "range_offsets": str(p1_stage / "range.off.tif"),
                        "azimuth_offsets": str(p1_stage / "azimuth.off.tif"),
                    },
                ),
                mock.patch("strip_insar._write_radar_amplitude_png", return_value=str(output_dir / "preview.png")),
            ):
                _run_p1_stage_from_cache(
                    output_dir=output_dir,
                    master_manifest_path=master_manifest,
                    slave_manifest_path=slave_manifest,
                    master_manifest={"slc": {"path": "master.tif"}},
                    slave_manifest={"slc": {"path": "slave.tif"}},
                    crop_request={"mode": "full", "master_window": {"row0": 0, "col0": 0, "rows": 100, "cols": 200}, "bbox": None},
                    backend_used="cpu",
                    gpu_id=0,
                    master_orbit_data=master_orbit,
                    master_acq_data=acq,
                    master_rg_data=rg,
                    slave_orbit_data=slave_orbit,
                    slave_acq_data=acq,
                    slave_rg_data=rg,
                    slave_dop_data=dop,
                )

            p1_record = json.loads((p1_stage / "stage.json").read_text(encoding="utf-8"))
            self.assertEqual(p1_record["output_files"]["coarse_geo2rdr_range_offsets"], str(coarse_range))
            self.assertEqual(p1_record["output_files"]["coarse_geo2rdr_azimuth_offsets"], str(coarse_az))
            self.assertEqual(p1_record["output_files"]["coarse_geo2rdr_model"], str(coarse_model))
            self.assertNotIn("/p1_geo2rdr_offsets/", p1_record["output_files"]["coarse_geo2rdr_model"])

    def test_load_offset_dataset_for_resample_supports_raw_float64(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_shape = (3, 4)
            path = Path(tmpdir) / "range.off"
            data = np.arange(12, dtype=np.float64).reshape(out_shape)
            mm = np.memmap(path, dtype=np.float64, mode="w+", shape=out_shape)
            mm[:] = data
            mm.flush()
            del mm

            loaded = _load_offset_dataset_for_resample(path, out_shape)

            self.assertIsInstance(loaded, np.memmap)
            self.assertEqual(loaded.shape, out_shape)
            self.assertTrue(np.allclose(np.asarray(loaded), data))


if __name__ == "__main__":
    unittest.main()
