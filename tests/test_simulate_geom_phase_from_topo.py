import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from osgeo import gdal
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

if "h5py" not in sys.modules:
    sys.modules["h5py"] = mock.MagicMock()

import simulate_geom_phase_from_topo as geom_phase


class SimulateGeomPhaseFormulaTests(unittest.TestCase):
    def test_compute_phase_products_matches_master_conj_slave_convention(self):
        master_range = np.array(
            [
                [850000.0, 850002.5],
                [850005.0, 850007.5],
            ],
            dtype=np.float64,
        )
        slave_true = np.array(
            [
                [850000.20, 850002.15],
                [850005.45, np.nan],
            ],
            dtype=np.float64,
        )
        slave_flat = np.array(
            [
                [850000.05, 850002.00],
                [850005.10, np.nan],
            ],
            dtype=np.float64,
        )
        wavelength = 0.05546576

        products = geom_phase.compute_phase_products(
            master_range_m=master_range,
            slave_true_range_m=slave_true,
            slave_flat_range_m=slave_flat,
            wavelength_m=wavelength,
        )

        expected_geom = geom_phase.wrap_phase(4.0 * np.pi * (slave_true - master_range) / wavelength)
        expected_flat = geom_phase.wrap_phase(4.0 * np.pi * (slave_flat - master_range) / wavelength)
        expected_topo = geom_phase.wrap_phase(
            4.0 * np.pi * (slave_true - slave_flat) / wavelength
        )
        valid = np.isfinite(expected_geom) & np.isfinite(expected_flat) & np.isfinite(expected_topo)

        self.assertTrue(np.allclose(products["phi_geom"][valid], expected_geom[valid], atol=1e-7))
        self.assertTrue(np.allclose(products["phi_flat"][valid], expected_flat[valid], atol=1e-7))
        self.assertTrue(np.allclose(products["phi_topo"][valid], expected_topo[valid], atol=1e-7))
        self.assertTrue(np.isnan(products["phi_geom"][1, 1]))
        self.assertTrue(np.isnan(products["phi_flat"][1, 1]))
        self.assertTrue(np.isnan(products["phi_topo"][1, 1]))

    def test_normalize_offsets_and_convert_to_slave_range(self):
        offsets = np.array(
            [
                [0.0, 1.25, -999999.0],
                [-0.5, -1000000.0, -1.0e6],
            ],
            dtype=np.float64,
        )
        normalized = geom_phase.normalize_geo2rdr_offsets(offsets)
        slave_range = geom_phase.compute_slave_range_from_offsets(
            normalized,
            slave_starting_range_m=910000.0,
            slave_range_spacing_m=2.5,
        )

        self.assertAlmostEqual(float(slave_range[0, 0]), 910000.0, places=6)
        self.assertAlmostEqual(float(slave_range[0, 1]), 910005.625, places=6)
        self.assertAlmostEqual(float(slave_range[1, 0]), 909998.75, places=6)
        self.assertTrue(np.isnan(slave_range[0, 2]))
        self.assertTrue(np.isnan(slave_range[1, 1]))
        self.assertTrue(np.isnan(slave_range[1, 2]))


class SimulateGeomPhaseOutputTests(unittest.TestCase):
    def test_write_phase_products_writes_npy_tif_png_and_summary(self):
        phase = np.array(
            [
                [0.0, np.pi / 2.0],
                [-np.pi / 2.0, np.nan],
            ],
            dtype=np.float32,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            paths = geom_phase.write_phase_products(
                out_dir,
                {
                    "phi_geom": phase,
                    "phi_flat": phase / 2.0,
                    "phi_topo": phase / 3.0,
                },
                summary={
                    "wavelength_m": 0.055,
                    "combined_phase_name": "phi_geom",
                },
            )

            self.assertTrue((out_dir / "phi_geom.npy").exists())
            self.assertTrue((out_dir / "phi_geom.tif").exists())
            self.assertTrue((out_dir / "phi_geom.png").exists())
            self.assertTrue((out_dir / "summary.json").exists())
            self.assertEqual(paths["combined_phase_name"], "phi_geom")

            ds = gdal.Open(str(out_dir / "phi_geom.tif"), gdal.GA_ReadOnly)
            self.assertIsNotNone(ds)
            tif_arr = ds.GetRasterBand(1).ReadAsArray()
            ds = None
            self.assertAlmostEqual(float(tif_arr[0, 1]), float(np.pi / 2.0), places=6)

            npy_arr = np.load(out_dir / "phi_geom.npy")
            self.assertAlmostEqual(float(npy_arr[1, 0]), float(-np.pi / 2.0), places=6)

            with Image.open(out_dir / "phi_geom.png") as img:
                self.assertEqual(img.mode, "RGB")
                self.assertEqual(img.size, (2, 2))

            summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["combined_phase_name"], "phi_geom")
            self.assertEqual(summary["products"]["phi_topo"]["png"], str(out_dir / "phi_topo.png"))


class SimulateGeomPhaseCliTests(unittest.TestCase):
    def test_parse_args_requires_flat_height(self):
        base_args = [
            "--master-acquisition-json",
            "m_acq.json",
            "--master-radargrid-json",
            "m_rg.json",
            "--master-orbit-json",
            "m_orbit.json",
            "--slave-acquisition-json",
            "s_acq.json",
            "--slave-radargrid-json",
            "s_rg.json",
            "--slave-orbit-json",
            "s_orbit.json",
            "--master-topo-vrt",
            "master_topo.vrt",
            "--out-dir",
            "out",
        ]
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                geom_phase.parse_args(base_args)

        args = geom_phase.parse_args(base_args + ["--flat-height", "0.0"])
        self.assertEqual(args.flat_height, 0.0)
        self.assertEqual(args.master_topo_vrt, "master_topo.vrt")


if __name__ == "__main__":
    unittest.main()
