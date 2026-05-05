import json
import sys
import tempfile
import types
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


class TopsInsarBackendTests(unittest.TestCase):
    def _write_tops_manifest(self, root: Path, name: str) -> Path:
        out = root / name
        metadata = out / "metadata"
        metadata.mkdir(parents=True)
        acquisition = {
            "startTimeUTC": "2023-11-10T04:39:48+00:00",
            "centerFrequency": 5405000454.33435,
            "wavelength": 0.05546576,
            "prf": 1717.0,
        }
        radargrid = {
            "numberOfRows": 8,
            "numberOfColumns": 4,
            "rangeTimeFirstPixel": 0.0045,
            "columnSpacing": 2.3,
            "groundRangeResolution": 5.0,
            "azimuthResolution": 15.0,
        }
        tops = {
            "swath": "IW2",
            "bursts": [
                {
                    "burstIndex": 1,
                    "lineOffset": 0,
                    "numberOfLines": 4,
                    "numberOfSamples": 4,
                    "firstValidLine": 0,
                    "numValidLines": 4,
                    "firstValidSample": 0,
                    "numValidSamples": 4,
                    "sensingStart": "2023-11-10T04:39:48+00:00",
                    "azimuthTimeInterval": 0.002,
                    "radarWavelength": 0.05546576,
                },
                {
                    "burstIndex": 2,
                    "lineOffset": 4,
                    "numberOfLines": 4,
                    "numberOfSamples": 4,
                    "firstValidLine": 0,
                    "numValidLines": 4,
                    "firstValidSample": 0,
                    "numValidSamples": 4,
                    "sensingStart": "2023-11-10T04:39:50+00:00",
                    "azimuthTimeInterval": 0.002,
                    "radarWavelength": 0.05546576,
                },
            ],
            "overlaps": [
                {
                    "previousBurstIndex": 1,
                    "nextBurstIndex": 2,
                    "estimatedOverlapLines": 2,
                }
            ],
        }
        for key, value in {
            "acquisition": acquisition,
            "radargrid": radargrid,
            "tops": tops,
        }.items():
            (metadata / f"{key}.json").write_text(json.dumps(value), encoding="utf-8")
        manifest = {
            "sensor": "sentinel-1",
            "slc": {"path": str(out / "dummy.slc")},
            "metadata": {
                "acquisition": str(metadata / "acquisition.json"),
                "radargrid": str(metadata / "radargrid.json"),
                "tops": str(metadata / "tops.json"),
            },
        }
        manifest_path = out / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest_path

    def test_local_backend_passes_tops_metadata_to_strip_insar2(self) -> None:
        with mock.patch.dict(sys.modules, {"h5py": types.ModuleType("h5py")}):
            import tops_insar
            import strip_insar2

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            master = self._write_tops_manifest(root, "master")
            slave = self._write_tops_manifest(root, "slave")
            args = Namespace(
                swath="IW2",
                start_stage="check",
                end_stage="p6",
                stop_after=None,
                resume=False,
                execute_stages=True,
                gpu_mode="cpu",
                topo_gpu=False,
                gpu_id=0,
                dem=str(root / "dem.tif"),
                burst_limit=None,
                resolution=20.0,
                range_looks=1,
                azimuth_looks=1,
                block_rows=None,
                dem_cache_dir=None,
                dem_margin_deg=0.2,
                no_kml=True,
                extra_esd_cycles=0.0,
                esd_coherence_threshold=0.85,
                unwrap_method="icu",
                execute_backend=False,
                backend_timeout_seconds=60,
                do_ionospheric_correction=False,
            )
            plan = tops_insar._build_plan_for_manifests(master, slave, root / "out", ["IW2"], args, [])

            def fake_process(*process_args, **kwargs):
                return {
                    "pair_name": "mock_pair",
                    "pair_dir": str(root / "backend_pair"),
                    "exports": {"interferogram_tif": str(root / "ifg.tif")},
                    "stage_backends": {"geo2rdr": "cpu", "crossmul": "cpu"},
                    "fallback_reasons": {},
                }

            with mock.patch.object(strip_insar2, "process_strip_insar2", side_effect=fake_process) as patched:
                record = tops_insar._run_local_tops_backend_for_swath(plan, "IW2", stop_after_stage="p6")

        self.assertEqual(record["backend"], "strip_insar2-local-tops")
        self.assertEqual(record["stop_after_stage"], "p6")
        patched.assert_called_once()
        call = patched.call_args
        self.assertEqual(Path(call.args[0]), master)
        self.assertEqual(Path(call.args[1]), slave)
        self.assertTrue(call.kwargs["tops_mode"])
        self.assertEqual(call.kwargs["gpu_mode"], "cpu")
        self.assertEqual(call.kwargs["unwrap_method"], "icu")
        self.assertEqual(call.kwargs["range_looks"], 1)
        self.assertEqual(call.kwargs["azimuth_looks"], 1)
        self.assertEqual(len(call.kwargs["master_bursts"]), 2)
        self.assertIsInstance(call.kwargs["master_bursts"][0], strip_insar2.TopsBurstInfo)
        self.assertEqual(call.kwargs["master_bursts"][0].burst_index, 0)
        self.assertEqual(call.kwargs["master_bursts"][1].burst_index, 1)
        self.assertEqual(len(call.kwargs["overlaps"]), 1)
        self.assertIsInstance(call.kwargs["overlaps"][0], strip_insar2.TopsOverlapInfo)
        self.assertEqual(call.kwargs["overlaps"][0].previous_burst_index, 0)
        self.assertEqual(call.kwargs["overlaps"][0].next_burst_index, 1)

    def test_process_strip_insar2_supports_stop_after_p0(self) -> None:
        with mock.patch.dict(sys.modules, {"h5py": types.ModuleType("h5py")}):
            import strip_insar2

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_context = strip_insar2.PairContext(
                master_manifest_path=root / "master.json",
                slave_manifest_path=root / "slave.json",
                master_manifest={"slc": {"path": str(root / "master.slc")}},
                slave_manifest={"slc": {"path": str(root / "slave.slc")}},
                master_orbit_data={},
                slave_orbit_data={},
                master_acq_data={"centerFrequency": 1.0},
                slave_acq_data={"centerFrequency": 1.0},
                master_rg_data={},
                slave_rg_data={},
                master_dop_data={},
                slave_dop_data={},
                output_root=root,
                pair_name="pair",
                pair_dir=root / "pair",
                output_paths={},
                resolved_dem=str(root / "dem.tif"),
                orbit_interp="Hermite",
                wavelength=0.24,
            )

            def fake_prepare(**kwargs):
                return {
                    "prepared_master_manifest": str(root / "prepared_master.json"),
                    "prepared_slave_manifest": str(root / "prepared_slave.json"),
                    "prepared_dem": str(root / "dem.tif"),
                    "effective_master_window": {"row0": 0, "col0": 0, "rows": 2, "cols": 2},
                }

            with mock.patch.object(strip_insar2, "_derive_pair_identity", return_value=("pair", root / "pair")), \
                mock.patch.object(strip_insar2, "_prepare_runtime_inputs", side_effect=fake_prepare), \
                mock.patch.object(strip_insar2, "load_pair_context", return_value=fake_context), \
                mock.patch.object(strip_insar2, "resolve_manifest_data_path", side_effect=lambda manifest, path: str(path)), \
                mock.patch.object(strip_insar2, "run_geo2rdr_stage", return_value=({"master_topo": "m", "slave_topo": "s"}, "cpu", None)) as geo2rdr, \
                mock.patch.object(strip_insar2, "run_resample_stage") as resample, \
                mock.patch.object(strip_insar2, "run_crossmul_stage") as crossmul:
                result = strip_insar2.process_strip_insar2(
                    root / "master.json",
                    root / "slave.json",
                    output_root=root,
                    gpu_mode="cpu",
                    tops_mode=True,
                    master_bursts=[],
                    slave_bursts=[],
                    overlaps=[],
                    stop_after_stage="p0",
                )

        self.assertEqual(result["stopped_after_stage"], "p0")
        self.assertEqual(result["stage_backends"], {"geo2rdr": "cpu"})
        geo2rdr.assert_called_once()
        resample.assert_not_called()
        crossmul.assert_not_called()

    def test_burst_merge_refreshes_filtered_interferogram_before_unwrap(self) -> None:
        with mock.patch.dict(sys.modules, {"h5py": types.ModuleType("h5py")}):
            import strip_insar2

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_context = strip_insar2.PairContext(
                master_manifest_path=root / "master.json",
                slave_manifest_path=root / "slave.json",
                master_manifest={"slc": {"path": str(root / "master.slc")}},
                slave_manifest={"slc": {"path": str(root / "slave.slc")}},
                master_orbit_data={},
                slave_orbit_data={},
                master_acq_data={"centerFrequency": 1.0},
                slave_acq_data={"centerFrequency": 1.0},
                master_rg_data={"numberOfRows": 4, "numberOfColumns": 4},
                slave_rg_data={"numberOfRows": 4, "numberOfColumns": 4},
                master_dop_data={},
                slave_dop_data={},
                output_root=root,
                pair_name="pair",
                pair_dir=root / "pair",
                output_paths={},
                resolved_dem=str(root / "dem.tif"),
                orbit_interp="Hermite",
                wavelength=0.24,
            )
            fake_context.pair_dir.mkdir(parents=True, exist_ok=True)

            interferogram = np.exp(1j * np.ones((4, 4), dtype=np.float32)).astype(np.complex64)
            filtered_interferogram = np.exp(1j * np.full((4, 4), 2.0, dtype=np.float32)).astype(np.complex64)
            coherence = np.full((4, 4), 0.8, dtype=np.float32)
            ifg_path = strip_insar2._save_stage_array(fake_context.pair_dir, "p2", "interferogram", interferogram)
            filtered_path = strip_insar2._save_stage_array(fake_context.pair_dir, "p2", "filtered_interferogram", filtered_interferogram)
            coh_path = strip_insar2._save_stage_array(fake_context.pair_dir, "p2", "coherence", coherence)
            strip_insar2._write_stage_outputs_record(
                output_dir=fake_context.pair_dir,
                stage="p2",
                master_manifest_path=fake_context.master_manifest_path,
                slave_manifest_path=fake_context.slave_manifest_path,
                backend_used="cpu",
                output_files={
                    "interferogram": ifg_path,
                    "filtered_interferogram": filtered_path,
                    "coherence": coh_path,
                },
                processing_options={"range_looks": 1, "azimuth_looks": 1},
            )

            bursts = [
                strip_insar2.TopsBurstInfo(
                    burst_index=0,
                    line_offset=0,
                    number_of_lines=2,
                    number_of_samples=4,
                    first_valid_line=0,
                    num_valid_lines=2,
                    first_valid_sample=0,
                    num_valid_samples=4,
                ),
                strip_insar2.TopsBurstInfo(
                    burst_index=1,
                    line_offset=2,
                    number_of_lines=2,
                    number_of_samples=4,
                    first_valid_line=0,
                    num_valid_lines=2,
                    first_valid_sample=0,
                    num_valid_samples=4,
                ),
            ]
            merged_interferogram = np.exp(1j * np.full((4, 4), 3.0, dtype=np.float32)).astype(np.complex64)
            merged_coherence = np.full((4, 4), 0.5, dtype=np.float32)

            with mock.patch.object(strip_insar2, "_merge_tops_burst_interferograms", return_value=(merged_interferogram, merged_coherence)):
                merge_result, backend, fallback = strip_insar2.run_burst_merge_stage(
                    fake_context,
                    master_bursts=bursts,
                    slave_bursts=bursts,
                    overlap_pairs=[{"previous_burst_index": 0, "next_burst_index": 1, "estimated_overlap_lines": 1}],
                    use_topo_flattening=False,
                    do_burst_seam_repair=False,
                )

            captured: dict[str, np.ndarray] = {}

            def fake_unwrap(interferogram_in: np.ndarray, coherence_in: np.ndarray, scratch_dir: Path):
                captured["interferogram"] = np.asarray(interferogram_in).copy()
                captured["coherence"] = np.asarray(coherence_in).copy()
                return np.zeros(interferogram_in.shape, dtype=np.float32), None

            with mock.patch.object(strip_insar2, "_unwrap_with_icu_profiles", side_effect=fake_unwrap):
                unwrap_result, unwrap_backend, unwrap_fallback = strip_insar2.run_unwrap_stage(
                    fake_context,
                    unwrap_method="icu",
                    block_rows=1,
                    range_looks=1,
                    azimuth_looks=1,
                    use_dolphin_unwrap=False,
                )
            self.assertEqual(backend, "cpu")
            self.assertIsNone(fallback)
            self.assertTrue(Path(merge_result["merged_interferogram"]).is_file())
            self.assertTrue(Path(merge_result["merged_coherence"]).is_file())
            np.testing.assert_allclose(np.load(merge_result["merged_interferogram"]), merged_interferogram)
            np.testing.assert_allclose(np.load(merge_result["merged_coherence"]), merged_coherence)
            self.assertIn("interferogram", captured)
            np.testing.assert_allclose(captured["interferogram"], merged_interferogram)
            np.testing.assert_allclose(captured["coherence"], merged_coherence)
            self.assertEqual(unwrap_backend, "cpu")
            self.assertIsNone(unwrap_fallback)
            self.assertTrue(Path(unwrap_result["unwrapped_phase"]).is_file())

    def test_strip_insar2_tops_mode_skips_esd_for_single_burst(self) -> None:
        with mock.patch.dict(sys.modules, {"h5py": types.ModuleType("h5py")}):
            import strip_insar2

        burst = strip_insar2.TopsBurstInfo(
            burst_index=0,
            line_offset=0,
            number_of_lines=2,
            number_of_samples=2,
            first_valid_line=0,
            num_valid_lines=2,
            first_valid_sample=0,
            num_valid_samples=2,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def fake_prepare(**kwargs):
                return {
                    "prepared_master_manifest": str(root / "master.json"),
                    "prepared_slave_manifest": str(root / "slave.json"),
                    "prepared_dem": str(root / "dem.tif"),
                    "effective_master_window": {"row0": 0, "col0": 0, "rows": 2, "cols": 2},
                }

            fake_context = strip_insar2.PairContext(
                master_manifest_path=root / "master.json",
                slave_manifest_path=root / "slave.json",
                master_manifest={"slc": {"path": str(root / "m.slc")}},
                slave_manifest={"slc": {"path": str(root / "s.slc")}},
                master_orbit_data={},
                slave_orbit_data={},
                master_acq_data={"centerFrequency": 1.0},
                slave_acq_data={},
                master_rg_data={},
                slave_rg_data={},
                master_dop_data={},
                slave_dop_data={},
                output_root=root,
                pair_name="pair",
                pair_dir=root / "pair",
                output_paths={},
                resolved_dem=str(root / "dem.tif"),
                orbit_interp="Hermite",
                wavelength=0.24,
            )

            with mock.patch.object(strip_insar2, "_derive_pair_identity", return_value=("pair", root / "pair")), \
                mock.patch.object(strip_insar2, "_prepare_runtime_inputs", side_effect=fake_prepare), \
                mock.patch.object(strip_insar2, "load_pair_context", return_value=fake_context), \
                mock.patch.object(strip_insar2, "resolve_manifest_data_path", side_effect=lambda manifest, path: str(path)), \
                mock.patch.object(strip_insar2, "run_geo2rdr_stage", return_value=({}, "cpu", None)), \
                mock.patch.object(strip_insar2, "run_resample_stage", return_value=({"fine_coreg_slave": str(root / "fine.slc")}, "cpu", None)), \
                mock.patch.object(strip_insar2, "run_crossmul_stage", return_value=({}, "cpu", None)), \
                mock.patch.object(strip_insar2, "run_esd_estimation_stage") as esd, \
                mock.patch.object(strip_insar2, "run_burst_merge_stage", return_value=({}, "cpu", None)), \
                mock.patch.object(strip_insar2, "run_unwrap_stage", return_value=({}, "cpu", None)):
                result = strip_insar2.process_strip_insar2(
                    root / "master.json",
                    root / "slave.json",
                    output_root=root,
                    gpu_mode="cpu",
                    tops_mode=True,
                    master_bursts=[burst],
                    slave_bursts=[burst],
                    overlaps=[],
                    stop_after_stage="p3",
                )

        esd.assert_not_called()
        self.assertEqual(result["stopped_after_stage"], "p3")
        self.assertEqual(result["stage_backends"]["burst_merge"], "cpu")


if __name__ == "__main__":
    unittest.main()
