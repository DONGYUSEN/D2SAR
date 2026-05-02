import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(PROJECT_ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from test_sentinel_importer import SentinelImporterTests, make_sentinel_safe_dir


class TopsGeometryTests(unittest.TestCase):
    def _make_manifest(self, root: Path) -> Path:
        from sentinel_importer import SentinelImporter

        safe = make_sentinel_safe_dir(root, SentinelImporterTests.MANIFEST_XML)
        return Path(SentinelImporter(str(safe)).run(str(root / "out"), download_dem=False))

    def test_iter_burst_radar_grids_derives_burst_geometry(self) -> None:
        from tops_geometry import iter_burst_radar_grids, load_tops_metadata, select_burst_doppler

        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = self._make_manifest(Path(tmp))
            metadata = load_tops_metadata(manifest_path)
            burst_grids = list(iter_burst_radar_grids(manifest_path))
            first_doppler = select_burst_doppler(metadata["tops"]["bursts"][0])

        self.assertEqual(len(burst_grids), 2)
        self.assertEqual(burst_grids[0]["source"], "sentinel-1-burst")
        self.assertEqual(burst_grids[0]["burstIndex"], 1)
        self.assertEqual(burst_grids[0]["numberOfRows"], 4)
        self.assertEqual(burst_grids[0]["numberOfColumns"], 4)
        self.assertEqual(burst_grids[0]["lineOffset"], 0)
        self.assertEqual(burst_grids[1]["lineOffset"], 4)
        self.assertEqual(burst_grids[0]["sensingStartUTC"], "2023-11-10T04:39:48.000000")
        self.assertAlmostEqual(burst_grids[0]["startingRange"], 674533.0305, places=3)
        self.assertEqual(burst_grids[0]["firstValidLine"], 0)
        self.assertEqual(burst_grids[0]["numValidLines"], 4)
        self.assertEqual(burst_grids[0]["firstValidSample"], 1)
        self.assertEqual(burst_grids[0]["numValidSamples"], 1)
        self.assertEqual(first_doppler["coefficients"], [1.0, 2.0, 3.0])

    def test_validate_burst_geometry_rejects_missing_orbit_coverage(self) -> None:
        from tops_geometry import iter_burst_radar_grids, load_tops_metadata, validate_burst_geometry

        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = self._make_manifest(Path(tmp))
            metadata = load_tops_metadata(manifest_path)
            metadata["orbit"]["stateVectors"] = []
            burst_grid = next(iter_burst_radar_grids(manifest_path))

        with self.assertRaisesRegex(ValueError, "orbit stateVectors"):
            validate_burst_geometry(
                metadata["acquisition"],
                metadata["orbit"],
                burst_grid,
                metadata["tops"]["bursts"][0],
            )


if __name__ == "__main__":
    unittest.main()
