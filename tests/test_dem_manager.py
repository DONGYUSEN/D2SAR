import sys
import unittest
from unittest import mock
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


from dem_manager import (
    _bbox_snwe_to_tile_names,
    dem_covers_scene_corners,
    scene_bbox_from_corners,
)


class DemManagerTests(unittest.TestCase):
    class _FakeBand:
        def GetNoDataValue(self):
            return None

        def ReadAsArray(self, x, y, width, height):
            import numpy as np

            return np.array([[1.0]])

    class _FakeDataset:
        RasterXSize = 2
        RasterYSize = 3

        def GetRasterBand(self, idx):
            return DemManagerTests._FakeBand()

        def GetGeoTransform(self):
            return (179.0, 1.0, 0.0, 32.0, 0.0, -1.0)

    def test_scene_bbox_from_corners_keeps_normal_bbox_monotonic(self):
        bbox = scene_bbox_from_corners(
            [{"lat": 29.5, "lon": 94.5}, {"lat": 30.2, "lon": 95.2}],
            margin_deg=0.05,
        )
        self.assertEqual(bbox, [94.45, 95.25, 29.45, 30.25])

    def test_scene_bbox_from_corners_unwraps_antimeridian_scene(self):
        bbox = scene_bbox_from_corners(
            [{"lat": 30.0, "lon": 179.5}, {"lat": 31.0, "lon": -179.5}],
            margin_deg=0.05,
        )
        self.assertEqual(bbox, [179.45, 180.55, 29.95, 31.05])

    def test_scene_bbox_from_corners_rejects_broad_longitude_span(self):
        with self.assertRaisesRegex(ValueError, "unsupported"):
            scene_bbox_from_corners(
                [
                    {"lat": 0.0, "lon": -170.0},
                    {"lat": 1.0, "lon": -20.0},
                    {"lat": 1.5, "lon": 30.0},
                    {"lat": 2.0, "lon": 170.0},
                ],
                margin_deg=0.05,
            )

    def test_bbox_snwe_to_tile_names_handles_antimeridian_bbox(self):
        tiles = _bbox_snwe_to_tile_names(29.95, 31.05, 179.45, 180.55)
        self.assertEqual(
            tiles, ["N29E179", "N29W180", "N30E179", "N30W180", "N31E179", "N31W180"]
        )

    def test_dem_covers_scene_corners_accepts_antimeridian_points_on_unwrapped_dem(
        self,
    ):
        with mock.patch(
            "dem_manager._dem_bbox",
            return_value=(self._FakeDataset(), 179.0, 181.0, 29.0, 32.0),
        ):
            check = dem_covers_scene_corners(
                "dummy.tif",
                [{"lat": 30.0, "lon": 179.5}, {"lat": 31.0, "lon": -179.5}],
                margin_deg=0.05,
            )
        self.assertTrue(check["ok"])


if __name__ == "__main__":
    unittest.main()
