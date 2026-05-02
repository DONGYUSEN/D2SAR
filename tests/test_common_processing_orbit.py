import sys
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


class CommonProcessingOrbitTests(unittest.TestCase):
    def test_construct_orbit_accepts_sentinel_nested_position_velocity(self) -> None:
        import common_processing

        orbit_json = {
            "header": {"firstStateTimeUTC": "2023-06-25T11:40:45.000000"},
            "stateVectors": [
                {
                    "timeUTC": "2023-06-25T11:40:45.000000",
                    "position": {"x": 1.0, "y": 2.0, "z": 3.0},
                    "velocity": {"x": 4.0, "y": 5.0, "z": 6.0},
                }
            ],
        }

        state_vectors = []

        class FakeDateTime:
            def __init__(self, value):
                self.value = value

        class FakeStateVector:
            def __init__(self, dt, pos, vel):
                self.dt = dt
                self.pos = pos
                self.vel = vel
                state_vectors.append(self)

        class FakeOrbit:
            def __init__(self, svs, ref_dt, method):
                self.svs = svs
                self.ref_dt = ref_dt
                self.method = method

        fake_core = mock.Mock()
        fake_core.DateTime = FakeDateTime
        fake_core.StateVector = FakeStateVector
        fake_core.Orbit = FakeOrbit
        fake_core.OrbitInterpMethod.HERMITE = "Hermite"
        fake_core.OrbitInterpMethod.LEGENDRE = "Legendre"

        with mock.patch.dict(sys.modules, {"isce3.core": fake_core, "isce3": mock.Mock(core=fake_core)}):
            orbit = common_processing.construct_orbit(orbit_json, "Legendre")

        self.assertEqual(orbit.method, "Legendre")
        self.assertEqual(state_vectors[0].pos.tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(state_vectors[0].vel.tolist(), [4.0, 5.0, 6.0])

    def test_construct_radar_grid_prefers_burst_sensing_start_gps_time(self) -> None:
        import common_processing

        captured = {}

        class FakeDateTime:
            def __init__(self, value):
                self.value = value

        class FakeRadarGridParameters:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        fake_core = mock.Mock()
        fake_core.speed_of_light = 299792458.0
        fake_core.LookSide.Left = "left"
        fake_core.LookSide.Right = "right"
        fake_core.DateTime = FakeDateTime
        fake_product = mock.Mock()
        fake_product.RadarGridParameters = FakeRadarGridParameters
        fake_isce3 = mock.Mock(core=fake_core, product=fake_product)

        radargrid_json = {
            "sensingStartGPSTime": 123.0,
            "rangeTimeFirstPixel": 0.004,
            "columnSpacing": 10.0,
            "numberOfRows": 4,
            "numberOfColumns": 5,
        }
        acquisition_json = {
            "startGPSTime": 100.0,
            "centerFrequency": 5.405e9,
            "prf": 486.0,
            "lookDirection": "RIGHT",
        }
        orbit_json = {"header": {"firstStateTimeUTC": "1980-01-06T00:00:00"}}

        with mock.patch.dict(
            sys.modules,
            {"isce3": fake_isce3, "isce3.core": fake_core, "isce3.product": fake_product},
        ):
            common_processing.construct_radar_grid(radargrid_json, acquisition_json, orbit_json)

        self.assertEqual(captured["sensing_start"], 123.0)

    def test_construct_radar_grid_prefers_row_spacing_for_prf(self) -> None:
        import common_processing

        captured = {}

        class FakeDateTime:
            def __init__(self, value):
                self.value = value

        class FakeRadarGridParameters:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        fake_core = mock.Mock()
        fake_core.speed_of_light = 299792458.0
        fake_core.LookSide.Left = "left"
        fake_core.LookSide.Right = "right"
        fake_core.DateTime = FakeDateTime
        fake_product = mock.Mock()
        fake_product.RadarGridParameters = FakeRadarGridParameters
        fake_isce3 = mock.Mock(core=fake_core, product=fake_product)

        radargrid_json = {
            "sensingStartGPSTime": 123.0,
            "rangeTimeFirstPixel": 0.004,
            "rowSpacing": 0.002,
            "columnSpacing": 10.0,
            "numberOfRows": 4,
            "numberOfColumns": 5,
        }
        acquisition_json = {
            "startGPSTime": 100.0,
            "centerFrequency": 5.405e9,
            "prf": 1717.0,
            "lookDirection": "RIGHT",
        }
        orbit_json = {"header": {"firstStateTimeUTC": "1980-01-06T00:00:00"}}

        with mock.patch.dict(
            sys.modules,
            {"isce3": fake_isce3, "isce3.core": fake_core, "isce3.product": fake_product},
        ):
            common_processing.construct_radar_grid(radargrid_json, acquisition_json, orbit_json)

        self.assertAlmostEqual(captured["prf"], 500.0)

    def test_construct_doppler_lut2d_prefers_burst_timing(self) -> None:
        import common_processing

        captured = {}

        class FakeLUT2d:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        fake_core = mock.Mock()
        fake_core.speed_of_light = 299792458.0
        fake_core.LUT2d = FakeLUT2d
        fake_isce3 = mock.Mock(core=fake_core)

        doppler_json = {
            "combinedDoppler": {
                "polynomialDegree": 0,
                "coefficients": [0.0],
                "referencePoint": 0.004,
            }
        }
        radargrid_json = {
            "sensingStartGPSTime": 123.0,
            "rangeTimeFirstPixel": 0.004,
            "rowSpacing": 0.002,
            "columnSpacing": 10.0,
            "numberOfRows": 4,
            "numberOfColumns": 5,
        }
        acquisition_json = {
            "startGPSTime": 100.0,
            "prf": 1717.0,
        }
        orbit_json = {"header": {"firstStateTimeUTC": "1980-01-06T00:00:00"}}

        with mock.patch.dict(sys.modules, {"isce3": fake_isce3, "isce3.core": fake_core}):
            common_processing.construct_doppler_lut2d(
                doppler_json,
                radargrid_json=radargrid_json,
                acquisition_json=acquisition_json,
                orbit_json=orbit_json,
            )

        self.assertEqual(captured["ystart"], 123.0)
        self.assertAlmostEqual(captured["dy"], 4.0 / 500.0)


if __name__ == "__main__":
    unittest.main()
