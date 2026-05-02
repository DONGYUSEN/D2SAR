import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


class StripInsar2OrbitInterpTests(unittest.TestCase):
    def test_lutan_pair_forces_legendre_orbit_interpolation(self):
        import strip_insar2

        with mock.patch("strip_insar2.choose_orbit_interp", return_value="Hermite"):
            interp = strip_insar2._choose_context_orbit_interp(
                {"sensor": "lutan"},
                {"sensor": "lutan"},
                {"stateVectors": [{} for _ in range(10)]},
                {"source": "lutan"},
            )

        self.assertEqual(interp, "Legendre")

    def test_non_lutan_pair_uses_default_orbit_interpolation_choice(self):
        import strip_insar2

        with mock.patch("strip_insar2.choose_orbit_interp", return_value="Hermite"):
            interp = strip_insar2._choose_context_orbit_interp(
                {"sensor": "tianyi"},
                {"sensor": "tianyi"},
                {"stateVectors": [{} for _ in range(10)]},
                {"source": "tianyi"},
            )

        self.assertEqual(interp, "Hermite")


if __name__ == "__main__":
    unittest.main()
