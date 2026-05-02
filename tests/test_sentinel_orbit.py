import json
import sys
import tempfile
import unittest
import zipfile
from datetime import datetime
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from test_sentinel_importer import SentinelImporterTests


EOF_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Earth_Explorer_File>
  <Data_Block>
    <List_of_OSVs>
      <OSV><UTC>UTC=2023-11-10T04:39:47.500000</UTC><X>100</X><Y>200</Y><Z>300</Z><VX>1</VX><VY>2</VY><VZ>3</VZ><Quality>NOMINAL</Quality></OSV>
      <OSV><UTC>UTC=2023-11-10T04:39:48.500000</UTC><X>400</X><Y>500</Y><Z>600</Z><VX>4</VX><VY>5</VY><VZ>6</VZ><Quality>NOMINAL</Quality></OSV>
    </List_of_OSVs>
  </Data_Block>
</Earth_Explorer_File>
"""


def orbit_name(kind: str) -> str:
    return f"S1A_OPER_AUX_{kind}_OPOD_20231111T000000_V20231110T030000_20231110T060000.EOF"


class SentinelOrbitTests(unittest.TestCase):
    def test_parse_product_filename_extracts_platform_and_sensing_times(self) -> None:
        from sentinel_orbit import parse_product_filename

        info = parse_product_filename(
            "S1A_IW_SLC__1SDV_20230625T114146_20230625T114213_049142_05E8CA_CCD3.zip"
        )

        self.assertEqual(info.platform, "S1A")
        self.assertEqual(info.start.isoformat(), "2023-06-25T11:41:46")
        self.assertEqual(info.stop.isoformat(), "2023-06-25T11:42:13")

    def test_resolve_prefers_precise_orbit_over_restituted_orbit(self) -> None:
        from sentinel_orbit import resolve_orbit_for_product

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            orbit_dir = root / "orbits"
            orbit_dir.mkdir()
            resorb = orbit_dir / orbit_name("RESORB")
            poeorb = orbit_dir / orbit_name("POEORB")
            resorb.write_text(EOF_XML, encoding="utf-8")
            poeorb.write_text(EOF_XML, encoding="utf-8")

            result = resolve_orbit_for_product(
                "S1A_IW_SLC__1SDV_20231110T043948_20231110T044000_TEST.zip",
                orbit_dir=orbit_dir,
                download=False,
            )

        self.assertEqual(result.orbit_type, "precise")
        self.assertEqual(Path(result.path).name, poeorb.name)
        self.assertEqual(result.source, "local")

    def test_resolve_materializes_zipped_eof_orbit(self) -> None:
        from sentinel_orbit import resolve_orbit_for_product

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            orbit_dir = root / "orbits"
            orbit_dir.mkdir()
            eof_name = orbit_name("POEORB")
            zip_path = orbit_dir / f"{eof_name}.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr(eof_name, EOF_XML)

            result = resolve_orbit_for_product(
                "S1A_IW_SLC__1SDV_20231110T043948_20231110T044000_TEST.zip",
                orbit_dir=orbit_dir,
                download=False,
            )

            materialized = Path(result.path)
            materialized_exists = materialized.is_file()

        self.assertEqual(materialized.name, eof_name)
        self.assertTrue(materialized_exists)
        self.assertEqual(result.orbit_type, "precise")

    def test_apply_updates_existing_manifest_orbit_metadata(self) -> None:
        from sentinel_importer import SentinelImporter
        from sentinel_orbit import apply_orbit_to_manifest

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            safe = SentinelImporterTests()._make_safe_dir(root)
            orbit_dir = root / "orbits"
            orbit_dir.mkdir()
            eof = orbit_dir / orbit_name("POEORB")
            eof.write_text(EOF_XML, encoding="utf-8")

            manifest_path = Path(SentinelImporter(str(safe)).run(str(root / "out"), download_dem=False))
            apply_orbit_to_manifest(manifest_path, orbit_dir=orbit_dir, download=False)

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            orbit = json.loads((manifest_path.parent / "metadata" / "orbit.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest["ancillary"]["orbitFile"], str(eof.resolve()))
        self.assertEqual(manifest["orbit"]["source"], "local")
        self.assertEqual(manifest["orbit"]["orbitType"], "precise")
        self.assertEqual(orbit["source"], "sentinel-1-eof")
        self.assertEqual(orbit["orbitFile"], str(eof.resolve()))

    def test_importer_auto_resolves_orbit_from_cache(self) -> None:
        from sentinel_importer import SentinelImporter

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            safe = SentinelImporterTests()._make_safe_dir(root)
            orbit_dir = root / "orbits"
            orbit_dir.mkdir()
            eof = orbit_dir / orbit_name("POEORB")
            eof.write_text(EOF_XML, encoding="utf-8")

            manifest_path = Path(
                SentinelImporter(str(safe), orbit_dir=orbit_dir).run(str(root / "out"), download_dem=False)
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            orbit = json.loads((manifest_path.parent / "metadata" / "orbit.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest["ancillary"]["orbitFile"], str(eof.resolve()))
        self.assertEqual(manifest["orbit"]["orbitType"], "precise")
        self.assertEqual(orbit["source"], "sentinel-1-eof")

    def test_download_orbit_uses_internal_downloader_not_fetchorbit_subprocess(self) -> None:
        import sentinel_orbit

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch("subprocess.run") as run_mock:
                with mock.patch("sentinel_orbit.download_orbit_file", side_effect=RuntimeError("offline")):
                    with self.assertRaises(RuntimeError):
                        sentinel_orbit._fetch_orbit(
                            "S1A_IW_SLC__1SDV_20231110T043948_20231110T044000_TEST.zip",
                            root,
                        )

        run_mock.assert_not_called()

    def test_internal_downloader_writes_eof_and_zip_cache_from_poeorb_url(self) -> None:
        from sentinel_orbit import ProductInfo, download_orbit_file

        eof_name = orbit_name("POEORB")

        def fake_urlopen(url, timeout=120):
            del timeout
            self.assertIn("/POEORB/S1A/2023/11/", url)

            class Response:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

                def read(self):
                    with tempfile.TemporaryDirectory() as tmp:
                        zip_path = Path(tmp) / "orbit.zip"
                        with zipfile.ZipFile(zip_path, "w") as zf:
                            zf.writestr(eof_name, EOF_XML)
                        return zip_path.read_bytes()

            return Response()

        with tempfile.TemporaryDirectory() as tmp:
            target_dir = Path(tmp)
            result = download_orbit_file(
                ProductInfo(
                    platform="S1A",
                    start=datetime(2023, 11, 10, 4, 39, 48),
                    stop=datetime(2023, 11, 10, 4, 40, 0),
                ),
                target_dir,
                urlopen=fake_urlopen,
            )
            eof = Path(result)

            self.assertEqual(eof.name, eof_name)
            self.assertTrue(eof.is_file())
            self.assertTrue(any(target_dir.glob("*.EOF.zip")))


if __name__ == "__main__":
    unittest.main()
