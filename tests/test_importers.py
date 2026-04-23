import sys
import tarfile
import tempfile
import unittest
import zipfile
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from tianyi_importer import TianyiImporter
from lutan_importer import LutanImporter
from common_processing import resolve_manifest_data_path, resolve_manifest_metadata_path


class TianyiTarTests(unittest.TestCase):
    def _make_tar(self, suffix: str, members: dict[str, str]) -> Path:
        fd, path = tempfile.mkstemp(suffix=suffix)
        import os

        os.close(fd)
        with tarfile.open(path, "w") as tf:
            for name, content in members.items():
                data = content.encode() if isinstance(content, str) else content
                import io

                tf.addfile(tarfile.TarInfo(name), io.BytesIO(data))
        return Path(path)

    def _make_zip(self, members: dict[str, str]) -> Path:
        fd, path = tempfile.mkstemp(suffix=".zip")
        import os

        os.close(fd)
        with zipfile.ZipFile(path, "w") as zf:
            for name, content in members.items():
                zf.writestr(name, content)
        return Path(path)

    def test_tianyi_tar_detects_tar(self):
        path = self._make_tar(".tar", {"x": ""})
        try:
            imp = TianyiImporter(str(path))
            self.assertTrue(imp.is_tar)
            self.assertFalse(imp.is_zip)
        finally:
            path.unlink()

    def test_tianyi_tar_detects_tar_gz(self):
        path = self._make_tar(".tar.gz", {"x": ""})
        try:
            imp = TianyiImporter(str(path))
            self.assertTrue(imp.is_tar)
        finally:
            path.unlink()

    def test_tianyi_tar_detects_tgz(self):
        path = self._make_tar(".tgz", {"x": ""})
        try:
            imp = TianyiImporter(str(path))
            self.assertTrue(imp.is_tar)
        finally:
            path.unlink()

    def test_tianyi_zip_still_works(self):
        path = self._make_zip({"x": ""})
        try:
            imp = TianyiImporter(str(path))
            self.assertTrue(imp.is_zip)
            self.assertFalse(imp.is_tar)
        finally:
            path.unlink()

    def test_tianyi_discover_tar_members(self):
        members = {
            "product/annotation/xml.xml": "<root/>",
            "product/annotation/calibration/cal.xml": "<root/>",
            "product/manifest.safe": "<root/>",
            "product/measurement/slc.tiff": "",
        }
        path = self._make_tar(".tar", members)
        try:
            imp = TianyiImporter(str(path))
            found = imp._discover_tar_members()
            self.assertEqual(found["annotation"], "product/annotation/xml.xml")
            self.assertEqual(
                found["calibration"], "product/annotation/calibration/cal.xml"
            )
            self.assertEqual(found["manifest"], "product/manifest.safe")
            self.assertEqual(found["tiff"], "product/measurement/slc.tiff")
        finally:
            path.unlink()

    def test_tianyi_build_slc_path_is_tar(self):
        path = self._make_tar(".tar", {"x": ""})
        try:
            imp = TianyiImporter(str(path))
            result = imp.build_slc_path("measurement/slc.tiff")
            self.assertEqual(result, f"/vsitar/{path}/measurement/slc.tiff")
        finally:
            path.unlink()


class LutanTarTests(unittest.TestCase):
    def _make_tar(self, suffix: str, members: dict[str, str]) -> Path:
        fd, path = tempfile.mkstemp(suffix=suffix)
        import os

        os.close(fd)
        with tarfile.open(path, "w") as tf:
            for name, content in members.items():
                data = content.encode() if isinstance(content, str) else content
                import io

                tf.addfile(tarfile.TarInfo(name), io.BytesIO(data))
        return Path(path)

    def _make_zip(self, members: dict[str, str]) -> Path:
        fd, path = tempfile.mkstemp(suffix=".zip")
        import os

        os.close(fd)
        with zipfile.ZipFile(path, "w") as zf:
            for name, content in members.items():
                zf.writestr(name, content)
        return Path(path)

    def test_lutan_tar_detects_tar(self):
        path = self._make_tar(".tar", {"x": ""})
        try:
            imp = LutanImporter(str(path))
            self.assertTrue(imp.is_tar)
            self.assertFalse(imp.is_zip)
        finally:
            path.unlink()

    def test_lutan_tar_detects_tar_gz(self):
        path = self._make_tar(".tar.gz", {"x": ""})
        try:
            imp = LutanImporter(str(path))
            self.assertTrue(imp.is_tar)
        finally:
            path.unlink()

    def test_lutan_tar_detects_tgz(self):
        path = self._make_tar(".tgz", {"x": ""})
        try:
            imp = LutanImporter(str(path))
            self.assertTrue(imp.is_tar)
        finally:
            path.unlink()

    def test_lutan_zip_still_works(self):
        path = self._make_zip({"x": ""})
        try:
            imp = LutanImporter(str(path))
            self.assertTrue(imp.is_zip)
            self.assertFalse(imp.is_tar)
        finally:
            path.unlink()

    def test_lutan_discover_tar_members(self):
        members = {
            "SLC.TIFF": "",
            "scene.meta.xml": "<root/>",
            "scene.incidence.xml": "<root/>",
            "scene.rpc": "",
        }
        path = self._make_tar(".tar", members)
        try:
            imp = LutanImporter(str(path))
            found = imp._discover_tar_members()
            self.assertEqual(found["tiff"], "SLC.TIFF")
            self.assertEqual(found["meta_xml"], "scene.meta.xml")
            self.assertEqual(found["incidence_xml"], "scene.incidence.xml")
            self.assertEqual(found["rpc"], "scene.rpc")
        finally:
            path.unlink()

    def test_lutan_build_slc_path_is_tar(self):
        path = self._make_tar(".tar", {"x": ""})
        try:
            imp = LutanImporter(str(path))
            result = imp.build_slc_path("SLC.TIFF")
            self.assertEqual(result, f"/vsitar/{path}/SLC.TIFF")
        finally:
            path.unlink()


class CommonProcessingTarTests(unittest.TestCase):
    def test_resolve_manifest_data_path_tar(self):
        fd, manifest_path = tempfile.mkstemp(suffix=".json")
        import os

        os.close(fd)
        manifest_path = Path(manifest_path)
        fd2, archive_path = tempfile.mkstemp(suffix=".tar")
        os.close(fd2)
        archive_path = Path(archive_path)

        entry = {
            "path": str(archive_path),
            "storage": "tar",
            "member": "product/measurement/slc.tiff",
        }
        result = resolve_manifest_data_path(manifest_path, entry)
        self.assertEqual(result, f"/vsitar/{archive_path}/product/measurement/slc.tiff")

        manifest_path.unlink()
        archive_path.unlink()

    def test_resolve_manifest_data_path_vsitar_prefix(self):
        fd, manifest_path = tempfile.mkstemp(suffix=".json")
        import os

        os.close(fd)
        manifest_path = Path(manifest_path)
        result = resolve_manifest_data_path(
            manifest_path, "/vsitar/some.tar/member.tif"
        )
        self.assertEqual(result, "/vsitar/some.tar/member.tif")
        manifest_path.unlink()

    def test_resolve_manifest_data_path_maps_legacy_results_absolute_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            manifest_dir = project_root / "results" / "20231110"
            manifest_dir.mkdir(parents=True, exist_ok=True)
            target = project_root / "results" / "20231121" / "metadata" / "orbit.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("{}", encoding="utf-8")
            manifest_path = manifest_dir / "manifest.json"
            manifest_path.write_text("{}", encoding="utf-8")

            cwd = Path.cwd()
            try:
                os.chdir(project_root)
                resolved = resolve_manifest_data_path(
                    manifest_path,
                    "/results/20231121/metadata/orbit.json",
                )
            finally:
                os.chdir(cwd)

            self.assertEqual(Path(resolved), target.resolve())

    def test_resolve_manifest_metadata_path_maps_legacy_results_absolute_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            manifest_dir = project_root / "results" / "20231110"
            manifest_dir.mkdir(parents=True, exist_ok=True)
            target = project_root / "results" / "20231121" / "metadata" / "orbit.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text('{"header": {}, "stateVectors": []}', encoding="utf-8")
            manifest_path = manifest_dir / "manifest.json"
            manifest = {"metadata": {"orbit": "/results/20231121/metadata/orbit.json"}}
            manifest_path.write_text("{}", encoding="utf-8")

            cwd = Path.cwd()
            try:
                os.chdir(project_root)
                resolved = resolve_manifest_metadata_path(
                    manifest_path,
                    manifest,
                    "orbit",
                )
            finally:
                os.chdir(cwd)

            self.assertEqual(resolved, target.resolve())


if __name__ == "__main__":
    unittest.main()
