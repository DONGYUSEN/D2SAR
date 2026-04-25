from pathlib import Path
import unittest


class DockerfileTest(unittest.TestCase):
    def test_main_dockerfile_installs_cupy_for_cuda12(self) -> None:
        dockerfile = Path(__file__).resolve().parents[1] / "docker" / "Dockerfile"
        content = dockerfile.read_text(encoding="utf-8")

        self.assertIn("cupy-cuda12x", content)
        self.assertIn("import cupy", content)


if __name__ == "__main__":
    unittest.main()
