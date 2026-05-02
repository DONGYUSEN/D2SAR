from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = PROJECT_ROOT / "docker" / "Dockerfile"


def test_dockerfile_sets_writable_matplotlib_config_dir() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")

    assert "MPLCONFIGDIR=/tmp/matplotlib" in text
    assert "XDG_CACHE_HOME=/tmp/.cache" in text
    assert "HOME=/tmp" in text
    assert "mkdir -p /tmp/matplotlib /tmp/.cache" in text


def test_dockerfile_installs_rich_for_dolphin_imports() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")

    assert "\n        rich \\" in text


def test_dockerfile_installs_setuptools_scm_for_dolphin_build_metadata() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")

    assert "\n        setuptools_scm \\" in text


def test_dockerfile_installs_dolphin_with_pip_instead_of_copying_package() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")

    assert "pip install /opt/dolphin_src --prefix=/opt/isce3 --no-build-isolation" in text
    assert "cp -r /opt/dolphin_src/src/dolphin" not in text


def test_dockerfile_verifies_dolphin_import_after_install() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")

    assert 'python3 -c "import dolphin; print(' in text


def test_dockerfile_installs_plant_isce3_with_pip_instead_of_copying_package() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")

    assert "pip install /opt/build/plant-isce3/src --prefix=/opt/isce3 --no-build-isolation" in text
    assert "cp -r src/src/plant_isce3" not in text
