"""Docker integration test for tops_insar.py using a two-burst TOPS subset.

This test follows the TOPS InSAR design docs:
- docs/2026-05-02-tops-insar-design.md
- docs/2026-05-02-tops-insar-implementation-plan.md
- docs/tops_insar_isce2_alignment.md

It runs the real CLI in the d2sar:cuda image and limits processing to two
common burst pairs (burst 1 and 2 in user-facing terms, implemented via
--burst-limit 2 in tops_insar.py).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
TEMP_DIR = Path("/home/ysdong/Temp")
MASTER_SAFE = TEMP_DIR / "s1/raw/S1A_IW_SLC__1SDV_20230625T114146_20230625T114213_049142_05E8CA_CCD3.SAFE"
SLAVE_SAFE = TEMP_DIR / "s1/raw/S1A_IW_SLC__1SDV_20230719T114147_20230719T114214_049492_05F38A_3C77.SAFE"
DEM = TEMP_DIR / "dem/dem_clip_wgs84.tif"
OUTPUT_NAME = "tops_insar_burst12_integration"


def _docker_available() -> bool:
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", "d2sar:cuda"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except FileNotFoundError:
        return False
    return result.returncode == 0


def _build_container_cmd(end_stage: str) -> str:
    template = r'''
set -euo pipefail
cd /work
python3 scripts/tops_insar.py \
  /results/tops_insar_burst12_integration \
  /temp/s1/raw/S1A_IW_SLC__1SDV_20230625T114146_20230625T114213_049142_05E8CA_CCD3.SAFE \
  /temp/s1/raw/S1A_IW_SLC__1SDV_20230719T114147_20230719T114214_049492_05F38A_3C77.SAFE \
  --swath IW1 \
  --dem /temp/dem/dem_clip_wgs84.tif \
  --orbit-dir /temp/s1/proc/orbits \
  --gpu-mode cpu \
  --burst-limit 2 \
  --start-stage preprocess \
  --end-stage __END_STAGE__ \
  --log-level INFO
python3 - <<'PY'
from pathlib import Path

root = Path('/results/tops_insar_burst12_integration/IW1')
assert (root / 'range_coreg_summary.json').exists(), 'range coreg summary missing'
assert (root / 'esd_summary' / 'esd_summary.json').exists(), 'ESD summary missing'
assert (root / 'burst_000' / 'fine_resampled_sec.npz').exists(), 'fine resampled burst 0 missing'
assert (root / 'burst_001' / 'fine_resampled_sec.npz').exists(), 'fine resampled burst 1 missing'
print('tops_insar burst12 fine_resamp validation OK')
PY
'''
    return template.replace('__END_STAGE__', end_stage)



@pytest.mark.skipif(
    os.environ.get("D2SAR_RUN_DOCKER_INTEGRATION") != "1",
    reason="set D2SAR_RUN_DOCKER_INTEGRATION=1 to run the d2sar:cuda integration test",
)
def test_tops_insar_docker_processes_only_two_bursts_and_merges_in_rd_domain() -> None:
    """Run tops_insar.py on IW1 with --burst-limit 2 and validate merge output."""
    if not _docker_available():
        pytest.skip("docker image d2sar:cuda is not available")
    for required in (MASTER_SAFE, SLAVE_SAFE, DEM):
        if not required.exists():
            pytest.skip(f"required integration input is missing: {required}")

    output_dir = RESULTS_DIR / OUTPUT_NAME
    shutil.rmtree(output_dir, ignore_errors=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "-v",
            f"{PROJECT_ROOT}:/work",
            "-v",
            f"{RESULTS_DIR}:/results",
            "-v",
            f"{TEMP_DIR}:/temp",
            "d2sar:cuda",
            "bash",
            "-c",
            'PYTHONPATH="/opt/isce3/packages:/opt/gdal38/lib/python3/dist-packages:/work/scripts:/work" '
            + _build_container_cmd("merge_bursts"),
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=1800,
    )

    assert result.returncode == 0, (
        "tops_insar.py docker integration test failed\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}\n"
    )
    assert "tops_insar burst12 integration validation OK" in result.stdout


def test_tops_insar_docker_runs_fine_resamp_for_two_bursts() -> None:
    """Run tops_insar.py through fine_resamp for a two-burst subset."""
    if os.environ.get("D2SAR_RUN_DOCKER_INTEGRATION") != "1":
        pytest.skip("set D2SAR_RUN_DOCKER_INTEGRATION=1 to run the d2sar:cuda integration test")
    if not _docker_available():
        pytest.skip("docker image d2sar:cuda is not available")
    for required in (MASTER_SAFE, SLAVE_SAFE, DEM):
        if not required.exists():
            pytest.skip(f"required integration input is missing: {required}")

    output_dir = RESULTS_DIR / OUTPUT_NAME
    shutil.rmtree(output_dir, ignore_errors=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "-v",
            f"{PROJECT_ROOT}:/work",
            "-v",
            f"{RESULTS_DIR}:/results",
            "-v",
            f"{TEMP_DIR}:/temp",
            "d2sar:cuda",
            "bash",
            "-c",
            'PYTHONPATH="/opt/isce3/packages:/opt/gdal38/lib/python3/dist-packages:/work/scripts:/work" '
            + _build_container_cmd("fine_resamp"),
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=1800,
    )

    assert result.returncode == 0, (
        "tops_insar.py fine_resamp integration failed\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}\n"
    )
    root = RESULTS_DIR / OUTPUT_NAME / "IW1"
    assert (root / "range_coreg_summary.json").exists()
    assert (root / "esd_summary" / "esd_summary.json").exists()
    assert (root / "burst_000" / "fine_resampled_sec.npz").exists()
    assert (root / "burst_001" / "fine_resampled_sec.npz").exists()
    assert "tops_insar burst12 integration validation OK" in result.stdout
