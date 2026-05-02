# Sentinel TOPS Geometry Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first Sentinel TOPS per-burst geometry adapter from importer metadata.

**Architecture:** Add a focused `scripts/tops_geometry.py` module that loads Sentinel manifest metadata, derives per-burst radargrid dictionaries, selects per-burst Doppler, and validates geometry readiness. Keep ISCE3 execution out of this step; this module prepares verified inputs for later `tops_rtc.py` and `tops_insar.py`.

**Tech Stack:** Python stdlib JSON/path/datetime, existing `scripts/common_processing.py` path helpers, pytest/unittest-style tests.

---

### Task 1: Add Per-Burst Geometry Tests

**Files:**
- Modify: `tests/test_sentinel_importer.py`
- Create: `tests/test_tops_geometry.py`

- [ ] **Step 1: Export the synthetic SAFE helper from tests**

Modify `tests/test_sentinel_importer.py` so the existing synthetic fixture can be reused from another test file without instantiating a test case. Add this function after `ANNOTATION_XML`:

```python
def make_sentinel_safe_dir(root: Path, manifest_xml: str) -> Path:
    safe = root / "S1A_IW_SLC__1SDV_TEST.SAFE"
    annotation = safe / "annotation"
    calibration = annotation / "calibration"
    measurement = safe / "measurement"
    annotation.mkdir(parents=True)
    calibration.mkdir()
    measurement.mkdir()
    (safe / "manifest.safe").write_text(manifest_xml, encoding="utf-8")
    (annotation / "s1a-iw2-slc-vv-test.xml").write_text(ANNOTATION_XML, encoding="utf-8")
    rfi = annotation / "rfi"
    rfi.mkdir()
    (rfi / "rfi-s1a-iw2-slc-vv-test.xml").write_text("<rfi/>\n", encoding="utf-8")
    (annotation / "s1a-iw1-slc-vh-test.xml").write_text(
        ANNOTATION_XML.replace("<swath>IW2</swath>", "<swath>IW1</swath>").replace(
            "<polarisation>VV</polarisation>", "<polarisation>VH</polarisation>"
        ),
        encoding="utf-8",
    )
    (calibration / "calibration-s1a-iw2-slc-vv-test.xml").write_text("<calibration/>\n", encoding="utf-8")
    (calibration / "noise-s1a-iw2-slc-vv-test.xml").write_text("<noise/>\n", encoding="utf-8")
    (measurement / "s1a-iw2-slc-vv-test.tiff").write_bytes(b"")
    (measurement / "s1a-iw1-slc-vh-test.tiff").write_bytes(b"")
    return safe
```

Then change `_make_safe_dir()` to:

```python
def _make_safe_dir(self, root: Path) -> Path:
    return make_sentinel_safe_dir(root, self.MANIFEST_XML)
```

- [ ] **Step 2: Create failing tests for `tops_geometry`**

Create `tests/test_tops_geometry.py`:

```python
import json
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
```

- [ ] **Step 3: Run the tests and confirm failure**

Run: `python -m pytest tests/test_tops_geometry.py`

Expected: FAIL with `ModuleNotFoundError: No module named 'tops_geometry'` or missing functions.

### Task 2: Implement `tops_geometry.py`

**Files:**
- Create: `scripts/tops_geometry.py`
- Test: `tests/test_tops_geometry.py`

- [ ] **Step 1: Add the module implementation**

Create `scripts/tops_geometry.py`:

```python
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from common_processing import resolve_manifest_metadata_path


GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)


def load_tops_metadata(manifest_path: str | Path) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    with manifest_path.open(encoding="utf-8") as f:
        manifest = json.load(f)
    if str(manifest.get("sensor", "")).lower() != "sentinel-1":
        raise ValueError("tops_geometry requires a sentinel-1 manifest")
    metadata = {"manifest": manifest}
    for name in ("acquisition", "orbit", "radargrid", "doppler", "tops"):
        with open(resolve_manifest_metadata_path(manifest_path, manifest, name), encoding="utf-8") as f:
            metadata[name] = json.load(f)
    return metadata


def iter_burst_radar_grids(manifest_path: str | Path, *, validate: bool = True) -> Iterator[dict[str, Any]]:
    metadata = load_tops_metadata(manifest_path)
    for burst in metadata["tops"].get("bursts", []):
        burst_grid = build_burst_radar_grid_json(
            metadata["radargrid"], metadata["acquisition"], metadata["orbit"], burst
        )
        if validate:
            validate_burst_geometry(metadata["acquisition"], metadata["orbit"], burst_grid, burst)
        yield burst_grid


def build_burst_radar_grid_json(
    radargrid: dict[str, Any],
    acquisition: dict[str, Any],
    orbit: dict[str, Any],
    burst: dict[str, Any],
) -> dict[str, Any]:
    del orbit
    sensing_start = str(burst.get("sensingStartUTC") or burst.get("azimuthTimeUTC") or "")
    return {
        "source": "sentinel-1-burst",
        "burstIndex": int(burst.get("index", 0)),
        "numberOfRows": int(burst.get("numberOfLines", radargrid.get("numberOfRows", 0))),
        "numberOfColumns": int(burst.get("numberOfSamples", radargrid.get("numberOfColumns", 0))),
        "rowSpacing": float(burst.get("azimuthTimeInterval", radargrid.get("rowSpacing", 0.0))),
        "columnSpacing": float(burst.get("rangePixelSize", radargrid.get("columnSpacing", 0.0))),
        "rangeTimeFirstPixel": float(radargrid.get("rangeTimeFirstPixel", 0.0)),
        "startingRange": float(burst.get("startingRange", radargrid.get("startingRange", 0.0))),
        "prf": float(burst.get("prf", radargrid.get("prf", 0.0))),
        "wavelength": float(burst.get("radarWavelength", radargrid.get("wavelength", 0.0))),
        "sensingStartUTC": sensing_start,
        "sensingStopUTC": str(burst.get("sensingStopUTC") or ""),
        "sensingStartGPSTime": _parse_timestamp(sensing_start),
        "lineOffset": int(burst.get("lineOffset", 0)),
        "firstValidLine": int(burst.get("firstValidLine", 0)),
        "numValidLines": int(burst.get("numValidLines", 0)),
        "firstValidSample": int(burst.get("firstValidSample", 0)),
        "lastValidSample": int(burst.get("lastValidSample", -1)),
        "numValidSamples": int(burst.get("numValidSamples", 0)),
        "swath": acquisition.get("swath") or radargrid.get("swath"),
        "polarisation": acquisition.get("polarisation") or radargrid.get("polarisation"),
        "lookDirection": acquisition.get("lookDirection", "RIGHT"),
    }


def select_burst_doppler(burst: dict[str, Any]) -> dict[str, Any]:
    doppler = burst.get("doppler")
    if not isinstance(doppler, dict) or not doppler.get("coefficients"):
        raise ValueError(f"burst {burst.get('index')} has no Doppler polynomial")
    return doppler


def validate_burst_geometry(
    acquisition: dict[str, Any],
    orbit: dict[str, Any],
    burst_grid: dict[str, Any],
    burst: dict[str, Any],
    *,
    orbit_margin_seconds: float = 5.0,
) -> None:
    if int(burst_grid.get("numberOfRows", 0)) <= 0:
        raise ValueError("burst radar grid numberOfRows must be positive")
    if int(burst_grid.get("numberOfColumns", 0)) <= 0:
        raise ValueError("burst radar grid numberOfColumns must be positive")
    if float(burst_grid.get("prf", 0.0)) <= 0.0:
        raise ValueError("burst radar grid prf must be positive")
    if float(burst_grid.get("columnSpacing", 0.0)) <= 0.0:
        raise ValueError("burst radar grid columnSpacing must be positive")
    if float(burst_grid.get("wavelength", 0.0)) <= 0.0:
        raise ValueError("burst radar grid wavelength must be positive")

    state_vectors = orbit.get("stateVectors") or []
    if not state_vectors:
        raise ValueError("orbit stateVectors must not be empty")
    _validate_orbit_covers_burst(state_vectors, burst_grid, orbit_margin_seconds)
    _validate_valid_region(burst_grid)
    select_burst_doppler(burst)

    if float(acquisition.get("centerFrequency", 0.0)) <= 0.0:
        raise ValueError("acquisition centerFrequency must be positive")


def _validate_valid_region(burst_grid: dict[str, Any]) -> None:
    first_line = int(burst_grid.get("firstValidLine", 0))
    num_lines = int(burst_grid.get("numValidLines", 0))
    first_sample = int(burst_grid.get("firstValidSample", 0))
    num_samples = int(burst_grid.get("numValidSamples", 0))
    rows = int(burst_grid.get("numberOfRows", 0))
    cols = int(burst_grid.get("numberOfColumns", 0))
    if first_line < 0 or num_lines <= 0 or first_line + num_lines > rows:
        raise ValueError("burst valid line region is outside burst bounds")
    if first_sample < 0 or num_samples <= 0 or first_sample + num_samples > cols:
        raise ValueError("burst valid sample region is outside burst bounds")


def _validate_orbit_covers_burst(
    state_vectors: list[dict[str, Any]], burst_grid: dict[str, Any], margin_seconds: float
) -> None:
    orbit_times = [_parse_datetime(str(sv.get("timeUTC", ""))) for sv in state_vectors]
    orbit_times = [time for time in orbit_times if time is not None]
    if not orbit_times:
        raise ValueError("orbit stateVectors contain no valid timeUTC values")
    start = _parse_datetime(str(burst_grid.get("sensingStartUTC", "")))
    stop = _parse_datetime(str(burst_grid.get("sensingStopUTC", "")))
    if start is None or stop is None:
        raise ValueError("burst sensingStartUTC/sensingStopUTC must be valid UTC timestamps")
    coverage_start = min(orbit_times)
    coverage_stop = max(orbit_times)
    if (start - coverage_start).total_seconds() < -margin_seconds:
        raise ValueError("orbit does not cover burst sensing start")
    if (coverage_stop - stop).total_seconds() < -margin_seconds:
        raise ValueError("orbit does not cover burst sensing stop")


def _parse_timestamp(value: str) -> float:
    dt = _parse_datetime(value)
    if dt is None:
        return 0.0
    return (dt - GPS_EPOCH).total_seconds()


def _parse_datetime(value: str) -> datetime | None:
    if not value:
        return None
    text = value.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Sentinel TOPS per-burst geometry metadata")
    parser.add_argument("manifest", help="Path to Sentinel importer manifest.json")
    args = parser.parse_args()
    burst_grids = list(iter_burst_radar_grids(args.manifest))
    print(json.dumps({"burst_count": len(burst_grids), "bursts": burst_grids}, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run focused tests**

Run: `python -m pytest tests/test_tops_geometry.py`

Expected: PASS, `2 passed`.

### Task 3: Verify Regression And Real ZIP

**Files:**
- Modify: `findings.md`
- Modify: `progress.md`

- [ ] **Step 1: Run importer and geometry tests together**

Run: `python -m pytest tests/test_sentinel_importer.py tests/test_tops_geometry.py`

Expected: PASS.

- [ ] **Step 2: Run compilation check**

Run: `python -m py_compile scripts/sentinel_importer.py scripts/tops_geometry.py tests/test_sentinel_importer.py tests/test_tops_geometry.py`

Expected: no output and exit code 0.

- [ ] **Step 3: Re-import the real Sentinel ZIP**

Run: `rm -rf /tmp/d2sar_sentinel_import_test && python scripts/sentinel_importer.py '/home/ysdong/Temp/S1A_IW_SLC__1SDV_20230625T114146_20230625T114213_049142_05E8CA_CCD3.zip' /tmp/d2sar_sentinel_import_test`

Expected: `/tmp/d2sar_sentinel_import_test/manifest.json`.

- [ ] **Step 4: Validate real ZIP burst geometry**

Run: `python scripts/tops_geometry.py /tmp/d2sar_sentinel_import_test/manifest.json`

Expected: JSON output with `burst_count` equal to `9` and no exception.

- [ ] **Step 5: Update project notes**

Append to `progress.md`:

```markdown
- 已新增 `scripts/tops_geometry.py`，从 Sentinel importer manifest 构造并校验 per-burst radargrid metadata。
- 验证命令：`python -m pytest tests/test_sentinel_importer.py tests/test_tops_geometry.py` 通过。
- 验证命令：`python -m py_compile scripts/sentinel_importer.py scripts/tops_geometry.py tests/test_sentinel_importer.py tests/test_tops_geometry.py` 通过。
- 真实 ZIP 几何验证命令：`python scripts/tops_geometry.py /tmp/d2sar_sentinel_import_test/manifest.json` 通过，burst_count=`9`。
```

Append to `findings.md`:

```markdown
- 已新增 TOPS per-burst 几何适配层 `scripts/tops_geometry.py`，第一版只负责 metadata 读取、burst radargrid 派生、Doppler 选择和几何有效性校验，不执行 RTC/Topo/GeoCode。
```

---

## Self-Review

- Spec coverage: 本计划覆盖当前 immediate task：新增 per-burst geometry 适配层、测试、真实 ZIP 验证。
- Placeholder scan: 无 TBD/TODO/模糊步骤。
- Type consistency: 函数名在测试和实现中一致：`load_tops_metadata`、`iter_burst_radar_grids`、`build_burst_radar_grid_json`、`validate_burst_geometry`、`select_burst_doppler`。
