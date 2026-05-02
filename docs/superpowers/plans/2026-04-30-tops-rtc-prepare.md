# Sentinel TOPS RTC Prepare Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `tops_rtc.py` first-stage dry-run preparation that writes a per-burst RTC execution plan.

**Architecture:** `scripts/tops_rtc.py` reads a Sentinel importer manifest, reuses `tops_geometry.py` for validated per-burst radargrid metadata, resolves the SLC reference, and writes `tops_rtc_plan.json`. It does not run RTC, DEM, HDF5, geocoding, GPU selection, or burst merge.

**Tech Stack:** Python stdlib JSON/path/argparse, existing `common_processing.resolve_manifest_data_path`, existing `tops_geometry` helpers, pytest/unittest tests.

---

### Task 1: Add Failing Tests

**Files:**
- Create: `tests/test_tops_rtc.py`

- [ ] **Step 1: Create tests for the dry-run RTC plan**

Create `tests/test_tops_rtc.py`:

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


class TopsRtcTests(unittest.TestCase):
    def _make_manifest(self, root: Path) -> Path:
        from sentinel_importer import SentinelImporter

        safe = make_sentinel_safe_dir(root, SentinelImporterTests.MANIFEST_XML)
        return Path(SentinelImporter(str(safe)).run(str(root / "imported"), download_dem=False))

    def test_prepare_tops_rtc_writes_per_burst_plan(self) -> None:
        from tops_rtc import prepare_tops_rtc

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = self._make_manifest(root)

            result = prepare_tops_rtc(manifest_path, root / "rtc")
            plan_path = Path(result["plan_path"])
            plan = json.loads(plan_path.read_text(encoding="utf-8"))

        self.assertTrue(plan_path.is_file())
        self.assertEqual(plan["sensor"], "sentinel-1")
        self.assertEqual(plan["swath"], "IW2")
        self.assertEqual(plan["polarisation"], "VV")
        self.assertEqual(plan["burst_count"], 2)
        self.assertEqual(len(plan["bursts"]), 2)
        self.assertEqual(plan["bursts"][0]["burstIndex"], 1)
        self.assertEqual(plan["bursts"][0]["slcWindow"]["xoff"], 0)
        self.assertEqual(plan["bursts"][0]["slcWindow"]["yoff"], 0)
        self.assertEqual(plan["bursts"][0]["slcWindow"]["xsize"], 4)
        self.assertEqual(plan["bursts"][0]["slcWindow"]["ysize"], 4)
        self.assertEqual(plan["bursts"][1]["slcWindow"]["yoff"], 4)
        self.assertEqual(plan["bursts"][0]["slcWindow"]["validWindow"]["xoff"], 1)
        self.assertEqual(plan["bursts"][0]["slcWindow"]["validWindow"]["ysize"], 4)
        self.assertTrue(plan["bursts"][0]["outputs"]["amplitude_h5"].endswith("burst_001/amplitude_fullres.h5"))
        self.assertEqual(plan["bursts"][0]["doppler"]["coefficients"], [1.0, 2.0, 3.0])

    def test_prepare_tops_rtc_rejects_non_sentinel_manifest(self) -> None:
        from tops_rtc import prepare_tops_rtc

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps({"sensor": "tianyi"}), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "sentinel-1"):
                prepare_tops_rtc(manifest_path, root / "rtc")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `python -m pytest tests/test_tops_rtc.py`

Expected: FAIL with `ModuleNotFoundError: No module named 'tops_rtc'`.

### Task 2: Implement `tops_rtc.py`

**Files:**
- Create: `scripts/tops_rtc.py`
- Test: `tests/test_tops_rtc.py`

- [ ] **Step 1: Create the module**

Create `scripts/tops_rtc.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common_processing import resolve_manifest_data_path
from tops_geometry import iter_burst_radar_grids, load_tops_metadata, select_burst_doppler


def prepare_tops_rtc(manifest_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_tops_metadata(manifest_path)
    manifest = metadata["manifest"]
    if str(manifest.get("sensor", "")).lower() != "sentinel-1":
        raise ValueError("tops_rtc requires a sentinel-1 manifest")

    slc_path = resolve_manifest_data_path(manifest_path, manifest.get("slc", {}).get("path"))
    if not slc_path:
        raise ValueError("manifest.slc.path is required")

    bursts = []
    burst_metadata = metadata["tops"].get("bursts", [])
    burst_grids = list(iter_burst_radar_grids(manifest_path))
    if len(burst_grids) != len(burst_metadata):
        raise ValueError("burst radar grid count does not match TOPS burst metadata count")

    for burst_grid, burst in zip(burst_grids, burst_metadata):
        burst_index = int(burst_grid["burstIndex"])
        bursts.append(
            {
                "burstIndex": burst_index,
                "radargrid": burst_grid,
                "doppler": select_burst_doppler(burst),
                "slcWindow": _build_slc_window(burst_grid),
                "outputs": _build_burst_outputs(output_dir, burst_index),
            }
        )

    plan = {
        "version": "1.0",
        "mode": "prepare-only",
        "sensor": manifest.get("sensor"),
        "swath": metadata["tops"].get("swath"),
        "polarisation": metadata["tops"].get("polarisation"),
        "burst_count": len(bursts),
        "input_manifest": str(manifest_path),
        "slc_path": slc_path,
        "bursts": bursts,
    }
    plan_path = output_dir / "tops_rtc_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"plan_path": str(plan_path), "burst_count": len(bursts)}


def _build_slc_window(burst_grid: dict[str, Any]) -> dict[str, Any]:
    return {
        "xoff": 0,
        "yoff": int(burst_grid["lineOffset"]),
        "xsize": int(burst_grid["numberOfColumns"]),
        "ysize": int(burst_grid["numberOfRows"]),
        "validWindow": {
            "xoff": int(burst_grid["firstValidSample"]),
            "yoff": int(burst_grid["firstValidLine"]),
            "xsize": int(burst_grid["numValidSamples"]),
            "ysize": int(burst_grid["numValidLines"]),
        },
    }


def _build_burst_outputs(output_dir: Path, burst_index: int) -> dict[str, str]:
    burst_dir = output_dir / f"burst_{burst_index:03d}"
    return {
        "directory": str(burst_dir),
        "amplitude_h5": str(burst_dir / "amplitude_fullres.h5"),
        "rtc_factor_tif": str(burst_dir / "rtc_factor.tif"),
        "topo_h5": str(burst_dir / "topo.h5"),
        "metadata_json": str(burst_dir / "metadata.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a Sentinel TOPS per-burst RTC execution plan")
    parser.add_argument("manifest", help="Path to Sentinel importer manifest.json")
    parser.add_argument("output_dir", help="Output directory for tops_rtc_plan.json")
    args = parser.parse_args()
    result = prepare_tops_rtc(args.manifest, args.output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run focused tests**

Run: `python -m pytest tests/test_tops_rtc.py`

Expected: PASS.

### Task 3: Verify And Record

**Files:**
- Modify: `progress.md`
- Modify: `findings.md`

- [ ] **Step 1: Run combined regression tests**

Run: `python -m pytest tests/test_sentinel_importer.py tests/test_tops_geometry.py tests/test_tops_rtc.py`

Expected: PASS.

- [ ] **Step 2: Run compile check**

Run: `python -m py_compile scripts/sentinel_importer.py scripts/tops_geometry.py scripts/tops_rtc.py tests/test_sentinel_importer.py tests/test_tops_geometry.py tests/test_tops_rtc.py`

Expected: no output and exit code 0.

- [ ] **Step 3: Generate a real ZIP TOPS RTC plan**

Run: `rm -rf /tmp/d2sar_sentinel_import_test /tmp/d2sar_tops_rtc_prepare && python scripts/sentinel_importer.py '/home/ysdong/Temp/S1A_IW_SLC__1SDV_20230625T114146_20230625T114213_049142_05E8CA_CCD3.zip' /tmp/d2sar_sentinel_import_test && python scripts/tops_rtc.py /tmp/d2sar_sentinel_import_test/manifest.json /tmp/d2sar_tops_rtc_prepare`

Expected: JSON output with `plan_path` and `burst_count` equal to `9`.

- [ ] **Step 4: Inspect the real plan**

Read `/tmp/d2sar_tops_rtc_prepare/tops_rtc_plan.json` and verify:
- `burst_count` is `9`
- first burst output path ends with `burst_001/amplitude_fullres.h5`
- first burst `slcWindow.yoff` is `0`
- last burst `slcWindow.yoff` is `11928`

- [ ] **Step 5: Update notes**

Append to `progress.md`:

```markdown
- 已新增 `scripts/tops_rtc.py` 第一版 prepare-only 阶段，生成 per-burst RTC 执行计划 `tops_rtc_plan.json`。
- 验证命令：`python -m pytest tests/test_sentinel_importer.py tests/test_tops_geometry.py tests/test_tops_rtc.py` 通过。
- 验证命令：`python -m py_compile scripts/sentinel_importer.py scripts/tops_geometry.py scripts/tops_rtc.py tests/test_sentinel_importer.py tests/test_tops_geometry.py tests/test_tops_rtc.py` 通过。
- 真实 ZIP 验证命令：`python scripts/tops_rtc.py /tmp/d2sar_sentinel_import_test/manifest.json /tmp/d2sar_tops_rtc_prepare` 通过，burst_count=`9`。
```

Append to `findings.md`:

```markdown
- 已新增 TOPS RTC prepare-only 阶段 `scripts/tops_rtc.py`，当前只生成 burst 级 SLC window、valid window、Doppler、radargrid 和输出路径计划，不执行 RTC 计算。
```

---

## Self-Review

- Spec coverage: 覆盖 `tops_rtc.py` prepare-only 阶段、输出 plan、测试和真实 ZIP 验证。
- Placeholder scan: 无 TBD/TODO/模糊步骤。
- Type consistency: 测试与实现函数名一致：`prepare_tops_rtc`。
