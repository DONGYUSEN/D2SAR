# Sentinel TOPS RTC Factor Entry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a single-burst RTC factor invocation path using burst-level temporary metadata manifests.

**Architecture:** Keep `tops_rtc.py` as the orchestration module. Add a burst manifest writer that projects the existing plan metadata into a temporary single-burst manifest, then call an injectable RTC factor function defaulting to `common_processing.compute_rtc_factor`. Tests validate the manifest and call path without requiring a real DEM/ISCE3 RTC run.

**Tech Stack:** Python stdlib JSON/path/tempfile, existing `common_processing.compute_rtc_factor`, existing TOPS RTC plan JSON, pytest/unittest.

---

### Task 1: Add Failing RTC Factor Tests

**Files:**
- Modify: `tests/test_tops_rtc.py`

- [ ] **Step 1: Add test for single-burst RTC factor invocation**

Add this test to `TopsRtcTests`:

```python
    def test_compute_burst_rtc_factor_writes_burst_manifest_and_calls_compute(self) -> None:
        from tops_rtc import compute_burst_rtc_factor, prepare_tops_rtc

        calls = []

        def fake_compute(manifest_path, dem_path, output_path, orbit_interp="Legendre"):
            calls.append(
                {
                    "manifest_path": manifest_path,
                    "dem_path": dem_path,
                    "output_path": output_path,
                    "orbit_interp": orbit_interp,
                }
            )
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text("rtc", encoding="utf-8")
            return output_path

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = self._make_manifest(root)
            result = prepare_tops_rtc(manifest_path, root / "rtc")

            outputs = compute_burst_rtc_factor(
                result["plan_path"],
                root / "dem.tif",
                burst_limit=1,
                compute_func=fake_compute,
            )
            burst_manifest = Path(outputs["bursts"][0]["burst_manifest"])
            burst_manifest_data = json.loads(burst_manifest.read_text(encoding="utf-8"))
            radargrid = json.loads(
                Path(burst_manifest_data["metadata"]["radargrid"]).read_text(encoding="utf-8")
            )
            doppler = json.loads(
                Path(burst_manifest_data["metadata"]["doppler"]).read_text(encoding="utf-8")
            )

        self.assertEqual(outputs["burst_count"], 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["dem_path"], str(root / "dem.tif"))
        self.assertTrue(calls[0]["output_path"].endswith("burst_001/rtc_factor.tif"))
        self.assertEqual(burst_manifest_data["sensor"], "sentinel-1")
        self.assertEqual(burst_manifest_data["tops"]["burst_index"], 1)
        self.assertEqual(radargrid["numberOfRows"], 4)
        self.assertEqual(radargrid["numberOfColumns"], 4)
        self.assertEqual(doppler["combinedDoppler"]["coefficients"], [1.0, 2.0, 3.0])
```

- [ ] **Step 2: Run test and confirm failure**

Run: `python -m pytest tests/test_tops_rtc.py::TopsRtcTests::test_compute_burst_rtc_factor_writes_burst_manifest_and_calls_compute`

Expected: FAIL with missing `compute_burst_rtc_factor`.

### Task 2: Implement Burst RTC Factor Entry

**Files:**
- Modify: `scripts/tops_rtc.py`

- [ ] **Step 1: Add imports**

Add to `scripts/tops_rtc.py`:

```python
import tempfile
from common_processing import compute_rtc_factor
```

- [ ] **Step 2: Add helper functions**

Add before `main()`:

```python
def compute_burst_rtc_factor(
    plan_path: str | Path,
    dem_path: str | Path,
    *,
    burst_limit: int = 1,
    orbit_interp: str = "Legendre",
    compute_func=compute_rtc_factor,
) -> dict[str, Any]:
    plan_path = Path(plan_path)
    with plan_path.open(encoding="utf-8") as f:
        plan = json.load(f)
    bursts = plan.get("bursts", [])[: max(0, int(burst_limit))]
    outputs = []
    with tempfile.TemporaryDirectory(prefix="tops_rtc_burst_", dir=str(plan_path.parent)) as tmp:
        tmp_dir = Path(tmp)
        for burst in bursts:
            burst_manifest = write_burst_metadata_manifest(plan, burst, tmp_dir)
            output_path = burst["outputs"]["rtc_factor_tif"]
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            compute_func(str(burst_manifest), str(dem_path), output_path, orbit_interp=orbit_interp)
            outputs.append(
                {
                    "burstIndex": burst["burstIndex"],
                    "rtc_factor_tif": output_path,
                    "burst_manifest": str(burst_manifest),
                }
            )
    return {"plan_path": str(plan_path), "burst_count": len(outputs), "bursts": outputs}


def write_burst_metadata_manifest(plan: dict[str, Any], burst: dict[str, Any], work_dir: str | Path) -> Path:
    work_dir = Path(work_dir) / f"burst_{int(burst['burstIndex']):03d}"
    metadata_dir = work_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    source_manifest_path = Path(plan["input_manifest"])
    with source_manifest_path.open(encoding="utf-8") as f:
        source_manifest = json.load(f)

    metadata = _load_source_metadata(source_manifest_path, source_manifest)
    radargrid_path = metadata_dir / "radargrid.json"
    orbit_path = metadata_dir / "orbit.json"
    acquisition_path = metadata_dir / "acquisition.json"
    doppler_path = metadata_dir / "doppler.json"
    scene_path = metadata_dir / "scene.json"

    radargrid_path.write_text(json.dumps(burst["radargrid"], indent=2), encoding="utf-8")
    orbit_path.write_text(json.dumps(metadata["orbit"], indent=2), encoding="utf-8")
    acquisition_path.write_text(json.dumps(metadata["acquisition"], indent=2), encoding="utf-8")
    doppler_path.write_text(json.dumps(_burst_doppler_as_combined(burst["doppler"]), indent=2), encoding="utf-8")
    scene_path.write_text(json.dumps(metadata["scene"], indent=2), encoding="utf-8")

    burst_manifest = {
        "version": "1.0",
        "sensor": "sentinel-1",
        "productType": source_manifest.get("productType", "SLC"),
        "platform": source_manifest.get("platform"),
        "polarisation": plan.get("polarisation"),
        "slc": {"path": plan["slc_path"]},
        "metadata": {
            "acquisition": str(acquisition_path),
            "orbit": str(orbit_path),
            "radargrid": str(radargrid_path),
            "doppler": str(doppler_path),
            "scene": str(scene_path),
        },
        "tops": {
            "mode": "IW",
            "swath": plan.get("swath"),
            "burst_index": burst["burstIndex"],
            "burst_count": 1,
        },
    }
    manifest_path = work_dir / "manifest.json"
    manifest_path.write_text(json.dumps(burst_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path


def _load_source_metadata(source_manifest_path: Path, source_manifest: dict[str, Any]) -> dict[str, Any]:
    from common_processing import resolve_manifest_metadata_path

    loaded = {}
    for key in ("acquisition", "orbit", "scene"):
        with open(resolve_manifest_metadata_path(source_manifest_path, source_manifest, key), encoding="utf-8") as f:
            loaded[key] = json.load(f)
    return loaded


def _burst_doppler_as_combined(doppler: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": "sentinel-1-burst",
        "combinedDoppler": {
            "polynomialDegree": max(len(doppler.get("coefficients", [])) - 1, 0),
            "referencePoint": doppler.get("t0", 0.0),
            "coefficients": doppler.get("coefficients", []),
        },
    }
```

- [ ] **Step 3: Extend CLI**

Add args:

```python
    parser.add_argument("--compute-rtc-factor", action="store_true", help="Compute per-burst RTC factor TIFFs")
    parser.add_argument("--dem", help="DEM path for RTC factor computation")
    parser.add_argument("--orbit-interp", default="Legendre", help="Orbit interpolation method")
```

After materialize block:

```python
    if args.compute_rtc_factor:
        if not args.dem:
            raise SystemExit("--dem is required with --compute-rtc-factor")
        result["rtc_factor"] = compute_burst_rtc_factor(
            result["plan_path"],
            args.dem,
            burst_limit=args.burst_limit or 1,
            orbit_interp=args.orbit_interp,
        )
```

- [ ] **Step 4: Run focused test**

Run: `python -m pytest tests/test_tops_rtc.py::TopsRtcTests::test_compute_burst_rtc_factor_writes_burst_manifest_and_calls_compute`

Expected: PASS.

### Task 3: Verify And Record

**Files:**
- Modify: `progress.md`
- Modify: `findings.md`

- [ ] **Step 1: Run TOPS regression tests**

Run: `python -m pytest tests/test_sentinel_importer.py tests/test_tops_geometry.py tests/test_tops_rtc.py`

Expected: PASS.

- [ ] **Step 2: Run compile check**

Run: `python -m py_compile scripts/common_processing.py scripts/sentinel_importer.py scripts/tops_geometry.py scripts/tops_rtc.py tests/test_sentinel_importer.py tests/test_tops_geometry.py tests/test_tops_rtc.py`

Expected: no output and exit code 0.

- [ ] **Step 3: Update notes**

Append to `progress.md`:

```markdown
- 已为 `scripts/tops_rtc.py` 增加单 burst RTC factor 调用链：写出临时 burst manifest，并调用可注入的 `compute_rtc_factor`。
- 验证命令：`python -m pytest tests/test_sentinel_importer.py tests/test_tops_geometry.py tests/test_tops_rtc.py` 通过。
- 验证命令：`python -m py_compile scripts/common_processing.py scripts/sentinel_importer.py scripts/tops_geometry.py scripts/tops_rtc.py tests/test_sentinel_importer.py tests/test_tops_geometry.py tests/test_tops_rtc.py` 通过。
```

Append to `findings.md`:

```markdown
- `tops_rtc.py` 已具备单 burst RTC factor 调用入口，测试通过注入 fake compute 验证 burst-level manifest/radargrid/doppler 映射和输出路径；真实 DEM/ISCE3 RTC 运行仍需单独验证。
```

---

## Self-Review

- Spec coverage: 覆盖单 burst RTC factor 调用入口、临时 burst manifest、注入测试和 CLI 参数。
- Placeholder scan: 无 TBD/TODO/模糊步骤。
- Type consistency: 函数名一致：`compute_burst_rtc_factor`、`write_burst_metadata_manifest`。
