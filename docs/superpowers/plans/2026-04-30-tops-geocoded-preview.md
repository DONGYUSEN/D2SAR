# Sentinel TOPS Geocoded Preview Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a single-burst geocoded export stage that writes UTM GeoTIFF and PNG previews from burst amplitude HDF5 plus topo coordinates.

**Architecture:** Reuse existing `common_processing` UTM/geocoded export functions. `export_burst_geocoded_preview()` works from a TOPS RTC plan and burst manifests, verifies each burst HDF already has topo datasets, appends UTM coordinates, computes an output grid shape, and writes GeoTIFF/PNG. No burst merge or multi-burst mosaic is included.

**Tech Stack:** Python stdlib JSON/path, h5py for dataset existence checks, existing `append_utm_coordinates_hdf`, `compute_utm_output_shape`, `write_geocoded_geotiff`, `write_geocoded_png`, pytest/unittest, Docker `d2sar:cuda` for real export.

---

### Task 1: Add Failing Export Tests

**Files:**
- Modify: `tests/test_tops_rtc.py`

- [ ] **Step 1: Add test for geocoded export call path**

Add this test to `TopsRtcTests`:

```python
    def test_export_burst_geocoded_preview_calls_utm_and_writers(self) -> None:
        from tops_rtc import export_burst_geocoded_preview, prepare_tops_rtc

        calls = []

        def fake_append_utm(output_h5, manifest_path, block_rows=32):
            calls.append(("utm", output_h5, manifest_path, block_rows))
            with h5py.File(output_h5, "a") as f:
                shape = f["slc_amplitude"].shape
                f.create_dataset("utm_x", data=np.zeros(shape, dtype=np.float32))
                f.create_dataset("utm_y", data=np.ones(shape, dtype=np.float32))
            return output_h5

        def fake_shape(output_h5, resolution_meters, block_rows=64):
            calls.append(("shape", output_h5, resolution_meters, block_rows))
            return 5, 6

        def fake_tif(output_h5, tif_path, target_width=None, target_height=None, block_rows=64):
            calls.append(("tif", output_h5, tif_path, target_width, target_height, block_rows))
            Path(tif_path).write_text("tif", encoding="utf-8")
            return tif_path

        def fake_png(output_h5, png_path, target_width=None, target_height=None, block_rows=64):
            calls.append(("png", output_h5, png_path, target_width, target_height, block_rows))
            Path(png_path).write_text("png", encoding="utf-8")
            return png_path

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = self._make_manifest(root)
            slc_path = root / "test_slc.tif"
            self._write_test_slc(slc_path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["slc"]["path"] = str(slc_path)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            result = prepare_tops_rtc(manifest_path, root / "rtc")
            h5_path = Path(root / "rtc" / "burst_001" / "amplitude_fullres.h5")
            from tops_rtc import materialize_tops_rtc_plan
            materialize_tops_rtc_plan(result["plan_path"], burst_limit=1, block_rows=2)
            with h5py.File(h5_path, "a") as f:
                shape = f["slc_amplitude"].shape
                f.create_dataset("longitude", data=np.zeros(shape, dtype=np.float32))
                f.create_dataset("latitude", data=np.ones(shape, dtype=np.float32))
                f.create_dataset("height", data=np.full(shape, 2, dtype=np.float32))

            exported = export_burst_geocoded_preview(
                result["plan_path"],
                burst_limit=1,
                resolution_meters=20.0,
                block_rows=4,
                append_utm_func=fake_append_utm,
                shape_func=fake_shape,
                geotiff_func=fake_tif,
                png_func=fake_png,
            )

        self.assertEqual(exported["burst_count"], 1)
        self.assertTrue(exported["bursts"][0]["geotiff"].endswith("burst_001/amplitude_utm_geocoded.tif"))
        self.assertTrue(exported["bursts"][0]["png"].endswith("burst_001/amplitude_utm_geocoded.png"))
        self.assertEqual([call[0] for call in calls], ["utm", "shape", "tif", "png"])
        self.assertEqual(exported["bursts"][0]["target_width"], 5)
        self.assertEqual(exported["bursts"][0]["target_height"], 6)
```

- [ ] **Step 2: Run test and confirm failure**

Run: `python -m pytest tests/test_tops_rtc.py::TopsRtcTests::test_export_burst_geocoded_preview_calls_utm_and_writers`

Expected: FAIL with missing `export_burst_geocoded_preview`.

### Task 2: Implement Export Stage

**Files:**
- Modify: `scripts/tops_rtc.py`

- [ ] **Step 1: Import export helpers**

Modify `common_processing` import to include:

```python
    append_utm_coordinates_hdf,
    compute_utm_output_shape,
    write_geocoded_geotiff,
    write_geocoded_png,
```

- [ ] **Step 2: Add output paths**

In `_build_burst_outputs()`, add:

```python
        "amplitude_utm_tif": str(burst_dir / "amplitude_utm_geocoded.tif"),
        "amplitude_utm_png": str(burst_dir / "amplitude_utm_geocoded.png"),
```

- [ ] **Step 3: Add `export_burst_geocoded_preview()`**

Add before `main()`:

```python
def export_burst_geocoded_preview(
    plan_path: str | Path,
    *,
    burst_limit: int = 1,
    resolution_meters: float = 20.0,
    block_rows: int = 64,
    append_utm_func=append_utm_coordinates_hdf,
    shape_func=compute_utm_output_shape,
    geotiff_func=write_geocoded_geotiff,
    png_func=write_geocoded_png,
) -> dict[str, Any]:
    plan_path = Path(plan_path)
    with plan_path.open(encoding="utf-8") as f:
        plan = json.load(f)
    bursts = plan.get("bursts", [])[: max(0, int(burst_limit))]
    work_dir = plan_path.parent / "burst_manifests"
    outputs = []
    for burst in bursts:
        burst_manifest = write_burst_metadata_manifest(plan, burst, work_dir)
        output_h5 = burst["outputs"]["amplitude_h5"]
        _require_topo_datasets(output_h5)
        append_utm_func(output_h5, str(burst_manifest), block_rows=min(block_rows, 64))
        target_width, target_height = shape_func(
            output_h5, resolution_meters, block_rows=min(block_rows, 64)
        )
        geotiff = burst["outputs"].get("amplitude_utm_tif") or str(
            Path(burst["outputs"]["directory"]) / "amplitude_utm_geocoded.tif"
        )
        png = burst["outputs"].get("amplitude_utm_png") or str(
            Path(burst["outputs"]["directory"]) / "amplitude_utm_geocoded.png"
        )
        geotiff_func(
            output_h5,
            geotiff,
            target_width=target_width,
            target_height=target_height,
            block_rows=min(block_rows, 64),
        )
        png_func(
            output_h5,
            png,
            target_width=target_width,
            target_height=target_height,
            block_rows=min(block_rows, 64),
        )
        outputs.append(
            {
                "burstIndex": burst["burstIndex"],
                "amplitude_h5": output_h5,
                "geotiff": geotiff,
                "png": png,
                "target_width": target_width,
                "target_height": target_height,
                "burst_manifest": str(burst_manifest),
            }
        )
    return {"plan_path": str(plan_path), "burst_count": len(outputs), "bursts": outputs}


def _require_topo_datasets(output_h5: str | Path) -> None:
    with h5py.File(output_h5, "r") as f:
        missing = [name for name in ("slc_amplitude", "longitude", "latitude", "height") if name not in f]
    if missing:
        raise ValueError(f"topo datasets missing from {output_h5}: {', '.join(missing)}")
```

- [ ] **Step 4: Extend CLI**

Add args:

```python
    parser.add_argument("--export-geocoded", action="store_true", help="Write per-burst UTM GeoTIFF and PNG previews")
    parser.add_argument("--resolution", type=float, default=20.0, help="Geocoded output resolution in meters")
```

After topo block:

```python
    if args.export_geocoded:
        result["geocoded"] = export_burst_geocoded_preview(
            result["plan_path"],
            burst_limit=args.burst_limit or 1,
            resolution_meters=args.resolution,
            block_rows=args.block_rows,
        )
```

- [ ] **Step 5: Run focused test**

Run: `python -m pytest tests/test_tops_rtc.py::TopsRtcTests::test_export_burst_geocoded_preview_calls_utm_and_writers`

Expected: PASS.

### Task 3: Verify Locally

**Files:**
- Modify: `progress.md`
- Modify: `findings.md`

- [ ] **Step 1: Run regression tests**

Run: `python -m pytest tests/test_sentinel_importer.py tests/test_tops_geometry.py tests/test_tops_rtc.py tests/test_common_processing_orbit.py`

Expected: PASS.

- [ ] **Step 2: Run compile check**

Run: `python -m py_compile scripts/common_processing.py scripts/sentinel_importer.py scripts/tops_geometry.py scripts/tops_rtc.py tests/test_sentinel_importer.py tests/test_tops_geometry.py tests/test_tops_rtc.py tests/test_common_processing_orbit.py`

Expected: no output and exit code 0.

### Task 4: Verify Docker Export

**Files:**
- Modify: `progress.md`

- [ ] **Step 1: Run Docker export command**

Use existing topo output from `/home/ysdong/Temp/d2sar_tops_rtc_topo`, or regenerate if missing:

```bash
docker run --rm --gpus all --user $(id -u):$(id -g) \
  -v /home/ysdong/Software/D2SAR:/work \
  -v /home/ysdong/Software/D2SAR/results:/results \
  -v /home/ysdong/Temp/:/temp \
  d2sar:cuda \
  python3 /work/scripts/tops_rtc.py \
    /temp/d2sar_sentinel_import_test/manifest.json \
    /temp/d2sar_tops_rtc_topo \
    --export-geocoded \
    --burst-limit 1 \
    --resolution 20 \
    --block-rows 64
```

Expected: command succeeds and writes:
- `/home/ysdong/Temp/d2sar_tops_rtc_topo/burst_001/amplitude_utm_geocoded.tif`
- `/home/ysdong/Temp/d2sar_tops_rtc_topo/burst_001/amplitude_utm_geocoded.png`

- [ ] **Step 2: Inspect outputs**

Run `gdalinfo` on the GeoTIFF and verify dimensions/projection are present.

- [ ] **Step 3: Update notes**

Append to `progress.md`:

```markdown
- 已为 `scripts/tops_rtc.py` 增加单 burst geocoded preview 导出阶段，复用 UTM 坐标追加和 GeoTIFF/PNG 写出函数。
- 验证命令：`python -m pytest tests/test_sentinel_importer.py tests/test_tops_geometry.py tests/test_tops_rtc.py tests/test_common_processing_orbit.py` 通过。
- Docker 真实导出验证命令：`docker run ... python3 /work/scripts/tops_rtc.py ... --export-geocoded --burst-limit 1 --resolution 20` 通过。
```

Append to `findings.md`:

```markdown
- `tops_rtc.py` 已支持单 burst UTM GeoTIFF/PNG 预览导出，前提是 burst HDF 已包含 `longitude/latitude/height` topo 数据集。
```

---

## Self-Review

- Spec coverage: 覆盖单 burst UTM/geocoded export API、CLI、fake writer 测试和 Docker 真实导出验证。
- Placeholder scan: 无 TBD/TODO/模糊步骤。
- Type consistency: 函数名一致：`export_burst_geocoded_preview`。
