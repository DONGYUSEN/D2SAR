# Sentinel TOPS RTC Amplitude Materialize Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a materialize stage that reads Sentinel TOPS per-burst SLC windows and writes burst amplitude HDF5 skeleton products.

**Architecture:** Keep `prepare_tops_rtc()` unchanged as plan generation. Add `write_burst_amplitude_hdf()` for one burst and `materialize_tops_rtc_plan()` for executing selected burst plans. The stage writes `slc_amplitude` and `valid_mask` only; RTC factor, DEM/topo, geocoding, GPU dispatch, and burst merge remain out of scope.

**Tech Stack:** Python, GDAL, NumPy, h5py, existing `tops_rtc.py` plan format, pytest/unittest.

---

### Task 1: Add Failing Materialize Tests

**Files:**
- Modify: `tests/test_tops_rtc.py`

- [ ] **Step 1: Add imports and helper for temporary complex GeoTIFF**

Add imports near the top of `tests/test_tops_rtc.py`:

```python
import h5py
import numpy as np
from osgeo import gdal
```

Add this helper inside `TopsRtcTests`:

```python
    def _write_test_slc(self, path: Path) -> None:
        driver = gdal.GetDriverByName("GTiff")
        ds = driver.Create(str(path), 4, 8, 1, gdal.GDT_CFloat32)
        if ds is None:
            raise RuntimeError("failed to create test SLC")
        arr = np.arange(32, dtype=np.float32).reshape(8, 4) + 1j
        ds.GetRasterBand(1).WriteArray(arr.astype(np.complex64))
        ds = None
```

- [ ] **Step 2: Add materialize behavior test**

Add test:

```python
    def test_materialize_tops_rtc_plan_writes_burst_amplitude_hdf(self) -> None:
        from tops_rtc import materialize_tops_rtc_plan, prepare_tops_rtc

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = self._make_manifest(root)
            slc_path = root / "test_slc.tif"
            self._write_test_slc(slc_path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["slc"]["path"] = str(slc_path)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            result = prepare_tops_rtc(manifest_path, root / "rtc")
            materialized = materialize_tops_rtc_plan(result["plan_path"], burst_limit=1, block_rows=2)
            output_h5 = Path(materialized["bursts"][0]["amplitude_h5"])

            with h5py.File(output_h5, "r") as f:
                amplitude = f["slc_amplitude"][:]
                valid_mask = f["valid_mask"][:]
                product_type = f.attrs["product_type"]
                burst_index = int(f.attrs["burst_index"])

        self.assertEqual(materialized["burst_count"], 1)
        self.assertTrue(output_h5.is_file())
        self.assertEqual(product_type, "sentinel_tops_burst_amplitude")
        self.assertEqual(burst_index, 1)
        self.assertEqual(amplitude.shape, (4, 4))
        self.assertAlmostEqual(float(amplitude[0, 0]), 1.0)
        self.assertAlmostEqual(float(amplitude[3, 3]), float(np.sqrt(15.0**2 + 1.0)))
        self.assertEqual(int(valid_mask.sum()), 4)
        self.assertEqual(int(valid_mask[:, 1].sum()), 4)
        self.assertEqual(int(valid_mask[:, 0].sum()), 0)
```

- [ ] **Step 3: Run test and confirm failure**

Run: `python -m pytest tests/test_tops_rtc.py::TopsRtcTests::test_materialize_tops_rtc_plan_writes_burst_amplitude_hdf`

Expected: FAIL with `ImportError` or `cannot import name 'materialize_tops_rtc_plan'`.

### Task 2: Implement Materialize Stage

**Files:**
- Modify: `scripts/tops_rtc.py`

- [ ] **Step 1: Add imports**

Add to `scripts/tops_rtc.py`:

```python
import h5py
import numpy as np
from osgeo import gdal
```

- [ ] **Step 2: Add `materialize_tops_rtc_plan()` and `write_burst_amplitude_hdf()`**

Add before `main()`:

```python
def materialize_tops_rtc_plan(
    plan_path: str | Path,
    *,
    burst_limit: int | None = None,
    block_rows: int = 256,
) -> dict[str, Any]:
    plan_path = Path(plan_path)
    with plan_path.open(encoding="utf-8") as f:
        plan = json.load(f)
    bursts = plan.get("bursts", [])
    if burst_limit is not None:
        bursts = bursts[: max(0, int(burst_limit))]

    materialized = []
    for burst in bursts:
        output_h5 = burst["outputs"]["amplitude_h5"]
        write_burst_amplitude_hdf(plan["slc_path"], burst, output_h5, block_rows=block_rows)
        materialized.append({"burstIndex": burst["burstIndex"], "amplitude_h5": output_h5})
    return {"plan_path": str(plan_path), "burst_count": len(materialized), "bursts": materialized}


def write_burst_amplitude_hdf(
    slc_path: str,
    burst_plan: dict[str, Any],
    output_h5: str | Path,
    *,
    block_rows: int = 256,
) -> str:
    slc_ds = gdal.Open(slc_path)
    if slc_ds is None:
        raise RuntimeError(f"failed to open SLC: {slc_path}")

    slc_window = burst_plan["slcWindow"]
    xoff = int(slc_window["xoff"])
    yoff = int(slc_window["yoff"])
    width = int(slc_window["xsize"])
    length = int(slc_window["ysize"])
    output_h5 = Path(output_h5)
    output_h5.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_h5, "w") as f:
        f.attrs["product_type"] = "sentinel_tops_burst_amplitude"
        f.attrs["burst_index"] = int(burst_plan["burstIndex"])
        f.attrs["source_slc"] = slc_path
        f.attrs["slc_window_xoff"] = xoff
        f.attrs["slc_window_yoff"] = yoff
        f.attrs["radiometry"] = "amplitude"
        f.attrs["value_domain"] = "linear"
        valid_window = slc_window["validWindow"]
        for key, value in valid_window.items():
            f.attrs[f"valid_window_{key}"] = int(value)

        chunk_rows = max(1, min(int(block_rows), length))
        chunk_cols = max(1, min(1024, width))
        d_amp = f.create_dataset(
            "slc_amplitude",
            shape=(length, width),
            dtype="f4",
            chunks=(chunk_rows, chunk_cols),
            compression="gzip",
            shuffle=True,
        )
        d_mask = f.create_dataset(
            "valid_mask",
            shape=(length, width),
            dtype="u1",
            chunks=(chunk_rows, chunk_cols),
            compression="gzip",
            shuffle=True,
        )
        d_mask[:, :] = 0
        valid_x = int(valid_window["xoff"])
        valid_y = int(valid_window["yoff"])
        valid_w = int(valid_window["xsize"])
        valid_h = int(valid_window["ysize"])
        d_mask[valid_y : valid_y + valid_h, valid_x : valid_x + valid_w] = 1

        band1 = slc_ds.GetRasterBand(1)
        band2 = slc_ds.GetRasterBand(2) if slc_ds.RasterCount >= 2 else None
        for row0 in range(0, length, block_rows):
            rows = min(block_rows, length - row0)
            block1 = band1.ReadAsArray(xoff, yoff + row0, width, rows)
            if block1 is None:
                raise RuntimeError("failed to read SLC block")
            if band2 is not None:
                block2 = band2.ReadAsArray(xoff, yoff + row0, width, rows)
                if block2 is None:
                    raise RuntimeError("failed to read SLC block")
                amplitude = np.sqrt(block1.astype(np.float32) ** 2 + block2.astype(np.float32) ** 2)
            else:
                amplitude = np.abs(block1).astype(np.float32)
            d_amp[row0 : row0 + rows, :] = amplitude.astype(np.float32)
    return str(output_h5)
```

- [ ] **Step 3: Extend CLI**

Modify `main()` to add:

```python
    parser.add_argument("--materialize", action="store_true", help="Write per-burst amplitude HDF5 files")
    parser.add_argument("--burst-limit", type=int, default=None, help="Limit materialization to first N bursts")
    parser.add_argument("--block-rows", type=int, default=256, help="Rows per SLC read block")
```

After `prepare_tops_rtc(...)`, add:

```python
    if args.materialize:
        result["materialized"] = materialize_tops_rtc_plan(
            result["plan_path"], burst_limit=args.burst_limit, block_rows=args.block_rows
        )
```

- [ ] **Step 4: Run focused materialize test**

Run: `python -m pytest tests/test_tops_rtc.py::TopsRtcTests::test_materialize_tops_rtc_plan_writes_burst_amplitude_hdf`

Expected: PASS.

### Task 3: Verify Regression And Real ZIP

**Files:**
- Modify: `progress.md`
- Modify: `findings.md`

- [ ] **Step 1: Run TOPS regression tests**

Run: `python -m pytest tests/test_sentinel_importer.py tests/test_tops_geometry.py tests/test_tops_rtc.py`

Expected: PASS.

- [ ] **Step 2: Run compile check**

Run: `python -m py_compile scripts/common_processing.py scripts/sentinel_importer.py scripts/tops_geometry.py scripts/tops_rtc.py tests/test_sentinel_importer.py tests/test_tops_geometry.py tests/test_tops_rtc.py`

Expected: no output and exit code 0.

- [ ] **Step 3: Materialize one real ZIP burst**

Run: `rm -rf /tmp/d2sar_sentinel_import_test /tmp/d2sar_tops_rtc_prepare && python scripts/sentinel_importer.py '/home/ysdong/Temp/S1A_IW_SLC__1SDV_20230625T114146_20230625T114213_049142_05E8CA_CCD3.zip' /tmp/d2sar_sentinel_import_test && python scripts/tops_rtc.py /tmp/d2sar_sentinel_import_test/manifest.json /tmp/d2sar_tops_rtc_prepare --materialize --burst-limit 1 --block-rows 256`

Expected: command succeeds and reports `materialized.burst_count` equal to `1`.

- [ ] **Step 4: Inspect real HDF**

Run a Python one-liner or small script to open `/tmp/d2sar_tops_rtc_prepare/burst_001/amplitude_fullres.h5` and verify:
- `slc_amplitude` shape is `(1491, 21677)`
- `valid_mask` shape is `(1491, 21677)`
- `valid_mask.sum()` is positive
- `product_type` is `sentinel_tops_burst_amplitude`

- [ ] **Step 5: Update notes**

Append to `progress.md`:

```markdown
- 已为 `scripts/tops_rtc.py` 增加 materialize 阶段，支持按 burst SLC window 写出 `amplitude_fullres.h5`。
- 验证命令：`python -m pytest tests/test_sentinel_importer.py tests/test_tops_geometry.py tests/test_tops_rtc.py` 通过。
- 真实 ZIP 验证命令：`python scripts/tops_rtc.py /tmp/d2sar_sentinel_import_test/manifest.json /tmp/d2sar_tops_rtc_prepare --materialize --burst-limit 1 --block-rows 256` 通过，写出 `burst_001/amplitude_fullres.h5`。
```

Append to `findings.md`:

```markdown
- `tops_rtc.py` materialize 阶段当前已能读取 per-burst SLC window 并写出 radar-domain amplitude HDF5 和 valid mask；尚未应用 RTC factor、DEM/topo 或 geocode。
```

---

## Self-Review

- Spec coverage: 覆盖 burst window 读取、amplitude HDF 写出、valid mask、CLI materialize、真实 ZIP 单 burst 验证。
- Placeholder scan: 无 TBD/TODO/模糊步骤。
- Type consistency: 函数名一致：`materialize_tops_rtc_plan`、`write_burst_amplitude_hdf`。
