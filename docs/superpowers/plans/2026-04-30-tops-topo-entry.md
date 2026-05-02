# Sentinel TOPS Topo Entry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a single-burst topo invocation path that appends longitude/latitude/height datasets to burst amplitude HDF5 files.

**Architecture:** Reuse the existing TOPS RTC plan and burst-level manifest writer. `compute_burst_topo()` ensures the burst amplitude HDF exists, writes/reuses a burst manifest, then calls an injectable topo function defaulting to `common_processing.append_topo_coordinates_hdf`. Unit tests inject a fake topo function; real ISCE3 topo validation runs in Docker.

**Tech Stack:** Python stdlib JSON/path, existing `common_processing.append_topo_coordinates_hdf`, existing `tops_rtc.py` plan/materialize helpers, pytest/unittest, Docker image `d2sar:cuda` for real ISCE3 calls.

---

### Task 1: Add Failing Topo Tests

**Files:**
- Modify: `tests/test_tops_rtc.py`

- [ ] **Step 1: Add test for topo call path**

Add this test to `TopsRtcTests`:

```python
    def test_compute_burst_topo_materializes_amplitude_and_calls_topo(self) -> None:
        from tops_rtc import compute_burst_topo, prepare_tops_rtc

        calls = []

        def fake_topo(manifest_path, dem_path, output_h5, block_rows=256, orbit_interp=None, use_gpu=False, gpu_id=0):
            calls.append(
                {
                    "manifest_path": manifest_path,
                    "dem_path": dem_path,
                    "output_h5": output_h5,
                    "block_rows": block_rows,
                    "orbit_interp": orbit_interp,
                    "use_gpu": use_gpu,
                    "gpu_id": gpu_id,
                }
            )
            with h5py.File(output_h5, "a") as f:
                shape = f["slc_amplitude"].shape
                f.create_dataset("longitude", data=np.zeros(shape, dtype=np.float32))
                f.create_dataset("latitude", data=np.ones(shape, dtype=np.float32))
                f.create_dataset("height", data=np.full(shape, 2, dtype=np.float32))
            return output_h5

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = self._make_manifest(root)
            slc_path = root / "test_slc.tif"
            self._write_test_slc(slc_path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["slc"]["path"] = str(slc_path)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            result = prepare_tops_rtc(manifest_path, root / "rtc")

            topo = compute_burst_topo(
                result["plan_path"],
                root / "dem.tif",
                burst_limit=1,
                block_rows=2,
                use_gpu=True,
                gpu_id=3,
                topo_func=fake_topo,
            )
            output_h5 = Path(topo["bursts"][0]["amplitude_h5"])
            burst_manifest = Path(topo["bursts"][0]["burst_manifest"])
            with h5py.File(output_h5, "r") as f:
                has_topo = all(name in f for name in ("longitude", "latitude", "height"))

        self.assertEqual(topo["burst_count"], 1)
        self.assertTrue(output_h5.is_file())
        self.assertTrue(burst_manifest.is_file())
        self.assertTrue(has_topo)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["dem_path"], str(root / "dem.tif"))
        self.assertEqual(calls[0]["block_rows"], 2)
        self.assertTrue(calls[0]["use_gpu"])
        self.assertEqual(calls[0]["gpu_id"], 3)
```

- [ ] **Step 2: Run test and confirm failure**

Run: `python -m pytest tests/test_tops_rtc.py::TopsRtcTests::test_compute_burst_topo_materializes_amplitude_and_calls_topo`

Expected: FAIL with missing `compute_burst_topo`.

### Task 2: Implement Topo Entry

**Files:**
- Modify: `scripts/tops_rtc.py`

- [ ] **Step 1: Import topo helper**

Modify import line to include:

```python
from common_processing import append_topo_coordinates_hdf, compute_rtc_factor, resolve_manifest_data_path, resolve_manifest_metadata_path
```

- [ ] **Step 2: Add `compute_burst_topo()`**

Add before `main()`:

```python
def compute_burst_topo(
    plan_path: str | Path,
    dem_path: str | Path,
    *,
    burst_limit: int = 1,
    block_rows: int = 256,
    orbit_interp: str | None = None,
    use_gpu: bool = False,
    gpu_id: int = 0,
    topo_func=append_topo_coordinates_hdf,
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
        if not Path(output_h5).is_file():
            write_burst_amplitude_hdf(plan["slc_path"], burst, output_h5, block_rows=block_rows)
        topo_func(
            str(burst_manifest),
            str(dem_path),
            output_h5,
            block_rows=block_rows,
            orbit_interp=orbit_interp,
            use_gpu=use_gpu,
            gpu_id=gpu_id,
        )
        outputs.append(
            {
                "burstIndex": burst["burstIndex"],
                "amplitude_h5": output_h5,
                "topo_h5": output_h5,
                "burst_manifest": str(burst_manifest),
            }
        )
    return {"plan_path": str(plan_path), "burst_count": len(outputs), "bursts": outputs}
```

- [ ] **Step 3: Extend CLI**

Add args:

```python
    parser.add_argument("--compute-topo", action="store_true", help="Append per-burst topo coordinates to amplitude HDF5 files")
    parser.add_argument("--topo-gpu", action="store_true", help="Use ISCE3 CUDA Rdr2Geo for topo")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU id for topo when --topo-gpu is set")
```

After RTC factor block:

```python
    if args.compute_topo:
        if not args.dem:
            raise SystemExit("--dem is required with --compute-topo")
        result["topo"] = compute_burst_topo(
            result["plan_path"],
            args.dem,
            burst_limit=args.burst_limit or 1,
            block_rows=args.block_rows,
            orbit_interp=args.orbit_interp,
            use_gpu=args.topo_gpu,
            gpu_id=args.gpu_id,
        )
```

- [ ] **Step 4: Run focused test**

Run: `python -m pytest tests/test_tops_rtc.py::TopsRtcTests::test_compute_burst_topo_materializes_amplitude_and_calls_topo`

Expected: PASS.

### Task 3: Verify Locally

**Files:**
- Modify: `progress.md`
- Modify: `findings.md`

- [ ] **Step 1: Run regression tests**

Run: `python -m pytest tests/test_sentinel_importer.py tests/test_tops_geometry.py tests/test_tops_rtc.py`

Expected: PASS.

- [ ] **Step 2: Run compile check**

Run: `python -m py_compile scripts/common_processing.py scripts/sentinel_importer.py scripts/tops_geometry.py scripts/tops_rtc.py tests/test_sentinel_importer.py tests/test_tops_geometry.py tests/test_tops_rtc.py`

Expected: no output and exit code 0.

### Task 4: Verify Real DEM In Docker

**Files:**
- Modify: `progress.md`

- [ ] **Step 1: Run Docker topo command**

Run:

```bash
docker run --rm --gpus all --user $(id -u):$(id -g) \
  -v /home/ysdong/Software/D2SAR:/work \
  -v /home/ysdong/Software/D2SAR/results:/results \
  -v /home/ysdong/Temp/:/temp \
  d2sar:cuda \
  python3 /work/scripts/tops_rtc.py \
    /temp/d2sar_sentinel_import_test/manifest.json \
    /temp/d2sar_tops_rtc_topo \
    --materialize \
    --compute-topo \
    --dem /temp/s1/proc/dem/dem.tif \
    --burst-limit 1 \
    --block-rows 256
```

Expected: command succeeds and reports `topo.burst_count` equal to `1`.

- [ ] **Step 2: Inspect real topo HDF**

Run:

```bash
python - <<'PY'
import h5py
p='/home/ysdong/Temp/d2sar_tops_rtc_topo/burst_001/amplitude_fullres.h5'
with h5py.File(p,'r') as f:
    print(f['slc_amplitude'].shape)
    print(f['longitude'].shape, f['latitude'].shape, f['height'].shape)
    print(float(f['longitude'][0,0]), float(f['latitude'][0,0]), float(f['height'][0,0]))
PY
```

Expected: topo datasets exist with shape `(1491, 21677)`.

- [ ] **Step 3: Update notes**

Append to `progress.md`:

```markdown
- 已为 `scripts/tops_rtc.py` 增加单 burst topo 调用链，复用 burst manifest 与 `append_topo_coordinates_hdf()`，可在 Docker/ISCE3 环境中执行。
- 验证命令：`python -m pytest tests/test_sentinel_importer.py tests/test_tops_geometry.py tests/test_tops_rtc.py` 通过。
- Docker 真实 DEM topo 验证命令：`docker run ... python3 /work/scripts/tops_rtc.py ... --compute-topo --dem /temp/s1/proc/dem/dem.tif --burst-limit 1`。
```

Append to `findings.md`:

```markdown
- `tops_rtc.py` 已具备单 burst topo 调用入口，输出仍写入 burst amplitude HDF，追加 `longitude/latitude/height`；真实 ISCE3 调用需在 `d2sar:cuda` Docker 镜像中执行。
```

---

## Self-Review

- Spec coverage: 覆盖单 burst topo API、CLI、fake topo 测试、Docker 真实 DEM 验证。
- Placeholder scan: 无 TBD/TODO/模糊步骤。
- Type consistency: 函数名一致：`compute_burst_topo`。
