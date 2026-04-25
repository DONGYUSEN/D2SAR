# Strip Insar2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a stable `strip_insar2.py` that uses a NISAR-style strip InSAR stage flow on top of manifest-based strip inputs, with GPU-first stage execution, CPU fallback, named single-product HDF5 output, and GeoTIFF/PNG/KML exports.

**Architecture:** Keep manifest parsing and product naming in a new orchestration layer, reuse stable strip-side implementations for stage internals, and push image export/KML generation into a separate helper module. Stages fall back independently from GPU to CPU rather than downgrading the whole pipeline at once.

**Tech Stack:** Python, GDAL, existing `strip_insar.py` helpers, `common_processing.py`, `insar_stage_cache.py`

---

### Task 1: Build Naming And Orchestration Tests

**Files:**
- Create: `tests/test_strip_insar2.py`

- [ ] **Step 1: Write failing tests for pair naming, KML, and fallback**

```python
class StripInsar2FallbackTests(unittest.TestCase):
    def test_run_stage_with_fallback_uses_cpu_when_gpu_stage_fails(self):
        ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest -v tests/test_strip_insar2.py`
Expected: import or behavior failures because `strip_insar2.py` does not exist yet

- [ ] **Step 3: Implement minimal test scaffolding**

```python
if "h5py" not in sys.modules:
    sys.modules["h5py"] = mock.MagicMock()
```

- [ ] **Step 4: Re-run test to confirm targeted failures remain**

Run: `python3 -m unittest -v tests/test_strip_insar2.py`
Expected: failures now point at missing strip_insar2 functionality instead of environment setup

### Task 2: Create Export And KML Helpers

**Files:**
- Create: `scripts/strip_insar2_export.py`
- Test: `tests/test_strip_insar2.py`

- [ ] **Step 1: Implement wrapped/scalar export helper**

```python
def export_insar_products(*, input_h5, output_paths, resolution_meters, block_rows=64, generate_kml=True):
    ...
```

- [ ] **Step 2: Implement GroundOverlay KML writer**

```python
def write_ground_overlay_kml_from_geotiff(*, tif_path, image_path, output_kml, overlay_name=None):
    ...
```

- [ ] **Step 3: Run focused tests**

Run: `python3 -m unittest -v tests/test_strip_insar2.py`
Expected: KML-related tests pass, orchestration tests still fail

### Task 3: Implement Strip Insar2 Orchestrator

**Files:**
- Create: `scripts/strip_insar2.py`
- Test: `tests/test_strip_insar2.py`

- [ ] **Step 1: Implement pair context and naming helpers**

```python
def extract_scene_date(acquisition_data, orbit_data=None) -> str:
    ...

def build_pair_name(master_date: str, slave_date: str) -> str:
    return f"{master_date}_{slave_date}"
```

- [ ] **Step 2: Implement stage fallback helper**

```python
def run_stage_with_fallback(*, stage_name, gpu_mode, gpu_id, gpu_runner, cpu_runner, gpu_check=None):
    ...
```

- [ ] **Step 3: Implement stage wrappers and main process function**

```python
def process_strip_insar2(master_manifest_path, slave_manifest_path, *, output_root="results", ...):
    ...
```

- [ ] **Step 4: Run focused tests**

Run: `python3 -m unittest -v tests/test_strip_insar2.py`
Expected: all tests pass

### Task 4: Verify And Extend

**Files:**
- Modify: `scripts/strip_insar2.py`
- Modify: `scripts/strip_insar2_export.py`
- Test: `tests/test_strip_insar2.py`

- [ ] **Step 1: Verify stage record writing for p5/p6**

```python
def _write_custom_stage_record(...):
    ...
```

- [ ] **Step 2: Re-run tests after record-handling fixes**

Run: `python3 -m unittest -v tests/test_strip_insar2.py`
Expected: pass

- [ ] **Step 3: Extend later with real NISAR module substitution**

```python
# Future work:
# - replace more strip-side wrappers with direct nisar.workflows stage calls
# - add integration tests on sample manifest pairs
```
