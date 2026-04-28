# StripInSAR2 Normal + Crop Migration Plan

## Goal

Migrate the `normal(prep)` and `crop` logic from `scripts/strip_insar.py` into the front of `scripts/strip_insar2.py`, so that:

- external crop input uses only geographic bbox (`min_lon, min_lat, max_lon, max_lat`)
- processing order is fixed as `normal -> crop`
- a final prepared runtime input set is generated under the pair work directory
- only final prepared artifacts are retained
- existing `p0/p1/p2/p3/p4/p5/p6` in `strip_insar2.py` can continue without major downstream changes

The prepared runtime set should include:

- prepared master manifest + metadata json files
- prepared slave manifest + metadata json files
- prepared master/slave SLC rasters
- prepared DEM
- one lightweight summary json for provenance


## Current Situation

### In `strip_insar.py`

The existing flow is effectively:

1. precheck
2. prep / normal
3. crop
4. downstream `p0-p5`

Relevant functions:

- `_write_prep_stage(...)`
- `_write_crop_stage(...)`
- `build_preprocess_plan(...)` in `scripts/insar_preprocess.py`
- `normalize_crop_request(...)` in `scripts/insar_crop.py`
- `build_cropped_manifest(...)` in `scripts/insar_subset.py`

Important behavior:

- `prep(normal)` mainly normalizes the slave scene to be compatible with the master geometry assumptions
- `crop` computes a master-driven crop window and applies the same window to master and normalized slave
- downstream runtime stages then run on cropped manifests, not on the original manifests

### In `strip_insar2.py`

`strip_insar2.py` currently enters the main chain directly through:

1. `load_pair_context(...)`
2. `run_geo2rdr_stage(...)` (`p0`)
3. `run_resample_stage(...)` (`p1`)
4. `run_crossmul_stage(...)` (`p2`)
5. `run_unwrap_stage(...)` (`p3`)
6. `run_los_stage(...)` (`p4`)
7. `write_primary_product(...)` (`p5`)
8. `export_insar_products(...)` (`p6`)

This means crop/normal cannot be inserted inside `run_geo2rdr_stage(...)`, because by that point the original manifests have already been loaded into the processing context.


## Recommendation

Do **not** port the `check/prep/crop` stage shell from `strip_insar.py` into `strip_insar2.py`.

Instead, implement a **runtime input preparation layer** in `strip_insar2.py` before `load_pair_context(...)`.

Recommended high-level flow:

1. read original master/slave manifests
2. build pair directory
3. run slave normalization
4. convert bbox to a master-driven crop window
5. crop master and normalized slave using the same window
6. prepare a local DEM for the prepared runtime inputs
7. write prepared manifests/json/slc/dem under `pair_dir/prepared/`
8. call `load_pair_context(...)` using the prepared manifests and prepared DEM
9. continue through existing `p0-p6` with minimal changes

This keeps `strip_insar2.py` structurally simple while still reusing the useful low-level logic from `strip_insar.py`.


## Proposed Prepared Directory Layout

Under:

```text
pair_dir/prepared/
```

create:

```text
prepared/
  master/
    manifest.json
    acquisition.json
    radargrid.json
    doppler.json
    orbit.json
    scene.json
    slc.tif
    slc.vrt
  slave/
    manifest.json
    acquisition.json
    radargrid.json
    doppler.json
    orbit.json
    scene.json
    slc.tif
    slc.vrt
  dem/
    dem.tif or dem.vrt
  prepare_summary.json
```

Only this final prepared set should be retained. Temporary normalize/crop scratch should be deleted after preparation completes.


## Processing Order

### Step 1: Normal

First normalize the slave input.

Why:

- the slave may need PRF normalization
- the slave may need radar-grid normalization
- the slave may need doppler/geometry-mode normalization
- crop should be applied after slave normalization so that prepared master/slave are already aligned to the intended runtime geometry assumptions

Recommended reused capability:

- `build_preprocess_plan(...)` from `scripts/insar_preprocess.py`

Do not reuse:

- `_write_prep_stage(...)`

Reason:

- it is stage-oriented
- it preserves intermediate stage artifacts
- it does not match the desired final prepared-input-only workflow

### Step 2: Crop

Then crop both master and normalized slave using the same master-derived crop window.

External interface should accept only bbox.

Internally:

- use `normalize_crop_request(...)` to convert bbox to `master_window`
- apply the same `master_window` to:
  - original master manifest
  - normalized slave manifest

Recommended reused capability:

- `normalize_crop_request(...)`
- `build_cropped_manifest(...)`


## Why This Should Be Considered a P0-Front Migration, Not a P0-Internal Migration

Logically this belongs before `p0`.

It should not be inserted into `run_geo2rdr_stage(...)` because:

- `PairContext` has already been built from the input manifests
- DEM has already been resolved
- downstream path resolution would become inconsistent if manifests/SLCs are replaced after context creation

So the correct insertion point is:

```text
process_strip_insar2(...)
  -> prepare runtime inputs
  -> load_pair_context(prepared manifests, prepared dem)
  -> existing p0-p6
```


## Detailed Modification Plan

### 1. `scripts/strip_insar2.py`

This is the main integration point.

#### 1.1 Extend public interface

Add bbox support to `process_strip_insar2(...)` and CLI.

Suggested new argument:

```python
bbox: tuple[float, float, float, float] | None = None
```

CLI should expose:

```text
--bbox MIN_LON MIN_LAT MAX_LON MAX_LAT
```

Only bbox should be exposed externally. No window-based crop interface is needed for the first migration.

#### 1.2 Build pair directory before context load

Currently `pair_dir` is derived inside `load_pair_context(...)`.

This needs to be split so that:

- original metadata can be loaded early
- `pair_name` and `pair_dir` can be computed
- `pair_dir/prepared/` can be created before calling `load_pair_context(...)`

This likely requires a lightweight helper such as:

```python
_derive_pair_identity(master_manifest_path, slave_manifest_path) -> (pair_name, pair_dir)
```

or refactoring a subset of `load_pair_context(...)`.

#### 1.3 Add `_prepare_runtime_inputs(...)`

Suggested signature:

```python
def _prepare_runtime_inputs(
    *,
    master_manifest_path: str | Path,
    slave_manifest_path: str | Path,
    pair_dir: Path,
    bbox: tuple[float, float, float, float] | None,
    dem_path: str | None,
    dem_cache_dir: str | None,
    dem_margin_deg: float,
) -> dict:
```

Suggested return:

```python
{
    "prepared_master_manifest": "...",
    "prepared_slave_manifest": "...",
    "prepared_dem": "...",
    "effective_bbox": [...],
    "effective_master_window": {...},
    "prepare_summary": "...",
}
```

#### 1.4 Add `_normalize_slave_for_runtime(...)`

This function should:

- read slave metadata
- reuse `build_preprocess_plan(...)`
- write final normalized outputs directly into `prepared/slave/`
- return the final normalized slave manifest path

The function should not preserve stage-style prep artifacts.

#### 1.5 Add `_crop_runtime_inputs_from_bbox(...)`

This function should:

- compute `master_window` from bbox
- apply the same window to:
  - original master
  - normalized slave
- write outputs directly into:
  - `prepared/master/`
  - `prepared/slave/`

It should return:

- prepared master manifest
- prepared slave manifest
- effective bbox
- effective master window

#### 1.6 Add `_prepare_runtime_dem(...)`

This function should:

- create a DEM local to `prepared/dem/`
- prefer bbox-scoped DEM if possible
- return a DEM path suitable for `load_pair_context(...)`

This should replace the current weak behavior in `_resolve_dem_path(...)`, which only supports explicit DEM or manifest DEM.

#### 1.7 Add `_write_prepare_summary(...)`

This should write:

- original input manifests
- prepared manifests
- effective bbox
- effective master window
- normalize actions
- DEM source and final DEM path

This summary is the only lightweight provenance file to retain.

#### 1.8 Update `process_strip_insar2(...)`

Recommended flow:

```text
process_strip_insar2(...)
  -> derive pair_dir
  -> _prepare_runtime_inputs(...)
  -> load_pair_context(prepared_master_manifest, prepared_slave_manifest, dem_path=prepared_dem)
  -> existing p0-p6 unchanged
```

This is the key architectural change.


### 2. `scripts/insar_preprocess.py`

Keep the algorithmic core.

Do not replace `build_preprocess_plan(...)`.

Recommended addition:

```python
build_prepared_normalized_slave(...)
```

Purpose:

- wrap `build_preprocess_plan(...)`
- control final output location and names under `prepared/slave/`
- avoid leaving prep-stage-oriented filenames or scratch behind

If no new wrapper is added here, `strip_insar2.py` will need to do more orchestration itself.


### 3. `scripts/insar_subset.py`

This file can mostly be reused, but it needs two important changes.

#### 3.1 Support stable prepared output naming

Current naming is stage-oriented.

Need support for final runtime names, such as:

- `prepared/master/manifest.json`
- `prepared/master/slc.tif`
- `prepared/master/acquisition.json`

Recommended approach:

Add an optional naming argument, for example:

```python
output_basename: str | None = None
```

#### 3.2 Update `scene.json`

This is a required fix.

Current `build_cropped_manifest(...)` copies `scene.json` without updating `sceneCorners`.

This is unsafe because later `strip_insar2.py` may use scene corners when selecting UTM zone.

At minimum:

- if bbox is provided, rewrite `sceneCorners` to the bbox rectangle corners

This is approximate, but still much better than keeping original full-scene corners after crop.


### 4. `scripts/insar_crop.py`

This file can be reused with minimal change.

Current `normalize_crop_request(...)` already supports:

- bbox -> master_window

Recommended first-phase policy:

- keep the existing implementation
- expose only bbox externally
- keep window support internal only if needed

Important note:

Current bbox-to-window conversion is based on scene-corner linear mapping, not strict topo-driven geometry.

This is acceptable for phase 1, but should be documented as approximate.


## DEM Handling Plan

Prepared runtime inputs should include DEM artifacts.

### Phase 1 DEM policy

- if `dem_path` is explicitly provided:
  - stage it into `prepared/dem/`
  - copy or build a local VRT
- if `dem_path` is not provided:
  - migrate the DEM resolution logic from `strip_insar.py`
  - resolve DEM using scene/bbox context

Recommended source to reuse:

- `_resolve_dem_path(...)` logic in `scripts/strip_insar.py`

Current `strip_insar2.py` DEM logic is too weak for this migration and should not remain the only path.


## UTM / Scene Corner Risk

This is the main downstream correctness risk.

Current `append_utm_coordinates_hdf(...)` in `strip_insar2.py` uses scene corners first to determine the center point for UTM EPSG selection.

If cropped manifests keep full-scene corners, UTM zone selection may be wrong for cropped edge cases.

Recommended fix in `strip_insar2.py`:

- if input is marked as prepared/cropped
  - prefer deriving center lon/lat from HDF `longitude/latitude`
  - use scene corners only as fallback

This should be done even if `scene.json` is also updated, because it makes the runtime more robust.


## What Should Not Be Migrated

Do not migrate the following from `strip_insar.py`:

- `check/prep/crop` stage wrappers
- stage success markers for prep/crop
- preview PNG generation for normal/crop
- stage-oriented intermediate record files
- cache semantics for prep/crop stages

Reason:

- the target workflow explicitly does not want to retain those intermediate outputs
- `strip_insar2.py` should remain a streamlined runtime pipeline


## Impact on Existing `strip_insar2.py` Stages

### p0

No core algorithm change required.

It will simply operate on prepared master/slave manifests and prepared DEM.

### p1

No core algorithm change required.

It will naturally work on cropped/normalized master/slave SLCs with smaller dimensions.

### p2 / p3 / p4

No core algorithm change required.

These stages should continue to operate normally once prepared inputs are used.

### p5

Needs care only for coordinate/UTM center selection logic, because cropped scene metadata may differ from original full-scene metadata.


## Minimum Test Plan

### Test 1: prepared inputs are created

Verify that bbox-driven runtime preparation produces:

- prepared master manifest
- prepared slave manifest
- prepared DEM

### Test 2: order is `normal -> crop`

Mock the preparation helpers and verify that crop runs on the normalized slave manifest, not the original slave manifest.

### Test 3: prepared manifests are fed into `load_pair_context(...)`

Mock `_prepare_runtime_inputs(...)` and assert that downstream context loading uses prepared paths.

### Test 4: cropped master/slave metadata dimensions match

Verify prepared radargrid sizes match after crop.

### Test 5: UTM selection is not tied to original full-scene corners

For prepared/cropped inputs, verify UTM center selection prefers actual lon/lat grids or updated scene corners.

### Test 6: no prep/crop intermediate artifacts remain

Verify only `prepared/` final artifacts are retained after preparation completes.


## Implementation Sequence

### Phase 1 ✅ COMPLETED

1. ✅ add bbox parameter to `strip_insar2.py`
2. ✅ derive pair directory before context construction
3. ✅ implement `_prepare_runtime_inputs(...)`
4. ✅ implement slave normalization wrapper
5. ✅ implement bbox crop wrapper
6. ✅ implement prepared DEM staging
7. ✅ switch `load_pair_context(...)` to prepared inputs
8. ✅ add `prepare_summary.json`
9. ✅ tests (see `implementation_status_strip_insar2_crop.md`)

### Phase 2 ✅ COMPLETED (partial)

1. ✅ update `scene.json` corners during crop (`_update_scene_corners_for_window()`)
2. ✅ make UTM center selection robust for prepared/cropped inputs (`append_utm_coordinates_hdf()`)
3. ✅ improve DEM clipping behavior (GDAL Warp with margin expansion)

### Phase 3 ⏳ Future

- Replace approximate scene-corner-based bbox-to-window conversion with geometry-driven crop-window derivation

---

## Implementation Status

**Date Completed**: 2025-04-25

**Files Modified**:
- `scripts/strip_insar2.py`: New functions `_derive_pair_identity()`, `_prepare_runtime_inputs()`
- `scripts/insar_subset.py`: New function `_update_scene_corners_for_window()`

**New File**: `implementation_status_strip_insar2_crop.md` - detailed implementation and test documentation

---

### Phase 3

Replace approximate scene-corner-based bbox-to-window conversion with a more geometry-driven crop-window derivation if needed.


## Final Recommendation

Adopt the following implementation boundary:

- `strip_insar2.py` handles orchestration and prepared runtime input generation
- `insar_preprocess.py` provides slave normalization capability
- `insar_subset.py` provides crop/subset capability, enhanced for prepared outputs
- `insar_crop.py` remains the bbox-to-window normalizer
- only `prepared/` final runtime inputs are retained
- existing `p0-p6` remain largely unchanged

This is the lowest-risk path that satisfies:

- geographic bbox crop only
- `normal -> crop` order
- final prepared master/slave/json/slc/dem artifacts under the work directory
- no intermediate prep/crop stage clutter
- minimal disruption to the existing `strip_insar2.py` downstream pipeline
