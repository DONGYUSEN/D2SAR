# strip_insar.py Steps / Crop / PRF-DC Redesign

## Date

2026-04-18

## Summary

`strip_insar.py` currently runs as a single monolithic pipeline. Any exception in a late stage causes the entire job to fall back to CPU and restart from P0, which is too expensive for real stripmap scenes. This redesign adds:

1. explicit step execution (`--step`, `--start-step/--end-step`, `--resume`)
2. crop support (`--bbox`, `--window`)
3. master/slave compatibility checking and preprocessing for PRF / DC mismatch

The design keeps the current P0-P6 stage model, but moves execution onto a stage-planner + stage-cache model so work can be resumed or rerun surgically.

## Goals

- Avoid rerunning the full pipeline when failure happens late (for example in P6 publishing)
- Allow operators to run exactly one step, a bounded step range, or auto-resume
- Allow smaller-area processing through simple CLI crop parameters
- Detect master/slave incompatibilities before expensive processing starts
- Apply ISCE2-style secondary normalization/resampling logic for PRF/DC mismatch instead of treating all mismatch as fatal

## Non-Goals

- Replacing the existing P0-P6 stage definitions
- Rewriting the full interferometric algorithm stack
- Generalizing all common_processing helpers to every product type in this iteration
- Making every stage independently GPU-resumable from arbitrary partial internal state

## Current Problems

### Monolithic fallback behavior

Current `process_strip_insar()` wraps the full GPU path in one `try/except`, and on any exception calls `_process_insar_cpu()` from the top. This means a late failure in P6 causes a complete CPU rerun from P0.

### No persistent stage contract

The pipeline writes final products, but not a formal per-stage contract. Later stages cannot reliably discover or reuse earlier stage outputs.

### No crop control

There is currently no user-facing mechanism to limit processing to a bbox or master-pixel window.

### No compatibility gate

Master/slave metadata is loaded, but there is no explicit precheck for:

- PRF mismatch
- Doppler / DC mismatch
- center frequency mismatch
- look direction mismatch
- polarization mismatch
- geometry/radar-grid mismatch requiring resampling

## Design Overview

The redesign introduces two pre-stages plus the existing six processing stages:

- `check` — compatibility inspection and crop normalization
- `prep` — optional preprocessing / normalization of slave inputs
- `p0`..`p6` — existing InSAR stages

The execution model becomes:

```text
check -> prep -> p0 -> p1 -> p2 -> p3 -> p4 -> p5 -> p6
```

Each stage writes a stable directory under `output_dir/work/` containing:

- stage outputs
- `stage.json`
- a success marker
- references to upstream stages

All execution modes (`--step`, range, `--resume`) are mapped onto this shared stage graph.

## CLI Changes

### New execution controls

Add three mutually exclusive execution modes:

```bash
--step check|prep|p0|p1|p2|p3|p4|p5|p6
--start-step check|prep|p0|p1|p2|p3|p4|p5|p6 --end-step check|prep|p0|p1|p2|p3|p4|p5|p6
--resume
```

Rules:

- `--step` runs exactly one stage
- `--start-step/--end-step` runs a closed interval
- `--resume` starts from the first incomplete stage based on cache state
- if none of the above are set, default behavior remains "full pipeline from check to p6"

### New crop controls

Add two mutually exclusive crop arguments:

```bash
--bbox min_lon min_lat max_lon max_lat
--window row0 col0 rows cols
```

Rules:

- `--window` is always interpreted in **master image pixel coordinates**
- `--bbox` is converted to a master-referenced processing window during `check`
- if neither is supplied, full scene is processed

### New validation controls

Add validation-oriented arguments:

```bash
--dc-policy auto|zero|strict
--prf-policy auto|strict
--skip-precheck
```

Recommended defaults:

- `--dc-policy auto`
- `--prf-policy auto`
- `--skip-precheck` absent by default

Meaning:

- `auto`: attempt safe normalization / preprocessing when possible
- `strict`: mismatch becomes fatal
- `zero`: force zero-Doppler geometry path where supported


## Stage Cache Layout

Under `output_dir/work/` create stable stage directories:

```text
work/
  check/
    stage.json
    precheck.json
    crop.json
    SUCCESS
  prep/
    stage.json
    preprocess_plan.json
    normalized_slave_manifest.json
    SUCCESS
  p0_geo2rdr/
  p1_dense_match/
  p2_crossmul/
  p3_unwrap/
  p4_geocode/
  p5_hdf/
  p6_publish/
```

Each `stage.json` must record:

- stage name
- input manifests
- effective crop
- backend used
- upstream stage dependencies
- key output file paths
- start/end timestamps
- success status
- relevant fallback reason if any

Resume logic uses `stage.json` + `SUCCESS` markers, not filename heuristics.

## Stage Semantics

### check

Purpose:

- load master/slave manifests and metadata
- validate compatibility
- resolve effective crop
- choose preprocessing strategy

Outputs:

- `precheck.json`
- `crop.json`
- normalized step plan embedded in `stage.json`

### prep

Purpose:

- normalize slave sampling when needed
- choose zero-Doppler vs explicit Doppler for geometry path
- produce a normalized secondary input manifest for downstream stages

Outputs:

- `preprocess_plan.json`
- `normalized_slave_manifest.json`
- optional normalized/resampled slave raster metadata

### p0..p6

These keep the existing logical stage boundaries, but now consume stage-cache inputs instead of assuming fresh full-pipeline execution.

## Crop Design

### Internal representation

Unify crop into one internal structure:

```json
{
  "mode": "full|bbox|window",
  "master_window": {
    "row0": 0,
    "col0": 0,
    "rows": 14580,
    "cols": 12544
  },
  "bbox": [min_lon, min_lat, max_lon, max_lat]
}
```

`check` resolves user input to `master_window`. All later stages use `master_window` only.

### `--window`

Directly sets `master_window`.

Validation:

- all four values must be non-negative integers
- `rows > 0`, `cols > 0`
- window must lie inside master radar-grid dimensions

### `--bbox`

Converted during `check` into a master-pixel processing window by intersecting bbox with scene corners / geolocation and choosing the smallest containing master window. The exact conversion may initially use corner/grid approximation; precise implementation details belong to the implementation plan, not this spec.

Validation:

- `min_lon < max_lon`
- `min_lat < max_lat`
- bbox must intersect the master scene

### Crop propagation

- master defines the reference ROI
- slave processing follows master ROI after preprocessing/normalization
- final HDF5 and `stage.json` record the effective master window and original request

## Compatibility Checks

### Required checks

The `check` stage must compute and classify:

1. **center frequency compatibility**
2. **PRF compatibility**
3. **Doppler/DC compatibility**
4. **look direction compatibility**
5. **polarization compatibility**
6. **radar-grid shape / spacing compatibility**
7. **time overlap / scene overlap sanity**

### Classification model

Each check returns one of:

- `ok`
- `warn`
- `fatal`

Overall precheck result is the max severity.

### Examples

- polarization mismatch -> `fatal`
- look direction mismatch -> `fatal`
- center frequency materially different -> `fatal`
- small PRF mismatch with known normalization path -> `warn`
- Doppler metadata unusable but zero-Doppler path allowed -> `warn`

## DC / PRF Handling Policy

### Guiding principle

Follow the same broad strategy as ISCE2 stripmap processing:

- keep **master** as the reference geometry and sampling definition
- **normalize / resample the slave** to match the reference where possible
- use zero-Doppler geometry where that is the safer, already validated path

This matches observed ISCE2 patterns in stripmap and stack processing:

- normalize secondary sampling
- resample secondary SLC to reference grid
- carry Doppler information into resampling when needed

### DC / Doppler policy

#### Geometry stages (`check`, `prep`, `p0`, HDF geolocation append)

Default to **zero-Doppler geometry** under `--dc-policy auto`.

Reason:

- current repo evidence shows topo/geo2rdr workflows are more stable with zero-Doppler LUT in this environment
- current `construct_doppler_lut2d()` is not a general 2D geometry-safe Doppler representation for topo use

#### Resampling-sensitive stages

If slave preprocessing requires explicit resampling, allow a later implementation to inject Doppler/carrier terms in the resampling stage, following ISCE2 `resamp_slc` style instead of forcing them through geometry stages.

### PRF policy

#### `strict`

Any PRF mismatch outside a very small tolerance is fatal.

#### `auto`

PRF mismatch is classified as:

- `ok`: negligible, no action
- `warn`: normalize/resample slave to master
- `fatal`: mismatch too large or required metadata absent

### Preprocessing outputs

When preprocessing is needed, `prep` writes:

- normalized slave manifest
- metadata describing original and normalized PRF/DC interpretation
- any generated intermediate raster references

The downstream pipeline then treats this normalized slave manifest as the slave input.

## Failure and Resume Semantics

### Current problem

Today, any GPU exception causes `_process_insar_cpu()` to restart from P0.

### New behavior

Under the redesigned stage model:

- stage-local failures only invalidate the current stage
- earlier successful stages remain reusable
- `--resume` continues from the first incomplete stage
- `--step` and `--start-step/--end-step` can rerun targeted sections without touching earlier stages

### Backend fallback semantics

For a stage that supports GPU and CPU:

- GPU attempt may fail
- fallback to CPU reruns **that stage only**, not the whole pipeline
- `stage.json` records both attempted backend and final backend

## Output Metadata Changes

The final HDF5 should include additional attributes/groups capturing:

- requested execution mode
- effective execution mode
- crop request
- effective master window
- precheck severity summary
- preprocessing decisions
- PRF/DC handling policy used

## File Responsibilities

### `scripts/strip_insar.py`

Owns:

- CLI parsing
- stage planning
- stage cache orchestration
- step/range/resume execution
- high-level precheck/prep dispatch

### `scripts/common_processing.py`

Owns shared low-level helpers only, such as:

- geocoding accumulation helpers
- coordinate appending helpers
- raster/grid utility functions

It should not own strip_insar-specific orchestration state.

### New helper modules (recommended)

To keep `strip_insar.py` from growing further, split new logic into focused helpers:

- `scripts/insar_stage_cache.py`
- `scripts/insar_precheck.py`
- `scripts/insar_preprocess.py`
- `scripts/insar_crop.py`

This is a recommendation, not a hard requirement, but the new logic should not be added as another large monolithic block if avoidable.

## Open Decisions Locked by This Spec

The following decisions are fixed by this design:

1. Support all three execution modes: single-step, step-range, resume
2. Use `--bbox` and `--window` as the only crop CLI
3. Interpret `--window` in master image pixel coordinates
4. Run compatibility checks before P0
5. Treat master as the reference scene
6. For PRF/DC mismatch, prefer ISCE2-style slave normalization/resampling over immediate failure when safe
7. Prefer zero-Doppler geometry path under `dc-policy=auto`

## Risks

- Step caching introduces more state to validate and clean
- Crop-to-master-window conversion from bbox may need iteration for accuracy on edge cases
- Slave normalization/resampling is the highest-risk implementation area and must be introduced with explicit tests
- A partial redesign that adds steps without cache contracts will not solve the restart problem

## Acceptance Criteria

This redesign is complete when:

1. `--step`, `--start-step/--end-step`, and `--resume` all work from a shared stage cache
2. A failure in P6 does not trigger a full rerun from P0
3. `--bbox` and `--window` both reduce processing extent correctly
4. `check` reports PRF/DC compatibility before expensive processing starts
5. A known PRF/DC warning case can continue through `prep` instead of failing immediately
6. Final metadata records crop and preprocessing decisions
