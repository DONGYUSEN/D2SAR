# Strip InSAR Stage V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `strip_insar` onto the approved Stage v2 pipeline: `check -> prep -> crop -> p0 -> p1 -> p2 -> p3 -> p4 -> p5 -> p6`, with `prep` limited to slave normalization, `crop` handling subset, ISCE2-style `resamp_slc` semantics moved into `p1`, and `p2` producing a Goldstein-filtered interferogram for `p3`.

**Architecture:** Keep the existing stage-cache/resume framework, but introduce a new `crop` stage and split registration semantics cleanly: `prep` produces normalized slave inputs only, while `p1` performs coarse offsets, resampling, and fine registration. Preserve the existing Python entrypoint and tests while adding dedicated modules for crop products, registration, and interferogram filtering.

**Tech Stack:** Python, NumPy, SciPy, GDAL, existing ISCE3 helpers, source-integrated ISCE2 `resamp_slc` logic adapted into local Python modules.

---

### Task 1: Stage graph and crop refactor

**Files:**
- Create: `scripts/insar_subset.py`
- Modify: `scripts/strip_insar.py`
- Modify: `scripts/insar_stage_cache.py`
- Test: `tests/test_strip_insar_stages.py`

- [ ] Add failing tests for a dedicated `crop` stage and downstream use of cropped manifests.
- [ ] Refactor `check` to stop owning crop outputs.
- [ ] Implement `crop` as a stage that writes cropped master/slave manifests, rasters, and metadata.
- [ ] Rewire `p0-p6` to consume `crop` outputs.

### Task 2: Prep scope reduction

**Files:**
- Modify: `scripts/insar_preprocess.py`
- Modify: `scripts/strip_insar.py`
- Test: `tests/test_insar_preprocess.py`

- [ ] Add failing tests that lock `prep` to normalization-only responsibilities.
- [ ] Remove registration-like resampling semantics from `prep` metadata/contracts.
- [ ] Keep normalized slave replacement behavior intact.

### Task 3: P1 registration and resamp integration

**Files:**
- Create: `scripts/insar_registration.py`
- Modify: `scripts/strip_insar.py`
- Test: `tests/test_strip_insar_stages.py`

- [ ] Add failing tests for p1 coarse offsets, coarse-coreg slave, fine offsets, and registration fit outputs.
- [ ] Integrate ISCE2 `resamp_slc`-style source logic into local code adapted for current manifests/cache.
- [ ] Implement p1 outputs and cache contracts.

### Task 4: P2 filter pipeline

**Files:**
- Create: `scripts/insar_filtering.py`
- Modify: `scripts/strip_insar.py`
- Test: `tests/test_strip_insar.py`
- Test: `tests/test_strip_insar_stages.py`

- [ ] Add failing tests for Goldstein filter output and p3 consuming filtered interferogram.
- [ ] Implement Goldstein filter.
- [ ] Store raw and filtered interferogram products in stage cache/HDF.

### Task 5: Product metadata and verification

**Files:**
- Modify: `scripts/strip_insar.py`
- Test: `tests/test_strip_insar.py`
- Test: `tests/test_strip_insar_stages.py`

- [ ] Add failing tests for HDF metadata recording crop/prep/registration/filter semantics.
- [ ] Update HDF writing and publish stages.
- [ ] Run unit tests and syntax verification.
