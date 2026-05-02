# TOPS RTC Factor Apply Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply computed RTC area factors to Sentinel TOPS burst amplitude products and export RTC-corrected geocoded previews.

**Architecture:** Extend `scripts/tops_rtc.py` minimally. Keep existing `amplitude_fullres.h5` materialization and topo products unchanged, add a separate `amplitude_rtc.h5` product and optional RTC-specific GeoTIFF/PNG export path.

**Tech Stack:** Python stdlib, NumPy, h5py, existing GDAL/ISCE3 helper functions in `scripts/common_processing.py`, pytest/unittest, Docker `d2sar:cuda` for real topo/RTC execution.

---

### Task 1: Apply RTC Factor To Burst HDF

**Files:**
- Modify: `scripts/tops_rtc.py`
- Modify: `tests/test_tops_rtc.py`

- [ ] Add a failing test that writes a tiny `amplitude_fullres.h5` with `slc_amplitude`, `valid_mask`, `longitude`, `latitude`, `height`, and a separate `rtc_factor.h5`, then expects `apply_burst_rtc_factor(...)` to write `amplitude_rtc.h5` with `rtc_amplitude = slc_amplitude / sqrt(rtc_factor)`.
- [ ] Implement `apply_burst_rtc_factor(amplitude_h5, rtc_factor_h5, output_h5)`.
- [ ] Preserve geolocation datasets in the RTC HDF so existing geocoding helpers can reuse the corrected product.

### Task 2: CLI Integration

**Files:**
- Modify: `scripts/tops_rtc.py`
- Modify: `tests/test_tops_rtc.py`

- [ ] Add `--apply-rtc` CLI flag.
- [ ] Make `tops_rtc.py` apply RTC for each selected burst after RTC factor exists.
- [ ] Return JSON summary with `rtc_h5` path per burst.

### Task 3: RTC Geocoded Preview

**Files:**
- Modify: `scripts/tops_rtc.py`
- Modify: `tests/test_tops_rtc.py`

- [ ] Add support for exporting geocoded preview from `amplitude_rtc.h5` instead of `amplitude_fullres.h5`.
- [ ] Use output names `amplitude_rtc_utm_geocoded.tif` and `amplitude_rtc_utm_geocoded.png`.

### Task 4: Real Data Verification

**Files:**
- Modify: `progress.md`
- Modify: `findings.md`
- Modify: `task_plan.md`

- [ ] Run unit tests and py_compile.
- [ ] Run Docker real single-burst materialize/topo/RTC factor/apply/geocode using `/home/ysdong/Temp/d2sar_real_orbit_test/import/manifest.json` and `/home/ysdong/Temp/s1/proc/dem/dem.tif`.
- [ ] Inspect output HDF, GeoTIFF, and PNG dimensions and record results.
