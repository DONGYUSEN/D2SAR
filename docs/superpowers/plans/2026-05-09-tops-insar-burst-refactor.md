# tops_insar Burst-Native ISCE3 Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `tops_insar` into a Sentinel-1 TOPS InSAR processor whose primary processing unit is the burst, whose registration/resampling/interferogram algorithms use ISCE3 native lower-level primitives, and whose flow is scientifically comparable to ISCE2 `topsApp`.

**Architecture:** `tops_insar.py` remains the CLI/orchestrator. All Sentinel-1 TOPS logic lives in new `scripts/tops_*` modules. `tops_insar` and all new TOPS modules must not import, call, wrap, copy helpers from, or execute `scripts/strip_insar.py` or `scripts/strip_insar.py`. High-level `nisar.workflows.*.run(cfg)` wrappers are not accepted as the algorithm implementation unless a task explicitly proves their Sentinel-1 burst compatibility; implementation should prefer lower-level ISCE3 primitives such as `isce3.geometry.Geo2Rdr`, `isce3.image.ResampSlc`, `isce3.image.v2.resample_slc_blocks`, `isce3.matchtemplate.PyCPUAmpcor` / `isce3.cuda.matchtemplate.PyCuAmpcor`, `isce3.math.offsets_polyfit`, and `isce3.signal.Crossmul`.

**Tech Stack:** Python, NumPy, GDAL/Raster IO already used in D2SAR, vendored ISCE3 lower-level APIs, pytest. Reference process: `/home/ysdong/Software/isce/isce2/applications/topsApp.py` and `/home/ysdong/Software/isce/isce2/components/isceobj/TopsProc/*.py`.

---

## 0. Non-Negotiable Constraints

1. **Burst is the fundamental unit.** Sentinel-1 TOPS processing must be modeled as `swath -> common burst pair -> overlap pair -> burst product -> merged product`.
2. **No strip backend dependency.** `tops_insar` and all `tops_*` modules must have zero imports/references to `strip_insar` and `strip_insar` except in documentation text and tests that assert absence.
3. **No empty shell algorithms.** Do not add modules that only call `pass`, return the input unchanged, raise “not implemented” for planned production paths, or merely wrap high-level NISAR workflow calls without proving Sentinel-1 burst compatibility.
4. **No silent physical simplification.** Deramp/reramp, ESD frequency separation, overlap slicing, common burst matching, and valid-window merge must be implemented or the relevant pipeline stage must remain blocked by a failing test.
5. **ISCE2 parity is the reference.** Each major algorithmic milestone must compare its intermediate products against ISCE2 `topsApp` on at least one Sentinel-1 pair.
6. **Lower-level ISCE3 primitives first.** The implementation should construct Sentinel-1 burst-level radar grids, orbit, Doppler LUTs, rasters, offsets, and products explicitly, then call ISCE3 primitives. NISAR workflow wrappers may be used only after a parity test proves they operate correctly on Sentinel-1 burst inputs.

---

## 1. Target Algorithmic Flow

The target flow is intentionally close to ISCE2 `topsApp`:

```text
preprocess / import
→ compute_common_bursts
→ build_burst_radar_grids
→ topo / geo2rdr per common burst
→ subset_overlaps
→ coarse_offsets
→ coarse_resamp
→ overlap_ifg
→ prep_esd
→ esd
→ range_coreg
→ fine_offsets
→ fine_resamp
→ burst_ifg
→ merge_bursts
→ optional_ionosphere_products
→ filter / unwrap / geocode / publish
```

Mapping to ISCE2 concepts:

| This plan | ISCE2 topsApp analogue | Required behavior |
| --- | --- | --- |
| `compute_common_bursts` | `runComputeBaseline.py`, `getCommonBurstLimits()` | Find continuous common burst span, not ordinal truncation. |
| `subset_overlaps` | `runSubsetOverlaps.py` | Materialize top/bottom overlap SLC windows and metadata. |
| `geo2rdr` | `runTopo.py`, `runCoarseOffsets.py`, `runFineOffsets.py` | Use burst radar grids and orbit/doppler to compute offsets. |
| `coarse_resamp`, `fine_resamp` | `runCoarseResamp.py`, `runFineResamp.py` | Resample secondary burst while preserving TOPS carrier semantics. |
| `prep_esd`, `esd` | `runPrepESD.py`, `runESD.py` | Build overlap IFG/frequency/coherence samples and estimate secondary timing correction. |
| `range_coreg` | `runRangeCoreg.py` | Estimate range residual correction from valid overlap/common-burst samples and inject it into fine range offsets. |
| `merge_bursts` | `runMergeBursts.py` | Valid-window-aware merge with seam diagnostics. |
| `optional_ionosphere_products` | `runIon.py` | Optional split-band burst/overlap processing path; disabled unless fully validated. |

---

## 2. Files and Responsibilities

### Existing files to modify

- `scripts/tops_insar.py`
  - CLI and orchestration only.
  - Remove all `strip_insar` / `strip_insar` dependencies.
  - Route stages to TOPS-native modules.

- `scripts/sentinel_importer.py`
  - Export complete Sentinel-1 TOPS burst metadata needed by ISCE3 primitives and TOPS algorithms.

- `tests/test_tops_insar_backend.py`
  - Update orchestration tests to expect TOPS-native backend calls, not `strip_insar`.

### New modules

- `scripts/tops_burst_model.py`
  - Dataclasses for burst identity, windows, radar grid metadata, overlap slices, common burst pair, timing correction, merge segment.

- `scripts/tops_no_strip_guard.py`
  - Optional runtime guard utilities used by tests to poison forbidden imports.

- `scripts/tops_common_bursts.py`
  - Global-offset continuous-span common burst matching.

- `scripts/tops_s1_metadata.py`
  - Sentinel-1 annotation normalization: sensing times, Doppler, azimuth FM-rate, steering rate, range timing, orbit references.

- `scripts/tops_isce3_geometry.py`
  - Construct ISCE3 `RadarGridParameters`, `Orbit`, Doppler LUT, and raster adapters for one burst/overlap.

- `scripts/tops_overlap.py`
  - Materialize top/bottom reference/secondary overlap windows and metadata.

- `scripts/tops_deramp.py`
  - Sentinel-1 TOPS deramp/reramp phase model.

- `scripts/tops_registration.py`
  - Coarse/fine registration, range coregistration, and timing/range correction orchestration using ISCE3 lower-level primitives.

- `scripts/tops_insar_utils.py`
  - TOPS-specific utility algorithms factored out of `tops_insar.py`: baseline diagnostics, range-coreg helpers, ionosphere helper structures, multilook/window utilities, valid-window math, polynomial evaluation, JSON diagnostics, and ISCE2-comparison helpers.
  - This file may reimplement concepts inspired by ISCE2 `TopsProc` files such as `runRangeCoreg.py` and `runIon.py`, but must not import ISCE2 or strip backend code.
  - Utilities must be concrete, tested algorithms; no empty wrappers or placeholder helpers.

- `scripts/tops_esd.py`
  - ESD prep, frequency separation raster, sample extraction, robust statistics, timing correction.

- `scripts/tops_merge.py`
  - Valid-window-aware burst merge implementation.

### Tests to create

- `tests/test_tops_no_strip_dependency.py`
- `tests/test_tops_s1_metadata.py`
- `tests/test_tops_common_bursts.py`
- `tests/test_tops_isce3_geometry.py`
- `tests/test_tops_overlap.py`
- `tests/test_tops_deramp.py`
- `tests/test_tops_registration.py`
- `tests/test_tops_insar_utils.py`
- `tests/test_tops_esd.py`
- `tests/test_tops_merge.py`
- `tests/test_tops_ion.py`
- `tests/test_tops_isce2_parity.py`

---

## 3. Implementation Tasks

### Task 0: Enforce no dependency on `strip_insar` or `strip_insar`

**Purpose:** Block any implementation path that reuses strip backends.

**Files:**
- Create: `tests/test_tops_no_strip_dependency.py`
- Modify: `scripts/tops_insar.py`

**Required implementation:**

- Add AST-based import checks for `tops_insar.py` and all `scripts/tops_*.py` modules.
- Add source checks that production TOPS modules do not reference `strip_insar` / `strip_insar` names.
- Add runtime poison-import test:

```python
import sys

class PoisonModule:
    def __getattr__(self, name):
        raise AssertionError(f"Forbidden strip backend accessed: {name}")

sys.modules["strip_insar"] = PoisonModule()
sys.modules["strip_insar"] = PoisonModule()
sys.modules["scripts.strip_insar"] = PoisonModule()
sys.modules["scripts.strip_insar"] = PoisonModule()
```

**Acceptance:**

- `pytest tests/test_tops_no_strip_dependency.py -v` passes.
- `tops_insar.py` imports only TOPS-native modules for TOPS execution.
- There is no transitional call to `strip_insar.process_strip_insar()`.

---

### Task 1: Normalize Sentinel-1 TOPS metadata completely

**Purpose:** Provide the physical inputs required by burst matching, ISCE3 geometry, deramp/reramp, ESD, and merge.

**Files:**
- Modify: `scripts/sentinel_importer.py`
- Create: `scripts/tops_s1_metadata.py`
- Create: `tests/test_tops_s1_metadata.py`

**Required metadata per burst:**

```text
swath
polarization
burstIndex
sensingStartUTC
sensingStopUTC
sensingMidUTC
lineOffset
numberOfLines
numberOfSamples
firstValidLine
numValidLines
firstValidSample
numValidSamples
startingRange
rangePixelSpacing
azimuthTimeInterval
radarWavelength
passDirection
orbitDirection
azimuthSteeringRate
dopplerCentroidPolynomial
azimuthFmRatePolynomial
```

**Algorithm requirements:**

- Parse all times into timezone-aware UTC `datetime` objects internally.
- Serialize times as ISO-8601 strings with `Z` suffix.
- Normalize old keys such as `index` into `burstIndex`.
- Reject bursts missing sensing start/stop, valid line/sample windows, range spacing, azimuth interval, Doppler, or FM-rate.
- Validate that `numValidLines > 0` and `numValidSamples > 0`.

**Acceptance:**

- Unit tests prove key normalization from current importer output.
- Unit tests fail on missing Doppler/FM-rate or invalid valid windows.
- Real Sentinel-1 annotation extraction produces all required fields for IW1/IW2/IW3 when available.

---

### Task 2: Add complete burst data model

**Purpose:** Represent common bursts, overlap slices, radar grids, timing corrections, and merge segments without relying on strip data structures.

**Files:**
- Create: `scripts/tops_burst_model.py`
- Create: `tests/test_tops_common_bursts.py`

**Required types:**

```text
BurstIdentity
BurstWindow
BurstRecord
BurstRadarGrid
CommonBurstPair
CommonBurstSelection
OverlapSlice
OverlapPair
TimingCorrection
MergeSegment
```

**Required computed properties:**

- absolute valid line/sample start/stop;
- burst duration seconds;
- sensing midpoint;
- local-to-full-image line mapping;
- range coordinate for sample index;
- azimuth time for line index;
- overlap duration seconds.

**Acceptance:**

- Tests verify absolute valid windows.
- Tests verify azimuth time mapping from line index.
- Tests verify range coordinate mapping from sample index.

---

### Task 3: Implement ISCE2-like common burst matching

**Purpose:** Replace ordinal or nearest-neighbor pairing with continuous common span matching equivalent in semantics to ISCE2 `getCommonBurstLimits()`.

**Files:**
- Create: `scripts/tops_common_bursts.py`
- Modify: `tests/test_tops_common_bursts.py`

**Algorithm:**

1. Group bursts by swath and polarization.
2. For each candidate integer offset `k`, pair `reference[i]` with `secondary[i + k]`.
3. A candidate pair is valid only if:
   - swath and polarization match;
   - sensing start difference <= tolerance;
   - sensing stop difference <= tolerance;
   - azimuth time interval difference <= tolerance;
   - burst duration difference <= tolerance;
   - both bursts have non-empty valid windows.
4. For each `k`, find the longest continuous valid span.
5. Choose the span with maximum number of common bursts; break ties by smallest median sensing time error.
6. Output:
   - `reference_start_index`
   - `secondary_start_index`
   - `number_of_common_bursts`
   - `burst_offset`
   - ordered `CommonBurstPair` list
   - diagnostics with rejected offsets and reasons.
7. If fewer than one common burst exists, raise a strict error.
8. If fewer than two common bursts exist, allow burst IFG but mark ESD unavailable.

**Acceptance:**

- Tests cover equal starts, secondary missing first burst, reference missing first burst, partial swath, non-continuous burst gaps.
- On one real Sentinel-1 pair, common burst start/count matches ISCE2 `topsApp` within exact index equality.
- `common_bursts.json` is written per swath.

---

### Task 3A: Add `tops_insar_utils.py` for TOPS-specific shared algorithms

**Purpose:** Provide a local, tested utility layer for TOPS algorithms that would otherwise tempt direct reuse of external workflow code such as ISCE2 `runRangeCoreg.py` or `runIon.py`. This file is an adjunct to `tops_insar`, not a dumping ground and not a strip backend replacement.

**Files:**
- Create: `scripts/tops_insar_utils.py`
- Create: `tests/test_tops_insar_utils.py`

**Allowed responsibilities:**

- valid-window and overlap window math;
- multilook shape/window adjustment;
- burst baseline diagnostic calculations;
- range-coreg sample filtering and robust statistics;
- ionosphere split-band metadata helpers and burst/overlap bookkeeping;
- polynomial evaluation for Doppler/FM-rate/carrier helper paths;
- JSON diagnostics serialization;
- ISCE2 parity report helpers.

**Disallowed responsibilities:**

- no direct import or execution of ISCE2 `TopsProc` modules;
- no direct import or execution of `strip_insar` / `strip_insar`;
- no placeholder functions;
- no functions that return input unchanged unless the function is explicitly a pure accessor tested as such;
- no broad “misc” helpers without tests.

**Required initial algorithms:**

1. `intersect_windows(a, b)` returning a real intersection or an explicit empty result.
2. `adjust_window_for_looks(window, range_looks, azimuth_looks)` matching the semantics needed by burst merge.
3. `robust_median_with_mad(values, mask)` for ESD/range-coreg diagnostics.
4. `evaluate_polynomial(coefficients, x)` with tests for Sentinel-style coefficient order.
5. `write_json_diagnostic(path, payload)` with stable key ordering.

**Acceptance:**

- Unit tests cover every utility function.
- `tests/test_tops_no_strip_dependency.py` includes `scripts/tops_insar_utils.py`.
- Any later helper added to `tops_insar_utils.py` must have a dedicated test.

---

### Task 4: Construct Sentinel-1 burst-level ISCE3 geometry inputs

**Purpose:** Convert one Sentinel-1 burst or overlap into the lower-level objects required by ISCE3 primitives.

**Files:**
- Create: `scripts/tops_isce3_geometry.py`
- Create: `tests/test_tops_isce3_geometry.py`

**Algorithm requirements:**

- Build `isce3.product.RadarGridParameters` for a burst using:
  - sensing start;
  - wavelength;
  - PRF derived from `azimuthTimeInterval`;
  - starting range;
  - range pixel spacing;
  - look side;
  - length and width.
- Build burst-local grids for:
  - full burst image window;
  - valid burst window;
  - overlap slices.
- Convert Sentinel-1 orbit metadata into `isce3.core.Orbit` or the equivalent object already used elsewhere in D2SAR/ISCE3.
- Convert Doppler centroid polynomial into an `isce3.core.LUT2d` or compatible Doppler model.
- Create raster adapters for burst windows inside measurement TIFF without copying full swath data.

**Acceptance:**

- Synthetic tests verify grid dimensions, PRF, sensing start, starting range, and line/sample coordinate mapping.
- A real Sentinel-1 burst can produce a valid ISCE3 radar grid object.
- No high-level NISAR `run(cfg)` workflow is required for this task.

---

### Task 5: Implement single-burst Geo2Rdr using ISCE3 lower-level primitive

**Purpose:** Prove and implement coarse geometry offsets for Sentinel-1 burst pairs.

**Files:**
- Modify: `scripts/tops_registration.py`
- Modify: `scripts/tops_isce3_geometry.py`
- Create: `tests/test_tops_registration.py`

**Algorithm:**

1. For each `CommonBurstPair`, construct reference burst radar grid.
2. Construct secondary orbit and Doppler model.
3. Use DEM raster and ISCE3 `Geo2Rdr` to generate burst-local `range.off` and `azimuth.off`.
4. Write offsets in burst-local coordinates.
5. Save diagnostics:
   - median range offset;
   - median azimuth offset;
   - valid sample count;
   - min/max offsets;
   - convergence/failure counts if exposed by ISCE3.

**Acceptance:**

- Synthetic/small fixture test verifies offset raster dimensions.
- Real Sentinel-1 single-burst prototype produces finite offsets over the valid window.
- Compare median offsets with ISCE2 `runCoarseOffsets.py` output for the same burst; document tolerance.
- This task must call `isce3.geometry.Geo2Rdr` or `isce3.cuda.geometry.Geo2Rdr` directly, not only `nisar.workflows.geo2rdr.run(cfg)`.

---

### Task 6: Materialize top/bottom overlap products

**Purpose:** Implement `subset_overlaps` equivalent to ISCE2, not just overlap JSON.

**Files:**
- Create: `scripts/tops_overlap.py`
- Create: `tests/test_tops_overlap.py`

**Algorithm:**

For each adjacent common burst pair `(top, bottom)`:

1. Compute overlap sensing interval:
   - `overlap_start = max(top.sensing_start, bottom.sensing_start)`;
   - `overlap_stop = min(top.sensing_stop, bottom.sensing_stop)`.
2. Convert overlap start/stop to burst-local line ranges separately for:
   - reference top;
   - reference bottom;
   - secondary top;
   - secondary bottom.
3. Intersect each local line range with each burst valid line range.
4. Intersect sample windows with valid sample windows.
5. Materialize four overlap SLC windows:
   - reference top overlap;
   - reference bottom overlap;
   - secondary top overlap;
   - secondary bottom overlap.
6. Create overlap radar grid metadata for each materialized product.
7. Write `overlaps.json` with top/bottom slice coordinates and product paths.

**Acceptance:**

- Unit tests verify line conversion from sensing time to local line index.
- Unit tests verify top and bottom slices are not assumed identical.
- Real pair overlap dimensions match ISCE2 `runSubsetOverlaps.py` for at least one swath/burst pair.
- Empty or too-small overlaps fail strict mode with diagnostic reason.

---

### Task 7: Implement Sentinel-1 TOPS deramp/reramp phase model

**Purpose:** Remove the most important physical gap before resampling, ESD, and burst merge.

**Files:**
- Create: `scripts/tops_deramp.py`
- Create: `tests/test_tops_deramp.py`

**Algorithm requirements:**

- Use burst-local azimuth time grid and range grid.
- Evaluate Doppler centroid and azimuth FM-rate polynomials from Sentinel-1 annotation on the burst grid.
- Compute deramp phase using the Sentinel-1 TOPS azimuth carrier model consistent with ISCE2 carrier handling.
- Implement:
  - `compute_tops_carrier_phase(burst, lines, samples)`;
  - `deramp_slc(slc, burst_metadata)`;
  - `reramp_slc(slc, burst_metadata)`.
- Preserve dtype and shape.
- No production path may use identity deramp/reramp.

**Acceptance:**

- Unit test: deramp followed by reramp returns original SLC within numerical tolerance.
- Unit test: carrier phase varies along azimuth for nonzero FM-rate/steering metadata.
- Real Sentinel-1 burst test: deramped phase/spectrum behavior is stable and documented.
- Compare carrier phase trend against ISCE2 `estimateAzimuthCarrierPolynomials` / `runFineResamp.py` output for one burst.

---

### Task 8: Implement TOPS-aware coarse resampling using ISCE3 ResampSlc

**Purpose:** Resample secondary burst/overlap products to reference burst/overlap grids while preserving TOPS carrier semantics.

**Files:**
- Modify: `scripts/tops_registration.py`
- Modify: `scripts/tops_deramp.py`
- Create/extend: `tests/test_tops_registration.py`

**Algorithm:**

1. Deramp secondary burst/overlap SLC using `tops_deramp.deramp_slc`.
2. Use ISCE3 `ResampSlc` or `image.v2.resample_slc_blocks` with Geo2Rdr offsets.
3. Reramp the resampled SLC onto the reference burst grid.
4. Adjust output valid lines/samples according to offset extrema and interpolation chip margin.
5. Save `coarse_coregistered_secondary.slc` and metadata.

**Acceptance:**

- Test verifies no identity deramp/reramp is used.
- Single-burst real data test produces output with reference burst dimensions.
- Cross-correlation between reference and coarse-resampled secondary improves compared with unresampled secondary.
- Compare output dimensions and valid window adjustment with ISCE2 `runCoarseResamp.py`.

---

### Task 9: Implement dense offsets and rubbersheet using ISCE3 lower-level algorithms

**Purpose:** Estimate residual offsets for fine resampling without reusing strip code.

**Files:**
- Modify: `scripts/tops_registration.py`
- Create/extend: `tests/test_tops_registration.py`

**Algorithm:**

1. Run ISCE3 CPU/GPU ampcor on deramped/coarse-resampled burst or overlap products.
2. Use gross offsets from Geo2Rdr.
3. Apply SNR/covariance/correlation peak filtering.
4. Fit or interpolate residual offsets with ISCE3 `offsets_polyfit` / rubbersheet-style filtering.
5. Produce burst-local fine `range.off` and `azimuth.off` rasters.
6. Save residual diagnostics and outlier masks.

**Acceptance:**

- Synthetic offset test recovers known range/azimuth shift within tolerance.
- Real burst residual offsets are finite over enough valid samples.
- Outlier filtering removes low-SNR samples.
- No `insar_registration`, `strip_insar`, or `strip_insar` helpers are used.

---

### Task 9A: Implement TOPS range coregistration utilities and correction injection

**Purpose:** Cover the ISCE2 `runRangeCoreg.py` role without importing external workflow code. Estimate range residuals from overlap/common-burst measurements and inject them into fine resampling offsets.

**Files:**
- Modify: `scripts/tops_registration.py`
- Modify: `scripts/tops_insar_utils.py`
- Create/extend: `tests/test_tops_registration.py`
- Create/extend: `tests/test_tops_insar_utils.py`

**Algorithm:**

1. Build candidate range-coreg chips from overlap products or high-coherence common-burst valid windows.
2. Use ISCE3 `PyCPUAmpcor` / `PyCuAmpcor` in range-sensitive configuration to estimate residual range offsets.
3. Filter samples by SNR, covariance, correlation peak, coherence, and valid overlap masks.
4. Use `tops_insar_utils.robust_median_with_mad` to estimate swath-level range correction and reject outliers.
5. Convert range correction into fine range-offset raster units using documented sign convention.
6. Inject range correction into fine resampling offsets before final `ResampSlc`.
7. Write `range_coreg_summary.json` with sample count, median, MAD, rejected count, and per-overlap contributions.

**Acceptance:**

- Synthetic range shift test recovers known shift within tolerance.
- Real overlap/common-burst test produces finite range correction when enough valid samples exist.
- Sign convention is validated by showing improved range alignment after correction.
- Compare correction magnitude with ISCE2 `runRangeCoreg.py` on one pair where range coreg is enabled.
- No ISCE2, `strip_insar`, `strip_insar`, or `insar_registration` helper is imported.

---

### Task 10: Implement overlap IFG and ESD prep

**Purpose:** Implement the full preparation stage required by ESD, not only the final median estimator.

**Files:**
- Create: `scripts/tops_esd.py`
- Create: `tests/test_tops_esd.py`

**Algorithm:**

For each overlap pair:

1. Coarse-resample secondary top and bottom overlap products.
2. Generate top overlap IFG and bottom overlap IFG with ISCE3 `Crossmul`.
3. Compute double-difference overlap phase used for ESD.
4. Generate coherence raster.
5. Generate ESD frequency separation raster from the TOPS carrier / Doppler / FM-rate model.
6. Multilook phase, frequency, and coherence consistently.
7. Extract valid ESD samples using:
   - finite phase;
   - finite nonzero frequency;
   - coherence threshold;
   - valid overlap mask.
8. Write per-overlap ESD sample diagnostics.

**Acceptance:**

- Unit tests verify phase/frequency/coherence masks.
- Synthetic test with known frequency and phase recovers known azimuth offset.
- Real overlap test produces non-empty ESD samples for a valid pair.
- Frequency raster shape and sign are documented and compared with ISCE2 `runPrepESD.py` output.

---

### Task 11: Implement ESD timing correction and fine resampling

**Purpose:** Turn ESD estimates into secondary timing correction and apply it to final burst resampling.

**Files:**
- Modify: `scripts/tops_esd.py`
- Modify: `scripts/tops_registration.py`
- Extend: `tests/test_tops_esd.py`

**Algorithm:**

1. Concatenate valid ESD samples from all overlap pairs in a swath.
2. Estimate azimuth offset pixels using robust statistics:
   - median;
   - mean;
   - std;
   - sample count;
   - per-overlap contribution.
3. Support `extra_esd_cycles` as ISCE2 does.
4. Convert median azimuth pixel offset into secondary timing correction using azimuth time interval.
5. Add timing correction to fine azimuth offsets.
6. Fine-resample full secondary bursts with deramp/resamp/reramp.
7. Write `esd_summary.json`.

**Acceptance:**

- Synthetic ESD test recovers known azimuth offset.
- Real swath ESD median/mean/std compared with ISCE2 `runESD.py` for the same pair.
- Applying ESD correction reduces overlap phase discontinuity or seam metric compared with no ESD.

---

### Task 12: Generate per-burst interferograms using ISCE3 Crossmul

**Purpose:** Produce final per-burst IFG/coherence after fine resampling.

**Files:**
- Modify: `scripts/tops_registration.py`
- Create/extend: `tests/test_tops_registration.py`

**Algorithm:**

1. Use reference burst SLC valid window.
2. Use fine-coregistered secondary burst SLC.
3. Run ISCE3 `Crossmul` for wrapped IFG and coherence.
4. Apply valid masks and looks consistently.
5. Save per-burst IFG/coherence metadata including valid line/sample window.

**Acceptance:**

- Synthetic complex SLC test produces expected phase difference.
- Real burst IFG dimensions match reference valid window or documented looks-adjusted size.
- Coherence values are finite and in `[0, 1]`.

---

### Task 13: Implement valid-window-aware burst merge

**Purpose:** Replace segment-only planning with real merge behavior comparable to ISCE2 `runMergeBursts.py`.

**Files:**
- Create: `scripts/tops_merge.py`
- Create: `tests/test_tops_merge.py`

**Algorithm:**

1. Build merged radar grid extent from all common burst valid windows.
2. For each burst product, compute output placement from absolute valid line/sample offsets.
3. Support overlap merge policies:
   - `top`;
   - `bottom`;
   - `average_complex` for IFG;
   - `average_float` for coherence/geometry.
4. Adjust valid windows for looks.
5. Preserve masks for gaps and invalid regions.
6. Merge IFG, coherence, and optional geometry layers consistently.
7. Compute seam diagnostics:
   - overlap phase difference median/std;
   - coherence drop across seam;
   - invalid gap count;
   - number of pixels contributed by top/bottom/average.
8. Write `burst_seam_diagnostics.json`.

**Acceptance:**

- Unit tests cover non-overlap, simple overlap, gap, and average merge policies.
- Synthetic complex IFG test verifies phase-preserving average.
- Real swath merge dimensions match ISCE2 merged product within documented look/window differences.
- Seam diagnostics improve after ESD compared with before ESD.

---

### Task 13A: Add optional TOPS ionosphere utility path without blocking core InSAR

**Purpose:** Provide a concrete extension point for ISCE2 `runIon.py`-style processing without importing ISCE2 code. This is optional for the first core InSAR milestone, but if enabled it must be a real algorithmic path, not a placeholder.

**Files:**
- Modify: `scripts/tops_insar_utils.py`
- Create: `scripts/tops_ion.py`
- Create: `tests/test_tops_ion.py`
- Extend: `tests/test_tops_no_strip_dependency.py`

**Algorithmic scope when ionosphere correction is enabled:**

1. Define split-band metadata for each common burst and overlap product.
2. Create low/high subband SLC product descriptors with burst-local valid windows.
3. Reuse the same common-burst and overlap materialization logic for low/high bands.
4. Generate low/high band IFGs with ISCE3 `Crossmul`.
5. Estimate dispersive phase using documented radar-frequency relationship.
6. Filter/multilook ionosphere phase using valid masks and coherence thresholds.
7. Merge ionosphere burst products with the same valid-window-aware merge machinery.
8. Write `ionosphere_summary.json` with per-burst/per-overlap diagnostics.

**Acceptance:**

- If ionosphere correction is disabled, no ion placeholder products are written.
- If enabled, synthetic two-frequency phase test recovers the known dispersive phase.
- Real-data implementation is gated behind an explicit validation dataset and comparison with ISCE2 `runIon.py` outputs.
- No `runIon.py`, ISCE2 module, `strip_insar`, or `strip_insar` code is imported or copied.

---

### Task 14: End-to-end ISCE2 parity validation

**Purpose:** Prevent a structurally complete but physically wrong implementation.

**Files:**
- Create: `tests/test_tops_isce2_parity.py`
- Add fixture metadata paths or documented manual validation script.

**Validation dataset:**

- One Sentinel-1 pair.
- Same track/pass.
- Single swath first, preferably IW2.
- 2-3 common bursts minimum.
- VV polarization.
- DEM available.
- ISCE2 `topsApp` output available for comparison.

**Required comparisons:**

1. common burst start/count;
2. overlap line/sample ranges;
3. Geo2Rdr offset median/std;
4. coarse-resampled burst dimensions;
5. ESD median/mean/std/sample count;
6. fine-resampled burst dimensions;
7. per-burst IFG dimensions and basic phase statistics;
8. merged IFG dimensions;
9. seam diagnostics;
10. geocoded footprint if geocode stage is enabled.

**Acceptance:**

- A report `tops_isce2_parity_report.json` is generated.
- Differences are within documented tolerances or explicitly explained by known implementation differences.
- The pipeline cannot be marked complete until this validation passes for at least one real pair.

---

## 4. Revised Migration Strategy

### Phase 0: Remove forbidden dependencies

Implement Task 0 first. If this fails, no further work should proceed.

### Phase 1: Build complete burst metadata and matching foundation

Implement Tasks 1-3A. These are pure metadata, matching, and shared TOPS utility tasks and should not depend on ISCE3 execution.

### Phase 2: Prove ISCE3 lower-level primitives on Sentinel-1 bursts

Implement Tasks 4-5. This phase must prove `RadarGridParameters`, orbit, Doppler LUT, DEM, and Geo2Rdr work on one Sentinel-1 burst.

### Phase 3: Implement physical TOPS overlap and carrier handling

Implement Tasks 6-8 and Task 9A (coarse resamp, deramp/reramp, range coreg). This is the highest-risk phase because overlap slicing, carrier handling, and range coreg determine whether ESD, fine resamp, and seams are meaningful.

### Phase 4: Fine registration and ESD

Implement Tasks 9, 9A, 10, 11 (dense offsets, range coreg, ESD prep and apply). This phase must compare ESD and range correction with ISCE2 before proceeding.

### Phase 5: IFG, merge, and parity

Implement Tasks 12-14. Do not claim completion until ISCE2 parity validation passes.

---

## 5. Validation Commands

Run after metadata/matching/utility tasks:

```bash
pytest tests/test_tops_no_strip_dependency.py tests/test_tops_s1_metadata.py tests/test_tops_common_bursts.py tests/test_tops_insar_utils.py -v
```

Run after ISCE3 geometry/registration tasks:

```bash
pytest tests/test_tops_isce3_geometry.py tests/test_tops_registration.py tests/test_tops_insar_utils.py tests/test_tops_no_strip_dependency.py -v
```

Run after ESD, range coreg, and merge tasks:

```bash
pytest tests/test_tops_overlap.py tests/test_tops_deramp.py tests/test_tops_esd.py tests/test_tops_merge.py tests/test_tops_insar_utils.py -v
```

Run before declaring the refactor complete:

```bash
pytest tests/test_tops_no_strip_dependency.py tests/test_tops_s1_metadata.py tests/test_tops_common_bursts.py tests/test_tops_isce3_geometry.py tests/test_tops_insar_utils.py tests/test_tops_overlap.py tests/test_tops_deramp.py tests/test_tops_registration.py tests/test_tops_esd.py tests/test_tops_merge.py tests/test_tops_isce2_parity.py -v
python3 scripts/tops_insar.py --help
```

---

## 6. Definition of Done

The refactor is complete only when all of the following are true:

- `tops_insar` and `scripts/tops_*.py` have no dependency on `strip_insar.py` or `strip_insar.py`.
- `scripts/tops_insar_utils.py` contains concrete, tested utility algorithms and no placeholder functions.
- Sentinel-1 burst metadata contains all required timing/range/Doppler/FM-rate fields.
- common burst matching is based on global offset and continuous span, not ordinal truncation.
- burst-level ISCE3 radar grids are constructed and tested on real Sentinel-1 bursts.
- Geo2Rdr offsets are produced through lower-level ISCE3 primitives and compared with ISCE2.
- overlap top/bottom products are materialized and compared with ISCE2 overlap dimensions.
- deramp/reramp is implemented with Sentinel-1 TOPS carrier phase and validated by roundtrip and ISCE2 carrier comparison.
- coarse/fine resampling uses ISCE3 resampling primitives with deramp/reramp handling.
- dense offsets and rubbersheet residuals are computed without strip helper code.
- range coreg is implemented with ISCE3 ampcor primitives, robust filtering, and sign-validated correction injection.
- ESD prep produces overlap phase, frequency, coherence, and valid samples.
- ESD timing correction is applied to fine azimuth offsets.
- per-burst IFG/coherence is produced with ISCE3 Crossmul.
- burst merge performs real valid-window-aware mosaic and writes seam diagnostics.
- `tops_insar_utils` is included in no-strip dependency checks.
- at least one real Sentinel-1 pair passes the ISCE2 parity report.

---

## 7. Explicitly Removed from the Previous Plan

The following approaches are no longer acceptable:

- An empty `tops_isce3_backend.py` that only defines an error class.
- A backend that only imports `nisar.workflows.*.run(cfg)` without proving Sentinel-1 burst compatibility.
- `allow_noop=True` deramp/reramp compatibility paths in production processing.
- ESD implemented only as `phase / frequency` without `prep_esd` frequency/coherence/sample generation.
- Overlap represented only by JSON/window intersections without materialized top/bottom overlap products.
- Merge represented only by segment planning without raster mosaic and seam diagnostics.
- Any transitional call into `strip_insar` or `strip_insar`.

---

## 8. Self-Review

### Coverage of issues found in feasibility review

- `strip_insar` / `strip_insar` dependency: Task 0 and Definition of Done.
- ISCE3/NISAR workflow mismatch: lower-level primitive requirement in Tasks 4-5, 8-12.
- common burst matching too simple: Task 3 global offset / continuous span algorithm.
- overlap not first-class: Task 6 materializes four overlap SLC windows.
- deramp/reramp missing: Task 7 implements physical carrier model and blocks identity behavior.
- ESD too shallow: Tasks 10-11 implement prep, samples, statistics, and timing correction.
- merge too shallow: Task 13 implements actual valid-window-aware raster merge.
- range coreg missing: Task 9A implements range residual estimation and correction injection.
- ionosphere / split-band missing: Task 13A provides optional concrete path.
- ISCE2 utilities reimplemented as placeholders: Task 3A (`tops_insar_utils.py`) provides tested shared algorithms without importing ISCE2.
- lack of parity proof: Task 14 requires ISCE2 comparison.

### Remaining high-risk areas

- Exact Sentinel-1 TOPS carrier phase formula must be validated against ISCE2 and annotation semantics.
- Direct construction of ISCE3 orbit/radar grid/Doppler objects from Sentinel-1 metadata must be proven on real data.
- Tolerances for parity tests must be established empirically using one small validation pair.

These are not deferred as vague future work; they are explicit blocking tasks before completion.
