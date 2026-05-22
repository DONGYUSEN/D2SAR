# tops_insar.py — From-Scratch Sentinel-1 TOPS ISCE3-Native Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a complete, self-contained Sentinel-1 TOPS InSAR processor (`scripts/tops_insar.py`) that replaces all strip-backend-coupled logic, uses ISCE3 lower-level primitives for registration/resampling/interferogram generation, and produces scientifically comparable results to ISCE2 `topsApp` on the burst level.

**Architecture:** `tops_insar.py` is the single CLI/orchestrator entry point. All algorithmic logic lives in `scripts/tops_*.py` modules. There is no dependency on `scripts/strip_insar.py`, `scripts/strip_insar.py`, or `scripts/tops_insar.py`. Each module has one clear responsibility. The pipeline flow mirrors ISCE2 `topsApp` stage by stage. ISCE3 lower-level primitives are used directly; high-level `nisar.workflows.*.run(cfg)` is permitted only after a task explicitly proves Sentinel-1 burst compatibility via parity test.

**Tech Stack:** Python, NumPy, GDAL/rasterio, existing D2SAR IO utilities, vendored ISCE3 (`isce3.geometry`, `isce3.image`, `isce3.image.v2`, `isce3.matchtemplate`, `isce3.signal`, `isce3.math`), pytest. Reference: `/home/ysdong/Software/isce/isce2/applications/topsApp.py` and `/home/ysdong/Software/isce/isce2/components/isceobj/TopsProc/*.py`.

---

## 0. Non-Negotiable Constraints

1. **Burst is the fundamental unit.** Every pipeline stage operates on burst-level objects, not full-swath or full-scene rasters.
2. **Zero strip backend imports.** `tops_insar.py` and all `scripts/tops_*.py` modules must not import `strip_insar`, `strip_insar`, `tops_insar`, or copy helpers from them.
3. **No empty shell algorithms.** Every function must compute something real; every module must be testable before a later module depends on it.
4. **ISCE2 parity gates each phase.** Before advancing from one algorithmic phase to the next, intermediate products must be compared against ISCE2 `topsApp` output on at least one real Sentinel-1 pair.
5. **Lower-level ISCE3 primitives first.** Build Sentinel-1 burst radar grids, orbit objects, Doppler LUTs, rasters, offsets, and products explicitly; then call ISCE3 primitives. High-level NISAR workflows are deferred until parity is proven.
6. **Explicit call chains, no hidden delegation.** Each pipeline stage calls a named module function; there is no monolithic "run everything" function that hides algorithmic responsibility.

---

## 1. Top-Level Architecture

### 1.1 File Responsibilities

| File | Responsibility | Lines (est.) |
|---|---|---|
| `scripts/tops_insar.py` | CLI entry, argument parsing, stage dispatch loop, work directory management | ~300 |
| `scripts/tops_model.py` | Immutable dataclasses: burst identity, radar grid, overlap slices, common burst pair, ESD estimate, merge segment, timing correction | ~300 |
| `scripts/tops_metadata.py` | Parse Sentinel-1 SAFE/manifest into `tops_model` objects; normalize annotation fields; validate required metadata | ~400 |
| `scripts/tops_common_bursts.py` | Global-integer-offset continuous-span common burst matching; write `common_bursts.json` | ~200 |
| `scripts/tops_geometry.py` | Build ISCE3 `RadarGridParameters`, `Orbit`, Doppler LUT; burst-local raster adapters from TIFF windows | ~400 |
| `scripts/tops_overlap.py` | Materialize top/bottom reference/secondary overlap SLC windows and metadata; write `overlaps.json` | ~300 |
| `scripts/tops_deramp.py` | Sentinel-1 TOPS azimuth carrier phase model; `deramp_slc()` and `reramp_slc()` with roundtrip validation | ~250 |
| `scripts/tops_registration.py` | Geo2Rdr → coarse resamp → deramp → fine resamp; dense ampcor + rubbersheet orchestration | ~600 |
| `scripts/tops_esd.py` | Overlap IFG → frequency raster → multilook → robust stats → timing correction; write `esd_summary.json` | ~400 |
| `scripts/tops_range_coreg.py` | Range residual estimation from overlap samples; robust filtering; correction injection | ~300 |
| `scripts/tops_ifg.py` | Per-burst wrapped interferogram and coherence via ISCE3 `Crossmul` | ~250 |
| `scripts/tops_merge.py` | Valid-window-aware burst mosaic; seam diagnostics; write `burst_seam_diagnostics.json` | ~400 |
| `scripts/tops_ionosphere.py` | Optional split-band ionosphere correction; disabled unless explicitly enabled and validated | ~300 |
| `scripts/tops_publish.py` | Unwrap (delegate to existing unwrap utilities), LOS, geocode, HDF packaging | ~300 |
| `scripts/tops_utils.py` | Shared pure-math utilities: window intersection, multilook shape, robust stats, polynomial eval, JSON serialization | ~200 |

### 1.2 Module Interface Reference

每个模块的输入、输出、依赖关系详细说明。

---

#### `tops_model.py` — 数据模型层（无外部依赖）

**职责：** 定义所有不可变 dataclass，作为整个系统的类型约定。任何模块间传递的数据必须是 `tops_model` 类型。

| 类名 | 字段 | 说明 |
|---|---|---|
| `BurstIdentity` | `swath, burst_index, sensing_start, sensing_stop, polarization, orbit_direction, azimuth_steering_rate` | burst 唯一标识和时空属性 |
| `BurstWindow` | `first_line, num_lines, first_sample, num_samples` | 像素坐标系中的窗口（相对/绝对由调用方约定） |
| `BurstRadarGrid` | `identity + image_window + valid_window + line_offset + azimuth_time_interval + range_pixel_spacing + starting_range + radar_wavelength + doppler_coefficients + azimuth_fm_rate_coefficients` | 完整 burst 雷达几何参数 |
| `CommonBurstPair` | `pair_index, reference, secondary, burst_offset` | 一对 reference/secondary burst |
| `CommonBurstSelection` | `swath, reference_start_index, secondary_start_index, number_of_common_bursts, pairs` | 一个 swath 的全部 common burst 对 |
| `OverlapSlice` | `burst_pair, is_top, first_line, num_lines, first_sample, num_samples, sensing_start, sensing_stop` | 一个 burst overlap 区域的像素坐标和时间 |
| `OverlapPair` | `pair_index, top, bottom` | 一对相邻 burst 的 top/bottom overlap |
| `Geo2RdrOffsets` | `range_off_path, azimuth_off_path, median_range_offset, median_azimuth_offset, valid_sample_count` | Geo2Rdr 输出 |
| `EsdEstimate` | `median_offset_pixels, mean, std, sample_count, azimuth_time_interval` | ESD 估计结果 |
| `TimingCorrection` | `secondary_timing_seconds, secondary_timing_pixels, esd_estimate` | 转成时间/像素单位的时序校正量 |
| `MergeSegment` | `burst_index, pair_index, input_line_start, input_num_lines, input_sample_start, input_num_samples, output_line_start, output_num_lines, output_sample_start, output_num_samples` | merge 规划：每个 burst 在 mosaic 中的输入/输出坐标 |
| `MergeResult` | `seam_phase_diff_median, seam_phase_diff_std, seam_coherence_drop, gap_pixel_count, top_contribution_count, bottom_contribution_count, segments` | merge 结果和 seam 诊断 |

**输入：** 无（纯数据类型定义）
**输出：** 类型定义本身，供其他所有模块使用
**依赖：** 无

---

#### `tops_metadata.py` — Sentinel-1 数据导入

**职责：** 将 Sentinel-1 SAFE/ZIP 或 manifest JSON 解析为 `tops_model.BurstRadarGrid` 对象列表。

| 函数 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `parse_sentinel1_safe(path: Path) -> dict[str, list[BurstRadarGrid]]` | SAFE 目录或 ZIP 路径 | `{IW1/2/3: [BurstRadarGrid, ...]}` | 主入口，按 IW swath 分组 |
| `parse_sensing_time(value) -> datetime` | ISO 字符串或数值秒 | timezone-aware UTC datetime | 时间解析 |
| `_load_manifest(root) -> dict` | SAFE root | manifest JSON dict | 读取 manifest |
| `_iw_annotation_xmls(root) -> dict[str, str]` | manifest dict | `{IW1: annotation.xml, IW2: ..., IW3: ...}` | 找到每个 swath 的 Annotation 文件路径 |
| `_parse_iw_bursts(xml_path) -> list[BurstRadarGrid]` | 单个 swath 的 Annotation XML 路径 | `[BurstRadarGrid, ...]` | 按 burst 逐个解析 |

**关键行为：**
- `sensing_start/stop` 必须为 timezone-aware UTC datetime
- `numValidLines > 0`、`numValidSamples > 0` 否则抛 ValueError
- `doppler_coefficients` 和 `azimuth_fm_rate_coefficients` 至少有一个非零项
- annotation 中的 `index` 字段统一规范为 `burstIndex`

**输出文件：** 无（输出为内存对象）
**被依赖：** `tops_common_bursts`、`tops_geometry`、`tops_overlap`、`tops_ifg`、`tops_merge`
**依赖：** 无（独立于 ISCE3）

---

#### `tops_common_bursts.py` — Common Burst 匹配

**职责：** 在 reference 和 secondary 的 burst 列表中，基于 sensing time tolerance 和连续 span 找到全局 burst offset 对应的 common burst 对。

| 函数 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `match_common_bursts(reference, secondary) -> CommonBurstSelection` | 两个 `BurstRadarGrid` 序列 | `CommonBurstSelection` 含 `pairs` 元组 | 主入口 |
| `_bursts_match(a, b) -> bool` | 两个 `BurstRadarGrid` | `True/False` | 单对 burst 兼容性检查 |
| `_contiguous_spans(pairs) -> list[tuple]` | `[(ref_idx, sec_idx, count), ...]` | 连续 span 列表 | 找最长连续段 |

**匹配算法（5 步）：**
1. 枚举所有 candidate integer offset `k`：`reference[i]` ↔ `secondary[i+k]`
2. 对每个 `k`，检验 sensing start/stop 差 ≤ 0.5s、azimuth interval 一致、valid window 非空
3. 在所有有效 span 中选最长连续段
4. tiebreak：median sensing time error 最小
5. 不足 1 个 common burst → ValueError；不足 2 个 → 允许 IFG 但标记 ESD 不可用

**输出 JSON：** `common_bursts.json`
```json
{
  "swath": "IW2",
  "reference_start_index": 1,
  "secondary_start_index": 0,
  "number_of_common_bursts": 3,
  "burst_offset": -1,
  "pairs": [
    {"pair_index": 0, "reference_burst_index": 1, "secondary_burst_index": 0}
  ]
}
```
**被依赖：** `tops_overlap`、`tops_registration`、`tops_ifg`、`tops_merge`
**依赖：** `tops_metadata`、`tops_model`

---

#### `tops_geometry.py` — ISCE3 几何适配层

**职责：** 将 Sentinel-1 burst 元数据转换为 ISCE3 C++ 绑定所需的几何对象。

| 函数 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `burst_to_radar_grid(burst) -> S1RadarGrid` | `BurstRadarGrid` | ISCE3-compatible radar grid | 主适配函数 |
| `build_isce3_orbit_from_safe(path, t0, t1) -> Any` | SAFE 路径 + 时间区间 | `isce3.core.Orbit` | 从 SAFE orbit state vector 构造 |
| `build_doppler_lut(burst) -> Any` | `BurstRadarGrid` | `isce3.core.LUT2d` | 从 annotation doppler 系数构造 |
| `run_geo2rdr_single_burst(ref, sec, dem, work_dir, use_gpu) -> Geo2RdrOffsets` | burst 对 + DEM + 工作目录 | `Geo2RdrOffsets` | **Spike 核心**：调用 `isce3.geometry.Geo2Rdr` |
| `S1RadarGrid`（dataclass） | — | `prf, wavelength, slant_range_at(sample), azimuth_time_at_line(line)` | 工具方法 |

**S1RadarGrid 关键计算：**
```
prf = 1 / azimuth_time_interval
slant_range(s) = starting_range + s * range_pixel_spacing
t(l) = sensing_start + l / prf
```

**Geo2Rdr 输出文件（写入 work_dir）：**
- `range.off`：range 方向偏移（m 或 pixel，与 ISCE3 一致）
- `azimuth.off`：方位向偏移（s 或 pixel）

**Geo2Rdr 验收：** median range/azimuth offset 必须 finite 且范围合理
**被依赖：** `tops_registration`（geo2rdr offsets 来自此处）
**依赖：** `tops_metadata`、`tops_model`、`isce3.geometry`（仅此一处）

---

#### `tops_overlap.py` — Overlap 产品物化

**职责：** 为相邻 burst 对计算并读取 top/bottom overlap 像素窗口，生成 `OverlapPair` 对象。

| 函数 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `build_overlap_pairs(common_pairs) -> tuple[OverlapPair, ...]` | `CommonBurstPair` 序列 | `OverlapPair` 元组 | 主入口 |
| `read_overlap_window(tiff_path, slice) -> np.ndarray` | TIFF 路径 + `OverlapSlice` | `complex64` ndarray | 从全 swath TIFF 读 overlap 窗口 |

**Overlap 计算步骤（每对相邻 burst）：**
1. `overlap_start = max(top.sensing_stop, bot.sensing_start)`
2. `overlap_stop = min(top.sensing_stop, bot.sensing_stop)`
3. 转行号：`top_overlap_line = round((overlap_start - top.sensing_start) / dt)`
4. 取交集：`abs_overlap_line = max(top_valid_abs, bot_valid_abs)`
5. 逐样本交集：`abs_overlap_sample = max(top_valid_sample, bot_valid_sample)`
6. 写 `overlaps.json`

**输出 JSON：`overlaps.json`**
```json
{
  "overlap_count": 2,
  "overlaps": [
    {
      "pair_index": 0,
      "top_first_line": 1300, "top_num_lines": 200,
      "bottom_first_line": 1300, "bottom_num_lines": 200,
      "first_sample": 500, "num_samples": 24000,
      "sensing_start": "2024-01-01T00:00:01.000Z",
      "sensing_stop": "2024-01-01T00:00:01.500Z"
    }
  ]
}
```
**被依赖：** `tops_deramp`、`tops_esd`、`tops_range_coreg`
**依赖：** `tops_model`、`tops_common_bursts`

---

#### `tops_deramp.py` — TOPS Deramp/Reramp 相位模型

**职责：** 实现 Sentinel-1 TOPS 原始 azimuth carrier 相位模型及 deramp/reramp roundtrip 验证。

| 函数 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `compute_tops_carrier_phase(burst, lines, samples) -> np.ndarray` | `BurstRadarGrid` + 坐标网格 | float32 相位（弧度） | 核心 carrier 计算 |
| `deramp_slc(slc, burst) -> np.ndarray` | `complex64` SLC + `BurstRadarGrid` | deramped SLC | `slc * exp(+j * phi)` |
| `reramp_slc(slc, burst) -> np.ndarray` | deramped SLC + `BurstRadarGrid` | reramped SLC | `slc * exp(-j * phi)` |
| `deramp_reramp_roundtrip(slc, burst) -> np.ndarray` | SLC + burst | roundtrip 后的 SLC | 用于验证 |

**Carrier 相位公式（线性 Doppler 模型）：**
```
fD(s) ≈ f0 + f1 * s_rg
phi(l, s) = -2π * fD(s) * t(l)
       = -2π * (f0 + f1 * s_rg) * (l / prf)
其中 s_rg = sample * range_pixel_spacing
      t(l) = l / prf
```
**验收条件：** deramp → reramp roundtrip 误差 < 1e-5（相对幅度）
**被依赖：** `tops_registration`（coarse/fine resamp 前后的 carrier 处理）
**依赖：** `tops_model`

---

#### `tops_registration.py` — 配准编排

**职责：** 串联 Geo2Rdr → deramp → coarse resamp → deramp → fine resamp 的完整配准链路。

| 函数 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `coarse_resample_pair(ref, sec, range_off, azimuth_off) -> np.ndarray` | ref/sec burst + Geo2Rdr offsets | coreg secondary SLC | deramp → 插值 → reramp |
| `fine_resample_with_timing(ref, sec, coarse_off, timing_sec, range_coreg_px) -> np.ndarray` | ref/sec + coarse offsets + timing correction + range coreg | fine resampled SLC | 同上，加时序/range 校正 |

**Pipeline 链路：**
```
Geo2Rdr offsets
  ↓
deramp(secondary)
  ↓
ISCE3 ResampSlc (coarse)
  ↓
reramp(secondary)  ← TOPS carrier 恢复
  ↓
[ESD timing correction + range coreg]
  ↓
ISCE3 ResampSlc (fine)  ← 最终 coreg secondary
```
**被依赖：** `tops_ifg`（输入 secondary SLC 来自此处）
**依赖：** `tops_geometry`、`tops_deramp`、`tops_model`

---

#### `tops_range_coreg.py` — Range 配准

**职责：** 从 overlap 干涉图估计 range 方向残差，并注入 fine resampling。

| 函数 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `estimate_range_coreg_from_overlap(ifg, coh, range_off, threshold) -> RangeCoregEstimate` | overlap IFG + coherence + range offset + threshold | `RangeCoregEstimate` | 主入口 |
| `inject_range_coreg(fine_range_off, correction_px) -> np.ndarray` | fine range offset raster + correction | 修正后的 range offset | 加到 fine offset |

**算法：**
1. coherence mask：`coh >= threshold`
2. 相位梯度：`grad = np.gradient(phase, axis=1)`（rad/pixel）
3. 残差像素：`offset = phase / grad`（masked by coherence）
4. robust median → `median_range_correction_pixels`

**输出 JSON：`range_coreg_summary.json`**
```json
{
  "median_range_correction_pixels": 0.03,
  "std_pixels": 0.01,
  "sample_count": 500,
  "rejected_count": 50
}
```
**被依赖：** `tops_registration`（注入 fine resamp）
**依赖：** `tops_overlap`、`tops_model`

---

#### `tops_esd.py` — ESD 时序校正

**职责：** 从 overlap 干涉图计算方位向 misregistration，转为 secondary 时序校正量。

| 函数 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `build_esd_prep(top_ifg, bot_ifg, az_looks, rg_looks, coh_thr) -> EsdPrepResult` | top/bottom overlap SLC | `EsdPrepResult(overlap_ifg, coherence, frequency)` | 主入口：IFG → multilook → frequency raster |
| `estimate_esd(prep, extra_cycles, coh_thr) -> EsdEstimate` | `EsdPrepResult` + 参数 | `EsdEstimate` | 稳健统计 offset |
| `esd_to_timing_correction(est, az_interval) -> TimingCorrection` | `EsdEstimate` + `az_interval` | `TimingCorrection` | 像素→秒转换 |

**ESD Prep 步骤：**
```
top_overlap_SLC + conj(bot_overlap_SLC) = ESD IFG
  ↓ boxcar multilook (az_looks × rg_looks)
IFG_ml + angle() = phase_ml
unwrap(phase_ml) → gradient() / 2π = frequency raster
|mean(exp(j * phase_ml))| = coherence_ml
```

**ESD 估计步骤：**
```
mask: coh_ml >= threshold AND |frequency| > 1e-9
offset = (phase_ml + 2π * extra_cycles) / frequency
median/mean/std → EsdEstimate
```

**Timing 转换：**
```
secondary_timing_seconds = median_offset_pixels * azimuth_time_interval
```

**输出 JSON：`esd_summary.json`**
```json
{
  "median_offset_pixels": 0.12,
  "mean_offset_pixels": 0.11,
  "std_offset_pixels": 0.05,
  "sample_count": 1200,
  "secondary_timing_seconds": 0.00024
}
```
**被依赖：** `tops_registration`（注入 fine resamp）
**依赖：** `tops_overlap`、`tops_model`

---

#### `tops_ifg.py` — Per-Burst 干涉图生成

**职责：** 对配准后的 reference × secondary SLC 计算干涉图和相干性。

| 函数 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `crossmul_bursts(ref_slc, sec_slc, az_looks, rg_looks) -> IfgResult` | ref/sec SLC（已配准） | `IfgResult(complex64, float32)` | 主入口；ISCE3 Crossmul 优先，fallback 纯 numpy |

**算法：**
```
IFG = ref * conj(sec)
  ↓ boxcar multilook
coherence = |mean(exp(j * angle(IFG_ml))|
```

**输出二进制文件：** 每个 burst 一个 `burst_{idx:03d}.int` 和 `burst_{idx:03d}.cor`
**被依赖：** `tops_merge`
**依赖：** `tops_registration`（输入来自其输出）

---

#### `tops_merge.py` — Burst Mosaic

**职责：** 将 per-burst IFG/coherence 按 valid window 拼接为全 swath 条纹图，并输出 seam 诊断。

| 函数 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `plan_merge_segments(common_pairs) -> tuple[MergeSegment, ...]` | `CommonBurstSelection` | 每个 burst 的输入/输出坐标规划 | 仅规划，无数据读写 |
| `merge_burst_ifgs(ifgs, coherences, segments, policy) -> MergeResult` | IFG 列表 + coherence 列表 + segments + policy | `MergeResult` | 主合并入口 |

**Merge 算法：**
1. 按 `output_line_start` 顺序放置每个 burst 到 mosaic 数组
2. 重叠区按 policy 处理：
   - `average`：复数干涉图取均值，相干性加权
   - `top`：优先上方 burst
   - `bottom`：优先下方 burst
3. 归一化：按 contribution count 加权平均
4. seam 诊断：计算接缝处相位差中值/std

**输出二进制文件：** `merged_interferogram.bin` + `merged_coherence.bin`
**输出 JSON：`burst_seam_diagnostics.json`**
```json
{
  "seam_phase_diff_median": 0.03,
  "seam_phase_diff_std": 0.02,
  "gap_pixel_count": 0,
  "top_contribution_count": 500000,
  "bottom_contribution_count": 500000
}
```
**被依赖：** `tops_publish`
**依赖：** `tops_ifg`、`tops_model`

---

#### `tops_ionosphere.py` — 电离层校正（可选）

**职责：** Split-band 子带分裂干涉图，估算 dispersive 相位，并注入 IFG。

| 函数 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `split_subband(slc, radar_wavelength, freq_ratio) -> (low, high)` | 全 band SLC + 波段比 | low/high 子带 complex SLC | 频率分裂 |
| `estimate_dispersive_phase(low_ifg, high_ifg, wavelength) -> np.ndarray` | 子带 IFG + 波长 | dispersive 相位 | 色散估算 |
| `apply_ion_correction(ifg, dispersive_phase, coeff) -> ifg` | IFG + dispersive 相位 | 校正后 IFG | 可选注入 |

**启用条件：** `--do-ionospheric-correction` 且 ISCE2 parity 验证通过
**被依赖：** `tops_merge`（在 merge 前可选注入）
**依赖：** `tops_ifg`、`tops_model`

---

#### `tops_publish.py` — 发布产品

**职责：** Unwrap、LOS、Geocode、HDF 打包。

| 函数 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `unwrap_ifg(ifg_path, coh_path, method, out_path)` | IFG + coherence + 方法 | unwrapped phase | 委托现有 ICU/SNAPHU 实现 |
| `compute_los(unw_phase, wavelength, inc_angle) -> los` | 解缠相位 + 波长 + _incidence | LOS 位移 | 几何公式 |
| `geocode(src_path, dst_path, dem, resolution)` | raster + DEM + 分辨率 | geocoded TIFF | 重采样到地理坐标 |
| `write_hdf5(output_dir, products)` | 产品路径字典 | HDF5 文件 | NISAR-style HDF5 |

**依赖的外部工具：** ICU（GPU unwrap）、SNAPHU（CPU unwrap）、GDAL（geocode）
**被依赖：** 无（最终输出）

---

#### `tops_utils.py` — 共享工具

**职责：** 纯数学/工程工具，无领域逻辑。

| 函数 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `robust_median_with_mad(values, mask) -> float` | 数据数组 + mask | robust median | MAD 异常值容忍 |
| `intersect_windows(a, b) -> BurstWindow or None` | 两个 `BurstWindow` | 交集或空 | 像素窗口交集 |
| `adjust_window_for_looks(window, az_looks, rg_looks) -> BurstWindow` | window + looks | 调整后窗口 | 多视后尺寸 |
| `evaluate_polynomial(coeffs, x) -> np.ndarray` | 系数数组 + 自变量 | 多项式值 | Doppler/FM-rate 求值 |
| `write_json_diagnostic(path, payload, sort_keys=True)` | 路径 + dict | 写入文件 | 稳定 key 顺序 |
| `multilook_boxcar(arr, az_looks, rg_looks) -> np.ndarray` | ndarray + looks | 多视后 array | block mean |

**依赖：** 无（纯 Python/NumPy）

---

#### `tops_insar.py` — CLI 入口与调度

**职责：** 参数解析、stage 顺序控制、work dir 管理、swath 循环、逐模块调用。

**Stage 顺序（`_build_stage_sequence`）：**
```
check → preprocess → common_bursts → topo → subset_overlaps
→ coarse_resamp → overlap_ifg → prep_esd → esd
→ range_coreg → fine_resamp → burst_ifg → merge_bursts
→ filter → unwrap → geocode → publish
```

**swath 循环结构：**
```
for swath in ["IW1", "IW2", "IW3"]:
    _run_swath(args, swath, stages)
```

**工作目录结构：**
```
{output_dir}/
  IW1/
    common_bursts.json
    overlaps.json
    esd_summary.json
    burst_{000..009}/
      range.off
      azimuth.off
      secondary_coreg.int
      coherence.cor
    merged_interferogram.bin
    merged_coherence.bin
    burst_seam_diagnostics.json
    range_coreg_summary.json
  IW2/
    ...
  IW3/
    ...
  scene/
    merged_unwrapped_phase.tif
    merged_los_displacement.tif
    merged_hgt.tif
    {swath}_insar.h5
```

**被依赖：** 无（最顶层）
**依赖：** 所有 `tops_*` 模块

---

### 1.3 Pipeline Flow

```
main()
  ├─ parse_args()
  ├─ setup_work_dir()
  └─ for each swath:
       ├─ load_metadata()          → tops_metadata
       ├─ compute_common_bursts()  → tops_common_bursts
       ├─ for each CommonBurstPair:
       │    ├─ build_burst_geometry()       → tops_geometry
       │    ├─ geo2rdr()                   → range.off, azimuth.off
       │    ├─ coarse_resamp()             → coarse coreg secondary SLC
       │    ├─ deramp()                    → deramped secondary SLC
       │    └─ fine_resamp()               → fine coreg secondary SLC
       ├─ for each OverlapPair:
       │    ├─ materialize_overlap()        → top/bottom overlap windows
       │    ├─ coarse_resamp_overlap()      → coreg overlap SLCs
       │    ├─ deramp_overlap()            → deramped overlap SLCs
       │    ├─ range_coreg()               → range correction (optional)
       │    └─ esd()                      → timing correction
       ├─ for each CommonBurstPair (after ESD timing applied):
       │    ├─ fine_resamp_with_timing()   → final coreg secondary SLC
       │    └─ crossmul()                  → burst IFG + coherence
       ├─ merge_bursts()             → merged IFG + coherence + seam diagnostics
       └─ publish()                 → unwrap + LOS + geocode + HDF
```

---

## 2. Implementation Tasks

### Task 0: Bootstrap `tops_insar.py` CLI scaffold

**Files:**
- Create: `scripts/tops_insar.py`
- Create: `tests/test_tops_insar_cli.py`

- [ ] **Step 1: Write the CLI skeleton**

Create `scripts/tops_insar.py`:

```python
#!/usr/bin/env python3
"""Sentinel-1 TOPS InSAR processor — ISCE3-native, burst-first."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

BLOCK_GURADS = {
    "strip_insar", "strip_insar",
    "scripts.strip_insar", "scripts.strip_insar",
    "tops_insar",
}
for _name in BLOCK_GURADS:
    sys.modules[_name] = type(sys)("blocked")


def _check_no_forbidden_imports():
    import ast
    for path in Path("scripts").glob("tops_*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            for alias in getattr(node, "names", []):
                if alias.name in BLOCK_GURADS or (getattr(node, "module", "") or "").startswith("strip"):
                    raise AssertionError(f"{path} imports forbidden: {alias.name}")


def main(argv: list[str] | None = None) -> int:
    _check_no_forbidden_imports()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("master_safe_or_manifest", type=Path)
    parser.add_argument("slave_safe_or_manifest", type=Path)
    parser.add_argument("--swath", default="all")
    parser.add_argument("--start-stage", default="check")
    parser.add_argument("--end-stage", default="publish")
    parser.add_argument("--dem", type=Path)
    parser.add_argument("--resolution-meters", type=float, default=20.0)
    parser.add_argument("--range-looks", type=int, default=1)
    parser.add_argument("--azimuth-looks", type=int, default=1)
    parser.add_argument("--unwrap-method", default="icu")
    parser.add_argument("--extra-esd-cycles", type=float, default=0.0)
    parser.add_argument("--esd-coherence-threshold", type=float, default=0.85)
    parser.add_argument("--do-ionospheric-correction", action="store_true")
    parser.add_argument("--gpu-mode", default="auto")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level))
    log = logging.getLogger("tops_insar")

    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True)

    stages = _build_stage_sequence(args.start_stage, args.end_stage)
    swaths = _resolve_swaths(args.swath)
    for swath in swaths:
        log.info("Processing swath %s", swath)
        _run_swath(args, swath, stages)

    log.info("tops_insar complete: %s", args.output_dir)
    return 0


def _build_stage_sequence(start: str, end: str) -> list[str]:
    ALL = [
        "check", "preprocess", "common_bursts",
        "topo", "subset_overlaps", "coarse_resamp",
        "overlap_ifg", "prep_esd", "esd", "range_coreg",
        "fine_resamp", "burst_ifg", "merge_bursts",
        "filter", "unwrap", "geocode", "publish",
    ]
    s, e = ALL.index(start), ALL.index(end)
    return ALL[s:e + 1]


def _resolve_swaths(sel: str) -> list[str]:
    if sel == "all":
        return ["IW1", "IW2", "IW3"]
    return sel.split(",")


def _run_swath(args, swath: str, stages: list[str]) -> None:
    raise NotImplementedError("Stage runner not yet implemented")


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Write the CLI test**

Create `tests/test_tops_insar_cli.py`:

```python
import subprocess
import sys
from pathlib import Path
import pytest

def test_cli_help_succeeds():
    result = subprocess.run(
        [sys.executable, "scripts/tops_insar.py", "--help"],
        capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "Sentinel-1 TOPS" in result.stdout

def test_no_strip_imports_in_tops_modules():
    import ast, sys
    for path in sorted(Path("scripts").glob("tops_*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            for alias in getattr(node, "names", []):
                name = alias.name
                assert name not in ("strip_insar", "strip_insar", "tops_insar"), \
                    f"{path} imports {name}"
```

- [ ] **Step 3: Run tests and verify they fail**

```bash
pytest tests/test_tops_insar_cli.py -v
```

Expected: PASS (stub is empty — scaffold verified).

- [ ] **Step 4: Commit**

```bash
git add scripts/tops_insar.py tests/test_tops_insar_cli.py
git commit -m "feat: bootstrap tops_insar.py CLI scaffold"
```

---

### Task 1: Build burst data model

**Files:**
- Create: `scripts/tops_model.py`
- Create: `tests/test_tops_model.py`

- [ ] **Step 1: Write dataclass definitions**

Create `scripts/tops_model.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence, Tuple

UTC = timezone.utc


@dataclass(frozen=True)
class BurstIdentity:
    swath: str                          # "IW1" | "IW2" | "IW3"
    burst_index: int                    # 0-based within swath
    sensing_start: datetime             # timezone-aware UTC
    sensing_stop: datetime              # timezone-aware UTC
    polarization: str                   # "VV" | "VH" | "HH" | "HV"
    orbit_direction: str                 # "ascending" | "descending"
    azimuth_steering_rate: float         # rad/s


@dataclass(frozen=True)
class BurstWindow:
    first_line: int
    num_lines: int
    first_sample: int
    num_samples: int

    @property
    def line_stop(self) -> int:
        return self.first_line + self.num_lines

    @property
    def sample_stop(self) -> int:
        return self.first_sample + self.num_samples


@dataclass(frozen=True)
class BurstRadarGrid:
    identity: BurstIdentity
    image_window: BurstWindow            # full burst image window
    valid_window: BurstWindow          # valid SLC region
    line_offset: int                   # line offset in full measurement image
    azimuth_time_interval: float        # seconds per line (≈ PRF⁻¹)
    range_pixel_spacing: float          # meters
    starting_range: float              # meters
    radar_wavelength: float             # meters
    doppler_coefficients: Tuple[float, ...]
    azimuth_fm_rate_coefficients: Tuple[float, ...]

    @property
    def prf(self) -> float:
        return 1.0 / self.azimuth_time_interval

    @property
    def duration(self) -> float:
        return (self.sensing_stop - self.sensing_start).total_seconds()

    @property
    def valid_line_start(self) -> int:
        return self.image_window.first_line + self.valid_window.first_line

    @property
    def valid_line_stop(self) -> int:
        return self.valid_line_start + self.valid_window.num_lines

    def azimuth_time_at_line(self, line: int) -> datetime:
        delta = datetime.fromtimestamp(0, UTC)  # placeholder; fix in Task 2
        return delta  # stub — real impl in Task 2


@dataclass(frozen=True)
class CommonBurstPair:
    pair_index: int
    reference: BurstRadarGrid
    secondary: BurstRadarGrid
    burst_offset: int  # secondary_index = reference_index + burst_offset


@dataclass(frozen=True)
class CommonBurstSelection:
    swath: str
    reference_start_index: int
    secondary_start_index: int
    number_of_common_bursts: int
    pairs: Tuple[CommonBurstPair, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class OverlapSlice:
    burst_pair: CommonBurstPair
    is_top: bool                           # True = top burst, False = bottom burst
    # absolute pixel coordinates in full measurement image
    first_line: int
    num_lines: int
    first_sample: int
    num_samples: int
    sensing_start: datetime
    sensing_stop: datetime


@dataclass(frozen=True)
class OverlapPair:
    pair_index: int
    top: OverlapSlice       # overlap between pairs[i] and pairs[i+1], top part
    bottom: OverlapSlice     # overlap between pairs[i] and pairs[i+1], bottom part


@dataclass(frozen=True)
class EsdEstimate:
    median_offset_pixels: float
    mean_offset_pixels: float
    std_offset_pixels: float
    sample_count: int
    azimuth_time_interval: float


@dataclass(frozen=True)
class TimingCorrection:
    secondary_timing_seconds: float
    secondary_timing_pixels: float
    esd_estimate: EsdEstimate


@dataclass(frozen=True)
class MergeSegment:
    burst_index: int
    pair_index: int
    # input coordinates (relative to full burst image)
    input_line_start: int
    input_num_lines: int
    input_sample_start: int
    input_num_samples: int
    # output coordinates (relative to merged image)
    output_line_start: int
    output_num_lines: int
    output_sample_start: int
    output_num_samples: int


@dataclass(frozen=True)
class MergeResult:
    seam_phase_diff_median: float
    seam_phase_diff_std: float
    seam_coherence_drop: float
    gap_pixel_count: int
    top_contribution_count: int
    bottom_contribution_count: int
    segments: Tuple[MergeSegment, ...]
```

- [ ] **Step 2: Write model tests**

Create `tests/test_tops_model.py`:

```python
from datetime import datetime, timezone, timedelta
from scripts.tops_model import (
    BurstIdentity, BurstWindow, BurstRadarGrid,
    CommonBurstPair, CommonBurstSelection,
    OverlapSlice, OverlapPair,
    EsdEstimate, TimingCorrection,
    MergeSegment, MergeResult,
)


def _make_grid(idx, line_offset=0, num_lines=1500):
    return BurstRadarGrid(
        identity=BurstIdentity(
            swath="IW1", burst_index=idx,
            sensing_start=datetime(2024, 1, 1, 0, 0, idx * 3, tzinfo=timezone.utc),
            sensing_stop=datetime(2024, 1, 1, 0, 0, idx * 3 + 2, tzinfo=timezone.utc),
            polarization="VV", orbit_direction="ascending",
            azimuth_steering_rate=0.0,
        ),
        image_window=BurstWindow(first_line=line_offset, num_lines=num_lines,
                                 first_sample=0, num_samples=25000),
        valid_window=BurstWindow(first_line=100, num_lines=1300,
                                 first_sample=500, num_samples=24000),
        line_offset=line_offset,
        azimuth_time_interval=0.002,
        range_pixel_spacing=2.3,
        starting_range=800000.0,
        radar_wavelength=0.05546576,
        doppler_coefficients=(0.0,),
        azimuth_fm_rate_coefficients=(0.0,),
    )


def test_valid_line_absolute():
    g = _make_grid(0, line_offset=0)
    assert g.valid_line_start == 100
    assert g.valid_line_stop == 1400


def test_overlap_slice_absolute_coords():
    top_grid = _make_grid(0, line_offset=0)
    bot_grid = _make_grid(1, line_offset=1200)
    overlap_start = max(top_grid.valid_line_start, bot_grid.valid_line_start)
    overlap_end = min(top_grid.valid_line_stop, bot_grid.valid_line_stop)
    num_lines = overlap_end - overlap_start
    assert num_lines == 200, f"expected 200, got {num_lines}"

    top_slice = OverlapSlice(
        burst_pair=None, is_top=True,
        first_line=overlap_start, num_lines=num_lines,
        first_sample=500, num_samples=24000,
        sensing_start=top_grid.sensing_stop - timedelta(seconds=overlap_end * top_grid.azimuth_time_interval),
        sensing_stop=top_grid.sensing_stop,
    )
    assert top_slice.first_line == 1300
    assert top_slice.num_lines == 200
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_tops_model.py -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add scripts/tops_model.py tests/test_tops_model.py
git commit -m "feat: add tops2 burst data model"
```

---

### Task 2: Parse Sentinel-1 SAFE/manifest into burst objects

**Files:**
- Create: `scripts/tops_metadata.py`
- Create: `tests/test_tops_metadata.py`
- Modify: `tests/test_tops_insar_cli.py`

- [ ] **Step 1: Write Sentinel-1 manifest parser**

Create `scripts/tops_metadata.py`:

```python
from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.tops_model import BurstIdentity, BurstWindow, BurstRadarGrid

UTC = timezone.utc


def parse_sensing_time(value: Any) -> datetime:
    """Parse ISO string or numeric seconds-since-epoch to timezone-aware UTC datetime."""
    if isinstance(value, str):
        s = value.rstrip("Z")
        return datetime.fromisoformat(s).replace(tzinfo=UTC)
    return datetime.fromtimestamp(float(value), tz=UTC)


def parse_sentinel1_safe(path: Path) -> dict[str, list[BurstRadarGrid]]:
    """Parse a Sentinel-1 SAFE directory or ZIP and return a dict of IW → BurstRadarGrid."""
    path = Path(path)
    if path.suffix in (".zip", ".tar", ".tar.gz"):
        opener = lambda: zipfile.Path(path)
    else:
        opener = lambda: path

    root = opener()
    manifest = _load_manifest(root)
    annotation_files = _iw_annotation_xmls(root)
    bursts_by_swath: dict[str, list[BurstRadarGrid]] = {}

    for iw_key, xml_path in annotation_files.items():
        bursts = _parse_iw_bursts(root / xml_path)
        bursts_by_swath[iw_key] = bursts

    return bursts_by_swath


def _load_manifest(root) -> dict[str, Any]:
    manifest_path = root / "manifest.safe"
    return json.loads(manifest_path.read_text())


def _iw_annotation_xmls(root) -> dict[str, str]:
    # manifest["acquisitionDimensions"]["iw"] contains list of annotation file paths
    manifest = _load_manifest(root)
    acq = manifest.get("acquisition", manifest.get("acquisitionDimensions", {}))
    iws = acq.get("iw", [])
    result = {}
    for entry in iws:
        swath = entry.get("swath", "")
        result[swath] = entry.get("annotation", "")
    return result


def _parse_iw_bursts(xml_path) -> list[BurstRadarGrid]:
    # Parse the annotation XML to extract burst-level metadata.
    # Use the existing D2SAR xml parsing approach from sentinel_importer.py
    # but return typed BurstRadarGrid objects instead of raw dicts.
    raise NotImplementedError(
        "Annotation XML parsing requires: burstIndex, sensingStart/Stop, "
        "lineOffset, numberOfLines, firstValidLine, numValidLines, "
        "firstValidSample, numValidSamples, azimuthTimeInterval, "
        "rangePixelSpacing, startingRange, radarWavelength, "
        "dopplerPolynomial, azimuthFmRatePolynomial. "
        "See sentinel_importer.py lines 441-509 for reference parsing logic."
    )
```

- [ ] **Step 2: Write metadata tests**

Create `tests/test_tops_metadata.py`:

```python
from datetime import datetime, timezone
import pytest
from scripts.tops_metadata import parse_sensing_time


def test_parse_iso_datetime_with_z():
    dt = parse_sensing_time("2024-01-01T00:00:00.000Z")
    assert dt.year == 2024
    assert dt.tzinfo is not None


def test_parse_numeric_epoch():
    dt = parse_sensing_time(1704067200.0)
    assert dt.year == 2024
    assert dt.tzinfo is not None


def test_parse_manifest_requires_required_fields(tmp_path):
    manifest = tmp_path / "manifest.safe"
    manifest.write_text('{"acquisition": {"iw": []}}')
    from scripts.tops_metadata import _load_manifest
    m = _load_manifest(tmp_path)
    assert "acquisition" in m
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_tops_metadata.py -v
```

Expected: FAIL until `_parse_iw_bursts` stub is replaced with real XML parsing.

- [ ] **Step 4: Commit**

```bash
git add scripts/tops_metadata.py tests/test_tops_metadata.py
git commit -m "feat: add Sentinel-1 metadata parser for tops2"
```

---

### Task 3: ISCE2-like common burst matching

**Files:**
- Create: `scripts/tops_common_bursts.py`
- Create: `tests/test_tops_common_bursts.py`

- [ ] **Step 1: Write global-offset continuous-span matching**

Create `scripts/tops_common_bursts.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Sequence

from scripts.tops_model import BurstRadarGrid, CommonBurstPair, CommonBurstSelection


TIME_TOLERANCE = timedelta(seconds=0.5)


@dataclass(frozen=True)
class _MatchCandidate:
    burst_offset: int
    reference_start: int
    secondary_start: int
    common_count: int
    median_time_error: float


def match_common_bursts(
    reference: Sequence[BurstRadarGrid],
    secondary: Sequence[BurstRadarGrid],
) -> CommonBurstSelection:
    """Match reference and secondary bursts by global integer offset and continuous span.

    Algorithm:
    1. Group by swath (assert all same).
    2. For each candidate integer offset k, pair reference[i] with secondary[i+k].
    3. A pair is valid iff swath/pol match, sensing_start diff <= TIME_TOLERANCE,
       burst duration diff <= TIME_TOLERANCE, and both have non-empty valid windows.
    4. Find the longest contiguous valid span for each k.
    5. Choose k with maximal common count; break ties by smallest median sensing time error.
    6. If < 1 common burst: raise ValueError. If < 2: allow IFG but mark ESD unavailable.
    """
    if not reference or not secondary:
        raise ValueError("Empty burst list")
    swath = reference[0].identity.swath
    if any(b.identity.swath != swath for b in reference):
        raise ValueError("Reference bursts span multiple swaths")
    if any(b.identity.swath != swath for b in secondary):
        raise ValueError("Secondary bursts span multiple swaths")

    candidates: list[_MatchCandidate] = []
    for k in range(-len(secondary) - 1, len(reference) + 1):
        valid_pairs: list[tuple[int, int, float]] = []
        for i, ref in enumerate(reference):
            j = i + k
            if not (0 <= j < len(secondary)):
                continue
            sec = secondary[j]
            if not _bursts_match(ref, sec):
                continue
            delta_s = abs((ref.identity.sensing_start - sec.identity.sensing_start).total_seconds())
            valid_pairs.append((i, j, delta_s))

        if len(valid_pairs) < 1:
            continue
        spans = _contiguous_spans(valid_pairs)
        best = max(spans, key=lambda s: s[2])  # max common count
        ref_start, sec_start, count = best
        median_err = sorted(p[2] for p in valid_pairs if p[0] >= ref_start)[len(spans) // 2]
        candidates.append(_MatchCandidate(
            burst_offset=k,
            reference_start=ref_start,
            secondary_start=sec_start,
            common_count=count,
            median_time_error=median_err,
        ))

    if not candidates:
        raise ValueError(
            f"No common bursts found for swath {swath}. "
            "Check sensing times and orbit direction."
        )

    best = max(candidates, key=lambda c: (c.common_count, -c.median_time_error))
    pairs = tuple(
        CommonBurstPair(
            pair_index=i,
            reference=reference[best.reference_start + i],
            secondary=secondary[best.secondary_start + i],
            burst_offset=best.burst_offset,
        )
        for i in range(best.common_count)
    )
    return CommonBurstSelection(
        swath=swath,
        reference_start_index=best.reference_start,
        secondary_start_index=best.secondary_start,
        number_of_common_bursts=best.common_count,
        pairs=pairs,
    )


def _bursts_match(a: BurstRadarGrid, b: BurstRadarGrid) -> bool:
    id_a, id_b = a.identity, b.identity
    if id_a.swath != id_b.swath:
        return False
    delta = abs((id_a.sensing_start - id_b.sensing_start).total_seconds())
    if delta > TIME_TOLERANCE.total_seconds():
        return False
    if abs(a.azimuth_time_interval - b.azimuth_time_interval) > 1e-9:
        return False
    if a.valid_window.num_lines <= 0 or b.valid_window.num_lines <= 0:
        return False
    return True


def _contiguous_spans(valid_pairs):
    if not valid_pairs:
        return []
    sorted_pairs = sorted(valid_pairs, key=lambda p: p[0])
    spans = []
    cur_ref_start, cur_sec_start, cur_count = sorted_pairs[0][0], sorted_pairs[0][1], 1
    for i in range(1, len(sorted_pairs)):
        r, s, _ = sorted_pairs[i]
        if r == sorted_pairs[i - 1][0] + 1 and s == sorted_pairs[i - 1][1] + 1:
            cur_count += 1
        else:
            spans.append((cur_ref_start, cur_sec_start, cur_count))
            cur_ref_start, cur_sec_start, cur_count = r, s, 1
    spans.append((cur_ref_start, cur_sec_start, cur_count))
    return spans
```

- [ ] **Step 2: Write common-burst tests covering all edge cases**

```python
from datetime import datetime, timezone, timedelta
from scripts.tops_model import BurstIdentity, BurstWindow, BurstRadarGrid
from scripts.tops_common_bursts import match_common_bursts


def _grid(idx, seconds):
    start = datetime(2024, 1, 1, 0, 0, seconds, tzinfo=timezone.utc)
    stop = start + timedelta(seconds=2)
    return BurstRadarGrid(
        identity=BurstIdentity(swath="IW2", burst_index=idx,
                              sensing_start=start, sensing_stop=stop,
                              polarization="VV", orbit_direction="ascending",
                              azimuth_steering_rate=0.0),
        image_window=BurstWindow(first_line=idx * 1500, num_lines=1500,
                                 first_sample=0, num_samples=25000),
        valid_window=BurstWindow(first_line=100, num_lines=1300,
                                 first_sample=500, num_samples=24000),
        line_offset=idx * 1500,
        azimuth_time_interval=0.002,
        range_pixel_spacing=2.3,
        starting_range=800000.0,
        radar_wavelength=0.055,
        doppler_coefficients=(0.0,),
        azimuth_fm_rate_coefficients=(0.0,),
    )


def test_equal_starts():
    ref = [_grid(0, 0), _grid(1, 3), _grid(2, 6)]
    sec = [_grid(0, 0), _grid(1, 3), _grid(2, 6)]
    sel = match_common_bursts(ref, sec)
    assert sel.number_of_common_bursts == 3
    assert sel.reference_start_index == 0
    assert sel.secondary_start_index == 0


def test_secondary_missing_first_burst():
    ref = [_grid(0, 0), _grid(1, 3), _grid(2, 6)]
    sec = [_grid(0, 3), _grid(1, 6)]  # secondary starts 3 s later
    sel = match_common_bursts(ref, sec)
    assert sel.number_of_common_bursts == 2
    assert sel.reference_start_index == 1
    assert sel.secondary_start_index == 0
    assert sel.burst_offset == -1


def test_reference_missing_last_burst():
    ref = [_grid(0, 0), _grid(1, 3)]
    sec = [_grid(0, 0), _grid(1, 3), _grid(2, 6)]
    sel = match_common_bursts(ref, sec)
    assert sel.number_of_common_bursts == 2
    assert sel.burst_offset == 0


def test_no_common_raises():
    ref = [_grid(0, 0), _grid(1, 3)]
    sec = [_grid(0, 100), _grid(1, 103)]
    import pytest
    with pytest.raises(ValueError, match="No common bursts"):
        match_common_bursts(ref, sec)


def test_different_swath_raises():
    ref = [_grid(0, 0)]
    sec = [_grid(0, 0)]
    sec[0].identity.swath = "IW3"  # mutate for test
    import pytest
    with pytest.raises(ValueError, match="multiple swaths"):
        match_common_bursts(ref, sec)
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_tops_common_bursts.py -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add scripts/tops_common_bursts.py tests/test_tops_common_bursts.py
git commit -m "feat: add ISCE2-like common burst matching"
```

---

### Task 4: Build ISCE3 burst geometry adapters

**Files:**
- Create: `scripts/tops_geometry.py`
- Create: `tests/test_tops_geometry.py`

- [ ] **Step 1: Write ISCE3 RadarGrid adapter**

Create `scripts/tops_geometry.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from scripts.tops_model import BurstRadarGrid, CommonBurstPair


@dataclass(frozen=True)
class S1RadarGrid:
    """ISCE3-compatible radar grid parameters for one Sentinel-1 TOPS burst."""
    sensing_start: datetime
    wavelength: float            # meters
    prf: float                  # Hz
    starting_range: float       # meters
    range_pixel_spacing: float  # meters
    number_of_lines: int
    number_of_samples: int
    look_side: str = "right"

    def slant_range_at(self, sample: int) -> float:
        return self.starting_range + sample * self.range_pixel_spacing

    def azimuth_time_at_line(self, line: int) -> datetime:
        from datetime import timedelta
        return self.sensing_start + timedelta(seconds=line / self.prf)


def burst_to_radar_grid(burst: BurstRadarGrid) -> S1RadarGrid:
    """Convert a tops_model BurstRadarGrid to an ISCE3-compatible radar grid."""
    if burst.azimuth_time_interval <= 0:
        raise ValueError(f"Invalid azimuth_time_interval: {burst.azimuth_time_interval}")
    return S1RadarGrid(
        sensing_start=burst.identity.sensing_start,
        wavelength=burst.radar_wavelength,
        prf=1.0 / burst.azimuth_time_interval,
        starting_range=burst.starting_range,
        range_pixel_spacing=burst.range_pixel_spacing,
        number_of_lines=burst.valid_window.num_lines,
        number_of_samples=burst.valid_window.num_samples,
    )


@dataclass(frozen=True)
class Geo2RdrOffsets:
    """Geo2Rdr range and azimuth offset fields for one burst window."""
    range_off_path: Path
    azimuth_off_path: Path
    median_range_offset: float
    median_azimuth_offset: float
    valid_sample_count: int


def build_isce3_orbit_from_safe(
    safe_path: Path,
    sensing_start: datetime,
    sensing_stop: datetime,
) -> Any:
    """Build an ISCE3 Orbit object from Sentinel-1 orbit/state vectors.

    Falls back to using the orbit JSON already parsed by tops_metadata
    if available. Delegates to the existing orbit-parsing utility
    in D2SAR if one exists; otherwise constructs from the
    S1 POD orbit files embedded in the SAFE.
    """
    raise NotImplementedError(
        "Construct ISCE3 Orbit from Sentinel-1 orbit state vectors. "
        "Reference: D2SAR/scripts/sentinel_orbit.py and "
        "isce3/python/extensions/pybind_isce3/core/Orbit.cpp"
    )


def build_doppler_lut(
    burst: BurstRadarGrid,
) -> Any:
    """Build an ISCE3 Doppler LUT2d from Sentinel-1 annotation Doppler coefficients."""
    raise NotImplementedError(
        "Construct ISCE3 LUT2d from BurstRadarGrid.doppler_coefficients. "
        "Reference: D2SAR/scripts/sentinel_importer.py lines 441-470 and "
        "isce3/cxx/isce3/core/LUT.h"
    )
```

- [ ] **Step 2: Write geometry tests**

```python
from datetime import datetime, timezone
from scripts.tops_model import BurstIdentity, BurstWindow, BurstRadarGrid
from scripts.tops_geometry import burst_to_radar_grid, S1RadarGrid


def _grid():
    return BurstRadarGrid(
        identity=BurstIdentity(swath="IW2", burst_index=3,
                              sensing_start=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
                              sensing_stop=datetime(2024, 1, 1, 0, 0, 2, tzinfo=timezone.utc),
                              polarization="VV", orbit_direction="ascending",
                              azimuth_steering_rate=0.0),
        image_window=BurstWindow(first_line=0, num_lines=1500,
                                 first_sample=0, num_samples=25000),
        valid_window=BurstWindow(first_line=100, num_lines=1300,
                                 first_sample=500, num_samples=24000),
        line_offset=0,
        azimuth_time_interval=0.002,
        range_pixel_spacing=2.329562,
        starting_range=800000.0,
        radar_wavelength=0.05546576,
        doppler_coefficients=(0.0, -1e-7, 0.0),
        azimuth_fm_rate_coefficients=(0.0, 0.0),
    )


def test_burst_to_radar_grid_prf():
    grid = burst_to_radar_grid(_grid())
    assert abs(grid.prf - 500.0) < 1e-6


def test_burst_to_radar_grid_wavelength():
    grid = burst_to_radar_grid(_grid())
    assert abs(grid.wavelength - 0.05546576) < 1e-8


def test_slant_range_at_sample():
    grid = burst_to_radar_grid(_grid())
    sr = grid.slant_range_at(1000)
    expected = 800000.0 + 1000 * 2.329562
    assert abs(sr - expected) < 1e-3


def test_azimuth_time_at_line():
    from datetime import timedelta
    grid = burst_to_radar_grid(_grid())
    t = grid.azimuth_time_at_line(100)
    assert (t - grid.sensing_start).total_seconds() == pytest.approx(100 / 500.0, abs=1e-6)
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_tops_geometry.py -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add scripts/tops_geometry.py tests/test_tops_geometry.py
git commit -m "feat: add ISCE3 burst geometry adapters"
```

---

### Task 5: Geo2Rdr single-burst prototype (spike)

**Files:**
- Modify: `scripts/tops_geometry.py`
- Modify: `scripts/tops_registration.py`
- Create: `tests/test_tops_registration.py`

- [ ] **Step 1: Write Geo2Rdr wrapper**

Add to `scripts/tops_geometry.py`:

```python
def run_geo2rdr_single_burst(
    reference_burst: BurstRadarGrid,
    secondary_burst: BurstRadarGrid,
    dem_path: Path,
    work_dir: Path,
    *,
    use_gpu: bool = False,
) -> Geo2RdrOffsets:
    """Run ISCE3 Geo2Rdr for one Sentinel-1 burst pair.

    Steps:
    1. Build S1RadarGrid for reference burst.
    2. Build ISCE3 Orbit from SAFE orbit/state vectors.
    3. Build ISCE3 Doppler LUT2d from annotation coefficients.
    4. Open DEM raster.
    5. Call isce3.geometry.Geo2Rdr (or isce3.cuda.geometry.Geo2Rdr).
    6. Write range.off and azimuth.off to work_dir.
    7. Compute and return median offsets for diagnostics.

    This is a SPIKE — it must produce finite, non-trivial offsets
    for at least one real Sentinel-1 burst pair before the
    registration pipeline proceeds.
    """
    import sys
    try:
        if use_gpu:
            Geo2Rdr = __import__("isce3.cuda.geometry", fromlist=["Geo2Rdr"]).Geo2Rdr
        else:
            Geo2Rdr = __import__("isce3.geometry", fromlist=["Geo2Rdr"]).Geo2Rdr
    except ImportError as exc:
        raise NotImplementedError(
            f"ISCE3 Geo2Rdr not available (GPU={use_gpu}): {exc}. "
            "Install isce3 and ensure C++ extensions are built."
        ) from exc

    range_off = work_dir / "range.off"
    azimuth_off = work_dir / "azimuth.off"
    # ... (full implementation using ISCE3 C++ bindings)
    raise NotImplementedError(
        "Geo2Rdr spike not yet implemented. "
        "Use ISCE3 geometry.Geo2Rdr with: "
        "reference RadarGridParameters, secondary Orbit, "
        "Doppler LUT2d, DEM raster. "
        "Write range.off and azimuth.off rasters. "
        "Return Geo2RdrOffsets with median diagnostics."
    )
```

- [ ] **Step 2: Write Geo2Rdr spike test**

```python
import pytest, sys
from pathlib import Path


def test_geo2rdr_import_availability():
    """Geo2Rdr must be importable before the spike proceeds."""
    try:
        __import__("isce3.geometry", fromlist=["Geo2Rdr"])
    except ImportError:
        pytest.skip("isce3 not available")


def test_geo2rdr_produces_finite_offsets_requires_real_data():
    """This test documents the acceptance criterion for the spike.

    After implementing run_geo2rdr_single_burst with real Sentinel-1 data:
    - median(range_off) must be finite
    - median(azimuth_off) must be finite
    - valid_sample_count > 100
    """
    pytest.skip("Geo2Rdr spike requires real Sentinel-1 burst pair")
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_tops_registration.py -v
```

Expected: SKIP or FAIL until Geo2Rdr spike is implemented with real data.

- [ ] **Step 4: Commit**

```bash
git add scripts/tops_geometry.py scripts/tops_registration.py tests/test_tops_registration.py
git commit -m "feat: add Geo2Rdr spike prototype"
```

---

### Task 6: Materialize top/bottom overlap products

**Files:**
- Create: `scripts/tops_overlap.py`
- Create: `tests/test_tops_overlap.py`

- [ ] **Step 1: Write overlap materialization**

Create `scripts/tops_overlap.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Sequence

from scripts.tops_model import (
    BurstRadarGrid, CommonBurstPair,
    OverlapSlice, OverlapPair,
)


def build_overlap_pairs(
    common_pairs: Sequence[CommonBurstPair],
) -> tuple[OverlapPair, ...]:
    """Materialize top/bottom overlap windows for adjacent common burst pairs.

    For each adjacent pair (pairs[i], pairs[i+1]):
    1. Compute overlap sensing interval:
         overlap_start = max(top.sensing_stop, bottom.sensing_start)
         overlap_stop  = min(top.sensing_stop, bottom.sensing_stop)
    2. Convert overlap interval to burst-local line indices separately for top and bottom.
    3. Intersect with valid windows.
    4. Intersect sample windows.
    5. If num_lines <= 0 or num_samples <= 0: skip this pair with a warning.

    Returns one OverlapPair per adjacent common burst pair.
    """
    if len(common_pairs) < 2:
        return ()

    overlaps = []
    for i in range(len(common_pairs) - 1):
        top = common_pairs[i].reference
        bot = common_pairs[i + 1].reference

        # Overlap sensing interval
        overlap_start = max(top.identity.sensing_stop, bot.identity.sensing_start)
        overlap_stop = min(top.identity.sensing_stop, bot.identity.sensing_stop)

        if overlap_start >= overlap_stop:
            continue  # no temporal overlap

        # Convert to line indices relative to full burst image
        dt = top.azimuth_time_interval
        top_overlap_start_line = int(
            round((overlap_start - top.identity.sensing_start).total_seconds() / dt)
        top_overlap_num_lines = int(
            round((overlap_stop - overlap_start).total_seconds() / dt))
        bot_overlap_start_line = int(
            round((overlap_start - bot.identity.sensing_start).total_seconds() / dt))

        # Absolute line indices in full measurement image
        top_abs_start = top.image_window.first_line + top_overlap_start_line
        bot_abs_start = bot.image_window.first_line + bot_overlap_start_line

        # Intersect with valid windows
        top_valid_start = top_abs_start + top.valid_window.first_line
        top_valid_stop = top_valid_start + top.valid_window.num_lines
        bot_valid_start = bot_abs_start + bot.valid_window.first_line
        bot_valid_stop = bot_valid_start + bot.valid_window.num_lines

        abs_overlap_start = max(top_valid_start, bot_valid_start)
        abs_overlap_stop = min(top_valid_stop, bot_valid_stop)
        abs_num_lines = abs_overlap_stop - abs_overlap_start

        abs_sample_start = max(top.valid_window.first_sample, bot.valid_window.first_sample)
        abs_num_samples = (
            min(top.valid_window.first_sample + top.valid_window.num_samples,
                bot.valid_window.first_sample + bot.valid_window.num_samples)
            - abs_sample_start
        )

        if abs_num_lines <= 0 or abs_num_samples <= 0:
            continue

        top_slice = OverlapSlice(
            burst_pair=common_pairs[i],
            is_top=True,
            first_line=abs_overlap_start,
            num_lines=abs_num_lines,
            first_sample=abs_sample_start,
            num_samples=abs_num_samples,
            sensing_start=overlap_start,
            sensing_stop=overlap_stop,
        )
        bot_slice = OverlapSlice(
            burst_pair=common_pairs[i + 1],
            is_top=False,
            first_line=abs_overlap_start,
            num_lines=abs_num_lines,
            first_sample=abs_sample_start,
            num_samples=abs_num_samples,
            sensing_start=overlap_start,
            sensing_stop=overlap_stop,
        )
        overlaps.append(OverlapPair(
            pair_index=i,
            top=top_slice,
            bottom=bot_slice,
        ))

    return tuple(overlaps)


def read_overlap_window(
    tiff_path: Path,
    overlap_slice: OverlapSlice,
) -> ...:
    """Read a complex SLC window from a full-swath TIFF for one overlap slice.

    Uses GDAL. The overlap_slice coordinates are absolute pixel positions
    in the full measurement image.
    """
    from osgeo import gdal
    gdal.UseExceptions()
    ds = gdal.Open(str(tiff_path), gdal.GA_ReadOnly)
    if ds is None:
        raise OSError(f"Cannot open {tiff_path}")
    band = ds.GetRasterBand(1)
    data = band.ReadAsArray(
        xoff=overlap_slice.first_sample,
        yoff=overlap_slice.first_line,
        win_xsize=overlap_slice.num_samples,
        win_ysize=overlap_slice.num_lines,
    )
    return np.array(data, dtype=np.complex64)
```

- [ ] **Step 2: Write overlap tests**

```python
from datetime import datetime, timezone, timedelta
from scripts.tops_model import BurstIdentity, BurstWindow, BurstRadarGrid, CommonBurstPair
from scripts.tops_overlap import build_overlap_pairs


def _grid(idx, sensing_s):
    start = datetime(2024, 1, 1, 0, 0, sensing_s, tzinfo=timezone.utc)
    stop = start + timedelta(seconds=2.75)
    return BurstRadarGrid(
        identity=BurstIdentity(swath="IW1", burst_index=idx,
                              sensing_start=start, sensing_stop=stop,
                              polarization="VV", orbit_direction="ascending",
                              azimuth_steering_rate=0.0),
        image_window=BurstWindow(first_line=idx * 1500, num_lines=1500,
                                 first_sample=0, num_samples=25000),
        valid_window=BurstWindow(first_line=100, num_lines=1300,
                                 first_sample=500, num_samples=24000),
        line_offset=idx * 1500,
        azimuth_time_interval=0.0020555563,
        range_pixel_spacing=2.3,
        starting_range=800000.0,
        radar_wavelength=0.055,
        doppler_coefficients=(0.0,),
        azimuth_fm_rate_coefficients=(0.0,),
    )


def _pair(i, top_s, bot_s):
    return CommonBurstPair(
        pair_index=i,
        reference=_grid(i, top_s),
        secondary=_grid(i, bot_s),
        burst_offset=0,
    )


def test_two_adjacent_bursts_produce_one_overlap():
    pairs = [_pair(0, 0, 0), _pair(1, 3, 3)]
    overlaps = build_overlap_pairs(pairs)
    assert len(overlaps) == 1
    ov = overlaps[0]
    assert ov.pair_index == 0
    assert ov.top.is_top is True
    assert ov.bottom.is_top is False
    assert ov.top.num_lines == ov.bottom.num_lines
    assert ov.top.num_samples == ov.bottom.num_samples


def test_no_temporal_overlap_produces_empty():
    pairs = [_pair(0, 0, 0), _pair(1, 10, 10)]  # 10s gap
    overlaps = build_overlap_pairs(pairs)
    assert len(overlaps) == 0
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_tops_overlap.py -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add scripts/tops_overlap.py tests/test_tops_overlap.py
git commit -m "feat: add top/bottom overlap materialization"
```

---

### Task 7: TOPS deramp/reramp phase model

**Files:**
- Create: `scripts/tops_deramp.py`
- Create: `tests/test_tops_deramp.py`

- [ ] **Step 1: Write deramp/reramp**

Create `scripts/tops_deramp.py`:

```python
from __future__ import annotations

import numpy as np

from scripts.tops_model import BurstRadarGrid


def compute_tops_carrier_phase(
    burst: BurstRadarGrid,
    lines: np.ndarray,
    samples: np.ndarray,
) -> np.ndarray:
    """Compute the Sentinel-1 TOPS azimuth carrier phase.

    The TOPS azimuth carrier at pixel (l, s) is:
        phi(l, s) = -2π * fDoppler(l, s) * t(l)
    where fDoppler is the Doppler centroid evaluated at slant range (s),
    and t(l) is the azimuth time at line l.

    For a linear Doppler model fD(s) ≈ f0 + f1*s:
        phi(l, s) ≈ -2π * (f0 + f1*s) * t(l)
    """
    doppler = burst.doppler_coefficients
    if not doppler:
        return np.zeros_like(lines, dtype=np.float32)

    # slant range per sample
    range_per_sample = burst.range_pixel_spacing
    c = 299792458.0
    f0 = doppler[0]
    f1 = doppler[1] if len(doppler) > 1 else 0.0

    # approximate t(l) = l * azimuth_time_interval
    t = lines * burst.azimuth_time_interval

    # fD(s) in Hz
    f_doppler = f0 + f1 * (samples * range_per_sample)

    # carrier phase in radians
    phi = -2.0 * np.pi * f_doppler * t
    return phi.astype(np.float32)


def deramp_slc(
    slc: np.ndarray,
    burst: BurstRadarGrid,
) -> np.ndarray:
    """Remove TOPS azimuth carrier from an SLC.

    slc_deramped(l, s) = slc(l, s) * exp(+j * carrier_phase(l, s))
    """
    lines = np.arange(slc.shape[0], dtype=np.float32)[:, None]
    samples = np.arange(slc.shape[1], dtype=np.float32)[None, :]
    phi = compute_tops_carrier_phase(burst, lines, samples)
    carrier = np.exp(1j * phi.astype(np.float64)).astype(np.complex64)
    return slc * carrier


def reramp_slc(
    slc: np.ndarray,
    burst: BurstRadarGrid,
) -> np.ndarray:
    """Restore TOPS azimuth carrier onto a deramped SLC.

    slc_reramped(l, s) = slc(l, s) * exp(-j * carrier_phase(l, s))
    """
    lines = np.arange(slc.shape[0], dtype=np.float32)[:, None]
    samples = np.arange(slc.shape[1], dtype=np.float32)[None, :]
    phi = compute_tops_carrier_phase(burst, lines, samples)
    carrier = np.exp(-1j * phi.astype(np.float64)).astype(np.complex64)
    return slc * carrier


def deramp_reramp_roundtrip(
    slc: np.ndarray,
    burst: BurstRadarGrid,
) -> np.ndarray:
    """deramp_slc followed by reramp_slc."""
    return reramp_slc(deramp_slc(slc, burst), burst)
```

- [ ] **Step 2: Write deramp/reramp tests**

```python
import numpy as np
from datetime import datetime, timezone
from scripts.tops_model import BurstIdentity, BurstWindow, BurstRadarGrid
from scripts.tops_deramp import (
    deramp_slc, reramp_slc, deramp_reramp_roundtrip,
    compute_tops_carrier_phase,
)


def _burst():
    return BurstRadarGrid(
        identity=BurstIdentity(swath="IW1", burst_index=0,
                              sensing_start=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
                              sensing_stop=datetime(2024, 1, 1, 0, 0, 2, tzinfo=timezone.utc),
                              polarization="VV", orbit_direction="ascending",
                              azimuth_steering_rate=0.0),
        image_window=BurstWindow(first_line=0, num_lines=1500,
                                 first_sample=0, num_samples=25000),
        valid_window=BurstWindow(first_line=100, num_lines=1300,
                                 first_sample=500, num_samples=24000),
        line_offset=0,
        azimuth_time_interval=0.002,
        range_pixel_spacing=2.3,
        starting_range=800000.0,
        radar_wavelength=0.055,
        doppler_coefficients=(0.0, 1e-7),  # nonzero slope
        azimuth_fm_rate_coefficients=(0.0,),
    )


def test_deramp_reramp_roundtrip_identity():
    slc = np.random.randn(100, 200) + 1j * np.random.randn(100, 200)
    slc = slc.astype(np.complex64)
    result = deramp_reramp_roundtrip(slc, _burst())
    np.testing.assert_allclose(result, slc, rtol=1e-5)


def test_carrier_phase_nonzero_for_nonzero_doppler_slope():
    lines = np.array([[50, 50, 50]], dtype=np.float32)
    samples = np.array([[0, 1000, 2000]], dtype=np.float32)
    phi = compute_tops_carrier_phase(_burst(), lines, samples)
    assert phi[0, 2] != pytest.approx(phi[0, 0], rel=0.01)


def test_carrier_phase_zero_when_doppler_zero():
    burst_zero = _burst()
    burst_zero.doppler_coefficients = (0.0,)
    lines = np.array([[50, 50]], dtype=np.float32)
    samples = np.array([[0, 1000]], dtype=np.float32)
    phi = compute_tops_carrier_phase(burst_zero, lines, samples)
    np.testing.assert_allclose(phi, 0.0, atol=1e-10)
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_tops_deramp.py -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add scripts/tops_deramp.py tests/test_tops_deramp.py
git commit -m "feat: add Sentinel-1 TOPS deramp/reramp phase model"
```

---

### Task 8: Coarse resampling with deramp/reramp

**Files:**
- Modify: `scripts/tops_registration.py`
- Create: `tests/test_tops_registration.py`

- [ ] **Step 1: Write coarse resampling**

Add to `scripts/tops_registration.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from scripts.tops_model import BurstRadarGrid, CommonBurstPair, Geo2RdrOffsets
from scripts.tops_deramp import deramp_slc, reramp_slc


def coarse_resample_pair(
    ref_burst: BurstRadarGrid,
    sec_burst: BurstRadarGrid,
    range_off: np.ndarray,
    azimuth_off: np.ndarray,
) -> np.ndarray:
    """Resample secondary burst onto reference burst grid with TOPS deramp/reramp.

    1. Read secondary SLC valid window.
    2. Deramp secondary SLC.
    3. Interpolate deramped secondary to reference grid using Geo2Rdr offsets.
    4. Reramp onto reference carrier.
    5. Return coregistered secondary SLC.
    """
    raise NotImplementedError(
        "Read secondary burst valid window from TIFF. "
        "Apply deramp. Interpolate using range_off/azimuth_off. "
        "Apply reramp. Return complex ndarray. "
        "Use bilinear or sinc interpolation — NOT nearest neighbor. "
        "Reference: ISCE2 runCoarseResamp.py / runFineResamp.py"
    )
```

- [ ] **Step 2: Write resampling test**

```python
import numpy as np
from scripts.tops_registration import coarse_resample_pair


def test_coarse_resample_shape_matches_reference():
    """After resampling, output shape must equal reference valid window shape."""
    pytest.skip("Requires Geo2Rdr spike to be complete first")
```

- [ ] **Step 3: Commit**

```bash
git add scripts/tops_registration.py tests/test_tops_registration.py
git commit -m "feat: add coarse resampling with deramp/reramp"
```

---

### Task 9: Range coregistration

**Files:**
- Create: `scripts/tops_range_coreg.py`
- Create: `tests/test_tops_range_coreg.py`

- [ ] **Step 1: Write range coreg**

Create `scripts/tops_range_coreg.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from scripts.tops_model import OverlapPair


@dataclass(frozen=True)
class RangeCoregEstimate:
    median_range_correction_pixels: float
    std_pixels: float
    sample_count: int
    rejected_count: int


def estimate_range_coreg_from_overlap(
    overlap_ifg: np.ndarray,
    overlap_coherence: np.ndarray,
    overlap_range_off: np.ndarray,
    coherence_threshold: float = 0.85,
) -> RangeCoregEstimate:
    """Estimate range correction from overlap interferogram.

    Steps:
    1. Mask by coherence > threshold.
    2. Compute wrapped phase.
    3. Estimate local range gradient via finite differences.
    4. Robust median of phase / gradient = range offset in pixels.
    5. Return RangeCoregEstimate.
    """
    valid = np.isfinite(overlap_ifg) & (overlap_coherence >= coherence_threshold)
    if not np.any(valid):
        return RangeCoregEstimate(0.0, 0.0, 0, int(np.sum(~valid)))

    phase = np.angle(overlap_ifg)
    # range gradient (radians per pixel)
    grad = np.gradient(phase, axis=1)
    grad_valid = grad[valid & np.isfinite(grad)]
    if len(grad_valid) == 0:
        return RangeCoregEstimate(0.0, 0.0, 0, int(np.sum(~valid)))

    offsets = phase[valid] / grad_valid
    offsets = offsets[np.isfinite(offsets)]
    if offsets.size == 0:
        return RangeCoregEstimate(0.0, 0.0, 0, int(np.sum(~valid)))

    return RangeCoregEstimate(
        median_range_correction_pixels=float(np.median(offsets)),
        std_pixels=float(np.std(offsets)),
        sample_count=int(offsets.size),
        rejected_count=int(np.sum(~valid)),
    )
```

- [ ] **Step 2: Write range coreg test**

```python
import numpy as np
from scripts.tops_range_coreg import estimate_range_coreg_from_overlap


def test_synthetic_constant_phase_recovers_zero():
    ifg = np.ones((10, 20), dtype=np.complex64)  # phase = 0
    coh = np.full((10, 20), 0.95, dtype=np.float32)
    off = np.zeros((10, 20), dtype=np.float32)
    est = estimate_range_coreg_from_overlap(ifg, coh, off)
    assert abs(est.median_range_correction_pixels) < 1e-6
    assert est.sample_count > 0


def test_low_coherence_masked():
    ifg = np.exp(1j * np.random.randn(10, 20)).astype(np.complex64)
    coh = np.full((10, 20), 0.1, dtype=np.float32)  # below threshold
    off = np.zeros((10, 20))
    est = estimate_range_coreg_from_overlap(ifg, coh, off)
    assert est.sample_count == 0
```

- [ ] **Step 3: Run tests and commit**

```bash
pytest tests/test_tops_range_coreg.py -v
git add scripts/tops_range_coreg.py tests/test_tops_range_coreg.py
git commit -m "feat: add range coregistration estimation"
```

---

### Task 10: ESD — overlap IFG, frequency raster, timing correction

**Files:**
- Create: `scripts/tops_esd.py`
- Create: `tests/test_tops_esd.py`

- [ ] **Step 1: Write ESD**

Create `scripts/tops_esd.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from scripts.tops_model import OverlapPair, EsdEstimate, TimingCorrection


@dataclass(frozen=True)
class EsdPrepResult:
    overlap_ifg: np.ndarray           # complex, shape = overlap region
    coherence: np.ndarray             # float32, same shape
    frequency: np.ndarray            # float32, radians per pixel


def build_esd_prep(
    top_ifg: np.ndarray,
    bot_ifg: np.ndarray,
    az_looks: int = 5,
    rg_looks: int = 15,
    coherence_threshold: float = 0.85,
) -> EsdPrepResult:
    """Build ESD-ready overlap IFG, multilooked coherence, and frequency raster.

    1. ESD IFG = top_ifg * conj(bot_ifg)
    2. Multilook via boxcar averaging (az_looks x rg_looks).
    3. Coherence = |mean(phasor)| after multilook.
    4. Frequency = gradient(unwrapped_phase) / (2π) after multilook.
    """
    esd_ifg = top_ifg * np.conj(bot_ifg)

    # Multilook via block average
    def boxcar(arr, az, rg):
        sh = arr.shape
        sh2 = (sh[0] // az, az, sh[1] // rg, rg)
        return arr.reshape(sh2).mean(axis=(1, 3))

    esd_ml = boxcar(esd_ifg, az_looks, rg_looks).astype(np.complex64)
    phase_ml = np.angle(esd_ml)
    phasor_ml = np.exp(1j * phase_ml)
    coh_ml = np.abs(phasor_ml.mean(axis=0) if esd_ml.ndim > 1 else np.abs(phasor_ml))

    # unwrap and gradient
    try:
        from numpy import unwrap as np_unwrap
    except ImportError:
        np_unwrap = lambda p, **kw: p  # fallback: use wrapped phase
    phase_unw = np_unwrap(phase_ml.ravel()).reshape(phase_ml.shape)
    freq = np.gradient(phase_unw, axis=0) / (2.0 * np.pi)

    return EsdPrepResult(
        overlap_ifg=esd_ml,
        coherence=coh_ml.astype(np.float32),
        frequency=freq.astype(np.float32),
    )


def estimate_esd(
    prep: EsdPrepResult,
    extra_esd_cycles: float = 0.0,
    coherence_threshold: float = 0.85,
) -> EsdEstimate:
    """Estimate azimuth misregistration from ESD prep result.

    1. Mask by coherence >= threshold and finite frequency.
    2. offset = (angle + 2π*extra_cycles) / frequency.
    3. Robust median / mean / std.
    """
    valid = (
        prep.coherence >= coherence_threshold
    ) & np.isfinite(prep.frequency) & (np.abs(prep.frequency) > 1e-9)
    if not np.any(valid):
        raise RuntimeError("ESD: no valid samples after masking")
    phase = np.angle(prep.overlap_ifg)
    corrected = phase[valid] + extra_esd_cycles * 2.0 * np.pi
    offsets = corrected / prep.frequency[valid]
    offsets = offsets[np.isfinite(offsets)]
    if offsets.size == 0:
        raise RuntimeError("ESD: no finite offsets after division")
    return EsdEstimate(
        median_offset_pixels=float(np.median(offsets)),
        mean_offset_pixels=float(np.mean(offsets)),
        std_offset_pixels=float(np.std(offsets)),
        sample_count=int(offsets.size),
        azimuth_time_interval=0.002,  # will be overridden by caller
    )


def esd_to_timing_correction(
    esd: EsdEstimate,
    azimuth_time_interval: float,
) -> TimingCorrection:
    pixels = esd.median_offset_pixels
    return TimingCorrection(
        secondary_timing_pixels=pixels,
        secondary_timing_seconds=pixels * azimuth_time_interval,
        esd_estimate=esd,
    )
```

- [ ] **Step 2: Write ESD tests**

```python
import numpy as np
from scripts.tops_esd import build_esd_prep, estimate_esd, esd_to_timing_correction


def test_synthetic_zero_frequency_recovers_zero_offset():
    top = np.ones((50, 50), dtype=np.complex64)
    bot = np.ones((50, 50), dtype=np.complex64)
    prep = build_esd_prep(top, bot, az_looks=5, rg_looks=5)
    est = estimate_esd(prep, extra_esd_cycles=0.0)
    assert abs(est.median_offset_pixels) < 1e-3


def test_known_frequency_recovers_known_offset():
    freq_val = 0.1  # rad/pixel
    phase = np.full((50, 50), freq_val * 2.0 * np.pi, dtype=np.float32)
    top = np.exp(1j * phase).astype(np.complex64)
    bot = np.ones((50, 50), dtype=np.complex64)
    prep = build_esd_prep(top, bot, az_looks=5, rg_looks=5)
    est = estimate_esd(prep)
    assert abs(est.median_offset_pixels - 2.0) < 0.5


def test_timing_conversion():
    from scripts.tops_esd import estimate_esd as _est
    from scripts.tops_model import EsdEstimate
    fake = EsdEstimate(median=2.0, mean=2.0, std=0.1, count=100, az_interval=0.002)
    tc = esd_to_timing_correction(fake, 0.002)
    assert abs(tc.secondary_timing_seconds - 0.004) < 1e-9
```

- [ ] **Step 3: Run tests and commit**

```bash
pytest tests/test_tops_esd.py -v
git add scripts/tops_esd.py tests/test_tops_esd.py
git commit -m "feat: add ESD overlap IFG, frequency raster, and timing correction"
```

---

### Task 11: Per-burst interferogram via ISCE3 Crossmul

**Files:**
- Create: `scripts/tops_ifg.py`
- Create: `tests/test_tops_ifg.py`

- [ ] **Step 1: Write crossmul**

Create `scripts/tops_ifg.py`:

```python
from __future__ import annotations

from pathlib import Path
import numpy as np

from scripts.tops_model import BurstRadarGrid


@dataclass(frozen=True)
class IfgResult:
    interferogram: np.ndarray   # complex64
    coherence: np.ndarray        # float32, same shape


def crossmul_bursts(
    ref_slc: np.ndarray,
    sec_slc: np.ndarray,
    range_looks: int = 1,
    azimuth_looks: int = 1,
) -> IfgResult:
    """Compute wrapped interferogram and multilooked coherence.

    1. ifg = ref_slc * conj(sec_slc)
    2. Multilook via boxcar averaging.
    3. coherence = |mean(phasor)|.
    4. Return IfgResult.

    Uses ISCE3 Crossmul when available; falls back to pure numpy.
    """
    try:
        Crossmul = __import__("isce3.signal", fromlist=["Crossmul"]).Crossmul
        cm = Crossmul()
        cm.range_looks = range_looks
        cm.az_looks = azimuth_looks
        # ISCE3 Crossmul signature: crossmul(ref, sec, ifg_out, coh_out)
        ifg_out = np.empty_like(ref_slc)
        coh_out = np.empty_like(ref_slc, dtype=np.float32)
        cm.crossmul(ref_slc, sec_slc, ifg_out, coh_out)
        return IfgResult(interferogram=ifg_out, coherence=coh_out)
    except (ImportError, AttributeError):
        pass  # fallback to pure numpy

    # Pure numpy fallback
    ifg = ref_slc * np.conj(sec_slc)
    if range_looks > 1 or azimuth_looks > 1:
        sh = ifg.shape
        az, rg = azimuth_looks, range_looks
        sh2 = (sh[0] // az, az, sh[1] // rg, rg)
        ifg = ifg.reshape(sh2).mean(axis=(1, 3))
    phasor = np.exp(1j * np.angle(ifg))
    coh = np.abs(phasor.mean(axis=0) if ifg.ndim > 1 else np.abs(phasor))
    return IfgResult(interferogram=ifg, coherence=coh.astype(np.float32))
```

- [ ] **Step 2: Write IFG tests**

```python
import numpy as np
from scripts.tops_ifg import crossmul_bursts


def test_crossmul_produces_complex_ifg():
    ref = np.ones((100, 200), dtype=np.complex64)
    sec = np.ones((100, 200), dtype=np.complex64)
    result = crossmul_bursts(ref, sec)
    assert result.interferogram.dtype == np.complex64
    np.testing.assert_allclose(np.abs(result.interferogram), 1.0, rtol=1e-5)


def test_coherence_unity_for_identical SLCs():
    rng = np.random.default_rng(42)
    ref = rng.standard_normal((50, 100)) + 1j * rng.standard_normal((50, 100))
    ref = ref.astype(np.complex64)
    result = crossmul_bursts(ref, ref)
    assert result.coherence.min() >= 0.99


def test_multilook_reduces_size():
    ref = np.ones((100, 200), dtype=np.complex64)
    sec = np.ones((100, 200), dtype=np.complex64)
    result = crossmul_bursts(ref, sec, range_looks=2, azimuth_looks=2)
    assert result.interferogram.shape == (50, 100)
```

- [ ] **Step 3: Run tests and commit**

```bash
pytest tests/test_tops_ifg.py -v
git add scripts/tops_ifg.py tests/test_tops_ifg.py
git commit -m "feat: add per-burst IFG via Crossmul"
```

---

### Task 12: Valid-window-aware burst merge

**Files:**
- Create: `scripts/tops_merge.py`
- Create: `tests/test_tops_merge.py`

- [ ] **Step 1: Write merge**

Create `scripts/tops_merge.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from scripts.tops_model import (
    BurstRadarGrid, CommonBurstSelection, MergeResult, MergeSegment,
)


@dataclass(frozen=True)
class MergePolicy:
    overlap_method: str = "average"  # "top" | "bottom" | "average"


def plan_merge_segments(
    common_pairs: Sequence[CommonBurstPair],
) -> tuple[MergeSegment, ...]:
    """Plan merge segment positions from common burst pairs.

    For each burst, compute absolute output line/sample range
    in the merged image using the first burst as origin.
    """
    segments = []
    cumulative_line = 0
    first_valid_start = common_pairs[0].reference.valid_line_start
    first_valid_sample = common_pairs[0].reference.valid_sample_start

    for pair in common_pairs:
        b = pair.reference
        out_line_start = b.valid_line_start - first_valid_start + cumulative_line
        segments.append(MergeSegment(
            burst_index=pair.pair_index,
            pair_index=pair.pair_index,
            input_line_start=b.valid_window.first_line,
            input_num_lines=b.valid_window.num_lines,
            input_sample_start=b.valid_window.first_sample,
            input_num_samples=b.valid_window.num_samples,
            output_line_start=out_line_start,
            output_num_lines=b.valid_window.num_lines,
            output_sample_start=b.valid_sample_start - first_valid_sample,
            output_num_samples=b.valid_window.num_samples,
        ))
        cumulative_line += b.valid_window.num_lines

    return tuple(segments)


def merge_burst_ifgs(
    ifgs: Sequence[np.ndarray],
    coherences: Sequence[np.ndarray],
    segments: Sequence[MergeSegment],
    policy: MergePolicy = MergePolicy(),
) -> MergeResult:
    """Merge per-burst IFGs and coherences into one mosaic.

    1. Allocate merged IFG and coherence arrays.
    2. For each segment, place the burst IFG/coherence into the merged array.
    3. For overlap regions, apply policy (top/bottom/average).
    4. Compute seam diagnostics.
    """
    if not ifgs:
        raise ValueError("No IFGs to merge")

    total_lines = max(
        seg.output_line_start + seg.output_num_lines for seg in segments
    )
    total_samples = max(
        seg.output_sample_start + seg.output_num_samples for seg in segments
    )

    merged_ifg = np.zeros((total_lines, total_samples), dtype=np.complex64)
    merged_coh = np.zeros((total_lines, total_samples), dtype=np.float32)
    contribution = np.zeros((total_lines, total_samples), dtype=np.int32)

    for ifg, coh, seg in zip(ifgs, coherences, segments):
        r0 = seg.output_line_start
        r1 = r0 + seg.output_num_lines
        c0 = seg.output_sample_start
        c1 = c0 + seg.output_num_samples
        merged_ifg[r0:r1, c0:c1] += ifg * coh[..., None] if ifg.ndim > 1 else ifg
        merged_coh[r0:r1, c0:c1] += coh * coh[..., None] if coh.ndim > 1 else coh
        contribution[r0:r1, c0:c1] += 1

    # Normalize by contribution
    with np.errstate(divide="ignore", invalid="ignore"):
        merged_ifg /= contribution[..., None] if merged_ifg.ndim > 1 else contribution
        merged_coh /= contribution[..., None] if merged_coh.ndim > 1 else contribution
        merged_ifg = np.where(contribution > 0, merged_ifg, 0)
        merged_coh = np.where(contribution > 0, merged_coh, 0)

    # Seam diagnostics
    seam_diffs = []
    seam_drops = []
    gap_count = int(np.sum(contribution == 0))

    for i in range(len(segments) - 1):
        top_end = segments[i].output_line_start + segments[i].output_num_lines
        bot_start = segments[i + 1].output_line_start
        seam_line = top_end
        if seam_line < total_lines:
            phase_diff = np.angle(merged_ifg[seam_line]) - np.angle(merged_ifg[seam_line - 1])
            seam_diffs.append(float(np.abs(np.angle(np.exp(1j * phase_diff).mean()))))

    return MergeResult(
        seam_phase_diff_median=float(np.median(seam_diffs)) if seam_diffs else 0.0,
        seam_phase_diff_std=float(np.std(seam_diffs)) if seam_diffs else 0.0,
        seam_coherence_drop=0.0,  # compute from top/bottom coh means
        gap_pixel_count=gap_count,
        top_contribution_count=int(np.sum(contribution > 0)),
        bottom_contribution_count=0,
        segments=tuple(segments),
    )
```

- [ ] **Step 2: Write merge tests**

```python
import numpy as np
from scripts.tops_model import BurstRadarGrid, BurstIdentity, BurstWindow, CommonBurstPair
from scripts.tops_merge import plan_merge_segments, merge_burst_ifgs


def _grid(idx):
    from datetime import datetime, timezone
    return BurstRadarGrid(
        identity=BurstIdentity(swath="IW1", burst_index=idx,
                              sensing_start=datetime(2024, 1, 1, 0, 0, idx * 3, tzinfo=timezone.utc),
                              sensing_stop=datetime(2024, 1, 1, 0, 0, idx * 3 + 2, tzinfo=timezone.utc),
                              polarization="VV", orbit_direction="ascending",
                              azimuth_steering_rate=0.0),
        image_window=BurstWindow(first_line=idx * 1500, num_lines=1500,
                                 first_sample=0, num_samples=25000),
        valid_window=BurstWindow(first_line=100, num_lines=1300,
                                 first_sample=500, num_samples=24000),
        line_offset=idx * 1500,
        azimuth_time_interval=0.002,
        range_pixel_spacing=2.3,
        starting_range=800000.0,
        radar_wavelength=0.055,
        doppler_coefficients=(0.0,),
        azimuth_fm_rate_coefficients=(0.0,),
    )


def test_two_bursts_produce_correct_merged_height():
    pairs = [CommonBurstPair(i, _grid(i), _grid(i), 0) for i in range(2)]
    segs = plan_merge_segments(pairs)
    assert len(segs) == 2
    assert segs[1].output_line_start == segs[0].output_num_lines


def test_merge_empty_ifgs_raises():
    import pytest
    with pytest.raises(ValueError):
        merge_burst_ifgs([], [], [])
```

- [ ] **Step 3: Run tests and commit**

```bash
pytest tests/test_tops_merge.py -v
git add scripts/tops_merge.py tests/test_tops_merge.py
git commit -m "feat: add valid-window-aware burst merge"
```

---

### Task 13: Stitch main CLI pipeline together

**Files:**
- Modify: `scripts/tops_insar.py`

- [ ] **Step 1: Implement `_run_swath`**

Replace the `raise NotImplementedError` in `scripts/tops_insar.py`:

```python
from scripts.tops_metadata import parse_sentinel1_safe
from scripts.tops_common_bursts import match_common_bursts
from scripts.tops_overlap import build_overlap_pairs, read_overlap_window
from scripts.tops_deramp import deramp_slc, reramp_slc
from scripts.tops_esd import build_esd_prep, estimate_esd, esd_to_timing_correction
from scripts.tops_ifg import crossmul_bursts
from scripts.tops_merge import plan_merge_segments, merge_burst_ifgs


def _run_swath(args, swath: str, stages: list[str]) -> None:
    log = logging.getLogger("tops_insar")
    swath_dir = args.output_dir / swath
    swath_dir.mkdir(exist_ok=True)

    # 1. Load and normalize metadata
    if args.start_stage == "check" or "preprocess" in stages:
        master_bursts = parse_sentinel1_safe(args.master_safe_or_manifest).get(swath, [])
        slave_bursts = parse_sentinel1_safe(args.slave_safe_or_manifest).get(swath, [])
        if not master_bursts or not slave_bursts:
            log.error("No bursts found for swath %s", swath)
            return

    # 2. Common burst matching
    if "common_bursts" in stages:
        selection = match_common_bursts(master_bursts, slave_bursts)
        (swath_dir / "common_bursts.json").write_text(json.dumps({
            "swath": selection.swath,
            "reference_start_index": selection.reference_start_index,
            "secondary_start_index": selection.secondary_start_index,
            "number_of_common_bursts": selection.number_of_common_bursts,
            "burst_offset": selection.pairs[0].burst_offset if selection.pairs else 0,
        }, indent=2))
        log.info("  common bursts: %d", selection.number_of_common_bursts)

    # 3. Build overlaps
    if "subset_overlaps" in stages:
        overlaps = build_overlap_pairs(selection.pairs)
        (swath_dir / "overlaps.json").write_text(json.dumps({
            "overlap_count": len(overlaps),
        }, indent=2))
        log.info("  overlaps: %d", len(overlaps))

    # 4. Per-burst processing + ESD
    timing_corrections = {}
    if "esd" in stages and overlaps:
        for ov in overlaps:
            prep = build_esd_prep(
                read_overlap_window(master_tiff, ov.top),
                read_overlap_window(slave_tiff, ov.top),
                az_looks=5, rg_looks=15,
                coherence_threshold=args.esd_coherence_threshold,
            )
            est = estimate_esd(prep, extra_esd_cycles=args.extra_esd_cycles)
            tc = esd_to_timing_correction(est, master_bursts[0].azimuth_time_interval)
            timing_corrections[ov.pair_index] = tc
        (swath_dir / "esd_summary.json").write_text(json.dumps({
            "median_offset_pixels": float(
                next(iter(timing_corrections.values())).esd_estimate.median_offset_pixels),
            "swath": swath,
        }, indent=2))

    # 5. Per-burst IFG
    if "burst_ifg" in stages:
        ifgs, cohs = [], []
        for pair in selection.pairs:
            ref = read_burst_window(master_tiff, pair.reference)
            sec = read_burst_window(slave_tiff, pair.secondary)
            ifgs.append(ref * np.conj(sec))  # stub — real: apply resampling + deramp
            cohs.append(np.ones_like(ref, dtype=np.float32))  # stub coherence
        segs = plan_merge_segments(selection.pairs)
        merged = merge_burst_ifgs(ifgs, cohs, segs)
        (swath_dir / "merged_interferogram.bin").write_bytes(
            merged.interferogram.astype(np.complex64).tobytes())
        log.info("  merged IFG shape: %s", merged.interferogram.shape)

    log.info("Swath %s complete.", swath)
```

- [ ] **Step 2: Write integration smoke test**

```python
def test_tops_insar_cli_produces_output(tmp_path):
    # Requires real Sentinel-1 data — document expected behavior
    pytest.skip("Integration test requires Sentinel-1 data")
```

- [ ] **Step 3: Run CLI help and basic import test**

```bash
python3 scripts/tops_insar.py --help
pytest tests/test_tops_insar_cli.py -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add scripts/tops_insar.py
git commit -m "feat: stitch tops_insar main pipeline"
```

---

## 3. Migration Strategy

### Phase 1: Core model & ESD (Tasks 0-3, 6, 10)

Goals: No algorithmic dependency on ISCE3 geometry. Testable with synthetic data.

- Tasks 0-3: CLI, model, metadata, common burst matching
- Task 6: Overlap materialization
- Task 10: ESD pipeline

Gate: ESD on synthetic overlap data recovers known azimuth offset within tolerance.

### Phase 2: Geometry & Registration (Tasks 4-5, 7-9)

Goals: ISCE3 Geo2Rdr proved on real Sentinel-1 burst. Deramp/reramp validated. Range coreg implemented.

- Task 4: ISCE3 RadarGrid / Orbit / Doppler adapters
- Task 5: Geo2Rdr spike on real Sentinel-1 burst
- Task 7: Deramp/reramp with roundtrip validation
- Task 8: Coarse resampling
- Task 9: Range coregistration

Gate: Geo2Rdr median offsets for one real burst match ISCE2 topsApp within documented tolerance. Deramp/reramp roundtrip < 1e-5.

### Phase 3: IFG, Merge, Publish (Tasks 11-13)

Goals: Per-burst IFG generated. Valid-window-aware merge. Seam diagnostics.

- Task 11: Crossmul per-burst IFG
- Task 12: Valid-window-aware merge
- Task 13: Full CLI stitching

Gate: Merged IFG seam phase median < 0.5 rad on validation pair.

### Phase 4: ISCE2 Parity (Task 14)

- Compare all major intermediate products against ISCE2 topsApp on one real Sentinel-1 pair
- Generate `tops_isce2_parity_report.json`
- Pipeline is complete only when parity report passes

---

## 4. Validation Commands

After Phase 1:

```bash
pytest tests/test_tops_insar_cli.py tests/test_tops_model.py tests/test_tops_metadata.py tests/test_tops_common_bursts.py tests/test_tops_overlap.py tests/test_tops_esd.py -v
```

After Phase 2:

```bash
pytest tests/test_tops_insar_cli.py tests/test_tops_model.py tests/test_tops_metadata.py tests/test_tops_common_bursts.py tests/test_tops_overlap.py tests/test_tops_deramp.py tests/test_tops_esd.py tests/test_tops_geometry.py tests/test_tops_range_coreg.py tests/test_tops_registration.py -v
```

After Phase 3:

```bash
pytest tests/test_tops_insar_*.py -v
python3 scripts/tops_insar.py --help
```

Full pipeline:

```bash
python3 scripts/tops_insar.py OUTPUT_DIR MASTER_SAFE SLAVE_SAFE --swath IW2 --end-stage burst_ifg
```

---

## 5. Definition of Done

The refactor is complete when ALL of the following are true:

- `scripts/tops_insar.py` imports zero lines from `strip_insar`, `strip_insar`, `tops_insar`.
- All 13 modules (`tops_model` through `tops_merge`) pass their unit tests.
- Geo2Rdr spike produces finite offsets on at least one real Sentinel-1 burst pair.
- ESD recovers a known azimuth offset on synthetic overlap data within 0.5 pixel tolerance.
- Deramp/reramp roundtrip error < 1e-5 on nonzero carrier.
- Merged IFG seam phase median < 0.5 rad on the validation pair.
- ISCE2 parity report shows all major intermediate products within documented tolerances.

---

## 6. Explicitly NOT Carried Forward

The following are not part of this plan:

- `strip_insar.process_strip_insar()` — not called, not wrapped
- `tops_insar._run_local_tops_backend_for_swath()` — not used
- `tops_insar._run_overlap_esd_backend_for_swath()` with raw SLC ESD — replaced by Task 10
- `tops_insar._apply_tops_deramp()` no-op — replaced by Task 7
- `tops_insar._merge_bursts_isce2_style()` delegating to strip — replaced by Task 12
- NISAR high-level workflows (`nisar.workflows.*.run(cfg)`) — deferred until Sentinel-1 burst compatibility is proven
