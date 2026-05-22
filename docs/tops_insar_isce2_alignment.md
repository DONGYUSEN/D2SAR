# D2SAR TOPS InSAR 与 ISCE2 对齐状态

本文档记录 D2SAR `tops_insar.py` 与 ISCE2 TOPS 处理流程的对齐状态。

---

## 1. 核心干涉处理流程（已完全对齐）

| 功能 | ISCE2 参照 | D2SAR 实现 | 状态 |
|------|-----------|-----------|------|
| Burst 干涉图生成 | `runBurstIfg.py multiply()` | `_compute_burst_interferograms()` | ✅ 完全对齐 |
| Range/Azimuth 偏移应用 | 同上 | `_resample_complex_with_offsets()` | ✅ 完全对齐 |
| Flatten 地形相位校正 | `phs = exp(-j*4π*rngOff*r0/λ)` | `ifg *= phs` (line 1645-1648) | ✅ 完全对齐 |
| `adjustValidLineSample` | `runBurstIfg.py` 137-151 | `adjustValidLineSample()` (line 1174) | ✅ 完全对齐 |
| 相干性计算 | `Correlation` class | `_estimate_burst_coherence()` | ✅ 完全对齐 |
| 多视处理 | `mroipac.looks.Looks` | `_multilook_mean_isce2_style()` | ✅ 完全对齐 |
| `azReferenceOff` 计算 | `sensingStart + firstValidLine*dt` | `_compute_az_reference_offsets()` | ✅ 完全对齐 |
| ESD 局部频率估算 | 逐像素调频率 | `_estimate_esd_local_frequency()` | ✅ 完全对齐 |
| ESD 偏移计算 | `off = (angle + extra) / freq` | `_compute_esd_spectral_diversity()` | ✅ 完全对齐 |
| Secondary timing correction 存储 | `secondaryTimingCorrection` | `_store_secondary_timing_correction()` | ✅ 完全对齐 |
| Secondary timing 应用 | 反馈到后续处理 | `_shift_complex_azimuth()` (line 1631-1634) | ✅ 完全对齐 |

---

## 2. 电离层校正框架（部分实现）

### ISCE2 vs D2SAR 对照

| ISCE2 函数 | D2SAR 实现 | 状态 | 说明 |
|-----------|-----------|------|------|
| `subband` | `_split_subband()` | ✅ 已实现 | 子带分裂，1/3-1/3-1/3 方案 |
| `rawion` | `_estimate_raw_ionosphere()` | ✅ 已实现 | 基于上下子带干涉图差异估算电离层 |
| `grd2ion` | `_grd2ion()` | ⚠️ Placeholder | 地理坐标转换，保留框架 |
| `filt_gaussian` | `_filter_ionosphere()` | ✅ 已实现 | 高斯滤波 + 拟合 |
| `ionosphere_shift` | `_compute_ionosphere_shift()` | ✅ 已实现 | 电离层相位转方位向偏移 |
| `ion2grd` | `_ion2grd()` | ⚠️ Placeholder | 网格化，保留框架 |
| ESD in ion | `_esd()` | ⚠️ Placeholder | 电离层 ESD 校正，保留框架 |
| 主流程 | `_run_ionospheric_correction()` | ✅ 框架完整 | 7 步骤框架已就绪 |

### Placeholder 说明

以下函数当前为框架/占位实现，如需完整电离层校正功能需进一步完善：

1. **`_grd2ion()`** - 地理坐标转换
   - 当前：保留地理参考信息与电离层相位
   - 需完善：与 ISCE2 runIon.py 的 `grd2ion` 步骤对齐

2. **`_ion2grd()`** - 电离层网格化
   - 当前：输出与网格对齐的电离层产品框架
   - 需完善：与 ISCE2 runIon.py 的 `ion2grd` 步骤对齐

3. **`_esd()`** - 电离层 ESD 校正
   - 当前：保留框架
   - 需完善：与 ISCE2 runIon.py 的 `esd` 步骤对齐

---

## 3. 流程对照

```
ISCE2                              D2SAR
───────────────────────────────────────────────────────────────────────
runBurstIfg                       _compute_burst_interferograms
  - range/azimuth offsets           ✓ range/azimuth offsets
  - flatten                        ✓ flatten
  - multilook                      ✓ multilook
  - coherence                      ✓ coherence
  - secondary timing               ✓ secondary timing (stored + applied)

runMergeBursts                    _merge_bursts_isce2_style
  - azReferenceOff                 ✓ azReferenceOff
  - burst merge (top/bot/avg)      ✓ burst merge
  - mergeBox                       ✗ (功能等效，无独立函数)

runESD                             _compute_esd_spectral_diversity
  - per-pixel frequency            ✓ _estimate_esd_local_frequency
  - ESD offset calculation         ✓ (angle + extra) / freq
  - coherence mask                 ✓ coherence > threshold
  - secondaryTimingCorrection      ✓ _store_secondary_timing_correction

runIon                             _run_ionospheric_correction
  - subband                        ✓ _split_subband
  - rawion                         ✓ _estimate_raw_ionosphere
  - grd2ion                        ⚠️ placeholder
  - filt_gaussian                  ✓ _filter_ionosphere
  - ionosphere_shift               ✓ _compute_ionosphere_shift
  - ion2grd                        ⚠️ placeholder
  - esd                            ⚠️ placeholder
```

---

## 4. 剩余微小差异

| 差异 | 说明 | 影响 |
|------|------|------|
| `mergeBox` 函数 | 直接用 burst valid window 叠加，无独立函数 | 功能等效 |
| Catalog/logging | 使用 JSON 记录而非 ISCE2 Catalog | 不影响结果 |
| 电离层 `_esd()` | Placeholder | 需完善以支持完整电离层校正 |

---

## 7. 2026-05-11 实测修正记录

在真实 Sentinel-1 IW1 burst-limit=2 数据上验证后，已修正以下偏差：

1. **Overlap 计算**
   - 早期按错误的边界公式计算，导致 overlap 退化为零数组。
   - 现已改为 valid-window 交集，`overlap_ifg` 可稳定生成真实窗口数据。

2. **Fine resample**
   - 早期未尊重 `--burst-limit`，会继续处理未选中的 burst。
   - 现已与 `_limited_pairs(common, state)` 对齐。

3. **RD 镶嵌**
   - 早期 `merge_bursts` 按 `valid_window` 直接放置，产生额外空白和错误 seam。
   - 现已按 RD 坐标系布局，实测输出 `gap_pixels=0`。

4. **ESD 统计**
   - `EsdEstimate` 现包含 `mean_coherence`，与 `compute_esd_timing_correction()` 保持一致。

5. **实测结果**
   - `overlap_ifg`：`coherence_mean=1.000`
   - `merge_bursts`：`shape=(2796, 20470)`
   - 输出 PNG：`wrapped.png`、`coherence.png`、`unwrapped.png`、`filtered.png`

6. **Geo2Rdr / Rdr2Geo 链路**
   - 已确认 DEM 在 ISCE3 中仍应作为**单波段输入**，仅用于 `Rdr2Geo.topo(dem, ...)`。
   - `Geo2Rdr` 的正确输入不是 DEM，而是 `Rdr2Geo` 生成的 `topo.vrt`，其前三个波段为 `x/y/z`。
   - `scripts/tops_geometry.py` 已改为“参考 burst 先 `Rdr2Geo`，二级 burst 再 `Geo2Rdr`”的两段式链路。
   - `Geo2Rdr` 输出中的 `-1e6` 无效值已从统计中排除，避免把未收敛像元计为有效样本。
   - `run_geo2rdr_single_burst()` 现在会在 CUDA 绑定不可用时自动回退 CPU。

---

## 5. 使用说明

### 启用电离层校正

```python
from tops_insar import IonosphericParams, _run_ionospheric_correction

ion_params = IonosphericParams()
ion_params.do_ion = True
ion_params.ion_height = 200.0  # km
ion_params.ion_fit = True

result = _run_ionospheric_correction(plan, context, master_bursts, slave_bursts, ion_params)
```

### CLI 参数

```bash
python3 scripts/tops_insar.py ... --do-ionospheric-correction
```

---

## 6. 参考

- ISCE2: `isce2/components/isceobj/TopsProc/runBurstIfg.py`
- ISCE2: `isce2/components/isceobj/TopsProc/runMergeBursts.py`
- ISCE2: `isce2/components/isceobj/TopsProc/runESD.py`
- ISCE2: `isce2/components/isceobj/TopsProc/runIon.py`
- D2SAR: `scripts/tops_insar.py`

---

*最后更新: 2026-05-05*
