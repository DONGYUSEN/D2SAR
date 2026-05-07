# tops_insar.py 规划与评估（Sentinel-1 TOPS InSAR）

## 1. 目标与范围

目标：新增统一入口 `scripts/tops_insar.py`，用于 Sentinel-1 TOPS 干涉处理，兼容当前仓库 `manifest` 数据契约，并借鉴 ISCE2 `topsApp.py` 的阶段化处理模式。

本阶段仅完成设计与评估，不实现代码。

---

## 2. 现状评估（D2SAR）

### 2.1 可复用能力

当前仓库已有较完整组件，可直接复用：

1. Sentinel 导入与元数据
- `scripts/sentinel_importer.py`
- `scripts/sentinel_orbit.py`
- `scripts/tops_geometry.py`

2. TOPS RTC 统一封装实践（可复用其工程化模式）
- `scripts/tops_rtc.py`
- 已具备：`--product-path`、`--swath`、静默日志、分阶段输出结构、swath 合并策略

3. InSAR 核心算法与阶段框架（strip 版）
- `scripts/strip_insar2.py`
- 已具备：阶段机（`check/prep/crop/p0..p6`）、CPU/GPU回退、阶段缓存、产品导出、unwrap 后端

### 2.2 当前缺口

1. 没有 TOPS 专用 InSAR 统一入口（只有 strip 与 tops_rtc）
2. 还没有把 TOPS burst/overlap/merge 语义与现有 `strip_insar2` 阶段体系打通
3. 多 swath（IW1/2/3）在 InSAR 侧缺少统一合并与一致性约束

---

## 3. ISCE2 参考模式（借鉴点）

参考：
- `~/Software/isce/isce2/applications/topsApp.py`
- `~/Software/isce/isce2/contrib/stack/topsStack/README.md`

`topsApp.py` 关键步骤（简化）：
1. preprocess
2. computeBaselines
3. verifyDEM
4. topo
5. subset overlaps / coarse offsets / coarse resamp
6. overlap ifg / prep ESD / ESD / range coreg
7. fine offsets / fine resamp
8. burst ifg
9. merge bursts
10. filter
11. unwrap
12. geocode

对 D2SAR 的启发：
1. 保留阶段化、可断点重跑、每阶段可验收
2. 先 burst 域完成配准与干涉，再 merge，再滤波/解缠/地理编码
3. ESD、overlap 相关能力建议在 MVP 中先做“可选/占位”，先确保主链可跑通

---

## 4. tops_insar.py 建议架构

建议采用“两层设计”：

1. Orchestration 层：`tops_insar.py`
- 负责 CLI、阶段调度、日志、产物路径、swath 批处理与合并

2. Compute 层：复用现有模块
- 复用 `strip_insar2.py` 内已成熟的注册、crossmul、filter、unwrap、geocode/HDF 发布函数
- 新增 TOPS 特有 bridge（burst 计划、merge、overlap/ESD 可选）

---

## 5. 阶段设计（建议 v1）

对齐 ISCE2 思路并兼容 D2SAR 风格，建议：

1. `check`
- 输入合法性、主辅景时间/轨道检查、swath 可用性

2. `prep`
- 导入/读取 manifest（支持 `--product-path` 与 manifest 两种入口）
- 解析 IW swath 与 burst 元数据

3. `p0_topo_geo2rdr`
- 生成 topo 与几何初值（支持 `--topo-gpu`）

4. `p1_coreg`
- 粗配准 + 精配准 + （可选）ESD 修正

5. `p2_ifg`
- burst interferogram / coherence / flatten / goldstein

6. `p3_merge`
- burst merge（同 swath）
- 多 swath：先分 swath 产物，后按规则合并（IW1+IW2、IW2+IW3）

7. `p4_unwrap`
- ICU / snaphu / dolphin（与 `strip_insar2` 同策略）

8. `p5_geocode_hdf`
- 地理编码与 HDF 产品组织

9. `p6_publish`
- GeoTIFF/PNG/KML/元数据汇总，输出结果 JSON

---

## 6. CLI 设计建议

建议与 `tops_rtc.py` 风格统一：

1. 输入模式
- `tops_insar.py <master_manifest> <slave_manifest> <output_dir>`
- `tops_insar.py <output_dir> --master-product-path ... --slave-product-path ...`

2. swath 选择
- `--swath IW1|IW2|IW3|IW1,IW2|IW2,IW3|IW1,IW3|all`
- `IW1,IW3` 给 warning，不做跨 swath 合并（与 tops_rtc 一致）

3. 阶段控制
- `--start-stage --end-stage --resume`
- `--burst-limit`（调试）

4. 后端控制
- `--gpu-mode auto|gpu|cpu`
- `--topo-gpu --gpu-id`

5. 算法配置
- `--range-looks --azimuth-looks`
- `--esd on|off`
- `--unwrap-method icu|snaphu|dolphin`

---

## 7. 数据与产物契约（建议）

单 swath 典型输出：
1. `tops_insar_plan.json`
2. `burst_*` 中间产物（ifg/coh/offset 等）
3. swath merged：
- `mosaic_interferogram_radar.h5`
- `mosaic_coherence_radar.h5`
- `mosaic_unwrapped_phase_radar.h5`
4. geocoded：
- `mosaic_interferogram_geocoded.tif/png`
- `mosaic_coherence_geocoded.tif/png`
- `mosaic_unwrapped_phase_geocoded.tif/png`

多 swath 输出：
1. `IW1/`, `IW2/`, `IW3/` 各自完整链路
2. 邻接 swath 合并产物（若满足规则）

---

## 8. MVP 边界（第一版必须/可后置）

### 8.1 第一版必须
1. 单 swath（IW1 或 IW2 或 IW3）主链跑通
2. 支持 `--product-path` 与 manifest 双入口
3. 支持阶段断点续跑
4. 支持基础 unwrap（默认 ICU）
5. 支持 geocoded interferogram/coherence/unwrapped 导出

### 8.2 可后置
1. 完整 ESD 时序优化
2. ionosphere 校正链路
3. dense offsets 全功能
4. 高级网络配对（stack 模式）

---

## 9. 实施顺序建议

1. M1：`check/prep/p0/p1`（先做配准可用）
2. M2：`p2/p3`（ifg + merge）
3. M3：`p4/p5/p6`（unwrap + geocode + publish）
4. M4：多 swath 邻接合并（IW1+IW2、IW2+IW3）
5. M5：ESD 与 ion 相关增强

---

## 10. 风险与对策

1. TOPS burst overlap 处理复杂，易出现缝与相位跳变
- 对策：先复用现有 merge 规则，建立 seam 质量检查

2. GPU/CPU 混合链路日志噪声与不一致
- 对策：统一静默包装 + 结构化进度输出（参考 tops_rtc 已实现）

3. 多 swath 非邻接拼接误用（IW1+IW3）
- 对策：CLI 明确 warning 并禁用自动合并

4. 算法参数组合过多导致可维护性下降
- 对策：先定义稳定默认参数集，再开放高级参数

---

## 11. 与 ISCE2 模式的对齐结论

结论：可采用“ISCE2 阶段语义 + D2SAR 工程化封装”的路线。

即：
1. 算法步骤对齐 `topsApp`（preprocess/coreg/ifg/merge/filter/unwrap/geocode）
2. 工程实现沿用 D2SAR 现有模式（manifest 契约、阶段缓存、CLI 统一、GPU/CPU回退）
3. 先单 swath 稳定，再做多 swath 邻接合并和高级物理改正

---

## 12. 下一步（文档后）

下一步建议产出实现计划文档（task-level）：
1. 文件级改动清单（新建/复用/重构）
2. 单元测试清单（按阶段）
3. 首个可运行命令与验收标准（含 IW1 最小样例）
