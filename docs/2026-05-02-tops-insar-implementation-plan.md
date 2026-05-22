# tops_insar.py 实施任务清单（按文件 + 测试 + 验收）

对应设计文档：
- `docs/2026-05-02-tops-insar-design.md`

## 0. 目标与完成定义

目标：交付 `scripts/tops_insar.py` MVP，支持 Sentinel-1 TOPS 单对干涉处理，具备：
1. `manifest` 与 `product-path` 双入口；
2. `--swath` 选择（单值/子集/all）；
3. 阶段化执行与断点续跑；
4. 产出 merged/geocoded 干涉结果；
5. 基础 unwrap（ICU）与结构化 JSON 输出。

完成定义（DoD）：
1. `python3 -m py_compile` 全部通过；
2. 新增测试通过；
3. IW1 最小真实样例链路跑通；
4. 结果目录结构与文档一致。

---

## 1. 阶段拆解与任务

## Phase A：CLI 与计划层（不触碰重算法）

### A1. 新建 `tops_insar.py` 入口骨架
文件：
1. 新建 `scripts/tops_insar.py`

任务：
1. 定义参数：
   - `manifest` 模式：`master_manifest slave_manifest output_dir`
   - `product` 模式：`--master-product-path --slave-product-path`
   - `--swath IW1|IW2|IW3|IW1,IW2|IW2,IW3|IW1,IW3|all`
   - `--start-stage --end-stage --resume`
   - `--gpu-mode --topo-gpu --gpu-id`
   - `--dem --burst-limit --resolution --unwrap-method`
2. 统一输出 JSON 根结构（mode/inputs/stages/outputs/warnings/errors）。

测试：
1. 新建 `tests/test_tops_insar_cli.py`
2. 覆盖参数合法性、冲突校验、`IW1,IW3` warning 规则。

验收命令：
1. `python3 -m py_compile scripts/tops_insar.py`
2. `python3 -m pytest tests/test_tops_insar_cli.py -q`

---

### A2. 计划文件生成（prepare-only）
文件：
1. 修改 `scripts/tops_insar.py`
2. （可选）新增 `scripts/tops_insar_plan.py`（若拆分）

任务：
1. 生成 `tops_insar_plan.json`：
   - master/slave manifest 路径；
   - swath 列表；
   - 每 swath burst 计数；
   - 输出路径模板；
   - stage 配置快照。
2. 若 `--resume`，读取并校验已有 plan。

测试：
1. 新增 `tests/test_tops_insar_plan.py`
2. 覆盖 plan 写入/读取、resume 校验失败分支。

验收命令：
1. `python3 -m pytest tests/test_tops_insar_plan.py -q`

---

## Phase B：单 swath 核心链路（MVP）

### B1. check/prep/crop/p0 接线
文件：
1. 修改 `scripts/tops_insar.py`
2. 复用 `scripts/sentinel_importer.py`、`scripts/tops_geometry.py`
3. 复用 `scripts/strip_insar.py` 中已稳定工具函数（通过导入）

任务：
1. 实现 stage runner（与 `strip_insar` 同风格）；
2. 先落地 `check/prep/p0`，产出可检查的中间文件与 stage record；
3. topo 阶段支持 `--topo-gpu` 与静默日志包装。

测试：
1. 新增 `tests/test_tops_insar_stages_a.py`
2. mock 方式验证 stage 顺序、resume、输出记录。

验收命令：
1. `python3 -m pytest tests/test_tops_insar_stages_a.py -q`

---

### B2. p1/p2/p3（coreg/ifg/merge）
文件：
1. 修改 `scripts/tops_insar.py`
2. 必要时新增 `scripts/tops_insar_merge.py`

任务：
1. 复用 `strip_insar` 的配准/crossmul/filter 能力；
2. 在 TOPS 语义下完成 burst 到 swath merge；
3. 输出：
   - `mosaic_interferogram_radar.h5`
   - `mosaic_coherence_radar.h5`

测试：
1. 新增 `tests/test_tops_insar_merge.py`
2. 覆盖 overlap 拼接与形状一致性。

验收命令：
1. `python3 -m pytest tests/test_tops_insar_merge.py -q`

---

### B3. p4/p5/p6（unwrap/geocode/publish）
文件：
1. 修改 `scripts/tops_insar.py`
2. 复用 `scripts/common_processing.py` 的 geocode 输出函数

任务：
1. unwrap（默认 ICU，预留 snaphu/dolphin）；
2. geocoded 导出 tif/png；
3. 发布阶段写 summary JSON 与最终输出索引。

测试：
1. 新增 `tests/test_tops_insar_publish.py`
2. 覆盖输出文件路径与元数据字段。

验收命令：
1. `python3 -m pytest tests/test_tops_insar_publish.py -q`

---

## Phase C：多 swath 编排与合并策略

### C1. swath 子集/all 编排
文件：
1. 修改 `scripts/tops_insar.py`

任务：
1. 支持 `IW1/IW2/IW3` 单 swath；
2. 支持 `IW1,IW2` 与 `IW2,IW3` 两 swath；
3. 支持 `all`（IW1/2/3 顺序处理）；
4. `IW1,IW3`：给 warning，不做跨 swath 合并。

测试：
1. 新增 `tests/test_tops_insar_swath_selection.py`
2. 覆盖上述 4 类选择逻辑与 warning 文案。

验收命令：
1. `python3 -m pytest tests/test_tops_insar_swath_selection.py -q`

---

### C2. 跨 swath 合并（仅邻接）
文件：
1. 修改 `scripts/tops_insar.py`

任务：
1. 邻接 swath geocoded 干涉产品合并（VRT + GeoTIFF）；
2. 合并输入来自每 swath `mosaic_*_geocoded.tif`；
3. 将 merge 结果写入 `swath_merge` 字段。

测试：
1. 新增 `tests/test_tops_insar_swath_merge.py`
2. mock GDAL 合并调用与结果路径。

验收命令：
1. `python3 -m pytest tests/test_tops_insar_swath_merge.py -q`

---

## 2. 建议文件清单（最终）

新增：
1. `scripts/tops_insar.py`
2. `tests/test_tops_insar_cli.py`
3. `tests/test_tops_insar_plan.py`
4. `tests/test_tops_insar_stages_a.py`
5. `tests/test_tops_insar_merge.py`
6. `tests/test_tops_insar_publish.py`
7. `tests/test_tops_insar_swath_selection.py`
8. `tests/test_tops_insar_swath_merge.py`

可选新增（若为降低耦合）：
1. `scripts/tops_insar_plan.py`
2. `scripts/tops_insar_merge.py`

需更新：
1. `docs/README.md`（新增 tops_insar 用法）

---

## 3. 里程碑验收命令（建议顺序）

1. 语法检查
```bash
python3 -m py_compile scripts/tops_insar.py
```

2. 单测（先新测后回归）
```bash
python3 -m pytest \
  tests/test_tops_insar_cli.py \
  tests/test_tops_insar_plan.py \
  tests/test_tops_insar_stages_a.py \
  tests/test_tops_insar_merge.py \
  tests/test_tops_insar_publish.py \
  tests/test_tops_insar_swath_selection.py \
  tests/test_tops_insar_swath_merge.py -q
```

3. 与现有 Sentinel/TOPS 回归（避免破坏）
```bash
python3 -m pytest \
  tests/test_sentinel_importer.py \
  tests/test_sentinel_orbit.py \
  tests/test_tops_rtc.py -q
```

4. 真实最小样例（IW1）
```bash
python3 scripts/tops_insar.py \
  --master-product-path /temp/S1A_...zip \
  --slave-product-path /temp/S1A_...zip \
  --swath IW1 \
  /results/tops_insar_iw1 \
  --dem /temp/s1/proc/dem/dem.tif \
  --topo-gpu --gpu-id 0
```

---

## 4. 风险控制点（执行时必须检查）

1. burst 有效窗一致性（主辅景、跨日期）
2. merge seam 连续性（相位跳变/空洞）
3. unwrap 失败回退策略（ICU -> snaphu/dolphin）
4. 多 swath 非邻接误合并保护（IW1,IW3）
5. 日志静默不吞异常（仅屏蔽噪声，不屏蔽错误）

---

## 5. 立即执行建议

按优先级：
1. 先做 Phase A（CLI + plan）；
2. 再做 Phase B（单 swath MVP）；
3. 最后做 Phase C（多 swath 编排/邻接合并）。

---

## 6. 实测回放（2026-05-11）

使用 `/home/ysdong/Temp` 下真实 Sentinel-1 IW1 数据，在 Docker 中完成 burst-limit=2 的处理验证。实测中逐项修复了以下问题：
1. `overlap_ifg` 的重叠窗口边界计算错误，导致 overlap 退化为零数组；修正为 valid-window 交集后恢复真实 overlap；
2. `fine_resamp` 阶段未尊重 `--burst-limit`，导致调试样例外的 burst 被误处理；已限制为 `_limited_pairs(common, state)`；
3. `merge_bursts` 早期按 `valid_window` 直接放置，导致输出并非 RD 镶嵌；已改为按 RD 坐标系布局，使用 `plan_merge_segments()` 生成 segment；

## 7. 几何链路修正任务（2026-05-12）

`tops_insar.py` 的 burst 级粗配准实现必须遵守 ISCE3 `rdr2geo/geo2rdr` 输入输出契约：
1. `DEM` 保持单波段，只作为 `Rdr2Geo.topo()` 输入；
2. `Rdr2Geo` 在参考 burst 的 RD 网格上输出 `topo.vrt`；
3. `Geo2Rdr` 在 secondary burst 的 RD 网格上读取该 `topo.vrt`，输出 secondary 相对参考 RD 网格的 range/azimuth offsets；
4. 禁止用“复制 DEM 为三波段”的方式伪造 `Geo2Rdr` 输入；
5. `Geo2Rdr` 输出的 `-1e6` 无效像元不参与 offset 统计；
6. 若 `--gpu-mode auto` 下 CUDA geometry 绑定不可用，应回退 CPU，避免真实数据批处理因环境差异中断。
4. `EsdEstimate` 增加 `mean_coherence` 字段后，`compute_esd_timing_correction()` 已同步更新。

实测结果：
1. `overlap_ifg` 成功读取真实窗口，`coherence_mean=1.000`；
2. `prep_esd`、`esd`、`range_coreg`、`fineoffsets`、`fine_resamp` 正常执行；
3. `merge_bursts` 输出 `shape=(2796, 20470)`，`gap_pixels=0`；
4. PNG 已导出：`wrapped.png`、`coherence.png`、`unwrapped.png`、`filtered.png`。
