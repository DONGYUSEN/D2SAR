# tops_insar2 代码完善计划

## 目标
系统梳理当前 tops_insar2 实现情况，识别缺失功能，补全实现。

## 审查范围（根据 PLAN.md）

### 模块接口核对清单

| # | 模块 | 状态 | 说明 |
|---|---|---|---|
| 1 | tops_model.py | ✅ 已实现 | 15 个 frozen dataclass |
| 2 | tops_metadata.py | ✅ 已实现 | SAFE/XML 解析 |
| 3 | tops_common_bursts.py | ✅ 已实现 | global integer offset 匹配 |
| 4 | tops_geometry.py | ✅ 已实现 | ISCE3 orbit/doppler/Geo2Rdr |
| 5 | tops_overlap.py | ✅ 已实现 | top/bottom overlap 物化 |
| 6 | tops_deramp.py | ✅ 已实现 | TOPS deramp/reramp |
| 7 | tops_registration.py | 🔶 部分 | coarse resamp OK, fine_resamp 待检查 |
| 8 | tops_range_coreg.py | ✅ 已实现 | range 配准 |
| 9 | tops_esd.py | ✅ 已实现 | ESD 时序校正 |
| 10 | tops_ifg.py | ✅ 已实现 | cross-multiply IFG |
| 11 | tops_merge.py | ✅ 已实现 | valid-window merge |
| 12 | tops_utils.py | ✅ 已实现 | 共享工具 |
| 13 | tops_publish.py | ✅ 已实现 | geocode/publish |
| 14 | tops_ionosphere.py | ✅ 已实现 | split-band ionosphere |
| 15 | tops_insar2.py | 🔶 部分 | CLI OK, _run_swath 待检查 |

## 待深入检查项

### A. tops_registration.py — fine_resamp 是否实现？
- PLAN: `fine_resample_with_timing` 函数应存在
- 当前实现: 只有 `run_coarse_registration`

### B. tops_insar2.py — 完整 stage 实现
- PLAN Section 1.3 pipeline flow vs 当前实现
- 检查 `_run_swath` 各 stage 是否完整

### C. ISCE3 ResampSlc 是否实现？
- PLAN: deramp → ISCE3 ResampSlc(coarse) → reramp
- 当前 tops_registration.py 用 scipy.ndimage 替代

### D. tops_overlap.py — overlap SLC 读取是否实现？
- PLAN: `read_overlap_window(tiff_path, slice)` 应从全 swath TIFF 读取
- 当前: `materialize_overlaps` 只返回 OverlapPair，未读取实际 SLC 数据

### E. tops_registration.py — coarse_resamp 读取 offsets
- 当前: `run_coarse_registration` 假设 offsets 已在 work_dir
- topo stage 应生成 offsets，但 Geo2Rdr 输出路径需确认

## 阶段划分

### 阶段 1: 审计 (Audit)
- [ ] 逐模块核对 PLAN.md 接口
- [ ] 确认每个函数/类的输入/输出/行为
- [ ] 记录缺失项

### 阶段 2: 补全 (Complete)
- [ ] 补全缺失的函数
- [ ] 补全缺失的测试
- [ ] 验证 pipeline 端到端

### 阶段 3: 验证 (Verify)
- [ ] 全量测试通过
- [ ] 无 strip_insar 导入
- [ ] commit

## 关键发现记录 → findings.md
## 进度 → progress.md