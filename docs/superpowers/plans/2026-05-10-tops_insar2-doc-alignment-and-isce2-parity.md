# tops_insar.py 文档对齐与 ISCE2 主链路补齐实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `scripts/tops_insar.py` 的阶段定义、数据流和关键处理行为与文档及 ISCE2 `topsApp` 主链路对齐，并逐步补齐缺失的 TOPS 扩展阶段。

**Architecture:** 以 `tops_insar.py` 作为唯一 CLI 编排入口，保留现有按 swath 分发的执行模型，但把文档、stage 顺序、状态流和 ISCE2 对照关系整理成可验证的设计基线。实现上先补齐主链路的真实数据流和错误语义，再按 ISCE2 参考顺序补上 `fineoffsets / ion / unwrap2stage / denseoffsets / filteroffsets / geocodeoffsets` 等扩展阶段，所有新行为都通过现有 `scripts/tops_*.py` 模块承载，避免把算法继续堆进 CLI。

**Tech Stack:** Python, NumPy, GDAL/rasterio, existing D2SAR IO utilities, vendored ISCE3 primitives, pytest, existing `scripts/tops_*.py` modules, ISCE2 `topsApp.py` / `TopsProc` as behavioral reference.

---

## 1. 文件结构与责任边界

### 1.1 需要修改的文件

| 文件 | 责任 |
|---|---|
| `scripts/tops_insar.py` | 调整 stage 序列、stage 分发、错误语义、阶段依赖和 CLI 参数使用情况 |
| `scripts/tops_overlap.py` | 让 overlap 窗口读取与物化行为更贴近真实 burst TIFF 读取 |
| `scripts/tops_registration.py` | 补齐 coarse/fine 配准链路的依赖检查、ESD/ range 校正注入、必要的失败策略 |
| `scripts/tops_esd.py` | 确保 ESD 估计结果真正进入后续时序校正链路 |
| `scripts/tops_publish.py` | 明确 unwrap / geocode / publish 的职责边界，区分中间产品与最终产品 |
| `tests/test_tops_insar_cli.py` | 更新 stage 列表、CLI 参数、帮助文本与 stage 选择行为的测试 |
| `tests/test_tops_insar_pipeline.py` | 增加主链路和扩展 stage 的回归测试，覆盖失败语义与状态传递 |
| `docs/superpowers/plans/2026-05-10-tops_insar-doc-alignment-and-isce2-parity.md` | 保存本实施计划 |

### 1.2 不修改的边界

- 不引入 `strip_insar` / `strip_insar` / `tops_insar` 依赖。
- 不把 `tops_insar.py` 改成算法容器；它只做 CLI 和 stage 编排。
- 不直接切到 `nisar.workflows.*` 作为主实现路径。
- 不把 ISCE2 的完整对象系统照搬进来；只对齐行为和 stage 流。

---

## 2. 现状与目标差异

### 2.1 当前已存在的 stage

`check → preprocess → common_bursts → topo → subset_overlaps → coarse_resamp → overlap_ifg → prep_esd → esd → range_coreg → fine_resamp → burst_ifg → merge_bursts → filter → unwrap → geocode → publish`

### 2.2 ISCE2 对照中需要补齐的阶段

- `fineoffsets`
- `ion`
- `unwrap2stage`
- `denseoffsets`
- `filteroffsets`
- `geocodeoffsets`
- `endup`（若保留为内部收尾动作，需要明确对应关系）

### 2.3 需要修正的关键行为

- `overlap_ifg` 不能继续依赖零填充合成数组。
- `topo` 不能在真实数据路径上静默退化为零偏移。
- `coarse_resamp` 失败时不能让后续阶段无声继续。
- `esd` 的估计结果必须进入后续的 resamp / timing 校正。
- `unwrap` 的 fallback 必须显式记录科学质量降级。
- `geocode` 必须区分“中间地理编码产品生成”和“最终发布”。

---

## 3. 任务拆分

### Task 1: 更新 stage 设计文档与顺序说明

**Files:**
- Modify: `docs/superpowers/plans/2026-05-10-tops_insar-doc-alignment-and-isce2-parity.md`

- [ ] **Step 1: 写出 stage 对照表**

```markdown
| ISCE2 stage | tops_insar.py stage | 当前状态 | 处理说明 |
|---|---|---|---|
| preprocess | preprocess | 已有 | SAFE/manifest 解析 |
| computeBaselines | common_bursts | 已有 | common burst 匹配 |
| verifyDEM | check | 已有 | DEM 存在性检查 |
| topo | topo | 需增强 | 真实 Geo2Rdr，不允许零偏移退化 |
| subsetoverlaps | subset_overlaps | 已有 | overlap 物化 |
| coarseoffsets | coarse_resamp | 需增强 | coarse offsets + resamp |
| coarseresamp | coarse_resamp | 需增强 | coarse resamp |
| overlapifg | overlap_ifg | 需增强 | 真实 TIFF 读取和 crossmul |
| prepesd | prep_esd | 已有/需校正 | ESD 输入准备 |
| esd | esd | 需增强 | ESD correction 真正生效 |
| rangecoreg | range_coreg | 已有/需校正 | range 校正传播 |
| fineoffsets | fine_offsets | 新增 | 密集精配准 |
| fineresamp | fine_resamp | 已有/需增强 | 精配准 + timing correction |
| ion | ion | 新增 | 可选 ionosphere correction |
| burstifg | burst_ifg | 已有/需增强 | 真正的 burst IFG 生成 |
| mergebursts | merge_bursts | 已有 | mosaic |
| filter | filter | 已有 | Goldstein filter |
| unwrap | unwrap | 已有/需增强 | unwrap fallback 显式告警 |
| unwrap2stage | unwrap2stage | 新增 | 两阶段 unwrap |
| geocode | geocode | 已有/需增强 | 生成中间地理编码产品 |
| denseoffsets | denseoffsets | 新增 | 全帧 dense offset |
| filteroffsets | filteroffsets | 新增 | offset field filter |
| geocodeoffsets | geocodeoffsets | 新增 | offset geocoding |
| publish | publish / endup | 已有/需增强 | 最终产品写出与收尾 |
```

- [ ] **Step 2: 明确文档中的“主链路”和“扩展链路”边界**

```markdown
- 主链路：check → preprocess → common_bursts → topo → subset_overlaps → coarse_resamp → overlap_ifg → prep_esd → esd → range_coreg → fine_resamp → burst_ifg → merge_bursts → filter → unwrap → geocode → publish
- 扩展链路：fineoffsets → ion → unwrap2stage → denseoffsets → filteroffsets → geocodeoffsets
```

- [ ] **Step 3: 写出与 ISCE2 的行为差异约束**

```markdown
- 任何 stage 失败必须返回可诊断状态，不允许静默吞掉。
- 合成零数组仅允许用于单元测试或不可用输入的明确测试分支，不允许默认生产路径。
- `geocode` 与 `publish` 必须分工：前者产出中间编码数组，后者负责最终打包/输出。
```

- [ ] **Step 4: 保存计划文档**

Run: `python - <<'PY' ... PY` 或使用现有写文件流程保存到 `docs/superpowers/plans/2026-05-10-tops_insar-doc-alignment-and-isce2-parity.md`
Expected: 文档存在且内容完整，没有 `TBD` / `TODO` / 占位文字。

---

### Task 2: 对齐 stage 序列与 CLI 入口

**Files:**
- Modify: `scripts/tops_insar.py`
- Modify: `tests/test_tops_insar_cli.py`

- [ ] **Step 1: 更新 stage 序列定义**

```python
STAGE_SEQUENCE: list[str] = [
    "check",
    "preprocess",
    "common_bursts",
    "topo",
    "subset_overlaps",
    "coarse_resamp",
    "overlap_ifg",
    "prep_esd",
    "esd",
    "range_coreg",
    "fineoffsets",
    "fine_resamp",
    "ion",
    "burst_ifg",
    "merge_bursts",
    "filter",
    "unwrap",
    "unwrap2stage",
    "geocode",
    "denseoffsets",
    "filteroffsets",
    "geocodeoffsets",
    "publish",
]
```

- [ ] **Step 2: 让 `--help` 和 stage 选择覆盖新序列**

```python
assert len(_build_stage_sequence("check", "publish")) == 23
assert _build_stage_sequence("esd", "unwrap") == [
    "esd", "range_coreg", "fineoffsets", "fine_resamp",
    "ion", "burst_ifg", "merge_bursts", "filter", "unwrap",
]
```

- [ ] **Step 3: 更新 CLI 参数说明与日志输出**

```python
parser.add_argument(
    "--do-ionospheric-correction",
    action="store_true",
    help="Enable split-band ionospheric phase correction; only applied when the ion stage is selected.",
)
```

- [ ] **Step 4: 运行 CLI 测试确认 stage 列表生效**

Run: `pytest tests/test_tops_insar_cli.py -v`
Expected: 所有 help / stage sequence 测试通过。

---

### Task 3: 补齐主链路的数据流与失败语义

**Files:**
- Modify: `scripts/tops_insar.py`
- Modify: `scripts/tops_registration.py`
- Modify: `scripts/tops_esd.py`
- Modify: `scripts/tops_publish.py`
- Modify: `tests/test_tops_insar_pipeline.py`

- [ ] **Step 1: 写出主链路失败语义测试**

```python
def test_coarse_resamp_failure_blocks_downstream_stages(monkeypatch):
    # mock run_coarse_registration -> False or raises
    # assert fine_resamp is not entered
    # assert pipeline state records the failure reason
```

- [ ] **Step 2: 让 coarse→fine 依赖显式化**

```python
if not state.get("coarse_resamp_ok", False):
    raise RuntimeError("fine_resamp requires successful coarse_resamp")
```

- [ ] **Step 3: 让 ESD correction 真正进入后续阶段**

```python
correction = compute_esd_timing_correction(...)
state["esd_correction"] = correction
state["timing_correction"] = apply_esd_correction(state, correction)
```

- [ ] **Step 4: 把 unwrap fallback 从静默降级改成显式降级**

```python
try:
    result = unwrap_ifg(...)
except Exception as exc:
    log.warning("unwrap fallback engaged: %s", exc)
    result = unwrap_phase_2d(...)
    state["unwrap_mode"] = "fallback_phase_2d"
```

- [ ] **Step 5: 将 geocode 拆成中间产物与最终产物**

```python
# geocode: write intermediate geocoded arrays
# publish: package final TIFF/HDF5/JSON products
```

- [ ] **Step 6: 运行主链路管线测试**

Run: `pytest tests/test_tops_insar_pipeline.py -v`
Expected: 主链路测试通过，失败语义测试可验证。

---

### Task 4: 补齐 ISCE2 扩展 stage

**Files:**
- Modify: `scripts/tops_insar.py`
- Modify: `scripts/tops_registration.py`
- Modify: `scripts/tops_publish.py`
- Create or modify: `scripts/tops_ionosphere.py`
- Modify: `tests/test_tops_insar_pipeline.py`

- [ ] **Step 1: 将 `fineoffsets` 插入 `fine_resamp` 前**

```python
state["fine_offsets"] = estimate_fine_offsets(
    state["overlaps"],
    state["coarse_resamp"],
)
```

- [ ] **Step 2: 添加 `ion` stage 的开关逻辑**

```python
if args.do_ionospheric_correction:
    state = run_ionosphere_correction(state)
else:
    log.info("ion stage skipped because --do-ionospheric-correction is not set")
```

- [ ] **Step 3: 让 `unwrap2stage` 复用现有 unwrap 结果**

```python
state["unwrap2stage"] = run_two_stage_unwrap(state["merged_ifg"], state["merged_coh"])
```

- [ ] **Step 4: 增加 dense offset 输出路径**

```python
state["dense_offsets"] = estimate_dense_offsets(state["merged_ifg"], state["merged_coh"])
state["filtered_offsets"] = filter_offsets(state["dense_offsets"])
state["geocoded_offsets"] = geocode_offsets(state["filtered_offsets"], dem=state["dem"])
```

- [ ] **Step 5: 为扩展 stage 添加覆盖测试**

```python
def test_optional_stages_are_skipped_when_disabled():
    ...
```

- [ ] **Step 6: 运行扩展 stage 测试**

Run: `pytest tests/test_tops_insar_pipeline.py -v -k "ion or unwrap2stage or denseoffsets"`
Expected: 新增扩展阶段测试通过。

---

### Task 5: 对齐 overlap / ifg / merge 的真实数据路径

**Files:**
- Modify: `scripts/tops_overlap.py`
- Modify: `scripts/tops_ifg.py`（如存在并需要接入）
- Modify: `scripts/tops_merge.py`
- Modify: `tests/test_tops_insar_pipeline.py`

- [ ] **Step 1: 把 overlap 读取改成真实 TIFF 窗口读取**

```python
def read_overlap_window(tiff_path, slice_):
    # Use GDAL/rasterio to read the exact pixel window
    # Return complex64 ndarray with the expected shape
```

- [ ] **Step 2: 确保 IFG 生成使用真实读入数据而不是零填充**

```python
ref = read_overlap_window(ref_tiff, overlap.top)
sec = read_overlap_window(sec_tiff, overlap.bottom)
ifg, coh = generate_ifg(ref, sec)
```

- [ ] **Step 3: merge_bursts 继续沿用 valid-window 逻辑，但补齐 seam 诊断**

```python
assert result.seam_phase_diff_median is not None
assert result.seam_coherence_drop is not None
```

- [ ] **Step 4: 增加真实数据路径失败测试**

```python
def test_overlap_ifg_fails_when_input_tiffs_missing():
    ...
```

- [ ] **Step 5: 运行 overlap / IFG / merge 测试**

Run: `pytest tests/test_tops_insar_pipeline.py -v -k "overlap or ifg or merge"`
Expected: 数据路径测试通过，缺失输入时错误清晰可诊断。

---

### Task 6: 文档自检与一致性修正

**Files:**
- Modify: `docs/superpowers/plans/2026-05-10-tops_insar-doc-alignment-and-isce2-parity.md`

- [ ] **Step 1: 扫描计划中的占位词**

```bash
python - <<'PY'
from pathlib import Path
text = Path('docs/superpowers/plans/2026-05-10-tops_insar-doc-alignment-and-isce2-parity.md').read_text()
for needle in ['TBD', 'TODO', 'implement later', 'fill in details']:
    assert needle not in text, needle
print('OK')
PY
```

- [ ] **Step 2: 检查任务编号和文件路径一致性**

```bash
python - <<'PY'
from pathlib import Path
p = Path('docs/superpowers/plans/2026-05-10-tops_insar-doc-alignment-and-isce2-parity.md')
text = p.read_text()
assert 'scripts/tops_insar.py' in text
assert 'tests/test_tops_insar_pipeline.py' in text
print('OK')
PY
```

- [ ] **Step 3: 确认计划能独立指导实现**

```markdown
- 每个任务都有明确文件。
- 每个代码变更都有测试命令。
- 没有未定义函数名或占位步骤。
```

---

## 4. 通过标准

- `scripts/tops_insar.py` 的 stage 序列与 ISCE2 主链路和扩展链路一致。
- `check → publish` 的 stage 选择能稳定返回完整顺序。
- 主链路 stage 之间的状态传递是显式的，不再依赖隐式 fallback。
- `overlap_ifg`、`esd`、`fine_resamp`、`unwrap`、`geocode` 的错误语义清晰。
- 新增的 `fineoffsets / ion / unwrap2stage / denseoffsets / filteroffsets / geocodeoffsets` 有明确入口和测试覆盖。
- 计划文档无占位词，无冲突表述，无未绑定到任务的需求。

---

## 5. 验证命令清单

```bash
pytest tests/test_tops_insar_cli.py -v
pytest tests/test_tops_insar_pipeline.py -v
python scripts/tops_insar.py --help
```

若需要真实 Sentinel-1 数据验证，再补充：

```bash
pytest tests/test_tops_insar_pipeline.py -v -m integration
```
