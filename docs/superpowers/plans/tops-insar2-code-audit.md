# tops_insar 文档对照审查记录

## 审查目标
对照开发文档，检查 `tops_insar.py` 功能实现与代码衔接情况，并记录差异。

## 已检查模块
- `tops_insar.py`
- `tops_registration.py`
- `tops_overlap.py`
- `tops_publish.py`
- `tops_utils.py`

## 当前结论

### 已对齐
- `STAGE_SEQUENCE` 完整覆盖文档中的 stage。
- `_dispatch_stage` 已连接所有 stage。
- `fine_resamp -> burst_ifg -> merge_bursts -> filter -> unwrap -> geocode -> publish` 衔接正常。
- `fine_resample_with_timing` 已接入 `_stage_fine_resamp`。
- `filter_ifg` 已接入 `_stage_filter`。
- `geocode_ifg / unwrap_ifg / write_product` 已接入 `_stage_geocode` / `_stage_publish`。

### 需要注意的差异（非阻塞）
1. `tops_insar.py` 中部分 stage 注释仍残留 “spike” 字样，但函数已实现。
2. `geocode` 阶段目前输出 `.npy` 中间文件，最终 TIFF/HDF5 由 `publish` 阶段统一写出。
3. `preprocess` / `common_bursts` 的职责拆分与文档描述略有差异，但不影响执行链路。

## 代码衔接检查
- `state["merged_ifg"]`、`state["merged_coh"]`、`state["unwrapped"]`、`state["geocoded_ifg"]`、`state["geocoded_coh"]`、`state["published_files"]` 传递一致。
- `merged/` 目录作为中间产品缓存目录使用，命名统一。

## 后续建议
- 清理 `tops_insar.py` 中残留的 spike 注释。
- 如需进一步贴合文档，可把 `geocode` 阶段输出说明改为“生成中间 geocoded arrays”，最终产品留给 `publish`。
