# GPU INSAR 全流程逐步测试计划

## 目标
从第一步开始，在 GPU 条件下对真实数据执行 D2SAR INSAR 完整处理流程，逐阶段验证输入、输出、日志和关键诊断，确认配准修正后全流程稳定。

## 输入数据
- Master manifest: `/home/ysdong/Software/D2SAR/results/20231110/manifest.json`
- Slave manifest: `/home/ysdong/Software/D2SAR/results/20231121/manifest.json`
- 测试输出目录: `/home/ysdong/Software/D2SAR/results/20231110_20231121_gpu_fulltest`
- 容器镜像: `d2sar:latest`
- GPU 模式: `--gpu-mode gpu`

## 阶段
- [complete] 阶段 0: 环境和脚本能力检查
- [complete] 阶段 1: check
- [complete] 阶段 2: prep
- [complete] 阶段 3: crop
- [complete] 阶段 4: p0 geo2rdr/coarse offset
- [complete] 阶段 5: p1 coarse/fine registration
- [complete] 阶段 6: p2 crossmul/filter/phase png
- [complete] 阶段 7: 如脚本支持后续阶段，继续执行到完整流程末端
- [complete] 阶段 8: 汇总诊断和输出验证

## 验证标准
- 每个阶段必须有对应 `SUCCESS` 或等价成功记录。
- GPU 请求必须被记录为 `backend_used=gpu` 或相关 GPU 处理路径。
- p1 必须检查 coarse/residual/final offset 统计，确认 residual 是小修正。
- p2 必须生成 `wrapped_phase_radar.png`。
- 后续阶段如果存在，必须检查最终产品是否生成。

## 当前状态
`p0-p6` 已全部完成，结果目录包含 HDF5、UTM geocoded TIFF 与 PNG。
