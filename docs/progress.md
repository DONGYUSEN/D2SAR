# GPU INSAR 全流程测试进度

## 2026-04-20
- 启动从第一步开始的 GPU 全流程逐步测试。
- 检查本机脚本帮助时发现本机 Python 环境缺少 `h5py`，决定使用 `d2sar:latest` 容器执行。
- 已发现输入 manifest 文件存在。
- 当前没有正在运行的 Docker 容器。
- 容器内 `nvidia-smi` 正常，GPU 可见。
- 容器内 `strip_insar.py --help` 成功，确认阶段支持到 `p6`。
- 已运行 `--step check`，输出目录 `/home/ysdong/Software/D2SAR/results/20231110_20231121_gpu_fulltest/work/check` 下有 `stage.json`、`precheck.json`、`SUCCESS`。
- `check` 的 `backend_used=cpu`，这是元数据检查阶段的预期结果。
- 已运行 `--step prep`，输出目录 `/home/ysdong/Software/D2SAR/results/20231110_20231121_gpu_fulltest/work/prep` 下有 `normalized_slave_manifest.json`、`normalized_doppler.json`、`normalized_radargrid.json`、`normalized_acquisition.json`、`preprocess_plan.json`、`stage.json`、`SUCCESS`。
- 已运行 `--step crop`，输出目录 `/home/ysdong/Software/D2SAR/results/20231110_20231121_gpu_fulltest/work/crop` 下有 crop manifest、radargrid、acquisition、doppler、orbit、scene、`crop.json`、`stage.json`、`SUCCESS`。
- 顶层生成 `master_normal_fullres.png`、`slave_normal_fullres.png`、`master_crop_fullres.png`、`slave_crop_fullres.png`。
- 已运行 `--step p0`，完成 master/slave topo 计算，生成 `geo2rdr_master.tif`、`geo2rdr_slave.tif`、`geo2rdr_master/topo.vrt`、`geo2rdr_slave/topo.vrt`。
- `p0` 的 `stage.json` 明确记录 `backend_used=gpu`。
- 已运行 `--step p1`，完成 coarse/fine 配准，生成 `p1_geo2rdr_offsets/range.off`、`azimuth.off`、`model.json`，以及 `work/p1_dense_match` 下的 coarse/fine coreg、dense match、residual、final offsets、diagnostics、`registration_model.json`、`stage.json`、`SUCCESS`。
- 已验证 `registration_model.json` 中 coarse/residual/final 统计与之前修复后的正确结果一致。
- 已运行 `--step p2`，生成 `interferogram.npy`、`filtered_interferogram.npy`、`coherence.npy` 和 `wrapped_phase_radar.png`。
- P2 日志提示 `ISCE3 CUDA Crossmul unavailable`，当前 crossmul 子步骤未真正走 CUDA。
- 已运行 `--step p3`，生成 `work/p3_unwrap/unwrapped_phase.npy`、`stage.json`、`SUCCESS`。
- 发现 `p3` 会创建 root 权限临时目录 `insar_unwrap_*`。
## 2026-04-20
- 接手继续 GPU 全流程测试，会话恢复后确认 `p5` 仍在容器 `unruffled_edison` 中运行。
- 通过现有 PTY 会话日志确认 `p5` 已完成 topo 与 layover/shadow mask 到 block 57/57。
- 当前 `interferogram_fullres.h5` 已增长到约 4.23 GB，但 `work/p5*` 目录下尚未出现 `stage.json`/`SUCCESS`，说明还在收尾写出。
## 2026-04-20
- `p5` 已完成，产物为 `/home/ysdong/Software/D2SAR/results/20231110_20231121_gpu_fulltest/interferogram_fullres.h5`。
- `work/p5_hdf/stage.json` 显示本阶段因复用已有 `p2-p4` 产品而走 CPU 发布路径，`backend_used=cpu`。
- 已启动 `--step p6`，继续验证最终发布阶段。

## 2026-04-20
- `p6` 已完成，生成 geocoded TIFF 和最终 interferogram/filtered interferogram PNG。
- `work/p6_publish/stage.json` 记录 `backend_used=cpu`。
- 本轮从 `check` 到 `p6` 的真实数据 GPU 全流程逐步测试已完整跑通。

## 2026-04-20
- 为 `p2` 补了两类能力：基于 `p1_geo2rdr_offsets/range.off` 的 flatten-aware crossmul，以及真正调用 `isce3.cuda.signal.Crossmul()` 的 GPU crossmul 路径。
- 先补了失败测试，再修改实现；当前 `/work/tests/test_strip_insar_stages.py` 在容器中已 23/23 通过。
- 下一步在真实结果目录上重跑 `p2`，检查 CUDA crossmul 是否接通，以及 `wrapped_phase_radar.png` 是否改为去平地后的相位。

## 2026-04-21
- 针对 `p2` 在写出 `range_flatten.off.tif` 后无日志退出的问题，定位到真实原因是 `isce3.cuda.signal.Crossmul` 的 native 崩溃，而不是 Python 正常退出。
- 关键根因之一：ISCE3 CUDA flatten 内核按 `double*` 读取 range offsets；已将 `range_flatten.off.tif` 改为 `Float64` 输出。
- 为避免 GPU 子进程 segfault 直接杀掉主流程，已把 CUDA crossmul 放入独立子进程执行，主进程现在会捕获退出码和 stderr，并显式打印回退日志后切回 CPU。
- 同时补充了 `work/p2_crossmul/gpu_fallback_reason.txt`，用于保存本次 GPU 回退原因。

## 2026-04-21
- 已将 `p2` GPU Crossmul 标记为实验特性并默认关闭；默认情况下即使 `--gpu-mode gpu`，Crossmul 子步骤也会走 CPU fallback，并明确记录原因。
- 显式启用方式为设置环境变量 `D2SAR_ENABLE_EXPERIMENTAL_GPU_CROSSMUL=1`。
- 已修正 full pipeline 中 `stage_backends["crossmul"]` 的记录，使其反映真实执行后端；当默认降级时记录为 `cpu`。
- 已修改 Dockerfile，通过 `GDAL_ENABLE_DRIVER_RAW=ON` 增加 ENVI 支持。
- 已构建验证镜像 `d2sar:envi-test`，确认 `gdal.GetDriverByName("ENVI")` 为 `True`，并可创建/打开 ENVI raster。
- 测试验证：`d2sar:latest` 与 `d2sar:envi-test` 中 `tests/test_strip_insar_stages.py` 均为 `28 passed, 1 warning`。
