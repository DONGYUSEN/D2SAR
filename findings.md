# GPU INSAR 全流程测试发现

## 已知信息
- 本机直接运行 `python3 scripts/strip_insar.py --help` 失败，原因是本机 Python 缺少 `h5py`。
- 前一次成功重跑使用容器 `d2sar:latest`，并用 `--gpus all` 启动 GPU 流程。
- 输入 manifest 存在：
  - `/home/ysdong/Software/D2SAR/results/20231110/manifest.json`
  - `/home/ysdong/Software/D2SAR/results/20231121/manifest.json`
- 容器内 GPU 可见：NVIDIA GeForce GTX 1060 6GB，驱动 550.163.01，CUDA 12.4。
- `strip_insar.py` 支持阶段：`check`、`prep`、`crop`、`p0`、`p1`、`p2`、`p3`、`p4`、`p5`、`p6`。
- `check` 阶段成功，precheck 总体为 `ok`，各项检查均为 `ok`。
- `prep` 阶段成功，`requires_normalization=false`，处理动作为 `pass-through`。
- `prep` 单阶段记录 `backend_used=cpu`，该阶段只处理元数据和 JSON。
- `crop` 阶段成功，生成 `master.crop.json`、`slave.crop.json` 和四张检查 PNG：normal/crop 的 master/slave fullres PNG。
- 当前全幅 crop window 为 row0=0、col0=0、rows=14580、cols=12544。
- `p0` 阶段成功，`backend_used=gpu`。
- `geo2rdr_master.tif`、`geo2rdr_slave.tif` 已生成；`geo2rdr_master/topo.vrt` 和 `geo2rdr_slave/topo.vrt` 使用 VRT 组织 lon/lat/hgt 三个单波段 TIFF。
- `p1` 阶段成功，`p1_geo2rdr_offsets/model.json` 仍为 `slave_minus_master_pixel_index`。
- `p1` 配准统计正常：
  - coarse range mean `+41.72`
  - residual range mean `-1.35`
  - final range mean `+40.37`
  - coarse azimuth mean `+6.97`
  - residual azimuth mean `-2.78`
  - final azimuth mean `+4.19`
- `fit_quality` 为 residual-based，`range_rms=1.35`、`azimuth_rms=2.78`、`retry_recommended=false`。
- P1 生成了 `slave_coarse_coreg_fullres.png` 和 `slave_fine_coreg_fullres.png`。
- `p2` 阶段成功，生成：
  - `work/p2_crossmul/interferogram.npy`
  - `work/p2_crossmul/filtered_interferogram.npy`
  - `work/p2_crossmul/coherence.npy`
  - `wrapped_phase_radar.png`
- `p2` 的 `stage.json` 记录 `backend_used=gpu`，但日志显示 CUDA Crossmul API 仍未接通，因此 crossmul 子步骤当前回退到 pure Python 实现。
- `p3` 阶段成功，生成 `work/p3_unwrap/unwrapped_phase.npy`，但 `stage.json` 明确记录 `backend_used=cpu`。
- `p3` 运行期间会在结果目录下创建临时解缠目录，例如 `insar_unwrap_5irdqc1l`，权限为 `root:root` 且 `700`，会影响宿主机直接遍历和后续清理。

## 待确认
- 后续 P0/P1/P2/P5/P6 阶段的实际 GPU/CPU 后端记录。
- 完整流程到 `p6` 的最终产品输出。

- `p4` 阶段成功，生成 `work/p4_geocode/los_displacement.npy`，`backend_used=cpu`。
- `p5` 阶段成功，生成 `/home/ysdong/Software/D2SAR/results/20231110_20231121_gpu_fulltest/interferogram_fullres.h5`，`work/p5_hdf/stage.json` 记录 `backend_used=cpu`。
- `p6` 阶段成功，生成：
  - `/home/ysdong/Software/D2SAR/results/20231110_20231121_gpu_fulltest/interferogram_utm_geocoded.tif`
  - `/home/ysdong/Software/D2SAR/results/20231110_20231121_gpu_fulltest/coherence_utm_geocoded.tif`
  - `/home/ysdong/Software/D2SAR/results/20231110_20231121_gpu_fulltest/unwrapped_phase_utm_geocoded.tif`
  - `/home/ysdong/Software/D2SAR/results/20231110_20231121_gpu_fulltest/los_displacement_utm_geocoded.tif`
  - `/home/ysdong/Software/D2SAR/results/20231110_20231121_gpu_fulltest/interferogram_utm_geocoded.png`
  - `/home/ysdong/Software/D2SAR/results/20231110_20231121_gpu_fulltest/filtered_interferogram_utm_geocoded.png`
- `p6` 的 `stage.json` 记录 `backend_used=cpu`，说明发布阶段当前仍是 CPU。
- 本次真实数据全流程中，明确记录为 GPU 的核心阶段是 `p0`、`p1`、`p2`；其中 `p2` 虽标记 GPU，但 crossmul 子步骤日志显示仍未接通 ISCE3 CUDA Crossmul，实际存在 CPU/pure Python 回退。
- `p3`、`p4`、`p5`、`p6` 当前均为 CPU 路径。
- `ISCE3 CUDA Crossmul unavailable` 的直接原因在 `/home/ysdong/Software/D2SAR/scripts/strip_insar.py`：`_run_crossmul()` 会先尝试调用 `_crossmul_isce3_gpu()`，但该函数当前直接 `raise NotImplementedError("ISCE3 CUDA Crossmul API not yet verified")`，随后异常被捕获并打印 “ISCE3 CUDA Crossmul unavailable ... using pure Python crossmul”。
- 当前 `wrapped_phase_radar.png` 确认没有去平地效应：`p2` 中先由 `_run_crossmul()` 生成 `interferogram`，随后直接调用 `_write_radar_wrapped_phase_png(interferogram, ...)`；而现有 `_crossmul_numpy()` 仅执行 `master * conj(slave)`，没有任何 flatten/flat-earth removal。
- ISCE3 参考流程 `isce3/python/packages/nisar/workflows/crossmul.py` 在 crossmul 前会根据 `flatten` 构造 `flatten_raster`，并在 `crossmul.crossmul(..., flatten_raster)` 中执行 flatten；D2SAR 当前 `p2` 没有对应实现。

## 2026-04-21 P2 CUDA Crossmul 降级与 ENVI 支持
- 真实崩溃根因已经收敛到 `isce3.cuda.signal.Crossmul` native segfault；主流程已通过子进程隔离避免 silent exit，并能记录 fallback reason。
- 公开 GitHub issue 搜索没有发现与本次 `p2` runtime segfault 完全相同的公开 issue；本地 ISCE3 历史中可见多个相关修复 PR/提交，包括 CUDA Crossmul size/initialization/range offset allocation 修复、upsample 修复、CPU/CUDA flatten 分离、flatten logic 修复。
- 当前策略：`p2` GPU Crossmul 明确降级为实验特性，默认关闭。只有设置 `D2SAR_ENABLE_EXPERIMENTAL_GPU_CROSSMUL=1` 才会尝试 CUDA Crossmul。
- 默认关闭时，`p2` 会直接使用 CPU/pure Python crossmul，并在 `work/p2_crossmul/gpu_fallback_reason.txt` 与 `stage.json` 中记录原因；full pipeline 的 `stage_backends["crossmul"]` 会标记为 `cpu`，不再误报为 `gpu`。
- Dockerfile 中 GDAL 的 ENVI 支持需要启用 `GDAL_ENABLE_DRIVER_RAW=ON`；构建日志显示 `frmts/raw/envidataset.cpp` 被编译，最终新镜像 `d2sar:envi-test` 中 `gdal.GetDriverByName("ENVI")` 为 `True`，并能创建/打开 ENVI raster。
