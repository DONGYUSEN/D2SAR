# D2SAR

## 当前主流程

当前仓库的主流程分为两段：

1. `importer`：读取第三方产品，输出 `manifest.json + metadata/*.json`
2. `process`：基于 `manifest.json` 通过统一入口 `scripts/strip_rtc.py` 执行幅度、DEM、UTM、GeoTIFF、PNG 等处理

---

## manifest 路径语义（当前规则）

### 1. metadata 路径

`manifest.json` 中的 `metadata/*` 字段统一写为**相对 `manifest.json` 的相对路径**。

例如：

```json
"metadata": {
  "acquisition": "metadata/acquisition.json",
  "orbit": "metadata/orbit.json",
  "radargrid": "metadata/radargrid.json",
  "doppler": "metadata/doppler.json",
  "scene": "metadata/scene.json"
}
```

### 2. 外部源数据路径

外部源数据（如原始 SLC、DEM、辅助 XML）不再强制写为相对路径，而是采用以下两种方式：

- **目录输入**：写为绝对路径
- **ZIP 输入**：写为结构化 ZIP 路径对象

目录输入示例：

```json
"slc": {
  "path": "/tmp/tianyi_test/.../measurement/product.tiff",
  "format": "TIFF",
  "complex": true
}
```

ZIP 输入示例：

```json
"slc": {
  "path": {
    "path": "/tmp/lutan_test.zip",
    "storage": "zip",
    "member": "LT1...SLC...tiff"
  },
  "format": "TIFF",
  "complex": true
}
```

辅助文件（如 `metaXML`、`rpc`、`manifestSafe`）也遵循同样规则。

### 3. DEM 路径

`manifest["dem"]` 保留下载后的绝对路径：

```json
"dem": {
  "path": "/path/to/out/dem/dem_clip_wgs84.tif",
  "source": "SRTMGL1",
  "directory": "/path/to/out/dem",
  "autoDownloaded": true
}
```

**自动下载默认行为**：当 `--download-dem` 且未指定 `--dem-dir` 时，DEM 自动下载到 **输出目录的 `dem/` 子目录**（即 `{output_dir}/dem`），不再使用全局缓存 `/tmp/d2sar_dem_cache`。这样每次导入的 DEM 都跟随各自的输出目录，便于迁移和清理。

### 4. consumer 解析规则

后续处理脚本不会直接假设 `manifest` 中的路径一定是绝对路径或 `/vsizip/...` 字符串，而是统一通过公共解析函数恢复真实可打开路径。

---

## importer 命令示例

### Tianyi

#### 目录输入，输出到当前目录

```bash
python3 scripts/tianyi_importer.py /path/to/extracted_product
```

#### ZIP / 归档输入，显式指定输出目录

```bash
python3 scripts/tianyi_importer.py /path/to/product.zip -o /path/to/out
```

#### 自动下载 DEM

`--download-dem` 时若未指定 `--dem-dir`，DEM 自动下载到输出目录的 `dem/` 子目录：

```bash
python3 scripts/tianyi_importer.py /path/to/product.zip -o /path/to/out \
  --download-dem
```

#### 使用已有 DEM 目录

```bash
python3 scripts/tianyi_importer.py /path/to/product.zip -o /path/to/out \
  --dem-dir /path/to/local_dem_dir
```

### Lutan

#### 目录输入，输出到当前目录

```bash
python3 scripts/lutan_importer.py /path/to/extracted_product
```

#### ZIP 输入，显式指定输出目录

```bash
python3 scripts/lutan_importer.py /path/to/product.zip -o /path/to/out
```

#### 自动下载 DEM

`--download-dem` 时若未指定 `--dem-dir`，DEM 自动下载到输出目录的 `dem/` 子目录：

```bash
python3 scripts/lutan_importer.py /path/to/product.zip -o /path/to/out \
  --download-dem
```

#### 使用已有 DEM 目录

```bash
python3 scripts/lutan_importer.py /path/to/product.zip -o /path/to/out \
  --dem-dir /path/to/local_dem_dir
```

---

## processing 命令示例

统一处理入口同时支持 Tianyi / Lutan，自动从 `manifest.json` 中识别 `sensor`。

```bash
python3 scripts/strip_rtc.py \
  /path/to/out/manifest.json \
  /path/to/out/processed
```

显式指定 DEM：

```bash
python3 scripts/strip_rtc.py \
  /path/to/out/manifest.json \
  /path/to/out/processed \
  --dem /path/to/dem_clip_wgs84.tif
```

**DEM 自动下载**：当 manifest 中无 DEM 路径且未指定 `--dem` 时，自动下载到 `manifest.json` 同目录的 `dem/` 子目录（即 `{manifest_dir}/dem`）。

指定最终 UTM GeoTIFF / PNG 地面分辨率（米）：

```bash
python3 scripts/strip_rtc.py \
  /path/to/out/manifest.json \
  /path/to/out/processed \
  --resolution 2.5
```

默认分辨率：`max(range_resolution, azimuth_resolution) × 2`（即 SAR 像元 2 倍采样），自动从 manifest 的 radargrid 元数据读取。

输出：

- `amplitude_fullres.h5`
- `amplitude_utm_geocoded.tif`
- `amplitude_utm_geocoded.png`

### GPU / CPU 后端语义

`strip_rtc.py` 当前采用**混合后端**：

- `rtc_factor`：CPU
- `amplitude_hdf`：CPU
- `topo_lonlatheight`：GPU（`isce3.cuda.geometry.Rdr2Geo`）
- `utm_transform`：CPU
- `utm_rasterize`：CPU
- `preview_png`：CPU

也就是说，当前仓库并**没有**声称提供完整的全 GPU RTC 链路；只有已经验证过的 topo 几何阶段走 GPU，其余阶段保持 CPU，以确保输出格式和现有主流程一致。

### GPU 运行模式

```bash
python3 scripts/strip_rtc.py /path/to/manifest.json /path/to/out --gpu-mode auto
```

- `--gpu-mode auto`：优先启用已验证的 GPU 阶段；不可用时自动回退 CPU
- `--gpu-mode cpu`：强制全 CPU
- `--gpu-mode gpu`：显式请求 GPU；若 GPU 阶段失败，处理脚本会回退到 CPU 并在结果中记录原因

可选参数：

- `--gpu-id <id>`：指定 CUDA 设备
- `--resolution <meters>`：输出地面分辨率（米），默认 `max(range_res, azimuth_res) × 2`

### 运行结果 JSON

`strip_rtc.py` 结束后会打印 JSON，除了输出文件路径外，还会显式报告后端执行情况，例如：

```json
{
  "backend_requested": "gpu",
  "backend_used": "gpu",
  "pipeline_mode": "hybrid",
  "stage_backends": {
    "rtc_factor": "cpu",
    "amplitude_hdf": "cpu",
    "topo_lonlatheight": "gpu",
    "utm_transform": "cpu",
    "utm_rasterize": "cpu",
    "preview_png": "cpu"
  }
}
```

如果 GPU 阶段失败并发生自动回退，则会返回：

- `pipeline_mode: "cpu-fallback"`
- `fallback_reasons`

---

## Docker 示例

### Tianyi importer

```bash
docker run --rm --gpus all \
  -v /home/ysdong/Software/D2SAR:/work \
  -v /tmp/tianyi_test:/tmp/tianyi_test \
  -v /tmp/dem_cache:/tmp/dem_cache \
  -v /tmp/tianyi_out:/out \
  d2sar:latest \
  python3 /work/scripts/tianyi_importer.py \
  /tmp/tianyi_test/BC3-SM-SLC-1SVV-20231110T043948-002131-000033-000853 \
  -o /out --download-dem --dem-dir /tmp/dem_cache
```

### Lutan importer（ZIP 输入）

```bash
docker run --rm --gpus all \
  -v /home/ysdong/Software/D2SAR:/work \
  -v /tmp/lutan_test.zip:/tmp/lutan_test.zip \
  -v /tmp/dem_cache:/tmp/dem_cache \
  -v /tmp/lutan_out:/out \
  d2sar:latest \
  python3 /work/scripts/lutan_importer.py \
  /tmp/lutan_test.zip \
  -o /out --download-dem --dem-dir /tmp/dem_cache
```

---

## 注意事项

1. `metadata/*` 是相对路径，便于随 `manifest.json` 一起搬运。
2. 外部源数据与 DEM 保持绝对路径/结构化 ZIP 路径，避免 Docker 挂载别名导致的错误相对路径。
3. `strip_rtc.py` 会优先尝试 GPU；若 GPU 不可用或当前仓库中缺少稳定 GPU strip 路径，则自动回退 CPU。
4. 处理脚本已兼容：
   - 旧绝对路径
   - 旧 `/vsizip/...`
   - 新结构化 ZIP 路径
   - 新相对 `metadata/*` 路径
