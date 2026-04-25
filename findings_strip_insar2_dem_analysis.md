# StripInSAR2 DEM 处理分析报告

## 分析时间
2025年4月25日

## 分析目标
分析 `strip_insar2.py` 中 DEM（数字高程模型）的处理流程，评估是否需要裁剪以及裁剪时的外扩策略。

---

## 1. 当前 DEM 处理流程

### 1.1 DEM 路径解析 (`_resolve_dem_path`)
```python
def _resolve_dem_path(...):
    if dem_path is not None:
        return str(Path(dem_path))  # 直接使用用户提供的 DEM
    
    manifest_dem = manifest.get("dem", {}).get("path")
    if manifest_dem is not None:
        return resolved_manifest_dem  # 使用 manifest 中指定的 DEM
    
    raise FileNotFoundError(...)  # 没有自动下载/裁剪逻辑
```

**结论**：只解析路径，不做裁剪。

### 1.2 DEM 使用 (`_run_rdr2geo_topo`)
```python
def _run_rdr2geo_topo(...):
    dem_raster = isce3.io.Raster(str(dem_path))  # 直接打开完整 DEM
    ...
    topo.topo(dem_raster, ...)  # 使用完整 DEM
```

**结论**：`isce3.geometry.Rdr2Geo` 直接处理整个 DEM，没有裁剪窗口参数。

### 1.3 数据流向
```
load_pair_context()
└── _resolve_dem_path() → 仅返回路径字符串
    └── context.resolved_dem

run_geo2rdr_stage()
└── _run_rdr2geo_topo(context.resolved_dem) → 直接使用完整 DEM
```

**没有裁剪环节！**

---

## 2. 发现的问题

| 问题 | 影响 | 严重程度 |
|------|------|----------|
| 使用全场景 DEM | 处理大范围场景时内存开销大 | 中 |
| 没有基于 bbox 的裁剪 | 与迁移计划要求的 "prepared/dem/" 局部 DEM 不符 | 高 |
| 依赖 `isce3` 内部裁剪 | 不会生成裁剪后的物理文件 | 中 |

---

## 3. 与迁移计划的差距

### 迁移计划要求（Phase 1 DEM Policy）
> - if `dem_path` is not provided: migrate the DEM resolution logic from `strip_insar.py`
> - resolve DEM using scene/**bbox** context
> - create a DEM local to `prepared/dem/`
> - **prefer bbox-scoped DEM if possible**

### 当前缺失
- ❌ 没有基于 bbox 的 DEM 裁剪逻辑
- ❌ 没有 `prepared/dem/` 目录结构
- ❌ DEM 路径解析过于简单
- ❌ 没有考虑裁剪后的外扩（buffer）策略

---

## 4. 推荐实现方案

### 4.1 DEM 裁剪外扩策略

**为什么需要外扩？**
- DEM 用于几何校正（Rdr2Geo）和配准
- SAR 图像边缘的像元需要周围 DEM 数据进行插值
- 如果没有外扩，裁剪后 DEM 的边缘会导致处理失败或精度下降

**推荐参数**
```python
DEM_CLIP_MARGIN_DEG = 0.05  # 约 5.5km，覆盖大部分 SAR 场景的边缘效应
```

### 4.2 裁剪实现建议

```python
def _prepare_runtime_dem(
    *,
    source_dem_path: str | Path,
    prepared_dir: Path,
    bbox: tuple[float, float, float, float] | None,
    margin_deg: float = 0.05,  # 外扩范围（度）
) -> str:
    """
    裁剪 DEM 并外扩一定范围，确保处理区域边缘有足够的 DEM 数据。
    
    Args:
        source_dem_path: 原始 DEM 路径
        prepared_dir: prepared/dem/ 目录路径
        bbox: (min_lon, min_lat, max_lon, max_lat) 裁剪范围
        margin_deg: 外扩范围（度），默认 0.05° (~5.5km)
    
    Returns:
        裁剪后的 DEM 路径
    """
    if bbox is None:
        # 没有 bbox，复用原 DEM
        return str(source_dem_path)
    
    min_lon, min_lat, max_lon, max_lat = bbox
    
    # 外扩 bbox
    expanded_bbox = (
        min_lon - margin_deg,
        min_lat - margin_deg,
        max_lon + margin_deg,
        max_lat + margin_deg,
    )
    
    prepared_dem_path = prepared_dir / "dem.tif"
    
    # 使用 GDAL Warp 裁剪并外扩
    from osgeo import gdal
    gdal.Warp(
        str(prepared_dem_path),
        str(source_dem_path),
        outputBounds=expanded_bbox,
        resampleAlg="bilinear",
        format="GTiff",
        options=["COMPRESS=LZW", "TILED=YES"],
    )
    
    print(f"[DEM] Clipped with {margin_deg}° margin: {expanded_bbox}")
    return str(prepared_dem_path)
```

### 4.3 外扩范围参考

| 应用场景 | 推荐外扩范围 | 说明 |
|----------|--------------|------|
| 常规 StripInSAR | 0.05° (~5.5km) | 覆盖边缘插值需求 |
| 大范围场景 | 0.1° (~11km) | 考虑地球曲率影响 |
| 陡峭地形 | 0.1° (~11km) | 地形阴影区域需要更多 DEM |

---

## 5. 关键注意事项

### 5.1 裁剪后的 DEM 完整性检查
```python
# 裁剪后检查 DEM 是否包含有效数据
import numpy as np
from osgeo import gdal

ds = gdal.Open(str(prepared_dem_path))
band = ds.GetRasterBand(1)
data = band.ReadAsArray()
valid_ratio = np.sum(data != band.GetNoDataValue()) / data.size

if valid_ratio < 0.9:
    raise RuntimeError(f"DEM 裁剪后有效数据比例过低: {valid_ratio:.2%}")
```

### 5.2 坐标系统一
- 确保 bbox 和 DEM 使用相同的坐标系（通常为 WGS84/EPSG:4326）
- 如果 DEM 是其他投影，需要先进行坐标转换

### 5.3 与 `strip_insar.py` 的对比
参考 `strip_insar.py` 中的 `_resolve_dem_path` 实现：
- 它支持自动下载 DEM
- 支持基于场景角点裁剪 DEM
- 但**没有显式的外扩参数**

**改进建议**：在迁移时显式添加 `dem_margin_deg` 参数，而非依赖隐式行为。

---

## 6. 与迁移计划的集成点

### 在 `_prepare_runtime_inputs()` 中的调用位置
```python
def _prepare_runtime_inputs(...):
    # ... 其他准备步骤 ...
    
    # Step 6: 准备 DEM
    prepared_dem = _prepare_runtime_dem(
        source_dem_path=original_dem_path,
        prepared_dir=pair_dir / "prepared" / "dem",
        bbox=bbox,
        margin_deg=0.05,  # 外扩 0.05 度
    )
    
    return {
        "prepared_master_manifest": ...,
        "prepared_slave_manifest": ...,
        "prepared_dem": prepared_dem,
        ...
    }
```

---

## 7. 结论与建议

### 7.1 当前状态
- `strip_insar2.py` **完全没有 DEM 裁剪功能**
- 直接使用全场景 DEM，内存开销大
- 不符合迁移计划的要求

### 7.2 需要实现的功能
1. ✅ 基于 bbox 的 DEM 裁剪
2. ✅ **裁剪时外扩一定范围**（推荐 0.05°）
3. ✅ 将裁剪后的 DEM 保存到 `prepared/dem/`
4. ✅ DEM 完整性检查

### 7.3 优先级
| 功能 | 优先级 | 原因 |
|------|--------|------|
| DEM 裁剪 + 外扩 | **P0（高）** | 影响处理正确性，边缘像元可能失败 |
| DEM 完整性检查 | P1（中） | 防止无效 DEM 进入下游处理 |
| 自动下载 DEM | P2（低） | 可以先要求用户显式提供 |

### 7.4 下一步行动
1. 在 `strip_insar2.py` 中实现 `_prepare_runtime_dem()` 函数
2. 添加 `dem_margin_deg` 参数到 CLI 和 `process_strip_insar2()`
3. 更新迁移计划文档，记录外扩策略决策

---

## 附录：相关代码位置

### strip_insar2.py 中的相关函数
- 行 1052-1074: `_resolve_dem_path()` - DEM 路径解析
- 行 2218-2329: `_run_rdr2geo_topo()` - 使用 DEM 进行几何定位
- 行 4804-4926: `process_strip_insar2()` - 主流程

### 参考实现
- `scripts/strip_insar.py`: `_resolve_dem_path()` - 包含自动下载和裁剪逻辑
- `scripts/dem_manager.py`: DEM 下载和裁剪工具

