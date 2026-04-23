# ISCE3 外部数据契约草案

> 目的：定义第三方 SAR 数据在进入 `isce3 / dolphin / plant-isce3` 处理前，外围系统必须提供的最小数据集合、元数据集合和目录组织方式。

---

## 1. 适用范围

本草案适用于：

- 第三方 SAR 产品接入
- 外围 Python 编排器与 reader 的接口统一
- CLI 流程处理的输入规范
- 后续 Docker / Web 系统的数据组织基础

本草案**不是 ISCE3 官方内部格式说明**，而是我们围绕 ISCE3 制定的外部工程契约。

---

## 2. 设计原则

### 2.1 不修改上游算法库

契约的目标是适配上游，而不是要求上游为我们改变接口。

### 2.2 以“可校验”为优先

所有输入不仅要存在，还要可验证：

- 文件存在
- 坐标系明确
- 单位明确
- 时间基准明确
- 元数据完整

### 2.3 以“单景 / 双景通用”为目标

同一份契约应支持：

- 单景处理（Focus / RTC）
- 双景处理（InSAR / DInSAR）

### 2.4 文件接口优先于内存对象接口

外部系统之间优先通过文件与 manifest 交互，而不是共享内部 Python 对象。

---

## 3. 推荐目录结构

```text
dataset/
├── manifest.json
├── source/
│   ├── raw/
│   └── product/
├── slc/
│   ├── master.slc
│   └── slave.slc
├── orbit/
│   ├── master_orbit.xml
│   └── slave_orbit.xml
├── dem/
│   └── dem.tif
├── metadata/
│   ├── master_radargrid.json
│   ├── slave_radargrid.json
│   ├── master_doppler.json
│   ├── slave_doppler.json
│   └── acquisition.json
├── aux/
│   ├── tec/
│   ├── weather/
│   └── masks/
└── output/
    ├── focus/
    ├── rtc/
    ├── insar/
    └── dinsar/
```

说明：

- `source/`：保存原始输入，不参与标准处理接口；
- `slc/ orbit/ dem/ metadata/`：构成标准处理输入层；
- `aux/`：保存可选辅助数据；
- `output/`：保存处理产物；
- `manifest.json`：总索引文件，是编排器的主要入口。

---

## 4. 最小输入对象集合

对于单景或双景处理，外围系统至少应能提供下列对象语义。

### 4.1 Raster

含义：

- SLC / RSLC 栅格
- 或后续处理中生成的中间栅格

最低要求：

- 文件路径
- 数据类型（复数 / 浮点）
- 行列大小
- 行列方向定义
- 像元顺序说明

### 4.2 Orbit

含义：

- 轨道状态向量信息

最低要求：

- 轨道时间参考
- 位置 / 速度序列
- 坐标系说明
- 插值或平滑来源说明

### 4.3 RadarGrid

含义：

- 雷达坐标网格定义

最低要求：

- 起始方位时间
- 起始斜距
- 方位向采样间隔
- 距离向采样间隔
- length / width
- wavelength
- look side
- PRF 或等效时间采样信息

### 4.4 Doppler

含义：

- 多普勒信息或可近似为零多普勒的说明

最低要求：

- doppler 类型（native / zero-doppler / estimated）
- 参考坐标系
- 数据来源或估计方法

### 4.5 DEM

最低要求：

- DEM 文件路径
- EPSG
- 分辨率
- 覆盖范围
- 高程单位

---

## 5. manifest.json 最小字段建议

> 当前 D2SAR 实现中，`manifest` 路径语义采用“**内部 metadata 相对化，外部源数据结构化/绝对化**”的策略：
>
> - `metadata/*`：相对 `manifest.json` 的相对路径；
> - `slc / ancillary / dem`：目录输入时通常为绝对路径；ZIP 输入时写为结构化对象；
> - consumer 统一通过路径解析函数恢复成真实可打开路径。

```json
{
  "dataset_id": "GF3_20260101_20260113_TRACK001",
  "sensor": "GF3",
  "mode": "stripmap",
  "processing_level": "external-normalized",
  "master": {
    "slc": {
      "path": "/data/master.zip",
      "storage": "zip",
      "member": "measurement/master.tiff"
    },
    "orbit": "/data/orbit/master_orbit.xml",
    "radargrid": "metadata/master_radargrid.json",
    "doppler": "metadata/master_doppler.json",
    "acquisition_time": "2026-01-01T10:15:00Z"
  },
  "slave": {
    "slc": "/data/slc/slave.slc",
    "orbit": "/data/orbit/slave_orbit.xml",
    "radargrid": "metadata/slave_radargrid.json",
    "doppler": "metadata/slave_doppler.json",
    "acquisition_time": "2026-01-13T10:15:00Z"
  },
  "dem": {
    "path": "/data/dem/dem.tif",
    "epsg": 4326
  },
  "aux": {
    "tec": null,
    "weather": null,
    "mask": null
  }
}
```

说明：

- 单景场景下 `slave` 可为空；
- `metadata/*.json` 推荐始终使用相对路径；
- 外部源数据路径不建议强制相对化，否则在 Docker 挂载别名场景下容易失真；
- ZIP 输入推荐写成 `{path, storage, member}` 结构，而不是预先展开成 `/vsizip/...`；
- `manifest.json` 负责把“文件集合”提升为“可编排数据包”。

---

## 6. 元数据文件最低要求

## 6.1 acquisition.json

建议包括：

- 传感器名
- 成像模式
- 极化
- 频段
- 产品来源
- 原始文件列表
- 适配器版本
- 数据生成时间

## 6.2 radargrid.json

建议包括：

- start sensing time
- starting range
- azimuth spacing
- range spacing
- length
- width
- wavelength
- look side
- PRF

## 6.3 doppler.json

建议包括：

- doppler 类型
- 估计方式
- 参考 epoch
- 插值或拟合参数

---

## 7. 不同处理链的最小输入要求

### 7.1 Focus

最低输入：

- 原始回波数据
- 轨道 / 姿态 / 定标参数
- 成像参数

### 7.2 RTC

最低输入：

- 单景 SLC / RSLC
- orbit
- DEM
- radar grid / doppler

### 7.3 InSAR

最低输入：

- master/slave 两景 SLC / RSLC
- 两景 orbit
- DEM
- radar grid
- doppler

### 7.4 DInSAR

最低输入：

- InSAR 所有输入
- 可选 mask
- 可选 TEC / weather / auxiliary corrections

---

## 8. 校验规则

外围系统至少应在进入算法前执行以下校验：

### 8.1 文件校验

- 路径存在
- 可读
- 文件大小非零

### 8.2 元数据校验

- EPSG 合法
- 时间字段可解析
- 单位字段明确
- 主从影像极化与频段兼容

### 8.3 几何校验

- DEM 覆盖处理区域
- orbit 时间范围覆盖成像时间
- radar grid 与 SLC 尺寸一致

### 8.4 流程校验

- 单景流程不应要求 slave
- InSAR / DInSAR 不允许缺主从任一关键对象

---

## 9. 与编排层的接口约定

编排层不直接理解某个传感器私有格式，而是只消费：

- `manifest.json`
- 标准目录结构
- 标准 metadata 文件

也就是说：

> **reader 负责把“传感器私有格式”翻译成“统一外部数据契约”；编排层只处理统一契约。**

---

## 10. 当前建议

第一阶段不要试图一次定义覆盖所有传感器的完美契约，而应：

1. 先选一个传感器；
2. 用真实样例文件验证本草案；
3. 在不破坏字段稳定性的前提下逐步扩展。

---

## 11. 小结

这份契约草案的目标不是复刻 ISCE3 内部实现，而是为外围系统建立一个稳定、清晰、可校验的输入边界。后续所有 reader、CLI pipeline、Docker 和 Web，都应建立在这层统一边界之上。
