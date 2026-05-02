# 任务计划：拆分 Sentinel TOPS RTC/InSAR 处理代码

## 目标
分析 `isce3` 目录中与 Sentinel-1/TOPS、RTC、InSAR 相关的处理流程，识别可复用入口、配置、数据流和依赖边界，为后续将哨兵数据处理逻辑单独抽取成 `tops_rtc` / `insar` 代码做准备。

## 阶段
| 阶段 | 状态 | 内容 |
|---|---|---|
| 1 | complete | 定位 `isce3` 中 Sentinel/TOPS/RTC/InSAR 相关文件和入口 |
| 2 | complete | 梳理处理链路：输入、配置、核心步骤、输出 |
| 3 | complete | 对比现有 D2SAR 外围脚本与 `isce3` 原生流程的边界 |
| 4 | complete | 提出 Sentinel 独立模块拆分方案 |
| 5 | complete | 总结风险、依赖和下一步实现建议 |
| 6 | complete | 实现 Sentinel-1 SAFE 导入模块 |
| 7 | complete | 对照 ISCE2 TOPS 和 ISCE3 输入需求复核 Sentinel importer |
| 8 | complete | 补充 Sentinel importer 的 burst-aware TOPS 元数据解析 |
| 9 | complete | 补充 swath/polarization 成员选择与 manifest.safe 处理信息解析 |
| 10 | complete | 补充 EOF 轨道导入与 overlap/ESD 派生元数据 |
| 11 | complete | 新增 Sentinel-1 EOF 轨道解析、自有下载代码和导入后 apply 工具 |
| 12 | complete | 使用真实 Sentinel ZIP 验证自有轨道下载、导入和 burst RTC/topo/geocode 链路 |
| 13 | complete | 全 9 burst materialize/topo/rtc/geocode 完整链路 |
| 14 | complete | swath mosaic：同 IW 内多 burst 按地理 UTM 网格拼接 |

## 决策记录
- 仅做代码分析和方案建议，不直接重构代码，除非用户后续确认实现。
- 用户确认首先完成哨兵数据导入模块；已新增 `scripts/sentinel_importer.py`，保持 D2SAR manifest/metadata 输出风格。
- Sentinel 轨道能力采用独立工具 `scripts/sentinel_orbit.py`，导入阶段仅在用户提供 `orbit_dir` 或 `download_orbit` 时启用，避免默认导入被网络下载阻塞。
- Sentinel 轨道下载必须使用 D2SAR `scripts` 内自有代码，不调用、不修改 ISCE2/ISCE3 的轨道下载程序。

## 错误记录
| 错误 | 处理 |
|---|---|
| 暂无 | 暂无 |
