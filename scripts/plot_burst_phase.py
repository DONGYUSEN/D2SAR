#!/usr/bin/env python3
"""生成干涉相位彩色 PNG 图（屏蔽无效值）"""

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

def phase_to_color(phase: np.ndarray, invalid_mask: np.ndarray | None = None) -> np.ndarray:
    """将相位转换为彩虹色图，无效值屏蔽为透明/黑色"""
    # 设置无效掩码（零值视为无效）
    if invalid_mask is None:
        invalid_mask = (phase == 0) | np.isnan(phase) | np.isinf(phase)

    # 相位范围 [-π, π] 映射到 [0, 1] 用于色彩
    normalized = (phase + np.pi) / (2.0 * np.pi)
    normalized = np.clip(normalized, 0, 1)

    # 使用 matplotlib 的 hue 色相映射 (0=红, 0.33=绿, 0.66=蓝, 1=红)
    hues = normalized
    saturation = np.ones_like(hues)
    value = np.ones_like(hues)

    # 转换为 RGB
    from matplotlib.colors import hsv_to_rgb
    rgb = hsv_to_rgb(np.stack([hues, saturation, value], axis=-1))

    # 无效区域设为黑色
    rgb[invalid_mask] = 0

    return (rgb * 255).astype(np.uint8)


def main():
    ifg_path = Path("/home/ysdong/Software/D2SAR/result/IW1/20230719_20230625/work/p2_burst_ifg/burst_000_interferogram.npy")
    output_path = ifg_path.parent / "burst_000_phase.png"

    print(f"读取干涉数据: {ifg_path}")
    ifg = np.load(ifg_path)
    print(f"数据形状: {ifg.shape}, 类型: {ifg.dtype}")

    # 计算相位
    phase = np.angle(ifg)
    print(f"相位范围: [{phase.min():.3f}, {phase.max():.3f}] rad")

    # 零值区域视为无效（burst valid window 外的区域）
    amplitude = np.abs(ifg)
    invalid = (amplitude == 0) | np.isnan(amplitude) | np.isinf(amplitude)
    print(f"无效像素数: {invalid.sum()} / {invalid.size} ({100*invalid.sum()/invalid.size:.1f}%)")

    # 生成彩色相位图
    print("生成彩色相位图...")
    rgb = phase_to_color(phase, invalid)

    # 保存 PNG
    print(f"保存图像: {output_path}")
    plt.imsave(output_path, rgb)
    print("完成!")

    # 同时保存一个带色彩条的版本
    output_path_colorbar = ifg_path.parent / "burst_000_phase_cbar.png"
    fig, ax = plt.subplots(figsize=(16, 6))
    im = ax.imshow(phase, cmap="hsv", vmin=-np.pi, vmax=np.pi)
    ax.set_title("Burst 000 Interferometric Phase (radians)")
    ax.set_xlabel("Range (pixels)")
    ax.set_ylabel("Azimuth (lines)")
    cbar = fig.colorbar(im, ax=ax, label="Phase (rad)")
    fig.savefig(output_path_colorbar, dpi=150, bbox_inches="tight")
    print(f"保存带色彩条图像: {output_path_colorbar}")


if __name__ == "__main__":
    main()
