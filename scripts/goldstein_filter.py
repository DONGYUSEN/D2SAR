"""
Goldstein-Werner 滤波模块

实现基于 ISCE2 的 power-spectral filter 算法，用于干涉图滤波。
算法参考: Goldstein, R. M., & Werner, C. L. (1998). Radar interferogram
          filtering using the power spectral density.
"""

import numpy as np
from typing import Optional


def goldstein_filter(
    interferogram: np.ndarray,
    alpha: float = 0.5,
    window_size: int = 32,
    step: Optional[int] = None,
) -> np.ndarray:
    if interferogram.ndim != 2:
        raise ValueError("goldstein_filter expects a 2D interferogram")

    if not interferogram.dtype.isbuiltin:
        raise ValueError("interferogram must be a numpy built-in complex dtype")

    if interferogram.dtype != np.complex64:
        interferogram = interferogram.astype(np.complex64)

    rows, cols = interferogram.shape

    if step is None:
        step = window_size // 2

    if window_size <= 0 or (window_size & (window_size - 1)) != 0:
        raise ValueError("window_size must be a positive power of 2")

    hanning = _create_hanning_window(window_size)

    filtered = np.zeros_like(interferogram, dtype=np.complex64)
    weight_sum = np.zeros((rows, cols), dtype=np.float64)

    for row_start in range(0, rows - window_size + 1, step):
        for col_start in range(0, cols - window_size + 1, step):
            window_data = interferogram[
                row_start : row_start + window_size,
                col_start : col_start + window_size,
            ].copy()

            windowed = window_data * hanning

            spectrum = np.fft.fft2(windowed)

            psd = np.abs(spectrum) ** 2

            weight = np.power(psd + 1e-10, alpha / 2.0)

            weighted_spectrum = spectrum * weight

            filtered_window = np.fft.ifft2(weighted_spectrum)

            windowed_output = filtered_window * hanning

            filtered[row_start : row_start + window_size, col_start : col_start + window_size] += windowed_output
            weight_sum[row_start : row_start + window_size, col_start : col_start + window_size] += hanning.real**2

    weight_sum[weight_sum == 0] = 1.0
    filtered = filtered / weight_sum

    return filtered.astype(np.complex64)


def _create_hanning_window(size: int) -> np.ndarray:
    h1d = np.hanning(size)
    return np.outer(h1d, h1d).astype(np.complex64)
