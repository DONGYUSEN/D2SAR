"""
Goldstein-Werner 滤波模块（已合并至 insar_filtering.py，保留作向后兼容导入）

实现基于 ISCE2 的 power-spectral filter 算法，用于干涉图滤波。
算法参考: Goldstein, R. M., & Werner, C. L. (1998). Radar interferogram
          filtering using the power spectral density.
"""

from insar_filtering import goldstein_filter as goldstein_filter

__all__ = ["goldstein_filter"]
