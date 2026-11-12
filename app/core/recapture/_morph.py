"""纯 numpy 形态学辅助（膨胀等）。"""
from __future__ import annotations

import numpy as np


def dilate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """二值掩码膨胀。radius<=0 时原样返回。"""
    mask = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return mask
    from numpy.lib.stride_tricks import sliding_window_view
    k = 2 * radius + 1
    ap = np.pad(mask, radius, mode="constant", constant_values=False)
    win = sliding_window_view(ap, (k, k))
    return win.any(axis=(2, 3))
