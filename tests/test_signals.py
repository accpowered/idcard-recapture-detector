"""信号分析在合成图像上的单调性测试。"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from app.core.recapture.signals import (  # noqa: E402
    moire_score, halftone_score, texture_degradation_score, collect,
)


def _base_card(seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((540, 856, 3), 235, dtype=np.uint8)
    yy, xx = np.mgrid[0:540, 0:856]
    tex = (np.sin(xx / 0.9) + np.sin(yy / 0.8)) * 3.0
    img = np.clip(img.astype(np.float32) + tex[..., None], 0, 255).astype(np.uint8)
    for _ in range(30):
        x0, y0 = int(rng.integers(0, 700)), int(rng.integers(0, 460))
        cv = int(rng.integers(0, 120))
        img[y0:y0 + 20, x0:x0 + 90] = cv
    return img


def _gray(img):
    return img.mean(axis=2).astype(np.float32)


def test_moire_monotonic():
    from tools.synthesize import add_moire
    base = _base_card()
    mo = add_moire(base, angle=45, freq=0.05, amp=0.25)
    s0, _ = moire_score(_gray(base))
    s1, _ = moire_score(_gray(mo))
    assert s1 >= s0


def test_halftone_monotonic():
    from tools.synthesize import add_halftone
    base = _base_card()
    hp = add_halftone(base, spacing=4, strength=0.6)
    s0, _ = halftone_score(_gray(base))
    s1, _ = halftone_score(_gray(hp))
    assert s1 >= s0


def test_texture_degradation_monotonic():
    base = _base_card()
    k = np.ones((9, 9), np.float32) / 81.0
    from numpy.lib.stride_tricks import sliding_window_view
    g = _gray(base)
    ap = np.pad(g, 4, mode="reflect")
    blur = sliding_window_view(ap, (9, 9)).mean(axis=(2, 3))[:g.shape[0], :g.shape[1]]
    d0, _ = texture_degradation_score(g)
    d1, _ = texture_degradation_score(blur)
    assert d1 >= d0  # 模糊后纹理退化得分更高


def test_collect_runs():
    base = _base_card()
    r = collect(base)
    assert "evidences" in r and len(r["evidences"]) == 9
    assert "features" in r and "moire" in r["features"]
