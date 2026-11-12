"""生成合成基准测试集（原件/屏幕翻拍/打印翻拍）用于验收与再训练演示。

用法: python tests/make_benchmark.py --out tests/data/benchmark --per 10
"""
from __future__ import annotations

import argparse
import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image  # noqa: E402
from tools.synthesize import add_halftone, add_moire, add_screen_grid  # noqa: E402


def _card():
    W, H, pad = 856, 540, 100
    img = np.full((H + 2 * pad, W + 2 * pad, 3), (70, 80, 90), dtype=np.uint8)
    card = np.full((H, W, 3), (242, 240, 235), dtype=np.uint8)
    yy, xx = np.mgrid[0:H, 0:W]
    card = np.clip(card.astype(np.float32) + (np.sin(xx / 0.9) + np.sin(yy / 0.8))[..., None] * 3.0, 0, 255)
    rng = np.random.default_rng(0)
    for _ in range(25):
        x0, y0 = int(rng.integers(0, W - 90)), int(rng.integers(0, H - 20))
        card[y0:y0 + 18, x0:x0 + 85] = int(rng.integers(0, 110))
    img[pad:pad + H, pad:pad + W] = card.astype(np.uint8)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tests/data/benchmark")
    ap.add_argument("--per", type=int, default=10)
    args = ap.parse_args()
    rng = np.random.default_rng(1)
    for cls in ("original", "screen_recapture", "print_recapture"):
        d = os.path.join(args.out, cls)
        os.makedirs(d, exist_ok=True)
        for i in range(args.per):
            img = _card()
            if cls == "screen_recapture":
                img = add_moire(img, rng.uniform(0, 180), rng.uniform(0.03, 0.07), rng.uniform(0.1, 0.25))
                img = add_screen_grid(img, rng.choice([3, 4, 5]), rng.uniform(0.4, 0.7))
            elif cls == "print_recapture":
                img = add_halftone(img, rng.choice([3, 4]), rng.uniform(0.4, 0.7))
                img = np.clip(img.astype(np.float32) * 0.85, 0, 255).astype(np.uint8)
            Image.fromarray(img).save(os.path.join(d, f"{cls}_{i:03d}.jpg"), quality=90)
    print(f"已生成基准集 → {args.out}（每类 {args.per} 张）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
