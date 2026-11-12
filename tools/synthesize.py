"""翻拍样本程序化合成（屏幕翻拍 / 打印翻拍）。

从任意身份证图/文档图生成仿真翻拍样本，用于：
  1) v1.0 基线权重训练底座；2) 再训练数据扩充。
用法:
  python tools/synthesize.py --src data/raw --out data/synthetic --type screen --n 3
"""
from __future__ import annotations

import argparse
import io
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.core.preprocess import decode_to_rgb  # noqa: E402

try:
    import cv2
    _HAS_CV2 = True
except Exception:  # pragma: no cover
    cv2 = None
    _HAS_CV2 = False


def _jpeg(data: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", data, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return data
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def add_moire(img: np.ndarray, angle: float, freq: float, amp: float) -> np.ndarray:
    """乘法叠加正弦摩尔纹。"""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    theta = np.deg2rad(angle)
    grid = xx * np.cos(theta) + yy * np.sin(theta)
    phase = random.uniform(0, 2 * np.pi)
    pattern = np.sin(2 * np.pi * freq * grid + phase)
    p3 = pattern[..., None].astype(np.float32)
    out = img.astype(np.float32) * (1.0 + amp * p3) + amp * 30.0 * p3
    return np.clip(out, 0, 255).astype(np.uint8)


def add_screen_grid(img: np.ndarray, spacing: int, strength: float) -> np.ndarray:
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    grid = (np.sin(2 * np.pi * xx / spacing) + np.sin(2 * np.pi * yy / spacing)) / 2.0
    return np.clip(img.astype(np.float32) + strength * 18.0 * grid[..., None], 0, 255).astype(np.uint8)


def add_halftone(img: np.ndarray, spacing: int, strength: float) -> np.ndarray:
    """半色调网点：周期点阵变暗。"""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dots = (np.sin(2 * np.pi * xx / spacing) * np.sin(2 * np.pi * yy / spacing))
    mask = (dots > 0.4).astype(np.float32)[..., None]
    return np.clip(img.astype(np.float32) * (1.0 - strength * 0.5 * mask), 0, 255).astype(np.uint8)


def add_paper_noise(img: np.ndarray, sigma: float) -> np.ndarray:
    h, w = img.shape[:2]
    # 低频亮度斑驳
    coarse = np.zeros((max(4, h // 16), max(4, w // 16)), dtype=np.float32)
    coarse = np.random.normal(0, sigma, coarse.shape).astype(np.float32)
    if _HAS_CV2:
        coarse = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_LINEAR)
    noise = coarse[..., None].astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def synthesize_screen(img: np.ndarray, rng: random.Random) -> np.ndarray:
    out = img.copy()
    out = add_moire(out, rng.uniform(0, 180), rng.uniform(0.02, 0.08), rng.uniform(0.05, 0.25))
    out = add_screen_grid(out, rng.choice([3, 4, 5, 6]), rng.uniform(0.3, 0.8))
    # 亮度/对比度衰减
    a, b = rng.uniform(0.75, 0.95), rng.uniform(-15, 5)
    out = np.clip(out.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
    # 色温偏移（蓝移）
    out = out.astype(np.float32)
    out[..., 2] += rng.uniform(8, 30)
    out[..., 0] -= rng.uniform(0, 10)
    out = np.clip(out, 0, 255).astype(np.uint8)
    if _HAS_CV2:
        out = cv2.GaussianBlur(out, (3, 3), rng.uniform(0.3, 1.2))
        out = _jpeg(out, rng.randint(55, 80))
    return out


def synthesize_print(img: np.ndarray, rng: random.Random) -> np.ndarray:
    out = img.copy()
    out = add_halftone(out, rng.choice([3, 4, 5]), rng.uniform(0.3, 0.7))
    out = add_paper_noise(out, rng.uniform(2, 8))
    # 饱和度下降
    if _HAS_CV2:
        hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 1] *= rng.uniform(0.4, 0.7)
        out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
    else:
        gray = (0.299 * out[..., 0] + 0.587 * out[..., 1] + 0.114 * out[..., 2]).astype(np.uint8)
        g = rng.uniform(0.4, 0.7)
        out = np.clip(out.astype(np.float32) * g + gray[..., None] * (1 - g), 0, 255).astype(np.uint8)
    if _HAS_CV2:
        out = cv2.GaussianBlur(out, (3, 3), rng.uniform(0.6, 1.5))
        out = _jpeg(out, rng.randint(50, 75))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--type", choices=["screen", "print"], required=True)
    ap.add_argument("--n", type=int, default=2, help="每张源图生成样本数")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    srcs = []
    for root, _, files in os.walk(args.src):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                srcs.append(os.path.join(root, f))
    if not srcs:
        print("未找到源图片")
        return 1
    os.makedirs(args.out, exist_ok=True)
    for i, sp in enumerate(srcs):
        try:
            img = decode_to_rgb(open(sp, "rb").read())
        except Exception as exc:  # noqa: BLE001
            print(f"跳过 {sp}: {exc}")
            continue
        for j in range(args.n):
            out = synthesize_screen(img, rng) if args.type == "screen" else synthesize_print(img, rng)
            name = f"{os.path.splitext(os.path.basename(sp))[0]}_{args.type}_{j}.jpg"
            dst = os.path.join(args.out, name)
            if _HAS_CV2:
                cv2.imwrite(dst, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
            else:
                from PIL import Image
                Image.fromarray(out).save(dst)
    print(f"完成：生成 {args.type} 样本 {len(srcs) * args.n} 张 → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
