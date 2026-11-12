"""翻拍分类器再训练（特征逻辑回归，纯 numpy，CPU 可跑）。

从 data/recapture/{original,screen_recapture,print_recapture}/ 提取信号特征，
训练 3 类 softmax 逻辑回归，导出 weights.json。
可选 --backbone efficientnet_b0 提示可切换到 CNN 路线（需 torch+timm，见注释）。

用法:
  python tools/train_recapture.py --data data/recapture --epochs 200 --out runs/recap_v1.1 --cpu
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.core import detect  # noqa: E402
from app.core.preprocess import decode_to_rgb, normalize, to_gray  # noqa: E402
from app.core.recapture.signals import collect as collect_signals  # noqa: E402

FEATURE_NAMES = ["moire", "screen_grid", "bezel", "highlight", "texture_degradation",
                 "halftone", "paper_noise", "saturation_drop", "color_shift"]
CLASSES = ["original", "screen_recapture", "print_recapture"]

SIGNAL_WEIGHTS = {"moire": 0.30, "screen_grid": 0.10, "bezel": 0.15, "highlight": 0.10,
                  "texture_degradation": 0.10, "halftone": 0.25, "paper_noise": 0.10,
                  "saturation_drop": 0.05, "color_shift": 0.05}


def extract_features(rgb: np.ndarray) -> Dict[str, float]:
    rgb = normalize(rgb, 1600)
    quad = detect.detect_card_quad(rgb)
    full_gray = to_gray(rgb)
    if quad is not None:
        card = detect.warp_card(rgb, quad)
    else:
        card = rgb
    sig = collect_signals(card, full_gray, quad)
    feats = {f: sig["features"].get(f, 0.0) for f in FEATURE_NAMES}
    return feats


def load_dataset(data_dir: str, limit_per_class: int = 0) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    X, y, names = [], [], []
    for ci, cls in enumerate(CLASSES):
        d = os.path.join(data_dir, cls)
        if not os.path.isdir(d):
            print(f"警告：目录不存在 {d}")
            continue
        files = [f for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))]
        if limit_per_class:
            random.Random(0).shuffle(files)
            files = files[:limit_per_class]
        for f in files:
            try:
                rgb = decode_to_rgb(open(os.path.join(d, f), "rb").read())
                feats = extract_features(rgb)
            except Exception as exc:  # noqa: BLE001
                print(f"跳过 {f}: {exc}")
                continue
            X.append([feats[n] for n in FEATURE_NAMES])
            y.append(ci)
            names.append(f"{cls}/{f}")
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64), names


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def train_logistic(X: np.ndarray, y: np.ndarray, epochs: int = 200, lr: float = 0.5,
                   l2: float = 1e-3, class_weights: Dict[int, float] | None = None,
                   seed: int = 0, verbose: bool = True) -> Dict:
    n, f = X.shape
    k = len(CLASSES)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.05, (k, f)).astype(np.float64)
    b = np.zeros((k,), dtype=np.float64)
    Y = np.eye(k)[y]

    sw = np.ones(n, dtype=np.float64)
    if class_weights:
        for ci, cw in class_weights.items():
            sw[y == ci] = cw

    history = []
    for ep in range(epochs):
        z = X.astype(np.float64) @ W.T + b
        p = _softmax(z)
        grad = (p - Y) * sw[:, None] / sw.sum()
        dW = grad.T @ X.astype(np.float64) + l2 * W
        db = grad.sum(axis=0)
        W -= lr * dW
        b -= lr * db
        loss = float(-(Y * np.log(p + 1e-9)).sum(axis=1).mean()) + 0.5 * l2 * float((W ** 2).sum())
        pred = p.argmax(axis=1)
        acc = float((pred == y).mean())
        history.append({"epoch": ep, "loss": round(loss, 5), "acc": round(acc, 4)})
        if verbose and (ep % 20 == 0 or ep == epochs - 1):
            print(f"epoch {ep:4d}  loss {loss:.5f}  acc {acc:.4f}")

    logits = {}
    for ci, cls in enumerate(CLASSES):
        logits[cls] = {"bias": round(float(b[ci]), 6),
                       "weights": {FEATURE_NAMES[j]: round(float(W[ci, j]), 6) for j in range(f)}}
    return {"logits": logits, "history": history}


def train(data_dir: str, out_dir: str, epochs: int = 200, limit_per_class: int = 0,
          class_weights: Dict[str, float] | None = None, lr: float = 0.5) -> str:
    X, y, names = load_dataset(data_dir, limit_per_class)
    if len(X) < 10:
        raise RuntimeError(f"训练样本不足（{len(X)} 张），请至少提供每类若干张图片")
    cw = None
    if class_weights:
        cw = {CLASSES.index(k): v for k, v in class_weights.items() if k in CLASSES}
    res = train_logistic(X, y, epochs=epochs, lr=lr, class_weights=cw)

    os.makedirs(out_dir, exist_ok=True)
    weights = {
        "version": os.path.basename(out_dir),
        "model_type": "feature_logistic",
        "feature_names": FEATURE_NAMES,
        "temperature": 1.5,
        "logits": res["logits"],
        "signal_weights": SIGNAL_WEIGHTS,
        "fusion_weights": {"w_screen": 0.45, "w_print": 0.35, "w_signal": 0.20},
        "thresholds": {"low": 0.30, "high": 0.60},
    }
    wp = os.path.join(out_dir, "weights.json")
    with open(wp, "w", encoding="utf-8") as fh:
        json.dump(weights, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump({"num_samples": len(X), "history": res["history"]}, fh, ensure_ascii=False, indent=2)
    print(f"训练完成：{len(X)} 样本 → {wp}")
    return wp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="runs/recap_latest")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--limit-per-class", type=int, default=0)
    ap.add_argument("--class-weights", type=str, default=None, help="如 original:0.5,screen_recapture:1,print_recapture:1")
    ap.add_argument("--backbone", default="feature_logistic", help="预留：feature_logistic 或 efficientnet_b0")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    cw = None
    if args.class_weights:
        cw = {}
        for part in args.class_weights.split(","):
            k, v = part.split(":")
            cw[k.strip()] = float(v)
    if args.backbone != "feature_logistic":
        print("提示：--backbone efficientnet_b0 为可选 CNN 路线，需安装 torch+timm（见 requirements-dev.txt）。"
              "当前使用 feature_logistic 训练。")
    train(args.data, args.out, epochs=args.epochs, limit_per_class=args.limit_per_class, class_weights=cw, lr=args.lr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
