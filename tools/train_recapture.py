"""翻拍分类器再训练（特征逻辑回归，纯 numpy，CPU 可跑）。

从 data/recapture/{original,screen_recapture,print_recapture}/ 提取信号特征，
训练 3 类 softmax 逻辑回归，导出 weights.json。

典型用法:
  python tools/train_recapture.py --data data/recapture --epochs 200 --out runs/recap_v1.1 --cpu

增量/多轮训练（在上一版权重基础上继续）:
  python tools/train_recapture.py --data data/recapture --epochs 100 --out runs/recap_v1.2 --init-from models/recapture/weights.json

注意:
  - 翻拍判定是「三类」分类，至少需要 original + 一种翻拍类别；只有单类样本无法训练有意义的判别边界。
  - 本模型为特征逻辑回归，训练是确定性的，数据变化后直接重新训练即可（无需 GPU）。
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


def _load_init_WB(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """从已有 weights.json 读取初始化权重（用于增量/多轮训练）。"""
    with open(path, "r", encoding="utf-8") as f:
        w = json.load(f)
    k, fdim = len(CLASSES), len(FEATURE_NAMES)
    W = np.zeros((k, fdim), dtype=np.float64)
    b = np.zeros((k,), dtype=np.float64)
    for ci, cls in enumerate(CLASSES):
        spec = w.get("logits", {}).get(cls, {})
        b[ci] = float(spec.get("bias", 0.0))
        for j, fn in enumerate(FEATURE_NAMES):
            W[ci, j] = float(spec.get("weights", {}).get(fn, 0.0))
    return W, b


def train_logistic(X: np.ndarray, y: np.ndarray, epochs: int = 200, lr: float = 0.5,
                   l2: float = 1e-3, class_weights: Dict[int, float] | None = None,
                   seed: int = 0, verbose: bool = True,
                   init_W: np.ndarray | None = None, init_b: np.ndarray | None = None) -> Dict:
    n, fdim = X.shape
    k = len(CLASSES)
    rng = np.random.default_rng(seed)
    if init_W is not None and init_b is not None:
        W = init_W.astype(np.float64).copy()
        b = init_b.astype(np.float64).copy()
    else:
        W = rng.normal(0, 0.05, (k, fdim)).astype(np.float64)
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
                       "weights": {FEATURE_NAMES[j]: round(float(W[ci, j]), 6) for j in range(fdim)}}
    return {"logits": logits, "history": history}


def _validate_classes(y: np.ndarray) -> None:
    """校验样本类别，避免只给单一类别时产出退化模型。"""
    counts = np.bincount(y, minlength=len(CLASSES))
    present = [cls for cls, c in zip(CLASSES, counts) if c > 0]
    for cls, c in zip(CLASSES, counts):
        if c == 0:
            print(f"警告：类别 {cls} 无样本，模型将无法学习该类边界")
        elif c < 5:
            print(f"警告：类别 {cls} 仅 {c} 张样本，建议至少 10 张")
    if len(present) < 2:
        raise RuntimeError(
            f"仅检测到 1 个类别（{present[0] if present else '无'}）。翻拍判定是三类分类，"
            "至少需要 original + 一种翻拍类别（screen_recapture 或 print_recapture）。\n"
            "说明：original 类不需要与翻拍样本是同一张卡，任意身份证原件的实拍照片即可；"
            "请补充原件实拍样本后重试。"
        )
    if len(present) < len(CLASSES):
        print(f"提示：当前仅有 {len(present)} 类样本（{'/'.join(present)}），缺失类别的判别能力有限；"
              "建议补齐三类以完整覆盖。")


def train(data_dir: str, out_dir: str, epochs: int = 200, limit_per_class: int = 0,
          class_weights: Dict[str, float] | None = None, lr: float = 0.5,
          init_from: str | None = None) -> str:
    X, y, names = load_dataset(data_dir, limit_per_class)
    if len(X) < 10:
        raise RuntimeError(f"训练样本不足（{len(X)} 张），请至少提供若干张图片")
    _validate_classes(y)

    cw = None
    if class_weights:
        cw = {CLASSES.index(k): v for k, v in class_weights.items() if k in CLASSES}
    init_W = init_b = None
    if init_from and os.path.exists(init_from):
        init_W, init_b = _load_init_WB(init_from)
        print(f"从已有权重热启动：{init_from}")
    res = train_logistic(X, y, epochs=epochs, lr=lr, class_weights=cw,
                         init_W=init_W, init_b=init_b)

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
        json.dump({"num_samples": len(X),
                   "class_counts": {c: int(n) for c, n in zip(CLASSES, np.bincount(y, minlength=len(CLASSES)))},
                   "init_from": init_from,
                   "history": res["history"]}, fh, ensure_ascii=False, indent=2)
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
    ap.add_argument("--init-from", "--resume", dest="init_from", default=None,
                    help="从已有 weights.json 热启动（增量/多轮训练）")
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
    train(args.data, args.out, epochs=args.epochs, limit_per_class=args.limit_per_class,
          class_weights=cw, lr=args.lr, init_from=args.init_from)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
