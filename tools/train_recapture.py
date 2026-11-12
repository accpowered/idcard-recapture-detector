"""翻拍分类器再训练（特征逻辑回归，纯 numpy，CPU 可跑）。

从 data/recapture/{original,screen_recapture[,print_recapture]}/ 提取信号特征，
训练 softmax 逻辑回归并导出 weights.json。类别集合由 --mode 决定：
  screen_only        -> original + screen_recapture
  screen_and_print   -> original + screen_recapture + print_recapture（默认）

典型用法:
  python tools/train_recapture.py --data data/recapture --epochs 200 --out runs/recap_v1.1
  python tools/train_recapture.py --data data/recapture --mode screen_only --out runs/recap_screen

注意:
  - 至少需要 original + 一种翻拍类别；只有单类样本无法训练有意义的判别边界。
  - 本模型为特征逻辑回归，训练是确定性的，数据变化后直接重新训练即可（无需 GPU）。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.core import detect  # noqa: E402
from app.core.preprocess import decode_to_rgb, normalize, to_gray  # noqa: E402
from app.core.recapture import modes as mode_defs  # noqa: E402
from app.core.recapture.signals import collect as collect_signals  # noqa: E402

FEATURE_NAMES = ["moire", "screen_grid", "bezel", "highlight", "texture_degradation",
                 "halftone", "paper_noise", "saturation_drop", "color_shift"]
CLASSES = list(mode_defs.ALL_CLASSES)  # 兼容旧引用
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

SIGNAL_WEIGHTS = {"moire": 0.30, "screen_grid": 0.10, "bezel": 0.15, "highlight": 0.10,
                  "texture_degradation": 0.10, "halftone": 0.25, "paper_noise": 0.10,
                  "saturation_drop": 0.05, "color_shift": 0.05}

ProgressFn = Optional[Callable[[int, int], None]]


def resolve_classes(mode: Optional[str] = None) -> List[str]:
    """按模式返回训练类别；mode=None 时保持三类（CLI 默认兼容）。"""
    if not mode:
        return list(CLASSES)
    return list(mode_defs.classes_for_mode(mode))


def extract_features(rgb: np.ndarray) -> Dict[str, float]:
    rgb = normalize(rgb, 1600)
    quad = detect.detect_card_quad(rgb)
    full_gray = to_gray(rgb)
    if quad is not None:
        card = detect.warp_card(rgb, quad)
    else:
        card = rgb
    sig = collect_signals(card, full_gray, quad)
    return {f: sig["features"].get(f, 0.0) for f in FEATURE_NAMES}


def load_dataset(data_dir: str, classes: Optional[List[str]] = None, limit_per_class: int = 0,
                 progress: ProgressFn = None
                 ) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, int]]:
    """按类别目录读取图片并提取特征，返回 (X, y, names, counts)。"""
    classes = list(classes or CLASSES)
    X, y, names = [], [], []
    counts: Dict[str, int] = {c: 0 for c in classes}
    for ci, cls in enumerate(classes):
        d = os.path.join(data_dir, cls)
        if not os.path.isdir(d):
            print("警告：目录不存在 {}".format(d))
            continue
        files = [f for f in os.listdir(d) if f.lower().endswith(IMAGE_EXTS)]
        files.sort()
        if limit_per_class:
            random.Random(0).shuffle(files)
            files = files[:limit_per_class]
        for f in files:
            try:
                with open(os.path.join(d, f), "rb") as fh:
                    rgb = decode_to_rgb(fh.read())
                feats = extract_features(rgb)
            except Exception as exc:  # noqa: BLE001
                print("跳过 {}: {}".format(f, exc))
                continue
            X.append([feats[n] for n in FEATURE_NAMES])
            y.append(ci)
            names.append("{}/{}".format(cls, f))
            counts[cls] = counts.get(cls, 0) + 1
            if progress is not None:
                try:
                    progress(len(names), 0)
                except Exception:  # noqa: BLE001
                    pass
    total = len(names)
    if progress is not None:
        try:
            progress(total, total)
        except Exception:  # noqa: BLE001
            pass
    return (np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64), names, counts)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _load_init_WB(path: str, classes: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """从已有 weights.json 读取初始化权重（用于增量/多轮训练）。"""
    with open(path, "r", encoding="utf-8") as f:
        w = json.load(f)
    stored = list(w.get("classes") or CLASSES)
    if set(stored) != set(classes):
        raise ValueError("初始化权重的类别 {} 与当前训练类别 {} 不一致".format(stored, classes))
    fdim = len(FEATURE_NAMES)
    W = np.zeros((len(classes), fdim), dtype=np.float64)
    b = np.zeros((len(classes),), dtype=np.float64)
    for ci, cls in enumerate(classes):
        spec = w.get("logits", {}).get(cls, {})
        b[ci] = float(spec.get("bias", 0.0))
        for j, fn in enumerate(FEATURE_NAMES):
            W[ci, j] = float(spec.get("weights", {}).get(fn, 0.0))
    return W, b


def train_logistic(X: np.ndarray, y: np.ndarray, epochs: int = 200, lr: float = 0.5,
                   l2: float = 1e-3, class_weights: Optional[Dict[int, float]] = None,
                   seed: int = 0, verbose: bool = True,
                   init_W: Optional[np.ndarray] = None, init_b: Optional[np.ndarray] = None,
                   classes: Optional[List[str]] = None,
                   progress: ProgressFn = None) -> Dict:
    classes = list(classes or CLASSES)
    n, fdim = X.shape
    k = len(classes)
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
            print("epoch {:4d}  loss {:.5f}  acc {:.4f}".format(ep, loss, acc))
        if progress is not None and (ep % 20 == 0 or ep == epochs - 1):
            try:
                progress(ep + 1, epochs)
            except Exception:  # noqa: BLE001
                pass

    logits = {}
    for ci, cls in enumerate(classes):
        logits[cls] = {"bias": round(float(b[ci]), 6),
                       "weights": {FEATURE_NAMES[j]: round(float(W[ci, j]), 6) for j in range(fdim)}}
    return {"logits": logits, "history": history}


def _validate_classes(y: np.ndarray, classes: List[str]) -> None:
    """校验样本类别，避免只给单一类别时产出退化模型。"""
    counts = np.bincount(y, minlength=len(classes))
    present = [cls for cls, c in zip(classes, counts) if c > 0]
    for cls, c in zip(classes, counts):
        if c == 0:
            print("警告：类别 {} 无样本，模型将无法学习该类边界".format(cls))
        elif c < 5:
            print("警告：类别 {} 仅 {} 张样本，建议至少 10 张".format(cls, c))
    if len(present) < 2:
        raise RuntimeError(
            "仅检测到 1 个类别（{}）。翻拍判定至少需要 original + 一种翻拍类别"
            "（screen_recapture 或 print_recapture）。\n"
            "说明：original 类不需要与翻拍样本是同一张卡，任意身份证原件的实拍照片即可；"
            "请补充原件实拍样本后重试。".format(present[0] if present else "无"))


def fit_weights(X: np.ndarray, y: np.ndarray, classes: List[str], epochs: int = 200,
                lr: float = 0.5, l2: float = 1e-3,
                class_weights: Optional[Dict[str, float]] = None,
                init_from: Optional[str] = None, seed: int = 0,
                progress: ProgressFn = None, verbose: bool = False) -> Dict:
    """在已提取的特征上训练并返回 weights 字典与训练历史。"""
    classes = list(classes)
    _validate_classes(y, classes)
    cw = None
    if class_weights:
        cw = {classes.index(k): v for k, v in class_weights.items() if k in classes}
    init_W = init_b = None
    if init_from and os.path.exists(init_from):
        init_W, init_b = _load_init_WB(init_from, classes)
    res = train_logistic(X, y, epochs=epochs, lr=lr, class_weights=cw,
                         init_W=init_W, init_b=init_b, classes=classes,
                         progress=progress, verbose=verbose)
    weights = {
        "version": "train",
        "model_type": "feature_logistic",
        "classes": classes,
        "recapture_mode": mode_defs.derive_mode(classes),
        "feature_names": FEATURE_NAMES,
        "temperature": 1.5,
        "logits": res["logits"],
        "signal_weights": SIGNAL_WEIGHTS,
        "fusion_weights": {"w_screen": 0.45, "w_print": 0.35, "w_signal": 0.20},
        "thresholds": {"low": 0.30, "high": 0.60},
    }
    return {"weights": weights, "history": res["history"]}


def train(data_dir: str, out_dir: str, epochs: int = 200, limit_per_class: int = 0,
          class_weights: Optional[Dict[str, float]] = None, lr: float = 0.5,
          init_from: Optional[str] = None, mode: Optional[str] = None,
          progress: ProgressFn = None) -> str:
    classes = resolve_classes(mode)
    X, y, names, counts = load_dataset(data_dir, classes=classes,
                                       limit_per_class=limit_per_class)
    if len(X) < 10:
        raise RuntimeError("训练样本不足（{} 张），请至少提供若干张图片".format(len(X)))
    fit = fit_weights(X, y, classes, epochs=epochs, lr=lr,
                      class_weights=class_weights, init_from=init_from,
                      progress=progress)
    weights = fit["weights"]
    weights["version"] = os.path.basename(out_dir)

    os.makedirs(out_dir, exist_ok=True)
    wp = os.path.join(out_dir, "weights.json")
    with open(wp, "w", encoding="utf-8") as fh:
        json.dump(weights, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump({"num_samples": len(X), "classes": classes, "class_counts": counts,
                   "init_from": init_from, "history": fit["history"]},
                  fh, ensure_ascii=False, indent=2)
    print("训练完成：{} 样本 → {}".format(len(X), wp))
    return wp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="runs/recap_latest")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--limit-per-class", type=int, default=0)
    ap.add_argument("--class-weights", type=str, default=None,
                    help="如 original:0.5,screen_recapture:1,print_recapture:1")
    ap.add_argument("--init-from", "--resume", dest="init_from", default=None,
                    help="从已有 weights.json 热启动（增量/多轮训练）")
    ap.add_argument("--mode", choices=list(mode_defs.MODES), default=None,
                    help="screen_only 或 screen_and_print；缺省为三类")
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
        print("提示：--backbone efficientb0 为可选 CNN 路线，需安装 torch+timm（见 requirements-dev.txt）。"
              "当前使用 feature_logistic 训练。")
    train(args.data, args.out, epochs=args.epochs, limit_per_class=args.limit_per_class,
          class_weights=cw, lr=args.lr, init_from=args.init_from, mode=args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
