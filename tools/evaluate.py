"""评估 + 阈值校准：输出 AUC/EER/Acc/F1/混淆矩阵，并按目标 FPR 校准阈值。

用法:
  python tools/evaluate.py --data data/recapture --weights runs/recap_v1.1/weights.json --target-fpr 0.05 --save-config
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.core.recapture.classifier import RecaptureClassifier  # noqa: E402
from tools.train_recapture import CLASSES, extract_features, load_dataset  # noqa: E402


def _roc_curve(y_true: np.ndarray, scores: np.ndarray):
    """返回 (fpr, tpr, thresholds)，按 scores 降序。"""
    order = np.argsort(-scores)
    y = y_true[order]
    s = scores[order]
    tpr, fpr = [], []
    tp = fp = 0
    P = int((y == 1).sum())
    N = int((y == 0).sum())
    for i in range(len(y)):
        if y[i] == 1:
            tp += 1
        else:
            fp += 1
        tpr.append(tp / P if P else 0.0)
        fpr.append(fp / N if N else 0.0)
    return np.asarray(fpr), np.asarray(tpr), s


def auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    return float(np.trapezoid(tpr, fpr))


def eer(fpr: np.ndarray, tpr: np.ndarray) -> float:
    fnr = 1.0 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def calibrate_threshold(scores: np.ndarray, y: np.ndarray, target_fpr: float) -> float:
    """按目标 FPR 找最低阈值（score 越大越可能是翻拍）。"""
    order = np.argsort(-scores)
    y_s = y[order]
    s_s = scores[order]
    N = int((y_s == 0).sum())
    if N == 0:
        return 0.5
    fp = 0
    best_t = 0.6
    best_fpr = 1.0
    for i in range(len(y_s)):
        if y_s[i] == 0:
            fp += 1
        if fp / N <= target_fpr:
            best_t = float(s_s[i])
            best_fpr = fp / N
        else:
            break
    return best_t


def evaluate(data_dir: str, weights_path: str | None = None, target_fpr: float = 0.05) -> Dict:
    X, y, names = load_dataset(data_dir)
    if len(X) == 0:
        raise RuntimeError("无样本")
    clf = RecaptureClassifier(weights_path=weights_path)
    probs_all, scores_all, preds = [], [], []
    for i in range(len(X)):
        feats = {CLASSES[0]: 0.0}
        feats = {n: float(X[i, j]) for j, n in enumerate(clf.weights["feature_names"])}
        r = clf.classify(feats)
        scores_all.append(r["score"])
        preds.append(CLASSES.index(r["class"]))
    preds = np.asarray(preds)
    scores_all = np.asarray(scores_all)

    acc = float((preds == y).mean())
    cm = np.zeros((3, 3), dtype=int)
    for a, b in zip(y, preds):
        cm[a, b] += 1

    per = {}
    for ci, cls in enumerate(CLASSES):
        tp = cm[ci, ci]
        fp = cm[:, ci].sum() - tp
        fn = cm[ci, :].sum() - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[cls] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}
    macro_f1 = float(np.mean([per[c]["f1"] for c in CLASSES]))

    y_bin = (y != 0).astype(int)
    fpr, tpr, _ = _roc_curve(y_bin, scores_all)
    auc_val = auc(fpr, tpr)
    eer_val = eer(fpr, tpr)
    th = calibrate_threshold(scores_all, y_bin, target_fpr)

    return {
        "num_samples": len(X), "accuracy": round(acc, 4), "macro_f1": round(macro_f1, 4),
        "per_class": per, "confusion_matrix": cm.tolist(),
        "auc_recapture": round(auc_val, 4), "eer": round(eer_val, 4),
        "threshold_high": round(th, 4), "target_fpr": target_fpr,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", default="models/recapture/weights.json")
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--save-config", action="store_true")
    args = ap.parse_args()

    res = evaluate(args.data, args.weights, args.target_fpr)
    print(json.dumps(res, ensure_ascii=False, indent=2))

    if args.save_config:
        wp = args.weights
        with open(wp, "r", encoding="utf-8") as f:
            w = json.load(f)
        w["thresholds"] = {"low": round(res["threshold_high"] * 0.5, 4), "high": res["threshold_high"]}
        with open(wp, "w", encoding="utf-8") as f:
            json.dump(w, f, ensure_ascii=False, indent=2)
        print(f"已按目标 FPR={args.target_fpr} 校准阈值并写入 {wp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
