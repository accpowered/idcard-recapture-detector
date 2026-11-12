"""分支 B + 融合：特征分类器与风险分级。

v1.1 支持按模式（仅屏幕 / 屏幕+打印）推理：模型类别由权重中的 classes 决定，
旧权重（无 classes）按三类处理；仅屏幕模式会屏蔽打印类并重新归一化融合分。
权重可由 tools/train_recapture.py 再训练覆盖。
"""
from __future__ import annotations

import json
import os
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from . import modes as mode_defs

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

CLASS_LABELS = mode_defs.ALL_CLASSES

DEFAULT_WEIGHTS: Dict = {
    "version": "1.0.0",
    "model_type": "feature_logistic",
    "classes": list(mode_defs.ALL_CLASSES),
    "recapture_mode": mode_defs.MODE_SCREEN_AND_PRINT,
    "feature_names": ["moire", "screen_grid", "bezel", "highlight", "texture_degradation",
                      "halftone", "paper_noise", "saturation_drop", "color_shift"],
    "temperature": 1.5,
    "logits": {
        "original": {"bias": 1.0, "weights": {}},
        "screen_recapture": {"bias": -1.5, "weights": {"moire": 3.0, "screen_grid": 2.0,
                                                        "bezel": 1.0, "highlight": 0.8,
                                                        "color_shift": 0.6}},
        "print_recapture": {"bias": -1.5, "weights": {"texture_degradation": 1.2, "halftone": 3.0,
                                                       "paper_noise": 0.8, "saturation_drop": 1.0}},
    },
    "signal_weights": {"moire": 0.30, "screen_grid": 0.10, "bezel": 0.15, "highlight": 0.10,
                       "texture_degradation": 0.10, "halftone": 0.25, "paper_noise": 0.10,
                       "saturation_drop": 0.05, "color_shift": 0.05},
    "fusion_weights": {"w_screen": 0.45, "w_print": 0.35, "w_signal": 0.20},
    "thresholds": {"low": 0.30, "high": 0.60},
}


class ModelModeError(ValueError):
    """模型类别与请求的识别模式不兼容。"""


class RecaptureClassifier:
    def __init__(self, weights_path: Optional[str] = None, thresholds: Optional[Dict] = None,
                 weights: Optional[Dict] = None):
        merged = self._load(weights_path)
        if isinstance(weights, dict):
            merged.update(json.loads(json.dumps(weights)))
        self.weights = merged
        if thresholds:
            self.weights["thresholds"] = thresholds
        classes = self.weights.get("classes") or CLASS_LABELS
        self.classes: Tuple[str, ...] = tuple(classes)
        self._onnx = None

    @staticmethod
    def _load(path: Optional[str]) -> Dict:
        w = json.loads(json.dumps(DEFAULT_WEIGHTS))
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                w.update(loaded)
            except Exception:  # noqa: BLE001
                pass
        return w

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        t = float(self.weights.get("temperature", 1.5))
        z = np.asarray(logits, dtype=np.float64) / max(t, 0.2)
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def _logit(self, cls: str, features: Dict[str, float]) -> float:
        spec = self.weights.get("logits", {}).get(cls, {"bias": 0.0, "weights": {}})
        v = float(spec.get("bias", 0.0))
        for f, c in spec.get("weights", {}).items():
            v += float(c) * float(features.get(f, 0.0))
        return v

    def classify(self, features: Dict[str, float], mode: Optional[str] = None) -> Dict:
        """输入信号特征向量，输出概率、翻拍分数与风险等级。

        mode=screen_only 时只计算 original / screen_recapture 两类，
        并把融合分按 w_screen + w_signal 重新归一化。
        """
        mode = mode_defs.normalize_mode(mode)
        allowed = mode_defs.allowed_classes(self.classes, mode)
        missing = sorted(set(mode_defs.classes_for_mode(mode)) - set(allowed))
        if missing:
            raise ModelModeError(
                "模型类别 {} 不满足模式 {}（缺少 {}）".format(
                    list(self.classes), mode, list(missing)))

        logits = [self._logit(c, features) for c in allowed]
        probs_arr = self._softmax(np.asarray(logits, dtype=np.float64))
        probs = {c: float(probs_arr[i]) for i, c in enumerate(allowed)}

        # 分支 A 信号聚合分
        sw = self.weights.get("signal_weights", {})
        signal_score = sum(float(sw.get(f, 0.0)) * float(features.get(f, 0.0)) for f in features)
        signal_score = float(min(1.0, signal_score))

        # 融合
        fw = self.weights.get("fusion_weights", {"w_screen": 0.45, "w_print": 0.35, "w_signal": 0.20})
        if mode == mode_defs.MODE_SCREEN_ONLY:
            w_screen = float(fw.get("w_screen", 0.45))
            w_signal = float(fw.get("w_signal", 0.20))
            denom = max(w_screen + w_signal, 1e-6)
            recapture_score = (w_screen * probs.get("screen_recapture", 0.0)
                               + w_signal * signal_score) / denom
        else:
            recapture_score = (
                float(fw.get("w_screen", 0.45)) * probs.get("screen_recapture", 0.0)
                + float(fw.get("w_print", 0.35)) * probs.get("print_recapture", 0.0)
                + float(fw.get("w_signal", 0.20)) * signal_score
            )
        recapture_score = float(min(1.0, max(0.0, recapture_score)))

        pred = max(allowed, key=lambda c: probs[c])
        th = self.weights.get("thresholds", {"low": 0.30, "high": 0.60})
        risk = "low"
        if recapture_score >= th.get("high", 0.60):
            risk = "high"
        elif recapture_score >= th.get("low", 0.30):
            risk = "medium"

        return {
            "score": round(recapture_score, 4),
            "risk": risk,
            "class": pred,
            "probs": {c: round(probs[c], 4) for c in allowed},
            "signal_score": round(signal_score, 4),
            "classes": list(allowed),
            "mode": mode,
        }
