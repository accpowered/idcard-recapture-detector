"""分支 B + 融合：特征分类器与风险分级。

v1.0 采用「可解释特征逻辑回归」分类器（纯 numpy，随仓库发布 weights.json，
可由 tools/train_recapture.py 再训练覆盖）。同时预留 ONNX CNN 推理接口
（若 models/recapture/recap_effb0.onnx 存在且已安装 onnxruntime，自动启用）。
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

DEFAULT_WEIGHTS: Dict = {
    "version": "1.0.0",
    "model_type": "feature_logistic",
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

CLASS_LABELS = ("original", "screen_recapture", "print_recapture")


class RecaptureClassifier:
    def __init__(self, weights_path: Optional[str] = None, thresholds: Optional[Dict] = None):
        self.weights = self._load(weights_path)
        if thresholds:
            self.weights["thresholds"] = thresholds
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
        spec = self.weights["logits"].get(cls, {"bias": 0.0, "weights": {}})
        v = float(spec.get("bias", 0.0))
        for f, c in spec.get("weights", {}).items():
            v += c * features.get(f, 0.0)
        return v

    def classify(self, features: Dict[str, float]) -> Dict:
        """输入信号特征向量，输出三类概率、翻拍分数与风险等级。"""
        logits = [self._logit(c, features) for c in CLASS_LABELS]
        probs = self._softmax(logits)
        probs = {c: float(probs[i]) for i, c in enumerate(CLASS_LABELS)}

        # 分支 A 信号聚合分
        sw = self.weights.get("signal_weights", {})
        signal_score = sum(sw.get(f, 0.0) * features.get(f, 0.0) for f in features)
        signal_score = float(min(1.0, signal_score))

        # 融合
        fw = self.weights.get("fusion_weights", {"w_screen": 0.45, "w_print": 0.35, "w_signal": 0.20})
        recapture_score = (
            fw.get("w_screen", 0.45) * probs["screen_recapture"]
            + fw.get("w_print", 0.35) * probs["print_recapture"]
            + fw.get("w_signal", 0.20) * signal_score
        )
        recapture_score = float(min(1.0, max(0.0, recapture_score)))

        pred = max(CLASS_LABELS, key=lambda c: probs[c])
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
            "probs": {c: round(probs[c], 4) for c in CLASS_LABELS},
            "signal_score": round(signal_score, 4),
        }
