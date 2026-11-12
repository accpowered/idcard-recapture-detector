"""OCR 引擎封装：RapidOCR（PP-OCRv5/v6 中文），优雅降级。

兼容 rapidocr 1.x/2.x（返回 (result, elapse)）与 3.x（返回 RapidOCROutput 对象）。
未安装 rapidocr 时 available=False，流水线自动跳过并标注状态。
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


class OCREngine:
    def __init__(self):
        self._engine = None
        self.available = False
        self.error: str = ""
        self._load()

    def _load(self):
        try:
            from rapidocr import RapidOCR
            self._engine = RapidOCR()
            self.available = True
        except Exception as exc:  # noqa: BLE001
            self._engine = None
            self.available = False
            self.error = str(exc)

    @staticmethod
    def _normalize(out) -> List[Tuple[list, str, float]]:
        items: List[Tuple[list, str, float]] = []
        # rapidocr 3.x: RapidOCROutput 对象（无文字时 txts/boxes 可能为 None）
        if hasattr(out, "txts") and hasattr(out, "scores"):
            txts = out.txts or ()
            scores = out.scores or ()
            boxes = getattr(out, "boxes", None)
            if boxes is None:
                boxes = [None] * len(txts)
            for box, text, score in zip(boxes, txts, scores):
                items.append((box, str(text), float(score)))
            return items
        # rapidocr 1.x/2.x: (result, elapse)
        result = out
        if isinstance(out, (tuple, list)) and len(out) == 2 and isinstance(out[1], (list, tuple, float, int)):
            result = out[0]
        if not result:
            return items
        for item in result:
            try:
                box = item[0]
                text = str(item[1])
                score = float(item[2]) if len(item) > 2 else 0.0
                items.append((box, text, score))
            except Exception:  # noqa: BLE001
                continue
        return items

    def recognize(self, rgb: np.ndarray) -> Optional[List[Tuple[list, str, float]]]:
        """返回 [(box, text, score), ...]，失败/未安装返回 None。"""
        if not self.available:
            return None
        img = rgb
        try:
            import cv2
            img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception:  # noqa: BLE001
            pass
        try:
            out = self._engine(img)
        except Exception as exc:  # noqa: BLE001
            self.error = str(exc)
            return None
        return self._normalize(out)
