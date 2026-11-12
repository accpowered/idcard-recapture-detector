"""推理流水线编排：归一化 → 卡证定位 → 翻拍判定 → OCR → 字段解析。

v1.1 支持按指定模型版本与全局识别模式推理；分类器实例由 ModelManager 缓存，
训练/启用新版本后无需重启即可生效。
"""
from __future__ import annotations

import base64
import io
import time
from typing import Dict, Optional

import numpy as np

from ..services.model_manager import ModelManager
from . import detect
from .config import load_settings
from .fields import parse_fields
from .preprocess import decode_to_rgb, normalize, to_gray
from .recapture import modes as mode_defs
from .recapture.classifier import ModelModeError
from .recapture.signals import collect as collect_signals


class Pipeline:
    def __init__(self, settings: Optional[Dict] = None, settings_store=None,
                 model_manager: Optional[ModelManager] = None):
        self.settings = settings if settings is not None else load_settings()
        self.settings_store = settings_store
        self.model_manager = model_manager or ModelManager()
        self.ocr = None  # 延迟加载

    # ------------------------------------------------------------------ 模型
    def _default_mode(self) -> str:
        if self.settings_store is not None:
            try:
                return self.settings_store.mode
            except Exception:  # noqa: BLE001
                pass
        return mode_defs.normalize_mode((self.settings.get("recapture") or {}).get("mode"))

    def invalidate_cache(self, version: Optional[str] = None) -> None:
        self.model_manager.invalidate(version)

    # ------------------------------------------------------------------ OCR
    def _ensure_ocr(self):
        if self.ocr is None:
            from .ocr_engine import OCREngine
            self.ocr = OCREngine()
        return self.ocr

    def analyze(self, data: bytes, model_version: Optional[str] = None,
                mode: Optional[str] = None) -> Dict:
        t0 = time.time()
        pipe_cfg = self.settings.get("pipeline", {})
        mode = mode_defs.normalize_mode(mode or self._default_mode())

        # 0. 选择模型版本
        requested = (model_version or "").strip()
        if requested in ("", "auto", "current"):
            version = self.model_manager.current_version()
        else:
            version = requested
            if not self.model_manager.has_version(version):
                return {"code": 404, "error": "模型版本不存在: {}".format(version)}
        classes = self.model_manager.classes_of(version)
        if not mode_defs.is_compatible(classes, mode):
            return {"code": 409,
                    "error": "模型 {} 缺少{}所需类别；请切换识别模式或在模型页训练/选择兼容版本".format(
                        version or "默认权重", mode_defs.mode_label(mode))}

        # 1. 输入归一化
        try:
            rgb = decode_to_rgb(data)
        except Exception as exc:  # noqa: BLE001
            return {"code": 400, "error": str(exc)}
        rgb = normalize(rgb, int(pipe_cfg.get("max_long_side", 1600)))
        h, w = rgb.shape[:2]

        # 2. 卡证定位
        quad = detect.detect_card_quad(rgb)
        if quad is None:
            return {
                "code": 404, "error": "未检测到身份证",
                "meta": {"width": w, "height": h, "processing_ms": int((time.time() - t0) * 1000)},
            }
        card_w, card_h = detect.CARD_W, detect.CARD_H
        card = detect.warp_card(rgb, quad, (card_w, card_h))

        # 3. 翻拍判定（双分支）
        full_gray = to_gray(rgb)
        sig = collect_signals(card, full_gray, quad)
        try:
            classifier = self.model_manager.classifier(version)
            recapture = classifier.classify(sig["features"], mode=mode)
        except ModelModeError as exc:
            return {"code": 409, "error": str(exc)}
        recapture["evidences"] = sig["evidences"]
        recapture["model_version"] = version or "default"
        recapture["mode"] = mode

        # 4. OCR（可选，未安装时降级）
        ocr_available = False
        ocr_lines: list = []
        if pipe_cfg.get("ocr", {}).get("enabled", True):
            eng = self._ensure_ocr()
            ocr_available = eng.available
            if eng.available:
                res = eng.recognize(card)
                if res:
                    # 按 y 坐标从上到下、x 从左到右排序，模拟阅读顺序
                    res_sorted = sorted(res, key=lambda r: (np.mean([p[1] for p in r[0]]), np.mean([p[0] for p in r[0]])))
                    ocr_lines = [(text, score) for _, text, score in res_sorted]

        # 5. 字段解析与校验
        mask = bool(self.settings.get("privacy", {}).get("mask_fields", True))
        text_lines = [t for t, _ in ocr_lines]
        fields = parse_fields(text_lines, mask=mask) if ocr_lines else {
            "name": "", "gender": "", "ethnicity": "", "birth": "", "address": "",
            "id_number": "", "authority": "", "valid_period": "",
            "id_checksum_valid": None, "gender_consistent": None, "birth_consistent": None,
            "confidence": "low", "raw_lines": [],
        }

        # 6. 结果可视化（标注卡框）
        annotated_b64 = self._annotate(rgb, quad)

        return {
            "code": 0,
            "result": {
                "card_bbox": [[round(float(x), 1), round(float(y), 1)] for x, y in quad],
                "recapture": recapture,
                "fields": fields,
                "confidence": {
                    "ocr": round(float(np.mean([s for _, s in ocr_lines])), 4) if ocr_lines else None,
                    "recapture": round(1.0 - abs(float(recapture["score"]) - (0.0 if recapture["risk"] == "low" else 1.0)), 4),
                },
                "ocr": {"available": ocr_available, "lines": [{"text": t, "score": round(s, 4)} for t, s in ocr_lines]},
                "annotated_b64": annotated_b64,
                "meta": {"width": w, "height": h, "processing_ms": int((time.time() - t0) * 1000)},
            },
        }

    @staticmethod
    def _annotate(rgb: np.ndarray, quad: np.ndarray) -> str:
        try:
            import cv2
            img = rgb.copy()
            pts = np.asarray(quad, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], True, (0, 255, 0), max(2, img.shape[0] // 300))
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            if ok:
                return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
        except Exception:  # noqa: BLE001
            pass
        try:
            from PIL import Image, ImageDraw
            img = Image.fromarray(rgb)
            dr = ImageDraw.Draw(img)
            dr.polygon([(float(x), float(y)) for x, y in quad], outline=(0, 255, 0), width=max(2, rgb.shape[0] // 300))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=88)
            return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:  # noqa: BLE001
            return ""
