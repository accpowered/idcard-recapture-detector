"""输入归一化：解码、EXIF 方向纠正、缩放。

仅依赖 numpy + Pillow；HEIC 在安装 pillow-heif 后自动支持。
"""
from __future__ import annotations

import io
from typing import Optional

import numpy as np

try:
    from PIL import Image, ImageOps
    _HAS_PIL = True
except Exception:  # pragma: no cover
    _HAS_PIL = False


class DecodeError(ValueError):
    """图片解码失败。"""


def decode_to_rgb(data: bytes) -> np.ndarray:
    """把任意图片字节解码为 RGB uint8 数组 (H, W, 3)。

    自动处理 EXIF 方向。支持 JPG/PNG/BMP/WebP/GIF/TIFF；
    安装 pillow-heif 后可支持 HEIC。
    """
    if not _HAS_PIL:
        raise DecodeError("缺少 Pillow，无法解码图片")
    if not data:
        raise DecodeError("空图片数据")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as exc:  # noqa: BLE001
        raise DecodeError(f"图片解码失败: {exc}") from exc
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:  # noqa: BLE001
        pass
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def normalize(arr: np.ndarray, max_long_side: int = 1600) -> np.ndarray:
    """长边缩放至上限，保持宽高比，避免超大图拖慢推理。"""
    h, w = arr.shape[:2]
    long_side = max(h, w)
    if long_side <= max_long_side:
        return arr
    scale = max_long_side / float(long_side)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    img = Image.fromarray(arr)
    img = img.resize((nw, nh), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def to_gray(rgb: np.ndarray) -> np.ndarray:
    """RGB → 灰度 float32，范围 [0, 255]。"""
    if rgb.ndim == 2:
        return rgb.astype(np.float32)
    r, g, b = rgb[..., 0].astype(np.float32), rgb[..., 1].astype(np.float32), rgb[..., 2].astype(np.float32)
    return 0.299 * r + 0.587 * g + 0.114 * b


def estimate_noise_level(gray: np.ndarray) -> float:
    """估计噪声水平（用于打印/纸张纹理线索）。"""
    gy, gx = np.gradient(gray.astype(np.float32))
    lap = np.gradient(gx, axis=1)[1] + np.gradient(gy, axis=0)[0]
    return float(np.std(lap))
