"""分支 A：可解释信号分析（规则 + 经典图像处理，纯 numpy，无训练依赖）。

每项线索输出 [0,1] 归一化得分 + 中文证据描述。
线索借鉴 CVPR2024 CMA（色度伪影）、IEEE2025「摩尔纹 vs 底纹解耦」等文献。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..preprocess import to_gray


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _rgb_to_hsv(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = rgb[..., 0].astype(np.float32) / 255.0
    g = rgb[..., 1].astype(np.float32) / 255.0
    b = rgb[..., 2].astype(np.float32) / 255.0
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    d = mx - mn
    s = np.zeros_like(mx)
    s[mx > 0] = d[mx > 0] / (mx[mx > 0] + 1e-9)
    return mx, s, None


def _hann_window(shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    return np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)


# --------------------------------------------------------------------------- #
# 1. 摩尔纹（2D FFT 频谱峰值检测 + 底纹掩码）
# --------------------------------------------------------------------------- #
def _local_median(a: np.ndarray, k: int = 5) -> np.ndarray:
    from numpy.lib.stride_tricks import sliding_window_view
    pad = k // 2
    ap = np.pad(a, pad, mode="reflect")
    win = sliding_window_view(ap, (k, k))
    return np.median(win, axis=(2, 3))


def moire_score(gray: np.ndarray) -> Tuple[float, str]:
    h, w = gray.shape
    if min(h, w) < 64:
        return 0.0, "图像过小，跳过摩尔纹检测"
    g = gray.astype(np.float32)
    # 下采样到 ~360 最小边，加速频谱与中值滤波（周期按比例缩放后仍在频段内）
    if min(h, w) > 360:
        from PIL import Image
        s = 360.0 / min(h, w)
        g = np.asarray(Image.fromarray(g.astype(np.uint8)).resize(
            (max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR), dtype=np.float32)
    h, w = g.shape
    g = g - g.mean()
    win = _hann_window((h, w))
    F = np.fft.fftshift(np.fft.fft2(g * win))
    mag = np.log1p(np.abs(F))

    # 顶帽（减去 5x5 局部中值）：真实摩尔纹是“尖锐单峰”，
    # 内容边缘（文字/人像）只产生平滑宽带，底纹泄漏也只造成整体抬升，
    # 因此顶帽响应可把摩尔纹从内容/底纹中解耦出来。
    top = mag - _local_median(mag, 5)

    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(xx - cx, yy - cy)
    # 摩尔纹频段（周期约 8~60 像素）
    d = min(h, w)
    rmin = max(5, d // 60)
    rmax_b = max(rmin + 6, d // 8)
    band = (r >= rmin) & (r <= rmax_b)
    if not band.any():
        return 0.0, "未检测到有效摩尔纹频段"

    vals = top[band]
    peak = float(vals.max())
    # 经验归一化：内容边缘（文字/人像）峰值约 1.5-2，真实翻拍摩尔纹约 2.7+；
    # 该阈值在验证集上用 evaluate.py 重新校准。
    score = _clip01((peak - 1.2) / 2.2)

    detail = "未检测到明显摩尔纹"
    if score > 0.25:
        thresh = np.percentile(vals, 99.9)
        py, px = np.nonzero(band & (top >= thresh))
        if len(px) > 0:
            ang = float(np.degrees(np.arctan2(py[:200].mean() - cy, px[:200].mean() - cx))) % 180
            detail = f"检测到约 {ang:.0f}° 方向摩尔纹峰值"
    return score, detail


# --------------------------------------------------------------------------- #
# 2. 屏幕栅格 / 色度伪影（液晶像素阵列、色偏）
# --------------------------------------------------------------------------- #
def screen_grid_score(rgb: np.ndarray) -> Tuple[float, str]:
    h, w = rgb.shape[:2]
    if min(h, w) < 32:
        return 0.0, "图像过小"
    # 色度通道高频能量（子像素 RGB 排列）: 屏幕翻拍会残留栅格
    cr = rgb[..., 2].astype(np.float32) - rgb[..., 0].astype(np.float32)
    cb = rgb[..., 2].astype(np.float32) - rgb[..., 1].astype(np.float32)
    hf_cr = _high_freq_energy(cr)
    hf_cb = _high_freq_energy(cb)
    luma = to_gray(rgb)
    hf_luma = _high_freq_energy(luma)
    # 栅格在色度通道比亮度通道更显著
    ratio = (hf_cr + hf_cb) / (2.0 * hf_luma + 1e-6)
    score = _clip01((ratio - 0.3) / 0.8)
    detail = f"色度高频能量比 {ratio:.2f}"
    return score, detail


def _high_freq_energy(ch: np.ndarray) -> float:
    ch = ch.astype(np.float32)
    ch = ch - ch.mean()
    gy, gx = np.gradient(ch)
    return float(np.mean(gx ** 2 + gy ** 2))


# --------------------------------------------------------------------------- #
# 3. 屏幕边框 / 黑边（卡域四周的暗色均匀带）
# --------------------------------------------------------------------------- #
def bezel_score(full_gray: np.ndarray, quad: np.ndarray) -> Tuple[float, str]:
    """在完整图中，检查卡域四边形外侧是否围绕一圈暗色边框（翻拍屏幕常见）。"""
    if full_gray is None or quad is None:
        return 0.0, "无边框上下文"
    h, w = full_gray.shape
    # 生成卡域内掩码，采样其外侧紧邻环形带
    mask = _polygon_mask((h, w), quad)
    # 膨胀掩码得外环
    from ._morph import dilate
    outer = dilate(mask, radius=max(4, int(min(h, w) * 0.015)))
    ring = outer & (~mask)
    if ring.sum() < 100:
        return 0.0, "边框区域过小"
    ring_vals = full_gray[ring]
    inner_vals = full_gray[mask]
    ring_mean = float(np.median(ring_vals))
    inner_mean = float(np.median(inner_vals))
    # 屏幕黑边应接近纯黑且与卡面形成强对比；普通桌面深色背景（中灰）不计入
    near_black = _clip01((55.0 - ring_mean) / 55.0)
    contrast = _clip01((inner_mean - ring_mean) / 100.0)
    uniformity = _clip01(1.0 - float(np.std(ring_vals)) / 60.0)
    score = near_black * contrast * 0.7 + near_black * uniformity * 0.3
    detail = f"卡域外黑边（亮度 {ring_mean:.0f}，对比 {inner_mean - ring_mean:.0f}）"
    return score, detail


def _polygon_mask(shape: Tuple[int, int], quad: np.ndarray) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    if quad is None or len(quad) < 4:
        return mask
    pts = np.asarray(quad, dtype=np.int32).reshape(-1, 2)
    # 扫描线填充
    pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)]
    if len(pts) < 3:
        return mask
    xmin, xmax = pts[:, 0].min(), pts[:, 0].max()
    ymin, ymax = pts[:, 1].min(), pts[:, 1].max()
    xmin, xmax = max(0, xmin), min(w - 1, xmax)
    ymin, ymax = max(0, ymin), min(h - 1, ymax)
    xs = np.arange(xmin, xmax + 1)
    for y in range(ymin, ymax + 1):
        # 求多边形与水平线交点
        inter = []
        n = len(pts)
        for i in range(n):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % n]
            if (y1 <= y < y2) or (y2 <= y < y1):
                if y2 != y1:
                    x = x1 + (y - y1) * (x2 - x1) / float(y2 - y1)
                    inter.append(x)
        inter.sort()
        for j in range(0, len(inter) - 1, 2):
            lo, hi = int(inter[j]), int(inter[j + 1])
            lo, hi = max(0, lo), min(w - 1, hi)
            if lo <= hi:
                mask[y, lo:hi + 1] = True
    return mask


# --------------------------------------------------------------------------- #
# 4. 高光 / 镜面反射（玻璃屏幕）
# --------------------------------------------------------------------------- #
def highlight_score(rgb: np.ndarray) -> Tuple[float, str]:
    gray = to_gray(rgb)
    h, w = gray.shape
    if min(h, w) < 32:
        return 0.0, "图像过小"
    # 高亮度 + 低局部方差（镜面）→ 高光
    bright = gray > 235
    if bright.sum() < 20:
        return 0.0, "未检测到高光"
    # 局部方差（用 3x3 窗口差分近似）
    var = _local_variance(gray)
    specular = bright & (var < 80.0)
    frac = float(specular.sum()) / max(1, float(bright.sum()))
    area_frac = float(specular.sum()) / (h * w)
    # 高光应是局部斑块：若整片均匀过亮，说明是亮背景而非镜面反射
    if area_frac > 0.35:
        return 0.0, f"大面积均匀高亮（{area_frac * 100:.1f}%），非镜面高光"
    score = _clip01(frac * 0.7 + _clip01(area_frac / 0.03) * 0.3)
    detail = f"高光像素占比 {area_frac * 100:.1f}%"
    return score, detail


def _local_variance(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32)
    k = np.ones((3, 3), dtype=np.float32) / 9.0
    mean = _convolve2d(g, k)
    sq = _convolve2d(g * g, k)
    return np.clip(sq - mean * mean, 0, None)


def _convolve2d(a: np.ndarray, k: np.ndarray) -> np.ndarray:
    from numpy.lib.stride_tricks import sliding_window_view
    h, w = a.shape
    kh, kw = k.shape
    pad = ((kh // 2, kh // 2), (kw // 2, kw // 2))
    ap = np.pad(a, pad, mode="reflect")
    win = sliding_window_view(ap, (kh, kw))
    return np.tensordot(win, k, axes=([2, 3], [0, 1]))


# --------------------------------------------------------------------------- #
# 5. 纹理退化（高频能量 / 清晰度下降 → 打印翻拍）
# --------------------------------------------------------------------------- #
def texture_degradation_score(gray: np.ndarray) -> Tuple[float, str]:
    h, w = gray.shape
    if min(h, w) < 32:
        return 0.0, "图像过小"
    g = gray.astype(np.float32)
    lap = _laplacian(g)
    var = float(lap.var())
    # 归一化：以 856x540 高清原件为参照，方差越大越清晰
    # 经验映射：var > 4000 清晰；< 800 严重退化
    clarity = _clip01((var - 300.0) / 4000.0)
    score = 1.0 - clarity  # 退化程度
    detail = f"高频清晰度方差 {var:.0f}"
    return score, detail


def _laplacian(g: np.ndarray) -> np.ndarray:
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    return _convolve2d(g, k)


# --------------------------------------------------------------------------- #
# 6. 半色调网点（打印/复印）
# --------------------------------------------------------------------------- #
def halftone_score(gray: np.ndarray) -> Tuple[float, str]:
    h, w = gray.shape
    if min(h, w) < 64:
        return 0.0, "图像过小"
    g = gray.astype(np.float32)
    g = g - g.mean()
    F = np.fft.fftshift(np.fft.fft2(g * _hann_window((h, w))))
    mag = np.abs(F)
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(xx - cx, yy - cy)
    # 半色调网点对应中高频率的规则点阵：四向/斜向周期峰
    rmin, rmax = int(min(h, w) * 0.08), int(min(cy, cx) * 0.85)
    band = (r > rmin) & (r < rmax)
    vals = mag[band]
    if len(vals) < 100:
        return 0.0, "频域样本不足"
    # 多峰等距性：取 top 峰的角度分布离散度
    thresh = np.percentile(vals, 99.8)
    peaks = band & (mag >= thresh)
    py, px = np.nonzero(peaks)
    score = 0.0
    detail = "未检测到半色调网点"
    if len(px) >= 3:
        angs = np.degrees(np.arctan2(py - cy, px - cx)) % 180
        # 等距周期峰 → 角度聚类成规则网格
        hist, _ = np.histogram(angs, bins=np.arange(0, 181, 15))
        peak_count = int((hist > hist.max() * 0.4).sum())
        score = _clip01((peak_count - 1) / 4.0)
        if score > 0.25:
            detail = f"检测到规则网点频率（{len(px)} 个周期峰）"
    return score, detail


# --------------------------------------------------------------------------- #
# 7. 纸张噪声（低频亮度斑驳）
# --------------------------------------------------------------------------- #
def paper_noise_score(gray: np.ndarray) -> Tuple[float, str]:
    """纸张/复印的细颗粒噪声（在平坦区域上测量），避免把文字/人像等真实内容当成斑驳。"""
    h, w = gray.shape
    if min(h, w) < 64:
        return 0.0, "图像过小"
    g = gray.astype(np.float32)
    # 只统计平坦区域（局部梯度小），排除文字/边缘等内容
    gy, gx = np.gradient(g)
    grad = np.hypot(gx, gy)
    flat = grad < np.percentile(grad, 40)
    if flat.sum() < h * w * 0.08:
        return 0.0, "平坦区域过少，跳过纸噪检测"
    # 细颗粒噪声：与 3x3 均值的偏差，在平坦区域上的离散度
    k = np.ones((3, 3), np.float32) / 9.0
    smooth = _convolve2d(g, k)
    grain = (g - smooth)[flat]
    grain_std = float(np.std(grain))
    # 归一化：细颗粒 std 与图像整体 std 之比
    ratio = grain_std / (float(np.std(g)) + 1e-6)
    score = _clip01((ratio - 0.12) / 0.30)
    detail = f"平坦区细颗粒噪声比 {ratio:.2f}"
    return score, detail


# --------------------------------------------------------------------------- #
# 8. 色彩饱和度下降（打印/复印）
# --------------------------------------------------------------------------- #
def saturation_drop_score(rgb: np.ndarray) -> Tuple[float, str]:
    mx, s, _ = _rgb_to_hsv(rgb)
    mean_s = float(s.mean())
    # 原件通常保留色彩；打印/复印（尤其黑白）下降。参考值取 0.30，避免浅色卡误判。
    drop = 1.0 - _clip01(mean_s / 0.30)
    detail = f"平均饱和度 {mean_s:.2f}"
    return drop, detail


# --------------------------------------------------------------------------- #
# 9. 色偏（屏幕蓝移 / 打印暖黄）
# --------------------------------------------------------------------------- #
def color_shift_score(rgb: np.ndarray) -> Tuple[float, str]:
    r = rgb[..., 0].astype(np.float32).mean()
    g = rgb[..., 1].astype(np.float32).mean()
    b = rgb[..., 2].astype(np.float32).mean()
    total = r + g + b + 1e-6
    # 蓝偏程度
    blue_bias = (b - r) / total
    score = _clip01((blue_bias + 0.03) / 0.12)
    detail = f"蓝偏 {blue_bias:.3f}"
    return score, detail


# --------------------------------------------------------------------------- #
# 汇总
# --------------------------------------------------------------------------- #
SIGNAL_FUNCS = {
    "moire": moire_score,
    "screen_grid": screen_grid_score,
    "bezel": bezel_score,
    "highlight": highlight_score,
    "texture_degradation": texture_degradation_score,
    "halftone": halftone_score,
    "paper_noise": paper_noise_score,
    "saturation_drop": saturation_drop_score,
    "color_shift": color_shift_score,
}


def collect(card_rgb: np.ndarray, full_gray: Optional[np.ndarray] = None,
            quad: Optional[np.ndarray] = None) -> Dict:
    """计算全部信号线索，返回 evidences 列表与原始特征向量。"""
    gray = to_gray(card_rgb)
    evidences: List[Dict] = []
    features: Dict[str, float] = {}

    def _run(name: str, fn, *args):
        try:
            score, detail = fn(*args)
        except Exception as exc:  # noqa: BLE001
            score, detail = 0.0, f"计算失败: {exc}"
        score = _clip01(score)
        features[name] = score
        evidences.append({"name": name, "score": round(score, 4), "detail": detail})

    _run("moire", moire_score, gray)
    _run("screen_grid", screen_grid_score, card_rgb)
    _run("bezel", bezel_score, full_gray, quad)
    _run("highlight", highlight_score, card_rgb)
    _run("texture_degradation", texture_degradation_score, gray)
    _run("halftone", halftone_score, gray)
    _run("paper_noise", paper_noise_score, gray)
    _run("saturation_drop", saturation_drop_score, card_rgb)
    _run("color_shift", color_shift_score, card_rgb)

    return {"evidences": evidences, "features": features}
