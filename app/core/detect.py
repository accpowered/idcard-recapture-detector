"""卡证定位与透视矫正。

优先使用 OpenCV；未安装时回退到纯 numpy/Pillow 实现。
输出：旋转框四角（有序 TL,TR,BR,BL）+ 透视矫正后的 856x540 卡面图。
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except Exception:  # pragma: no cover
    cv2 = None
    _HAS_CV2 = False

from .preprocess import to_gray

CARD_W, CARD_H = 856, 540  # 二代身份证宽高比 85.6:54 ≈ 1.585:1


def _order_points(pts: np.ndarray) -> np.ndarray:
    """把 4 个点按 TL, TR, BR, BL 顺序排列。"""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = pts[:, 1] - pts[:, 0]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _quad_area(quad: np.ndarray) -> float:
    """四点包围框面积（shoelace）。"""
    x = quad[:, 0]
    y = quad[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def detect_card_quad(rgb: np.ndarray) -> Optional[np.ndarray]:
    """检测身份证四角。返回 (4,2) float32，或 None 表示未检出。"""
    if _HAS_CV2:
        return _detect_cv2(rgb)
    return _detect_numpy(rgb)


def _detect_cv2(rgb: np.ndarray) -> Optional[np.ndarray]:
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # 大图先下采样加速，坐标再映射回原图
    scale = 1.0
    long_side = max(h, w)
    if long_side > 1000:
        scale = 1000.0 / long_side
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 140)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    image_area = float(gray.shape[0] * gray.shape[1])
    best = None
    for cnt in contours[:12]:
        area = cv2.contourArea(cnt)
        if area < 0.03 * image_area:  # 太小，不是卡
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quad = approx.reshape(4, 2).astype(np.float32)
            best = quad
            break
        # 矩形退化为最小外接矩形
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        if box is not None and len(box) == 4:
            best = box.astype(np.float32)
            break

    if best is None:
        return None

    if scale != 1.0:
        best = best / scale
    return _order_points(best)


def _detect_numpy(rgb: np.ndarray) -> Optional[np.ndarray]:
    """纯 numpy 兜底：Sobel 边缘 + Hough 直线 → 四点。"""
    gray = to_gray(rgb)
    h, w = gray.shape
    scale = 1.0
    if max(h, w) > 800:
        scale = 800.0 / max(h, w)
        from PIL import Image
        gray = np.asarray(Image.fromarray(gray.astype(np.uint8)).resize(
            (int(w * scale), int(h * scale)), Image.BILINEAR), dtype=np.float32)

    gy, gx = np.gradient(gray)
    mag = np.hypot(gx, gy)
    mag = mag / (mag.max() + 1e-9)
    edge = mag > 0.12
    ys, xs = np.nonzero(edge)
    if len(xs) < 50:
        return None
    # 采样最多 8000 个边缘点参与 Hough
    if len(xs) > 8000:
        idx = np.random.default_rng(0).choice(len(xs), 8000, replace=False)
        xs, ys = xs[idx], ys[idx]

    thetas = np.deg2rad(np.arange(0, 180, 1.0))
    cos_t, sin_t = np.cos(thetas), np.sin(thetas)
    # rho = x cos + y sin
    rhos = np.round(xs[None, :] * cos_t[:, None] + ys[None, :] * sin_t[:, None]).astype(np.int32)
    rho_min, rho_max = rhos.min(), rhos.max()
    lines = []
    for i in range(len(thetas)):
        vals = rhos[i]
        lo, hi = vals.min(), vals.max()
        if hi - lo < 5:
            continue
        hist, bins = np.histogram(vals, bins=np.arange(lo, hi + 2))
        if hist.max() < 15:
            continue
        rho = bins[np.argmax(hist)]
        lines.append((thetas[i], float(rho), int(hist.max())))
    if len(lines) < 4:
        return None

    # 按角度聚类成 4 组，每组取最强线
    lines = sorted(lines, key=lambda x: -x[2])
    groups = []
    for theta, rho, votes in lines:
        placed = False
        for g in groups:
            if abs(_angle_diff(theta, g[0])) < np.deg2rad(20):
                g[1].append((theta, rho, votes))
                placed = True
                break
        if not placed:
            groups.append([theta, [(theta, rho, votes)]])
        if len(groups) == 4:
            break
    if len(groups) < 4:
        return None

    chosen = []
    for theta0, members in groups:
        members.sort(key=lambda x: -x[2])
        chosen.append(members[0])
    # 两两求交，取面积最大的凸四边
    quad = None
    best_area = 0.0
    hh, ww = gray.shape
    from itertools import combinations
    for pair in combinations(chosen, 4):
        pts = []
        for (a1, r1, _), (a2, r2, _) in combinations(pair, 2):
            p = _line_intersect(a1, r1, a2, r2)
            if p is not None:
                pts.append(p)
        if len(pts) < 4:
            continue
        pts = np.array(pts, dtype=np.float32)
        o = _order_points(pts)
        area = _quad_area(o)
        if 0.01 * hh * ww < area < 0.95 * hh * ww and area > best_area:
            best_area = area
            quad = o
    if quad is None:
        return None
    if scale != 1.0:
        quad = quad / scale
    return _order_points(quad)


def _angle_diff(a: float, b: float) -> float:
    d = abs(a - b) % np.pi
    return min(d, np.pi - d)


def _line_intersect(t1: float, r1: float, t2: float, r2: float) -> Optional[Tuple[float, float]]:
    A = np.array([[np.cos(t1), np.sin(t1)], [np.cos(t2), np.sin(t2)]], dtype=np.float64)
    b = np.array([r1, r2], dtype=np.float64)
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    if abs(det) < 1e-6:
        return None
    x, y = np.linalg.solve(A, b)
    return float(x), float(y)


def warp_card(rgb: np.ndarray, quad: np.ndarray, size: Tuple[int, int] = (CARD_W, CARD_H)) -> np.ndarray:
    """把卡域四点透视矫正为 856x540 基准图。"""
    w, h = size
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    src = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    if _HAS_CV2:
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(rgb, M, (w, h), flags=cv2.INTER_LINEAR)
    # Pillow 兜底
    from PIL import Image
    H = _homography(src, dst)
    img = Image.fromarray(rgb)
    coeffs = H.ravel()[:8].tolist()
    warped = img.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BILINEAR)
    return np.asarray(warped, dtype=np.uint8)


def _homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    A = np.asarray(A, dtype=np.float64)
    _, _, V = np.linalg.svd(A)
    H = V[-1].reshape(3, 3)
    return H / H[2, 2]
