"""端到端流水线测试（合成身份证图）。"""
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402


def make_card_image(aspect=(856, 540), bg_color=(70, 80, 90)):
    """生成一张较逼真的身份证人像面：浅色底 + 彩色人像区 + 深色文字 + 细底纹。"""
    W, H = aspect
    pad = 120
    rng = np.random.default_rng(7)
    img = np.full((H + 2 * pad, W + 2 * pad, 3), bg_color, dtype=np.uint8)
    card = np.zeros((H, W, 3), dtype=np.float32)
    card[:] = (222, 232, 240)  # 浅蓝白底
    yy, xx = np.mgrid[0:H, 0:W]
    card += ((np.sin(xx / 0.9) + np.sin(yy / 0.8))[..., None] * 2.0)
    # 人像区（左下）：肤色 + 纹理
    py0, py1, px0, px1 = H - 300, H - 40, 60, 280
    card[py0:py1, px0:px1] = (200, 165, 135)
    card[py0:py1, px0:px1] += rng.normal(0, 10, (py1 - py0, px1 - px0, 1))
    # 文字行（深色）
    for i in range(12):
        y = 40 + i * 30
        x = 330
        card[y:y + 15, x:x + 380] = (35, 35, 40)
    # 标题与号码行
    card[30:60, 60:260] = (30, 30, 35)
    card[H - 70:H - 40, 60:620] = (30, 30, 35)
    card = np.clip(card, 0, 255)
    img[pad:pad + H, pad:pad + W] = card.astype(np.uint8)
    return img


def to_jpeg(img, quality=92):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def test_pipeline_detect_and_classify():
    from app.core.pipeline import Pipeline
    p = Pipeline()
    data = to_jpeg(make_card_image())
    res = p.analyze(data)
    assert res["code"] == 0, res.get("error")
    r = res["result"]
    assert len(r["card_bbox"]) == 4
    rc = r["recapture"]
    assert rc["class"] in ("original", "screen_recapture", "print_recapture")
    assert 0.0 <= rc["score"] <= 1.0
    assert rc["risk"] in ("low", "medium", "high")
    assert isinstance(rc["evidences"], list) and len(rc["evidences"]) == 9


def test_pipeline_screen_recapture_higher_than_original():
    from app.core.pipeline import Pipeline
    from tools.synthesize import add_moire, add_screen_grid
    p = Pipeline()
    base = make_card_image()
    screen = add_moire(base, 45, 0.05, 0.22)
    screen = add_screen_grid(screen, 4, 0.6)
    r0 = p.analyze(to_jpeg(base))
    r1 = p.analyze(to_jpeg(screen))
    assert r0["code"] == 0 and r1["code"] == 0
    e0 = {e["name"]: e["score"] for e in r0["result"]["recapture"]["evidences"]}
    e1 = {e["name"]: e["score"] for e in r1["result"]["recapture"]["evidences"]}
    assert e1["moire"] >= e0["moire"]
    assert r1["result"]["recapture"]["score"] >= r0["result"]["recapture"]["score"] - 0.02


def test_pipeline_reports_model_and_mode():
    from app.core.pipeline import Pipeline
    p = Pipeline()
    data = to_jpeg(make_card_image())
    res = p.analyze(data)
    assert res["code"] == 0, res.get("error")
    rc = res["result"]["recapture"]
    assert rc["model_version"] == "1.0.0"
    assert rc["mode"] == "screen_and_print"
    assert "print_recapture" in rc["probs"]

    explicit = p.analyze(data, model_version="1.0.0")
    assert explicit["code"] == 0
    assert explicit["result"]["recapture"]["model_version"] == "1.0.0"

    missing = p.analyze(data, model_version="version-does-not-exist")
    assert missing["code"] == 404


def test_pipeline_screen_only_masks_print():
    from app.core.pipeline import Pipeline
    p = Pipeline()
    res = p.analyze(to_jpeg(make_card_image()), mode="screen_only")
    assert res["code"] == 0, res.get("error")
    rc = res["result"]["recapture"]
    assert rc["mode"] == "screen_only"
    assert "print_recapture" not in rc["probs"]
    assert abs(sum(rc["probs"].values()) - 1.0) < 1e-3
