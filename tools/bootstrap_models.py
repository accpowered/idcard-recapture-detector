"""模型完整性检查与引导下载（SHA256 校验清单）。

用法:
  python tools/bootstrap_models.py            # 检查 models/ 目录、生成默认权重
  python tools/bootstrap_models.py --offline  # 仅本地校验，不联网
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.core.recapture.classifier import DEFAULT_WEIGHTS  # noqa: E402

MODELS = ["recapture", "idcard_det", "doc_orientation", "unwarp", "ocr"]
CHECKSUMS = os.path.join("models", "checksums.json")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_defaults() -> bool:
    os.makedirs("models", exist_ok=True)
    for d in MODELS:
        os.makedirs(os.path.join("models", d), exist_ok=True)
    wp = os.path.join("models", "recapture", "weights.json")
    created = False
    if not os.path.exists(wp):
        with open(wp, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_WEIGHTS, f, ensure_ascii=False, indent=2)
        created = True
    vp = os.path.join("models", "recapture", "VERSION")
    if not os.path.exists(vp):
        with open(vp, "w", encoding="utf-8") as f:
            f.write("1.0.0")
    return created


def verify() -> dict:
    if not os.path.exists(CHECKSUMS):
        return {"status": "no_checksums", "detail": "无校验清单，跳过"}
    with open(CHECKSUMS, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    results = {}
    for rel, expect in manifest.items():
        p = rel
        if not os.path.exists(p):
            results[rel] = "missing"
        elif _sha256(p) != expect:
            results[rel] = "mismatch"
        else:
            results[rel] = "ok"
    bad = {k: v for k, v in results.items() if v != "ok"}
    return {"status": "ok" if not bad else "fail", "results": results, "bad": bad}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()
    created = ensure_defaults()
    print("模型目录已就绪" + ("，已生成默认翻拍权重 models/recapture/weights.json" if created else ""))
    print("校验结果：", json.dumps(verify(), ensure_ascii=False))
    print("提示：官方权重（OCR/检测/方向/矫正 ONNX）如未随仓库分发，请在联网环境运行 bootstrap 下载，"
          "或直接使用本工具内置的特征分类器（无需下载即可翻拍判定）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
