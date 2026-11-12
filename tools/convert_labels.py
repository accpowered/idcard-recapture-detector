"""标注格式转换：LabelMe → 分类目录 / LabelMe → YOLO。

用法:
  python tools/convert_labels.py labelme --src data/labelme --out data/recapture
  python tools/convert_labels.py labelme2yolo --src data/labelme --out data/idcard_det
"""
from __future__ import annotations

import argparse
import json
import os
import shutil


def labelme_to_cls(src: str, out: str):
    os.makedirs(out, exist_ok=True)
    for root, _, files in os.walk(src):
        for f in files:
            if not f.endswith(".json"):
                continue
            jp = os.path.join(root, f)
            try:
                meta = json.load(open(jp, "r", encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                print(f"跳过 {jp}: {exc}")
                continue
            cls = None
            flags = meta.get("flags") or {}
            for cand in ("original", "screen_recapture", "print_recapture"):
                if cand in jp or str(flags).find(cand) >= 0:
                    cls = cand
                    break
            if cls not in ("original", "screen_recapture", "print_recapture"):
                print(f"跳过 {jp}：无法确定类别（请在文件名或 flags 中标注）")
                continue
            img = os.path.join(root, meta.get("imagePath", ""))
            if not os.path.exists(img):
                img = os.path.splitext(jp)[0] + ".jpg"
            if os.path.exists(img):
                dst = os.path.join(out, cls, os.path.basename(img))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(img, dst)
    print("转换完成 →", out)


def labelme_to_yolo(src: str, out: str):
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    os.makedirs(os.path.join(out, "labels"), exist_ok=True)
    cls_id = 0  # idcard
    for root, _, files in os.walk(src):
        for f in files:
            if not f.endswith(".json"):
                continue
            jp = os.path.join(root, f)
            try:
                meta = json.load(open(jp, "r", encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            img = os.path.join(root, meta.get("imagePath", ""))
            if not os.path.exists(img):
                continue
            W, H = meta.get("imageWidth", 1), meta.get("imageHeight", 1)
            shutil.copy2(img, os.path.join(out, "images", os.path.basename(img)))
            lines = []
            for shape in meta.get("shapes", []):
                if shape.get("label", "idcard") != "idcard":
                    continue
                pts = shape["points"]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                xc = sum(xs) / len(xs) / W
                yc = sum(ys) / len(ys) / H
                bw = (max(xs) - min(xs)) / W
                bh = (max(ys) - min(ys)) / H
                lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            lbl = os.path.splitext(os.path.basename(img))[0] + ".txt"
            with open(os.path.join(out, "labels", lbl), "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines))
    print("转换完成 →", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["labelme", "labelme2yolo"])
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.mode == "labelme":
        labelme_to_cls(args.src, args.out)
    else:
        labelme_to_yolo(args.src, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
