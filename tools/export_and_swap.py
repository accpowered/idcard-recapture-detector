"""导出并热替换模型权重 / 回滚。

用法:
  python tools/export_and_swap.py --weights runs/recap_v1.1/weights.json   # 上线新权重
  python tools/export_and_swap.py --rollback v1.0.0                        # 回滚到某版本
  python tools/export_and_swap.py --list                                   # 查看版本
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.services.model_manager import ModelManager  # noqa: E402

ROOT = "models/recapture"


def swap(weights_path: str) -> int:
    if not os.path.exists(weights_path):
        print(f"权重不存在：{weights_path}")
        return 1
    mm = ModelManager(ROOT)
    cur = mm.current().get("version") or "unknown"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    ver = os.path.basename(os.path.dirname(weights_path)) or "v"
    new_ver = f"{ver}-{stamp}"
    os.makedirs(os.path.join(ROOT, "versions"), exist_ok=True)

    wp = os.path.join(ROOT, "weights.json")
    if os.path.exists(wp):
        bak = os.path.join(ROOT, "versions", cur)
        os.makedirs(bak, exist_ok=True)
        shutil.copy2(wp, os.path.join(bak, "weights.json"))

    shutil.copy2(weights_path, wp)
    with open(os.path.join(ROOT, "VERSION"), "w", encoding="utf-8") as f:
        f.write(new_ver)
    metrics_src = os.path.join(os.path.dirname(weights_path), "metrics.json")
    if os.path.exists(metrics_src):
        with open(metrics_src, "r", encoding="utf-8") as f:
            m = json.load(f)
        md = os.path.join(ROOT, "METRICS.md")
        body = json.dumps(m, ensure_ascii=False, indent=2)
        with open(md, "w", encoding="utf-8") as f:
            f.write("# 翻拍分类器指标\n\n版本：" + new_ver + "\n\n" + body + "\n")
    # 清理旧版本，仅保留最近 3 版
    mm = ModelManager(ROOT)
    vers = [v for v in mm.list_versions() if v != new_ver]
    for old in vers[2:]:
        shutil.rmtree(os.path.join(ROOT, "versions", old), ignore_errors=True)
    print(f"已上线新权重版本 {new_ver}（旧版已备份）")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None)
    ap.add_argument("--rollback", default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    mm = ModelManager(ROOT)
    if args.list:
        print("当前版本：", mm.current().get("version"))
        print("历史版本：", mm.list_versions())
        return 0
    if args.rollback:
        ok = mm.rollback(args.rollback)
        print("回滚成功" if ok else "回滚失败（版本不存在）")
        return 0 if ok else 1
    if args.weights:
        return swap(args.weights)
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
