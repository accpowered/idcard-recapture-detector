"""导出并热替换模型权重 / 回滚。

用法:
  python tools/export_and_swap.py --weights runs/recap_v1.1/weights.json   # 上线新权重
  python tools/export_and_swap.py --rollback 1.0.0                          # 回滚到某版本
  python tools/export_and_swap.py --list                                    # 查看版本
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.core.recapture import modes as mode_defs  # noqa: E402
from app.services.model_manager import ModelManager  # noqa: E402


def _load_metrics(weights_path: str):
    path = os.path.join(os.path.dirname(os.path.abspath(weights_path)), "metrics.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def swap(weights_path: str, label: str | None = None, mode: str | None = None,
         activate: bool = True, keep: int = 10) -> int:
    if not os.path.exists(weights_path):
        print("权重不存在：{}".format(weights_path))
        return 1
    try:
        with open(weights_path, "r", encoding="utf-8") as f:
            weights = json.load(f)
    except Exception as exc:  # noqa: BLE001
        print("权重解析失败：{}".format(exc))
        return 1
    if not isinstance(weights, dict) or not weights.get("logits"):
        print("权重文件无效（缺少 logits）")
        return 1

    classes = list(weights.get("classes") or mode_defs.ALL_CLASSES)
    mode = mode_defs.normalize_mode(mode or weights.get("recapture_mode")
                                    or mode_defs.derive_mode(classes))
    if not mode_defs.is_compatible(classes, mode):
        print("警告：权重类别 {} 不满足模式 {}，启用后识别会被拦截".format(classes, mode))

    mm = ModelManager(keep_versions=keep)
    default_label = os.path.basename(os.path.dirname(os.path.abspath(weights_path)))[:24] or "import"
    version = mm.new_version_id(label or default_label)
    meta = {
        "version": version,
        "created_at": time.time(),
        "source": "cli_import",
        "label": label,
        "parent_version": mm.current_version(),
        "mode": mode,
        "classes": classes,
        "params": {},
        "dataset": None,
        "metrics": _load_metrics(weights_path),
        "thresholds": weights.get("thresholds"),
        "promoted": activate,
        "gate": "passed",
        "note": "export_and_swap",
    }
    mm.promote(weights_path, meta, activate=activate)
    print("已{}版本 {}（类别 {}，模式 {}）".format(
        "上线" if activate else "导入（未启用）", version, classes, mode))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None)
    ap.add_argument("--label", default=None, help="版本标签，最终版本号为 <label>-<时间戳>")
    ap.add_argument("--mode", choices=list(mode_defs.MODES), default=None)
    ap.add_argument("--keep", type=int, default=10, help="保留的版本数量")
    ap.add_argument("--no-activate", action="store_true", help="只导入为候选，不切换当前版本")
    ap.add_argument("--rollback", default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    mm = ModelManager(keep_versions=args.keep)
    if args.list:
        info = mm.current_info()
        print("当前版本：", info.get("version"), "模式：", info.get("mode"), "类别：", info.get("classes"))
        for item in mm.list_versions():
            print(" - {:<32} mode={:<16} active={} metrics={}".format(
                item["version"], item.get("mode"), item.get("active"),
                bool(item.get("metrics"))))
        return 0
    if args.rollback:
        ok = mm.activate(args.rollback)
        print("回滚成功" if ok else "回滚失败（版本不存在或权重无效）")
        return 0 if ok else 1
    if args.weights:
        return swap(args.weights, label=args.label, mode=args.mode,
                    activate=not args.no_activate, keep=args.keep)
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
