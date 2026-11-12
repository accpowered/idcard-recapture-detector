"""模型版本管理：查看当前权重、指标、回滚。"""
from __future__ import annotations

import json
import os
import shutil
import time
from typing import Dict, List, Optional


class ModelManager:
    def __init__(self, root: str = "models/recapture"):
        self.root = root
        self.versions_dir = os.path.join(root, "versions")
        os.makedirs(self.versions_dir, exist_ok=True)

    def _weights_path(self) -> str:
        return os.path.join(self.root, "weights.json")

    def current(self) -> Dict:
        info = {"version": None, "weights_path": self._weights_path(), "metrics": None}
        vpath = os.path.join(self.root, "VERSION")
        if os.path.exists(vpath):
            with open(vpath, "r", encoding="utf-8") as f:
                info["version"] = f.read().strip()
        mpath = os.path.join(self.root, "METRICS.md")
        if os.path.exists(mpath):
            with open(mpath, "r", encoding="utf-8") as f:
                info["metrics"] = f.read()
        return info

    def list_versions(self) -> List[str]:
        if not os.path.isdir(self.versions_dir):
            return []
        return sorted(os.listdir(self.versions_dir), reverse=True)

    def rollback(self, version: str) -> bool:
        vdir = os.path.join(self.versions_dir, version)
        if not os.path.isdir(vdir):
            return False
        src = os.path.join(vdir, "weights.json")
        if not os.path.exists(src):
            return False
        # 备份当前
        self._backup_current()
        shutil.copy2(src, self._weights_path())
        with open(os.path.join(self.root, "VERSION"), "w", encoding="utf-8") as f:
            f.write(version)
        return True

    def _backup_current(self):
        wp = self._weights_path()
        if not os.path.exists(wp):
            return
        cur = self.current().get("version") or "unknown"
        stamp = time.strftime("%Y%m%d-%H%M%S")
        dst = os.path.join(self.versions_dir, f"{cur}-rollback-{stamp}")
        os.makedirs(dst, exist_ok=True)
        shutil.copy2(wp, os.path.join(dst, "weights.json"))
