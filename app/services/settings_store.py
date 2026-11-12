"""运行时配置：settings.yaml 出厂默认 + settings.local.json 叠加层。

- 叠加层只保存被修改过的白名单键（JSON），无 PyYAML 时也能持久化。
- 对外只允许修改 ui.show_training、recapture.mode、training.keep_versions / target_fpr。
"""
from __future__ import annotations

import copy
import json
import os
import tempfile
import threading
from typing import Any, Dict, List, Optional

from ..core.config import ROOT, load_settings
from ..core.recapture import modes as mode_defs

DEFAULT_SETTINGS: Dict[str, Any] = {
    "ui": {"show_training": True},
    "recapture": {"mode": mode_defs.DEFAULT_MODE},
    "training": {
        "epochs": 200,
        "lr": 0.5,
        "target_fpr": 0.05,
        "keep_versions": 10,
        "min_macro_f1": 0.5,
    },
}

EDITABLE: Dict[str, Dict[str, type]] = {
    "ui": {"show_training": bool},
    "recapture": {"mode": str},
    "training": {"keep_versions": int, "target_fpr": float},
}


class SettingsError(ValueError):
    """非法的配置修改。"""


def _deep_copy(value):
    return copy.deepcopy(value)


def deep_merge(base: Dict, extra: Optional[Dict]) -> Dict:
    out = copy.deepcopy(base)
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _atomic_write_json(path: str, payload: Dict) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


class SettingsStore:
    def __init__(self, base_path: Optional[str] = None, override_path: Optional[str] = None):
        self.base_path = base_path or os.path.join(ROOT, "config/settings.yaml")
        self.override_path = override_path or os.path.join(ROOT, "config/settings.local.json")
        self._lock = threading.RLock()
        self.warnings: List[str] = []
        self._data: Dict[str, Any] = {}
        self.reload()

    # ------------------------------------------------------------------ 读取
    def _read_overlay(self) -> Dict:
        if not os.path.exists(self.override_path):
            return {}
        try:
            with open(self.override_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception as exc:  # noqa: BLE001
            self.warnings.append("settings.local.json 解析失败，已忽略：{}".format(exc))
            return {}

    def reload(self) -> Dict:
        base: Dict = {}
        try:
            base = load_settings(self.base_path) or {}
        except Exception as exc:  # noqa: BLE001
            self.warnings.append("settings.yaml 读取失败，使用内置默认值：{}".format(exc))
        data = deep_merge(DEFAULT_SETTINGS, base)
        data = deep_merge(data, self._read_overlay())
        with self._lock:
            self._data = data
        return self.get_all()

    def get_all(self) -> Dict:
        with self._lock:
            return _deep_copy(self._data)

    def get(self, dotted: str, default=None):
        node: Any = self._data
        for part in str(dotted).split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    @property
    def mode(self) -> str:
        return mode_defs.normalize_mode(self.get("recapture.mode"))

    def editable(self) -> Dict:
        return {
            "ui": {"show_training": bool(self.get("ui.show_training", True))},
            "recapture": {"mode": self.mode},
            "training": {
                "keep_versions": int(self.get("training.keep_versions", 10)),
                "target_fpr": float(self.get("training.target_fpr", 0.05)),
            },
        }

    def sources(self) -> Dict:
        return {
            "base": self.base_path,
            "base_exists": os.path.exists(self.base_path),
            "override": self.override_path,
            "override_exists": os.path.exists(self.override_path),
        }

    # ------------------------------------------------------------------ 写入
    @staticmethod
    def validate(patch: Dict) -> None:
        if not isinstance(patch, dict) or not patch:
            raise SettingsError("配置补丁必须是非空对象")
        for section, values in patch.items():
            if section not in EDITABLE:
                raise SettingsError("不可修改的配置项：{}".format(section))
            if not isinstance(values, dict):
                raise SettingsError("配置项 {} 必须是对象".format(section))
            for key, value in values.items():
                if key not in EDITABLE[section]:
                    raise SettingsError("不可修改的配置项：{}.{}".format(section, key))
                expected = EDITABLE[section][key]
                if expected is bool and not isinstance(value, bool):
                    raise SettingsError("{}.{} 必须是布尔值".format(section, key))
                if expected is int and (isinstance(value, bool) or not isinstance(value, int)):
                    raise SettingsError("{}.{} 必须是整数".format(section, key))
                if expected is float and not isinstance(value, (int, float)):
                    raise SettingsError("{}.{} 必须是数字".format(section, key))
                if expected is str and not isinstance(value, str):
                    raise SettingsError("{}.{} 必须是字符串".format(section, key))
        mode = patch.get("recapture", {}).get("mode")
        if mode is not None and mode not in mode_defs.MODES:
            raise SettingsError("未知的识别模式：{}（可选 {}）".format(
                mode, ", ".join(mode_defs.MODES)))
        keep = patch.get("training", {}).get("keep_versions")
        if keep is not None and not (1 <= int(keep) <= 50):
            raise SettingsError("training.keep_versions 需在 1–50 之间")
        fpr = patch.get("training", {}).get("target_fpr")
        if fpr is not None and not (0.01 <= float(fpr) <= 0.5):
            raise SettingsError("training.target_fpr 需在 0.01–0.5 之间")

    def update(self, patch: Dict) -> Dict:
        self.validate(patch)
        with self._lock:
            overlay = self._read_overlay()
            merged = deep_merge(overlay, patch)
            _atomic_write_json(self.override_path, merged)
            self._data = deep_merge(deep_merge(DEFAULT_SETTINGS, load_settings(self.base_path) or {}),
                                    merged)
            return self.get_all()
