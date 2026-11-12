"""配置加载：settings.yaml / fields.yaml。"""
from __future__ import annotations

import os
from typing import Any, Dict

try:
    import yaml
    _HAS_YAML = True
except Exception:  # pragma: no cover
    yaml = None
    _HAS_YAML = False

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_yaml(path: str) -> Dict[str, Any]:
    if _HAS_YAML:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    # 无 PyYAML 时的极简兜底：仅返回空字典，调用方使用内置默认值
    return {}


def load_settings(path: str | None = None) -> Dict[str, Any]:
    p = path or os.path.join(ROOT, "config", "settings.yaml")
    if os.path.exists(p):
        return _load_yaml(p)
    return {}


def load_fields(path: str | None = None) -> Dict[str, Any]:
    p = path or os.path.join(ROOT, "config", "fields.yaml")
    if os.path.exists(p):
        return _load_yaml(p)
    return {}
