"""模型版本管理：版本元数据、启用/回退、训练晋升与运行时分类器缓存。"""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import threading
import time
from typing import Dict, List, Optional, Tuple

from ..core.config import ROOT
from ..core.recapture import modes as mode_defs
from ..core.recapture.classifier import DEFAULT_WEIGHTS, RecaptureClassifier

_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
BASELINE_VERSION = "1.0.0"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _atomic_write_text(path: str, text: str) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _atomic_copy(src: str, dst: str) -> None:
    directory = os.path.dirname(dst) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=directory)
    os.close(fd)
    try:
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _read_json(path: Optional[str], default=None):
    if not path:
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return default


class ModelManager:
    """管理 models/recapture 下的版本目录、当前指针与分类器实例缓存。"""

    def __init__(self, root: Optional[str] = None, keep_versions: int = 10):
        self.root = os.path.abspath(root or os.path.join(ROOT, "models/recapture"))
        self.versions_dir = os.path.join(self.root, "versions")
        os.makedirs(self.versions_dir, exist_ok=True)
        self.keep_versions = max(1, int(keep_versions or 10))
        self._lock = threading.RLock()
        self._cache: Dict[str, Tuple[Optional[str], float, RecaptureClassifier]] = {}
        self.warnings: List[str] = []

    # ------------------------------------------------------------------ 路径
    def weights_path(self, version: Optional[str] = None) -> Optional[str]:
        if version:
            if not self._safe_version(version):
                return None
            path = os.path.join(self.versions_dir, version, "weights.json")
            return path if os.path.isfile(path) else None
        path = os.path.join(self.root, "weights.json")
        return path if os.path.isfile(path) else None

    def _version_file(self) -> str:
        return os.path.join(self.root, "VERSION")

    def _read_pointer(self) -> Optional[str]:
        try:
            with open(self._version_file(), "r", encoding="utf-8") as f:
                value = f.read().strip()
            return value or None
        except OSError:
            return None

    def has_version(self, version: Optional[str]) -> bool:
        return bool(version) and self.weights_path(version) is not None

    def _safe_version(self, version: str) -> bool:
        if not version or not _VERSION_RE.match(version):
            return False
        target = os.path.realpath(os.path.join(self.versions_dir, version))
        base = os.path.realpath(self.versions_dir)
        return target == base or target.startswith(base + os.sep)

    # ------------------------------------------------------------------ 查询
    def current_version(self) -> Optional[str]:
        pointer = self._read_pointer()
        if pointer and self.has_version(pointer):
            return pointer
        return None

    def classes_of(self, version: Optional[str] = None) -> Tuple[str, ...]:
        path = self.weights_path(version)
        data = _read_json(path, {}) or {}
        classes = data.get("classes")
        if classes:
            return tuple(classes)
        if version is None and path:
            return mode_defs.ALL_CLASSES
        return mode_defs.ALL_CLASSES

    def _load_meta(self, version: str) -> Dict:
        vdir = os.path.join(self.versions_dir, version)
        meta = _read_json(os.path.join(vdir, "meta.json"))
        weights = _read_json(os.path.join(vdir, "weights.json"), {}) or {}
        classes = tuple(weights.get("classes") or mode_defs.ALL_CLASSES)
        mode = weights.get("recapture_mode") or mode_defs.derive_mode(classes)
        path = os.path.join(vdir, "weights.json")
        created = os.path.getmtime(path) if os.path.isfile(path) else time.time()
        base = {
            "version": version,
            "created_at": created,
            "source": "unknown",
            "label": None,
            "parent_version": None,
            "mode": mode,
            "classes": list(classes),
            "params": {},
            "dataset": None,
            "metrics": None,
            "thresholds": weights.get("thresholds"),
            "promoted": True,
            "gate": None,
            "note": "",
        }
        if isinstance(meta, dict):
            base.update(meta)
        base["version"] = version
        base["classes"] = list(base.get("classes") or classes)
        base["mode"] = base.get("mode") or mode
        return base

    def list_versions(self, mode: Optional[str] = None) -> List[Dict]:
        mode = mode_defs.normalize_mode(mode) if mode else None
        current = self.current_version()
        items: List[Dict] = []
        try:
            names = os.listdir(self.versions_dir)
        except OSError:
            names = []
        for name in names:
            vdir = os.path.join(self.versions_dir, name)
            if not os.path.isdir(vdir) or not _VERSION_RE.match(name):
                continue
            valid = os.path.isfile(os.path.join(vdir, "weights.json"))
            item = self._load_meta(name)
            item["active"] = name == current
            item["valid"] = valid
            item["compatible"] = bool(valid and (mode is None or
                                                 mode_defs.is_compatible(item["classes"], mode)))
            item["reason"] = None
            if not valid:
                item["reason"] = "权重文件缺失"
            elif mode and not item["compatible"]:
                item["reason"] = "该版本缺少 {} 所需的类别".format(mode_defs.mode_label(mode))
            items.append(item)
        items.sort(key=lambda x: (-(x.get("created_at") or 0.0), x["version"]))
        return items

    def current_info(self, mode: Optional[str] = None) -> Dict:
        mode = mode_defs.normalize_mode(mode)
        version = self.current_version()
        if version:
            meta = self._load_meta(version)
            classes = tuple(meta["classes"])
            return {
                "version": version,
                "mode": meta.get("mode") or mode_defs.derive_mode(classes),
                "classes": list(classes),
                "created_at": meta.get("created_at"),
                "source": meta.get("source"),
                "metrics": meta.get("metrics"),
                "thresholds": meta.get("thresholds"),
                "compatible": mode_defs.is_compatible(classes, mode),
                "available": True,
                "weights_path": self.weights_path(version),
                "pointer": self._read_pointer(),
            }
        path = self.weights_path(None)
        weights = _read_json(path, {}) or {}
        classes = tuple(weights.get("classes") or mode_defs.ALL_CLASSES)
        return {
            "version": self._read_pointer() or "unknown",
            "mode": weights.get("recapture_mode") or mode_defs.derive_mode(classes),
            "classes": list(classes),
            "created_at": None,
            "source": "root_weights" if path else "builtin_default",
            "metrics": None,
            "thresholds": weights.get("thresholds"),
            "compatible": mode_defs.is_compatible(classes, mode),
            "available": False,
            "weights_path": path,
            "pointer": self._read_pointer(),
        }

    # ------------------------------------------------------------------ 缓存
    def classifier(self, version: Optional[str] = None) -> RecaptureClassifier:
        path = self.weights_path(version)
        try:
            mtime = os.path.getmtime(path) if path else 0.0
        except OSError:
            mtime = 0.0
        key = version or "__root__"
        with self._lock:
            cached = self._cache.get(key)
            if cached and cached[0] == path and abs(cached[1] - mtime) < 1e-6:
                return cached[2]
            clf = RecaptureClassifier(weights_path=path)
            self._cache[key] = (path, mtime, clf)
            return clf

    def invalidate(self, version: Optional[str] = None) -> None:
        with self._lock:
            if version is None:
                self._cache.clear()
            else:
                self._cache.pop(version, None)
                self._cache.pop("__root__", None)

    # ------------------------------------------------------------------ 写入
    def new_version_id(self, label: Optional[str] = None) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", (label or "train").strip().lower()).strip("-")[:24]
        slug = slug or "train"
        base = "{}-{}".format(slug, time.strftime("%Y%m%d-%H%M%S"))
        candidate = base
        index = 2
        while os.path.exists(os.path.join(self.versions_dir, candidate)):
            candidate = "{}-{}".format(base, index)
            index += 1
        return candidate

    def promote(self, weights_path: str, meta: Dict, activate: bool = True) -> str:
        if not os.path.isfile(weights_path):
            raise FileNotFoundError("权重文件不存在: {}".format(weights_path))
        version = str(meta.get("version") or "")
        if not _VERSION_RE.match(version):
            raise ValueError("非法版本号: {!r}".format(version))
        data = _read_json(weights_path, None)
        if not isinstance(data, dict) or not data.get("logits"):
            raise ValueError("权重文件无效（缺少 logits）: {}".format(weights_path))
        vdir = os.path.join(self.versions_dir, version)
        os.makedirs(vdir, exist_ok=True)
        _atomic_copy(weights_path, os.path.join(vdir, "weights.json"))
        payload = dict(meta)
        payload["version"] = version
        _atomic_write_text(os.path.join(vdir, "meta.json"),
                           json.dumps(payload, ensure_ascii=False, indent=2))
        self.invalidate(version)
        if activate:
            if not self.activate(version):
                raise RuntimeError("启用新版本失败: {}".format(version))
        self.prune()
        return version

    def activate(self, version: str) -> bool:
        path = self.weights_path(version)
        if not path:
            return False
        data = _read_json(path, None)
        if not isinstance(data, dict) or not data.get("logits"):
            return False
        pointer_before = self._read_pointer()
        with self._lock:
            try:
                _atomic_copy(path, os.path.join(self.root, "weights.json"))
                _atomic_write_text(self._version_file(), version + "\n")
            except Exception:  # noqa: BLE001
                if pointer_before and self.has_version(pointer_before):
                    try:
                        _atomic_copy(self.weights_path(pointer_before),
                                     os.path.join(self.root, "weights.json"))
                    except OSError:
                        pass
                raise
            self._cache.clear()
        self._write_metrics(version)
        return True

    def _write_metrics(self, version: str) -> None:
        """把当前版本的指标写到 models/recapture/METRICS.md（尽力而为）。"""
        try:
            meta = self._load_meta(version)
        except Exception:  # noqa: BLE001
            return
        metrics = meta.get("metrics")
        if not metrics:
            return
        body = "# 翻拍分类器指标\n\n当前版本：{}\n\n模式：{}（{}）\n\n".format(
            version, meta.get("mode"), ", ".join(meta.get("classes") or []))
        body += "```json\n" + json.dumps(metrics, ensure_ascii=False, indent=2) + "\n```\n"
        _atomic_write_text(os.path.join(self.root, "METRICS.md"), body)

    def prune(self) -> List[str]:
        items = self.list_versions()
        current = self.current_version()
        protected = {BASELINE_VERSION}
        if current:
            protected.add(current)
        keep = set(protected)
        count = 0
        for item in items:
            name = item["version"]
            if name in protected:
                continue
            if count < self.keep_versions:
                keep.add(name)
                count += 1
        removed: List[str] = []
        for item in items:
            name = item["version"]
            if name in keep:
                continue
            path = os.path.join(self.versions_dir, name)
            if not self._safe_version(name):
                continue
            shutil.rmtree(path, ignore_errors=True)
            removed.append(name)
            self.invalidate(name)
        return removed
