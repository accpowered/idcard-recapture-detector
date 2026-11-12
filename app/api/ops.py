"""FastAPI 与标准库兜底服务共用的业务层（路由适配只做参数解析）。"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple

from .. import __version__
from ..core.config import ROOT
from ..core.preprocess import decode_to_rgb
from ..core.recapture import modes as mode_defs
from ..services.settings_store import SettingsError
from ..services.training import (IMAGE_EXTS, MAX_UPLOAD_BYTES, MAX_UPLOAD_FILES,
                                 TrainingBusy, TrainingError, dataset_report,
                                 resolve_data_dir)

DATA_ROOT = os.path.join(ROOT, "data")


class ApiError(Exception):
    """带 HTTP 语义的业务错误。"""

    def __init__(self, code: int, message: str, payload: Optional[Dict] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.payload = payload or {}


def _ok(**fields) -> Dict:
    payload: Dict = {"code": 0}
    payload.update(fields)
    return payload


def _hash_index_path(directory: str) -> str:
    return os.path.join(directory, ".upload_index.json")


def _load_hash_index(directory: str) -> set:
    try:
        with open(_hash_index_path(directory), "r", encoding="utf-8") as f:
            data = json.load(f)
        return set(data) if isinstance(data, list) else set()
    except Exception:  # noqa: BLE001
        return set()


def _save_hash_index(directory: str, hashes: set) -> None:
    try:
        with open(_hash_index_path(directory), "w", encoding="utf-8") as f:
            json.dump(sorted(hashes), f, ensure_ascii=False, indent=1)
    except OSError:
        pass


class ApiOps:
    def __init__(self, ctx):
        self.ctx = ctx

    @property
    def mode(self) -> str:
        return self.ctx.settings_store.mode

    # ------------------------------------------------------------------ 健康
    def health(self) -> Dict:
        ocr_installed = importlib.util.find_spec("rapidocr") is not None
        info = self.ctx.model_manager.current_info(self.mode)
        return _ok(
            status="ok",
            ocr_installed=ocr_installed,
            version=__version__,
            recapture={"mode": self.mode, "mode_label": mode_defs.mode_label(self.mode)},
            model={"version": info["version"], "mode": info["mode"],
                   "classes": info["classes"], "compatible": info["compatible"]},
            training={"enabled": bool(self.ctx.settings_store.get("ui.show_training", True)),
                      "running": self.ctx.training.is_busy()},
        )

    # ------------------------------------------------------------------ 识别
    def analyze(self, data: bytes, file_name: str = "upload.jpg",
                model_version: Optional[str] = None) -> Dict:
        return self.ctx.analyze(data, file_name, model_version=model_version)

    def _check_version(self, model_version: Optional[str]) -> Optional[str]:
        version = (model_version or "").strip()
        if version in ("", "auto", "current"):
            return None
        if not self.ctx.model_manager.has_version(version):
            raise ApiError(404, "模型版本不存在: {}".format(version))
        classes = self.ctx.model_manager.classes_of(version)
        if not mode_defs.is_compatible(classes, self.mode):
            raise ApiError(409, "版本 {} 缺少{}所需类别，请切换识别模式或选择其他版本".format(
                version, mode_defs.mode_label(self.mode)))
        return version

    def batch(self, items: List[tuple], model_version: Optional[str] = None) -> Dict:
        version = self._check_version(model_version)
        task_id = self.ctx.batch(items, model_version=version)
        return _ok(task_id=task_id)

    def batch_status(self, task_id: str) -> Dict:
        status = self.ctx.batch_status(task_id)
        if not status or status.get("status") == "not_found":
            raise ApiError(404, "批量任务不存在: {}".format(task_id))
        return _ok(**status)

    # ------------------------------------------------------------------ 模型
    def models(self) -> Dict:
        manager = self.ctx.model_manager
        info = manager.current_info(self.mode)
        return _ok(
            current=info,
            versions=manager.list_versions(self.mode),
            modes={"current": self.mode,
                   "labels": dict(mode_defs.MODE_LABELS)},
            warnings=list(manager.warnings),
        )

    def activate(self, version: Optional[str]) -> Dict:
        version = (version or "").strip()
        manager = self.ctx.model_manager
        if not version or not manager.has_version(version):
            raise ApiError(404, "模型版本不存在: {}".format(version or "(空)"))
        classes = manager.classes_of(version)
        if not mode_defs.is_compatible(classes, self.mode):
            raise ApiError(409, "版本 {} 缺少{}所需类别，请先切换识别模式".format(
                version, mode_defs.mode_label(self.mode)))
        try:
            ok = manager.activate(version)
        except Exception as exc:  # noqa: BLE001
            raise ApiError(500, "启用版本失败: {}".format(exc))
        if not ok:
            raise ApiError(500, "启用版本失败（权重无效）: {}".format(version))
        return _ok(current=manager.current_info(self.mode))

    # ------------------------------------------------------------------ 设置
    def settings_get(self) -> Dict:
        store = self.ctx.settings_store
        info = self.ctx.model_manager.current_info(self.mode)
        return _ok(
            editable=store.editable(),
            effective=store.get_all(),
            sources=store.sources(),
            model={"current_version": info["version"], "compatible": info["compatible"],
                   "mode": self.mode},
            warnings=list(store.warnings),
        )

    def settings_save(self, patch: Dict) -> Dict:
        store = self.ctx.settings_store
        if not isinstance(patch, dict):
            raise ApiError(400, "配置补丁必须是对象")
        try:
            store.validate(patch)
        except SettingsError as exc:
            raise ApiError(400, str(exc))
        old_mode = store.mode
        new_mode = (patch.get("recapture") or {}).get("mode") or old_mode
        new_mode = mode_defs.normalize_mode(new_mode)
        try:
            store.update(patch)
        except SettingsError as exc:
            raise ApiError(400, str(exc))
        except Exception as exc:  # noqa: BLE001
            raise ApiError(500, "保存配置失败: {}".format(exc))

        switched_to = None
        warnings: List[str] = []
        if new_mode != old_mode:
            manager = self.ctx.model_manager
            info = manager.current_info(new_mode)
            if not info["compatible"]:
                candidates = [v for v in manager.list_versions(new_mode)
                              if v.get("valid") and v.get("compatible")]
                if candidates:
                    target = candidates[0]["version"]
                    try:
                        manager.activate(target)
                        switched_to = target
                    except Exception as exc:  # noqa: BLE001
                        try:
                            store.update({"recapture": {"mode": old_mode}})
                        except Exception:  # noqa: BLE001
                            pass
                        raise ApiError(500, "切换到兼容模型版本失败: {}".format(exc))
                else:
                    warnings.append(
                        "当前没有与「{}」兼容的模型版本，识别请求会返回 409；"
                        "请先在模型页训练或选择兼容版本".format(mode_defs.mode_label(new_mode)))
        return _ok(settings=store.get_all(), editable=store.editable(),
                   switched_to=switched_to, warnings=warnings)

    # ------------------------------------------------------------------ 数据集
    def dataset(self, data_dir: Optional[str] = None, mode: Optional[str] = None) -> Dict:
        try:
            path = resolve_data_dir(data_dir, root=DATA_ROOT, strict=True)
        except TrainingError as exc:
            raise ApiError(exc.code, exc.message)
        return _ok(**dataset_report(path, mode or self.mode))

    def upload_samples(self, class_name: Optional[str], files: List[tuple]) -> Dict:
        if not class_name:
            raise ApiError(400, "缺少 class 字段")
        allowed = mode_defs.classes_for_mode(self.mode)
        if class_name not in allowed:
            raise ApiError(400, "当前识别模式「{}」不接受类别 {}（可选：{}）".format(
                mode_defs.mode_label(self.mode), class_name, ", ".join(allowed)))
        items = [item for item in (files or []) if isinstance(item, tuple)]
        if not items:
            raise ApiError(400, "没有收到上传文件")
        if len(items) > MAX_UPLOAD_FILES:
            raise ApiError(413, "单次最多上传 {} 张图片".format(MAX_UPLOAD_FILES))
        data_dir = resolve_data_dir(None, root=DATA_ROOT, strict=True)
        target = os.path.join(data_dir, class_name)
        os.makedirs(target, exist_ok=True)
        hashes = _load_hash_index(target)
        saved: List[str] = []
        skipped: List[Dict[str, str]] = []
        stamp = int(time.time() * 1000)
        for name, data in items:
            if len(data) > MAX_UPLOAD_BYTES:
                skipped.append({"name": name, "reason": "超过 12MB"})
                continue
            ext = os.path.splitext(name or "")[1].lower()
            if ext not in IMAGE_EXTS:
                skipped.append({"name": name, "reason": "不支持的扩展名"})
                continue
            try:
                decode_to_rgb(data)
            except Exception as exc:  # noqa: BLE001
                skipped.append({"name": name, "reason": "图片解码失败: {}".format(exc)})
                continue
            digest = hashlib.sha256(data).hexdigest()
            if digest in hashes:
                skipped.append({"name": name, "reason": "重复样本（内容哈希已存在）"})
                continue
            stem = re.sub(r"[^A-Za-z0-9._-]", "_",
                          os.path.splitext(os.path.basename(name or "sample"))[0])[:40] or "sample"
            dest_name = "{}_{}{}".format(stamp + len(saved), stem, ext)
            dest = os.path.join(target, dest_name)
            index = 2
            while os.path.exists(dest):
                dest_name = "{}_{}-{}{}".format(stamp + len(saved), stem, index, ext)
                dest = os.path.join(target, dest_name)
                index += 1
            try:
                with open(dest, "wb") as f:
                    f.write(data)
            except OSError as exc:
                skipped.append({"name": name, "reason": "写入失败: {}".format(exc)})
                continue
            hashes.add(digest)
            saved.append(dest_name)
        _save_hash_index(target, hashes)
        report = dataset_report(data_dir, self.mode)
        return _ok(saved=saved, skipped=skipped, dataset=report)

    def delete_sample(self, class_name: Optional[str], name: Optional[str]) -> Dict:
        if class_name not in mode_defs.ALL_CLASSES:
            raise ApiError(400, "未知类别: {}".format(class_name))
        name = (name or "").strip()
        if not name or os.path.basename(name) != name:
            raise ApiError(400, "非法文件名")
        if os.path.splitext(name)[1].lower() not in IMAGE_EXTS:
            raise ApiError(400, "只能删除图片样本")
        data_dir = resolve_data_dir(None, root=DATA_ROOT, strict=True)
        class_dir = os.path.join(data_dir, class_name)
        path = os.path.join(class_dir, name)
        real = os.path.realpath(path)
        base = os.path.realpath(class_dir)
        if not real.startswith(base + os.sep) or not os.path.isfile(real):
            raise ApiError(404, "样本不存在: {}".format(name))
        try:
            with open(real, "rb") as f:
                digest = hashlib.sha256(f.read()).hexdigest()
            os.remove(real)
            hashes = _load_hash_index(class_dir)
            hashes.discard(digest)
            _save_hash_index(class_dir, hashes)
        except OSError as exc:
            raise ApiError(500, "删除失败: {}".format(exc))
        report = dataset_report(data_dir, self.mode)
        return _ok(deleted=name, dataset=report)

    # ------------------------------------------------------------------ 训练
    def train_start(self, params: Dict) -> Dict:
        if not isinstance(params, dict):
            raise ApiError(400, "训练参数必须是对象")
        try:
            data_dir = resolve_data_dir(params.get("data_dir"), root=DATA_ROOT, strict=True)
            if not os.path.isdir(data_dir):
                raise ApiError(400, "训练数据目录不存在: {}".format(
                    params.get("data_dir") or data_dir))
            job = self.ctx.training.start(dict(params, data_dir=data_dir, mode=self.mode))
        except TrainingBusy as exc:
            raise ApiError(409, exc.message)
        except TrainingError as exc:
            raise ApiError(exc.code, exc.message, {"issues": exc.issues})
        return _ok(job_id=job["id"], version=job["version"],
                   job=self.ctx.training.status(job["id"]))

    def train_status(self, job_id: str) -> Dict:
        status = self.ctx.training.status(job_id)
        if status.get("status") == "not_found":
            raise ApiError(404, "训练任务不存在: {}".format(job_id))
        return _ok(**status)
