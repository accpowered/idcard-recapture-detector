"""Web 训练闭环：数据集体检、素材管理、训练 → 评估 → 自动上线。

本模块只做编排，特征提取/训练/评估复用 tools/ 下的函数（CLI 与 Web 同源）。
"""
from __future__ import annotations

import json
import os
import threading
import time
import traceback
import uuid
from typing import Dict, List, Optional

from ..core.config import ROOT
from ..core.recapture import modes as mode_defs

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MIN_PER_CLASS = 5
MIN_TOTAL = 10
RECOMMENDED_PER_CLASS = 50
MAX_UPLOAD_FILES = 100
MAX_UPLOAD_BYTES = 12 * 1024 * 1024


class TrainingError(Exception):
    def __init__(self, message: str, issues: Optional[list] = None, code: int = 400):
        super().__init__(message)
        self.message = message
        self.issues = issues or []
        self.code = code


class TrainingBusy(TrainingError):
    def __init__(self):
        super().__init__("已有训练任务在进行，请等待完成", code=409)


def resolve_data_dir(data_dir: Optional[str] = None, root: Optional[str] = None,
                     strict: bool = True) -> str:
    """把训练数据目录解析为绝对路径；strict 时要求位于 data/ 之下。"""
    root_abs = os.path.abspath(root or os.path.join(ROOT, "data"))
    if not data_dir:
        return os.path.join(root_abs, "recapture")
    path = os.path.abspath(data_dir if os.path.isabs(data_dir)
                           else os.path.join(ROOT, data_dir))
    if strict:
        real = os.path.realpath(path)
        base = os.path.realpath(root_abs)
        if real != base and not real.startswith(base + os.sep):
            raise TrainingError("训练数据目录必须位于 {} 之下".format(root_abs), code=400)
    return path


def _list_images(directory: str) -> List[os.DirEntry]:
    if not os.path.isdir(directory):
        return []
    try:
        entries = [e for e in os.scandir(directory)
                   if e.is_file() and e.name.lower().endswith(IMAGE_EXTS)]
    except OSError:
        return []
    try:
        entries.sort(key=lambda e: e.stat().st_mtime, reverse=True)
    except OSError:
        pass
    return entries


def _rel(path: str) -> str:
    try:
        rel = os.path.relpath(path, ROOT)
    except ValueError:
        return path
    return rel.replace(os.sep, "/") if not rel.startswith("..") else path


def dataset_report(data_dir: str, mode: Optional[str] = None) -> Dict:
    mode = mode_defs.normalize_mode(mode)
    classes = list(mode_defs.classes_for_mode(mode))
    required = list(mode_defs.required_classes(mode))
    optional = list(mode_defs.optional_classes(mode))

    counts: Dict[str, int] = {}
    recent: Dict[str, List[str]] = {}
    for cls in classes:
        entries = _list_images(os.path.join(data_dir, cls))
        counts[cls] = len(entries)
        recent[cls] = [e.name for e in entries[:10]]

    total = sum(counts.values())
    issues: List[Dict[str, str]] = []

    def add(level: str, code: str, message: str) -> None:
        issues.append({"level": level, "code": code, "message": message})

    if total < MIN_TOTAL:
        add("error", "too_few",
            "有效训练样本共 {} 张，至少需要 {} 张（original 与 screen_recapture 各 ≥{} 张）".format(
                total, MIN_TOTAL, MIN_PER_CLASS))
    for cls in required:
        count = counts.get(cls, 0)
        if count == 0:
            if cls == "original":
                message = ("缺少 original（原件实拍）样本：请上传任意身份证原件的实拍照片，"
                           "不必与翻拍样本是同一张卡")
            else:
                message = "缺少 screen_recapture（屏幕翻拍）样本：请上传对屏幕显示再拍摄的图片"
            add("error", "missing_class", message)
        elif count < MIN_PER_CLASS:
            add("error", "low_class",
                "{}（{}）仅 {} 张，至少需要 {} 张".format(
                    cls, mode_defs.label_of(cls), count, MIN_PER_CLASS))
    for cls in optional:
        if counts.get(cls, 0) == 0:
            add("warn", "optional_missing",
                "未提供 print_recapture（打印翻拍）样本：打印件将无法被正确识别；"
                "可补充样本，或把识别模式切到「仅屏幕翻拍」")
    present = [counts[c] for c in required if counts.get(c, 0) > 0]
    if len(present) >= 2 and min(present) > 0 and max(present) > 3 * min(present):
        add("warn", "imbalanced",
            "必备类别样本不均衡（最多/最少 = {:.1f}），建议补齐较少的一类".format(
                max(present) / min(present)))

    class_rows = []
    for cls in classes:
        count = counts.get(cls, 0)
        if count == 0:
            status = "missing"
        elif count < MIN_PER_CLASS:
            status = "low"
        elif count >= RECOMMENDED_PER_CLASS:
            status = "good"
        else:
            status = "ok"
        class_rows.append({
            "name": cls,
            "label": mode_defs.label_of(cls),
            "count": count,
            "min_required": MIN_PER_CLASS,
            "recommended": RECOMMENDED_PER_CLASS,
            "status": status,
            "recent_files": recent.get(cls, []),
        })

    guidance = [
        "最少：original 与 screen_recapture 各 ≥{} 张、合计 ≥{} 张，缺一不可。".format(
            MIN_PER_CLASS, MIN_TOTAL),
        "建议：每类 ≥{} 张；正式使用 200–500 张，覆盖不同机型/光照/角度。".format(
            RECOMMENDED_PER_CLASS),
        "同一张卡分别拍原件与屏幕翻拍（成对样本）更利于模型学到翻拍伪影差异。",
        "可把「复核」页确认的样本自动并入 data/recapture，也可在本页按类别上传素材。",
    ]
    if mode == mode_defs.MODE_SCREEN_AND_PRINT:
        guidance.append("当前模式为「屏幕+打印翻拍」：补充 print_recapture 后模型才能识别打印件。")
    else:
        guidance.append("当前模式为「仅屏幕翻拍」：print_recapture 样本不会参与训练。")

    ready = not any(item["level"] == "error" for item in issues)
    return {
        "mode": mode,
        "mode_label": mode_defs.mode_label(mode),
        "data_dir": data_dir,
        "data_dir_display": _rel(data_dir),
        "total": total,
        "ready": ready,
        "required_classes": required,
        "optional_classes": optional,
        "classes": class_rows,
        "issues": issues,
        "guidance": guidance,
    }


class TrainingRunner:
    def __init__(self, model_manager, settings_store=None,
                 data_root: Optional[str] = None, runs_root: Optional[str] = None):
        self.model_manager = model_manager
        self.settings_store = settings_store
        self.data_root = os.path.abspath(data_root or os.path.join(ROOT, "data"))
        self.runs_root = os.path.abspath(runs_root or os.path.join(ROOT, "runs"))
        self.jobs: Dict[str, Dict] = {}
        self._lock = threading.RLock()
        self._active_id: Optional[str] = None

    # ------------------------------------------------------------------ 配置
    @property
    def mode(self) -> str:
        if self.settings_store is not None:
            try:
                return self.settings_store.mode
            except Exception:  # noqa: BLE001
                pass
        return mode_defs.DEFAULT_MODE

    def _setting(self, key: str, default):
        if self.settings_store is not None:
            try:
                value = self.settings_store.get(key, default)
                return default if value is None else value
            except Exception:  # noqa: BLE001
                return default
        return default

    # ------------------------------------------------------------------ 任务
    def is_busy(self) -> bool:
        with self._lock:
            job = self.jobs.get(self._active_id) if self._active_id else None
            return bool(job and job.get("status") in ("queued", "running"))

    def status(self, job_id: str) -> Dict:
        job = self.jobs.get(job_id)
        if not job:
            return {"status": "not_found"}
        with self._lock:
            snapshot = dict(job)
            snapshot["logs"] = list(job.get("logs") or [])
            snapshot["params"] = dict(job.get("params") or {})
        return snapshot

    def _validate_params(self, params: Dict) -> Dict:
        def as_int(key, default, lo, hi):
            try:
                value = int(params.get(key, default))
            except (TypeError, ValueError):
                raise TrainingError("参数 {} 必须是整数".format(key))
            if not (lo <= value <= hi):
                raise TrainingError("参数 {} 需在 {}–{} 之间".format(key, lo, hi))
            return value

        def as_float(key, default, lo, hi):
            try:
                value = float(params.get(key, default))
            except (TypeError, ValueError):
                raise TrainingError("参数 {} 必须是数字".format(key))
            if not (lo <= value <= hi):
                raise TrainingError("参数 {} 需在 {}–{} 之间".format(key, lo, hi))
            return value

        label = params.get("label") or None
        if label is not None:
            label = str(label).strip()
            if len(label) > 40:
                raise TrainingError("版本标签过长（最多 40 字符）")
        return {
            "epochs": as_int("epochs", self._setting("training.epochs", 200), 10, 2000),
            "lr": as_float("lr", self._setting("training.lr", 0.5), 0.01, 2.0),
            "limit_per_class": as_int("limit_per_class", 0, 0, 10000),
            "target_fpr": as_float("target_fpr", self._setting("training.target_fpr", 0.05), 0.01, 0.5),
            "label": label,
            "init_from_current": bool(params.get("init_from_current", False)),
            "auto_activate": bool(params.get("auto_activate", True)),
        }

    def start(self, params: Dict) -> Dict:
        with self._lock:
            if self.is_busy():
                raise TrainingBusy()
            mode = mode_defs.normalize_mode(params.get("mode") or self.mode)
            data_dir = resolve_data_dir(params.get("data_dir"), root=self.data_root, strict=False)
            report = dataset_report(data_dir, mode)
            if not report["ready"]:
                raise TrainingError("训练数据不满足要求", issues=report["issues"], code=400)
            validated = self._validate_params(params)
            version = self.model_manager.new_version_id(validated.get("label"))
            job = {
                "id": uuid.uuid4().hex,
                "status": "running",
                "stage": "dataset",
                "progress": 0,
                "logs": [],
                "error": None,
                "issues": [],
                "params": dict(validated, data_dir=data_dir, mode=mode,
                               data_dir_display=report["data_dir_display"]),
                "result": None,
                "started_at": time.time(),
                "finished_at": None,
                "version": version,
            }
            self.jobs[job["id"]] = job
            self._active_id = job["id"]
            self._log(job, "数据集体检通过：共 {} 张，模式 {}".format(
                report["total"], mode_defs.mode_label(mode)))
            thread = threading.Thread(target=self._run, args=(job, report), daemon=True)
            thread.start()
            return job

    # ------------------------------------------------------------------ 执行
    @staticmethod
    def _log(job: Dict, message: str) -> None:
        logs = job.setdefault("logs", [])
        logs.append("[{}] {}".format(time.strftime("%H:%M:%S"), message))
        if len(logs) > 200:
            del logs[: len(logs) - 200]

    def _set_stage(self, job: Dict, stage: str, progress: float, message: Optional[str] = None) -> None:
        job["stage"] = stage
        job["progress"] = max(0, min(100, int(round(progress))))
        if message:
            self._log(job, message)

    def gate(self, metrics: Dict, mode: str, parent_version: Optional[str],
             min_f1: Optional[float] = None) -> Optional[str]:
        """质量门：返回 None 表示通过，否则返回未通过原因代码。"""
        min_f1 = float(self._setting("training.min_macro_f1", 0.5) if min_f1 is None else min_f1)
        if float(metrics.get("macro_f1", 0.0)) < min_f1:
            return "below_min_f1"
        for cls in mode_defs.required_classes(mode):
            recall = float(((metrics.get("per_class") or {}).get(cls) or {}).get("recall", 0.0))
            if recall <= 0:
                return "missing_class_recall"
        if self.model_manager.current_version() != parent_version:
            return "current_changed"
        if self.mode != mode:
            return "mode_changed"
        return None

    def _run(self, job: Dict, report: Dict) -> None:
        params = job["params"]
        mode = params["mode"]
        classes = list(mode_defs.classes_for_mode(mode))
        parent = self.model_manager.current_version()
        try:
            from tools.evaluate import evaluate_from_features
            from tools.train_recapture import fit_weights, load_dataset

            self._set_stage(job, "extract", 5, "开始提取特征（{} 类）…".format(len(classes)))
            total_hint = max(1, report["total"])

            def on_extract(done, total):
                total = total or total_hint
                job["progress"] = 5 + int(40 * done / max(1, total))
                if done == total or (done and done % 25 == 0):
                    self._log(job, "特征提取 {}/{}".format(done, total))

            X, y, names, counts = load_dataset(
                params["data_dir"], classes=classes,
                limit_per_class=params["limit_per_class"], progress=on_extract)
            if len(X) < MIN_TOTAL:
                raise TrainingError(
                    "有效样本不足（成功提取 {} 张，至少需要 {} 张）".format(len(X), MIN_TOTAL))
            present = {classes[int(i)] for i in set(y.tolist())} if len(y) else set()
            missing_required = [c for c in mode_defs.required_classes(mode) if c not in present]
            if missing_required:
                raise TrainingError("缺少必备类别样本：{}".format(", ".join(missing_required)))
            self._log(job, "特征提取完成：{} 张（{}）".format(
                len(X), ", ".join("{}={}".format(k, v) for k, v in counts.items())))

            init_from = None
            if params["init_from_current"] and parent:
                candidate = self.model_manager.weights_path(parent)
                if candidate:
                    try:
                        with open(candidate, "r", encoding="utf-8") as f:
                            stored = json.load(f).get("classes") or []
                    except Exception:  # noqa: BLE001
                        stored = []
                    if set(classes).issubset(set(stored)):
                        init_from = candidate
                        self._log(job, "从当前版本 {} 热启动".format(parent))
                    else:
                        self._log(job, "当前版本类别与训练模式不匹配，改为从零训练")

            self._set_stage(job, "train", 45, "开始训练：epochs={} lr={}".format(
                params["epochs"], params["lr"]))

            def on_train(epoch, total):
                total = total or params["epochs"]
                job["progress"] = 45 + int(30 * epoch / max(1, total))

            fit = fit_weights(X, y, classes, epochs=params["epochs"], lr=params["lr"],
                              init_from=init_from, progress=on_train, verbose=False)
            weights = fit["weights"]
            self._log(job, "训练完成，开始评估与阈值校准…")
            self._set_stage(job, "evaluate", 75)

            metrics = evaluate_from_features(X, y, weights, classes,
                                             target_fpr=params["target_fpr"], mode=mode)
            high = float(metrics["threshold_high"])
            weights["thresholds"] = {"low": round(high * 0.5, 4), "high": round(high, 4)}
            weights["version"] = job["version"]
            weights["classes"] = classes
            weights["recapture_mode"] = mode
            self._log(job, "评估：Acc={:.4f} macro-F1={:.4f} AUC={:.4f}".format(
                metrics["accuracy"], metrics["macro_f1"], metrics.get("auc_recapture", 0.0)))

            run_dir = os.path.join(self.runs_root, job["version"])
            os.makedirs(run_dir, exist_ok=True)
            weights_path = os.path.join(run_dir, "weights.json")
            with open(weights_path, "w", encoding="utf-8") as f:
                json.dump(weights, f, ensure_ascii=False, indent=2)
            with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)

            self._set_stage(job, "promote", 90)
            reason = None
            promoted = False
            if params["auto_activate"]:
                reason = self.gate(metrics, mode, parent)
            else:
                reason = "auto_activate_disabled"
            meta = {
                "version": job["version"],
                "created_at": time.time(),
                "source": "web_train",
                "label": params.get("label"),
                "parent_version": parent,
                "mode": mode,
                "classes": classes,
                "params": {
                    "epochs": params["epochs"],
                    "lr": params["lr"],
                    "limit_per_class": params["limit_per_class"],
                    "init_from": _rel(init_from) if init_from else None,
                    "target_fpr": params["target_fpr"],
                },
                "dataset": {
                    "data_dir": params.get("data_dir_display") or _rel(params["data_dir"]),
                    "total": int(len(X)),
                    "counts": counts,
                },
                "metrics": metrics,
                "thresholds": weights["thresholds"],
                "promoted": False,
                "gate": reason or "passed",
                "note": "",
            }
            if reason is None:
                meta["promoted"] = True
                self.model_manager.promote(weights_path, meta, activate=True)
                promoted = True
                self._log(job, "已自动升级当前版本为 {}".format(job["version"]))
            else:
                self.model_manager.promote(weights_path, meta, activate=False)
                self._log(job, "未自动启用（{}）：版本 {} 已保存为候选".format(reason, job["version"]))

            job["result"] = {
                "version": job["version"],
                "promoted": promoted,
                "gate": reason,
                "previous_version": parent,
                "metrics": metrics,
                "thresholds": weights["thresholds"],
                "dataset": meta["dataset"],
                "message": ("训练完成，已自动升级版本 {}".format(job["version"]) if promoted
                            else "训练完成，未自动启用（{}），版本 {} 已保存为候选".format(
                                reason, job["version"])),
            }
            job["status"] = "done"
            job["progress"] = 100
            job["finished_at"] = time.time()
        except TrainingError as exc:
            job["status"] = "failed"
            job["error"] = exc.message
            job["issues"] = exc.issues
            job["finished_at"] = time.time()
            self._log(job, "失败：{}".format(exc.message))
        except Exception as exc:  # noqa: BLE001
            job["status"] = "failed"
            job["error"] = str(exc)
            job["finished_at"] = time.time()
            self._log(job, "异常：{}".format(exc))
            traceback.print_exc()
