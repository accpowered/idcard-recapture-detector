"""应用上下文：装配流水线、存储、队列、模型管理与训练闭环，供 FastAPI 与 stdlib 服务共用。"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

from .core.config import ROOT
from .core.pipeline import Pipeline
from .services.model_manager import ModelManager
from .services.queue import BatchQueue
from .services.settings_store import SettingsStore
from .services.storage import Storage
from .services.training import TrainingRunner


class AppContext:
    def __init__(self, settings_store: Optional[SettingsStore] = None,
                 model_manager: Optional[ModelManager] = None):
        self.settings_store = settings_store or SettingsStore()
        self.settings = self.settings_store.get_all()
        keep_versions = int(self.settings_store.get("training.keep_versions", 10) or 10)
        self.model_manager = model_manager or ModelManager(keep_versions=keep_versions)
        self.pipeline = Pipeline(self.settings, settings_store=self.settings_store,
                                 model_manager=self.model_manager)
        self.storage = Storage(os.path.join(
            ROOT, self.settings.get("storage", {}).get("db_path", "data/app.db")))
        self.queue = BatchQueue(self.pipeline, max_workers=2)
        self.training = TrainingRunner(
            self.model_manager, self.settings_store,
            data_root=os.path.join(ROOT, "data"),
            runs_root=os.path.join(ROOT, "runs"))
        self.train_jobs = self.training.jobs

    # ---- 分析 ----
    def analyze(self, data: bytes, file_name: str = "upload.jpg", save: bool = True,
                model_version: Optional[str] = None) -> Dict:
        res = self.pipeline.analyze(data, model_version=model_version)
        if res.get("code") == 0 and save:
            store = bool(self.settings.get("privacy", {}).get("store_original", False)) is False
            if store:
                self.storage.add_history(file_name, res["result"])
        return res

    # ---- 批量 ----
    def batch(self, files: List[tuple], model_version: Optional[str] = None) -> str:
        return self.queue.submit(files, model_version=model_version)

    def batch_status(self, tid: str) -> Dict:
        return self.queue.get(tid)

    # ---- 反馈 ----
    def submit_feedback(self, file_name: str, true_class: str, image_data: bytes) -> str:
        fb_dir = os.path.join(ROOT, self.settings.get("storage", {}).get("feedback_dir", "data/feedback"))
        os.makedirs(fb_dir, exist_ok=True)
        safe = "".join(c for c in file_name if c.isalnum() or c in "._-") or "sample.jpg"
        path = os.path.join(fb_dir, "{}_{}".format(int(__import__("time").time() * 1000), safe))
        with open(path, "wb") as f:
            f.write(image_data)
        return self.storage.add_feedback(file_name, true_class, path)

    # ---- 训练（兼容旧调用） ----
    def start_train(self, params: Dict) -> str:
        return self.training.start(params)["id"]

    def train_status(self, tid: str) -> Dict:
        return self.training.status(tid)
