"""应用上下文：装配流水线、存储、队列、模型管理，供 FastAPI 与 stdlib 服务共用。"""
from __future__ import annotations

import os
import threading
from typing import Dict, List

from .core.config import ROOT, load_settings
from .core.pipeline import Pipeline
from .services.model_manager import ModelManager
from .services.queue import BatchQueue
from .services.storage import Storage


class AppContext:
    def __init__(self):
        self.settings = load_settings()
        self.pipeline = Pipeline(self.settings)
        self.storage = Storage(os.path.join(ROOT, self.settings.get("storage", {}).get("db_path", "data/app.db")))
        self.queue = BatchQueue(self.pipeline, max_workers=2)
        self.model_manager = ModelManager(os.path.join(ROOT, "models/recapture"))
        self.train_jobs: Dict[str, Dict] = {}
        self._train_lock = threading.Lock()

    # ---- 分析 ----
    def analyze(self, data: bytes, file_name: str = "upload.jpg", save: bool = True) -> Dict:
        res = self.pipeline.analyze(data)
        if res.get("code") == 0 and save:
            store = bool(self.settings.get("privacy", {}).get("store_original", False)) is False
            if store:
                self.storage.add_history(file_name, res["result"])
        return res

    # ---- 批量 ----
    def batch(self, files: List[tuple]) -> str:
        return self.queue.submit(files)

    def batch_status(self, tid: str) -> Dict:
        return self.queue.get(tid)

    # ---- 反馈 ----
    def submit_feedback(self, file_name: str, true_class: str, image_data: bytes) -> str:
        fb_dir = os.path.join(ROOT, self.settings.get("storage", {}).get("feedback_dir", "data/feedback"))
        os.makedirs(fb_dir, exist_ok=True)
        safe = "".join(c for c in file_name if c.isalnum() or c in "._-") or "sample.jpg"
        path = os.path.join(fb_dir, f"{int(__import__('time').time() * 1000)}_{safe}")
        with open(path, "wb") as f:
            f.write(image_data)
        return self.storage.add_feedback(file_name, true_class, path)

    # ---- 训练 ----
    def start_train(self, params: Dict) -> str:
        import uuid
        tid = uuid.uuid4().hex
        with self._train_lock:
            self.train_jobs[tid] = {"id": tid, "status": "running", "progress": 0, "logs": [], "error": None}
        t = threading.Thread(target=self._run_train, args=(tid, params), daemon=True)
        t.start()
        return tid

    def _run_train(self, tid: str, params: Dict):
        job = self.train_jobs[tid]
        try:
            from tools.train_recapture import train
            out = params.get("out") or "runs/recap_web"
            wp = train(
                data_dir=params.get("data_dir", "data/recapture"),
                out_dir=out,
                epochs=int(params.get("epochs", 60)),
                limit_per_class=int(params.get("limit_per_class", 0)),
                class_weights=params.get("class_weights"),
            )
            job["weights_path"] = wp
            job["status"] = "done"
            job["progress"] = 100
        except Exception as exc:  # noqa: BLE001
            job["status"] = "failed"
            job["error"] = str(exc)

    def train_status(self, tid: str) -> Dict:
        return self.train_jobs.get(tid, {"status": "not_found"})
