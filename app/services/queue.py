"""进程内批量处理队列（线程池）。"""
from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List


class BatchQueue:
    def __init__(self, pipeline, max_workers: int = 2):
        self.pipeline = pipeline
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.tasks: Dict[str, Dict] = {}
        self.lock = threading.Lock()

    def submit(self, files: List[tuple]) -> str:
        """files: [(file_name, bytes), ...]"""
        tid = uuid.uuid4().hex
        task = {
            "id": tid, "total": len(files), "done": 0, "status": "running",
            "results": [], "errors": [], "created_at": time.time(),
        }
        with self.lock:
            self.tasks[tid] = task
        self.executor.submit(self._run, tid, files)
        return tid

    def _run(self, tid: str, files: List[tuple]):
        task = self.tasks[tid]
        for name, data in files:
            try:
                res = self.pipeline.analyze(data)
                entry = {"file_name": name, "result": res.get("result") if res.get("code") == 0 else None,
                         "error": res.get("error")}
                task["results"].append(entry)
                if res.get("code") != 0:
                    task["errors"].append(name)
            except Exception as exc:  # noqa: BLE001
                task["results"].append({"file_name": name, "error": str(exc)})
                task["errors"].append(name)
            task["done"] += 1
        task["status"] = "done"
        task["finished_at"] = time.time()

    def get(self, tid: str) -> Dict:
        return self.tasks.get(tid, {"status": "not_found"})
