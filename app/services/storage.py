"""本地 SQLite 存储（历史记录 + 误判反馈，敏感字段已脱敏）。"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from typing import Dict, List, Optional


class Storage:
    def __init__(self, db_path: str):
        self.db_path = db_path
        d = os.path.dirname(db_path)
        if d:
            os.makedirs(d, exist_ok=True)
        self._init()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self):
        with self._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS history (
                    id TEXT PRIMARY KEY,
                    created_at REAL,
                    file_name TEXT,
                    result_json TEXT
                )""")
            c.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY,
                    created_at REAL,
                    file_name TEXT,
                    true_class TEXT,
                    image_path TEXT,
                    note TEXT,
                    status TEXT DEFAULT 'pending'
                )""")

    def add_history(self, file_name: str, result: Dict) -> str:
        rid = uuid.uuid4().hex
        with self._conn() as c:
            c.execute("INSERT INTO history (id, created_at, file_name, result_json) VALUES (?,?,?,?)",
                      (rid, time.time(), file_name, json.dumps(result, ensure_ascii=False)))
        self._trim()
        return rid

    def _trim(self, limit: int = 5000):
        with self._conn() as c:
            c.execute("DELETE FROM history WHERE id NOT IN (SELECT id FROM history ORDER BY created_at DESC LIMIT ?)", (limit,))

    def list_history(self, limit: int = 200, offset: int = 0) -> List[Dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM history ORDER BY created_at DESC LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        out = []
        for r in rows:
            try:
                res = json.loads(r["result_json"])
            except Exception:  # noqa: BLE001
                res = {}
            out.append({
                "id": r["id"], "created_at": r["created_at"], "file_name": r["file_name"],
                "summary": {
                    "class": res.get("recapture", {}).get("class"),
                    "risk": res.get("recapture", {}).get("risk"),
                    "score": res.get("recapture", {}).get("score"),
                    "name": res.get("fields", {}).get("name"),
                    "id_number": res.get("fields", {}).get("id_number"),
                },
            })
        return out

    def get_history(self, rid: str) -> Optional[Dict]:
        with self._conn() as c:
            r = c.execute("SELECT * FROM history WHERE id=?", (rid,)).fetchone()
        if not r:
            return None
        try:
            return json.loads(r["result_json"])
        except Exception:  # noqa: BLE001
            return {}

    def add_feedback(self, file_name: str, true_class: str, image_path: str, note: str = "") -> str:
        fid = uuid.uuid4().hex
        with self._conn() as c:
            c.execute("INSERT INTO feedback (id, created_at, file_name, true_class, image_path, note, status) VALUES (?,?,?,?,?,?, 'pending')",
                      (fid, time.time(), file_name, true_class, image_path, note))
        return fid

    def list_feedback(self, status: str = "pending") -> List[Dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM feedback WHERE status=? ORDER BY created_at DESC", (status,)).fetchall()
        return [dict(r) for r in rows]

    def confirm_feedback(self, fid: str, confirmed_class: str, target_dir: str) -> bool:
        """确认反馈样本标签，并移动到再训练目录。"""
        with self._conn() as c:
            r = c.execute("SELECT * FROM feedback WHERE id=?", (fid,)).fetchone()
            if not r:
                return False
            src = r["image_path"]
        if src and os.path.exists(src):
            dst_dir = os.path.join(target_dir, confirmed_class)
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, os.path.basename(src))
            try:
                os.replace(src, dst)
            except OSError:
                import shutil
                shutil.copy2(src, dst)
        with self._conn() as c:
            c.execute("UPDATE feedback SET status='confirmed', true_class=? WHERE id=?", (confirmed_class, fid))
        return True

    def reject_feedback(self, fid: str) -> bool:
        with self._conn() as c:
            c.execute("UPDATE feedback SET status='rejected' WHERE id=?", (fid,))
        return True
