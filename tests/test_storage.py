"""SQLite 存储回归测试：历史/反馈读写，且不依赖系统临时目录。"""
import os
import shutil
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import ROOT  # noqa: E402
from app.services.storage import Storage  # noqa: E402


def _tmpdir(name):
    path = os.path.join(ROOT, ".tmp", "tests", "{}-{}".format(name, uuid.uuid4().hex[:8]))
    os.makedirs(path, exist_ok=True)
    return path


def test_history_roundtrip_with_large_payload():
    tmp = _tmpdir("storage-history")
    try:
        storage = Storage(os.path.join(tmp, "app.db"))
        payload = {
            "annotated_b64": "A" * 200000,
            "recapture": {"class": "original", "risk": "low", "score": 0.12,
                          "model_version": "1.0.0", "mode": "screen_and_print"},
            "fields": {"name": "张*", "id_number": "110101********1234"},
        }
        rid = storage.add_history("a.jpg", payload)
        items = storage.list_history(10, 0)
        assert len(items) == 1
        assert items[0]["id"] == rid
        assert items[0]["summary"]["class"] == "original"
        assert items[0]["summary"]["model_version"] == "1.0.0"
        assert storage.get_history(rid)["fields"]["name"] == "张*"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_feedback_roundtrip():
    tmp = _tmpdir("storage-feedback")
    try:
        storage = Storage(os.path.join(tmp, "app.db"))
        image = os.path.join(tmp, "img.jpg")
        with open(image, "wb") as f:
            f.write(b"x")
        fid = storage.add_feedback("b.jpg", "screen_recapture", image)
        pending = storage.list_feedback("pending")
        assert len(pending) == 1
        assert pending[0]["id"] == fid
        assert storage.reject_feedback(fid)
        assert storage.list_feedback("pending") == []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
