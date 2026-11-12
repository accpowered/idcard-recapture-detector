"""模型版本管理测试。"""
import json
import os
import shutil
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import ROOT  # noqa: E402
from app.core.recapture.classifier import DEFAULT_WEIGHTS  # noqa: E402
from app.services.model_manager import ModelManager  # noqa: E402


def _tmpdir(name):
    path = os.path.join(ROOT, ".tmp", "tests", "{}-{}".format(name, uuid.uuid4().hex[:8]))
    os.makedirs(path, exist_ok=True)
    return path


def _write_weights(path, weights=None):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(weights or DEFAULT_WEIGHTS, f)


def _two_class_weights():
    weights = json.loads(json.dumps(DEFAULT_WEIGHTS))
    weights["classes"] = ["original", "screen_recapture"]
    weights["logits"] = {k: v for k, v in weights["logits"].items() if k != "print_recapture"}
    return weights


def _baseline(mm, tmp):
    path = os.path.join(tmp, "baseline.json")
    _write_weights(path)
    mm.promote(path, {"version": "1.0.0", "created_at": time.time(), "source": "baseline",
                      "mode": "screen_and_print",
                      "classes": list(DEFAULT_WEIGHTS["classes"]),
                      "metrics": None, "promoted": True}, activate=True)


def test_promote_activate_rollback():
    tmp = _tmpdir("mm-rollback")
    try:
        mm = ModelManager(root=os.path.join(tmp, "models"), keep_versions=5)
        _baseline(mm, tmp)
        assert mm.current_version() == "1.0.0"
        assert os.path.exists(os.path.join(tmp, "models", "weights.json"))
        assert os.path.exists(os.path.join(tmp, "models", "versions", "1.0.0", "meta.json"))

        new_weights = os.path.join(tmp, "v2.json")
        _write_weights(new_weights)
        version = mm.new_version_id("demo")
        mm.promote(new_weights, {"version": version, "created_at": time.time(),
                                 "source": "test", "mode": "screen_and_print",
                                 "classes": list(DEFAULT_WEIGHTS["classes"]),
                                 "metrics": {"accuracy": 0.9, "macro_f1": 0.9},
                                 "promoted": True}, activate=True)
        assert mm.current_version() == version
        assert mm.activate("1.0.0")
        assert mm.current_version() == "1.0.0"
        with open(os.path.join(tmp, "models", "weights.json"), encoding="utf-8") as f:
            assert json.load(f)["version"] == "1.0.0"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_incompatible_version_listing():
    tmp = _tmpdir("mm-compat")
    try:
        mm = ModelManager(root=os.path.join(tmp, "models"), keep_versions=5)
        _baseline(mm, tmp)
        path = os.path.join(tmp, "two.json")
        _write_weights(path, _two_class_weights())
        version = mm.new_version_id("two")
        mm.promote(path, {"version": version, "created_at": time.time(),
                          "source": "test", "mode": "screen_only",
                          "classes": ["original", "screen_recapture"],
                          "metrics": None, "promoted": False}, activate=False)
        items = {item["version"]: item for item in mm.list_versions("screen_and_print")}
        assert items[version]["compatible"] is False
        assert items[version]["reason"]
        assert items[version]["active"] is False
        screen_items = {item["version"]: item for item in mm.list_versions("screen_only")}
        assert screen_items[version]["compatible"] is True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_reject_unsafe_version():
    tmp = _tmpdir("mm-safe")
    try:
        mm = ModelManager(root=os.path.join(tmp, "models"), keep_versions=5)
        assert mm.weights_path("../escape") is None
        path = os.path.join(tmp, "w.json")
        _write_weights(path)
        try:
            mm.promote(path, {"version": "../escape", "created_at": time.time()})
        except ValueError:
            pass
        else:
            raise AssertionError("非法版本名应被拒绝")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_prune_keeps_recent_and_protected():
    tmp = _tmpdir("mm-prune")
    try:
        root = os.path.join(tmp, "models")
        mm = ModelManager(root=root, keep_versions=2)
        _baseline(mm, tmp)
        versions_dir = os.path.join(root, "versions")
        for index in range(4):
            name = "train-2026010{}".format(index + 1)
            vdir = os.path.join(versions_dir, name)
            os.makedirs(vdir, exist_ok=True)
            _write_weights(os.path.join(vdir, "weights.json"))
            with open(os.path.join(vdir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump({"version": name, "created_at": 1000 + index,
                           "mode": "screen_and_print",
                           "classes": list(DEFAULT_WEIGHTS["classes"])}, f)
        removed = mm.prune()
        remaining = {item["version"] for item in mm.list_versions()}
        assert "1.0.0" in remaining
        assert "train-20260104" in remaining
        assert "train-20260103" in remaining
        assert "train-20260102" not in remaining
        assert set(removed) == {"train-20260101", "train-20260102"}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_legacy_version_without_meta():
    tmp = _tmpdir("mm-legacy")
    try:
        root = os.path.join(tmp, "models")
        mm = ModelManager(root=root, keep_versions=5)
        vdir = os.path.join(root, "versions", "legacy-20260101")
        os.makedirs(vdir, exist_ok=True)
        _write_weights(os.path.join(vdir, "weights.json"))
        items = {item["version"]: item for item in mm.list_versions("screen_and_print")}
        assert "legacy-20260101" in items
        assert items["legacy-20260101"]["source"] == "unknown"
        assert items["legacy-20260101"]["compatible"] is True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
