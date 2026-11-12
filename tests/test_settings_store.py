"""运行时设置叠加层测试。"""
import json
import os
import shutil
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import ROOT  # noqa: E402
from app.services.settings_store import SettingsError, SettingsStore  # noqa: E402


def _tmpdir(name):
    path = os.path.join(ROOT, ".tmp", "tests", "{}-{}".format(name, uuid.uuid4().hex[:8]))
    os.makedirs(path, exist_ok=True)
    return path


def _store(tmp):
    return SettingsStore(base_path=os.path.join(tmp, "settings.yaml"),
                         override_path=os.path.join(tmp, "settings.local.json"))


def test_defaults():
    tmp = _tmpdir("settings-default")
    try:
        store = _store(tmp)
        assert store.get("ui.show_training") is True
        assert store.mode == "screen_and_print"
        assert abs(float(store.get("training.target_fpr")) - 0.05) < 1e-9
        assert int(store.get("training.keep_versions")) == 10
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_update_persists_overlay():
    tmp = _tmpdir("settings-update")
    try:
        store = _store(tmp)
        store.update({"ui": {"show_training": False},
                      "recapture": {"mode": "screen_only"},
                      "training": {"keep_versions": 4, "target_fpr": 0.02}})
        assert store.mode == "screen_only"
        assert store.get("ui.show_training") is False
        path = os.path.join(tmp, "settings.local.json")
        assert os.path.exists(path)
        with open(path, encoding="utf-8") as f:
            overlay = json.load(f)
        assert overlay["recapture"]["mode"] == "screen_only"
        assert overlay["training"]["keep_versions"] == 4
        reloaded = _store(tmp)
        assert reloaded.mode == "screen_only"
        assert reloaded.get("ui.show_training") is False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_invalid_patch_rejected():
    tmp = _tmpdir("settings-invalid")
    try:
        store = _store(tmp)
        for patch in (
            {"unknown": {"x": 1}},
            {"recapture": {"mode": "screen_only", "extra": 1}},
            {"recapture": {"mode": "bogus"}},
            {"training": {"keep_versions": 0}},
            {"training": {"target_fpr": 0.9}},
            {"ui": {"show_training": "yes"}},
        ):
            try:
                store.update(patch)
            except SettingsError:
                continue
            raise AssertionError("非法配置应被拒绝: {}".format(patch))
        assert store.mode == "screen_and_print"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_corrupt_overlay_tolerated():
    tmp = _tmpdir("settings-corrupt")
    try:
        path = os.path.join(tmp, "settings.local.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write("{not json")
        store = _store(tmp)
        assert store.get("ui.show_training") is True
        assert store.warnings
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_yaml_base_overrides_defaults():
    tmp = _tmpdir("settings-yaml")
    try:
        with open(os.path.join(tmp, "settings.yaml"), "w", encoding="utf-8") as f:
            f.write("ui:\n  show_training: false\nrecapture:\n  mode: screen_only\n")
        store = _store(tmp)
        assert store.get("ui.show_training") is False
        assert store.mode == "screen_only"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
