"""训练闭环集成测试（小样本合成数据，验证训练→评估→自动上线→回退）。"""
import json
import os
import shutil
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from app.core.config import ROOT  # noqa: E402
from app.core.recapture.classifier import DEFAULT_WEIGHTS  # noqa: E402
from app.services.model_manager import ModelManager  # noqa: E402
from app.services.training import TrainingRunner, dataset_report  # noqa: E402
from tests.make_benchmark import _card  # noqa: E402
from tools.synthesize import add_halftone, add_moire, add_screen_grid  # noqa: E402


class FakeSettings:
    def __init__(self, mode="screen_and_print", min_macro_f1=0.5):
        self._mode = mode
        self._min = min_macro_f1

    @property
    def mode(self):
        return self._mode

    def get(self, key, default=None):
        values = {
            "training.epochs": 30,
            "training.lr": 0.5,
            "training.target_fpr": 0.05,
            "training.min_macro_f1": self._min,
            "training.keep_versions": 5,
        }
        return values.get(key, default)


def _tmpdir(name):
    path = os.path.join(ROOT, ".tmp", "tests", "{}-{}".format(name, uuid.uuid4().hex[:8]))
    os.makedirs(path, exist_ok=True)
    return path


def _make_dataset(root):
    rng = np.random.default_rng(11)
    for cls in ("original", "screen_recapture", "print_recapture"):
        directory = os.path.join(root, cls)
        os.makedirs(directory, exist_ok=True)
        for index in range(5):
            image = _card()
            if cls == "screen_recapture":
                image = add_moire(image, rng.uniform(0, 180), rng.uniform(0.03, 0.07),
                                  rng.uniform(0.1, 0.25))
                image = add_screen_grid(image, int(rng.choice([3, 4, 5])), rng.uniform(0.4, 0.7))
            elif cls == "print_recapture":
                image = add_halftone(image, int(rng.choice([3, 4])), rng.uniform(0.4, 0.7))
            Image.fromarray(image).save(
                os.path.join(directory, "{}_{:02d}.jpg".format(cls, index)), quality=90)


def _baseline(mm, tmp):
    path = os.path.join(tmp, "baseline.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_WEIGHTS, f)
    mm.promote(path, {"version": "1.0.0", "created_at": time.time(),
                      "source": "baseline", "mode": "screen_and_print",
                      "classes": list(DEFAULT_WEIGHTS["classes"]),
                      "metrics": None, "promoted": True}, activate=True)


def test_training_flow_promotes_and_rolls_back():
    tmp = _tmpdir("train-flow")
    try:
        mm = ModelManager(root=os.path.join(tmp, "models"), keep_versions=5)
        _baseline(mm, tmp)
        data = os.path.join(tmp, "data", "recapture")
        _make_dataset(data)
        report = dataset_report(data, "screen_and_print")
        assert report["ready"] is True, report["issues"]

        runner = TrainingRunner(mm, FakeSettings(),
                                data_root=os.path.join(tmp, "data"),
                                runs_root=os.path.join(tmp, "runs"))
        job = runner.start({"data_dir": data, "epochs": 30, "label": "unittest"})
        deadline = time.time() + 240
        status = None
        while time.time() < deadline:
            status = runner.status(job["id"])
            if status["status"] in ("done", "failed"):
                break
            time.sleep(0.3)
        assert status is not None
        assert status["status"] == "done", status.get("error")
        result = status["result"]
        assert result["promoted"] is True, result
        assert result["gate"] is None
        assert mm.current_version() == result["version"]
        assert os.path.isdir(os.path.join(tmp, "models", "versions", result["version"]))
        assert os.path.exists(os.path.join(tmp, "models", "METRICS.md"))
        assert result["metrics"]["macro_f1"] > 0.5
        assert result["metrics"]["classes"] == [
            "original", "screen_recapture", "print_recapture"]
        assert mm.activate("1.0.0") is True
        assert mm.current_version() == "1.0.0"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_gate_reasons():
    tmp = _tmpdir("train-gate")
    try:
        mm = ModelManager(root=os.path.join(tmp, "models"), keep_versions=5)
        runner = TrainingRunner(mm, FakeSettings(), runs_root=os.path.join(tmp, "runs"))
        metrics = {"macro_f1": 0.4,
                   "per_class": {"original": {"recall": 1.0},
                                 "screen_recapture": {"recall": 1.0}}}
        assert runner.gate(metrics, "screen_and_print", mm.current_version(),
                           min_f1=0.5) == "below_min_f1"
        good = {"macro_f1": 0.9,
                "per_class": {"original": {"recall": 1.0},
                              "screen_recapture": {"recall": 1.0}}}
        assert runner.gate(good, "screen_and_print", "other-version",
                           min_f1=0.5) == "current_changed"
        screen_runner = TrainingRunner(mm, FakeSettings(mode="screen_only"),
                                       runs_root=os.path.join(tmp, "runs"))
        assert screen_runner.gate(good, "screen_and_print", mm.current_version(),
                                  min_f1=0.5) == "mode_changed"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
