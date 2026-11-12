"""数据集体检与目录约束测试。"""
import os
import shutil
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import ROOT  # noqa: E402
from app.services.training import (  # noqa: E402
    TrainingError, dataset_report, resolve_data_dir,
)


def _tmpdir(name):
    path = os.path.join(ROOT, ".tmp", "tests", "{}-{}".format(name, uuid.uuid4().hex[:8]))
    os.makedirs(path, exist_ok=True)
    return path


def _samples(root, class_name, count):
    directory = os.path.join(root, class_name)
    os.makedirs(directory, exist_ok=True)
    for index in range(count):
        with open(os.path.join(directory, "sample_{:03d}.jpg".format(index)), "wb") as f:
            f.write(b"fake-image-bytes")


def test_empty_dataset_not_ready():
    tmp = _tmpdir("dataset-empty")
    try:
        report = dataset_report(os.path.join(tmp, "recapture"), "screen_and_print")
        assert report["ready"] is False
        assert report["total"] == 0
        codes = {issue["code"] for issue in report["issues"]}
        assert "too_few" in codes
        assert "missing_class" in codes
        assert any("original" in issue["message"] for issue in report["issues"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_only_screen_samples_blocked():
    tmp = _tmpdir("dataset-screen")
    try:
        root = os.path.join(tmp, "recapture")
        _samples(root, "screen_recapture", 6)
        report = dataset_report(root, "screen_and_print")
        assert report["ready"] is False
        assert any(issue["level"] == "error" and "original" in issue["message"]
                   for issue in report["issues"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_minimum_pairs_ready_and_print_warning():
    tmp = _tmpdir("dataset-ready")
    try:
        root = os.path.join(tmp, "recapture")
        _samples(root, "original", 5)
        _samples(root, "screen_recapture", 5)
        report = dataset_report(root, "screen_and_print")
        assert report["ready"] is True
        assert report["total"] == 10
        assert any(issue["code"] == "optional_missing" for issue in report["issues"])
        by_name = {item["name"]: item for item in report["classes"]}
        assert by_name["original"]["count"] == 5
        assert by_name["print_recapture"]["count"] == 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_screen_only_mode_uses_two_classes():
    tmp = _tmpdir("dataset-screenonly")
    try:
        root = os.path.join(tmp, "recapture")
        _samples(root, "original", 5)
        _samples(root, "screen_recapture", 5)
        _samples(root, "print_recapture", 5)
        report = dataset_report(root, "screen_only")
        assert [item["name"] for item in report["classes"]] == [
            "original", "screen_recapture"]
        assert report["ready"] is True
        assert not any(issue["code"] == "optional_missing" for issue in report["issues"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_strict_data_dir():
    tmp = _tmpdir("dataset-strict")
    try:
        data_root = os.path.join(tmp, "data")
        os.makedirs(os.path.join(data_root, "recapture"), exist_ok=True)
        inside = resolve_data_dir(os.path.join(data_root, "recapture"),
                                  root=data_root, strict=True)
        assert inside == os.path.join(data_root, "recapture")
        outside = os.path.join(tmp, "elsewhere")
        os.makedirs(outside, exist_ok=True)
        try:
            resolve_data_dir(outside, root=data_root, strict=True)
        except TrainingError:
            pass
        else:
            raise AssertionError("data 目录之外的路径应被拒绝")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
