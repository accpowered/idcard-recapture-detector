"""识别模式与类别契约测试。"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.recapture import modes  # noqa: E402
from app.core.recapture.classifier import (  # noqa: E402
    DEFAULT_WEIGHTS, ModelModeError, RecaptureClassifier,
)

FEATURES = {
    "moire": 0.8, "screen_grid": 0.6, "bezel": 0.2, "highlight": 0.3,
    "texture_degradation": 0.2, "halftone": 0.1, "paper_noise": 0.2,
    "saturation_drop": 0.1, "color_shift": 0.3,
}


def _two_class_weights():
    weights = json.loads(json.dumps(DEFAULT_WEIGHTS))
    weights["classes"] = ["original", "screen_recapture"]
    weights["recapture_mode"] = modes.MODE_SCREEN_ONLY
    weights["logits"] = {k: v for k, v in weights["logits"].items()
                         if k != "print_recapture"}
    return weights


def test_mode_mapping():
    assert modes.classes_for_mode("screen_only") == ("original", "screen_recapture")
    assert modes.classes_for_mode("screen_and_print") == (
        "original", "screen_recapture", "print_recapture")
    assert modes.required_classes("screen_and_print") == ("original", "screen_recapture")
    assert modes.optional_classes("screen_only") == ()
    assert modes.is_compatible(["original", "screen_recapture"], "screen_only")
    assert not modes.is_compatible(["original", "screen_recapture"], "screen_and_print")
    assert modes.is_compatible(
        ["original", "screen_recapture", "print_recapture"], "screen_and_print")
    assert modes.derive_mode(["original", "screen_recapture"]) == "screen_only"
    assert modes.normalize_mode("bogus") == "screen_and_print"


def test_screen_only_masks_print_and_renormalizes():
    clf = RecaptureClassifier(weights=json.loads(json.dumps(DEFAULT_WEIGHTS)))
    result = clf.classify(FEATURES, mode="screen_only")
    assert "print_recapture" not in result["probs"]
    assert abs(sum(result["probs"].values()) - 1.0) < 1e-3
    assert result["class"] in ("original", "screen_recapture")
    expected = (0.45 * result["probs"]["screen_recapture"] + 0.20 * result["signal_score"]) / 0.65
    assert abs(result["score"] - round(expected, 4)) < 1e-3
    assert result["mode"] == "screen_only"


def test_both_mode_keeps_three_classes():
    clf = RecaptureClassifier(weights=json.loads(json.dumps(DEFAULT_WEIGHTS)))
    result = clf.classify(FEATURES, mode="screen_and_print")
    assert set(result["probs"]) == {"original", "screen_recapture", "print_recapture"}
    assert result["mode"] == "screen_and_print"


def test_two_class_model_incompatible_in_both_mode():
    clf = RecaptureClassifier(weights=_two_class_weights())
    try:
        clf.classify(FEATURES, mode="screen_and_print")
    except ModelModeError:
        return
    raise AssertionError("2 类模型在屏幕+打印模式下应抛 ModelModeError")


def test_two_class_model_works_in_screen_only():
    clf = RecaptureClassifier(weights=_two_class_weights())
    result = clf.classify(FEATURES, mode="screen_only")
    assert set(result["probs"]) == {"original", "screen_recapture"}
    assert result["class"] in ("original", "screen_recapture")
