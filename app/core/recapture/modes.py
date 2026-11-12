"""识别模式与类别契约（推理、训练、评估、版本兼容与 API 校验的唯一来源）。"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

MODE_SCREEN_ONLY = "screen_only"
MODE_SCREEN_AND_PRINT = "screen_and_print"
DEFAULT_MODE = MODE_SCREEN_AND_PRINT

ALL_CLASSES: Tuple[str, ...] = ("original", "screen_recapture", "print_recapture")

CLASS_LABELS: Dict[str, str] = {
    "original": "原件实拍",
    "screen_recapture": "屏幕翻拍",
    "print_recapture": "打印翻拍",
}

MODE_LABELS: Dict[str, str] = {
    MODE_SCREEN_ONLY: "仅屏幕翻拍",
    MODE_SCREEN_AND_PRINT: "屏幕+打印翻拍",
}

# 每种模式训练/推理可用的类别全集。
MODES: Dict[str, Tuple[str, ...]] = {
    MODE_SCREEN_ONLY: ("original", "screen_recapture"),
    MODE_SCREEN_AND_PRINT: ALL_CLASSES,
}


class ModeError(ValueError):
    """非法的识别模式。"""


def normalize_mode(mode: Optional[str]) -> str:
    """把任意输入归一化为合法模式，未知值回落到默认模式。"""
    return mode if mode in MODES else DEFAULT_MODE


def classes_for_mode(mode: Optional[str]) -> Tuple[str, ...]:
    """该模式下训练使用的类别集合。"""
    return MODES[normalize_mode(mode)]


def required_classes(mode: Optional[str]) -> Tuple[str, ...]:
    """训练/兼容性判断中必须出现的类别（打印类在屏幕+打印模式下为强烈建议）。"""
    return ("original", "screen_recapture")


def optional_classes(mode: Optional[str]) -> Tuple[str, ...]:
    """可选但建议补齐的类别。"""
    return ("print_recapture",) if normalize_mode(mode) == MODE_SCREEN_AND_PRINT else ()


def allowed_classes(classes: Iterable[str], mode: Optional[str]) -> Tuple[str, ...]:
    """推理时实际参与计算的类别（= 模型类别 ∩ 模式类别）。"""
    allowed = set(MODES[normalize_mode(mode)])
    return tuple(c for c in classes if c in allowed)


def is_compatible(classes: Iterable[str], mode: Optional[str]) -> bool:
    """模型类别是否覆盖该模式的全部推理类别。

    仅屏幕模式接受 2 类或 3 类模型（3 类会屏蔽打印）；
    屏幕+打印模式要求模型含 print_recapture。
    """
    return set(classes_for_mode(mode)).issubset(set(classes))


def derive_mode(classes: Iterable[str]) -> str:
    """由类别集合反推训练模式。"""
    return MODE_SCREEN_AND_PRINT if "print_recapture" in set(classes) else MODE_SCREEN_ONLY


def label_of(class_name: str) -> str:
    return CLASS_LABELS.get(class_name, class_name)


def mode_label(mode: Optional[str]) -> str:
    return MODE_LABELS.get(normalize_mode(mode), str(mode))
