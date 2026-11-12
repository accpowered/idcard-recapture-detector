"""身份证关键字段解析与校验（GB 11643-1999）。

输入 OCR 文本行（列表），输出结构化字段 + 校验结果 + 敏感信息打码。
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .config import load_fields

# GB 11643-1999 校验位加权因子
_WEIGHTS = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
_CHECK = "10X98765432"


def gb11643_checksum(id_number: str) -> bool:
    """校验 18 位公民身份号码校验位。"""
    if not id_number or len(id_number) != 18:
        return False
    s = id_number.upper()
    if not re.fullmatch(r"[0-9]{17}[0-9X]", s):
        return False
    total = 0
    for w, c in zip(_WEIGHTS, s[:17]):
        if not c.isdigit():
            return False
        total += w * int(c)
    return _CHECK[total % 11] == s[-1]


def _mask_id(id_number: str) -> str:
    if len(id_number) == 18:
        return id_number[:6] + "********" + id_number[-4:]
    return "*" * len(id_number)


def _mask_name(name: str) -> str:
    if not name:
        return ""
    return name[0] + "*" * (len(name) - 1)


def _mask_address(addr: str) -> str:
    if not addr:
        return ""
    if len(addr) <= 3:
        return addr[0] + "**"
    return addr[:3] + "***（打码）"


def parse_fields(text_lines: List[str], mask: bool = True) -> Dict:
    """从 OCR 文本行解析字段。text_lines 为纯文本行列表（已按阅读顺序）。"""
    cfg = load_fields()
    ethnicities = set(cfg.get("ethnicities", ["汉", "其他"]))

    result: Dict = {
        "name": "", "gender": "", "ethnicity": "", "birth": "", "address": "",
        "id_number": "", "authority": "", "valid_period": "",
        "id_checksum_valid": None, "gender_consistent": None, "birth_consistent": None,
        "confidence": "low", "raw_lines": text_lines,
    }

    full_text = " ".join(text_lines)
    compact = re.sub(r"\s+", "", full_text)

    # 公民身份号码：18 位（末尾可能 X）
    m = re.search(r"(?<![0-9])([1-9][0-9]{5}(?:19|20)[0-9]{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12][0-9]|3[01])[0-9]{3}[0-9Xx])(?![0-9])", compact)
    if m:
        result["id_number"] = m.group(1).upper()
        result["id_checksum_valid"] = gb11643_checksum(result["id_number"])

    # 性别
    m = re.search(r"性别[:：]?\s*([男女])", full_text)
    if m:
        result["gender"] = m.group(1)
    elif "男" in full_text and "女" not in full_text:
        result["gender"] = "男"
    elif "女" in full_text:
        result["gender"] = "女"

    # 民族
    m = re.search(r"民族[:：]?\s*([\u4e00-\u9fff]{1,4})", full_text)
    if m and m.group(1) in ethnicities:
        result["ethnicity"] = m.group(1)

    # 出生
    m = re.search(r"出生[:：]?\s*(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日", full_text)
    if m:
        result["birth"] = f"{m.group(1)}年{int(m.group(2))}月{int(m.group(3))}日"
    elif re.search(r"出生", full_text):
        m2 = re.search(r"(\d{4})[年.]?(\d{1,2})[月.](\d{1,2})", compact)
        if m2:
            result["birth"] = f"{m2.group(1)}年{int(m2.group(2))}月{int(m2.group(3))}日"

    # 姓名：无关键字，取“姓名”后 2-4 汉字，或首行
    m = re.search(r"姓名[:：]?\s*([\u4e00-\u9fff]{1,15})", full_text)
    if m:
        result["name"] = m.group(1)

    # 住址：以“住址”锚定，跨行拼接
    m = re.search(r"住址[:：]?\s*([\u4e00-\u9fff0-9号镇乡县区市省路街道村组栋单元室]+)", full_text)
    if m:
        result["address"] = m.group(1)

    # 签发机关
    m = re.search(r"签发机关[:：]?\s*([\u4e00-\u9fff]+)", full_text)
    if m:
        result["authority"] = m.group(1)

    # 有效期限
    m = re.search(r"有效期限[:：]?\s*([0-9]{4}\.?[0-9]{1,2}\.?[0-9]{1,2}\s*[-—~至]\s*(?:[0-9]{4}\.?[0-9]{1,2}\.?[0-9]{1,2}|长期))", full_text)
    if m:
        result["valid_period"] = re.sub(r"\s+", "", m.group(1))
    elif "长期" in full_text:
        m2 = re.search(r"(\d{4}[.年]\d{1,2}[.月]\d{1,2})[-—~至]?长期", compact)
        if m2:
            result["valid_period"] = m2.group(1) + "-长期"

    # 交叉校验：性别 vs 号码第 17 位；出生 vs 号码 7-14 位
    if result["id_number"] and result["gender"]:
        digit = int(result["id_number"][16])
        expected = "男" if digit % 2 == 1 else "女"
        result["gender_consistent"] = (expected == result["gender"])
    if result["id_number"] and result["birth"]:
        ymd = re.findall(r"\d+", result["birth"])
        if len(ymd) >= 3:
            dob = f"{int(ymd[0]):04d}{int(ymd[1]):02d}{int(ymd[2]):02d}"
            result["birth_consistent"] = (result["id_number"][6:14] == dob)

    # 置信度粗略估计
    filled = sum(1 for k in ("name", "gender", "ethnicity", "birth", "address", "id_number", "authority", "valid_period") if result[k])
    if filled >= 7 and result["id_checksum_valid"]:
        result["confidence"] = "high"
    elif filled >= 4:
        result["confidence"] = "medium"

    # 打码
    result["masked"] = mask
    if mask:
        result["name"] = _mask_name(result["name"]) if result["name"] else ""
        result["id_number"] = _mask_id(result["id_number"]) if result["id_number"] else ""
        result["address"] = _mask_address(result["address"]) if result["address"] else ""

    return result


def validate_date(ymd: str) -> bool:
    try:
        datetime.strptime(ymd, "%Y%m%d")
        return True
    except Exception:  # noqa: BLE001
        return False
