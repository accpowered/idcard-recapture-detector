"""字段解析与 GB11643 校验位测试。"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.fields import gb11643_checksum, parse_fields, _mask_id  # noqa: E402


def test_checksum_valid():
    # 11010519491231002X 为公开示例有效号码
    assert gb11643_checksum("11010519491231002X") is True
    assert gb11643_checksum("11010519491231002x") is True


def test_checksum_invalid():
    assert gb11643_checksum("110105194912310021") is False
    assert gb11643_checksum("123") is False
    assert gb11643_checksum("") is False


def test_parse_fields_basic():
    lines = [
        "姓名 张三",
        "性别 女 民族 汉",
        "出生 1949年12月31日",
        "住址 北京市海淀区中关村大街1号",
        "公民身份号码 11010519491231002X",
        "签发机关 北京市公安局海淀分局",
        "有效期限 2020.01.01-2040.01.01",
    ]
    r = parse_fields(lines, mask=False)
    assert r["name"] == "张三"
    assert r["gender"] == "女"
    assert r["ethnicity"] == "汉"
    assert r["birth"] == "1949年12月31日"
    assert r["id_number"] == "11010519491231002X"
    assert r["id_checksum_valid"] is True
    assert r["gender_consistent"] is True   # 第17位 2 为偶 → 女
    assert r["birth_consistent"] is True    # 号码 7-14 位 == 19491231
    assert r["authority"] == "北京市公安局海淀分局"


def test_masking():
    r = parse_fields(["姓名 张三", "公民身份号码 11010519491231002X"], mask=True)
    assert r["name"] == "张*"
    assert r["id_number"] == "110105********002X"


def test_mask_id_helper():
    assert _mask_id("11010519491231002X") == "110105********002X"
