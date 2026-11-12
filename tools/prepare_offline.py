"""构建离线发行包：下载依赖 wheel + 导出 OCR 模型。

在有网络的机器上运行（建议在项目 venv 中）：
    python tools/prepare_offline.py --out offline

产物 offline/ 结构：
    offline/wheels/            # pip 依赖 wheel
    offline/rapidocr_models/   # PP-OCR 中文 det/rec/cls ONNX（约 30MB）
    offline/install_offline.bat / .sh
    offline/README.md

把「源码目录 + offline/」一起拷到离线机器，执行 install_offline.* 即可。
注意：wheel 与平台相关，Windows 包需在 Windows 上准备，Linux 同理。
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def download_wheels(out_dir: str, req_file: str) -> bool:
    wheels = os.path.join(out_dir, "wheels")
    os.makedirs(wheels, exist_ok=True)
    cmd = [sys.executable, "-m", "pip", "download", "-r", req_file, "-d", wheels]
    print("[1/2] 下载依赖 wheel：", " ".join(cmd))
    try:
        return subprocess.run(cmd, check=False).returncode == 0
    except Exception as exc:  # noqa: BLE001
        print("下载 wheel 失败：", exc)
        return False


def export_models(out_dir: str) -> bool:
    dst = os.path.join(out_dir, "rapidocr_models")
    os.makedirs(dst, exist_ok=True)
    try:
        import rapidocr
        from rapidocr import RapidOCR
        print("[2/2] 初始化 RapidOCR（首次会下载模型）…")
        RapidOCR()  # 触发模型下载/校验
        src = os.path.join(os.path.dirname(rapidocr.__file__), "models")
        if not os.path.isdir(src):
            print("未找到 rapidocr 模型目录：", src)
            return False
        n = 0
        for f in os.listdir(src):
            sp = os.path.join(src, f)
            if os.path.isfile(sp) and f.endswith(".onnx"):
                shutil.copy2(sp, os.path.join(dst, f))
                n += 1
        print(f"已导出 {n} 个 OCR 模型 → {dst}")
        return n > 0
    except Exception as exc:  # noqa: BLE001
        print("导出 OCR 模型失败（可跳过，离线时仅无 OCR）：", exc)
        return False


def write_scripts(out_dir: str):
    bat = [
        "@echo off",
        "chcp 65001 >nul",
        "cd /d \"%~dp0\\..\"",
        "echo [1/2] 离线安装依赖...",
        "python -m pip install --no-index --find-links \"offline\\wheels\" -r requirements.txt",
        "echo [2/2] 安装 OCR 模型...",
        "python -c \"import os,shutil,rapidocr; d=os.path.join(os.path.dirname(rapidocr.__file__),'models'); os.makedirs(d,exist_ok=True); s='offline/rapidocr_models'; [shutil.copy2(os.path.join(s,f),d) for f in os.listdir(s) if f.endswith('.onnx')]\"",
        "echo 完成。运行 run.bat 启动。",
        "pause",
    ]
    with open(os.path.join(out_dir, "install_offline.bat"), "w", encoding="utf-8", newline="\r\n") as f:
        f.write("\r\n".join(bat) + "\r\n")

    sh = [
        "#!/usr/bin/env bash",
        "set -e",
        "cd \"$(dirname \"$0\")/..\"",
        "echo '[1/2] 离线安装依赖...'",
        "python3 -m pip install --no-index --find-links offline/wheels -r requirements.txt",
        "echo '[2/2] 安装 OCR 模型...'",
        "python3 -c \"import os,shutil,rapidocr; d=os.path.join(os.path.dirname(rapidocr.__file__),'models'); os.makedirs(d,exist_ok=True); s='offline/rapidocr_models'; [shutil.copy2(os.path.join(s,f),d) for f in os.listdir(s) if f.endswith('.onnx')]\"",
        "echo '完成。运行 ./run.sh 启动。'",
    ]
    with open(os.path.join(out_dir, "install_offline.sh"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(sh) + "\n")

    readme = """# 离线发行包

本目录由 tools/prepare_offline.py 生成。

## 使用
1. 把「整个项目目录」与 offline/ 一起拷贝到离线机器（同操作系统、同 Python 大版本）。
2. 在项目根目录执行：
   - Windows: offline\\install_offline.bat
   - Linux:   bash offline/install_offline.sh
3. 运行 run.bat / ./run.sh 启动。

## 说明
- wheels 为平台相关；Windows 包需在 Windows 上生成。
- rapidocr_models 为 PP-OCRv6 中文 det/rec/cls ONNX，安装后复制到 rapidocr 包内 models/。
- 若目标机已具备依赖与模型，可跳过安装，直接 python run.py。
"""
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="offline")
    ap.add_argument("--requirements", default=os.path.join(ROOT, "requirements.txt"))
    ap.add_argument("--skip-wheels", action="store_true", help="仅导出模型（不下载 wheel）")
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    ok_w = True
    if not args.skip_wheels:
        ok_w = download_wheels(out, args.requirements)
    ok_m = export_models(out)
    write_scripts(out)
    print(f"\n离线包已生成：{out}")
    print("  wheel:", "OK" if ok_w else "失败", " | OCR 模型:", "OK" if ok_m else "跳过/失败")
    return 0 if (ok_m or ok_w) else 1


if __name__ == "__main__":
    raise SystemExit(main())
