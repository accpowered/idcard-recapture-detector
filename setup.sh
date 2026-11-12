#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[错误] 未找到 python3，请先安装 Python 3.10+" >&2
  exit 1
fi

if [ ! -d .venv ]; then
  echo "[1/3] 创建虚拟环境 .venv ..."
  python3 -m venv .venv
fi

echo "[2/3] 安装核心依赖 ..."
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

echo "[3/3] 补齐模型目录 ..."
.venv/bin/python tools/bootstrap_models.py

echo "安装完成。运行 ./run.sh 启动服务。"
