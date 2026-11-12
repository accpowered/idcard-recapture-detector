@echo off
chcp 65001 >nul
REM 一键安装：创建虚拟环境并安装核心依赖
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo [错误] 未找到 Python，请先安装 Python 3.10+ 并加入 PATH。
  pause
  exit /b 1
)

if not exist .venv (
  echo [1/3] 创建虚拟环境 .venv ...
  python -m venv .venv
)

echo [2/3] 安装核心依赖（清华镜像）...
".venv\Scripts\python.exe" -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
".venv\Scripts\python.exe" -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

echo [3/3] 补齐模型目录 ...
".venv\Scripts\python.exe" tools\bootstrap_models.py

echo.
echo 安装完成。运行 run.bat 启动服务。
endlocal
