# 部署与离线发行

## 环境要求

- Python 3.10+（Windows 10/11 或主流 Linux）
- 4 核 CPU、内存 ≥ 2.5 GB
- 无 GPU 要求

## 一键安装与启动

Windows：

    setup.bat
    run.bat

Linux / macOS：

    ./setup.sh
    ./run.sh

启动后浏览器自动访问 http://127.0.0.1:8000 （如未自动打开请手动访问）。

## 模型引导

首次启动运行 tools/bootstrap_models.py：

- 创建 models/ 目录结构；
- 生成随仓库发布的翻拍分类基线权重 models/recapture/weights.json（无下载即可翻拍判定）；
- 依据 models/checksums.json 对随仓库分发的模型二进制做 SHA256 校验。

官方权重（PP-OCRv5 中文 OCR、方向分类 PP-LCNet、UVDoc、卡证检测 YOLOv8n）体积较大，
按 §5.2 三级策略分发：随仓库 LFS / 引导下载（ModelScope/HF/官方源回退）/ 离线发行包。
未安装这些官方权重不影响翻拍判定核心能力（信号分析 + 特征分类器）。

## 离线发行包

对完全离线内网，打包：

    # 1) 导出依赖 wheel
    pip download -r requirements.txt -d dist/wheels
    # 2) 复制模型与脚本
    cp -r models dist/
    cp run.py setup.sh dist/

离线机执行：

    python -m pip install --no-index --find-links dist/wheels -r requirements.txt
    python run.py --offline

## 常见问题

- 端口占用：python run.py --port 8080
- HEIC 支持：pip install pillow-heif
- OpenVINO 加速（Intel CPU）：pip install onnxruntime-openvino
