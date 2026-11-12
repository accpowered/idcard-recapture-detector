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

## 联网要求（重要）

解压源码后，**翻拍判定核心能力**（信号分析 + 特征分类器）不依赖外部下载，
但以下两项首次运行需要联网：

1. `pip install -r requirements.txt` —— 安装 numpy/opencv/fastapi/rapidocr/onnxruntime 等依赖；
2. 首次初始化 OCR 时，rapidocr 会自动下载 PP-OCRv6 中文 det/rec/cls ONNX（约 30MB）到其包内 models/。

如果目标机已装好依赖、或已按下面方式打包，即可完全离线运行。

## 完全离线发行包（推荐）

在有网的机器上（建议在项目 venv 内）运行：

    python tools/prepare_offline.py --out offline

产物：`offline/wheels/`（依赖 wheel）、`offline/rapidocr_models/`（OCR 模型）、
`offline/install_offline.bat` / `.sh`、`offline/README.md`。

把「整个项目目录 + offline/」拷到离线机器（同操作系统、同 Python 大版本），执行：

    # Windows
    offline\install_offline.bat
    # Linux
    bash offline/install_offline.sh

然后 `run.bat` / `./run.sh` 启动即可，全程不联网。

注意：wheel 与平台相关（Windows 包在 Windows 准备，Linux 同理）；`--skip-wheels` 可只导出 OCR 模型。

## 更省事的离线方式

若目标机已有 Python 环境，直接**整包拷贝项目目录（连同 `.venv` 或 `vendor/`）**，
用 `.venv\Scripts\python.exe run.py` 启动即可，无需任何安装。
（要求：同操作系统、同 Python 大版本。）

## 常见问题

- 端口占用：python run.py --port 8080
- HEIC 支持：pip install pillow-heif
- OpenVINO 加速（Intel CPU）：pip install onnxruntime-openvino
