# 居民身份证翻拍识别工具

对本地 Web 上传的身份证照片做 **卡证定位 → 翻拍判定（原件实拍 / 屏幕翻拍 / 打印翻拍）→ 关键字段 OCR → 结构化输出与风险报告**，仅依赖 CPU，开箱即用，并支持一键再训练与热替换。

## 特性

- 三级翻拍判定：original / screen_recapture / print_recapture，输出分数 + 风险等级 + 可解释线索（摩尔纹、屏幕边框、高光、纹理退化、半色调网点、纸面噪声、色偏等）。
- 双分支判定：可解释信号分析（纯 numpy）+ 特征分类器（逻辑回归，随仓库发布、可再训练）；预留 ONNX CNN 接口。
- 关键字段 OCR（RapidOCR/PP-OCR，可选安装）与 GB 11643-1999 校验位、性别/出生交叉校验；敏感信息默认打码。
- Web 应用：识别 / 批量 / 历史 / 复核 / 模型 / 设置 六个页面。
- 再训练闭环：标注数据 → train_recapture.py → evaluate.py 阈值校准 → export_and_swap.py 热替换与回滚。

## 快速开始

### Windows

    setup.bat   REM 创建 venv 并安装依赖、补齐模型
    run.bat     REM 启动服务并访问 http://127.0.0.1:8000

### 直接使用项目 venv / uv

    uv pip install -r requirements.txt
    uv run run.py

### Linux / macOS

    ./setup.sh
    ./run.sh

### 零依赖启动（仅需 numpy + Pillow）

若环境中已有 numpy 与 Pillow（未安装 FastAPI/OpenCV 亦可），可直接运行：

    python run.py

此时自动回退到纯标准库 HTTP 服务 + 纯 numpy/Pillow 推理，翻拍判定功能完整可用；OCR 与卡证检测（cv2 加速）在安装 requirements.txt 后进一步增强。

## 样例图片

仓库内置两张合成样例（`samples/idcard_original.jpg` 原件、`samples/idcard_screen.jpg` 屏幕翻拍），可直接拖入页面验证。

## 依赖

| 文件 | 内容 |
| --- | --- |
| requirements.txt | 核心推理（numpy、opencv-python-headless、pillow）+ Web（fastapi、uvicorn）+ OCR（rapidocr、onnxruntime） |
| requirements-ocr.txt | 仅补装 OCR（已包含在 requirements.txt） |
| requirements-dev.txt | 训练/开发（torch、timm、ultralytics、pytest，可选） |

## 目录结构

见开发计划 §8，核心实现：

- app/core/：预处理、卡证定位、翻拍判定（signals/classifier）、OCR、字段解析、流水线。
- app/main.py：FastAPI 后端；app/run_stdlib.py：标准库兜底服务。
- app/web/：前端单页应用。
- tools/：合成、转换、训练、评估、导出热替换、模型引导。
- models/recapture/weights.json：随仓库发布的翻拍分类基线权重。

## 再训练

    # 1) 合成样本（可选）
    python tools/synthesize.py --src data/raw --out data/synthetic --type screen --n 2
    python tools/synthesize.py --src data/raw --out data/synthetic --type print --n 2

    # 2) 训练（按 data/recapture/{original,screen_recapture,print_recapture} 分类目录）
    python tools/train_recapture.py --data data/recapture --epochs 200 --out runs/recap_v1.1

    # 3) 评估 + 阈值校准（目标 FPR 5%）
    python tools/evaluate.py --data data/recapture --weights runs/recap_v1.1/weights.json --target-fpr 0.05 --save-config

    # 4) 导出并热替换（自动备份、更新 VERSION/METRICS）
    python tools/export_and_swap.py --weights runs/recap_v1.1/weights.json

详细见 docs/RETRAIN.md。

## 隐私与合规

- 全流程本地推理，无云端调用；原图默认仅驻留内存（可在 config/settings.yaml 中配置）。
- 身份证号、姓名默认打码展示与落库；日志不记录完整身份证号。
- 仅用于授权业务场景（自有证件核验、内部质检），使用方须遵守《个人信息保护法》等法规。

## 文档

- docs/DEPLOY.md：部署与离线发行
- docs/RETRAIN.md：再训练闭环
- docs/API.md：接口说明
- 身份证翻拍识别工具开发计划.md：完整开发计划