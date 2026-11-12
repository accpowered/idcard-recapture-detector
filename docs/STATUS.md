# 实现状态与落地说明

本文档对照《居民身份证翻拍识别工具开发计划》逐项说明已落地内容、运行方式与生产化建议。

## 已落地（可运行、已自测）

| 需求 | 状态 | 实现位置 |
| --- | --- | --- |
| 图片上传（JPG/PNG/BMP/WebP，HEIC 需 pillow-heif） | ✅ | app/core/preprocess.py、app/web |
| 卡证定位（四角 + 透视矫正，cv2 优先 / numpy 兜底） | ✅ | app/core/detect.py |
| 翻拍判定（三类概率 + 分数 + 风险等级） | ✅ | app/core/recapture/classifier.py |
| 可解释线索（摩尔纹、屏幕栅格、黑边、高光、纹理退化、半色调、纸噪、饱和度、色偏） | ✅ | app/core/recapture/signals.py（9 项证据） |
| 关键字段 OCR（RapidOCR PP-OCRv6 中文，CPU） | ✅ 实测可用 | app/core/ocr_engine.py |
| 字段解析与 GB11643 校验位、性别/出生交叉校验 | ✅ | app/core/fields.py |
| 结果可视化（卡框标注、风险徽章、概率条、线索、字段表格、JSON 导出） | ✅ | app/web |
| 批量处理（线程池队列 + 进度 + CSV/JSON 导出） | ✅ | app/services/queue.py |
| 历史记录（SQLite 脱敏） | ✅ | app/services/storage.py |
| 误判反馈 → 复核 → 并入训练集 | ✅ | app/services/storage.py、app/web |
| 再训练闭环（train → evaluate → export/热替换 → 回滚） | ✅ 已跑通 | tools/*.py |
| 模型管理（版本、指标、回滚） | ✅ | app/services/model_manager.py |

测试：python tests/run_tests.py（11 项全通过）或 pytest tests/。

## 启动方式

1. 标准（推荐）：Windows 运行 setup.bat 后 run.bat；Linux 运行 ./setup.sh 后 ./run.sh。
2. 零依赖兜底：环境已有 numpy + Pillow 时直接 python run.py（自动回退标准库 HTTP 服务）。

服务默认 http://127.0.0.1:8000，提供识别/批量/历史/复核/模型/设置六个页面与 /api/* 接口。

## 与计划的技术取舍（生产化建议）

1. 翻拍分类基线
   - 计划：EfficientNet-B0 / PP-LCNet CNN 三分类，随仓库发布 ONNX 权重。
   - 现状：本仓库发布「信号特征 + 逻辑回归」基线（models/recapture/weights.json，
     纯 numpy 推理、开箱即用、可解释、可再训练）。这是计划 §4.3「分类不可用时仅信号分支」的
     稳健落地形态，且训练闭环已完全打通。
   - 生产化：用真实自采+合成数据运行 tools/train_recapture.py 与 tools/evaluate.py
     校准后 tools/export_and_swap.py 上线；如需 CNN，安装 requirements-dev.txt（torch+timm）
     训练 EfficientNet-B0 并导出 ONNX，接口已预留。

2. 卡证检测
   - 计划：YOLOv8n 微调 ONNX。
   - 现状：OpenCV 轮廓+四边形+透视矫正（CPU 毫秒级），numpy Hough 兜底。
   - 生产化：用 data/idcard_det（YOLO 格式，tools/convert_labels.py 可转换 LabelMe）训练
     YOLOv8n，导出 ONNX 替换 detect 中的检测分支。

3. OCR
   - 现状：app/core/ocr_engine.py 封装 RapidOCR（3.x，PP-OCRv6 中文 det/rec ONNX，CNN+ONNX Runtime CPU），
     实测可识别卡面全部文字并完成字段解析与 GB11643 校验；未安装依赖时自动降级并标注 ocr.available=false。
   - 依赖：pip install -r requirements-ocr.txt（rapidocr + onnxruntime）；模型首次运行自动下载到
     vendor/rapidocr/models（约 20MB）。

4. 模型完整性
   - models/checksums.json 支持 SHA256 校验；随仓库分发的特征权重已入库。
   - 官方 ONNX 权重按计划 §5.2 以 LFS/引导下载/离线包分发，未随本次交付附带大文件。

## 隐私与合规

- 全流程本地推理，无云端调用；原图默认仅驻留内存（config/settings.yaml privacy.store_original）。
- 身份证号、姓名默认打码展示与落库。
- 仅用于授权业务场景，使用方须遵守《个人信息保护法》等法规。