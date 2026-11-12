# 再训练闭环

翻拍分类器 v1.0 为「信号特征 + 逻辑回归」，无需 GPU，CPU 即可再训练。
如需 CNN 路线（EfficientNet-B0），安装 requirements-dev.txt 后使用 torch+timm，
导出 ONNX 放入 models/recapture/ 即可（预留接口）。

## 数据准备

按类别目录放置图片：

    data/recapture/
    ├── original/           # 原件实拍
    ├── screen_recapture/   # 屏幕翻拍
    └── print_recapture/    # 打印翻拍

标注规范：直接按目录归类即可；LabelMe 多边形标注可用 tools/convert_labels.py 转换。

## 数据合成（可选，扩充样本）

    python tools/synthesize.py --src data/raw --out data/synthetic --type screen --n 3
    python tools/synthesize.py --src data/raw --out data/synthetic --type print --n 3

将合成样本并入对应类别目录后参与训练。

## 训练

    python tools/train_recapture.py         --data data/recapture         --epochs 200         --out runs/recap_v1.1

支持：--resume 增量训练（预留）、--class-weights 应对样本不均衡、--limit-per-class 限流。

## 评估与阈值校准

    python tools/evaluate.py         --data data/recapture         --weights runs/recap_v1.1/weights.json         --target-fpr 0.05         --save-config

输出 Acc / macro-F1 / 各类 Precision/Recall/F1 / 混淆矩阵 / AUC / EER，
并按目标 FPR（默认 5%）把「高风险阈值」写入 weights.json。

## 导出与热替换 / 回滚

    python tools/export_and_swap.py --weights runs/recap_v1.1/weights.json   # 上线
    python tools/export_and_swap.py --list                                   # 查看版本
    python tools/export_and_swap.py --rollback 1.0.0                         # 回滚

上线时自动备份当前版本至 models/recapture/versions/，仅保留最近 3 版。

## Web 反馈闭环

1. 用户在历史/识别结果点击「判错」，选择真实类别；
2. 样本进入 data/feedback/ 待复核队列；
3. 复核页人工确认 → 自动并入 data/recapture/<class>/；
4. 达到阈值数量（默认 50）后在「模型」页一键触发再训练。
