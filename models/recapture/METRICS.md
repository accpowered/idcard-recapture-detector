# 翻拍分类器指标

版本：1.0.0（基线，特征逻辑回归）

说明：v1.0 基线权重为「可解释信号特征 + 逻辑回归」，随仓库发布、开箱即用。
由于未引入受限许可的公开翻拍数据，基线采用手工校准系数；指标需用自采/合成数据集
运行 tools/evaluate.py 复现。接入自采+合成数据后运行 tools/train_recapture.py 即可获得
数据驱动权重，并由 tools/export_and_swap.py 在此文件写入 AUC/EER/Acc/F1 与混淆矩阵。

## 复现方式

    python tests/make_benchmark.py --out data/benchmark --per 20
    python tools/train_recapture.py --data data/benchmark --epochs 150 --out runs/recap_v1
    python tools/evaluate.py --data data/benchmark --weights runs/recap_v1/weights.json --target-fpr 0.05 --save-config
