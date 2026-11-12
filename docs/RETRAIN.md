# 再训练闭环

翻拍分类器 v1.0 为「信号特征 + 逻辑回归」，无需 GPU，CPU 即可再训练。
如需 CNN 路线（EfficientNet-B0），安装 requirements-dev.txt 后使用 torch+timm，
导出 ONNX 放入 models/recapture/ 即可（预留接口）。

## 数据准备

按类别目录放置图片（目录名即标签，**不需要画框、不需要标字段**）：

    data/recapture/
    ├── original/           # 原件实拍：对实体身份证直接拍摄
    ├── screen_recapture/   # 屏幕翻拍：对手机/显示器/平板显示的身份证再拍摄
    └── print_recapture/    # 打印翻拍：对打印件、复印件、彩印件拍摄

标注规范：
- 直接把图片拷进对应类别文件夹即可。
- LabelMe 多边形标注可用 tools/convert_labels.py 转换：
      python tools/convert_labels.py labelme --src data/labelme --out data/recapture

样本要求（重要）：
- 翻拍判定是**三类分类**，至少需要 original + 一种翻拍类别；**只有单类样本无法训练有意义的判别边界**（脚本会报错提示）。
- original 类**不需要**与翻拍样本是同一张卡——任意身份证原件的实拍照片都可作为「原件」负样本。
- 建议每类 ≥ 200–500 张，覆盖不同机型/光照/角度；三类尽量均衡。
- 不均衡时用 --class-weights 调整，例如：
      python tools/train_recapture.py --data data/recapture --class-weights original:0.5,screen_recapture:1.5,print_recapture:1

## 数据合成（可选，扩充样本）

    python tools/synthesize.py --src data/raw --out data/recapture/screen_recapture --type screen --n 3
    python tools/synthesize.py --src data/raw --out data/recapture/print_recapture --type print --n 3

把合成样本直接放进对应类别目录即可参与训练。

## 训练

    python tools/train_recapture.py --data data/recapture --epochs 200 --out runs/recap_v1.1

参数：
- --epochs          训练轮数（多轮训练直接调大即可）
- --limit-per-class 每类最多用多少张（快速试跑）
- --class-weights   样本不均衡加权
- --init-from       从已有 weights.json 热启动（增量/多轮训练）

增量/多轮训练示例（在上一版权重基础上继续）：

    python tools/train_recapture.py --data data/recapture --epochs 100 --out runs/recap_v1.2 --init-from models/recapture/weights.json

## 评估与阈值校准

    python tools/evaluate.py --data data/recapture --weights runs/recap_v1.1/weights.json --target-fpr 0.05 --save-config

输出 Acc / macro-F1 / 各类 Precision/Recall/F1 / 混淆矩阵 / AUC / EER，
并按目标 FPR（默认 5%）把「高风险阈值」写入 weights.json。

## 导出与热替换 / 回滚

    python tools/export_and_swap.py --weights runs/recap_v1.1/weights.json   # 上线（自动备份）
    python tools/export_and_swap.py --list                                   # 查看当前与历史版本
    python tools/export_and_swap.py --rollback 1.0.0                         # 回滚到指定版本

上线时自动把当前版本备份到 models/recapture/versions/<版本>/，仅保留最近 3 版；
回滚会恢复该版本的 weights.json 并更新 VERSION，服务热加载即可生效。

## Web 反馈闭环

1. 用户在历史/识别结果点击「判错」，选择真实类别；
2. 样本进入 data/feedback/ 待复核队列；
3. 复核页人工确认 → 自动并入 data/recapture/<class>/；
4. 达到阈值数量（默认 50）后在「模型」页一键触发再训练（或直接用上面的 CLI）。

## 常见问题

- **只有屏幕翻拍样本，能训练吗？** 不能得到有意义的模型：缺少 original 负样本，模型会把所有图判成翻拍。
  需要补任意原件实拍图作为 original 类（不要求同卡）。
- **能训练多轮吗？** 可以：调大 --epochs 继续训练，或用 --init-from 从上一版权重热启动；
  数据增加后每次重新训练、再 export_and_swap 上线即可。
- **训练能回退吗？** 可以：export_and_swap 会保留最近 3 个版本，用 --rollback <版本> 一键回退。
