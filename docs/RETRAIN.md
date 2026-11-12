# 再训练闭环

v1.1 起推荐直接用 Web 完成「素材 → 训练 → 评估 → 上线 → 回退」，CLI 保留等价能力。
翻拍分类器为「信号特征 + 逻辑回归」，CPU 几秒即可训完，无需 GPU。

## 识别模式与类别契约

| 模式 | 配置值 | 训练类别 | 推理行为 |
| --- | --- | --- | --- |
| 仅屏幕翻拍 | `screen_only` | original + screen_recapture | 不输出打印类；三类模型会自动屏蔽打印项并重新归一化分数 |
| 屏幕翻拍 + 打印翻拍 | `screen_and_print`（默认） | original + screen_recapture + print_recapture | 三类融合；打印样本缺失时告警「打印件无法被正确识别」 |

- 训练模式始终跟随「设置 → 识别模式」，页面上只读展示。
- 2 类模型只能用于仅屏幕模式；3 类模型两种模式都可用。
- 同一张卡分别拍「原件」与「屏幕翻拍」（成对样本）更利于模型学到翻拍伪影差异；
  但 original 类**不要求**与翻拍样本是同一张卡，任意原件实拍都可以。

## Web 流程

1. **设置**页选择识别模式（可选：关闭「模型 → 再训练」区域的显示）。
2. **模型 → 再训练**：数据集体检显示每类数量、最低/建议门槛与缺类指引；
   可直接按类别上传素材，或从「复核」页确认样本并入 `data/recapture`。
3. 填写轮数/学习率/目标 FPR/版本标签（可选热启动），点击「开始训练」。
4. 后台依次执行：提取特征 → 训练 → 评估 + 阈值校准 → 生成版本。
5. 通过质量门时**自动切换**为新版本（无需重启）；未通过则保存为候选版本，可点「仍然启用此版本」。
6. 识别页 / 批量页可选「跟随当前版本」或具体版本；模型页可随时启用任意兼容版本（即回退）。

## 数据准备

按类别目录放图（目录名即标签，不需要画框或标字段）：

    data/recapture/
    ├── original/           # 原件实拍：对实体身份证直接拍摄
    ├── screen_recapture/   # 屏幕翻拍：对手机/显示器/平板显示的身份证再拍摄
    └── print_recapture/    # 打印翻拍：对打印件、复印件、彩印件拍摄（仅屏幕+打印模式使用）

样本要求（Web 与 CLI 一致）：

- **最低**：original 与 screen_recapture 各 ≥5 张、合计 ≥10 张；缺任一类无法训练。
- **建议**：每类 ≥50 张；正式使用 200–500 张，覆盖不同机型 / 光照 / 角度。
- screen_and_print 模式下 print_recapture 为强烈建议项：缺失只告警、不阻断，但打印件识别能力会很弱。
- 类别不均衡（最多/最少 > 3）会告警，建议补齐较少类，或用 CLI 的 --class-weights 调整。
- 合成样本（可选，命令在 v1.0 文档中保留）：

      python tools/synthesize.py --src data/raw --out data/recapture/screen_recapture --type screen --n 3
      python tools/synthesize.py --src data/raw --out data/recapture/print_recapture --type print --n 3

## 质量门与候选版本

- 门限：`macro_f1 >= training.min_macro_f1`（默认 0.5），且必备类别召回 > 0；
  同时要求「训练开始时的父版本仍是当前版本」且「识别模式未变」。
- 通过：版本写入 `models/recapture/versions/<version>/`，并自动启用。
- 未通过：同样生成版本与指标，但只保存为候选（不影响当前版本），原因会显示在页面与 job.result.gate：
  `below_min_f1` / `missing_class_recall` / `current_changed` / `mode_changed` / `auto_activate_disabled`。
- 注意：评估使用训练集自评，仅用于版本比较与阈值校准，不代表泛化性能。

## 版本管理与回退

- 版本号：`<label>-<YYYYMMDD-HHMMSS>`，label 默认 `train`。
- 每个版本包含 `weights.json` + `meta.json`（模式、类别、参数、数据量、指标、阈值、父版本）。
- `models/recapture/VERSION` 是当前版本指针；`weights.json` / `METRICS.md` 同步为当前版本内容。
- 保留策略：`training.keep_versions`（默认 10）个训练版本 + 当前版本 + 基线 `1.0.0`。
- 回退：识别页/模型页启用旧版本即可，立即生效；或在 CLI 执行 --rollback。
- 版本与模型兼容：屏幕+打印模式下只能启用含打印类的版本；仅屏幕模式可用 2 类或 3 类版本。

## CLI（等价能力）

    # 训练（默认三类；仅屏幕模式加 --mode screen_only）
    python tools/train_recapture.py --data data/recapture --epochs 200 --out runs/recap_v1.1
    python tools/train_recapture.py --data data/recapture --mode screen_only --out runs/recap_screen

    # 评估 + 阈值校准（目标 FPR 5%）
    python tools/evaluate.py --data data/recapture --weights runs/recap_v1.1/weights.json --target-fpr 0.05 --save-config

    # 导入/上线 / 回滚 / 列表
    python tools/export_and_swap.py --weights runs/recap_v1.1/weights.json
    python tools/export_and_swap.py --weights runs/recap_v1.1/weights.json --no-activate   # 只导入为候选
    python tools/export_and_swap.py --rollback 1.0.0
    python tools/export_and_swap.py --list

## 与 git 的关系

- 训练/启用会改写被 git 跟踪的 `models/recapture/weights.json`、`VERSION`、`METRICS.md`，
  这是「本地热切换」的预期行为。需要恢复出厂基线时：

      git checkout -- models/recapture/weights.json models/recapture/VERSION models/recapture/METRICS.md

- 自动生成的版本目录（`models/recapture/versions/*`，除基线 `1.0.0/`）与 `runs/` 已被 .gitignore 忽略。
- 训练素材与 `config/settings.local.json` 同样不入库。

## 常见问题

- **只有屏幕翻拍样本，没有对应原件**：不能训练出有意义的模型。请补任意身份证原件实拍（不要求同卡）。
- **「成对出现」是必须同卡吗**：不是。最低要求是 original 与 screen_recapture 两类都存在；同卡成对只是更优的数据组织方式。
- **训练后想回退**：模型页「启用」上一版本，或在 job 结果横幅点「回退到 <上一版本>」。
- **切换识别模式后识别报 409**：当前版本缺少新模式所需类别；页面会自动尝试启用兼容版本，若没有则需先训练/导入。
- **旧版权重没有 classes/mode 字段**：按三类、屏幕+打印模式兼容加载，行为与 v1.0 一致。
