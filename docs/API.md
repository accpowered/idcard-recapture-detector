# API 说明

Base URL：http://127.0.0.1:8000

约定：成功响应为 `{"code":0, ...}`；业务失败时 HTTP 状态码与 `code` 一致，并带 `error` 文案
（个别历史接口为兼容保留 HTTP 200 + 非 0 code，见下）。

## 识别

### POST /api/analyze

multipart：`file`（必填）、`model_version`（可选，`auto`/空=跟随当前版本，或具体版本号）。

    {
      "code": 0,
      "result": {
        "card_bbox": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]],
        "recapture": {
          "score": 0.86, "risk": "high", "class": "screen_recapture",
          "probs": {"original":0.08,"screen_recapture":0.85,"print_recapture":0.07},
          "classes": ["original","screen_recapture","print_recapture"],
          "model_version": "1.0.0", "mode": "screen_and_print",
          "signal_score": 0.42,
          "evidences": [{"name":"moire","score":0.72,"detail":"检测到约 45° 方向摩尔纹峰值"}]
        },
        "fields": {"name":"张*","id_number":"110101********1234","id_checksum_valid":true},
        "confidence": {"ocr":0.97,"recapture":0.88},
        "ocr": {"available":true,"lines":[{"text":"姓名 张三","score":0.99}]},
        "annotated_b64": "data:image/jpeg;base64,..."
      }
    }

- `mode=screen_only` 时 `probs` 只含 `original` / `screen_recapture`，不含打印类。
- 失败：未检测到身份证 / 版本不存在 → `code 404`；所选版本与当前识别模式不兼容 → `code 409`；
  图片解码失败 → `code 400`。

### POST /api/batch

multipart `files`（多文件）+ 可选 `model_version` → `{"code":0,"task_id":"..."}`。
`GET /api/batch/{task_id}` 轮询进度与结果，返回 `{"code":0, ...任务对象}`；任务不存在返回 404 + `{"code":404,...}`。

## 反馈（不变）

- POST /api/feedback：multipart `file` + `true_class`
- GET /api/feedback?status=pending
- POST /api/feedback/{id}/confirm（body `confirmed_class`）
- POST /api/feedback/{id}/reject

## 训练

### GET /api/train/dataset?data_dir=&mode=

返回数据集体检报告：

    {
      "code": 0, "mode": "screen_and_print", "mode_label": "屏幕+打印翻拍",
      "data_dir": "...", "data_dir_display": "data/recapture",
      "total": 12, "ready": true,
      "required_classes": ["original","screen_recapture"],
      "optional_classes": ["print_recapture"],
      "classes": [
        {"name":"original","label":"原件实拍","count":6,"min_required":5,
         "recommended":50,"status":"ok","recent_files":["a.jpg"]}
      ],
      "issues": [{"level":"warn","code":"optional_missing","message":"..."}],
      "guidance": ["最少：...", "建议：..."]
    }

`data_dir` 只允许仓库 `data/` 之下的路径，默认 `data/recapture`。

### POST /api/train/samples

multipart：`class`（original / screen_recapture / print_recapture，受当前模式限制）+ `files`（多文件）。

返回 `{"code":0,"saved":["..."],"skipped":[{"name":"...","reason":"..."}],"dataset":{...}}`。
单请求 ≤100 张、单张 ≤12MB；按内容哈希去重。

### POST /api/train/samples/delete

body `{"class":"original","name":"xxx.jpg"}` → 删除单个样本并返回最新 `dataset`。

### POST /api/train

    {"data_dir":"data/recapture","epochs":200,"lr":0.5,"limit_per_class":0,
     "target_fpr":0.05,"label":"demo","init_from_current":false,"auto_activate":true}

训练模式固定跟随全局设置。返回 `{"code":0,"job_id":"...","version":"demo-20261113-001230","job":{...}}`。

- 数据不满足要求：`400` + `issues`
- 已有训练任务：`409`

### GET /api/train/{job_id}

    {"code":0,"id":"...","status":"running|done|failed","stage":"extract|train|evaluate|promote",
     "progress":72,"logs":["[12:00:01] ..."],"error":null,"issues":[],
     "params":{...},
     "result":{"version":"...","promoted":true,"gate":null,"previous_version":"1.0.0",
               "metrics":{"accuracy":0.94,"macro_f1":0.93,"auc_recapture":0.98,...},
               "thresholds":{"low":0.17,"high":0.34},"dataset":{...},"message":"..."}}

`gate` 取值：`null`（通过）、`below_min_f1`、`missing_class_recall`、`current_changed`、`mode_changed`。

## 模型

### GET /api/models

    {
      "code": 0,
      "current": {"version":"1.0.0","mode":"screen_and_print",
                  "classes":["original","screen_recapture","print_recapture"],
                  "compatible":true,"metrics":null,"weights_path":"...","source":"baseline"},
      "versions": [
        {"version":"1.0.0","active":true,"valid":true,"compatible":true,"reason":null,
         "created_at":1762905600,"source":"baseline","mode":"screen_and_print",
         "classes":["..."],"metrics":null,"dataset":null,"note":""}
      ],
      "modes": {"current":"screen_and_print","labels":{"screen_only":"仅屏幕翻拍","screen_and_print":"屏幕+打印翻拍"}},
      "warnings": []
    }

> v1.1 变更：`versions` 由字符串数组改为版本对象数组。

### POST /api/models/activate

body `{"version":"train-..."}` → `{"code":0,"current":{...}}`。启用后立即生效（无需重启）。
未知版本 `404`；与当前模式不兼容 `409`。

### POST /api/models/rollback

与 `/api/models/activate` 等价（保留旧路径）。

## 设置

### GET /api/settings

    {
      "code": 0,
      "editable": {"ui":{"show_training":true},
                   "recapture":{"mode":"screen_and_print"},
                   "training":{"keep_versions":10,"target_fpr":0.05}},
      "effective": { "...": "完整合并后的配置" },
      "sources": {"base":"config/settings.yaml","override":"config/settings.local.json"},
      "model": {"current_version":"1.0.0","compatible":true,"mode":"screen_and_print"},
      "warnings": []
    }

### POST /api/settings

body 为白名单补丁，例如：

    {"ui":{"show_training":false},
     "recapture":{"mode":"screen_only"},
     "training":{"keep_versions":10,"target_fpr":0.05}}

保存后立即生效，写入 `config/settings.local.json`（本地、不入库）。
切换模式时若当前版本不兼容，会自动启用最新的兼容版本并返回 `switched_to`；
若没有任何兼容版本，则允许切换但识别请求返回 409，需先训练/导入兼容模型。
非法键或值返回 `400`。

## 健康

`GET /api/health` 返回服务状态、OCR 安装情况、当前模型/模式与训练开关。

## 历史

- GET /api/history?limit=&offset=：`summary` 含 `class/risk/score/model_version/mode/name/id_number`
- GET /api/history/{id}：详情
