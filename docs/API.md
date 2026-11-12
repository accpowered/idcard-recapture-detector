# API 说明

Base URL：http://127.0.0.1:8000

## POST /api/analyze

multipart 上传单张图片（字段名 file），同步返回完整结果。

    {
      "code": 0,
      "result": {
        "card_bbox": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]],
        "recapture": {
          "score": 0.86, "risk": "high", "class": "screen_recapture",
          "probs": {"original":0.08,"screen_recapture":0.85,"print_recapture":0.07},
          "signal_score": 0.42,
          "evidences": [{"name":"moire","score":0.72,"detail":"检测到约 45° 方向摩尔纹峰值"}]
        },
        "fields": {"name":"张*","gender":"男","ethnicity":"汉","birth":"19**年*月**日",
                   "address":"北京市***（打码）","id_number":"110101********1234",
                   "authority":"北京市公安局***分局","valid_period":"2020.01.01-2040.01.01",
                   "id_checksum_valid":true},
        "confidence": {"ocr":0.97,"recapture":0.88},
        "ocr": {"available":true,"lines":[{"text":"姓名 张三","score":0.99}]},
        "annotated_b64": "data:image/jpeg;base64,..."
      }
    }

失败时 code 非 0 并带 error（如未检测到身份证返回 404）。

## 批量

- POST /api/batch：multipart 多文件（字段名 files），返回 task_id
- GET /api/batch/{task_id}：轮询进度与结果

## 反馈

- POST /api/feedback：multipart 上传 file + true_class（原图进入复核队列）
- GET /api/feedback?status=pending：待复核列表
- POST /api/feedback/{id}/confirm：确认并并入训练集（body: confirmed_class）
- POST /api/feedback/{id}/reject：拒绝

## 训练

- POST /api/train：body { data_dir, epochs, out }
- GET /api/train/{job_id}：轮询状态（running/done/failed + weights_path）

## 模型

- GET /api/models：当前版本、指标、历史版本
- POST /api/models/rollback：body { version }

## 历史

- GET /api/history：脱敏历史列表
- GET /api/history/{id}：详情
