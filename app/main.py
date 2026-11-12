"""FastAPI 入口（生产/标准运行）。

启动：uvicorn app.main:app --host 127.0.0.1 --port 8000
或直接：python -m app.main
"""
from __future__ import annotations

import os
from typing import List, Optional

from fastapi import Body, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import __version__
from .api.ops import ApiError, ApiOps
from .app_context import AppContext
from .core.config import ROOT

ctx = AppContext()
ops = ApiOps(ctx)

app = FastAPI(title="居民身份证翻拍识别工具", version=__version__)


@app.exception_handler(ApiError)
async def _handle_api_error(request: Request, exc: ApiError):  # noqa: ARG001
    payload = {"code": exc.code, "error": exc.message}
    payload.update(exc.payload)
    return JSONResponse(payload, status_code=exc.code)


# --------------------------------------------------------------------------- 健康
@app.get("/api/health")
def health():
    return ops.health()


# --------------------------------------------------------------------------- 识别
@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), model_version: str = Form("auto")):
    data = await file.read()
    return JSONResponse(ops.analyze(data, file.filename or "upload.jpg", model_version))


@app.post("/api/batch")
async def batch(files: List[UploadFile] = File(...), model_version: str = Form("auto")):
    items = []
    for f in files:
        items.append((f.filename or "image.jpg", await f.read()))
    return JSONResponse(ops.batch(items, model_version))


@app.get("/api/batch/{task_id}")
async def batch_status(task_id: str):
    return ops.batch_status(task_id)


# --------------------------------------------------------------------------- 反馈
@app.post("/api/feedback")
async def feedback(file: UploadFile = File(...), true_class: str = "screen_recapture", note: str = ""):
    data = await file.read()
    fid = ctx.submit_feedback(file.filename or "sample.jpg", true_class, data)
    return {"code": 0, "feedback_id": fid}


@app.get("/api/feedback")
async def list_feedback(status: str = "pending"):
    return {"code": 0, "items": ctx.storage.list_feedback(status)}


class FeedbackAction(BaseModel):
    confirmed_class: Optional[str] = None


@app.post("/api/feedback/{fid}/confirm")
async def confirm_feedback(fid: str, body: FeedbackAction):
    cls = body.confirmed_class or "screen_recapture"
    ok = ctx.storage.confirm_feedback(fid, cls, os.path.join(ROOT, "data/recapture"))
    return {"code": 0 if ok else 1, "ok": ok}


@app.post("/api/feedback/{fid}/reject")
async def reject_feedback(fid: str):
    return {"code": 0, "ok": ctx.storage.reject_feedback(fid)}


# --------------------------------------------------------------------------- 训练
@app.get("/api/train/dataset")
async def train_dataset(data_dir: Optional[str] = None, mode: Optional[str] = None):
    return ops.dataset(data_dir, mode)


@app.post("/api/train/samples")
async def train_samples(class_name: str = Form(..., alias="class"),
                        files: List[UploadFile] = File(...)):
    items = []
    for f in files:
        items.append((f.filename or "sample.jpg", await f.read()))
    return ops.upload_samples(class_name, items)


@app.post("/api/train/samples/delete")
async def train_samples_delete(body: dict = Body(...)):
    return ops.delete_sample(body.get("class"), body.get("name"))


@app.post("/api/train")
async def train(body: dict = Body(...)):
    return ops.train_start(body)


@app.get("/api/train/{job_id}")
async def train_status(job_id: str):
    return ops.train_status(job_id)


# --------------------------------------------------------------------------- 模型
@app.get("/api/models")
async def models():
    return ops.models()


class VersionBody(BaseModel):
    version: str


@app.post("/api/models/activate")
async def activate(body: VersionBody):
    return ops.activate(body.version)


@app.post("/api/models/rollback")
async def rollback(body: VersionBody):
    return ops.activate(body.version)


# --------------------------------------------------------------------------- 设置
@app.get("/api/settings")
async def get_settings():
    return ops.settings_get()


@app.post("/api/settings")
async def save_settings(body: dict = Body(...)):
    return ops.settings_save(body)


# --------------------------------------------------------------------------- 历史
@app.get("/api/history")
async def history(limit: int = 100, offset: int = 0):
    return {"code": 0, "items": ctx.storage.list_history(limit, offset)}


@app.get("/api/history/{rid}")
async def history_detail(rid: str):
    item = ctx.storage.get_history(rid)
    if item is None:
        return JSONResponse({"code": 404, "error": "not found"}, status_code=404)
    return {"code": 0, "result": item}


# 静态前端
WEB_DIR = os.path.join(ROOT, "app", "web")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


def run():
    import uvicorn
    host = ctx.settings.get("app", {}).get("host", "127.0.0.1")
    port = int(ctx.settings.get("app", {}).get("port", 8000))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run()
