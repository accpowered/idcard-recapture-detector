"""FastAPI 入口（生产/标准运行）。

启动：uvicorn app.main:app --host 127.0.0.1 --port 8000
或直接：python -m app.main
"""
from __future__ import annotations

import os
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .app_context import AppContext
from .core.config import ROOT

ctx = AppContext()

app = FastAPI(title="居民身份证翻拍识别工具", version="1.0.0")


@app.get("/api/health")
def health():
    try:
        ocr_available = bool(ctx.pipeline._ensure_ocr().available)
    except Exception:  # noqa: BLE001
        ocr_available = False
    return {"status": "ok", "ocr_available": ocr_available, "version": "1.0.0"}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    data = await file.read()
    res = ctx.analyze(data, file.filename or "upload.jpg")
    return JSONResponse(res)


@app.post("/api/batch")
async def batch(files: List[UploadFile] = File(...)):
    items = []
    for f in files:
        data = await f.read()
        items.append((f.filename or "image.jpg", data))
    tid = ctx.batch(items)
    return {"code": 0, "task_id": tid}


@app.get("/api/batch/{task_id}")
async def batch_status(task_id: str):
    return ctx.batch_status(task_id)


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


class TrainParams(BaseModel):
    data_dir: str = "data/recapture"
    epochs: int = 60
    out: str = "runs/recap_web"
    limit_per_class: int = 0
    class_weights: Optional[dict] = None


@app.post("/api/train")
async def train(body: TrainParams):
    tid = ctx.start_train(body.model_dump())
    return {"code": 0, "job_id": tid}


@app.get("/api/train/{job_id}")
async def train_status(job_id: str):
    return ctx.train_status(job_id)


@app.get("/api/models")
async def models():
    return {"code": 0, "current": ctx.model_manager.current(), "versions": ctx.model_manager.list_versions()}


class RollbackBody(BaseModel):
    version: str


@app.post("/api/models/rollback")
async def rollback(body: RollbackBody):
    ok = ctx.model_manager.rollback(body.version)
    return {"code": 0 if ok else 1, "ok": ok}


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
