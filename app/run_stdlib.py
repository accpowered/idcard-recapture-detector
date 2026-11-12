"""零依赖 HTTP 服务（纯标准库），用于无 FastAPI 环境下的开箱即用。

启动：python app/run_stdlib.py [--host 127.0.0.1] [--port 8000]
"""
from __future__ import annotations

import json
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import __version__  # noqa: E402
from app.api.ops import ApiError, ApiOps  # noqa: E402
from app.app_context import AppContext  # noqa: E402
from app.core.config import ROOT  # noqa: E402

ctx = AppContext()
ops = ApiOps(ctx)
WEB_DIR = os.path.join(ROOT, "app", "web")

_MIME = {
    ".html": "text/html; charset=utf-8", ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8", ".json": "application/json; charset=utf-8",
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
}


def parse_multipart(body: bytes, content_type: str):
    """解析 multipart，同名字段收集为列表（修复多文件互相覆盖）。"""
    m = re.search(r"boundary=(.+)$", content_type or "")
    if not m:
        return {}
    boundary = m.group(1).strip().strip('"').encode()
    parts = body.split(b"--" + boundary)
    fields = {}
    for part in parts:
        if b"Content-Disposition" not in part:
            continue
        header, _, content = part.partition(b"\r\n\r\n")
        content = content.rstrip(b"\r\n")
        hdr = header.decode("utf-8", "ignore")
        dm = re.search(r'name="([^"]+)"', hdr)
        fm = re.search(r'filename="([^"]+)"', hdr)
        if not dm:
            continue
        name = dm.group(1)
        value = (fm.group(1), content) if fm else content.decode("utf-8", "ignore")
        fields.setdefault(name, []).append(value)
    return fields


def first_field(fields, name, default=None):
    values = fields.get(name) or []
    return values[0] if values else default


def file_items(fields, name):
    return [v for v in (fields.get(name) or []) if isinstance(v, tuple)]


class Handler(BaseHTTPRequestHandler):
    server_version = "IDReshootCheck/1.1"

    def _send(self, code, obj, ctype="application/json; charset=utf-8"):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8") if not isinstance(obj, bytes) else obj
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _route(self, method, path, body=b""):
        u = urlparse(path)
        p = u.path
        qs = parse_qs(u.query)
        try:
            if p == "/api/health":
                return 200, ops.health()
            if p == "/api/analyze" and method == "POST":
                files = parse_multipart(body, self.headers.get("Content-Type", ""))
                items = file_items(files, "file")
                if not items:
                    return 400, {"code": 400, "error": "缺少 file"}
                name, data = items[0]
                return 200, ops.analyze(data, name, first_field(files, "model_version", "auto"))
            if p == "/api/batch" and method == "POST":
                files = parse_multipart(body, self.headers.get("Content-Type", ""))
                items = [(name, data) for name, data in file_items(files, "files")]
                return 200, ops.batch(items, first_field(files, "model_version", "auto"))
            if p.startswith("/api/batch/") and method == "GET":
                return 200, ops.batch_status(p.rsplit("/", 1)[1])
            if p == "/api/feedback" and method == "POST":
                files = parse_multipart(body, self.headers.get("Content-Type", ""))
                items = file_items(files, "file")
                if not items:
                    return 400, {"code": 400, "error": "缺少 file"}
                name, data = items[0]
                true_class = first_field(files, "true_class", "screen_recapture")
                fid = ctx.submit_feedback(name, true_class, data)
                return 200, {"code": 0, "feedback_id": fid}
            if p == "/api/feedback" and method == "GET":
                return 200, {"code": 0, "items": ctx.storage.list_feedback(qs.get("status", ["pending"])[0])}
            m = re.match(r"^/api/feedback/([^/]+)/confirm$", p)
            if m and method == "POST":
                cls = json.loads(body.decode() or "{}").get("confirmed_class", "screen_recapture")
                ok = ctx.storage.confirm_feedback(m.group(1), cls, os.path.join(ROOT, "data/recapture"))
                return 200, {"code": 0 if ok else 1, "ok": ok}
            m = re.match(r"^/api/feedback/([^/]+)/reject$", p)
            if m and method == "POST":
                return 200, {"code": 0, "ok": ctx.storage.reject_feedback(m.group(1))}
            if p == "/api/train/dataset" and method == "GET":
                return 200, ops.dataset(qs.get("data_dir", [None])[0], qs.get("mode", [None])[0])
            if p == "/api/train/samples" and method == "POST":
                files = parse_multipart(body, self.headers.get("Content-Type", ""))
                class_name = first_field(files, "class")
                return 200, ops.upload_samples(class_name, file_items(files, "files"))
            if p == "/api/train/samples/delete" and method == "POST":
                payload = json.loads(body.decode() or "{}")
                return 200, ops.delete_sample(payload.get("class"), payload.get("name"))
            if p == "/api/train" and method == "POST":
                return 200, ops.train_start(json.loads(body.decode() or "{}"))
            if p.startswith("/api/train/") and method == "GET":
                return 200, ops.train_status(p.rsplit("/", 1)[1])
            if p == "/api/models" and method == "GET":
                return 200, ops.models()
            if p == "/api/models/activate" and method == "POST":
                payload = json.loads(body.decode() or "{}")
                return 200, ops.activate(payload.get("version"))
            if p == "/api/models/rollback" and method == "POST":
                payload = json.loads(body.decode() or "{}")
                return 200, ops.activate(payload.get("version"))
            if p == "/api/settings" and method == "GET":
                return 200, ops.settings_get()
            if p == "/api/settings" and method == "POST":
                return 200, ops.settings_save(json.loads(body.decode() or "{}"))
            if p == "/api/history" and method == "GET":
                return 200, {"code": 0, "items": ctx.storage.list_history()}
            if p.startswith("/api/history/") and method == "GET":
                item = ctx.storage.get_history(p.rsplit("/", 1)[1])
                return (404, {"code": 404, "error": "not found"}) if item is None else (200, {"code": 0, "result": item})
            # 静态资源：/static/xxx → app/web/xxx；/ 或未知路由 → index.html
            rel = p.lstrip("/")
            if rel.startswith("static/"):
                rel = rel[len("static/"):]
            if rel in ("", "index.html"):
                return 200, open(os.path.join(WEB_DIR, "index.html"), "rb").read(), "text/html; charset=utf-8"
            fp = os.path.normpath(os.path.join(WEB_DIR, rel))
            if os.path.commonpath([WEB_DIR, fp]) != WEB_DIR:
                return 404, {"error": "forbidden"}
            if os.path.isfile(fp):
                ext = os.path.splitext(fp)[1].lower()
                return 200, open(fp, "rb").read(), _MIME.get(ext, "application/octet-stream")
            if rel.startswith("static/"):
                return 404, {"error": "not found"}
            return 200, open(os.path.join(WEB_DIR, "index.html"), "rb").read(), "text/html; charset=utf-8"
        except ApiError as exc:
            payload = {"code": exc.code, "error": exc.message}
            payload.update(exc.payload)
            return exc.code, payload
        except Exception as exc:  # noqa: BLE001
            return 500, {"code": 500, "error": str(exc)}

    def do_GET(self):
        code, obj, *ct = self._route("GET", self.path)
        self._send(code, obj, ct[0] if ct else "application/json; charset=utf-8")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        code, obj, *ct = self._route("POST", self.path, body)
        self._send(code, obj, ct[0] if ct else "application/json; charset=utf-8")

    def log_message(self, fmt, *args):  # 安静日志
        pass


def run(host="127.0.0.1", port=8000):
    srv = ThreadingHTTPServer((host, port), Handler)
    print("服务已启动：http://{}:{}（版本 {}）".format(host, port, __version__))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    run(args.host, args.port)
