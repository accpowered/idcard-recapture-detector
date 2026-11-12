"""标准库兜底服务：路由一致性与 multipart 多文件解析测试。"""
import json
import os
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.run_stdlib import Handler, ctx, first_field, parse_multipart  # noqa: E402


def _start_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, "http://127.0.0.1:{}".format(port)


def _get(url):
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_status(url):
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _post(url, payload):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def test_multipart_collects_multiple_files():
    boundary = "----dshtest"
    chunks = [
        ("--" + boundary + "\r\nContent-Disposition: form-data; name=\"class\"\r\n\r\n"
         "original\r\n").encode("utf-8"),
    ]
    for name, content in (("a.jpg", b"aaa"), ("b.jpg", b"bbb")):
        chunks.append((
            "--" + boundary + "\r\nContent-Disposition: form-data; name=\"files\"; "
            "filename=\"" + name + "\"\r\nContent-Type: image/jpeg\r\n\r\n"
        ).encode("utf-8") + content + b"\r\n")
    body = b"".join(chunks) + ("--" + boundary + "--\r\n").encode("utf-8")
    fields = parse_multipart(body, "multipart/form-data; boundary=" + boundary)
    assert len(fields["files"]) == 2
    assert [item[0] for item in fields["files"]] == ["a.jpg", "b.jpg"]
    assert first_field(fields, "class") == "original"


def test_stdlib_routes():
    override = ctx.settings_store.override_path
    original = None
    if os.path.exists(override):
        with open(override, "rb") as f:
            original = f.read()
    server = None
    try:
        server, base = _start_server()
        health = _get(base + "/api/health")
        assert health["code"] == 0
        assert health["version"] == "1.1.0"

        models = _get(base + "/api/models")
        assert models["code"] == 0
        assert any(item["version"] == "1.0.0" for item in models["versions"])

        settings = _get(base + "/api/settings")
        assert settings["code"] == 0
        assert "editable" in settings

        history = _get(base + "/api/history")
        assert history["code"] == 0
        assert isinstance(history["items"], list)

        status, body = _get_status(base + "/api/batch/not-a-real-task")
        assert status == 404 and body["code"] == 404

        status, body = _post(base + "/api/models/activate",
                             {"version": "definitely-missing"})
        assert status == 404 and body["code"] == 404

        status, body = _post(base + "/api/settings", {"ui": {"show_training": True}})
        assert status == 200 and body["code"] == 0

        status, body = _post(base + "/api/settings", {"recapture": {"mode": "bogus"}})
        assert status == 400 and body["code"] == 400
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        if original is None:
            if os.path.exists(override):
                os.remove(override)
        else:
            with open(override, "wb") as f:
                f.write(original)
        try:
            ctx.settings_store.reload()
        except Exception:  # noqa: BLE001
            pass
