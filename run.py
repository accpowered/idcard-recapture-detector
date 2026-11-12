"""一键启动入口：自动补齐模型，优先 FastAPI，回退标准库服务。

用法：python run.py [--host 127.0.0.1] [--port 8000] [--offline]

说明：优先使用当前环境（如项目 venv）中已安装的 FastAPI；
仅在当前环境缺少依赖时，才尝试本地 vendor/ 包，最后回退到纯标准库服务。
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _fastapi_importable() -> bool:
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
        from fastapi import FastAPI  # noqa: F401
        from pydantic import BaseModel  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _try_add_vendor() -> bool:
    """仅在当前环境不可用时尝试本地 vendor（需与当前解释器兼容）。"""
    vendor = os.path.join(HERE, "vendor")
    if not os.path.isdir(vendor) or vendor in sys.path:
        return False
    sys.path.insert(0, vendor)
    if _fastapi_importable():
        return True
    sys.path.remove(vendor)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    # 补齐模型目录与默认权重
    from tools.bootstrap_models import ensure_defaults
    ensure_defaults()

    from app.core.config import load_settings
    settings = load_settings()
    host = args.host or settings.get("app", {}).get("host", "127.0.0.1")
    port = args.port or int(settings.get("app", {}).get("port", 8000))

    ok = _fastapi_importable()
    if not ok:
        ok = _try_add_vendor()

    if ok:
        try:
            from app.main import app as fastapi_app
            import uvicorn
            print(f"[启动] 使用 FastAPI 启动：http://{host}:{port}")
            uvicorn.run(fastapi_app, host=host, port=port, log_level="info")
            return
        except Exception as exc:  # noqa: BLE001
            print(f"[启动] FastAPI 启动失败（{exc}），回退标准库服务")

    print(f"[启动] 使用标准库服务启动：http://{host}:{port}")
    from app.run_stdlib import run as run_std
    run_std(host, port)


if __name__ == "__main__":
    main()
