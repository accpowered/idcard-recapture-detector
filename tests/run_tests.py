"""轻量测试运行器（无需 pytest），也可用 pytest 运行。

用法: python tests/run_tests.py
"""
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
# vendor 仅作兜底：放到 sys.path 末尾，优先使用当前环境已安装的依赖
_vendor = os.path.join(ROOT, "vendor")
if os.path.isdir(_vendor):
    sys.path.append(_vendor)

MODULES = [
    "tests.test_fields",
    "tests.test_signals",
    "tests.test_recapture_modes",
    "tests.test_model_manager",
    "tests.test_settings_store",
    "tests.test_storage",
    "tests.test_train_dataset",
    "tests.test_pipeline",
    "tests.test_stdlib_api",
    "tests.test_train_flow",
]


def discover(mod_name):
    mod = __import__(mod_name, fromlist=["x"])
    return [(n, getattr(mod, n)) for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n))]


def main():
    passed = failed = 0
    for mod in MODULES:
        for name, fn in discover(mod):
            try:
                fn()
                print(f"PASS {mod}.{name}")
                passed += 1
            except Exception:  # noqa: BLE001
                print(f"FAIL {mod}.{name}")
                traceback.print_exc()
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
