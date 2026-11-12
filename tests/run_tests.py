"""轻量测试运行器（无需 pytest），也可用 pytest 运行。

用法: python tests/run_tests.py
"""
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
_vendor = os.path.join(ROOT, "vendor")
if os.path.isdir(_vendor):
    sys.path.insert(0, _vendor)

MODULES = ["tests.test_fields", "tests.test_signals", "tests.test_pipeline"]


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
