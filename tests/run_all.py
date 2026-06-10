"""Run every test in this folder. No pytest dependency.

    python run_all.py
"""
import importlib
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODULES = ["test_metrics", "test_video_io", "test_pipeline", "test_temporal"]


def main():
    passed = failed = 0
    for modname in MODULES:
        mod = importlib.import_module(modname)
        for name in sorted(dir(mod)):
            if not name.startswith("test_"):
                continue
            fn = getattr(mod, name)
            if not callable(fn):
                continue
            try:
                fn()
                print(f"  PASS  {modname}.{name}")
                passed += 1
            except Exception:
                print(f"  FAIL  {modname}.{name}")
                traceback.print_exc()
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
