"""Environment check — run this first.

Reports the Python / numpy / OpenCV / torch / CUDA state. The CPU-only parts of
this folder (metrics, video_io, pipeline, the tests) need just numpy + OpenCV;
torch + a GPU are needed only to run actual models. The thesis targets a single
NVIDIA T4, so this also warns if a different GPU is attached.
"""
from __future__ import annotations

import sys


def main():
    print("python      :", sys.version.split()[0])

    try:
        import numpy
        print("numpy       :", numpy.__version__)
    except ImportError:
        print("numpy       : MISSING  (required)")

    try:
        import cv2
        print("opencv      :", cv2.__version__)
    except ImportError:
        print("opencv      : MISSING  (required)")

    try:
        import torch
        print("torch       :", torch.__version__)
        cuda = torch.cuda.is_available()
        print("cuda        :", cuda)
        if cuda:
            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"gpu         : {name}  ({total:.1f} GB)")
            if "T4" not in name:
                print("  WARNING   : not a T4 — the thesis targets a single T4; "
                      "throughput/VRAM numbers will not transfer.")
    except ImportError:
        print("torch       : not installed  (needed only for model inference, "
              "not for the CPU tests)")

    print("\nOK — CPU pipeline + tests can run if numpy and opencv are present.")


if __name__ == "__main__":
    main()
