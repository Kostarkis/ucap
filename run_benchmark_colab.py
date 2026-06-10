"""End-to-end benchmark runner for the egocentric-anonymization study (Colab T4).

Runs the **headline paradigm comparison** — Paradigm A (specialist zoo:
``backends.EgoBlurDetector`` + ``backends.PaddleOCRDetector``) vs Paradigm B
(``pipeline.Sam3UltralyticsSegmenter``) — on one annotated eval subset and one
T4, reporting per category: AP@0.5, precision, recall, recall@P95, FPS, peak
VRAM (Layer 1) **and** the temporal recall / leak-event table (the supporting
metric, `planning/12` §3). Use the same call on a CogniCap clip and on the
public ``xperience-10m-sample`` (the supervisor's public-dataset validation).

------------------------------------------------------------------------------
Colab quick start (T4 runtime; HF token in a Secret named HF_TOKEN for SAM 3):

    !pip install numpy opencv-python-headless torch torchvision \
                 paddlepaddle-gpu paddleocr supervision ultralytics
    # download EgoBlur ego_blur_face.jit / ego_blur_lp.jit from the EgoBlur repo
    !python run_benchmark_colab.py --data anno.json --egoblur-face ego_blur_face.jit

Annotation JSON schema (produced on Day 2 — semi-auto pre-label then correct):

    {
      "fps": 30,
      "root": "frames",                       # optional base dir for "image"
      "frames": [
        {"image": "clip01/000001.jpg",
         "boxes": {"face": [[x1,y1,x2,y2], ...], "text": []}},
        ...                                    # one entry per sampled frame
      ]
    }

A category is *present* in a frame iff its box list is non-empty — that presence
timeline drives temporal recall, so it is cheap to annotate (scrub + mark spans)
relative to exhaustive boxes. Predicted-label strings must match the category
keys (alias with ``--label-map`` if a SAM 3 prompt differs, e.g. document=text).
------------------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
import json
import os

import cv2

import bench_harness
import metrics
import temporal


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_dataset(path):
    """Return ``(frames_bgr, annotations, fps, categories)`` from an annotation
    JSON (schema in the module docstring)."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    root = data.get("root", os.path.dirname(os.path.abspath(path)))
    fps = float(data.get("fps", 30))
    frames, ann, categories = [], [], set()
    for entry in data["frames"]:
        img = cv2.imread(os.path.join(root, entry["image"]))
        if img is None:
            raise FileNotFoundError(entry["image"])
        frames.append(img)
        boxes = {k: [tuple(b) for b in v] for k, v in entry.get("boxes", {}).items()}
        ann.append(boxes)
        categories.update(boxes.keys())
    return frames, ann, fps, sorted(categories)


# --------------------------------------------------------------------------
# Benchmark one detector across all categories it can emit
# --------------------------------------------------------------------------
def benchmark_detector(detector, frames, ann, categories, name, fps_capture,
                       ks=(0, 2, 5, 10), iou_thr=0.5, label_map=None):
    """Time one detection pass, then score every category from those predictions.

    Returns ``{"fps", "peak_vram_gb", "layer1": {cat: BenchResult},
    "temporal": {cat: [TemporalResult]}}``. ``label_map`` aliases predicted
    labels onto category keys (e.g. ``{"document": "text"}``).
    """
    import time

    label_map = label_map or {}
    metrics.reset_vram()
    t0 = time.perf_counter()
    preds_per_frame = [detector.detect(f) for f in frames]
    elapsed = time.perf_counter() - t0
    fps = len(frames) / elapsed if elapsed > 0 else float("inf")
    vram = metrics.peak_vram_gb()

    out = {"fps": fps, "peak_vram_gb": vram, "layer1": {}, "temporal": {}}
    for cat in categories:
        gt_cat = [a.get(cat, []) for a in ann]
        preds_cat = [
            [r for r in preds if label_map.get(r.label, r.label) == cat]
            for preds in preds_per_frame
        ]
        if not any(len(g) for g in gt_cat):
            continue  # category not present in this clip's ground truth
        out["layer1"][cat] = bench_harness.score_from_preds(
            preds_cat, gt_cat, f"{name} · {cat}", fps=fps, iou_thr=iou_thr,
            peak_vram_gb=vram)
        present, fired = temporal.presence_and_hits(preds_cat, gt_cat, iou_thr)
        out["temporal"][cat] = temporal.temporal_sweep(present, fired, ks)

    print(f"\n### {name} — FPS {fps:.1f} (capture {fps_capture:.0f}), "
          f"peak VRAM {vram if vram is not None else 'n/a'} GB")
    if out["layer1"]:
        print("\n" + bench_harness.results_table(list(out["layer1"].values())))
    for cat, rows in out["temporal"].items():
        print(f"\nTemporal — {cat}:")
        print(temporal.temporal_table(rows, fps=fps_capture))
    return out


# --------------------------------------------------------------------------
# Example wiring (edit for your run)
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="annotation JSON")
    ap.add_argument("--egoblur-face", help="ego_blur_face.jit path")
    ap.add_argument("--egoblur-lp", help="ego_blur_lp.jit path")
    ap.add_argument("--paddleocr", action="store_true", help="run PaddleOCR text")
    ap.add_argument("--sam3", action="store_true",
                    help="run SAM 3 (Ultralytics; needs HF access to facebook/sam3)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    frames, ann, fps_capture, categories = load_dataset(args.data)
    print(f"Loaded {len(frames)} frames · categories {categories} · "
          f"capture {fps_capture:.0f} fps")

    results = {}

    # Paradigm A — specialist zoo.
    if args.egoblur_face or args.egoblur_lp:
        from backends import EgoBlurDetector
        det = EgoBlurDetector(face_model=args.egoblur_face,
                              lp_model=args.egoblur_lp, device=args.device)
        results["A:EgoBlur"] = benchmark_detector(
            det, frames, ann, categories, "Paradigm A · EgoBlur", fps_capture)

    if args.paddleocr:
        from backends import PaddleOCRDetector
        det = PaddleOCRDetector(use_gpu=(args.device == "cuda"))
        results["A:PaddleOCR"] = benchmark_detector(
            det, frames, ann, categories, "Paradigm A · PaddleOCR", fps_capture)

    # Paradigm B — unified promptable concept segmentation (SAM 3).
    if args.sam3:
        from pipeline import Sam3UltralyticsSegmenter
        prompts = categories  # one prompt per annotated category
        det = Sam3UltralyticsSegmenter(prompts=prompts)
        # SAM 3 prompt strings already equal the category keys here.
        results["B:SAM3"] = benchmark_detector(
            det, frames, ann, categories, "Paradigm B · SAM 3", fps_capture)

    print("\nDone. Paste the tables above into drafts/section_results.md and "
          "planning/06_sam3_exploration.md §3.")
    return results


if __name__ == "__main__":
    main()
