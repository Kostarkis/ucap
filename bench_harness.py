"""Generic per-model benchmark runner.

Wraps any ``Detector`` (a Paradigm-A specialist model or a Paradigm-B
promptable backend) behind one uniform measurement: detection accuracy
(AP / precision / recall) against ground truth, throughput (FPS), and peak
VRAM. Every candidate model is scored the same way on the same egocentric
evaluation subset, so the results tables are directly comparable.

The benchmark *protocol* (how the eval subset is built and annotated) lives in
``planning/07_benchmark_methodology.md``; this file is just the runner.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import metrics
import temporal


@dataclass
class BenchResult:
    name: str
    ap: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    recall_at_p95: float = 0.0
    fps: float = 0.0
    peak_vram_gb: Optional[float] = None
    n_frames: int = 0

    def as_markdown_row(self):
        vram = f"{self.peak_vram_gb:.2f}" if self.peak_vram_gb is not None else "n/a"
        return (
            f"| {self.name} | {self.ap:.3f} | {self.precision:.3f} | "
            f"{self.recall:.3f} | {self.recall_at_p95:.3f} | {self.fps:.1f} | {vram} |"
        )


MARKDOWN_HEADER = (
    "| Model | AP@0.5 | Precision | Recall | Recall@P95 | FPS (T4) | Peak VRAM (GB) |\n"
    "|---|---|---|---|---|---|---|"
)


def score_from_preds(preds_per_frame, ground_truth, name, fps=0.0, iou_thr=0.5,
                     peak_vram_gb=None):
    """Accuracy-only scoring from **precomputed** per-frame predictions (one
    category). Lets a single detection pass be scored per category without
    re-running the model — used by the Colab runner, where EgoBlur emits faces +
    plates and SAM 3 emits several concepts in one pass. ``fps`` / ``peak_vram_gb``
    are the model-level numbers measured once by the caller.
    """
    if len(preds_per_frame) != len(ground_truth):
        raise ValueError("preds_per_frame and ground_truth must be the same length")
    images = []
    tp = fp = fn = 0
    for regions, gts in zip(preds_per_frame, ground_truth):
        pred_boxes = [r.box for r in regions]
        pred_scores = [getattr(r, "score", 1.0) for r in regions]
        gts = list(gts)
        images.append({"pred_boxes": pred_boxes, "pred_scores": pred_scores,
                       "gt_boxes": gts})
        a, b, c, _ = metrics.match_detections(pred_boxes, pred_scores, gts, iou_thr)
        tp += a
        fp += b
        fn += c
    ap = metrics.average_precision(images, iou_thr)
    precision, recall = metrics.precision_recall(tp, fp, fn)
    recall_at_p95 = metrics.recall_at_precision(images, 0.95, iou_thr)
    return BenchResult(name=name, ap=ap, precision=precision, recall=recall,
                       recall_at_p95=recall_at_p95, fps=fps,
                       peak_vram_gb=peak_vram_gb, n_frames=len(preds_per_frame))


def run_benchmark(detector, frames, ground_truth, name, iou_thr=0.5):
    """Score one detector end to end.

    ``frames``        : list of images (numpy arrays).
    ``ground_truth``  : list (parallel to ``frames``) of gt-box lists.
    Times one detection pass (FPS), then delegates accuracy to
    ``score_from_preds``. Returns a ``BenchResult``.
    """
    if len(frames) != len(ground_truth):
        raise ValueError("frames and ground_truth must be the same length")

    metrics.reset_vram()
    t0 = time.perf_counter()
    preds_per_frame = [detector.detect(f) for f in frames]
    elapsed = time.perf_counter() - t0
    fps = len(frames) / elapsed if elapsed > 0 else float("inf")

    return score_from_preds(preds_per_frame, ground_truth, name, fps=fps,
                            iou_thr=iou_thr, peak_vram_gb=metrics.peak_vram_gb())


def results_table(results):
    """Render a list of BenchResult as a Markdown table for the manuscript."""
    lines = [MARKDOWN_HEADER]
    lines += [r.as_markdown_row() for r in results]
    return "\n".join(lines)


def temporal_pass(preds_per_frame, ground_truth, ks=(0, 2, 5, 10), iou_thr=0.5):
    """Video-level temporal scoring for one category — the *supporting* metric
    (`planning/12` §3).

    ``preds_per_frame`` is exactly what ``run_benchmark``'s detection pass
    produces (per-frame lists of ``Region``), so call ``detector.detect`` once
    and feed the result to both. Returns a list of ``temporal.TemporalResult``
    across propagation windows ``ks``: ``k=0`` is the raw per-frame detector;
    ``k>0`` emulates a bolt-on tracker (Paradigm A) or SAM 3's native video
    memory (Paradigm B). Render with ``temporal.temporal_table``.
    """
    present, fired = temporal.presence_and_hits(preds_per_frame, ground_truth, iou_thr)
    return temporal.temporal_sweep(present, fired, ks)
