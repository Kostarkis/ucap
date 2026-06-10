"""Detection / segmentation metrics and throughput helpers.

Used by the benchmark to score every Paradigm-A (specialist zoo) and Paradigm-B
(promptable concept segmentation) model on the same egocentric evaluation
subset. Pure Python + numpy; torch is imported lazily and only for VRAM.

Boxes are ``(x1, y1, x2, y2)`` in pixel coordinates.

A note on the metric choice: for *anonymization* a false negative (un-redacted
PII) is far worse than a false positive (a harmless over-blur). Standard mAP
weights precision and recall equally, so this module also exposes
``recall_at_precision`` and ``fbeta`` (use beta > 1 to favour recall) — see
``planning/05_model_landscape.md`` section 6.

The three headline metrics (the blur-vs-utility trade-off, §2.4 of the paper;
all **lower = better**) are reported on top of the per-frame accuracy above. Each
failure — under-blur (privacy) or over-blur (utility) — is either *caught* by the
conflict flag or *silent*:

* **M1 — face-leak rate**: present frames left un-redacted-and-unflagged / present
  frames (``frame_leak_rate``). Equal to one minus the temporal recall in
  ``temporal.py``. The *silent* privacy failure (under-blur). The same function
  scores any other prompt category (documents, screens) if one is added.
* **M2 — conflict / human-review rate**: frames where a blur box overlaps a *detected*
  interaction-zone box (so it is flagged) / **total** frames (``conflict_review_rate``).
  Not damage but the *cost* of the conflicts the pipeline catches — reported in %,
  the share of footage a human must review.
* **M3 — lost-training-value rate**: frames where a blur box covers an active interaction
  area (GT hand/object) **with no review flag** / **total** frames
  (``lost_training_value_rate``). The *silent* utility failure — the dual of a leak, from
  a missed hand rather than missed PII.

M1/M3 are silent damage; M2 is the caught-conflict cost. Driving the leak rate down
demands more aggressive redaction, which raises M2 and M3, so the method is judged by its
operating point across all three, not one number.
"""
from __future__ import annotations

import time

import numpy as np


# --------------------------------------------------------------------------
# Overlap
# --------------------------------------------------------------------------
def iou(box_a, box_b) -> float:
    """Intersection-over-union of two axis-aligned boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def mask_iou(mask_a, mask_b) -> float:
    """IoU of two binary masks (Paradigm B produces masks, not just boxes)."""
    a = np.asarray(mask_a).astype(bool)
    b = np.asarray(mask_b).astype(bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum() / union)


# --------------------------------------------------------------------------
# Matching and precision / recall
# --------------------------------------------------------------------------
def match_detections(pred_boxes, pred_scores, gt_boxes, iou_thr=0.5):
    """Greedy IoU matching, highest score first.

    Returns ``(tp, fp, fn, pairs)`` where ``pairs`` is a list of
    ``(pred_index, gt_index)`` tuples.
    """
    order = list(np.argsort(pred_scores)[::-1]) if len(pred_scores) else []
    gt_used = [False] * len(gt_boxes)
    tp = fp = 0
    pairs = []
    for i in order:
        best_iou, best_j = 0.0, -1
        for j, gt in enumerate(gt_boxes):
            if gt_used[j]:
                continue
            v = iou(pred_boxes[i], gt)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_iou >= iou_thr and best_j >= 0:
            tp += 1
            gt_used[best_j] = True
            pairs.append((int(i), best_j))
        else:
            fp += 1
    fn = gt_used.count(False)
    return tp, fp, fn, pairs


def precision_recall(tp, fp, fn):
    """Precision and recall from a TP/FP/FN count."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall


def fbeta(precision, recall, beta=2.0):
    """F-beta score. beta > 1 weights recall higher — the right default for a
    privacy task, where a missed detection is the costly error."""
    b2 = beta * beta
    denom = b2 * precision + recall
    return (1 + b2) * precision * recall / denom if denom > 0 else 0.0


# --------------------------------------------------------------------------
# Average precision
# --------------------------------------------------------------------------
def _pr_curve(images, iou_thr=0.5):
    """Build the global precision/recall curve over a set of images.

    ``images`` is a list of dicts with keys ``pred_boxes``, ``pred_scores``,
    ``gt_boxes``. Returns ``(recalls, precisions, n_gt)``.
    """
    entries = []  # (score, is_tp)
    n_gt = 0
    for im in images:
        gt = list(im["gt_boxes"])
        n_gt += len(gt)
        pred_boxes = im["pred_boxes"]
        pred_scores = im["pred_scores"]
        order = list(np.argsort(pred_scores)[::-1]) if len(pred_scores) else []
        used = [False] * len(gt)
        for i in order:
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(gt):
                if used[j]:
                    continue
                v = iou(pred_boxes[i], g)
                if v > best_iou:
                    best_iou, best_j = v, j
            is_tp = best_iou >= iou_thr and best_j >= 0
            if is_tp:
                used[best_j] = True
            entries.append((float(pred_scores[i]), is_tp))
    if not entries:
        return np.array([]), np.array([]), n_gt
    entries.sort(key=lambda e: e[0], reverse=True)
    tp = np.array([1 if e[1] else 0 for e in entries], dtype=float)
    fp = 1.0 - tp
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / n_gt if n_gt else np.zeros_like(tp_cum)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    return recalls, precisions, n_gt


def average_precision(images, iou_thr=0.5):
    """All-point-interpolated AP for one class over a set of images."""
    recalls, precisions, n_gt = _pr_curve(images, iou_thr)
    if n_gt == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def mean_average_precision(per_class_images, iou_thr=0.5):
    """mAP over classes. ``per_class_images`` maps class name -> image list."""
    if not per_class_images:
        return 0.0
    aps = {c: average_precision(ims, iou_thr) for c, ims in per_class_images.items()}
    return float(np.mean(list(aps.values()))), aps


def recall_at_precision(images, target_precision=0.95, iou_thr=0.5):
    """Highest recall achievable while keeping precision >= target.

    The privacy-relevant question: 'if we accept at most a 5% false-positive
    rate, how much of the PII do we still catch?'
    """
    recalls, precisions, n_gt = _pr_curve(images, iou_thr)
    if n_gt == 0 or len(recalls) == 0:
        return 0.0
    ok = recalls[precisions >= target_precision]
    return float(ok.max()) if len(ok) else 0.0


# --------------------------------------------------------------------------
# The three headline metrics (the blur-vs-utility trade-off; paper §2.4)
# --------------------------------------------------------------------------
def frame_leak_rate(present, covered) -> float:
    """Per-frame privacy-leak rate for one category — **M1** for faces (the paper's
    headline); the same function scores any other category (documents, screens).

    A frame *leaks* when the category is present but no redaction covers it.
    ``present`` and ``covered`` are equal-length per-frame boolean sequences
    (``covered[i]`` = a present instance was actually blurred in frame ``i``).

    Returns ``leaked_present_frames / present_frames`` (0.0 if never present);
    lower is better. This is one minus the temporal recall in ``temporal.py``.
    """
    present = [bool(p) for p in present]
    n_present = sum(present)
    if n_present == 0:
        return 0.0
    leaked = sum(1 for p, c in zip(present, covered) if p and not c)
    return leaked / n_present


def _any_overlap(boxes_a, boxes_b, min_area=1.0) -> bool:
    """True if any box in ``boxes_a`` intersects any box in ``boxes_b`` by >= ``min_area``."""
    for a in boxes_a:
        for b in boxes_b:
            iw = min(a[2], b[2]) - max(a[0], b[0])
            ih = min(a[3], b[3]) - max(a[1], b[1])
            if iw > 0 and ih > 0 and iw * ih >= min_area:
                return True
    return False


def conflict_review_rate(blur_boxes_per_frame, keep_boxes_per_frame, min_area=1.0) -> float:
    """Per-clip conflict / human-review rate — **M2**.

    A frame is a *conflict frame* when a redaction (blur) box overlaps a preserved
    interaction-zone (hand/object) box: the privacy/utility collision a reviewer must
    adjudicate (e.g. a held phone screen against the hand holding it). Returns
    ``conflict_frames / total_frames`` over the clip; lower is better.

    ``blur_boxes_per_frame`` and ``keep_boxes_per_frame`` are equal-length lists, one
    entry per frame, each a list of ``(x1, y1, x2, y2)`` boxes in pixel coordinates.
    """
    total = len(blur_boxes_per_frame)
    if total == 0:
        return 0.0
    conflicts = sum(
        1
        for blur, keep in zip(blur_boxes_per_frame, keep_boxes_per_frame)
        if _any_overlap(blur, keep, min_area)
    )
    return conflicts / total


def lost_training_value_rate(
    blur_boxes_per_frame, interaction_boxes_per_frame, flagged_per_frame, min_area=1.0
) -> float:
    """Per-clip lost-training-value rate — **M3** (the *silent* over-blur).

    A frame loses training value when a redaction (blur) box covers an *active interaction
    area* (a ground-truth hand / manipulated object) yet the frame was **not** flagged for
    review — so the manipulation signal is destroyed silently. This is the utility-side dual
    of a privacy leak (M1): there a missed PII detection leaks; here a missed hand/object
    detection over-blurs. Returns ``silent_loss_frames / total_frames``; lower is better.

    ``blur_boxes_per_frame`` and ``interaction_boxes_per_frame`` are equal-length lists of
    per-frame ``(x1, y1, x2, y2)`` box lists (the second is the GT interaction zone);
    ``flagged_per_frame`` is a per-frame boolean — did the pipeline raise a review flag.

    Contrast ``conflict_review_rate`` (M2), which counts overlaps the pipeline *did* flag:
    M2 + M3 partition the frames where blur meets the interaction zone into caught vs silent.
    """
    total = len(blur_boxes_per_frame)
    if total == 0:
        return 0.0
    lost = sum(
        1
        for blur, inter, flagged in zip(
            blur_boxes_per_frame, interaction_boxes_per_frame, flagged_per_frame
        )
        if not flagged and _any_overlap(blur, inter, min_area)
    )
    return lost / total


# --------------------------------------------------------------------------
# Throughput and VRAM
# --------------------------------------------------------------------------
def measure_fps(fn, iters=30, warmup=3):
    """Frames per second of a zero-argument callable that processes one frame."""
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    dt = time.perf_counter() - t0
    return iters / dt if dt > 0 else float("inf")


def reset_vram():
    """Reset the CUDA peak-memory counter (no-op without torch/CUDA)."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def peak_vram_gb():
    """Peak CUDA memory in GB since the last reset, or None if unavailable."""
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1e9, 3)
    except Exception:
        pass
    return None
