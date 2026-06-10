"""Tests for metrics.py — hand-computed cases, no torch needed."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

import metrics


def _approx(a, b, tol=1e-6):
    assert abs(a - b) <= tol, f"{a} != {b}"


def test_iou_partial_overlap():
    # boxes (0,0,10,10) & (5,5,15,15): inter 25, union 175
    _approx(metrics.iou((0, 0, 10, 10), (5, 5, 15, 15)), 25.0 / 175.0)


def test_iou_identical_and_disjoint():
    _approx(metrics.iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
    _approx(metrics.iou((0, 0, 1, 1), (5, 5, 6, 6)), 0.0)


def test_mask_iou():
    a = np.zeros((10, 10), bool)
    a[:5, :] = True  # 50 px
    b = np.zeros((10, 10), bool)
    b[3:8, :] = True  # 50 px; overlap = rows 3,4 = 20 px; union = 80
    _approx(metrics.mask_iou(a, b), 20.0 / 80.0)


def test_match_detections():
    preds = [(0, 0, 10, 10), (100, 100, 110, 110)]
    scores = [0.9, 0.8]
    gts = [(1, 1, 11, 11)]  # overlaps the first pred (IoU ~0.68), not the second
    tp, fp, fn, pairs = metrics.match_detections(preds, scores, gts, iou_thr=0.5)
    assert (tp, fp, fn) == (1, 1, 0)
    assert pairs == [(0, 0)]


def test_precision_recall_and_fbeta():
    p, r = metrics.precision_recall(1, 1, 0)
    _approx(p, 0.5)
    _approx(r, 1.0)
    _approx(metrics.fbeta(1.0, 1.0), 1.0)
    _approx(metrics.fbeta(0.5, 1.0, beta=2.0), 2.5 / 3.0)


def test_average_precision_perfect():
    images = [{"pred_boxes": [(0, 0, 10, 10)], "pred_scores": [0.9],
               "gt_boxes": [(0, 0, 10, 10)]}]
    _approx(metrics.average_precision(images), 1.0)


def test_average_precision_partial():
    # 2 ground truths, 1 correct detection -> AP = 0.5
    images = [{"pred_boxes": [(0, 0, 10, 10)], "pred_scores": [0.9],
               "gt_boxes": [(0, 0, 10, 10), (100, 100, 110, 110)]}]
    _approx(metrics.average_precision(images), 0.5)


def test_average_precision_no_predictions():
    images = [{"pred_boxes": [], "pred_scores": [], "gt_boxes": [(0, 0, 10, 10)]}]
    _approx(metrics.average_precision(images), 0.0)


def test_recall_at_precision():
    # one TP, no FP -> precision 1.0 at recall 0.5
    images = [{"pred_boxes": [(0, 0, 10, 10)], "pred_scores": [0.9],
               "gt_boxes": [(0, 0, 10, 10), (100, 100, 110, 110)]}]
    _approx(metrics.recall_at_precision(images, 0.95), 0.5)


def test_mean_average_precision():
    img = {"pred_boxes": [(0, 0, 10, 10)], "pred_scores": [0.9],
           "gt_boxes": [(0, 0, 10, 10)]}
    m, aps = metrics.mean_average_precision({"face": [img], "text": [img]})
    _approx(m, 1.0)
    assert set(aps) == {"face", "text"}


def test_measure_fps():
    assert metrics.measure_fps(lambda: None, iters=5, warmup=1) > 0


# --- the four headline rates (blur-vs-utility trade-off) ------------------
def test_frame_leak_rate():
    # present at 0,1,3 (n=3); covered only at 0 -> leaked 1,3 -> 2/3
    present = [True, True, False, True]
    covered = [True, False, False, False]
    _approx(metrics.frame_leak_rate(present, covered), 2.0 / 3.0)
    # category never present -> no leak possible -> 0.0
    _approx(metrics.frame_leak_rate([False, False], [False, False]), 0.0)


def test_conflict_review_rate():
    blur = [[(0, 0, 10, 10)], [(0, 0, 10, 10)], []]
    keep = [[(5, 5, 15, 15)], [(100, 100, 110, 110)], [(0, 0, 10, 10)]]
    # only frame 0's blur overlaps a keep box -> 1/3
    _approx(metrics.conflict_review_rate(blur, keep), 1.0 / 3.0)


def test_lost_training_value_rate():
    blur = [[(0, 0, 10, 10)], [(0, 0, 10, 10)], [(0, 0, 10, 10)]]
    inter = [[(5, 5, 15, 15)], [(5, 5, 15, 15)], [(100, 100, 110, 110)]]
    flagged = [False, True, False]
    # frame 0: overlap & un-flagged -> silent loss (M3); frame 1: overlap but flagged
    # -> caught (M2, not M3); frame 2: un-flagged but no overlap -> fine.  -> 1/3
    _approx(metrics.lost_training_value_rate(blur, inter, flagged), 1.0 / 3.0)
    # M2/M3 partition the 2 overlap frames: conflict_review_rate sees both -> 2/3
    _approx(metrics.conflict_review_rate(blur, inter), 2.0 / 3.0)
