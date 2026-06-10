"""Tests for temporal.py — hand-computed cases, no torch/cv2 needed."""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

import temporal


def _approx(a, b, tol=1e-9):
    assert abs(a - b) <= tol, f"{a} != {b}"


def _region(box, score=1.0):
    return SimpleNamespace(box=box, score=score)


# --- runs_of_true ---------------------------------------------------------
def test_runs_of_true():
    assert temporal.runs_of_true([False, True, True, False, True]) == [2, 1]
    assert temporal.runs_of_true([False, False]) == []
    assert temporal.runs_of_true([True, True, True]) == [3]


# --- propagate ------------------------------------------------------------
def test_propagate_identity_when_k_zero():
    fired = np.array([True, False, False, True, False])
    assert list(temporal.propagate(fired, 0)) == list(fired)


def test_propagate_bridges_short_interior_gap_and_holds():
    fired = np.array([True, False, False, True, False])
    # k=2: gap of 3 between hits 0 and 3 is bridged; 2-frame hold after hit 3.
    assert list(temporal.propagate(fired, 2)) == [True, True, True, True, True]


def test_propagate_does_not_bridge_long_gap():
    fired = np.array([True, False, False, False, False, False, True])
    # gap of 6 > k+1=3 -> not bridged; only the 2-frame forward holds apply.
    assert list(temporal.propagate(fired, 2)) == [
        True, True, True, False, False, False, True]


def test_propagate_empty():
    fired = np.zeros(4, bool)
    assert list(temporal.propagate(fired, 5)) == [False, False, False, False]


# --- evaluate_temporal ----------------------------------------------------
def test_evaluate_temporal_known_per_frame():
    present = np.array([True, True, True, True, True])
    fired = np.array([True, False, False, True, False])
    r = temporal.evaluate_temporal(present, fired, k=0)
    _approx(r.temporal_recall, 0.4)          # 3 of 5 present frames leak
    assert r.leaked_frames == 3
    assert r.leak_events == 2                 # runs {1,2} and {4}
    assert r.longest_leak_gap == 2
    assert r.over_redaction_frames == 0
    assert r.n_present == 5


def test_evaluate_temporal_propagation_closes_leaks():
    present = np.array([True, True, True, True, True])
    fired = np.array([True, False, False, True, False])
    r = temporal.evaluate_temporal(present, fired, k=2)
    _approx(r.temporal_recall, 1.0)
    assert r.leaked_frames == 0
    assert r.over_redaction_frames == 0


def test_evaluate_temporal_over_redaction_counts_hold_spill():
    # face present only in frame 0; a 2-frame hold spills into 1 and 2 (no face).
    present = np.array([True, False, False, False])
    fired = np.array([True, False, False, False])
    r = temporal.evaluate_temporal(present, fired, k=2)
    _approx(r.temporal_recall, 1.0)
    assert r.over_redaction_frames == 2      # frames 1 and 2 redacted, no PII


# --- presence_and_hits (benchmark bridge) ---------------------------------
def test_presence_and_hits():
    preds = [
        [_region((0, 0, 10, 10), 0.9)],   # present, matches gt -> fired
        [],                                # present, no detection -> miss
        [_region((0, 0, 10, 10), 0.9)],   # not present (no gt) -> not fired
    ]
    gt = [[(0, 0, 10, 10)], [(0, 0, 10, 10)], []]
    present, fired = temporal.presence_and_hits(preds, gt, iou_thr=0.5)
    assert list(present) == [True, True, False]
    assert list(fired) == [True, False, False]


def test_presence_and_hits_length_mismatch():
    try:
        temporal.presence_and_hits([[]], [[], []])
    except ValueError:
        return
    raise AssertionError("expected ValueError on length mismatch")


# --- sweep + rendering ----------------------------------------------------
def test_temporal_sweep_recall_monotonic_in_k():
    present = np.ones(20, bool)
    fired = np.zeros(20, bool)
    fired[::4] = True                        # a hit every 4th frame
    results = temporal.temporal_sweep(present, fired, ks=(0, 2, 5, 10))
    recalls = [r.temporal_recall for r in results]
    assert recalls == sorted(recalls)        # non-decreasing with k
    assert len(results) == 4


def test_markdown_row_and_table():
    present = np.array([True, True, True, True, True])
    fired = np.array([True, False, False, True, False])
    rows = temporal.temporal_sweep(present, fired, ks=(0, 2))
    table = temporal.temporal_table(rows, fps=30)
    assert "Temporal recall" in table
    assert "per-frame (k=0)" in table
    assert "propagate k=2" in table
    assert table.count("\n") == 3            # header line + separator + 2 rows
