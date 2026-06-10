"""Temporal (video-level) anonymization metrics — the supporting angle.

Per-frame detection accuracy (``metrics.py``) misses a privacy-critical fact: a
face is identifying if it is visible in **any** frame, so coverage must be
measured across a track's lifetime, not per frame. A per-frame detector that
*blinks* on hard egocentric frames (motion blur from head-turns, oblique angle,
small scale, occlusion) leaves brief leaks even at high per-frame recall; a
tracker (ByteTrack/BoT-SORT) or a video-native model with memory (SAM 3, SAM 2)
**propagates** the redaction across the blink and closes them.

This module scores that. It is the *supporting* metric for the headline
paradigm comparison (`planning/12` §3): Paradigm A reaches temporal recall only
with a **bolt-on** tracker (``k>0`` here); SAM 3 (Paradigm B) gets it
**natively** — a concrete, measured reason to prefer one, or a hybrid.

It works on per-frame boolean timelines for one PII category:

* ``present[t]`` — ground truth: is the category visible in frame ``t``?
* ``fired[t]``   — did the detector catch it in frame ``t`` (a true hit)?

``propagate`` turns ``fired`` into the redaction footprint a tracker / memory
model yields (hold + bidirectional gap-fill, window ``k``). Over-redaction here
is the *propagation* cost (a held redaction spilling past the track) — distinct
from Layer-1 false-positive over-blur, which ``metrics.precision`` already
covers. See ``code/experiments/temporal_recall_demo.py`` for the illustration.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import metrics


# --------------------------------------------------------------------------
# Run-length helpers
# --------------------------------------------------------------------------
def runs_of_true(mask) -> list[int]:
    """Lengths of maximal runs of True in a boolean sequence."""
    out, c = [], 0
    for v in np.asarray(mask, bool):
        if v:
            c += 1
        elif c:
            out.append(c)
            c = 0
    if c:
        out.append(c)
    return out


# --------------------------------------------------------------------------
# Temporal propagation — the tracker / video-memory footprint
# --------------------------------------------------------------------------
def propagate(fired, k: int):
    """Redaction footprint from raw per-frame hits under a propagation window.

    Fills any interior gap of un-fired frames of length ``<= k`` between two hits
    (interpolation) and holds ``k`` frames after each hit (track persistence).
    ``k=0`` reduces to per-frame redaction (no tracker). Returns a bool array.
    """
    fired = np.asarray(fired, bool)
    redacted = fired.copy()
    n = len(fired)
    idx = np.flatnonzero(fired)
    if idx.size == 0:
        return redacted
    for a, b in zip(idx[:-1], idx[1:]):       # bridge short interior gaps
        if 1 < b - a <= k + 1:
            redacted[a:b] = True
    for i in idx:                              # hold forward after each hit
        redacted[i:min(n, i + k + 1)] = True
    return redacted


# --------------------------------------------------------------------------
# Temporal metrics
# --------------------------------------------------------------------------
@dataclass
class TemporalResult:
    """Video-level anonymization outcome for one category at one window ``k``."""

    k: int = 0
    temporal_recall: float = 1.0       # share of present frames actually redacted
    frame_leak_rate: float = 0.0       # 1 - temporal_recall
    leaked_frames: int = 0             # present but not redacted
    leak_events: int = 0              # maximal runs of leaked frames
    longest_leak_gap: int = 0          # worst single visible streak (frames)
    over_redaction_frames: int = 0     # redacted with no present PII (hold spill)
    n_present: int = 0

    def as_markdown_row(self, label=None, fps=None):
        name = label if label is not None else (
            "per-frame (k=0)" if self.k == 0 else f"propagate k={self.k}")
        gap = f"{self.longest_leak_gap}"
        if fps:
            gap += f" ({self.longest_leak_gap / fps:.2f}s)"
        return (
            f"| {name} | {self.temporal_recall:.3f} | {self.frame_leak_rate:.3f} "
            f"| {self.leaked_frames} | {self.leak_events} | {gap} "
            f"| {self.over_redaction_frames} |"
        )


TEMPORAL_HEADER = (
    "| Strategy | Temporal recall | Leak rate | Leaked frames | Leak events "
    "| Longest gap | Over-redaction |\n"
    "|---|---|---|---|---|---|---|"
)


def evaluate_temporal(present, fired, k=0) -> TemporalResult:
    """Score one category timeline at propagation window ``k``."""
    present = np.asarray(present, bool)
    redacted = propagate(fired, k)
    leaked = present & ~redacted
    over = redacted & ~present
    n_present = int(present.sum())
    leak_runs = runs_of_true(leaked)
    return TemporalResult(
        k=k,
        temporal_recall=1 - leaked.sum() / max(n_present, 1),
        frame_leak_rate=leaked.sum() / max(n_present, 1),
        leaked_frames=int(leaked.sum()),
        leak_events=len(leak_runs),
        longest_leak_gap=max(leak_runs) if leak_runs else 0,
        over_redaction_frames=int(over.sum()),
        n_present=n_present,
    )


# --------------------------------------------------------------------------
# Bridge from the per-frame benchmark to temporal timelines
# --------------------------------------------------------------------------
def presence_and_hits(preds_per_frame, ground_truth, iou_thr=0.5):
    """Derive ``(present, fired)`` for one category from the benchmark's
    per-frame predictions.

    ``preds_per_frame`` — list (per frame) of detected ``Region``\\s (one
    category). ``ground_truth`` — parallel list of gt-box lists. A frame is
    *present* if it has any gt box, and *fired* if the detector produced a
    true-positive match there (``metrics.match_detections``). The two arrays
    feed ``evaluate_temporal``.
    """
    n = len(ground_truth)
    if len(preds_per_frame) != n:
        raise ValueError("preds_per_frame and ground_truth must be the same length")
    present = np.zeros(n, bool)
    fired = np.zeros(n, bool)
    for t, (regions, gts) in enumerate(zip(preds_per_frame, ground_truth)):
        gts = list(gts)
        present[t] = len(gts) > 0
        if not present[t]:
            continue
        boxes = [r.box for r in regions]
        scores = [getattr(r, "score", 1.0) for r in regions]
        tp, _, _, _ = metrics.match_detections(boxes, scores, gts, iou_thr)
        fired[t] = tp > 0
    return present, fired


def temporal_sweep(present, fired, ks=(0, 2, 5, 10)) -> list[TemporalResult]:
    """Evaluate a timeline across several propagation windows (for the figure)."""
    return [evaluate_temporal(present, fired, k) for k in ks]


def temporal_table(results, fps=None) -> str:
    """Render a list of ``TemporalResult`` as a Markdown table."""
    return "\n".join([TEMPORAL_HEADER] + [r.as_markdown_row(fps=fps) for r in results])
