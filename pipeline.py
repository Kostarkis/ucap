"""Anonymization pipeline — a privacy/utility-aware detect -> protect -> resolve
-> redact chain.

Egocentric training data sits under two opposing forces:

* **Privacy** — identifying content (faces, documents, screens, plates) must be
  redacted.
* **Utility** — the manipulation signal (the operator's hands and the objects /
  regions they interact with) must be preserved; blurring it destroys the
  training value of the data.

These conflict when a detected PII region overlaps a manipulation-critical
region. The pipeline resolves this by estimating a *protected interaction zone*
— in egocentric video the near field, recoverable from stereo or monocular
depth — and, where a PII detection overlaps that zone, marking the redaction as
**flagged** for lightweight human review rather than silently deciding. Output
is fail-safe: every detected PII region is still redacted, so nothing leaks
before review; the flag list is the human reviewer's targeted worklist.

The pipeline is paradigm-agnostic — a Paradigm-A specialist detector and a
Paradigm-B promptable backend both satisfy the same ``Detector`` contract.
``Sam3UltralyticsSegmenter`` is the Paradigm-B (axis B1) backend; the remaining
concrete backends (specialist detectors, depth/zone estimator) are filled in
after the Phase-2 spike. ``FixedBoxDetector`` and ``BoxZoneEstimator`` are
dependency-free stubs used by the tests and as backend templates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol

import cv2
import numpy as np

import video_io


@dataclass
class Region:
    """A detected sensitive region.

    ``status`` is set by ``resolve_conflicts``: ``'redact'`` (clear of the
    interaction zone) or ``'flagged'`` (overlaps it — privacy/utility conflict,
    routed to human review). ``mask`` is optional: Paradigm B yields instance
    masks (tighter redaction, smaller conflict surface); Paradigm A usually
    yields only a box.
    """

    box: tuple  # (x1, y1, x2, y2) in pixels
    label: str  # 'face', 'licence plate', 'document', ...
    score: float = 1.0
    mask: Optional[np.ndarray] = None
    track_id: Optional[int] = None
    status: str = "redact"          # 'redact' | 'flagged'
    protected_overlap: float = 0.0  # fraction of the region inside the interaction zone


class Detector(Protocol):
    """A specialist-zoo detector OR a promptable-segmentation backend."""

    def detect(self, frame: np.ndarray) -> list[Region]:
        ...


class Tracker(Protocol):
    """Propagates regions across frames so a redaction never flickers."""

    def update(self, frame: np.ndarray, regions: list[Region]) -> list[Region]:
        ...


class InteractionZoneEstimator(Protocol):
    """Returns a frame-sized boolean mask of the manipulation-critical zone to
    **protect** from over-redaction — the operator's hands and the near-field
    objects they interact with. In egocentric video this zone is recoverable
    from depth (stereo matching on the stereo pair, or a monocular depth
    model)."""

    def protected_mask(self, frame: np.ndarray) -> np.ndarray:
        ...


# --------------------------------------------------------------------------
# Redaction strategies
# --------------------------------------------------------------------------
def _clip_box(box, w, h):
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def blur_regions(frame, regions, ksize=31):
    """Gaussian-blur every region. Mask-aware: if a Region carries a mask, only
    the masked pixels inside its box are blurred (tighter, less collateral)."""
    out = frame.copy()
    h, w = frame.shape[:2]
    k = ksize | 1  # GaussianBlur needs an odd kernel
    for r in regions:
        x1, y1, x2, y2 = _clip_box(r.box, w, h)
        if x2 <= x1 or y2 <= y1:
            continue
        roi = out[y1:y2, x1:x2]
        blurred = cv2.GaussianBlur(roi, (k, k), 0)
        if r.mask is not None:
            sub = np.asarray(r.mask)[y1:y2, x1:x2].astype(bool)
            roi[sub] = blurred[sub]
        else:
            out[y1:y2, x1:x2] = blurred
    return out


def mask_regions(frame, regions, color=(0, 0, 0)):
    """Replace every region with a solid colour (maximal information loss)."""
    out = frame.copy()
    h, w = frame.shape[:2]
    for r in regions:
        x1, y1, x2, y2 = _clip_box(r.box, w, h)
        if x2 <= x1 or y2 <= y1:
            continue
        if r.mask is not None:
            sub = np.asarray(r.mask)[y1:y2, x1:x2].astype(bool)
            out[y1:y2, x1:x2][sub] = color
        else:
            out[y1:y2, x1:x2] = color
    return out


# --------------------------------------------------------------------------
# Privacy / utility conflict resolution
# --------------------------------------------------------------------------
def region_protected_overlap(region, protected_mask):
    """Fraction of a region that lies inside the protected interaction zone.

    Uses the instance mask when present (more precise — and the reason
    mask-producing models reduce the conflict surface), else the box.
    """
    pm = np.asarray(protected_mask).astype(bool)
    h, w = pm.shape[:2]
    if region.mask is not None:
        rm = np.asarray(region.mask).astype(bool)
        denom = rm.sum()
        if denom == 0:
            return 0.0
        return float(np.logical_and(rm, pm).sum() / denom)
    x1, y1, x2, y2 = _clip_box(region.box, w, h)
    area = (x2 - x1) * (y2 - y1)
    if area <= 0:
        return 0.0
    return float(pm[y1:y2, x1:x2].sum() / area)


def resolve_conflicts(regions, protected_mask, overlap_threshold=0.15):
    """Classify each PII region against the protected interaction zone.

    A region overlapping the zone beyond ``overlap_threshold`` is a
    privacy/utility conflict: its status becomes ``'flagged'`` (route to human
    review). Otherwise ``'redact'``. Returns the same list, mutated.
    """
    for r in regions:
        ov = region_protected_overlap(r, protected_mask)
        r.protected_overlap = ov
        r.status = "flagged" if ov >= overlap_threshold else "redact"
    return regions


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------
@dataclass
class VideoResult:
    """Outcome of anonymizing a video."""

    frames_written: int = 0
    flagged: list = field(default_factory=list)  # list of (frame_index, Region)

    @property
    def n_flagged(self):
        return len(self.flagged)


class AnonymizationPipeline:
    """Chains a detector, an optional tracker, an optional interaction-zone
    estimator, and a redaction strategy.

    With a ``zone_estimator``, every detected PII region is checked against the
    protected interaction zone and flagged on conflict. Redaction is always
    fail-safe — every detected PII region is redacted in the output — so the
    flag list is a review worklist, not an un-redacted leak.
    """

    def __init__(self, detector: Detector, tracker: Optional[Tracker] = None,
                 zone_estimator: Optional[InteractionZoneEstimator] = None,
                 redactor=blur_regions, overlap_threshold=0.15):
        self.detector = detector
        self.tracker = tracker
        self.zone_estimator = zone_estimator
        self.redactor = redactor
        self.overlap_threshold = overlap_threshold

    def process_frame(self, frame):
        """Return ``(redacted_frame, regions)``. Each Region carries ``.status``
        — a ``'flagged'`` region overlaps the interaction zone and needs human
        review. All detected PII regions are redacted in ``redacted_frame``."""
        regions = self.detector.detect(frame)
        if self.tracker is not None:
            regions = self.tracker.update(frame, regions)
        if self.zone_estimator is not None:
            protected = self.zone_estimator.protected_mask(frame)
            resolve_conflicts(regions, protected, self.overlap_threshold)
        redacted = self.redactor(frame, regions)
        return redacted, regions

    def process_video(self, in_path, out_path, max_frames=None):
        """Anonymize a whole MP4. Returns a ``VideoResult`` with the frame count
        and the per-frame list of flagged regions for human review."""
        info = video_io.probe(in_path)
        result = VideoResult()
        with video_io.VideoWriter(out_path, info.fps, (info.width, info.height)) as w:
            for idx, frame in enumerate(
                video_io.read_frames(in_path, max_frames=max_frames)
            ):
                redacted, regions = self.process_frame(frame)
                w.write(redacted)
                result.frames_written += 1
                for r in regions:
                    if r.status == "flagged":
                        result.flagged.append((idx, r))
        return result


# --------------------------------------------------------------------------
# Dependency-free stubs (tests; templates for real backends)
# --------------------------------------------------------------------------
class FixedBoxDetector:
    """Returns one fixed Region per frame. The skeleton a real ``Detector``
    backend should match."""

    def __init__(self, box, label="test", score=1.0):
        self.box = tuple(box)
        self.label = label
        self.score = score

    def detect(self, frame):
        return [Region(box=self.box, label=self.label, score=self.score)]


class BoxZoneEstimator:
    """An ``InteractionZoneEstimator`` that protects a fixed rectangle. The
    skeleton a real depth-derived zone estimator should match."""

    def __init__(self, box):
        self.box = tuple(box)

    def protected_mask(self, frame):
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), bool)
        x1, y1, x2, y2 = _clip_box(self.box, w, h)
        mask[y1:y2, x1:x2] = True
        return mask


# --------------------------------------------------------------------------
# Paradigm-B backend — Ultralytics SAM 3 (axis B1)
# --------------------------------------------------------------------------
class Sam3UltralyticsSegmenter:
    """Paradigm-B (axis B1) detector backend — Ultralytics SAM 3.

    Wraps ``ultralytics.models.sam.SAM3VideoSemanticPredictor``: one
    text-prompted model that detects + segments every instance of each concept
    in ``prompts``. This is the BrainHack-validated path — video-native and
    FP16 (``half=True``); it yields instance masks, which tighten redaction and
    shrink the privacy/utility conflict surface.

    ``ultralytics`` and the gated ``facebook/sam3`` weights (``sam3.pt``) are
    runtime dependencies, so the import is deferred — the tests and the
    dependency-free stubs above do not need them. ``detect`` runs a single
    frame to satisfy the per-frame ``Detector`` contract; for production B1 use
    the predictor's native video stream (``mode='track'``), which also tracks.

    Template — verify the predictor and ``Results`` attributes against the
    ``spike_sam3_t4.ipynb`` run before relying on it.
    """

    def __init__(self, prompts, model="sam3.pt", score_threshold=0.5,
                 half=True):
        from ultralytics.models.sam import SAM3VideoSemanticPredictor

        self.prompts = list(prompts)
        overrides = dict(task="segment", mode="predict", model=model,
                         half=half, save=False, retina_masks=True,
                         verbose=False)
        self._predictor = SAM3VideoSemanticPredictor(
            overrides=overrides, score_threshold_detection=score_threshold)

    def detect(self, frame):
        """Return one ``Region`` per detected instance, across all prompts."""
        regions = []
        for label in self.prompts:
            for res in self._predictor(source=frame, text=[label],
                                       stream=False):
                masks = getattr(res, "masks", None)
                if masks is None:
                    continue
                conf = getattr(getattr(res, "boxes", None), "conf", None)
                for i, m in enumerate(masks.data.cpu().numpy()):
                    mask = m.astype(bool)
                    ys, xs = np.where(mask)
                    if xs.size == 0:
                        continue
                    box = (int(xs.min()), int(ys.min()),
                           int(xs.max()) + 1, int(ys.max()) + 1)
                    score = (float(conf[i])
                             if conf is not None and i < len(conf) else 1.0)
                    regions.append(Region(box=box, label=label, score=score,
                                          mask=mask))
        return regions
