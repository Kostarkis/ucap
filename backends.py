"""Concrete Paradigm-A detector / tracker backends for the benchmark (Colab T4).

Each class satisfies a ``pipeline.py`` contract:

* a **Detector** exposes ``detect(frame_bgr) -> list[Region]``;
* a **Tracker** exposes ``update(frame_bgr, regions) -> list[Region]``.

Heavy dependencies (torch, paddleocr, supervision) and the model weights are
imported **lazily**, so importing this module — and the CPU test suite — needs
none of them. Install per backend on Colab; see each docstring. These are
**templates checked against each project's current API**; verify the exact I/O
against the upstream repo before trusting the numbers — the same caveat that
``pipeline.Sam3UltralyticsSegmenter`` carries.

Paradigm B (SAM 3) already lives in ``pipeline.Sam3UltralyticsSegmenter``.
NSFW is a frame-level *triage* classifier (NudeNet / Falconsai ViT), not a
region Detector — run it as a separate stage in the runner, not here
(`planning/05_model_landscape.md` §3.3).
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# Faces + licence plates — Meta EgoBlur (the key egocentric prior work)
# --------------------------------------------------------------------------
def _unpack_detection_output(out):
    """Normalise an EgoBlur / torchvision-detection forward output to
    ``(boxes, scores)`` as plain Python lists. Tolerant of the shapes the
    TorchScript models return across versions: a list/tuple of dicts with
    ``boxes``/``scores`` (torchvision convention), a bare ``(boxes, scores)``
    tuple, or a single dict."""
    def to_list(x):
        return x.detach().cpu().tolist() if hasattr(x, "detach") else list(x)

    if isinstance(out, dict):
        return to_list(out["boxes"]), to_list(out.get("scores", []))
    if isinstance(out, (list, tuple)):
        if len(out) and isinstance(out[0], dict):          # [{'boxes','scores'}]
            return to_list(out[0]["boxes"]), to_list(out[0].get("scores", []))
        if len(out) >= 2:                                   # (boxes, scores, ...)
            return to_list(out[0]), to_list(out[1])
        if len(out) == 1:
            return to_list(out[0]), []
    raise TypeError(f"Unrecognised EgoBlur output type: {type(out)}")


class EgoBlurDetector:
    """Paradigm-A face + licence-plate detector (Meta EgoBlur).

    Two TorchScript models from ``facebookresearch/EgoBlur`` (also on HF
    ``projectaria/EgoBlur``): ``ego_blur_face.jit`` and ``ego_blur_lp.jit`` —
    FasterRCNN-ResNeXt, ~104 M params each, trained on 23 M egocentric images.
    Meta ran them on a V100 16 GB, so a T4 is in budget. **Per-frame, no
    tracking** — pair with ``ByteTrackTracker`` to recover temporal recall
    (`planning/12` §3); that gap is precisely the missed-frame leak the project
    measures.

    Colab: ``pip install torch torchvision`` and download the two ``.jit`` files.
    Verify the forward signature against EgoBlur ``script/demo_ego_blur.py``.
    """

    def __init__(self, face_model=None, lp_model=None, device="cuda",
                 score_threshold=0.5):
        import torch

        self._torch = torch
        self.device = device
        self.score_threshold = score_threshold
        self.models = []  # (label, module)
        if face_model:
            self.models.append(
                ("face", torch.jit.load(face_model, map_location=device).eval()))
        if lp_model:
            self.models.append(
                ("licence plate",
                 torch.jit.load(lp_model, map_location=device).eval()))
        if not self.models:
            raise ValueError("EgoBlurDetector needs face_model and/or lp_model")

    def _to_tensor(self, frame_bgr):
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])     # BGR -> RGB
        t = self._torch.from_numpy(rgb).permute(2, 0, 1).float()
        return t.to(self.device)

    def detect(self, frame):
        from pipeline import Region

        regions = []
        img = self._to_tensor(frame)
        with self._torch.no_grad():
            for label, model in self.models:
                boxes, scores = _unpack_detection_output(model(img))
                for i, box in enumerate(boxes):
                    score = float(scores[i]) if i < len(scores) else 1.0
                    if score < self.score_threshold:
                        continue
                    x1, y1, x2, y2 = (int(round(float(v))) for v in box[:4])
                    regions.append(Region(box=(x1, y1, x2, y2), label=label,
                                          score=score))
        return regions


# --------------------------------------------------------------------------
# Scene text / documents — PaddleOCR (detection only)
# --------------------------------------------------------------------------
class PaddleOCRDetector:
    """Paradigm-A scene-text detector (PaddleOCR, **detection only**).

    For anonymization, locating text is what matters; recognition is off (it only
    helps triage which text is sensitive). PP-OCRv4/v5 mobile det is light and
    T4-friendly, Apache-2.0.

    Colab: ``pip install paddlepaddle-gpu paddleocr``. API note — this targets
    PaddleOCR **2.x** ``ocr.ocr(img, det=True, rec=False)`` (returns quad
    polygons); 3.x renamed the call to ``predict`` and changed the result schema.
    """

    def __init__(self, lang="en", use_gpu=True, score_threshold=0.5,
                 label="text"):
        from paddleocr import PaddleOCR

        self._ocr = PaddleOCR(use_angle_cls=False, lang=lang, use_gpu=use_gpu,
                              show_log=False)
        self.label = label
        self.score_threshold = score_threshold

    def detect(self, frame):
        from pipeline import Region

        regions = []
        result = self._ocr.ocr(frame, det=True, rec=False)
        for per_image in (result or []):
            for poly in (per_image or []):
                pts = np.asarray(poly, dtype=float).reshape(-1, 2)
                x1, y1 = pts.min(0)
                x2, y2 = pts.max(0)
                regions.append(Region(box=(int(x1), int(y1), int(x2), int(y2)),
                                      label=self.label, score=1.0))
        return regions


# --------------------------------------------------------------------------
# Tracking — ByteTrack (Paradigm A's path to temporal recall)
# --------------------------------------------------------------------------
class ByteTrackTracker:
    """Bolt-on tracker (ByteTrack via ``supervision``).

    Assigns persistent ``track_id``\\s so a redaction survives a detector blink —
    Paradigm A's route to the temporal recall SAM 3 gets natively (`planning/12`
    §3). **Fail-safe:** every input region is returned (and still redacted);
    tracking only *adds* ``track_id`` and is the production analogue of the
    benchmark's ``temporal.propagate`` emulation.

    Colab: ``pip install supervision``. ByteTrack tracks by box; any instance
    ``mask`` passes through untouched.
    """

    def __init__(self, **kwargs):
        import supervision as sv

        self._sv = sv
        self._tracker = sv.ByteTrack(**kwargs)

    def update(self, frame, regions):
        import metrics

        if not regions:
            self._tracker.update_with_detections(self._sv.Detections.empty())
            return regions
        xyxy = np.array([r.box for r in regions], dtype=float)
        conf = np.array([getattr(r, "score", 1.0) for r in regions], dtype=float)
        dets = self._sv.Detections(
            xyxy=xyxy, confidence=conf,
            class_id=np.zeros(len(regions), dtype=int))
        tracked = self._tracker.update_with_detections(dets)
        # supervision may drop/reorder; re-associate track_id to the original
        # regions by IoU so the returned list stays complete (fail-safe).
        for tb, tid in zip(tracked.xyxy, tracked.tracker_id):
            best, best_iou = None, 0.0
            for r in regions:
                v = metrics.iou(r.box, tuple(tb))
                if v > best_iou:
                    best, best_iou = r, v
            if best is not None and best_iou >= 0.5:
                best.track_id = int(tid)
        return regions
