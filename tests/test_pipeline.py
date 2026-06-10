"""Tests for pipeline.py — redaction, privacy/utility conflict resolution,
and the end-to-end chain."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import cv2
import numpy as np

import video_io
from pipeline import (AnonymizationPipeline, BoxZoneEstimator, FixedBoxDetector,
                      Region, blur_regions, mask_regions,
                      region_protected_overlap, resolve_conflicts)


def _noisy(h=48, w=64, seed=0):
    return np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)


# --- redaction -------------------------------------------------------------
def test_blur_changes_only_the_region():
    frame = _noisy()
    out = blur_regions(frame, [Region(box=(10, 10, 40, 30), label="x")], ksize=11)
    assert not np.array_equal(out[10:30, 10:40], frame[10:30, 10:40])  # inside changed
    assert np.array_equal(out[0, 0], frame[0, 0])  # outside preserved
    assert np.array_equal(out[45, 60], frame[45, 60])


def test_mask_regions_solid_fill():
    out = mask_regions(_noisy(), [Region(box=(5, 5, 20, 20), label="x")], color=(0, 0, 0))
    assert (out[5:20, 5:20] == 0).all()


def test_blur_with_instance_mask():
    frame = _noisy(40, 40, seed=2)
    m = np.zeros((40, 40), bool)
    m[10:20, 10:20] = True
    out = blur_regions(frame, [Region(box=(0, 0, 40, 40), label="x", mask=m)], ksize=9)
    assert np.array_equal(out[~m], frame[~m])  # unmasked pixels untouched
    assert not np.array_equal(out[10:20, 10:20], frame[10:20, 10:20])  # masked changed


def test_clip_out_of_bounds_box():
    frame = _noisy()
    out = blur_regions(frame, [Region(box=(-20, -20, 30, 30), label="x")])
    assert out.shape == frame.shape  # partly-offscreen box must not crash


# --- privacy / utility conflict resolution ---------------------------------
def test_resolve_conflicts_flags_overlap_with_interaction_zone():
    mask = np.zeros((48, 64), bool)
    mask[:, :32] = True  # protect the left half (the interaction zone)
    inside = Region(box=(5, 5, 25, 40), label="document")   # within the zone
    outside = Region(box=(40, 5, 60, 40), label="face")     # clear of the zone
    resolve_conflicts([inside, outside], mask)
    assert inside.status == "flagged"
    assert outside.status == "redact"
    assert inside.protected_overlap > 0.9
    assert outside.protected_overlap == 0.0


def test_region_protected_overlap_uses_instance_mask():
    protected = np.zeros((40, 40), bool)
    protected[:20, :] = True  # top half protected
    rmask = np.zeros((40, 40), bool)
    rmask[:10, :10] = True  # region mask fully inside the protected half
    r = Region(box=(0, 0, 40, 40), label="x", mask=rmask)
    assert abs(region_protected_overlap(r, protected) - 1.0) < 1e-6


# --- pipeline --------------------------------------------------------------
def test_pipeline_process_frame():
    frame = _noisy()
    pipe = AnonymizationPipeline(FixedBoxDetector(box=(10, 10, 40, 30), label="face"))
    redacted, regions = pipe.process_frame(frame)
    assert len(regions) == 1 and regions[0].label == "face"
    assert not np.array_equal(redacted[10:30, 10:40], frame[10:30, 10:40])


def test_pipeline_flags_conflict_but_still_redacts():
    frame = _noisy()
    pipe = AnonymizationPipeline(
        FixedBoxDetector(box=(5, 5, 25, 40), label="document"),
        zone_estimator=BoxZoneEstimator(box=(0, 0, 32, 48)),
    )
    redacted, regions = pipe.process_frame(frame)
    assert regions[0].status == "flagged"  # overlaps the interaction zone
    # fail-safe: a flagged region is still redacted in the output
    assert not np.array_equal(redacted[5:40, 5:25], frame[5:40, 5:25])


def test_pipeline_no_conflict_when_clear_of_zone():
    frame = _noisy()
    pipe = AnonymizationPipeline(
        FixedBoxDetector(box=(40, 5, 60, 40), label="face"),
        zone_estimator=BoxZoneEstimator(box=(0, 0, 32, 48)),
    )
    _, regions = pipe.process_frame(frame)
    assert regions[0].status == "redact"


def _write_video(path, n, h=48, w=64):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    for i in range(n):
        vw.write(_noisy(h, w, seed=i))
    vw.release()


def test_pipeline_process_video_preserves_frame_count():
    with tempfile.TemporaryDirectory() as d:
        src, dst = os.path.join(d, "in.mp4"), os.path.join(d, "out.mp4")
        _write_video(src, 9)
        pipe = AnonymizationPipeline(FixedBoxDetector(box=(5, 5, 30, 25), label="x"))
        result = pipe.process_video(src, dst)
        assert result.frames_written == 9
        assert os.path.exists(dst)
        assert video_io.count_frames(dst) == 9


def test_pipeline_process_video_collects_flags():
    with tempfile.TemporaryDirectory() as d:
        src, dst = os.path.join(d, "in.mp4"), os.path.join(d, "out.mp4")
        _write_video(src, 6)
        pipe = AnonymizationPipeline(
            FixedBoxDetector(box=(5, 5, 25, 40), label="document"),
            zone_estimator=BoxZoneEstimator(box=(0, 0, 32, 48)),
        )
        result = pipe.process_video(src, dst)
        assert result.frames_written == 6
        assert result.n_flagged == 6  # the region overlaps the zone every frame
        frame_idx, region = result.flagged[0]
        assert frame_idx == 0 and region.status == "flagged"
