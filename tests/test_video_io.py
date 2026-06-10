"""Tests for video_io.py — synthetic MP4s, no licensed data needed."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import cv2
import numpy as np

import video_io


def _make_video(path, n_frames, w=64, h=48, fps=30, color=(235, 10, 10)):
    """Write a solid-colour MP4 (mp4v). color is BGR."""
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert vw.isOpened(), f"could not open writer for {path}"
    frame = np.zeros((h, w, 3), np.uint8)
    frame[:] = color
    for _ in range(n_frames):
        vw.write(frame)
    vw.release()


def test_probe_and_count():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "v.mp4")
        _make_video(p, 12, w=64, h=48, fps=30)
        info = video_io.probe(p)
        assert info.width == 64 and info.height == 48
        assert abs(info.fps - 30) < 1.0
        assert video_io.count_frames(p) == 12


def test_read_frames_shape_and_limit():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "v.mp4")
        _make_video(p, 10)
        frames = list(video_io.read_frames(p))
        assert len(frames) == 10
        assert frames[0].shape == (48, 64, 3)
        assert len(list(video_io.read_frames(p, max_frames=4))) == 4


def test_bgr_rgb_channel_order():
    # frame is strongly blue in BGR: B channel high, R channel low.
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "v.mp4")
        _make_video(p, 6, color=(240, 0, 0))
        bgr = list(video_io.read_frames(p))[0]
        assert bgr[:, :, 0].mean() > 150  # blue dominant at index 0
        assert bgr[:, :, 2].mean() < 100
        rgb = list(video_io.read_frames(p, to_rgb=True))[0]
        assert rgb[:, :, 2].mean() > 150  # blue moved to index 2
        assert rgb[:, :, 0].mean() < 100


def test_writer_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "out.mp4")
        with video_io.VideoWriter(p, 30, (64, 48)) as w:
            for _ in range(8):
                w.write(np.full((48, 64, 3), 120, np.uint8))
        assert video_io.count_frames(p) == 8


def test_stereo_pairing_with_length_mismatch():
    with tempfile.TemporaryDirectory() as d:
        sess = os.path.join(d, "20260517_120000")
        os.makedirs(sess)
        _make_video(os.path.join(sess, "left.mp4"), 10)
        _make_video(os.path.join(sess, "right.mp4"), 6)
        pairs = list(video_io.read_stereo(sess))
        assert len(pairs) == 10  # length of the longer stream
        for i, (left, right) in enumerate(pairs):
            assert left is not None
            assert (right is not None) == (i < 6)


def test_session_pair_requires_two_files():
    with tempfile.TemporaryDirectory() as d:
        sess = os.path.join(d, "s")
        os.makedirs(sess)
        _make_video(os.path.join(sess, "only.mp4"), 3)
        try:
            video_io.find_session_pair(sess)
            assert False, "expected ValueError for a non-pair session"
        except ValueError:
            pass
