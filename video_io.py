"""Video I/O for the licensed egocentric corpus.

Format contract (``context/03_capture_pipeline_output.md``): MP4 / ``mp4v``
codec, 960x1080 per eye, ~30 fps, BGR pixel order, no audio. One recording
session is a directory holding two MP4 files — the stereo pair. The two files
are time-aligned at capture but **not** guaranteed frame-index identical, and
either may be truncated; the pipeline treats them as parallel-but-temporally-
independent and processes whatever frames are present.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    header_frame_count: int  # from the header; unreliable if the file is truncated


def probe(path) -> VideoInfo:
    """Read a video's properties without decoding it fully."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")
    try:
        return VideoInfo(
            path=str(path),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(cap.get(cv2.CAP_PROP_FPS)),
            header_frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()


def read_frames(path, to_rgb=False, max_frames=None):
    """Yield frames in linear order.

    Stops cleanly at EOF or at a truncation point — a short read is treated as
    end-of-stream, not an error, so a truncated session still yields all of its
    valid frames.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")
    try:
        i = 0
        while max_frames is None or i < max_frames:
            ok, frame = cap.read()
            if not ok:
                break  # EOF or truncation
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if to_rgb else frame
            i += 1
    finally:
        cap.release()


def count_frames(path) -> int:
    """Actual decodable frame count (robust to a wrong header count)."""
    return sum(1 for _ in read_frames(path))


def find_session_pair(session_dir):
    """Return the two MP4 files of a session directory, sorted by filename.

    Exact filenames are an implementation detail of the corpus; the pipeline
    just takes whichever two MP4s are present as the stereo pair.
    """
    mp4s = sorted(Path(session_dir).glob("*.mp4"))
    if len(mp4s) != 2:
        raise ValueError(
            f"expected exactly 2 MP4 files in {session_dir}, found {len(mp4s)}"
        )
    return mp4s[0], mp4s[1]


def read_stereo(session_dir, to_rgb=False, max_frames=None):
    """Yield ``(left_frame, right_frame)`` pairs for a session.

    The two streams may differ in length; once one is exhausted its slot is
    ``None`` while the other continues. Iteration stops when both are done.
    """
    left, right = find_session_pair(session_dir)
    gen_l = read_frames(left, to_rgb, max_frames)
    gen_r = read_frames(right, to_rgb, max_frames)
    sentinel = object()
    while True:
        frame_l = next(gen_l, sentinel)
        frame_r = next(gen_r, sentinel)
        if frame_l is sentinel and frame_r is sentinel:
            break
        yield (
            None if frame_l is sentinel else frame_l,
            None if frame_r is sentinel else frame_r,
        )


class VideoWriter:
    """Write an anonymized MP4 with the corpus codec (``mp4v``).

    Use as a context manager. ``from_rgb=True`` converts RGB frames back to BGR
    before writing.
    """

    def __init__(self, path, fps, size, from_rgb=False):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(path), fourcc, float(fps or 30.0), size)
        if not self._writer.isOpened():
            raise IOError(f"cannot open video writer: {path}")
        self._from_rgb = from_rgb

    def write(self, frame):
        if self._from_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self._writer.write(frame)

    def close(self):
        self._writer.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
