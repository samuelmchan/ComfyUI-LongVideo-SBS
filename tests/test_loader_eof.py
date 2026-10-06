"""Loader meta-batch EOF regression.

ffprobe can over-report the frame count (AV1 commonly lists a trailing packet
that yields no picture). VHS turns that estimate into the meta-batch count, so a
source whose real decoded length is an exact multiple of ``frames_per_batch``
gets one batch too many and fails with ``No frames generated``. The VAAPI
generator must pin the meta-batch total to the decoder's actual EOF instead.

Run from the repository root:

    PYTHONPATH=. python tests/test_loader_eof.py
"""
from __future__ import annotations

import io

import numpy as np

import video_loader as vl


class _Progress:
    def __init__(self, *_a, **_k):
        self.count = 0

    def update(self, n):
        self.count += n


class _VHS:
    ProgressBar = _Progress
    ffmpeg_path = "/usr/bin/ffmpeg"


class _Proc:
    def __init__(self, payload):
        self.stdout = io.BytesIO(payload)
        self.stderr = io.BytesIO()
        self._rc = None

    def poll(self):
        return self._rc

    def wait(self, timeout=None):
        self._rc = 0
        return 0

    def terminate(self):
        self._rc = 0

    def kill(self):
        self._rc = 0


class _Meta:
    def __init__(self, frames_per_batch):
        self.frames_per_batch = frames_per_batch
        self.inputs = {}
        self.has_closed_inputs = False
        # Deliberately over-reported container estimate.
        self.total_frames = 9999


FRAMES = 6
payload = b"".join(bytes([i, i + 1, i + 2, i + 3, i + 4, i + 5]) for i in range(FRAMES))
probe = vl.VideoProbe(2, 1, 30.0, 0.2, FRAMES, "yuv420p", "av1", False)

_original = (vl._vhs_module, vl.probe_video, vl.build_vaapi_command, vl.subprocess.Popen)
vl._vhs_module = lambda: _VHS
vl.probe_video = lambda *_a, **_k: probe
vl.build_vaapi_command = lambda **_k: (["fake-ffmpeg"], 2, 1, np.dtype(np.uint8), 3)
vl.subprocess.Popen = lambda *_a, **_k: _Proc(payload)
try:
    for prefetch in (1, 0):
        meta = _Meta(frames_per_batch=4)
        gen = vl.vaapi_frame_generator(
            "tiny.mp4", 0, 0, 0, 0, 0,
            downscale_ratio=1, meta_batch=meta, unique_id="loader",
            vaapi_device="/dev/dri/renderD128", prefetch_batches=prefetch,
        )
        next(gen)
        meta.inputs["loader"] = (gen,)
        frames = 0
        while True:
            try:
                next(gen)
                frames += 1
            except StopIteration:
                break
        assert frames == FRAMES, (prefetch, frames)
        assert meta.total_frames == FRAMES, (prefetch, meta.total_frames)
finally:
    (vl._vhs_module, vl.probe_video, vl.build_vaapi_command, vl.subprocess.Popen) = _original

print("loader meta-batch EOF regression: PASS")
