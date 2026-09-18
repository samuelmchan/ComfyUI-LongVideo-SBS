from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class LongVideoSession:
    key: Tuple[Any, ...]
    config_signature: Tuple[Any, ...]
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)

    # Source / window accounting.
    total_frames: Optional[int] = None
    input_count: int = 0
    next_window_start: int = 0
    window_index: int = 0
    finished: bool = False
    source_name: Optional[str] = None

    # Raw RGB buffer. Each entry is a CPU BHWC frame tensor [H,W,C].
    raw_buffer: List[torch.Tensor] = field(default_factory=list)
    last_source_frame: Optional[torch.Tensor] = None

    # Official VDA temporal-reuse state. We retain only the ten keyframes
    # needed by the next 32-frame window rather than the full previous input.
    pre_keyframes: Optional[torch.Tensor] = None

    # Alignment state matching VDA's offline algorithm.
    ref_align: List[Any] = field(default_factory=list)
    pending_depth: List[Any] = field(default_factory=list)
    pending_rgb: List[torch.Tensor] = field(default_factory=list)

    # Long-video normalization state used after VDA alignment.
    norm_low: Optional[float] = None
    norm_high: Optional[float] = None
    prev_scene_thumb: Optional[torch.Tensor] = None

    # Optional DEV2 stereo-comfort temporal state. Kept on CPU and only used
    # when temporal_disparity_stability > 0.
    stereo_prev_depth: Optional[torch.Tensor] = None

    # Diagnostic counters.
    emitted_frames: int = 0
    warnings: List[str] = field(default_factory=list)

    lock: threading.RLock = field(default_factory=threading.RLock)

    def touch(self) -> None:
        self.last_used = time.time()

    def set_total_from_meta_batch(self, meta_batch: Any) -> None:
        total = getattr(meta_batch, "total_frames", None)
        if total is None:
            return
        try:
            if math.isfinite(float(total)):
                self.total_frames = max(0, int(round(float(total))))
        except (TypeError, ValueError):
            pass

    def clear_gpu_state(self) -> None:
        self.pre_keyframes = None


class SessionRegistry:
    def __init__(self) -> None:
        self._sessions: Dict[Tuple[Any, ...], LongVideoSession] = {}
        self._lock = threading.RLock()

    def get(self, key: Tuple[Any, ...]) -> Optional[LongVideoSession]:
        with self._lock:
            return self._sessions.get(key)

    def get_or_create(
        self,
        key: Tuple[Any, ...],
        config_signature: Tuple[Any, ...],
        force_new: bool = False,
    ) -> LongVideoSession:
        with self._lock:
            current = self._sessions.get(key)
            if (
                force_new
                or current is None
                or current.finished
                or current.config_signature != config_signature
            ):
                current = LongVideoSession(
                    key=key,
                    config_signature=config_signature,
                )
                self._sessions[key] = current
            current.touch()
            return current

    def delete(self, key: Tuple[Any, ...]) -> None:
        with self._lock:
            session = self._sessions.pop(key, None)
            if session is not None:
                session.clear_gpu_state()

    def cleanup(self, max_age_seconds: float = 24 * 3600) -> None:
        now = time.time()
        with self._lock:
            dead = [
                key
                for key, session in self._sessions.items()
                if (now - session.last_used) > max_age_seconds
            ]
            for key in dead:
                self._sessions[key].clear_gpu_state()
                del self._sessions[key]


SESSIONS = SessionRegistry()
