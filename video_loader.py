from __future__ import annotations

import importlib
import json
import math
import os
import queue
import shutil
import subprocess
import sys
import threading
import types
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np


class VAAPIPreflightError(RuntimeError):
    """Raised only before a VHS meta-batch generator has been registered."""




class _PrefetchError:
    __slots__ = ("exc",)

    def __init__(self, exc: BaseException):
        self.exc = exc


class _RawFramePrefetcher:
    """Continuously drain FFmpeg stdout into a bounded raw-frame queue.

    The queue deliberately stores the decoder's raw RGB24/RGB48 bytes instead of
    float32 ComfyUI IMAGE frames.  That lets VAAPI/FFmpeg keep decoding while the
    current meta-batch is in VDA/DIBR without doubling the prefetch RAM footprint.
    Conversion to float32 remains on the consumer thread and therefore preserves
    the exact image representation used by v3.3.1.
    """

    _EOF = object()

    def __init__(self, proc: subprocess.Popen, frame_bytes: int, max_frames: int):
        self.proc = proc
        self.frame_bytes = int(frame_bytes)
        self.max_frames = max(1, int(max_frames))
        self.frames: queue.Queue = queue.Queue(maxsize=self.max_frames)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._producer, name="LongVideo-VAAPI-Prefetch", daemon=True
        )
        self.thread.start()

    def _put(self, item) -> bool:
        while not self.stop_event.is_set():
            try:
                self.frames.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _read_one(self):
        stdout = self.proc.stdout
        if stdout is None:
            raise RuntimeError("FFmpeg VAAPI decoder stdout pipe is unavailable")
        buf = bytearray(self.frame_bytes)
        view = memoryview(buf)
        offset = 0
        while offset < self.frame_bytes and not self.stop_event.is_set():
            n = stdout.readinto(view[offset:])
            if n is None:
                continue
            if n == 0:
                break
            offset += n
        if self.stop_event.is_set():
            return None
        if offset == 0:
            return self._EOF
        if offset != self.frame_bytes:
            err = b"" if self.proc.stderr is None else self.proc.stderr.read()
            raise RuntimeError(
                f"VAAPI decoder ended with a partial frame ({offset}/{self.frame_bytes} bytes): "
                + err.decode("utf-8", errors="replace")
            )
        return buf

    def _producer(self):
        try:
            while not self.stop_event.is_set():
                # Do not read a (potentially ~50 MiB) frame until the bounded queue
                # has room. With a single producer this keeps queued + in-flight raw
                # frame memory at or below max_frames.
                while self.frames.full() and not self.stop_event.is_set():
                    self.stop_event.wait(0.05)
                if self.stop_event.is_set():
                    return
                item = self._read_one()
                if self.stop_event.is_set():
                    return
                if item is self._EOF:
                    rc = self.proc.wait()
                    if rc != 0:
                        err = b"" if self.proc.stderr is None else self.proc.stderr.read()
                        raise RuntimeError(
                            f"FFmpeg VAAPI decoder exited with code {rc}: "
                            + err.decode("utf-8", errors="replace")
                        )
                    self._put(self._EOF)
                    return
                if not self._put(item):
                    return
        except BaseException as exc:
            if not self.stop_event.is_set():
                self._put(_PrefetchError(exc))

    def get(self):
        item = self.frames.get()
        if item is self._EOF:
            raise StopIteration
        if isinstance(item, _PrefetchError):
            raise item.exc
        return item

    def close(self):
        self.stop_event.set()
        if self.proc.poll() is None:
            try:
                self.proc.terminate()
            except Exception:
                pass
        self.thread.join(timeout=2.0)
        if self.thread.is_alive() and self.proc.poll() is None:
            try:
                self.proc.kill()
            except Exception:
                pass
            self.thread.join(timeout=1.0)

@dataclass(frozen=True)
class VideoProbe:
    width: int
    height: int
    fps: float
    duration: float
    total_frames: int
    pix_fmt: str
    codec_name: str
    ten_bit: bool


_VHS_REQUIRED_ATTRS = (
    "load_video", "target_size", "ffmpeg_path", "strip_path", "ProgressBar",
)


def _valid_vhs_module(candidate):
    return candidate is not None and all(hasattr(candidate, name) for name in _VHS_REQUIRED_ATTRS)


def _module_from_vhs_node_class(cls):
    """Recover VHS load_video_nodes from an already-registered ComfyUI node class.

    ComfyUI can load a custom node package without adding that package's root to
    sys.path.  In that case ``import videohelpersuite`` fails even though VHS is
    installed and its node classes are live.  The registered LoadVideo class was
    defined inside load_video_nodes.py, so its method globals/module give us a
    reliable handle to the exact loaded VHS implementation.
    """
    if cls is None:
        return None

    module_name = getattr(cls, "__module__", "")
    candidate = sys.modules.get(module_name) if module_name else None
    if candidate is None and module_name:
        try:
            candidate = importlib.import_module(module_name)
        except Exception:
            candidate = None
    if _valid_vhs_module(candidate):
        return candidate

    # Some ComfyUI custom-node loaders use synthetic module names that are not
    # importable later.  A function's __globals__ is still the live module
    # namespace, so construct a tiny adapter from the exact objects VHS loaded.
    method = getattr(cls, "load_video", None)
    func = getattr(method, "__func__", method)
    globs = getattr(func, "__globals__", None)
    if isinstance(globs, dict) and all(name in globs for name in _VHS_REQUIRED_ATTRS):
        return types.SimpleNamespace(**{name: globs[name] for name in _VHS_REQUIRED_ATTRS})
    return None


def _vhs_module():
    # Fast path for installs where the VHS package root is directly importable.
    try:
        from videohelpersuite import load_video_nodes as vhs
        if _valid_vhs_module(vhs):
            return vhs
    except Exception:
        pass

    # Robust ComfyUI path: VHS may be loaded under a synthetic/custom-node module
    # name rather than as top-level ``videohelpersuite``.
    try:
        import nodes as comfy_nodes
        mappings = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {})
        for node_name in ("VHS_LoadVideo", "VHS_LoadVideoFFmpeg"):
            candidate = _module_from_vhs_node_class(mappings.get(node_name))
            if _valid_vhs_module(candidate):
                return candidate
    except Exception:
        pass

    # Last-resort discovery for custom loaders that registered the VHS module in
    # sys.modules but not under the normal package name.
    for module_name, candidate in tuple(sys.modules.items()):
        if module_name.endswith("videohelpersuite.load_video_nodes") and _valid_vhs_module(candidate):
            return candidate

    raise RuntimeError(
        "ComfyUI-VideoHelperSuite is loaded but LongVideo could not resolve its "
        "load_video_nodes module. Restart ComfyUI after updating VHS; if this "
        "continues, reinstall/enable ComfyUI-VideoHelperSuite."
    )


def _ffprobe_path(ffmpeg_path: str) -> str:
    sibling = os.path.join(os.path.dirname(os.path.abspath(ffmpeg_path)), "ffprobe")
    if os.path.isfile(sibling) and os.access(sibling, os.X_OK):
        return sibling
    return shutil.which("ffprobe") or "ffprobe"


def _parse_rate(value: str | None) -> float:
    if not value or value in {"0/0", "N/A"}:
        return 0.0
    try:
        return float(Fraction(value))
    except Exception:
        try:
            return float(value)
        except Exception:
            return 0.0


def probe_video(video: str, ffmpeg_path: str | None = None) -> VideoProbe:
    vhs = _vhs_module()
    ffmpeg_path = ffmpeg_path or vhs.ffmpeg_path
    ffprobe = _ffprobe_path(ffmpeg_path)
    cmd = [
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,pix_fmt,codec_name:format=duration",
        "-of", "json", video,
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        data = json.loads(res.stdout.decode("utf-8", errors="replace"))
    except (OSError, subprocess.CalledProcessError, ValueError, KeyError, IndexError) as exc:
        detail = ""
        if isinstance(exc, subprocess.CalledProcessError):
            detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise VAAPIPreflightError(f"ffprobe could not inspect {video!r}: {detail or exc}") from exc

    streams = data.get("streams") or []
    if not streams:
        raise VAAPIPreflightError(f"ffprobe found no video stream in {video!r}")
    st = streams[0]
    width = int(st.get("width") or 0)
    height = int(st.get("height") or 0)
    if width <= 0 or height <= 0:
        raise VAAPIPreflightError(f"ffprobe returned invalid dimensions for {video!r}")

    fps = _parse_rate(st.get("avg_frame_rate")) or _parse_rate(st.get("r_frame_rate")) or 1.0
    try:
        duration = float((data.get("format") or {}).get("duration") or 0.0)
    except Exception:
        duration = 0.0
    try:
        total_frames = int(st.get("nb_frames") or 0)
    except Exception:
        total_frames = 0
    if total_frames <= 0 and duration > 0:
        total_frames = int(round(duration * fps))
    if duration <= 0 and total_frames > 0 and fps > 0:
        duration = total_frames / fps

    pix_fmt = str(st.get("pix_fmt") or "").lower()
    codec_name = str(st.get("codec_name") or "unknown").lower()
    # ffprobe formats are normally yuv420p10le/p010le for 10-bit consumer video.
    ten_bit = any(token in pix_fmt for token in ("10", "12", "16", "p010", "p012", "p016"))
    return VideoProbe(width, height, fps, duration, total_frames, pix_fmt, codec_name, ten_bit)


def _download_format(probe: VideoProbe) -> tuple[str, str, np.dtype, int]:
    # Keep source precision through the pipe: 10-bit decode -> RGB48, 8-bit -> RGB24.
    if probe.ten_bit:
        return "p010le", "rgb48le", np.dtype("<u2"), 6
    return "nv12", "rgb24", np.dtype(np.uint8), 3


def _build_filters(probe: VideoProbe, force_rate: float, custom_width: int,
                   custom_height: int, downscale_ratio: int | None) -> tuple[list[str], int, int, str, np.dtype, int]:
    vhs = _vhs_module()
    surface_fmt, raw_fmt, dtype, bytes_per_pixel = _download_format(probe)
    filters = ["hwdownload", f"format={surface_fmt}"]
    if force_rate:
        filters.append(f"fps=fps={force_rate}")

    if custom_width or custom_height:
        size = vhs.target_size(
            probe.width, probe.height, int(custom_width), int(custom_height),
            downscale_ratio=downscale_ratio,
        )
        ar = float(size[0]) / float(size[1])
        if abs(probe.width * ar - probe.height) >= 1:
            filters.append(
                f"crop=if(gt({ar}\\,a)\\,iw\\,ih*{ar}):if(gt({ar}\\,a)\\,iw/{ar}\\,ih)"
            )
        filters.append(f"scale={size[0]}:{size[1]}")
        out_w, out_h = int(size[0]), int(size[1])
    else:
        out_w, out_h = probe.width, probe.height

    filters.append(f"format={raw_fmt}")
    return filters, out_w, out_h, raw_fmt, dtype, bytes_per_pixel


def build_vaapi_command(video: str, vaapi_device: str, probe: VideoProbe,
                        force_rate: float = 0.0, custom_width: int = 0,
                        custom_height: int = 0, frame_load_cap: int = 0,
                        downscale_ratio: int | None = 1,
                        ffmpeg_path: str | None = None) -> tuple[list[str], int, int, np.dtype, int]:
    vhs = _vhs_module()
    ffmpeg_path = ffmpeg_path or vhs.ffmpeg_path
    filters, out_w, out_h, raw_fmt, dtype, bpp = _build_filters(
        probe, float(force_rate), int(custom_width), int(custom_height), downscale_ratio
    )
    cmd = [
        ffmpeg_path, "-hide_banner", "-loglevel", "error", "-nostdin",
        "-hwaccel", "vaapi", "-hwaccel_device", vaapi_device,
        "-hwaccel_output_format", "vaapi",
        "-i", video, "-map", "0:v:0", "-an", "-sn", "-dn",
        "-vf", ",".join(filters), "-pix_fmt", raw_fmt,
    ]
    if frame_load_cap > 0:
        cmd += ["-frames:v", str(int(frame_load_cap))]
    cmd += ["-f", "rawvideo", "pipe:1"]
    return cmd, out_w, out_h, dtype, bpp


def preflight_vaapi(video: str, vaapi_device: str, probe: VideoProbe | None = None,
                    ffmpeg_path: str | None = None) -> VideoProbe:
    vhs = _vhs_module()
    ffmpeg_path = ffmpeg_path or vhs.ffmpeg_path
    if not vaapi_device:
        raise VAAPIPreflightError("VAAPI device is empty")
    if not os.path.exists(vaapi_device):
        raise VAAPIPreflightError(f"VAAPI device does not exist: {vaapi_device}")
    if not os.access(vaapi_device, os.R_OK | os.W_OK):
        raise VAAPIPreflightError(f"VAAPI device is not readable/writable: {vaapi_device}")

    probe = probe or probe_video(video, ffmpeg_path)
    filters, _w, _h, _raw, _dtype, _bpp = _build_filters(probe, 0.0, 0, 0, 1)
    # Decode one frame all the way through hwdownload. This catches unsupported
    # codec/profile/surface-format combinations before VHS registers a generator.
    cmd = [
        ffmpeg_path, "-hide_banner", "-loglevel", "error", "-nostdin",
        "-hwaccel", "vaapi", "-hwaccel_device", vaapi_device,
        "-hwaccel_output_format", "vaapi",
        "-i", video, "-map", "0:v:0", "-an", "-sn", "-dn",
        "-vf", ",".join(filters[:-1]), "-frames:v", "1", "-f", "null", "-",
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = ""
        if isinstance(exc, subprocess.CalledProcessError):
            detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise VAAPIPreflightError(
            f"VAAPI decode preflight failed for {probe.codec_name}/{probe.pix_fmt} on {vaapi_device}: "
            f"{detail or exc}"
        ) from exc
    return probe


def vaapi_frame_generator(video: str, force_rate: float, frame_load_cap: int,
                           start_time: float, custom_width: int, custom_height: int,
                           downscale_ratio: int = 8, meta_batch: Any = None,
                           unique_id: Any = None, vaapi_device: str = "/dev/dri/renderD128",
                           prefetch_batches: int = 1):
    """VHS-compatible generator using FFmpeg VAAPI decode and precision-aware raw RGB."""
    if start_time:
        # Production wrapper currently always uses 0. Keeping this explicit avoids
        # silently changing seek semantics if the generator is reused elsewhere.
        raise VAAPIPreflightError("LongVideo VAAPI production loader currently requires start_time=0")

    vhs = _vhs_module()
    probe = probe_video(video, vhs.ffmpeg_path)
    cmd, out_w, out_h, dtype, bytes_per_pixel = build_vaapi_command(
        video=video,
        vaapi_device=vaapi_device,
        probe=probe,
        force_rate=force_rate,
        custom_width=custom_width,
        custom_height=custom_height,
        frame_load_cap=frame_load_cap,
        downscale_ratio=downscale_ratio,
        ffmpeg_path=vhs.ffmpeg_path,
    )

    target_fps = float(force_rate or probe.fps)
    target_frame_time = 1.0 / target_fps
    yieldable_frames = target_fps * probe.duration if probe.duration > 0 else probe.total_frames
    if frame_load_cap > 0:
        yieldable_frames = min(yieldable_frames or frame_load_cap, frame_load_cap)
    total_frames = probe.total_frames or int(round(probe.fps * probe.duration))

    yield (
        probe.width, probe.height, probe.fps, probe.duration, total_frames,
        target_frame_time, yieldable_frames, out_w, out_h, False,
    )

    pbar = vhs.ProgressBar(yieldable_frames)
    frame_bytes = out_w * out_h * bytes_per_pixel
    denom = 65535.0 if bytes_per_pixel == 6 else 255.0

    proc = None
    prefetcher = None
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        assert proc.stdout is not None

        prefetch_batches = max(0, min(1, int(prefetch_batches)))
        if prefetch_batches and meta_batch is not None:
            prefetch_frames = max(1, int(getattr(meta_batch, "frames_per_batch", 1)))
            max_raw_mib = frame_bytes * prefetch_frames / (1024.0 * 1024.0)
            print(
                f"[LongVideo Loader] VAAPI prefetch=1: up to {prefetch_frames} raw frames "
                f"(~{max_raw_mib:.0f} MiB) decoded ahead"
            )
            prefetcher = _RawFramePrefetcher(proc, frame_bytes, prefetch_frames)

            while True:
                try:
                    raw = prefetcher.get()
                except StopIteration:
                    break
                arr = np.frombuffer(raw, dtype=dtype).reshape(out_h, out_w, 3)
                frame = arr.astype(np.float32)
                frame *= (1.0 / denom)
                control = yield frame
                pbar.update(1)
                # VHS BatchManager.close_inputs() sends a non-None value into live
                # generators. Honor it so the background decoder cannot outlive a
                # workflow reset/cancel.
                if control is not None:
                    break
        else:
            # Exact v3.3.1 synchronous path, retained for A/B testing/fallback.
            stopped_early = False
            buf = bytearray(frame_bytes)
            view = memoryview(buf)
            while True:
                offset = 0
                while offset < frame_bytes:
                    n = proc.stdout.readinto(view[offset:])
                    if n is None:
                        continue
                    if n == 0:
                        break
                    offset += n
                if offset == 0:
                    break
                if offset != frame_bytes:
                    err = b"" if proc.stderr is None else proc.stderr.read()
                    raise RuntimeError(
                        f"VAAPI decoder ended with a partial frame ({offset}/{frame_bytes} bytes): "
                        + err.decode("utf-8", errors="replace")
                    )
                arr = np.frombuffer(buf, dtype=dtype).reshape(out_h, out_w, 3)
                frame = arr.astype(np.float32)
                frame *= (1.0 / denom)
                control = yield frame
                pbar.update(1)
                if control is not None:
                    stopped_early = True
                    break

            if not stopped_early and proc.poll() is None:
                rc = proc.wait()
                if rc != 0:
                    err = b"" if proc.stderr is None else proc.stderr.read()
                    raise RuntimeError(
                        f"FFmpeg VAAPI decoder exited with code {rc}: "
                        + err.decode("utf-8", errors="replace")
                    )
    finally:
        if prefetcher is not None:
            prefetcher.close()
        elif proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except Exception:
                proc.kill()
        if meta_batch is not None:
            # Only the consumer removes the VHS input. The producer may hit EOF
            # while a prefetched final batch is still queued; removing it early
            # would make VHS reopen the file and duplicate frames.
            try:
                meta_batch.inputs.pop(unique_id, None)
                meta_batch.has_closed_inputs = True
            except Exception:
                pass


def load_video_vaapi(common: dict[str, Any], vaapi_device: str, prefetch_batches: int = 1):
    """Run VHS load_video with our VAAPI generator while keeping VHS batch/audio semantics."""
    import folder_paths

    vhs = _vhs_module()
    kwargs = dict(common)
    raw_video = vhs.strip_path(kwargs.get("video") or "")
    if not str(raw_video).strip():
        raise VAAPIPreflightError(
            "No video selected. Choose a video in LV_ProductionVideoLoader before running."
        )
    kwargs["video"] = folder_paths.get_annotated_filepath(raw_video)
    if not os.path.isfile(kwargs["video"]):
        raise VAAPIPreflightError(
            f"Selected video path is not a file: {kwargs['video']!r}. Choose a video file in LV_ProductionVideoLoader."
        )
    meta_batch = kwargs.get("meta_batch")
    unique_id = kwargs.get("unique_id")

    # Only preflight when creating a new generator; continuing meta-batches must never
    # reopen the file or re-run a decoder probe.
    if meta_batch is None or unique_id not in getattr(meta_batch, "inputs", {}):
        probe = probe_video(kwargs["video"], vhs.ffmpeg_path)
        preflight_vaapi(kwargs["video"], vaapi_device, probe, vhs.ffmpeg_path)

    image, frame_count, audio, video_info = vhs.load_video(
        **kwargs,
        generator=lambda **gen_kwargs: vaapi_frame_generator(
            **gen_kwargs, vaapi_device=vaapi_device, prefetch_batches=prefetch_batches
        ),
        start_time=0.0,
    )
    return image, int(frame_count), audio, video_info
