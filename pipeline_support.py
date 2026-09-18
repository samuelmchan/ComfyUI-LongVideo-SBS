import atexit
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    import folder_paths
except Exception:
    folder_paths = None


CATEGORY = "Long Video SBS"
_ENCODERS = {}
_ENCODERS_LOCK = threading.Lock()


def _stream_run_token(stream):
    """Return a token that is stable across meta-batches but changes on a new prompt run."""
    state = getattr(stream, "state", None)
    state_key = getattr(stream, "state_key", None)
    if isinstance(state_key, tuple) and len(state_key) == 2:
        return (int(state_key[0]), str(state_key[1]), id(state))
    return ("state", id(state))


def _format_ifnet_stage_profile(stage_seconds, stage_calls, unattributed, profiled_forwards):
    """Compact stable ordering for DEV2.6 IFNet stage telemetry."""
    stage_seconds = dict(stage_seconds or {})
    stage_calls = dict(stage_calls or {})
    preferred = ["encode"] + [f"block{i}" for i in range(8)]
    names = [name for name in preferred if name in stage_seconds]
    names += sorted(name for name in stage_seconds if name not in names)
    parts = [
        f"{name}={float(stage_seconds.get(name, 0.0)):.3f}s/{int(stage_calls.get(name, 0))}x"
        for name in names
    ]
    parts.append(f"outer={float(unattributed):.3f}s")
    parts.append(f"profiled={int(profiled_forwards)}fwd")
    return " ".join(parts)


def _retire_stale_encoder_for_run(key, run_token):
    """Abort stale FFmpeg state when the same Comfy node starts a new video run.

    Comfy/VHS requeues reuse the node UNIQUE_ID, so UNIQUE_ID alone cannot
    distinguish a legitimate next meta-batch from a user-cancelled prompt that is
    started again.  The interpolation state/run token can.
    """
    stale = None
    with _ENCODERS_LOCK:
        state = _ENCODERS.get(key)
        if state is not None and getattr(state, "_longvideo_run_token", None) != run_token:
            stale = _ENCODERS.pop(key, None)
            state = None
    if stale is not None:
        try:
            stale.abort(remove_partial=True)
        except TypeError:
            stale.abort()
        print(
            "[LongVideo Stream Encoder] retired stale interrupted encoder state "
            "before starting the new run"
        )
    return state


def _close_meta_batch_inputs_for_final(meta_batch):
    """Close suspended VHS input generators after our downstream stream knows EOF.

    VHS normally discovers EOF by requesting one frame beyond the final frame.  When
    the source length is an exact multiple of frames_per_batch, that extra request
    would require a whole extra workflow requeue.  The LongVideo RIFE stream already
    has authoritative final-batch state from the VDA session, so on that final batch
    we can safely retire any still-suspended VHS input generator ourselves.

    Clear the dictionary *before* closing generators.  Several VHS/custom generators
    remove their own entry in a finally block; pre-clearing makes that cleanup
    idempotent and avoids mutating a dict while BatchManager is iterating it.
    """
    if meta_batch is None:
        return 0

    inputs = getattr(meta_batch, "inputs", None)
    entries = []
    if isinstance(inputs, dict) and inputs:
        entries = list(inputs.values())
        inputs.clear()

    closed = 0
    for entry in entries:
        gen = entry[0] if isinstance(entry, (tuple, list)) and entry else entry
        close = getattr(gen, "close", None)
        if callable(close):
            try:
                close()
                closed += 1
            except Exception as exc:
                print(f"[LongVideo Stream Encoder] warning: loader generator close failed: {exc}")

    try:
        meta_batch.has_closed_inputs = True
    except Exception:
        pass
    return closed


def _as_depth_bchw(depth: torch.Tensor):
    """Return depth as Bx1xHxW plus a small token describing original layout."""
    if depth.ndim == 3:  # B,H,W
        return depth.unsqueeze(1), "bhw"
    if depth.ndim == 4:
        if depth.shape[1] == 1:  # B,1,H,W
            return depth, "b1hw"
        if depth.shape[-1] == 1:  # B,H,W,1
            return depth.permute(0, 3, 1, 2), "bhw1"
    raise ValueError(f"Unsupported LV_DEPTH shape: {tuple(depth.shape)}")


def _restore_depth_layout(depth_bchw: torch.Tensor, layout: str):
    if layout == "bhw":
        return depth_bchw[:, 0]
    if layout == "b1hw":
        return depth_bchw
    if layout == "bhw1":
        return depth_bchw.permute(0, 2, 3, 1)
    raise ValueError(layout)


def _box_mean(x: torch.Tensor, radius: int):
    if radius <= 0:
        return x
    k = radius * 2 + 1
    # A box filter is separable. Two 1D averages are numerically equivalent
    # (apart from tiny FP rounding) to a kxk average, while doing far less work
    # for the production radius=4 (9+9 taps instead of 81 taps per pixel).
    x = F.avg_pool2d(
        x, kernel_size=(1, k), stride=1, padding=(0, radius),
        count_include_pad=False
    )
    return F.avg_pool2d(
        x, kernel_size=(k, 1), stride=1, padding=(radius, 0),
        count_include_pad=False
    )


def _fast_depth_guide(rgb: torch.Tensor, target_hw):
    """
    Build a BT.709 luma guide near the depth-map resolution without first
    converting/resizing the entire full-resolution RGB batch.

    For 4K -> ~518p depth this reduces the RGB FP32 working set by ~16x:
    take a centered strided view first, convert only that smaller view to FP32,
    then do at most a small final bilinear resize.
    """
    if rgb.ndim != 4 or rgb.shape[-1] < 3:
        raise ValueError(f"Expected RGB IMAGE BxHxWxC, got {tuple(rgb.shape)}")

    th, tw = int(target_hw[0]), int(target_hw[1])
    sh, sw = int(rgb.shape[1]), int(rgb.shape[2])

    # Round rather than floor so common 4K -> 518p lands around a 4x stride
    # instead of unnecessarily retaining a ~1280p intermediate.
    sy = max(1, int(round(sh / max(1, th))))
    sx = max(1, int(round(sw / max(1, tw))))
    oy = sy // 2 if sy > 1 else 0
    ox = sx // 2 if sx > 1 else 0

    # Slice before float conversion. This is the main performance win because
    # Comfy IMAGE tensors can be multi-gigabyte at 4K x 44 frames.
    sampled = rgb[:, oy::sy, ox::sx, :3]
    g = sampled.permute(0, 3, 1, 2).float()

    if g.shape[-2:] != (th, tw):
        g = F.interpolate(g, size=(th, tw), mode="bilinear", align_corners=False)

    # BT.709 luma edge guide, normalized to 0..1.
    return (0.2126 * g[:, 0:1] + 0.7152 * g[:, 1:2] + 0.0722 * g[:, 2:3]).clamp(0, 1)


def edge_aware_guided_smooth(rgb: torch.Tensor, depth: torch.Tensor, strength: float,
                             radius: int, edge_protection: float):
    """
    Guided-filter depth smoothing using RGB luminance as guidance.
    Runs at the depth-map resolution, so radius is stable across source resolutions.
    """
    if strength <= 0.0 or radius <= 0:
        return depth

    d, layout = _as_depth_bchw(depth)
    out_device = d.device
    out_dtype = d.dtype

    # Compute in fp32 for stable local statistics.
    d32 = d.float()

    I = _fast_depth_guide(rgb, d32.shape[-2:])

    # The original implementation ran four independent box filters here.
    # Pool the statistics as channels in one pass instead; this is exactly the
    # same local-mean math with much less launch/loop overhead.
    stats = torch.cat((I, d32, I * I, I * d32), dim=1)
    stats_mean = _box_mean(stats, radius)
    mean_I, mean_p, corr_I, corr_Ip = stats_mean.chunk(4, dim=1)

    var_I = (corr_I - mean_I * mean_I).clamp_min(0)
    cov_Ip = corr_Ip - mean_I * mean_p

    # High edge_protection -> smaller epsilon -> stronger respect for RGB edges.
    ep = float(max(0.0, min(1.0, edge_protection)))
    eps = 1.0e-6 + ((1.0 - ep) ** 2) * 2.5e-2
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    # Same optimization for the final two local means: one 2-channel pool.
    mean_ab = _box_mean(torch.cat((a, b), dim=1), radius)
    mean_a, mean_b = mean_ab.chunk(2, dim=1)
    q = mean_a * I + mean_b

    s = float(max(0.0, min(1.0, strength)))
    result = torch.lerp(d32, q, s).clamp(0, 1).to(dtype=out_dtype, device=out_device)
    return _restore_depth_layout(result, layout)


class LVEdgeAwareDepthSmooth:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rgb": ("IMAGE",),
                "normalized_depth": ("LV_DEPTH",),
                "strength": ("FLOAT", {"default": 0.14, "min": 0.0, "max": 1.0, "step": 0.01}),
                "radius": ("INT", {"default": 4, "min": 0, "max": 32, "step": 1}),
                "edge_protection": ("FLOAT", {"default": 0.80, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LV_DEPTH",)
    RETURN_NAMES = ("smoothed_depth",)
    FUNCTION = "smooth"
    CATEGORY = CATEGORY

    def smooth(self, rgb, normalized_depth, strength, radius, edge_protection):
        return (edge_aware_guided_smooth(rgb, normalized_depth, strength, radius, edge_protection),)


def _depth_edge_magnitude(depth_bchw: torch.Tensor) -> torch.Tensor:
    """Return a same-size local depth-gradient magnitude in Bx1xHxW."""
    dx = torch.zeros_like(depth_bchw)
    dy = torch.zeros_like(depth_bchw)
    dx[..., :, 1:] = (depth_bchw[..., :, 1:] - depth_bchw[..., :, :-1]).abs()
    dy[..., 1:, :] = (depth_bchw[..., 1:, :] - depth_bchw[..., :-1, :]).abs()
    return torch.maximum(dx, dy)


def _luma_edge_magnitude(luma_bchw: torch.Tensor) -> torch.Tensor:
    dx = torch.zeros_like(luma_bchw)
    dy = torch.zeros_like(luma_bchw)
    dx[..., :, 1:] = (luma_bchw[..., :, 1:] - luma_bchw[..., :, :-1]).abs()
    dy[..., 1:, :] = (luma_bchw[..., 1:, :] - luma_bchw[..., :-1, :]).abs()
    return torch.maximum(dx, dy)


def tune_depth_for_stereo(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    depth_range_scale: float = 0.90,
    zero_parallax: float = 0.15,
    near_compression: float = 0.0,
    far_compression: float = 0.0,
    feather_enabled: bool = False,
    feather_mode: str = "depth_edges",
    feather_strength: float = 0.15,
    feather_radius: int = 1,
    edge_threshold: float = 0.04,
    rgb_guidance: float = 0.50,
):
    """Stereo-comfort remap with zero-parallax as the single fixed pivot.

    DEV2 intentionally removes a separate range-center control.  Range scaling
    and near/far compression are anchored to the same normalized depth value
    that DIBR uses as zero parallax, so changing depth range cannot silently
    move the screen-plane reference.
    """
    rs = float(depth_range_scale)
    c = float(max(0.0, min(1.0, zero_parallax)))
    near_c = float(max(0.0, min(0.95, near_compression)))
    far_c = float(max(0.0, min(0.95, far_compression)))
    fs = float(max(0.0, min(1.0, feather_strength)))
    fr = max(0, int(feather_radius))
    feather_active = bool(feather_enabled) and fs > 0.0 and fr > 0

    if abs(rs - 1.0) <= 1e-12 and near_c <= 0.0 and far_c <= 0.0 and not feather_active:
        return depth

    d, layout = _as_depth_bchw(depth)
    out_device = d.device
    out_dtype = d.dtype
    x = d.float()

    if abs(rs - 1.0) > 1e-12:
        x = c + (x - c) * rs

    # Compress only the extremes while preserving zero parallax exactly.
    if near_c > 0.0 and c < 1.0:
        delta = (x - c).clamp_min(0.0)
        t = (delta / max(1.0 - c, 1e-6)).clamp(0.0, 1.0)
        near = c + delta * (1.0 - near_c * t)
        x = torch.where(x > c, near, x)
    if far_c > 0.0 and c > 0.0:
        delta = (c - x).clamp_min(0.0)
        t = (delta / max(c, 1e-6)).clamp(0.0, 1.0)
        far = c - delta * (1.0 - far_c * t)
        x = torch.where(x < c, far, x)

    x = x.clamp(0.0, 1.0)

    if feather_active:
        mode = str(feather_mode)
        if mode not in ("depth_edges", "rgb_guided"):
            raise ValueError("feather_mode must be depth_edges or rgb_guided")

        thr = max(1e-6, float(edge_threshold))
        edge = _depth_edge_magnitude(x)
        mask = ((edge - thr) / thr).clamp(0.0, 1.0)
        k = fr * 2 + 1
        mask = F.max_pool2d(mask, kernel_size=k, stride=1, padding=fr)

        if mode == "rgb_guided":
            guide = _fast_depth_guide(rgb, x.shape[-2:])
            rgb_edge = _luma_edge_magnitude(guide)
            rgb_mask = (rgb_edge / 0.08).clamp(0.0, 1.0)
            g = float(max(0.0, min(1.0, rgb_guidance)))
            mask = mask * ((1.0 - g) + g * rgb_mask)

        softened = _box_mean(x, fr)
        x = torch.lerp(x, softened, mask * fs).clamp(0.0, 1.0)

    result = x.to(device=out_device, dtype=out_dtype)
    return _restore_depth_layout(result, layout)


def _stabilize_stereo_depth(depth: torch.Tensor, session, stability: float, scene_cut_threshold: float = 0.20):
    """EMA the final tuned depth field across frames, resetting on large cuts.

    With depth_gamma=1 this is equivalent to stabilizing the disparity-driving
    field.  A value of 0 is a hard bypass and clears stale state.
    """
    a = float(max(0.0, min(0.95, stability)))
    if session is None:
        return depth
    if a <= 0.0:
        try:
            session.stereo_prev_depth = None
        except Exception:
            pass
        return depth

    d, layout = _as_depth_bchw(depth)
    device, dtype = d.device, d.dtype
    x = d.float()
    out = torch.empty_like(x)
    prev = getattr(session, "stereo_prev_depth", None)
    if isinstance(prev, torch.Tensor):
        prev = prev.to(device=x.device, dtype=x.dtype)
        if tuple(prev.shape) != tuple(x[:1].shape):
            prev = None

    threshold = float(max(0.0, scene_cut_threshold))
    for i in range(x.shape[0]):
        cur = x[i:i+1]
        if prev is None:
            mixed = cur
        else:
            mad = float((cur - prev).abs().mean().item())
            mixed = cur if mad >= threshold else torch.lerp(cur, prev, a)
        out[i:i+1] = mixed
        prev = mixed.detach()

    try:
        session.stereo_prev_depth = prev.detach().cpu() if prev is not None else None
    except Exception:
        pass
    return _restore_depth_layout(out.to(device=device, dtype=dtype), layout)


class LVDepthStereoTuning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rgb": ("IMAGE",),
                "normalized_depth": ("LV_DEPTH",),
                "session": ("LV_SESSION",),
                "zero_parallax": ("FLOAT", {"default": 0.00, "min": 0.00, "max": 1.00, "step": 0.01}),
                "depth_range_scale": ("FLOAT", {
                    "default": 1.00, "min": 0.25, "max": 1.50, "step": 0.01,
                    "display": "slider",
                }),
                "near_compression": ("FLOAT", {
                    "default": 0.00, "min": 0.00, "max": 0.95, "step": 0.01,
                    "display": "slider",
                }),
                "far_compression": ("FLOAT", {
                    "default": 0.00, "min": 0.00, "max": 0.95, "step": 0.01,
                    "display": "slider",
                }),
                "temporal_disparity_stability": ("FLOAT", {
                    "default": 0.15, "min": 0.00, "max": 0.95, "step": 0.01,
                    "display": "slider",
                    "tooltip": "Temporal stabilization of tuned depth immediately before DIBR. 0.00 = off; higher values retain more prior-frame depth.",
                }),
                "feather_enabled": ("BOOLEAN", {"default": False}),
                "feather_mode": (["depth_edges", "rgb_guided"], {"default": "depth_edges"}),
                "feather_strength": ("FLOAT", {
                    "default": 0.15, "min": 0.00, "max": 1.00, "step": 0.01,
                    "display": "slider",
                }),
                "feather_radius": ("INT", {"default": 1, "min": 1, "max": 6, "step": 1}),
                "edge_threshold": ("FLOAT", {
                    "default": 0.04, "min": 0.005, "max": 0.25, "step": 0.005,
                }),
                "rgb_guidance": ("FLOAT", {
                    "default": 0.50, "min": 0.00, "max": 1.00, "step": 0.05,
                }),
            }
        }

    RETURN_TYPES = ("LV_DEPTH", "STRING")
    RETURN_NAMES = ("tuned_depth", "status")
    FUNCTION = "tune"
    CATEGORY = CATEGORY

    def tune(
        self, rgb, normalized_depth, session, zero_parallax, depth_range_scale,
        near_compression, far_compression, temporal_disparity_stability,
        feather_enabled, feather_mode, feather_strength, feather_radius,
        edge_threshold, rgb_guidance,
    ):
        out = tune_depth_for_stereo(
            rgb=rgb,
            depth=normalized_depth,
            depth_range_scale=depth_range_scale,
            zero_parallax=zero_parallax,
            near_compression=near_compression,
            far_compression=far_compression,
            feather_enabled=feather_enabled,
            feather_mode=feather_mode,
            feather_strength=feather_strength,
            feather_radius=feather_radius,
            edge_threshold=edge_threshold,
            rgb_guidance=rgb_guidance,
        )
        out = _stabilize_stereo_depth(out, session, temporal_disparity_stability)
        status = (
            f"range={float(depth_range_scale):.2f} pivot=zero_parallax:{float(zero_parallax):.2f} "
            f"near_comp={float(near_compression):.2f} far_comp={float(far_compression):.2f} "
            f"temporal={float(temporal_disparity_stability):.2f} "
            f"feather={'on' if feather_enabled else 'off'}:{feather_mode} "
            f"strength={float(feather_strength):.2f} radius={int(feather_radius)} "
            f"threshold={float(edge_threshold):.3f} rgb_guide={float(rgb_guidance):.2f}"
        )
        return (out, status)


class LVResolutionStereoControls:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rgb": ("IMAGE",),
                "output_mode": (["half_sbs", "full_sbs"], {"default": "half_sbs"}),
                "stereo_strength_percent": ("FLOAT", {
                    "default": 1.40, "min": 0.10, "max": 5.00, "step": 0.05,
                    "display": "slider"
                }),
                "zero_parallax": ("FLOAT", {
                    "default": 0.00, "min": 0.00, "max": 1.00, "step": 0.01,
                    "display": "slider"
                }),
                "hole_fill_multiplier": ("FLOAT", {
                    "default": 3.00, "min": 0.50, "max": 10.00, "step": 0.10
                }),
            }
        }

    RETURN_TYPES = ("FLOAT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("max_disparity_eye_px", "max_fill_distance", "zero_parallax", "status")
    FUNCTION = "calculate"
    CATEGORY = CATEGORY

    def calculate(self, rgb, output_mode, stereo_strength_percent, zero_parallax, hole_fill_multiplier):
        if rgb.ndim != 4:
            raise ValueError(f"Expected IMAGE BxHxWxC, got {tuple(rgb.shape)}")
        source_w = int(rgb.shape[2])
        eye_w = source_w // 2 if output_mode == "half_sbs" else source_w
        disparity = eye_w * (float(stereo_strength_percent) / 100.0)
        total_lr = disparity * 2.0
        total_ratio = (total_lr / max(1, eye_w)) * 100.0
        fill = max(1, int(round(disparity * float(hole_fill_multiplier))))
        zp = float(max(0.0, min(1.0, zero_parallax)))
        status = (
            f"source={source_w}px eye={eye_w}px strength={float(stereo_strength_percent):.2f}% | "
            f"max={disparity:.2f}px/eye total_LR={total_lr:.2f}px ({total_ratio:.2f}% eye width) | "
            f"zero_parallax={zp:.2f} fill={fill}px ({float(hole_fill_multiplier):.2f}x)"
        )
        return (float(disparity), fill, zp, status)


def _find_vhs_utils():
    """Import VHS utilities without requiring a fixed custom-node import name."""
    try:
        from videohelpersuite.utils import ffmpeg_path, requeue_workflow_unchecked
        return ffmpeg_path, requeue_workflow_unchecked
    except Exception:
        here = Path(__file__).resolve().parent
        candidates = [
            here.parent / "ComfyUI-VideoHelperSuite",
            here.parent / "comfyui-videohelpersuite",
        ]
        for root in candidates:
            if (root / "videohelpersuite" / "utils.py").is_file():
                if str(root) not in sys.path:
                    sys.path.insert(0, str(root))
                from videohelpersuite.utils import ffmpeg_path, requeue_workflow_unchecked
                return ffmpeg_path, requeue_workflow_unchecked
        return shutil.which("ffmpeg"), None


def _next_output_path(prefix: str, save_output: bool):
    if folder_paths is None:
        raise RuntimeError("ComfyUI folder_paths is unavailable")
    output_dir = folder_paths.get_output_directory() if save_output else folder_paths.get_temp_directory()
    full_output_folder, filename, _, subfolder, _ = folder_paths.get_save_image_path(prefix, output_dir)
    os.makedirs(full_output_folder, exist_ok=True)
    matcher = re.compile(rf"{re.escape(filename)}_(\d+)\D*\.mp4", re.IGNORECASE)
    max_counter = 0
    for existing in os.listdir(full_output_folder):
        m = matcher.fullmatch(existing)
        if m:
            max_counter = max(max_counter, int(m.group(1)))
    counter = max_counter + 1
    return full_output_folder, filename, subfolder, counter


def _sanitize_source_stem(name: str) -> str:
    base = os.path.basename(str(name or "").strip())
    stem, _ext = os.path.splitext(base)
    stem = stem or base or "video"
    # Keep spaces and ordinary Unicode; only remove path/control characters and
    # the Windows-reserved punctuation that commonly causes portable-name issues.
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", stem).strip(" .")
    return stem or "video"


def _resolve_stream_filename_prefix(filename_prefix: str, filename_mode: str,
                                    filename_suffix: str, source_name: str) -> str:
    mode = str(filename_mode)
    if mode == "custom":
        return str(filename_prefix)
    if mode != "source_name":
        raise ValueError(f"Unknown filename_mode: {filename_mode}")
    if not source_name:
        raise ValueError(
            "filename_mode=source_name requested, but no source filename reached the RIFE stream. "
            "Use the LongVideo Production Video Loader or switch filename_mode to custom."
        )
    raw_prefix = str(filename_prefix or "")
    # In source-name mode the prefix field is treated as an output folder. A
    # trailing slash therefore means exactly that folder, rather than having
    # dirname() discard the final component.
    if raw_prefix.endswith(("/", "\\")):
        folder = raw_prefix.rstrip("/\\")
    else:
        folder = os.path.dirname(raw_prefix.rstrip("/\\"))
    suffix = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(filename_suffix or ""))
    stem = _sanitize_source_stem(source_name) + suffix
    return os.path.join(folder, stem) if folder else stem


def _next_source_named_output_path(prefix: str, save_output: bool):
    """Return a human-readable source-name path without underscore counters.

    First choice is exactly ``<source> Half-SBS.mp4``. Existing files are never
    overwritten; collisions become ``<source> Half-SBS (2).mp4`` etc.
    """
    if folder_paths is None:
        raise RuntimeError("ComfyUI folder_paths is unavailable")
    output_dir = folder_paths.get_output_directory() if save_output else folder_paths.get_temp_directory()
    full_output_folder, filename, _, subfolder, _ = folder_paths.get_save_image_path(prefix, output_dir)
    os.makedirs(full_output_folder, exist_ok=True)
    stem = filename
    if not os.path.exists(os.path.join(full_output_folder, stem + ".mp4")):
        return full_output_folder, stem, subfolder
    n = 2
    while os.path.exists(os.path.join(full_output_folder, f"{filename} ({n}).mp4")):
        n += 1
    return full_output_folder, f"{filename} ({n})", subfolder


def _bulk_rgb_cpu(images: torch.Tensor, bit_depth="8bit"):
    """Vectorized RGB handoff for the persistent FFmpeg pipe.

    8-bit uses packed rgb24, matching the historical production path.
    10-bit uses packed rgb48le so ComfyUI's float IMAGE precision is not
    quantized down to 8-bit before FFmpeg converts it to a 10-bit YUV surface.
    """
    t0 = time.perf_counter()
    if not isinstance(images, torch.Tensor):
        images = torch.as_tensor(images)
    if images.ndim != 4 or images.shape[-1] not in (3, 4):
        raise ValueError(f"Expected IMAGE BxHxWx3/4, got {tuple(images.shape)}")
    images = images[..., :3]
    if str(bit_depth) == "10bit":
        out = images.clamp(0, 1).mul(65535.0).add_(0.5).to(torch.uint16)
        raw_pix_fmt = "rgb48le"
    elif str(bit_depth) == "8bit":
        out = images.clamp(0, 1).mul(255.0).add_(0.5).to(torch.uint8)
        raw_pix_fmt = "rgb24"
    else:
        raise ValueError(f"Unsupported bit depth: {bit_depth}")
    if out.device.type != "cpu":
        out = out.cpu()
    out = out.contiguous()
    return out, raw_pix_fmt, time.perf_counter() - t0


def _bulk_u8_cpu(images: torch.Tensor):
    """Legacy helper retained for v3.1.x preset workflows."""
    out, _fmt, elapsed = _bulk_rgb_cpu(images, "8bit")
    return out, elapsed


ENCODER_PRESETS = ["lossless_x264_qp0", "av1_vaapi_150m_10bit"]
ENCODER_CODECS = ["x264", "x265", "av1_vaapi"]
ENCODER_BIT_DEPTHS = ["10bit", "8bit"]
ENCODER_RATE_CONTROLS = ["lossless", "constant_quality", "cqp", "vbr"]
SOFTWARE_PRESETS = ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"]
_ENCODER_HELP_CACHE = {}


def _build_video_ffmpeg_args(ffmpeg, frame_rate, width, height, video_path, encoder_preset, pixel_format):
    """Build the persistent video-only FFmpeg command for the selected production preset."""
    preset = str(encoder_preset)
    args = [ffmpeg, "-v", "error", "-y"]
    if preset == "av1_vaapi_150m_10bit":
        args += [
            "-init_hw_device", "vaapi=foo:/dev/dri/renderD128",
            "-filter_hw_device", "foo",
        ]
    args += [
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-color_range", "pc", "-colorspace", "rgb",
        "-color_primaries", "bt709", "-color_trc", "bt709",
        "-s", f"{int(width)}x{int(height)}", "-r", str(float(frame_rate)), "-i", "-",
    ]
    if preset == "lossless_x264_qp0":
        args += [
            "-vf", "scale=out_color_matrix=bt709:out_range=tv",
            "-c:v", "libx264", "-preset", "ultrafast", "-qp", "0",
            "-pix_fmt", str(pixel_format),
        ]
    elif preset == "av1_vaapi_150m_10bit":
        args += [
            "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=p010le,hwupload",
            "-c:v", "av1_vaapi",
            "-b:v", "150M", "-maxrate", "150M", "-bufsize", "300M",
            "-rc_mode", "VBR",
        ]
    else:
        raise ValueError(f"Unknown LongVideo encoder preset: {preset}")
    args += [
        "-color_range", "tv", "-colorspace", "bt709",
        "-color_primaries", "bt709", "-color_trc", "bt709",
        "-movflags", "+faststart",
        str(video_path),
    ]
    return args


class _AsyncEncoderState:
    def __init__(self, ffmpeg, frame_rate, width, height, pixel_format, queue_batches,
                 video_path, final_path, audio, save_output, subfolder, filename,
                 encoder_preset="lossless_x264_qp0", source_format_label=None):
        self.ffmpeg = ffmpeg
        self.frame_rate = float(frame_rate)
        self.width = int(width)
        self.height = int(height)
        self.pixel_format = pixel_format
        self.encoder_preset = str(encoder_preset)
        self.video_path = video_path
        self.final_path = final_path
        self.audio = audio
        self.save_output = save_output
        self.subfolder = subfolder
        self.filename = filename
        if source_format_label is None:
            source_format_label = "video/av1-mp4" if self.encoder_preset == "av1_vaapi_150m_10bit" else "video/h264-mp4"
        self.source_format_label = source_format_label
        self.q = queue.Queue(maxsize=max(1, int(queue_batches)))
        self.error = None
        self.closed = False
        self.frames_written = 0
        self.bytes_written = 0
        self.write_seconds = 0.0
        self.started = time.perf_counter()
        self._proc = self._start_ffmpeg()
        self._thread = threading.Thread(target=self._worker, name="LongVideoAsyncEncoder", daemon=True)
        self._thread.start()

    def _start_ffmpeg(self):
        args = _build_video_ffmpeg_args(
            self.ffmpeg, self.frame_rate, self.width, self.height, self.video_path,
            self.encoder_preset, self.pixel_format,
        )
        return subprocess.Popen(args, stdin=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)

    def _write_tensor(self, batch_u8: torch.Tensor):
        arr = batch_u8.numpy()
        view = memoryview(arr).cast("B")
        # Large writes reduce Python overhead while avoiding a second 1+ GiB bytes copy.
        step = 64 * 1024 * 1024
        t0 = time.perf_counter()
        for off in range(0, len(view), step):
            self._proc.stdin.write(view[off:off + step])
        self.write_seconds += time.perf_counter() - t0
        self.frames_written += int(batch_u8.shape[0])
        self.bytes_written += int(batch_u8.numel() * batch_u8.element_size())

    def _worker(self):
        try:
            while True:
                item = self.q.get()
                try:
                    if item is None:
                        break
                    self._write_tensor(item)
                finally:
                    self.q.task_done()
        except Exception as exc:
            self.error = exc
            # Unblock any final queue.join if FFmpeg fails.
            while True:
                try:
                    self.q.get_nowait()
                    self.q.task_done()
                except queue.Empty:
                    break
        finally:
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
            except Exception:
                pass
            try:
                stderr = self._proc.stderr.read() if self._proc.stderr else b""
                rc = self._proc.wait()
                if rc != 0 and self.error is None:
                    self.error = RuntimeError(stderr.decode("utf-8", "replace") or f"FFmpeg exited {rc}")
            except Exception as exc:
                if self.error is None:
                    self.error = exc
            self.closed = True

    def check(self):
        if self.error is not None:
            raise RuntimeError(f"LongVideo async encoder failed: {self.error}") from self.error

    def enqueue(self, batch_u8: torch.Tensor):
        self.check()
        while True:
            try:
                self.q.put(batch_u8, timeout=0.25)
                return
            except queue.Full:
                self.check()

    def finish(self):
        self.check()
        while True:
            try:
                self.q.put(None, timeout=0.25)
                break
            except queue.Full:
                self.check()
        self.q.join()
        self._thread.join()
        self.check()
        return self._mux_audio()

    def abort(self, remove_partial=True):
        """Best-effort cancellation that also wakes and retires the writer thread.

        DEV2.3.1 needs this to be safe after a Comfy prompt interruption: killing
        FFmpeg alone leaves the daemon worker blocked on q.get() and leaves the
        partial output/state available to the next prompt.
        """
        try:
            self._proc.kill()
        except Exception:
            pass

        # Drop queued frame batches and balance Queue.unfinished_tasks.
        while True:
            try:
                self.q.get_nowait()
                self.q.task_done()
            except queue.Empty:
                break
            except Exception:
                break

        # Wake a worker that is blocked waiting for its next batch.
        try:
            self.q.put_nowait(None)
        except Exception:
            pass

        try:
            if self._proc.stdin:
                self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._thread.join(timeout=2.0)
        except Exception:
            pass
        try:
            self._proc.wait(timeout=1.0)
        except Exception:
            pass
        self.closed = True

        if remove_partial:
            for path in {self.video_path, self.final_path}:
                try:
                    if path and os.path.isfile(path):
                        os.remove(path)
                except OSError:
                    pass

    def _mux_audio(self):
        if self.audio is None:
            return self.video_path

        audio_file = getattr(self.audio, "file", None)
        if audio_file and os.path.isfile(audio_file):
            # First try bitstream-copying the original audio. Fall back to AAC for MP4-incompatible codecs.
            copy_cmd = [
                self.ffmpeg, "-v", "error", "-y", "-i", self.video_path, "-i", audio_file,
                "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", "-c:a", "copy",
                "-shortest", "-movflags", "+faststart", self.final_path,
            ]
            res = subprocess.run(copy_cmd, capture_output=True)
            if res.returncode != 0:
                aac_cmd = [
                    self.ffmpeg, "-v", "error", "-y", "-i", self.video_path, "-i", audio_file,
                    "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "320k", "-shortest", "-movflags", "+faststart",
                    self.final_path,
                ]
                subprocess.run(aac_cmd, capture_output=True, check=True)
            try:
                os.remove(self.video_path)
            except OSError:
                pass
            return self.final_path

        # Fallback for ordinary ComfyUI AUDIO dictionaries.
        try:
            waveform = self.audio["waveform"]
            sample_rate = int(self.audio["sample_rate"])
            channels = int(waveform.size(1))
            audio_data = waveform.squeeze(0).transpose(0, 1).contiguous().cpu().numpy().tobytes()
            cmd = [
                self.ffmpeg, "-v", "error", "-y", "-i", self.video_path,
                "-ar", str(sample_rate), "-ac", str(channels), "-f", "f32le", "-i", "-",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "320k", "-shortest",
                "-movflags", "+faststart", self.final_path,
            ]
            subprocess.run(cmd, input=audio_data, capture_output=True, check=True)
            try:
                os.remove(self.video_path)
            except OSError:
                pass
            return self.final_path
        except Exception:
            # Do not lose the video just because audio extraction/muxing failed.
            return self.video_path


class LVAsyncLosslessVideoEncoder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_rate": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 240.0, "step": 0.001}),
                "filename_prefix": ("STRING", {"default": "halfsbs/vr"}),
                "pixel_format": (["yuv420p10le", "yuv420p"], {"default": "yuv420p10le"}),
                # Use dropdown widgets here rather than raw INT/BOOLEAN widgets.
                # Recent ComfyUI frontend validation can deserialize custom output-node
                # INT/BOOLEAN widgets as unconnected sockets in imported workflows.
                "queue_batches": (["1", "2", "3", "4"], {"default": "1"}),
                "save_output": (["output", "temp"], {"default": "output"}),
                # Added last to preserve positional compatibility with v3.0.x workflows.
                "encoder_preset": (ENCODER_PRESETS, {"default": "lossless_x264_qp0"}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "meta_batch": ("VHS_BatchManager",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("VHS_FILENAMES",)
    RETURN_NAMES = ("Filenames",)
    OUTPUT_NODE = True
    FUNCTION = "encode"
    CATEGORY = CATEGORY

    def encode(self, images, frame_rate, filename_prefix, pixel_format, queue_batches, save_output, encoder_preset="lossless_x264_qp0",
               audio=None, meta_batch=None, unique_id=None):
        queue_batches = int(queue_batches)
        save_output = (save_output == "output") if isinstance(save_output, str) else bool(save_output)
        if images is None or (isinstance(images, torch.Tensor) and images.shape[0] == 0):
            return ((save_output, []),)
        if meta_batch is None:
            raise ValueError("LongVideo Async Lossless Encoder requires VHS_BatchManager/meta_batch")

        ffmpeg, requeue_unchecked = _find_vhs_utils()
        if not ffmpeg:
            raise RuntimeError("ffmpeg was not found")
        if requeue_unchecked is None:
            raise RuntimeError("Could not import VHS requeue_workflow_unchecked; is ComfyUI-VideoHelperSuite installed?")

        key = str(unique_id)
        with _ENCODERS_LOCK:
            state = _ENCODERS.get(key)

        if state is None:
            h, w = int(images.shape[1]), int(images.shape[2])
            folder, filename, subfolder, counter = _next_output_path(filename_prefix, save_output)
            stem = f"{filename}_{counter:05}"
            # Encode to a temporary video-only file if audio exists, then mux to final.
            if audio is not None:
                video_path = os.path.join(folder, stem + "-video.mp4")
                final_path = os.path.join(folder, stem + ".mp4")
            else:
                video_path = os.path.join(folder, stem + ".mp4")
                final_path = video_path
            state = _AsyncEncoderState(
                ffmpeg, frame_rate, w, h, pixel_format, queue_batches,
                video_path, final_path, audio, save_output, subfolder, stem,
                encoder_preset=encoder_preset,
            )
            with _ENCODERS_LOCK:
                _ENCODERS[key] = state
            print(
                f"[LongVideo Encoder] started preset={state.encoder_preset} "
                f"{w}x{h}@{float(frame_rate):.6g} -> {video_path}"
            )

        if int(images.shape[1]) != state.height or int(images.shape[2]) != state.width:
            raise ValueError("Video dimensions changed during meta-batch encoding")
        if abs(float(frame_rate) - state.frame_rate) > 1e-6:
            raise ValueError("Frame rate changed during meta-batch encoding")
        if str(encoder_preset) != state.encoder_preset:
            raise ValueError("Encoder preset changed during meta-batch encoding")

        batch_u8, convert_sec = _bulk_u8_cpu(images)
        t0 = time.perf_counter()
        state.enqueue(batch_u8)
        enqueue_sec = time.perf_counter() - t0
        print(
            f"[LongVideo Encoder] queued {int(batch_u8.shape[0])} frames | "
            f"convert={convert_sec:.3f}s queue_wait={enqueue_sec:.3f}s "
            f"queue={state.q.qsize()}/{state.q.maxsize}"
        )

        if not meta_batch.has_closed_inputs:
            requeue_unchecked()
            return {"ui": {"unfinished_batch": [True]}, "result": ((save_output, []),)}

        # Final meta-batch: drain the bounded queue, close FFmpeg, then mux audio.
        final_path = state.finish()
        elapsed = time.perf_counter() - state.started
        print(
            f"[LongVideo Encoder] finished frames={state.frames_written} "
            f"encoder_write={state.write_seconds:.2f}s wall={elapsed:.2f}s -> {final_path}"
        )
        with _ENCODERS_LOCK:
            _ENCODERS.pop(key, None)

        preview_file = os.path.basename(final_path)
        preview = {
            "filename": preview_file,
            "subfolder": state.subfolder,
            "type": "output" if save_output else "temp",
            "format": state.source_format_label,
            "frame_rate": frame_rate,
            "fullpath": final_path,
        }
        return {"ui": {"gifs": [preview]}, "result": ((save_output, [final_path]),)}


def _fmt_mbps(value):
    value = float(value)
    if value <= 0:
        raise ValueError("Bitrate values must be greater than 0 Mbps")
    return f"{value:g}M"


def _ffmpeg_encoder_help(ffmpeg, encoder):
    key = (str(ffmpeg), str(encoder))
    cached = _ENCODER_HELP_CACHE.get(key)
    if cached is not None:
        return cached
    res = subprocess.run(
        [ffmpeg, "-hide_banner", "-h", f"encoder={encoder}"],
        capture_output=True, text=True,
    )
    text = (res.stdout or "") + (res.stderr or "")
    if res.returncode != 0 or f"Encoder {encoder}" not in text:
        raise RuntimeError(
            f"FFmpeg encoder '{encoder}' is unavailable in this FFmpeg build. "
            f"Run: {ffmpeg} -hide_banner -h encoder={encoder}"
        )
    _ENCODER_HELP_CACHE[key] = text
    return text


def _validate_advanced_encoder_settings(ffmpeg, codec, bit_depth, rate_control, crf,
                                        av1_qp, bitrate_mbps, maxrate_mbps,
                                        bufsize_mbps, software_preset, vaapi_device):
    codec = str(codec)
    bit_depth = str(bit_depth)
    rate_control = str(rate_control)
    software_preset = str(software_preset)
    vaapi_device = str(vaapi_device)

    if codec not in ENCODER_CODECS:
        raise ValueError(f"Unsupported codec: {codec}")
    if bit_depth not in ENCODER_BIT_DEPTHS:
        raise ValueError(f"Unsupported bit depth: {bit_depth}")
    if rate_control not in ENCODER_RATE_CONTROLS:
        raise ValueError(f"Unsupported rate control: {rate_control}")
    if software_preset not in SOFTWARE_PRESETS:
        raise ValueError(f"Unsupported software preset: {software_preset}")

    encoder = {"x264": "libx264", "x265": "libx265", "av1_vaapi": "av1_vaapi"}[codec]
    help_text = _ffmpeg_encoder_help(ffmpeg, encoder)

    if codec in ("x264", "x265"):
        if bit_depth == "10bit" and "yuv420p10le" not in help_text:
            raise RuntimeError(
                f"Selected {codec} 10-bit, but this FFmpeg/{encoder} build does not advertise yuv420p10le."
            )
        if rate_control == "cqp":
            raise ValueError(f"{codec} does not use the AV1 VAAPI CQP mode; use constant_quality (CRF), lossless, or vbr")
        if rate_control == "constant_quality" and not (0.0 <= float(crf) <= 51.0):
            raise ValueError(f"{codec} CRF must be in the range 0..51")
    else:
        if rate_control == "lossless":
            raise ValueError(
                "av1_vaapi does not expose a guaranteed lossless mode here; "
                "use cqp (or legacy constant_quality) or vbr."
            )
        if rate_control in ("cqp", "constant_quality") and not (1 <= int(av1_qp) <= 255):
            raise ValueError("av1_vaapi QP/global_quality must be in the range 1..255")
        if not vaapi_device:
            raise ValueError("VAAPI device path is empty")
        if not os.path.exists(vaapi_device):
            raise RuntimeError(f"VAAPI device does not exist: {vaapi_device}")

    if rate_control == "vbr":
        b = float(bitrate_mbps)
        m = float(maxrate_mbps)
        buf = float(bufsize_mbps)
        if min(b, m, buf) <= 0:
            raise ValueError("VBR bitrate, maxrate, and bufsize must all be greater than 0 Mbps")
        if m < b:
            raise ValueError("VBR maxrate_mbps must be >= bitrate_mbps")

    return encoder


def _advanced_encoder_signature(codec, bit_depth, rate_control, crf, av1_qp,
                                bitrate_mbps, maxrate_mbps, bufsize_mbps,
                                software_preset, vaapi_device):
    return (
        str(codec), str(bit_depth), str(rate_control), round(float(crf), 6), int(av1_qp),
        round(float(bitrate_mbps), 6), round(float(maxrate_mbps), 6),
        round(float(bufsize_mbps), 6), str(software_preset), str(vaapi_device),
    )


def _build_advanced_video_ffmpeg_args(ffmpeg, frame_rate, width, height, video_path,
                                      codec, bit_depth, rate_control, crf, av1_qp,
                                      bitrate_mbps, maxrate_mbps, bufsize_mbps,
                                      software_preset, vaapi_device):
    """Build the persistent FFmpeg command for the configurable production encoder."""
    codec = str(codec)
    bit_depth = str(bit_depth)
    rate_control = str(rate_control)
    raw_pix_fmt = "rgb48le" if bit_depth == "10bit" else "rgb24"

    args = [ffmpeg, "-v", "error", "-y"]
    if codec == "av1_vaapi":
        args += [
            "-init_hw_device", f"vaapi=foo:{vaapi_device}",
            "-filter_hw_device", "foo",
        ]

    args += [
        "-f", "rawvideo", "-pix_fmt", raw_pix_fmt,
        "-color_range", "pc", "-colorspace", "rgb",
        "-color_primaries", "bt709", "-color_trc", "bt709",
        "-s", f"{int(width)}x{int(height)}", "-r", str(float(frame_rate)), "-i", "-",
    ]

    if codec == "x264":
        target_fmt = "yuv420p10le" if bit_depth == "10bit" else "yuv420p"
        args += [
            "-vf", f"scale=out_color_matrix=bt709:out_range=tv,format={target_fmt}",
            "-c:v", "libx264", "-preset", str(software_preset),
        ]
        if rate_control == "lossless":
            args += ["-qp", "0"]
        elif rate_control == "constant_quality":
            args += ["-crf", f"{float(crf):g}"]
        elif rate_control == "vbr":
            args += [
                "-b:v", _fmt_mbps(bitrate_mbps),
                "-maxrate", _fmt_mbps(maxrate_mbps),
                "-bufsize", _fmt_mbps(bufsize_mbps),
            ]
        args += ["-pix_fmt", target_fmt]

    elif codec == "x265":
        target_fmt = "yuv420p10le" if bit_depth == "10bit" else "yuv420p"
        args += [
            "-vf", f"scale=out_color_matrix=bt709:out_range=tv,format={target_fmt}",
            "-c:v", "libx265", "-preset", str(software_preset),
        ]
        if rate_control == "lossless":
            args += ["-x265-params", "lossless=1"]
        elif rate_control == "constant_quality":
            args += ["-crf", f"{float(crf):g}"]
        elif rate_control == "vbr":
            args += [
                "-b:v", _fmt_mbps(bitrate_mbps),
                "-maxrate", _fmt_mbps(maxrate_mbps),
                "-bufsize", _fmt_mbps(bufsize_mbps),
            ]
        args += ["-pix_fmt", target_fmt]

    elif codec == "av1_vaapi":
        target_fmt = "p010le" if bit_depth == "10bit" else "nv12"
        args += [
            "-vf", f"scale=out_color_matrix=bt709:out_range=tv,format={target_fmt},hwupload",
            "-c:v", "av1_vaapi",
        ]
        if rate_control in ("cqp", "constant_quality"):
            args += ["-rc_mode", "CQP", "-global_quality", str(int(av1_qp))]
        elif rate_control == "vbr":
            args += [
                "-rc_mode", "VBR",
                "-b:v", _fmt_mbps(bitrate_mbps),
                "-maxrate", _fmt_mbps(maxrate_mbps),
                "-bufsize", _fmt_mbps(bufsize_mbps),
            ]
        elif rate_control == "lossless":
            raise ValueError("av1_vaapi lossless mode is not supported by the production encoder UI")
    else:
        raise ValueError(f"Unsupported codec: {codec}")

    args += [
        "-color_range", "tv", "-colorspace", "bt709",
        "-color_primaries", "bt709", "-color_trc", "bt709",
        "-movflags", "+faststart",
        str(video_path),
    ]
    return args


class _AsyncAdvancedEncoderState(_AsyncEncoderState):
    def __init__(self, ffmpeg, frame_rate, width, height, queue_batches,
                 video_path, final_path, audio, save_output, subfolder, filename,
                 settings_signature, codec, bit_depth, rate_control, crf, av1_qp,
                 bitrate_mbps, maxrate_mbps, bufsize_mbps, software_preset,
                 vaapi_device):
        self.settings_signature = settings_signature
        self.codec = str(codec)
        self.bit_depth = str(bit_depth)
        self.rate_control = str(rate_control)
        self.crf = float(crf)
        self.av1_qp = int(av1_qp)
        self.bitrate_mbps = float(bitrate_mbps)
        self.maxrate_mbps = float(maxrate_mbps)
        self.bufsize_mbps = float(bufsize_mbps)
        self.software_preset = str(software_preset)
        self.vaapi_device = str(vaapi_device)
        source_label = {
            "x264": "video/h264-mp4",
            "x265": "video/h265-mp4",
            "av1_vaapi": "video/av1-mp4",
        }[self.codec]
        super().__init__(
            ffmpeg, frame_rate, width, height,
            "yuv420p10le" if self.bit_depth == "10bit" else "yuv420p",
            queue_batches, video_path, final_path, audio, save_output,
            subfolder, filename, encoder_preset="advanced", source_format_label=source_label,
        )

    def _start_ffmpeg(self):
        args = _build_advanced_video_ffmpeg_args(
            self.ffmpeg, self.frame_rate, self.width, self.height, self.video_path,
            self.codec, self.bit_depth, self.rate_control, self.crf, self.av1_qp,
            self.bitrate_mbps, self.maxrate_mbps, self.bufsize_mbps,
            self.software_preset, self.vaapi_device,
        )
        return subprocess.Popen(args, stdin=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)


class LVAsyncVideoEncoder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_rate": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 240.0, "step": 0.001}),
                "filename_prefix": ("STRING", {"default": "halfsbs/vr"}),
                "codec": (ENCODER_CODECS, {"default": "x264"}),
                "bit_depth": (ENCODER_BIT_DEPTHS, {"default": "10bit"}),
                "rate_control": (ENCODER_RATE_CONTROLS, {"default": "lossless"}),
                "crf": ("FLOAT", {"default": 18.0, "min": 0.0, "max": 51.0, "step": 0.5}),
                "av1_qp": ("INT", {"default": 24, "min": 1, "max": 255, "step": 1, "tooltip": "AV1 VAAPI q_idx/global_quality for CQP. Lower = higher quality/larger files; higher = lower quality/smaller files."}),
                "bitrate_mbps": ("FLOAT", {"default": 150.0, "min": 1.0, "max": 1000.0, "step": 1.0}),
                "maxrate_mbps": ("FLOAT", {"default": 150.0, "min": 1.0, "max": 1000.0, "step": 1.0}),
                "bufsize_mbps": ("FLOAT", {"default": 300.0, "min": 1.0, "max": 4000.0, "step": 1.0}),
                "software_preset": (SOFTWARE_PRESETS, {"default": "ultrafast"}),
                "vaapi_device": ("STRING", {"default": "/dev/dri/renderD128"}),
                "queue_batches": (["1", "2", "3", "4"], {"default": "1"}),
                "save_output": (["output", "temp"], {"default": "output"}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "meta_batch": ("VHS_BatchManager",),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("VHS_FILENAMES",)
    RETURN_NAMES = ("Filenames",)
    OUTPUT_NODE = True
    FUNCTION = "encode"
    CATEGORY = CATEGORY

    def encode(self, images, frame_rate, filename_prefix, codec, bit_depth, rate_control,
               crf, av1_qp, bitrate_mbps, maxrate_mbps, bufsize_mbps, software_preset,
               vaapi_device, queue_batches, save_output, audio=None, meta_batch=None,
               unique_id=None):
        queue_batches = int(queue_batches)
        save_output = (save_output == "output") if isinstance(save_output, str) else bool(save_output)
        if images is None or (isinstance(images, torch.Tensor) and images.shape[0] == 0):
            return ((save_output, []),)
        if meta_batch is None:
            raise ValueError("LongVideo Async Video Encoder requires VHS_BatchManager/meta_batch")

        ffmpeg, requeue_unchecked = _find_vhs_utils()
        if not ffmpeg:
            raise RuntimeError("ffmpeg was not found")
        if requeue_unchecked is None:
            raise RuntimeError("Could not import VHS requeue_workflow_unchecked; is ComfyUI-VideoHelperSuite installed?")

        signature = _advanced_encoder_signature(
            codec, bit_depth, rate_control, crf, av1_qp, bitrate_mbps,
            maxrate_mbps, bufsize_mbps, software_preset, vaapi_device,
        )
        key = str(unique_id)
        with _ENCODERS_LOCK:
            state = _ENCODERS.get(key)

        if state is None:
            _validate_advanced_encoder_settings(
                ffmpeg, codec, bit_depth, rate_control, crf, av1_qp,
                bitrate_mbps, maxrate_mbps, bufsize_mbps, software_preset,
                vaapi_device,
            )
            h, w = int(images.shape[1]), int(images.shape[2])
            folder, filename, subfolder, counter = _next_output_path(filename_prefix, save_output)
            stem = f"{filename}_{counter:05}"
            if audio is not None:
                video_path = os.path.join(folder, stem + "-video.mp4")
                final_path = os.path.join(folder, stem + ".mp4")
            else:
                video_path = os.path.join(folder, stem + ".mp4")
                final_path = video_path
            state = _AsyncAdvancedEncoderState(
                ffmpeg, frame_rate, w, h, queue_batches, video_path, final_path,
                audio, save_output, subfolder, stem, signature, codec, bit_depth,
                rate_control, crf, av1_qp, bitrate_mbps, maxrate_mbps,
                bufsize_mbps, software_preset, vaapi_device,
            )
            with _ENCODERS_LOCK:
                _ENCODERS[key] = state
            quality_desc = (
                "lossless" if rate_control == "lossless" else
                (f"CRF={float(crf):g}" if rate_control == "constant_quality" and codec in ("x264", "x265")
                 else f"CQP q_idx={int(av1_qp)}" if codec == "av1_vaapi" and rate_control in ("cqp", "constant_quality")
                 else f"VBR={float(bitrate_mbps):g}/{float(maxrate_mbps):g}/{float(bufsize_mbps):g}M")
            )
            print(
                f"[LongVideo Encoder] started codec={codec} bit_depth={bit_depth} "
                f"rc={quality_desc} {w}x{h}@{float(frame_rate):.6g} -> {video_path}"
            )

        if not isinstance(state, _AsyncAdvancedEncoderState):
            raise ValueError("Encoder node identity is already occupied by a legacy encoder state")
        if state.settings_signature != signature:
            raise ValueError("Encoder settings changed during meta-batch encoding")
        if int(images.shape[1]) != state.height or int(images.shape[2]) != state.width:
            raise ValueError("Video dimensions changed during meta-batch encoding")
        if abs(float(frame_rate) - state.frame_rate) > 1e-6:
            raise ValueError("Frame rate changed during meta-batch encoding")

        batch_rgb, raw_fmt, convert_sec = _bulk_rgb_cpu(images, bit_depth)
        expected_fmt = "rgb48le" if state.bit_depth == "10bit" else "rgb24"
        if raw_fmt != expected_fmt:
            raise RuntimeError(f"Internal raw pixel-format mismatch: {raw_fmt} != {expected_fmt}")
        t0 = time.perf_counter()
        state.enqueue(batch_rgb)
        enqueue_sec = time.perf_counter() - t0
        print(
            f"[LongVideo Encoder] queued {int(batch_rgb.shape[0])} frames | "
            f"raw={raw_fmt} convert={convert_sec:.3f}s queue_wait={enqueue_sec:.3f}s "
            f"queue={state.q.qsize()}/{state.q.maxsize}"
        )

        if not meta_batch.has_closed_inputs:
            requeue_unchecked()
            return {"ui": {"unfinished_batch": [True]}, "result": ((save_output, []),)}

        final_path = state.finish()
        elapsed = time.perf_counter() - state.started
        print(
            f"[LongVideo Encoder] finished frames={state.frames_written} "
            f"raw_gib={state.bytes_written / (1024**3):.2f} "
            f"encoder_write={state.write_seconds:.2f}s wall={elapsed:.2f}s -> {final_path}"
        )
        with _ENCODERS_LOCK:
            _ENCODERS.pop(key, None)

        preview = {
            "filename": os.path.basename(final_path),
            "subfolder": state.subfolder,
            "type": "output" if save_output else "temp",
            "format": state.source_format_label,
            "frame_rate": frame_rate,
            "fullpath": final_path,
        }
        return {"ui": {"gifs": [preview]}, "result": ((save_output, [final_path]),)}


class LVStreamingVideoEncoder:
    """DEV2 sink: consume LV_INTERP_STREAM in bounded chunks and encode immediately."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stream": ("LV_INTERP_STREAM",),
                "filename_prefix": ("STRING", {"default": "halfsbs/", "tooltip": "Output subfolder in source_name mode; no filename prefix is added. Example: halfsbs/"}),
                "filename_mode": (["custom", "source_name"], {"default": "source_name"}),
                "filename_suffix": ("STRING", {"default": " Half-SBS"}),
                "codec": (ENCODER_CODECS, {"default": "av1_vaapi"}),
                "bit_depth": (ENCODER_BIT_DEPTHS, {"default": "10bit"}),
                "rate_control": (ENCODER_RATE_CONTROLS, {"default": "vbr"}),
                "crf": ("FLOAT", {"default": 18.0, "min": 0.0, "max": 51.0, "step": 0.5}),
                "av1_qp": ("INT", {"default": 24, "min": 1, "max": 255, "step": 1, "tooltip": "AV1 VAAPI q_idx/global_quality for CQP. Lower = higher quality/larger files; higher = lower quality/smaller files."}),
                "bitrate_mbps": ("FLOAT", {"default": 100.0, "min": 1.0, "max": 1000.0, "step": 1.0}),
                "maxrate_mbps": ("FLOAT", {"default": 135.0, "min": 1.0, "max": 1000.0, "step": 1.0}),
                "bufsize_mbps": ("FLOAT", {"default": 280.0, "min": 1.0, "max": 4000.0, "step": 1.0}),
                "software_preset": (SOFTWARE_PRESETS, {"default": "ultrafast"}),
                "vaapi_device": ("STRING", {"default": "/dev/dri/renderD128"}),
                "queue_batches": (["1", "2", "3", "4"], {"default": "1"}),
                "save_output": (["output", "temp"], {"default": "output"}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "meta_batch": ("VHS_BatchManager",),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("VHS_FILENAMES",)
    RETURN_NAMES = ("Filenames",)
    OUTPUT_NODE = True
    FUNCTION = "encode"
    CATEGORY = CATEGORY

    def encode(self, stream, filename_prefix, filename_mode, filename_suffix, codec, bit_depth, rate_control,
               crf, av1_qp, bitrate_mbps, maxrate_mbps, bufsize_mbps, software_preset,
               vaapi_device, queue_batches, save_output, audio=None, meta_batch=None,
               unique_id=None):
        queue_batches = int(queue_batches)
        save_output = (save_output == "output") if isinstance(save_output, str) else bool(save_output)
        if meta_batch is None:
            raise ValueError("LongVideo Streaming Video Encoder requires VHS_BatchManager/meta_batch")
        if stream is None or not hasattr(stream, "iter_chunks"):
            raise ValueError("Expected LV_INTERP_STREAM from LongVideo Frame Interpolation Stream")

        ffmpeg, requeue_unchecked = _find_vhs_utils()
        if not ffmpeg:
            raise RuntimeError("ffmpeg was not found")
        if requeue_unchecked is None:
            raise RuntimeError("Could not import VHS requeue_workflow_unchecked; is ComfyUI-VideoHelperSuite installed?")

        frame_rate = float(stream.output_fps)
        source_name = str(getattr(stream, "source_name", "") or "")
        effective_prefix = _resolve_stream_filename_prefix(
            filename_prefix, filename_mode, filename_suffix, source_name
        )
        signature = _advanced_encoder_signature(
            codec, bit_depth, rate_control, crf, av1_qp, bitrate_mbps,
            maxrate_mbps, bufsize_mbps, software_preset, vaapi_device,
        )
        key = str(unique_id)
        run_token = _stream_run_token(stream)
        state = _retire_stale_encoder_for_run(key, run_token)

        if state is None:
            _validate_advanced_encoder_settings(
                ffmpeg, codec, bit_depth, rate_control, crf, av1_qp,
                bitrate_mbps, maxrate_mbps, bufsize_mbps, software_preset,
                vaapi_device,
            )
            h, w = int(stream.height), int(stream.width)
            if str(filename_mode) == "source_name":
                folder, stem, subfolder = _next_source_named_output_path(effective_prefix, save_output)
            else:
                folder, filename, subfolder, counter = _next_output_path(effective_prefix, save_output)
                stem = f"{filename}_{counter:05}"
            if audio is not None:
                video_path = os.path.join(folder, stem + "-video.mp4")
                final_path = os.path.join(folder, stem + ".mp4")
            else:
                video_path = os.path.join(folder, stem + ".mp4")
                final_path = video_path
            state = _AsyncAdvancedEncoderState(
                ffmpeg, frame_rate, w, h, queue_batches, video_path, final_path,
                audio, save_output, subfolder, stem, signature, codec, bit_depth,
                rate_control, crf, av1_qp, bitrate_mbps, maxrate_mbps,
                bufsize_mbps, software_preset, vaapi_device,
            )
            state._longvideo_run_token = run_token
            with _ENCODERS_LOCK:
                _ENCODERS[key] = state
            state._longvideo_profile = {
                "batches": 0, "rife": 0.0, "ifnet": 0.0, "rife_post": 0.0, "rife_pack": 0.0,
                "rife_calls": 0, "rife_pairs": 0, "pair_batch": 1,
                "feature_cache": "off", "feature_cache_extract_calls": 0,
                "feature_cache_reuses": 0, "feature_cache_pairs": 0, "feature_cache_resets": 0,
                "ifnet_stage_seconds": {}, "ifnet_stage_calls": {},
                "ifnet_stage_unattributed": 0.0, "ifnet_stage_profiled_forwards": 0,
                "d2h": 0.0, "d2h_wait": 0.0, "scene": 0.0,
                "prepare": 0.0, "layout": 0.0, "h2d": 0.0, "h2d_wait": 0.0, "gpu_prepare": 0.0,
                "assemble": 0.0, "convert": 0.0, "queue_wait": 0.0,
                "pin_alloc": 0.0, "pin_new": 0, "pin_reuse": 0, "pin_peak": 0,
                "h2d_pin_alloc": 0.0, "h2d_pin_new": 0, "h2d_pin_reuse": 0, "h2d_pin_peak": 0,
                "stream_wall": 0.0, "finish": 0.0,
            }
            print(
                f"[LongVideo Stream Encoder] started {w}x{h}@{frame_rate:.9f} "
                f"codec={codec} bit_depth={bit_depth} naming={filename_mode} "
                f"source={source_name or '-'} -> {video_path}"
            )

        if not isinstance(state, _AsyncAdvancedEncoderState):
            raise ValueError("Encoder node identity is already occupied by a legacy encoder state")
        if getattr(state, "_longvideo_run_token", None) != run_token:
            raise RuntimeError("Internal LongVideo encoder run-token mismatch")
        if state.settings_signature != signature:
            raise ValueError("Encoder settings changed during meta-batch encoding")
        if int(stream.height) != state.height or int(stream.width) != state.width:
            raise ValueError("Video dimensions changed during streaming interpolation")
        if abs(frame_rate - state.frame_rate) > 1e-6:
            raise ValueError("Frame rate changed during streaming interpolation")

        chunks = 0
        frames = 0
        convert_total = 0.0
        queue_wait_total = 0.0
        batch_started = time.perf_counter()
        writer_seconds_before = float(state.write_seconds)
        writer_frames_before = int(state.frames_written)
        try:
            for chunk in stream.iter_chunks():
                batch_rgb, raw_fmt, convert_sec = _bulk_rgb_cpu(chunk, bit_depth)
                expected_fmt = "rgb48le" if state.bit_depth == "10bit" else "rgb24"
                if raw_fmt != expected_fmt:
                    raise RuntimeError(f"Internal raw pixel-format mismatch: {raw_fmt} != {expected_fmt}")
                t0 = time.perf_counter()
                state.enqueue(batch_rgb)
                queue_wait = time.perf_counter() - t0
                chunks += 1
                frames += int(batch_rgb.shape[0])
                convert_total += float(convert_sec)
                queue_wait_total += float(queue_wait)
                # Queue owns batch_rgb after enqueue; dropping the IMAGE chunk here
                # keeps the expanded 120-fps working set bounded.
                del chunk, batch_rgb
        except Exception:
            state.abort()
            with _ENCODERS_LOCK:
                _ENCODERS.pop(key, None)
            raise

        batch_wall = time.perf_counter() - batch_started
        rife_sec = float(stream.stats.get("rife_inference_seconds", stream.stats.get("rife_seconds", 0.0)))
        ifnet_sec = float(stream.stats.get("ifnet_seconds", rife_sec))
        rife_post_sec = float(stream.stats.get("rife_post_seconds", 0.0))
        rife_pack_sec = float(stream.stats.get("rife_pack_seconds", 0.0))
        rife_calls = int(stream.stats.get("rife_forward_calls", 0))
        rife_pairs = int(stream.stats.get("rife_pairs_submitted", 0))
        pair_batch_active = int(stream.stats.get("pair_batch_active", 1))
        pair_batch_requested = int(stream.stats.get("pair_batch_requested", pair_batch_active))
        ifnet_stage_seconds = dict(stream.stats.get("ifnet_stage_seconds", {}) or {})
        ifnet_stage_calls = dict(stream.stats.get("ifnet_stage_calls", {}) or {})
        ifnet_stage_unattributed = float(stream.stats.get("ifnet_stage_unattributed_seconds", 0.0))
        ifnet_stage_profiled_forwards = int(stream.stats.get("ifnet_stage_profiled_forwards", 0))
        ifnet_profile_active = str(stream.stats.get("ifnet_profile_active", "off"))
        feature_cache_active = str(stream.stats.get("feature_cache_active", "off"))
        feature_cache_requested = str(stream.stats.get("feature_cache_requested", feature_cache_active))
        feature_cache_extract_calls = int(stream.stats.get("feature_cache_extract_calls", 0))
        feature_cache_reuses = int(stream.stats.get("feature_cache_reuses", 0))
        feature_cache_pairs = int(stream.stats.get("feature_cache_pairs", 0))
        feature_cache_resets = int(stream.stats.get("feature_cache_resets", 0))
        ifnet_model_class = str(stream.stats.get("ifnet_model_class", "?"))
        ifnet_model_dtype = str(stream.stats.get("ifnet_model_dtype", stream.stats.get("ifnet_input_dtype", "?")))
        ifnet_input_contiguous = bool(stream.stats.get("ifnet_input_contiguous", False))
        ifnet_input_channels_last = bool(stream.stats.get("ifnet_input_channels_last", False))
        d2h_sec = float(stream.stats.get("d2h_seconds", 0.0))
        d2h_wait_sec = float(stream.stats.get("d2h_wait_seconds", 0.0))
        d2h_active = str(stream.stats.get("d2h_mode_active", "sync"))
        d2h_requested = str(stream.stats.get("d2h_mode_requested", d2h_active))
        scene_sec = float(stream.stats.get("scene_seconds", 0.0))
        prepare_sec = float(stream.stats.get("prepare_seconds", 0.0))
        layout_sec = float(stream.stats.get("layout_seconds", 0.0))
        h2d_sec = float(stream.stats.get("h2d_seconds", 0.0))
        h2d_wait_sec = float(stream.stats.get("h2d_wait_seconds", 0.0))
        gpu_prepare_sec = float(stream.stats.get("gpu_prepare_seconds", 0.0))
        h2d_active = str(stream.stats.get("h2d_mode_active", "sync"))
        h2d_requested = str(stream.stats.get("h2d_mode_requested", h2d_active))
        assemble_sec = float(stream.stats.get("assemble_seconds", 0.0))
        pin_alloc_sec = float(stream.stats.get("pinned_alloc_seconds", 0.0))
        pin_new = int(stream.stats.get("pinned_allocations", 0))
        pin_reuse = int(stream.stats.get("pinned_reuses", 0))
        pin_peak = int(stream.stats.get("pinned_peak_inflight", 0))
        h2d_pin_alloc_sec = float(stream.stats.get("h2d_pinned_alloc_seconds", 0.0))
        h2d_pin_new = int(stream.stats.get("h2d_pinned_allocations", 0))
        h2d_pin_reuse = int(stream.stats.get("h2d_pinned_reuses", 0))
        h2d_pin_peak = int(stream.stats.get("h2d_pinned_peak_inflight", 0))
        writer_observed = max(0.0, float(state.write_seconds) - writer_seconds_before)
        writer_frames_observed = max(0, int(state.frames_written) - writer_frames_before)
        # H2D/D2H engine times can overlap GPU compute. Count only the unhidden
        # dependency waits plus CPU layout/assembly/conversion in serial wall math.
        accounted_serial = (
            rife_sec + rife_pack_sec + d2h_wait_sec + scene_sec + layout_sec + h2d_wait_sec
            + assemble_sec + convert_total + queue_wait_total
        )
        other_wall = max(0.0, batch_wall - accounted_serial)

        profile = getattr(state, "_longvideo_profile", None)
        if profile is None:
            profile = state._longvideo_profile = {
                "batches": 0, "rife": 0.0, "ifnet": 0.0, "rife_post": 0.0, "rife_pack": 0.0,
                "rife_calls": 0, "rife_pairs": 0, "pair_batch": 1,
                "ifnet_stage_seconds": {}, "ifnet_stage_calls": {},
                "ifnet_stage_unattributed": 0.0, "ifnet_stage_profiled_forwards": 0,
                "d2h": 0.0, "d2h_wait": 0.0, "scene": 0.0,
                "prepare": 0.0, "layout": 0.0, "h2d": 0.0, "h2d_wait": 0.0, "gpu_prepare": 0.0,
                "assemble": 0.0, "convert": 0.0, "queue_wait": 0.0,
                "pin_alloc": 0.0, "pin_new": 0, "pin_reuse": 0, "pin_peak": 0,
                "h2d_pin_alloc": 0.0, "h2d_pin_new": 0, "h2d_pin_reuse": 0, "h2d_pin_peak": 0,
                "stream_wall": 0.0, "finish": 0.0,
            }
        profile["batches"] += 1
        profile["rife"] += rife_sec
        profile["ifnet"] += ifnet_sec
        profile["rife_post"] += rife_post_sec
        profile["rife_pack"] += rife_pack_sec
        profile["rife_calls"] += rife_calls
        profile["rife_pairs"] += rife_pairs
        profile["pair_batch"] = max(int(profile.get("pair_batch", 1)), pair_batch_active)
        if feature_cache_active != "off":
            profile["feature_cache"] = feature_cache_active
        profile["feature_cache_extract_calls"] = int(profile.get("feature_cache_extract_calls", 0)) + feature_cache_extract_calls
        profile["feature_cache_reuses"] = int(profile.get("feature_cache_reuses", 0)) + feature_cache_reuses
        profile["feature_cache_pairs"] = int(profile.get("feature_cache_pairs", 0)) + feature_cache_pairs
        profile["feature_cache_resets"] = int(profile.get("feature_cache_resets", 0)) + feature_cache_resets
        profile_stage_seconds = profile.setdefault("ifnet_stage_seconds", {})
        profile_stage_calls = profile.setdefault("ifnet_stage_calls", {})
        for _stage, _sec in ifnet_stage_seconds.items():
            profile_stage_seconds[_stage] = float(profile_stage_seconds.get(_stage, 0.0)) + float(_sec)
        for _stage, _calls in ifnet_stage_calls.items():
            profile_stage_calls[_stage] = int(profile_stage_calls.get(_stage, 0)) + int(_calls)
        profile["ifnet_stage_unattributed"] = float(profile.get("ifnet_stage_unattributed", 0.0)) + ifnet_stage_unattributed
        profile["ifnet_stage_profiled_forwards"] = int(profile.get("ifnet_stage_profiled_forwards", 0)) + ifnet_stage_profiled_forwards
        profile["d2h"] += d2h_sec
        profile["d2h_wait"] += d2h_wait_sec
        profile["scene"] += scene_sec
        profile["prepare"] += prepare_sec
        profile["layout"] += layout_sec
        profile["h2d"] += h2d_sec
        profile["h2d_wait"] += h2d_wait_sec
        profile["gpu_prepare"] += gpu_prepare_sec
        profile["assemble"] += assemble_sec
        profile["convert"] += convert_total
        profile["queue_wait"] += queue_wait_total
        profile["pin_alloc"] += pin_alloc_sec
        profile["pin_new"] += pin_new
        profile["pin_reuse"] += pin_reuse
        profile["pin_peak"] = max(int(profile.get("pin_peak", 0)), pin_peak)
        profile["h2d_pin_alloc"] += h2d_pin_alloc_sec
        profile["h2d_pin_new"] += h2d_pin_new
        profile["h2d_pin_reuse"] += h2d_pin_reuse
        profile["h2d_pin_peak"] = max(int(profile.get("h2d_pin_peak", 0)), h2d_pin_peak)
        profile["stream_wall"] += batch_wall

        print(
            f"[LongVideo RIFE Stream] {getattr(stream, 'summary', '')} | "
            f"cuts={int(stream.stats.get('scene_cuts', 0))}/"
            f"{int(stream.stats.get('scene_pairs', 0))}"
        )
        print(
            f"[LongVideo Stream Profile] scene={scene_sec:.3f}s "
            f"layout={layout_sec:.3f}s h2d={h2d_sec:.3f}s h2d_wait={h2d_wait_sec:.3f}s "
            f"gpu_prep={gpu_prepare_sec:.3f}s h2d_mode={h2d_active}"
            + (f"(requested={h2d_requested})" if h2d_active != h2d_requested else "") + " "
            f"rife={rife_sec:.3f}s(ifnet={ifnet_sec:.3f}s post={rife_post_sec:.3f}s pack={rife_pack_sec:.3f}s "
            f"calls={rife_calls} pairs={rife_pairs} pair_batch={pair_batch_active}"
            + (f" requested={pair_batch_requested}" if pair_batch_active != pair_batch_requested else "") + ") "
            f"feature_cache={feature_cache_active}"
            + (f"(requested={feature_cache_requested})" if feature_cache_active != feature_cache_requested else "")
            + f" feat_extract={feature_cache_extract_calls} feat_reuse={feature_cache_reuses} "
            f"d2h={d2h_sec:.3f}s d2h_wait={d2h_wait_sec:.3f}s "
            f"d2h_mode={d2h_active}" + (f"(requested={d2h_requested})" if d2h_active != d2h_requested else "") + " "
            f"h2d_pin_alloc={h2d_pin_alloc_sec:.3f}s h2d_pin_new={h2d_pin_new} "
            f"h2d_pin_reuse={h2d_pin_reuse} h2d_pin_peak={h2d_pin_peak} "
            f"d2h_pin_alloc={pin_alloc_sec:.3f}s d2h_pin_new={pin_new} "
            f"d2h_pin_reuse={pin_reuse} d2h_pin_peak={pin_peak} "
            f"assemble={assemble_sec:.3f}s rgb_convert={convert_total:.3f}s "
            f"queue_wait={queue_wait_total:.3f}s other~={other_wall:.3f}s wall={batch_wall:.3f}s | "
            f"writer_overlap={writer_observed:.3f}s/{writer_frames_observed}f"
        )
        if ifnet_stage_profiled_forwards:
            print(
                f"[LongVideo IFNet Stage Profile] mode={ifnet_profile_active} model={ifnet_model_class} "
                f"dtype={ifnet_model_dtype} contiguous={int(ifnet_input_contiguous)} "
                f"channels_last={int(ifnet_input_channels_last)} "
                + _format_ifnet_stage_profile(
                    ifnet_stage_seconds, ifnet_stage_calls, ifnet_stage_unattributed,
                    ifnet_stage_profiled_forwards,
                )
            )
        print(
            f"[LongVideo Stream Encoder] batch frames={frames} chunks={chunks} "
            f"queue={state.q.qsize()}/{state.q.maxsize}"
        )

        stream_final = bool(getattr(stream, "final", False))
        vhs_closed = bool(getattr(meta_batch, "has_closed_inputs", False))

        if not stream_final:
            if vhs_closed:
                state.abort()
                with _ENCODERS_LOCK:
                    _ENCODERS.pop(key, None)
                raise RuntimeError(
                    "VHS closed the input stream before the RIFE stream marked its final batch; "
                    "refusing to finalize because exact-duration tail frames may be missing."
                )
            requeue_unchecked()
            return {"ui": {"unfinished_batch": [True]}, "result": ((save_output, []),)}

        # The RIFE stream's final flag is authoritative because it comes from the
        # VDA session's exact source-frame accounting.  VHS normally learns EOF only
        # when its generator is resumed once past the final frame.  For an exact
        # multiple of frames_per_batch (e.g. 132 = 3x44), that would cause a bogus
        # fourth workflow run and then `RuntimeError: No frames generated`.
        # Retire any still-suspended loader generator here instead.
        if not vhs_closed:
            closed = _close_meta_batch_inputs_for_final(meta_batch)
            print(
                f"[LongVideo Stream Encoder] final RIFE batch closed {closed} lingering "
                "VHS input generator(s); no EOF probe requeue needed"
            )

        tf = time.perf_counter()
        final_path = state.finish()
        finish_sec = time.perf_counter() - tf
        profile = getattr(state, "_longvideo_profile", {})
        if profile:
            profile["finish"] = float(finish_sec)
            print(
                f"[LongVideo Stream Profile TOTAL] batches={int(profile.get('batches', 0))} "
                f"scene={float(profile.get('scene', 0.0)):.3f}s "
                f"layout={float(profile.get('layout', 0.0)):.3f}s "
                f"h2d={float(profile.get('h2d', 0.0)):.3f}s "
                f"h2d_wait={float(profile.get('h2d_wait', 0.0)):.3f}s "
                f"gpu_prep={float(profile.get('gpu_prepare', 0.0)):.3f}s "
                f"rife={float(profile.get('rife', 0.0)):.3f}s "
                f"ifnet={float(profile.get('ifnet', 0.0)):.3f}s "
                f"rife_post={float(profile.get('rife_post', 0.0)):.3f}s "
                f"rife_pack={float(profile.get('rife_pack', 0.0)):.3f}s "
                f"rife_calls={int(profile.get('rife_calls', 0))} rife_pairs={int(profile.get('rife_pairs', 0))} "
                f"pair_batch={int(profile.get('pair_batch', 1))} "
                f"feature_cache={str(profile.get('feature_cache', 'off'))} "
                f"feat_extract={int(profile.get('feature_cache_extract_calls', 0))} "
                f"feat_reuse={int(profile.get('feature_cache_reuses', 0))} "
                f"feat_pairs={int(profile.get('feature_cache_pairs', 0))} "
                f"feat_resets={int(profile.get('feature_cache_resets', 0))} "
                f"d2h={float(profile.get('d2h', 0.0)):.3f}s "
                f"d2h_wait={float(profile.get('d2h_wait', 0.0)):.3f}s "
                f"h2d_pin_alloc={float(profile.get('h2d_pin_alloc', 0.0)):.3f}s "
                f"h2d_pin_new={int(profile.get('h2d_pin_new', 0))} h2d_pin_reuse={int(profile.get('h2d_pin_reuse', 0))} "
                f"h2d_pin_peak={int(profile.get('h2d_pin_peak', 0))} "
                f"d2h_pin_alloc={float(profile.get('pin_alloc', 0.0)):.3f}s "
                f"d2h_pin_new={int(profile.get('pin_new', 0))} d2h_pin_reuse={int(profile.get('pin_reuse', 0))} "
                f"d2h_pin_peak={int(profile.get('pin_peak', 0))} "
                f"assemble={float(profile.get('assemble', 0.0)):.3f}s "
                f"rgb_convert={float(profile.get('convert', 0.0)):.3f}s "
                f"queue_wait={float(profile.get('queue_wait', 0.0)):.3f}s "
                f"stream_wall={float(profile.get('stream_wall', 0.0)):.3f}s "
                f"finish={finish_sec:.3f}s"
            )
            if int(profile.get("ifnet_stage_profiled_forwards", 0)):
                print(
                    "[LongVideo IFNet Stage Profile TOTAL] "
                    + _format_ifnet_stage_profile(
                        profile.get("ifnet_stage_seconds", {}),
                        profile.get("ifnet_stage_calls", {}),
                        float(profile.get("ifnet_stage_unattributed", 0.0)),
                        int(profile.get("ifnet_stage_profiled_forwards", 0)),
                    )
                )
        elapsed = time.perf_counter() - state.started
        expected_frames = int(getattr(stream.state, "output_frames", state.frames_written))
        if state.frames_written != expected_frames:
            with _ENCODERS_LOCK:
                _ENCODERS.pop(key, None)
            state.abort(remove_partial=True)
            raise RuntimeError(
                f"Final encoded frame count mismatch: wrote {state.frames_written}, expected {expected_frames}"
            )
        print(
            f"[LongVideo Stream Encoder] finished frames={state.frames_written} "
            f"raw_gib={state.bytes_written / (1024**3):.2f} "
            f"encoder_write={state.write_seconds:.2f}s wall={elapsed:.2f}s -> {final_path}"
        )
        with _ENCODERS_LOCK:
            _ENCODERS.pop(key, None)

        preview = {
            "filename": os.path.basename(final_path),
            "subfolder": state.subfolder,
            "type": "output" if save_output else "temp",
            "format": state.source_format_label,
            "frame_rate": frame_rate,
            "fullpath": final_path,
        }
        return {"ui": {"gifs": [preview]}, "result": ((save_output, [final_path]),)}


class LVEncoderCleanup:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"cleanup": ("BOOLEAN", {"default": True})}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "cleanup"
    CATEGORY = CATEGORY

    def cleanup(self, cleanup=True):
        if not cleanup:
            return ("No cleanup requested",)
        with _ENCODERS_LOCK:
            items = list(_ENCODERS.items())
            _ENCODERS.clear()
        for _, state in items:
            state.abort()
        return (f"Aborted {len(items)} LongVideo encoder worker(s)",)


def _cleanup_all():
    with _ENCODERS_LOCK:
        items = list(_ENCODERS.values())
        _ENCODERS.clear()
    for state in items:
        try:
            state.abort()
        except Exception:
            pass


atexit.register(_cleanup_all)

NODE_CLASS_MAPPINGS = {
    "LV_EdgeAwareDepthSmooth": LVEdgeAwareDepthSmooth,
    "LV_DepthStereoTuning": LVDepthStereoTuning,
    "LV_ResolutionStereoControls": LVResolutionStereoControls,
    "LV_AsyncVideoEncoder": LVAsyncVideoEncoder,
    "LV_StreamingVideoEncoder": LVStreamingVideoEncoder,
    "LV_AsyncLosslessVideoEncoder": LVAsyncLosslessVideoEncoder,
    "LV_EncoderCleanup": LVEncoderCleanup,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LV_EdgeAwareDepthSmooth": "LongVideo • Edge-Aware Depth Smooth",
    "LV_DepthStereoTuning": "LongVideo • Stereo Comfort / Feather Tuning",
    "LV_ResolutionStereoControls": "LongVideo • Resolution-Normalized 3D Controls",
    "LV_AsyncVideoEncoder": "LongVideo • Async Video Encoder",
    "LV_StreamingVideoEncoder": "LongVideo • RIFE Streaming Video Encoder",
    "LV_AsyncLosslessVideoEncoder": "LongVideo • Async Video Encoder (Legacy Presets)",
    "LV_EncoderCleanup": "LongVideo • Encoder Cleanup",
}
