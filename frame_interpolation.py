from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import torch


@dataclass
class LongVideoInterpolationState:
    signature: Tuple[Any, ...]
    last_frame: Optional[torch.Tensor] = None
    input_frames: int = 0
    output_frames: int = 0
    source_name: Optional[str] = None
    # DEV2.3.1: retained per-run pool for async GPU->CPU interpolation outputs.
    # It is deliberately not part of the signature/equality semantics.
    pinned_pool: Optional[Any] = None
    # DEV2.4: retained pinned BCHW staging buffers for async CPU->GPU source uploads.
    h2d_pinned_pool: Optional[Any] = None
    # DEV2.8: cumulative detail-profiler counters across meta-batches.
    ifnet_detail_totals: Optional[Dict[str, Any]] = None
    # DEV2.10: cumulative exact-math FP32 warp-source cache counters.
    warp_input_cache_totals: Optional[Dict[str, int]] = None
    # DEV2.11: cumulative exact-math repeated-flow grid reuse counters.
    warp_grid_cache_totals: Optional[Dict[str, int]] = None
    # DEV2.12: cumulative block4 internal profiler counters.
    ifnet_block4_totals: Optional[Dict[str, Any]] = None
    # DEV2.14: cumulative exact-output block4 ResConv in-place residual counters.
    block4_res_inplace_totals: Optional[Dict[str, int]] = None
    # DEV2.15: cumulative exact-output block4 scale=1 interpolate bypass counters.
    block4_scale1_totals: Optional[Dict[str, int]] = None
    # DEV2.16: cumulative post-optimization convolution-family profiler counters.
    ifnet_conv_family_totals: Optional[Dict[str, Any]] = None
    # DEV2.17: cumulative optimized IFNet outer-path profiler counters.
    ifnet_outer_totals: Optional[Dict[str, Any]] = None
    # DEV2.18: cumulative exact-output FP16 outer-cat pushdown counters.
    outer_cat_pushdown_totals: Optional[Dict[str, int]] = None
    # DEV2.19: cumulative post-optimization warp-inner profiler totals.
    warp_postopt_totals: Optional[Dict[str, Any]] = None


_STATES: Dict[Tuple[int, str], LongVideoInterpolationState] = {}
_STATES_LOCK = threading.RLock()


class _PinnedCpuBufferPool:
    """Small reusable pool for async RIFE D2H destinations.

    Pinned allocations are comparatively expensive and can force allocator/driver
    synchronization on ROCm.  The interpolation stream only needs a handful of
    in-flight output frames for chunk_frames=8, so recycle those buffers across
    temporal pairs *and* meta-batches instead of allocating a new pinned 4K tensor
    for every prediction.
    """

    def __init__(self):
        self._free: Dict[Tuple[Tuple[int, ...], torch.dtype], list[torch.Tensor]] = {}
        self.allocations = 0
        self.reuses = 0
        self.inflight = 0
        self.peak_inflight = 0
        self.alloc_seconds = 0.0

    @staticmethod
    def _key(shape, dtype):
        return (tuple(int(x) for x in shape), dtype)

    def acquire(self, shape, dtype):
        key = self._key(shape, dtype)
        bucket = self._free.get(key)
        if bucket:
            out = bucket.pop()
            self.reuses += 1
        else:
            t0 = _time.perf_counter()
            out = torch.empty(key[0], dtype=dtype, device="cpu", pin_memory=True)
            self.alloc_seconds += _time.perf_counter() - t0
            self.allocations += 1
        self.inflight += 1
        self.peak_inflight = max(self.peak_inflight, self.inflight)
        return out

    def release(self, tensor):
        if tensor is None:
            return
        key = self._key(tensor.shape, tensor.dtype)
        self._free.setdefault(key, []).append(tensor)
        self.inflight = max(0, self.inflight - 1)

    def snapshot(self):
        return {
            "allocations": int(self.allocations),
            "reuses": int(self.reuses),
            "peak_inflight": int(self.peak_inflight),
            "alloc_seconds": float(self.alloc_seconds),
        }

    def clear(self):
        self._free.clear()
        self.inflight = 0


# v3.7.1: treat a source that is already effectively at the requested target
# cadence as a 1x passthrough instead of forcing RIFE to double it.  The window
# is intentionally narrow: it covers common clock/cadence variants such as
# 119.88/120.000/120.006 for a 120-fps target, but does not silently accept a
# materially lower source (for example 100 fps -> 120 fps).
_TARGET_RATE_BYPASS_REL_TOL = 0.005  # 0.5%
_TARGET_RATE_BYPASS_ABS_TOL = 0.25   # fps floor for lower target rates


def _target_rate_bypass_tolerance(target_fps: float) -> float:
    return max(_TARGET_RATE_BYPASS_ABS_TOL, abs(float(target_fps)) * _TARGET_RATE_BYPASS_REL_TOL)


def resolve_integer_multiplier(source_fps: float, target_fps: float) -> Tuple[int, float]:
    """Resolve a cadence-preserving integer RIFE multiplier.

    23.976/29.97/59.94 sources intentionally resolve to 119.88 for target=120,
    rather than inventing a non-uniform cadence merely to label the stream 120.000.

    v3.7.1 adds a 1x passthrough when the source is already within the narrow
    target-rate tolerance.  Passthrough keeps the source cadence as output_fps;
    it never relabels 119.88 as 120.000.
    """
    src = float(source_fps)
    target = float(target_fps)
    if not math.isfinite(src) or src <= 0:
        raise ValueError(f"source_fps must be positive, got {source_fps!r}")
    if not math.isfinite(target) or target <= 0:
        raise ValueError(f"target_fps must be positive, got {target_fps!r}")

    tolerance = _target_rate_bypass_tolerance(target)
    if abs(src - target) <= tolerance:
        return 1, src

    if target <= src:
        raise ValueError(
            f"source_fps {src:.6f} is already above target_fps {target:.6f} and outside "
            f"the {tolerance:.3f}-fps target-rate bypass tolerance"
        )

    multiplier = int(round(target / src))
    if multiplier < 2 or multiplier > 16:
        raise ValueError(
            f"Resolved interpolation multiplier {multiplier}x is outside supported range 2..16 "
            f"for {src:.6f} -> {target:.6f} fps; source is not close enough to target for 1x bypass"
        )
    return multiplier, src * multiplier


def _state_key(session: Any, unique_id: Any) -> Tuple[int, str]:
    return (id(session), str(unique_id))


def _scene_cut_pair(a: torch.Tensor, b: torch.Tensor, threshold: float) -> bool:
    """Hard-cut detector using an area-averaged 48x64 luminance proxy.

    DEV2.2 keeps the DEV2.1 metric semantics but changes the operation order:
    average RGB to 48x64 in the source dtype first, then convert only the tiny
    proxy to float32 for BT.709 luma/MAD. This avoids allocating/converting full
    4K float32 frames while preserving the low-pass behavior of area resize.
    """
    if threshold <= 0:
        return False
    if a.ndim != 4 or b.ndim != 4 or a.shape[-1] < 3 or b.shape[-1] < 3:
        raise ValueError("Scene-cut detector expects BHWC RGB frames")

    target = (48, 64)
    # Average is linear, so RGB-average then luma is equivalent to luma then
    # area-average apart from tiny source-dtype rounding. Keeping this step in
    # float16 on our production SBS path removes the expensive full-frame fp32
    # conversion that dominated DEV2.1 scene detection.
    x = torch.nn.functional.adaptive_avg_pool2d(a[..., :3].movedim(-1, 1), target).float()
    y = torch.nn.functional.adaptive_avg_pool2d(b[..., :3].movedim(-1, 1), target).float()
    x = x[:, 0:1] * 0.2126 + x[:, 1:2] * 0.7152 + x[:, 2:3] * 0.0722
    y = y[:, 0:1] * 0.2126 + y[:, 1:2] * 0.7152 + y[:, 2:3] * 0.0722
    mad = torch.mean(torch.abs(x - y)).item()
    return mad >= float(threshold)


def _precompute_scene_flags(images: torch.Tensor, threshold: float) -> tuple[list[bool], float]:
    """Evaluate the unchanged DEV2.2 hard-cut detector before GPU transfer overlap begins.

    DEV2.4 evaluated scene pairs inside the RIFE loop while async H2D/D2H DMA was
    active. On the production 7900 XTX run that increased CPU detector wall time
    about 4x through memory-bandwidth contention. DEV2.4.1 performs the exact
    same pairwise detector in a short CPU prepass, preserving decisions while
    keeping the detector out of the transfer window.
    """
    total_pairs = max(0, int(images.shape[0]) - 1)
    if total_pairs <= 0 or float(threshold) <= 0:
        return [False] * total_pairs, 0.0
    t0 = _time.perf_counter()
    with torch.inference_mode():
        flags = [
            _scene_cut_pair(images[i:i + 1], images[i + 1:i + 2], float(threshold))
            for i in range(total_pairs)
        ]
    return flags, _time.perf_counter() - t0


def _hold_intermediates(frame0: torch.Tensor, count: int, dtype: torch.dtype) -> torch.Tensor:
    if count <= 0:
        return frame0[:0].to(dtype=dtype, device="cpu")
    return frame0.to(device="cpu", dtype=dtype).expand(count, -1, -1, -1).clone()


def stitch_batch_with_backend(
    images: torch.Tensor,
    state: LongVideoInterpolationState,
    multiplier: int,
    backend: Callable[[torch.Tensor, int], torch.Tensor],
) -> torch.Tensor:
    """Reference state-stitching helper used by regression tests.

    backend(sequence, multiplier) must return (N-1)*multiplier+1 frames. The
    previous source frame is prepended on later meta-batches and exactly one
    duplicate frame is removed, preserving the boundary interpolation interval.
    """
    if images.ndim != 4 or images.shape[0] < 1:
        raise ValueError("Expected non-empty BHWC IMAGE batch")
    work = images
    drop = 0
    if state.last_frame is not None:
        work = torch.cat((state.last_frame.to(images.dtype), images), dim=0)
        drop = 1
    out = backend(work, int(multiplier))
    expected = (work.shape[0] - 1) * int(multiplier) + 1
    if int(out.shape[0]) != expected:
        raise RuntimeError(f"Interpolation backend returned {out.shape[0]} frames; expected {expected}")
    if drop:
        out = out[1:]
    state.last_frame = images[-1:].detach().cpu().clone()
    state.input_frames += int(images.shape[0])
    state.output_frames += int(out.shape[0])
    return out


def _prepare_frame(frame: torch.Tensor, device: torch.device, dtype: torch.dtype, align: int) -> torch.Tensor:
    x = frame.movedim(-1, 1).to(device=device, dtype=dtype)
    if align > 1:
        from comfy.ldm.common_dit import pad_to_patch_size
        x = pad_to_patch_size(x, (align, align), padding_mode="reflect")
    return x


def interpolate_sequence_core(
    interp_model: Any,
    images: torch.Tensor,
    multiplier: int,
    stereo_mode: str = "split_eyes",
    scene_cut: bool = True,
    scene_threshold: float = 0.22,
    cpu_output: str = "float16",
    timestep_batch: int = 1,
) -> torch.Tensor:
    """Long-video-friendly RIFE execution using ComfyUI's loaded INTERP_MODEL.

    Unlike the generic ComfyUI node, this routine writes directly into a CPU result
    allocation and supports treating Half-SBS eyes as independent streams while
    keeping their timestamps locked. It intentionally does not carry model feature
    caches across meta-batches yet; boundary *frames* are stateful and exact.
    """
    from comfy import model_management

    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"Expected BHWC RGB IMAGE, got shape {tuple(images.shape)}")
    if images.shape[0] < 2:
        return images.to(device="cpu", dtype=torch.float16 if cpu_output == "float16" else torch.float32)
    if multiplier < 2 or multiplier > 16:
        raise ValueError("multiplier must be in 2..16")
    if stereo_mode not in ("split_eyes", "full_sbs"):
        raise ValueError("stereo_mode must be split_eyes or full_sbs")
    if cpu_output not in ("float16", "float32"):
        raise ValueError("cpu_output must be float16 or float32")

    H, W = int(images.shape[1]), int(images.shape[2])
    if stereo_mode == "split_eyes" and W % 2:
        raise ValueError(f"split_eyes requires even SBS width, got {W}")

    device = interp_model.load_device
    model_dtype = interp_model.model_dtype()
    inference_model = interp_model.model
    align = int(getattr(inference_model, "pad_align", 1))
    out_dtype = torch.float16 if cpu_output == "float16" else torch.float32

    # Load the model once. Pixel count rather than expanded output size drives
    # inference activation memory; split-eyes still processes the same total pixels.
    activation_mem = inference_model.memory_used_forward(images.shape, model_dtype)
    model_management.load_models_gpu([interp_model], memory_required=activation_mem)

    total_pairs = int(images.shape[0]) - 1
    total_out = total_pairs * int(multiplier) + 1
    result = torch.empty((total_out, H, W, 3), dtype=out_dtype, device="cpu")
    result[0].copy_(images[0].to(device="cpu", dtype=out_dtype))
    out_idx = 1

    n_mid = int(multiplier) - 1
    timesteps = [j / float(multiplier) for j in range(1, int(multiplier))]
    timestep_batch = max(1, min(int(timestep_batch), n_mid))

    # Retain prepared frame1 as frame0 of the next temporal pair. This is an easy
    # source upload win within each meta-batch and mirrors ComfyUI core's logic.
    prepared_next = None

    with torch.inference_mode():
        for pair_idx in range(total_pairs):
            src0 = images[pair_idx : pair_idx + 1]
            src1 = images[pair_idx + 1 : pair_idx + 2]

            is_cut = bool(scene_cut) and _scene_cut_pair(src0, src1, float(scene_threshold))
            if is_cut:
                mids = _hold_intermediates(src0, n_mid, out_dtype)
                result[out_idx : out_idx + n_mid].copy_(mids)
                out_idx += n_mid
                result[out_idx].copy_(src1[0].to(device="cpu", dtype=out_dtype))
                out_idx += 1
                prepared_next = None
                continue

            if stereo_mode == "split_eyes":
                half = W // 2
                if prepared_next is None:
                    p0 = torch.cat(
                        (
                            _prepare_frame(src0[:, :, :half], device, model_dtype, align),
                            _prepare_frame(src0[:, :, half:], device, model_dtype, align),
                        ),
                        dim=0,
                    )
                else:
                    p0 = prepared_next
                p1 = torch.cat(
                    (
                        _prepare_frame(src1[:, :, :half], device, model_dtype, align),
                        _prepare_frame(src1[:, :, half:], device, model_dtype, align),
                    ),
                    dim=0,
                )
                prepared_next = p1
                streams = 2
                crop_w = half
            else:
                p0 = prepared_next if prepared_next is not None else _prepare_frame(src0, device, model_dtype, align)
                p1 = _prepare_frame(src1, device, model_dtype, align)
                prepared_next = p1
                streams = 1
                crop_w = W

            pH, pW = int(p0.shape[2]), int(p0.shape[3])
            mids_by_t = []
            j = 0
            while j < n_mid:
                count = min(timestep_batch, n_mid - j)
                ts = timesteps[j : j + count]

                # Task layout: [stream0@t0..tn, stream1@t0..tn].
                a = p0.repeat_interleave(count, dim=0)
                b = p1.repeat_interleave(count, dim=0)
                t = torch.tensor(ts, device=device, dtype=model_dtype).repeat(streams)
                t = t.view(-1, 1, 1, 1).expand(-1, 1, pH, pW)

                try:
                    pred = inference_model(a, b, timestep=t, cache=None).clamp_(0.0, 1.0)
                except TypeError:
                    # Older detected IFNet variants may not expose the cache kwarg.
                    pred = inference_model(a, b, timestep=t).clamp_(0.0, 1.0)
                pred = pred[:, :, :H, :crop_w].detach().to(device="cpu", dtype=out_dtype)

                if streams == 1:
                    chunk = pred.movedim(1, -1)
                    mids_by_t.extend(chunk[k : k + 1] for k in range(count))
                else:
                    left = pred[:count].movedim(1, -1)
                    right = pred[count : count * 2].movedim(1, -1)
                    for k in range(count):
                        mids_by_t.append(torch.cat((left[k : k + 1], right[k : k + 1]), dim=2))
                j += count

            mids = torch.cat(mids_by_t, dim=0)
            result[out_idx : out_idx + n_mid].copy_(mids)
            out_idx += n_mid
            result[out_idx].copy_(src1[0].to(device="cpu", dtype=out_dtype))
            out_idx += 1

    if out_idx != total_out:
        raise RuntimeError(f"Interpolation accounting error: wrote {out_idx}, expected {total_out}")
    return result


def find_rife_models() -> Dict[str, str]:
    """Return display-name -> absolute-path for core and legacy RIFE model folders."""
    import folder_paths

    found: Dict[str, str] = {}
    try:
        for name in folder_paths.get_filename_list("frame_interpolation"):
            path = folder_paths.get_full_path("frame_interpolation", name)
            if path and os.path.isfile(path):
                found[f"frame_interpolation/{name}"] = path
    except Exception:
        pass

    models_dir = getattr(folder_paths, "models_dir", None)
    if models_dir:
        for rel in ("rife", os.path.join("vfi", "rife")):
            rife_dir = os.path.join(models_dir, rel)
            if not os.path.isdir(rife_dir):
                continue
            for name in sorted(os.listdir(rife_dir)):
                if name.lower().endswith((".pth", ".pt", ".safetensors")):
                    path = os.path.join(rife_dir, name)
                    if os.path.isfile(path):
                        found.setdefault(f"{rel}/{name}", path)
    return found


class _RIFEBypassModel:
    """Typed lightweight sentinel used when the user explicitly disables RIFE."""
    _longvideo_model_name = "RIFE-disabled"
    _longvideo_manual_bypass = True


_RIFE_BYPASS_MODEL = _RIFEBypassModel()


def _manual_rife_bypass(interp_model: Any) -> bool:
    return bool(getattr(interp_model, "_longvideo_manual_bypass", False))


class LV_RIFEModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        models = find_rife_models()
        choices = ["auto"] + sorted(models)
        return {
            "required": {
                "model": (choices, {"default": "auto"}),
                "enable_rife": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("INTERP_MODEL",)
    RETURN_NAMES = ("interp_model",)
    FUNCTION = "load"
    CATEGORY = "Long Video SBS"

    def load(self, model, enable_rife=True):
        if not bool(enable_rife):
            # Do not touch the model files or Comfy model-management stack.  The
            # downstream interpolation nodes recognize this sentinel and emit a
            # true 1x source-cadence stream.
            return (_RIFE_BYPASS_MODEL,)

        models = find_rife_models()
        if model == "auto":
            # DEV2 production candidate: prefer Practical-RIFE 4.25, then 4.26.
            ranked = sorted(
                models.items(),
                key=lambda kv: (
                    0 if ("425" in kv[0].lower() or "4.25" in kv[0].lower()) else
                    1 if ("426" in kv[0].lower() or "4.26" in kv[0].lower()) else 2,
                    kv[0].lower(),
                ),
            )
            path = ranked[0][1] if ranked else None
        else:
            path = models.get(model)
        if not path:
            raise RuntimeError(
                "No RIFE model found. Put a RIFE 4.25 checkpoint in ComfyUI/models/frame_interpolation/ "
                "(or keep an existing copy in ComfyUI/models/rife/ or models/vfi/rife/) and restart ComfyUI."
            )

        import comfy.model_patcher
        import comfy.utils
        from comfy import model_management
        from comfy_extras.nodes_frame_interpolation import FrameInterpolationModelLoader

        sd = comfy.utils.load_torch_file(path, safe_load=True)
        net = FrameInterpolationModelLoader._detect_and_load(sd)
        dtype = torch.float16 if model_management.should_use_fp16(model_management.get_torch_device()) else torch.float32
        net.eval().to(dtype)
        patcher = comfy.model_patcher.CoreModelPatcher(
            net,
            load_device=model_management.get_torch_device(),
            offload_device=model_management.unet_offload_device(),
        )
        patcher._longvideo_model_name = os.path.basename(path)
        return (patcher,)


class LV_LongVideoFrameInterpolation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "interp_model": ("INTERP_MODEL",),
                "images": ("IMAGE",),
                "session": ("LV_SESSION",),
                "source_fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 240.0, "step": 0.001}),
                "target_fps": ("FLOAT", {"default": 120.0, "min": 2.0, "max": 480.0, "step": 0.001}),
                "stereo_mode": (["split_eyes", "full_sbs"], {"default": "split_eyes"}),
                "scene_cut": ("BOOLEAN", {"default": True}),
                "scene_threshold": ("FLOAT", {"default": 0.22, "min": 0.02, "max": 1.0, "step": 0.01}),
                "cpu_output": (["float16", "float32"], {"default": "float16"}),
                "timestep_batch": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "STRING")
    RETURN_NAMES = ("interpolated", "output_fps", "status")
    FUNCTION = "interpolate"
    CATEGORY = "Long Video SBS"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def interpolate(
        self,
        interp_model,
        images,
        session,
        source_fps,
        target_fps,
        stereo_mode,
        scene_cut,
        scene_threshold,
        cpu_output,
        timestep_batch,
        unique_id=None,
    ):
        manual_bypass = _manual_rife_bypass(interp_model)
        if manual_bypass:
            multiplier, output_fps = 1, float(source_fps)
        else:
            multiplier, output_fps = resolve_integer_multiplier(source_fps, target_fps)
        if multiplier == 1:
            key = _state_key(session, unique_id)
            with _STATES_LOCK:
                stale_keys = [k for k in _STATES if k[1] == str(unique_id)]
                for stale_key in stale_keys:
                    stale = _STATES.pop(stale_key, None)
                    if stale is not None and getattr(stale, "pinned_pool", None) is not None:
                        stale.pinned_pool.clear()
                    if stale is not None and getattr(stale, "h2d_pinned_pool", None) is not None:
                        stale.h2d_pinned_pool.clear()
            reason = "manual enable_rife=false" if manual_bypass else "source already at target cadence"
            status = (
                f"RIFE BYPASS 1x | {float(source_fps):.6f}->{output_fps:.6f} fps | "
                f"{reason}; frames unchanged"
            )
            return (images, float(output_fps), status)

        signature = (
            float(source_fps), float(target_fps), int(multiplier), str(stereo_mode),
            bool(scene_cut), float(scene_threshold), str(cpu_output), int(timestep_batch),
        )
        key = _state_key(session, unique_id)
        with _STATES_LOCK:
            # DEV2.3.1: interrupted prompts can leave the prior run's small state
            # (last frame / pinned pool) resident.  A new VDA session is a new run;
            # retire older states for this same interpolation node before starting.
            stale_keys = [k for k in _STATES if k[1] == str(unique_id) and k != key]
            for stale_key in stale_keys:
                stale = _STATES.pop(stale_key, None)
                if stale is not None and getattr(stale, "pinned_pool", None) is not None:
                    stale.pinned_pool.clear()
                    stale.pinned_pool = None
                if stale is not None and getattr(stale, "h2d_pinned_pool", None) is not None:
                    stale.h2d_pinned_pool.clear()
                    stale.h2d_pinned_pool = None
            state = _STATES.get(key)
            if state is None or state.signature != signature:
                state = LongVideoInterpolationState(signature=signature)
                _STATES[key] = state

        work = images
        drop_first = False
        if state.last_frame is not None:
            work = torch.cat((state.last_frame.to(dtype=images.dtype), images), dim=0)
            drop_first = True

        out = interpolate_sequence_core(
            interp_model=interp_model,
            images=work,
            multiplier=multiplier,
            stereo_mode=str(stereo_mode),
            scene_cut=bool(scene_cut),
            scene_threshold=float(scene_threshold),
            cpu_output=str(cpu_output),
            timestep_batch=int(timestep_batch),
        )
        if drop_first:
            # Prior meta-batch already emitted the retained source frame. Keep all
            # newly generated 43->44 intermediates, but remove that one duplicate.
            out = out[1:]

        state.last_frame = images[-1:].detach().cpu().clone()
        state.input_frames += int(images.shape[0])
        state.output_frames += int(out.shape[0])

        finished = bool(getattr(session, "finished", False))
        status = (
            f"RIFE {multiplier}x | {float(source_fps):.6f}->{output_fps:.6f} fps | "
            f"stereo={stereo_mode} | input={state.input_frames} | output={state.output_frames} | "
            f"final={finished}"
        )
        if abs(output_fps - float(target_fps)) > 0.01:
            status += f" | cadence-preserving target differs by {output_fps - float(target_fps):+.3f} fps"

        if finished:
            with _STATES_LOCK:
                _STATES.pop(key, None)

        return (out, float(output_fps), status)

# -----------------------------------------------------------------------------
# v3.5 DEV2 bounded streaming interpolation
# -----------------------------------------------------------------------------

import time as _time
from typing import Iterator as _Iterator, List as _List


def _model_label(interp_model: Any) -> str:
    return str(getattr(interp_model, "_longvideo_model_name", "RIFE"))


class _IFNetStageProfiler:
    """Low-overhead stage profiler for ComfyUI's RIFE IFNet.

    The profiler is installed only for a profiled stream batch, so production/off
    runs pay no forward-hook cost.  On current ComfyUI RIFE models we time the
    reusable encode head plus each IFBlock.  The IFNet envelope timer already
    present in DEV2.5 lets us attribute the remainder to outer warp/grid-sample,
    concatenation, mask/flow bookkeeping, and final blend work.
    """

    def __init__(self, model: Any, profile_warp: bool = False, profile_warp_inner: bool = False, profile_block4: bool = False, profile_conv_family: bool = False, profile_outer: bool = False):
        self.model = model
        self._handles = []
        self._active = None
        self.stage_names = []
        self._profile_warp = bool(profile_warp)
        self._profile_warp_inner = bool(profile_warp_inner)
        self._profile_block4 = bool(profile_block4)
        self._profile_conv_family = bool(profile_conv_family)
        self._profile_outer = bool(profile_outer)
        self._warp_had_instance_attr = False
        self._warp_instance_attr = None
        self._forward_had_instance_attr = False
        self._forward_instance_attr = None


        stages = []
        encode = getattr(model, "encode", None)
        blocks = getattr(model, "blocks", None)
        if isinstance(encode, torch.nn.Module):
            stages.append(("encode", encode))
        if blocks is not None:
            try:
                for i, block in enumerate(blocks):
                    if isinstance(block, torch.nn.Module):
                        stages.append((f"block{i}", block))
            except TypeError:
                pass

        # Generic fallback for future ComfyUI interpolation models. Avoid
        # container modules whose children would otherwise double-count time.
        if not stages and isinstance(model, torch.nn.Module):
            for name, child in model.named_children():
                if isinstance(child, (torch.nn.ModuleList, torch.nn.Sequential)):
                    continue
                if isinstance(child, torch.nn.Module):
                    stages.append((str(name), child))

        for stage, module in stages:
            self.stage_names.append(stage)
            self._handles.append(module.register_forward_pre_hook(self._make_pre_hook(stage)))
            self._handles.append(module.register_forward_hook(self._make_post_hook(stage)))

        # DEV2.12 instrumentation-only: block4 is the dominant remaining IFBlock
        # on the 4K ROCm benchmark. Profile its child modules without changing
        # execution. Parent block4 timing remains present; block4_misc is derived
        # later as parent time minus these non-overlapping children.
        if self._profile_block4:
            try:
                block4 = list(getattr(model, "blocks"))[4]
                children = []
                conv0 = getattr(block4, "conv0", None)
                if isinstance(conv0, torch.nn.Sequential) and len(conv0) >= 2:
                    children.extend([("block4_conv0_0", conv0[0]), ("block4_conv0_1", conv0[1])])
                convblock = getattr(block4, "convblock", None)
                if isinstance(convblock, torch.nn.Sequential):
                    for i, child in enumerate(convblock):
                        children.append((f"block4_res{i}", child))
                        # DEV2.13: current ComfyUI ResConv exposes an explicit
                        # Conv2d and LeakyReLU. Time them as nested children so
                        # the remainder of the parent ResConv is the exact
                        # addcmul/residual bookkeeping cost. These nested stages
                        # are excluded from parent/misc attribution below.
                        child_conv = getattr(child, "conv", None)
                        child_relu = getattr(child, "relu", None)
                        if isinstance(child_conv, torch.nn.Module):
                            children.append((f"block4_res{i}_conv", child_conv))
                        if isinstance(child_relu, torch.nn.Module):
                            children.append((f"block4_res{i}_relu", child_relu))
                lastconv = getattr(block4, "lastconv", None)
                if isinstance(lastconv, torch.nn.Sequential) and len(lastconv) >= 1:
                    children.append(("block4_deconv", lastconv[0]))
                    if len(lastconv) >= 2:
                        children.append(("block4_pixelshuffle", lastconv[1]))
                for stage, module in children:
                    if isinstance(module, torch.nn.Module):
                        self.stage_names.append(stage)
                        self._handles.append(module.register_forward_pre_hook(self._make_pre_hook(stage)))
                        self._handles.append(module.register_forward_hook(self._make_post_hook(stage)))
            except Exception:
                # Keep the generic stage profiler usable on future/non-IFNet models.
                pass

        # DEV2.16 instrumentation-only: refresh the convolution hotspot map
        # after adjacent feature reuse, grid reuse, and the locked block4 scale=1
        # bypass. Time the learned convolution family in every IFBlock without
        # changing execution or memory format. Parent block timings above remain
        # available; these nested child timings are excluded from outer attribution.
        if self._profile_conv_family:
            try:
                for bi, block in enumerate(list(getattr(model, "blocks"))):
                    children = []
                    conv0 = getattr(block, "conv0", None)
                    if isinstance(conv0, torch.nn.Sequential) and len(conv0) >= 2:
                        children.extend([(f"block{bi}_conv0_0", conv0[0]), (f"block{bi}_conv0_1", conv0[1])])
                    convblock = getattr(block, "convblock", None)
                    if isinstance(convblock, torch.nn.Sequential):
                        for ri, child in enumerate(convblock):
                            child_conv = getattr(child, "conv", None)
                            if isinstance(child_conv, torch.nn.Module):
                                children.append((f"block{bi}_res{ri}_conv", child_conv))
                    lastconv = getattr(block, "lastconv", None)
                    if isinstance(lastconv, torch.nn.Sequential) and len(lastconv) >= 1:
                        children.append((f"block{bi}_deconv", lastconv[0]))
                        if len(lastconv) >= 2:
                            children.append((f"block{bi}_pixelshuffle", lastconv[1]))
                    for stage, module in children:
                        if isinstance(module, torch.nn.Module):
                            self.stage_names.append(stage)
                            self._handles.append(module.register_forward_pre_hook(self._make_pre_hook(stage)))
                            self._handles.append(module.register_forward_hook(self._make_post_hook(stage)))
            except Exception:
                pass

        # DEV2.17 instrumentation-only: decompose the optimized IFNet outer path
        # before changing any more kernels. This exact copy of current ComfyUI's
        # IFNet.forward times only the non-module operations surrounding the
        # already-hooked encode/blocks: concatenations, feature/image warps,
        # in-place flow accumulation, sigmoid, and final lerp. The installed
        # grid-reuse warp and block4 scale=1 bypass remain in effect.
        if self._profile_outer:
            try:
                blocks = list(getattr(model, "blocks"))
                scales = list(getattr(model, "scale_list"))
                if len(blocks) != 5 or len(scales) != 5:
                    raise ValueError("outer_detail requires current five-block IFNet")
                if not callable(getattr(model, "warp", None)) or not callable(getattr(model, "_build_warp_grids", None)):
                    raise ValueError("outer_detail requires current IFNet warp/grid contract")
                self._forward_had_instance_attr = "forward" in getattr(model, "__dict__", {})
                if self._forward_had_instance_attr:
                    self._forward_instance_attr = model.__dict__.get("forward")
                original_forward = model.forward

                def _outer_timed(trace, stage, fn):
                    if trace["mode"] == "cuda":
                        start = torch.cuda.Event(enable_timing=True)
                        start.record()
                        out = fn()
                        end = torch.cuda.Event(enable_timing=True)
                        end.record()
                    else:
                        start = _time.perf_counter()
                        out = fn()
                        end = _time.perf_counter()
                    trace["events"].setdefault(stage, []).append((start, end))
                    return out

                def profiled_forward(img0, img1, timestep=0.5, cache=None, _model=model, _orig=original_forward):
                    trace = self._active
                    if trace is None:
                        return _orig(img0, img1, timestep=timestep, cache=cache)
                    if not isinstance(timestep, torch.Tensor):
                        timestep = _outer_timed(
                            trace, "outer_timestep",
                            lambda: torch.full((img0.shape[0], 1, img0.shape[2], img0.shape[3]), timestep, device=img0.device, dtype=img0.dtype),
                        )
                    _model._build_warp_grids(img0.shape[2], img0.shape[3], img0.device)
                    B = img0.shape[0]
                    f0 = cache["img0"].expand(B, -1, -1, -1) if cache and "img0" in cache else _model.encode(img0)
                    f1 = cache["img1"].expand(B, -1, -1, -1) if cache and "img1" in cache else _model.encode(img1)
                    flow = mask = feat = None
                    warped_img0, warped_img1 = img0, img1
                    for i, block in enumerate(_model.blocks):
                        if flow is None:
                            block_input = _outer_timed(
                                trace, "outer_cat_initial",
                                lambda: torch.cat((img0, img1, f0, f1, timestep), 1),
                            )
                            flow, mask, feat = block(block_input, None, scale=_model.scale_list[i])
                        else:
                            wf0 = _outer_timed(trace, "outer_warp_feature", lambda: _model.warp(f0, flow[:, :2]))
                            wf1 = _outer_timed(trace, "outer_warp_feature", lambda: _model.warp(f1, flow[:, 2:4]))
                            block_input = _outer_timed(
                                trace, "outer_cat_refine",
                                lambda: torch.cat((warped_img0, warped_img1, wf0, wf1, timestep, mask, feat), 1),
                            )
                            fd, mask, feat = block(block_input, flow, scale=_model.scale_list[i])
                            flow = _outer_timed(trace, "outer_flow_add", lambda: flow.add_(fd))
                        warped_img0 = _outer_timed(trace, "outer_warp_image", lambda: _model.warp(img0, flow[:, :2]))
                        warped_img1 = _outer_timed(trace, "outer_warp_image", lambda: _model.warp(img1, flow[:, 2:4]))
                    blend = _outer_timed(trace, "outer_sigmoid", lambda: torch.sigmoid(mask))
                    return _outer_timed(trace, "outer_lerp", lambda: torch.lerp(warped_img1, warped_img0, blend))

                model.forward = profiled_forward
                for name in (
                    "outer_timestep", "outer_cat_initial", "outer_cat_refine",
                    "outer_warp_feature", "outer_flow_add", "outer_warp_image",
                    "outer_sigmoid", "outer_lerp",
                ):
                    if name not in self.stage_names:
                        self.stage_names.append(name)
            except Exception:
                self._profile_outer = False

        # DEV2.8/DEV2.9 detail modes: current ComfyUI IFNet performs its
        # grid-sample warps through model.warp(), outside the child modules above.
        # ``detail`` times each complete image/feature warp. ``warp_detail``
        # reproduces the current ComfyUI IFNet warp expression operation-for-
        # operation and times its normalization, grid construction, fp32 casts,
        # grid_sample, and cast-back separately. The bound method is restored
        # exactly on close().
        if (self._profile_warp or self._profile_warp_inner) and callable(getattr(model, "warp", None)):
            original = getattr(model, "warp")
            self._warp_had_instance_attr = "warp" in getattr(model, "__dict__", {})
            if self._warp_had_instance_attr:
                self._warp_instance_attr = model.__dict__.get("warp")

            def _timed(trace, stage, fn):
                if trace["mode"] == "cuda":
                    start = torch.cuda.Event(enable_timing=True)
                    start.record()
                    out = fn()
                    end = torch.cuda.Event(enable_timing=True)
                    end.record()
                else:
                    start = _time.perf_counter()
                    out = fn()
                    end = _time.perf_counter()
                trace["events"].setdefault(stage, []).append((start, end))
                return out

            def profiled_warp(img, flow):
                trace = self._active
                if trace is None:
                    return original(img, flow)
                try:
                    channels = int(img.shape[1])
                except Exception:
                    channels = -1
                if channels == 3:
                    kind = "image"
                elif channels > 0:
                    kind = "feature"
                else:
                    kind = "other"

                if not self._profile_warp_inner:
                    return _timed(trace, f"warp_{kind}", lambda: original(img, flow))

                # Current ComfyUI RIFE IFNet _warp() contract (2026-08):
                #   flow_norm = cat(flow/div).float()
                #   grid = (base_grid.expand + flow_norm).permute(...)
                #   grid_sample(img.float(), grid, border, align_corners=True)
                #   .to(img.dtype)
                # If a future model no longer exposes the expected cached grids,
                # fall back to the original warp rather than changing behavior.
                try:
                    B, _, H, W = img.shape
                    warp_grids = getattr(model, "_warp_grids")
                    base_grid, flow_div = warp_grids[(H, W)]
                except Exception:
                    return _timed(trace, f"warp_{kind}_fallback", lambda: original(img, flow))

                prefix = f"warp_{kind}"
                flow_norm = _timed(
                    trace, prefix + "_norm",
                    lambda: torch.cat([
                        flow[:, 0:1] / flow_div[0],
                        flow[:, 1:2] / flow_div[1],
                    ], 1).float(),
                )
                grid = _timed(
                    trace, prefix + "_grid",
                    lambda: (base_grid.expand(B, -1, -1, -1) + flow_norm).permute(0, 2, 3, 1),
                )
                img32 = _timed(trace, prefix + "_cast_in", lambda: img.float())
                sampled = _timed(
                    trace, prefix + "_sample",
                    lambda: torch.nn.functional.grid_sample(
                        img32, grid, mode="bilinear", padding_mode="border", align_corners=True
                    ),
                )
                return _timed(trace, prefix + "_cast_out", lambda: sampled.to(img.dtype))

            model.warp = profiled_warp
            if self._profile_warp_inner:
                for kind in ("image", "feature", "other"):
                    for part in ("norm", "grid", "cast_in", "sample", "cast_out", "fallback"):
                        name = f"warp_{kind}_{part}"
                        if name not in self.stage_names:
                            self.stage_names.append(name)
            else:
                for name in ("warp_image", "warp_feature", "warp_other"):
                    if name not in self.stage_names:
                        self.stage_names.append(name)

    @property
    def usable(self) -> bool:
        return bool(self.stage_names)

    def begin(self, mode: str):
        if self._active is not None:
            raise RuntimeError("IFNet stage profiler received overlapping forwards")
        self._active = {"mode": str(mode), "events": {}, "stacks": {}}

    def end(self):
        trace = self._active
        self._active = None
        return trace

    def _make_pre_hook(self, stage: str):
        def hook(_module, _inputs):
            trace = self._active
            if trace is None:
                return
            if trace["mode"] == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                start.record()
            else:
                start = _time.perf_counter()
            trace["stacks"].setdefault(stage, []).append(start)
        return hook

    def _make_post_hook(self, stage: str):
        def hook(_module, _inputs, _output):
            trace = self._active
            if trace is None:
                return
            stack = trace["stacks"].get(stage)
            if not stack:
                return
            start = stack.pop()
            if trace["mode"] == "cuda":
                end = torch.cuda.Event(enable_timing=True)
                end.record()
            else:
                end = _time.perf_counter()
            trace["events"].setdefault(stage, []).append((start, end))
        return hook

    def close(self):
        self._active = None
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._handles.clear()
        if self._profile_outer:
            try:
                if self._forward_had_instance_attr:
                    self.model.forward = self._forward_instance_attr
                elif "forward" in getattr(self.model, "__dict__", {}):
                    delattr(self.model, "forward")
            except Exception:
                pass
        if self._profile_warp or self._profile_warp_inner:
            try:
                if self._warp_had_instance_attr:
                    self.model.warp = self._warp_instance_attr
                elif "warp" in getattr(self.model, "__dict__", {}):
                    delattr(self.model, "warp")
            except Exception:
                pass


class _IFNetWarpInputCache:
    """Exact-math reuse of IFNet's repeated ``img.float()`` warp sources.

    Current ComfyUI RIFE calls ``warp()`` 18 times per forward but those calls
    sample only four distinct source tensors: img0, img1, f0 and f1.  The stock
    warp converts its source to float32 every time.  DEV2.10 keeps the stock
    float32 grid and ``grid_sample`` math intact, but converts each distinct
    source only once per IFNet forward and reuses that float32 tensor.

    The cache lifetime is exactly one model forward.  This avoids retaining 4K
    float32 tensors across temporal pairs/meta-batches and makes restoration of
    the model's original bound method deterministic.
    """

    def __init__(self, model: Any):
        self.model = model
        self._cache: Dict[int, torch.Tensor] = {}
        self._active = False
        self.casts = 0
        self.reuses = 0
        self.peak_entries = 0
        self.forwards = 0
        self.fallbacks = 0
        self._warp_had_instance_attr = "warp" in getattr(model, "__dict__", {})
        self._warp_instance_attr = model.__dict__.get("warp") if self._warp_had_instance_attr else None
        self._original = getattr(model, "warp", None)
        if not callable(self._original):
            raise ValueError("model has no callable warp()")
        if not isinstance(getattr(model, "_warp_grids", None), dict):
            raise ValueError("model has no IFNet _warp_grids cache")

        def cached_warp(img, flow):
            if not self._active:
                return self._original(img, flow)
            try:
                B, _, H, W = img.shape
                base_grid, flow_div = self.model._warp_grids[(H, W)]
            except Exception:
                self.fallbacks += 1
                return self._original(img, flow)

            key = id(img)
            img32 = self._cache.get(key)
            if img32 is None:
                # This is exactly the stock ComfyUI conversion, merely retained
                # for the remaining warps in this same IFNet forward.
                img32 = img.float()
                self._cache[key] = img32
                self.casts += 1
                self.peak_entries = max(self.peak_entries, len(self._cache))
            else:
                self.reuses += 1

            # Preserve current ComfyUI IFNet warp expression/order exactly.
            flow_norm = torch.cat([
                flow[:, 0:1] / flow_div[0],
                flow[:, 1:2] / flow_div[1],
            ], 1).float()
            grid = (base_grid.expand(B, -1, -1, -1) + flow_norm).permute(0, 2, 3, 1)
            return torch.nn.functional.grid_sample(
                img32, grid, mode="bilinear", padding_mode="border", align_corners=True
            ).to(img.dtype)

        model.warp = cached_warp

    def begin_forward(self):
        self._cache.clear()
        self._active_version = None
        self._active = True
        self.forwards += 1

    def end_forward(self):
        self._active = False
        self._cache.clear()
        self._active_version = None

    def snapshot(self):
        return {
            "casts": int(self.casts),
            "reuses": int(self.reuses),
            "peak_entries": int(self.peak_entries),
            "forwards": int(self.forwards),
            "fallbacks": int(self.fallbacks),
        }

    def close(self):
        self.end_forward()
        try:
            if self._warp_had_instance_attr:
                self.model.warp = self._warp_instance_attr
            elif "warp" in getattr(self.model, "__dict__", {}):
                delattr(self.model, "warp")
        except Exception:
            pass



class _IFNetWarpGridCache:
    """Exact-math reuse of IFNet sampling grids across adjacent stages.

    Production ComfyUI runs IFNet under ``torch.inference_mode()``, whose tensors
    intentionally do not expose a usable version counter.  DEV2.11 originally
    tried to invalidate cached grids via ``flow._version`` and therefore fell
    back on every ROCm warp.

    DEV2.11.2 instead keys grids by the flow slice storage/view and invalidates
    them with IFBlock forward hooks.  This matches IFNet's execution order:

      previous-stage image warps -> next-stage feature warps -> block forward
      -> flow.add_(fd) -> next image warps

    The feature warps therefore reuse the exact grids from the preceding image
    warps.  The block hook clears those grids before the in-place flow update is
    consumed by the following image warps.  Sampling, source casting, grid math,
    padding, align_corners, and output dtype remain unchanged.
    """

    def __init__(self, model: Any, profile_inner: bool = False):
        self.model = model
        self._cache: Dict[Tuple[Any, ...], torch.Tensor] = {}
        self.profile_inner = bool(profile_inner)
        self._trace = None
        self._active = False
        self._active_version: Optional[Tuple[int, int]] = None
        self.builds = 0
        self.reuses = 0
        self.peak_entries = 0
        self.forwards = 0
        self.fallbacks = 0
        self.invalidations = 0
        self.strategy = "off"
        self._hook_handles = []
        self._warp_had_instance_attr = "warp" in getattr(model, "__dict__", {})
        self._warp_instance_attr = model.__dict__.get("warp") if self._warp_had_instance_attr else None
        self._original = getattr(model, "warp", None)
        if not callable(self._original):
            raise ValueError("model has no callable warp()")
        if not isinstance(getattr(model, "_warp_grids", None), dict):
            raise ValueError("model has no IFNet _warp_grids cache")

        # Real IFNet has a ModuleList of five IFBlocks.  Hooks are preferred to
        # tensor version counters because inference tensors do not track versions.
        blocks = getattr(model, "blocks", None)
        hookable = isinstance(blocks, (torch.nn.ModuleList, list, tuple)) and len(blocks) > 0
        if hookable and all(isinstance(block, torch.nn.Module) for block in blocks):
            def _after_block(_module, _inputs, _output):
                if self._active:
                    self._cache.clear()
                    self.invalidations += 1
            self._hook_handles = [block.register_forward_hook(_after_block) for block in blocks]
            self.strategy = "block_hooks"
        else:
            # Kept only for toy/future models outside inference_mode.  Production
            # IFNet should always use block_hooks.
            self.strategy = "tensor_version"

        def _timed(kind, part, fn):
            trace = self._trace
            if not self.profile_inner or trace is None:
                return fn()
            stage = f"warp_post_{kind}_{part}"
            if trace["mode"] == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                start.record()
                out = fn()
                end = torch.cuda.Event(enable_timing=True)
                end.record()
            else:
                start = _time.perf_counter()
                out = fn()
                end = _time.perf_counter()
            trace["events"].setdefault(stage, []).append((start, end))
            return out

        def cached_warp(img, flow):
            if not self._active:
                return self._original(img, flow)
            try:
                B, _, H, W = img.shape
                base_grid, flow_div = self.model._warp_grids[(H, W)]
                storage = flow.untyped_storage()
                storage_ptr = int(storage.data_ptr())
                slice_base = (
                    storage_ptr,
                    int(flow.storage_offset()),
                    tuple(int(v) for v in flow.shape),
                    tuple(int(v) for v in flow.stride()),
                    int(H), int(W),
                )
                if self.strategy == "tensor_version":
                    # This path intentionally falls back if an inference tensor is
                    # presented without block hooks; using a stale grid is never OK.
                    version = int(getattr(flow, "_version"))
                    version_key = (storage_ptr, version)
                    if self._active_version != version_key:
                        self._cache.clear()
                        self._active_version = version_key
                    slice_key = slice_base + (version,)
                else:
                    slice_key = slice_base
            except Exception:
                self.fallbacks += 1
                return self._original(img, flow)

            try:
                channels = int(img.shape[1])
            except Exception:
                channels = -1
            kind = "image" if channels == 3 else ("feature" if channels > 0 else "other")

            grid = self._cache.get(slice_key)
            if grid is None:
                # Preserve the stock expression/order exactly.
                flow_norm = _timed(kind, "norm", lambda: torch.cat([
                    flow[:, 0:1] / flow_div[0],
                    flow[:, 1:2] / flow_div[1],
                ], 1).float())
                grid = _timed(kind, "grid", lambda: (base_grid.expand(B, -1, -1, -1) + flow_norm).permute(0, 2, 3, 1))
                self._cache[slice_key] = grid
                self.builds += 1
                self.peak_entries = max(self.peak_entries, len(self._cache))
            else:
                self.reuses += 1

            img32 = _timed(kind, "cast_in", lambda: img.float())
            sampled = _timed(kind, "sample", lambda: torch.nn.functional.grid_sample(
                img32, grid, mode="bilinear", padding_mode="border", align_corners=True
            ))
            return _timed(kind, "cast_out", lambda: sampled.to(img.dtype))

        model.warp = cached_warp

    def begin_forward(self, mode: str = "cpu"):
        self._cache.clear()
        self._active = True
        self.forwards += 1
        self._trace = {"mode": str(mode), "events": {}} if self.profile_inner else None

    def end_forward(self):
        self._active = False
        self._cache.clear()
        trace = self._trace
        self._trace = None
        return trace

    def snapshot(self):
        return {
            "builds": int(self.builds),
            "reuses": int(self.reuses),
            "peak_entries": int(self.peak_entries),
            "forwards": int(self.forwards),
            "fallbacks": int(self.fallbacks),
            "invalidations": int(self.invalidations),
            "strategy": str(self.strategy),
        }

    def close(self):
        self.end_forward()
        for handle in self._hook_handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._hook_handles = []
        try:
            if self._warp_had_instance_attr:
                self.model.warp = self._warp_instance_attr
            elif "warp" in getattr(self.model, "__dict__", {}):
                delattr(self.model, "warp")
        except Exception:
            pass


class _IFNetBlock4ResConvInplace:
    """Inference-only exact-output ResConv residual update for IFNet block4.

    Current ComfyUI ResConv computes::

        relu(torch.addcmul(x, conv(x), beta))

    The block4 ResConvs are a simple Sequential chain, so each input ``x`` is
    dead after its residual result is produced.  DEV2.14 keeps Conv2d, beta,
    addcmul arithmetic and in-place LeakyReLU unchanged, but writes the addcmul
    result back into that dead input tensor::

        y = conv(x)
        x.addcmul_(y, beta)
        relu(x)

    ``torch.addcmul`` and ``Tensor.addcmul_`` are bit-identical for the tested
    float16/float32 path; no channel layout, convolution algorithm or dtype is
    changed.  The original forwards are restored on close().
    """

    def __init__(self, model: Any):
        self.model = model
        self.calls = 0
        self.patched = 0
        self._saved = []
        blocks = getattr(model, "blocks", None)
        if not isinstance(blocks, (torch.nn.ModuleList, list, tuple)) or len(blocks) < 5:
            raise ValueError("model has no IFNet block4")
        block4 = blocks[4]
        convblock = getattr(block4, "convblock", None)
        if not isinstance(convblock, torch.nn.Sequential) or len(convblock) != 8:
            raise ValueError("block4 does not expose the expected 8 ResConv modules")

        for res in convblock:
            conv = getattr(res, "conv", None)
            beta = getattr(res, "beta", None)
            relu = getattr(res, "relu", None)
            if not isinstance(conv, torch.nn.Module) or not isinstance(beta, torch.Tensor) or not isinstance(relu, torch.nn.Module):
                self.close()
                raise ValueError("block4 ResConv layout is not compatible with exact in-place residual mode")
            had = "forward" in getattr(res, "__dict__", {})
            old = res.__dict__.get("forward") if had else None
            self._saved.append((res, had, old))

            def _forward(x, _res=res, _self=self):
                y = _res.conv(x)
                x.addcmul_(y, _res.beta)
                _self.calls += 1
                return _res.relu(x)

            res.forward = _forward
            self.patched += 1

        if self.patched != 8:
            self.close()
            raise ValueError(f"expected 8 block4 ResConvs, patched {self.patched}")

    def snapshot(self):
        return {"calls": int(self.calls), "patched": int(self.patched)}

    def close(self):
        for res, had, old in reversed(self._saved):
            try:
                if had:
                    res.forward = old
                elif "forward" in getattr(res, "__dict__", {}):
                    delattr(res, "forward")
            except Exception:
                pass
        self._saved = []


class _IFNetBlock4Scale1Bypass:
    """Exact-output bypass for block4's identity bilinear resizes.

    Current ComfyUI IFNet runs its fifth block at ``scale=1`` but IFBlock.forward
    still executes three bilinear ``F.interpolate`` calls: on ``x``, on ``flow``
    (followed by ``div_(1)``), and on the ``lastconv`` output.  With scale exactly
    one these resizes preserve every element, so DEV2.15 bypasses only those
    identity kernels while keeping conv0, all eight ResConvs, ConvTranspose2d,
    PixelShuffle, slicing, scale multiplication, dtype and ordering unchanged.

    The patch is restricted to block4 and falls back to the original forward for
    any non-unit scale.  The original forward is restored on close().
    """

    def __init__(self, model: Any):
        self.model = model
        self.calls = 0
        self.bypassed_interpolates = 0
        self.fallbacks = 0
        blocks = getattr(model, "blocks", None)
        if not isinstance(blocks, (torch.nn.ModuleList, list, tuple)) or len(blocks) < 5:
            raise ValueError("model has no IFNet block4")
        self.block = blocks[4]
        for name in ("conv0", "convblock", "lastconv"):
            if not isinstance(getattr(self.block, name, None), torch.nn.Module):
                raise ValueError(f"block4 missing expected {name} module")
        self._had_instance_attr = "forward" in getattr(self.block, "__dict__", {})
        self._instance_attr = self.block.__dict__.get("forward") if self._had_instance_attr else None
        self._original_forward = self.block.forward

        def _forward(x, flow=None, scale=1, _block=self.block, _self=self):
            _self.calls += 1
            try:
                unit_scale = float(scale) == 1.0
            except Exception:
                unit_scale = False
            if not unit_scale:
                _self.fallbacks += 1
                return _self._original_forward(x, flow, scale)

            # Exact stock ordering with only scale=1 interpolation kernels removed.
            # x interpolate skipped.
            if flow is not None:
                # flow interpolate + div_(1) skipped; cat reads the same values.
                x = torch.cat((x, flow), 1)
                _self.bypassed_interpolates += 2
            else:
                _self.bypassed_interpolates += 1
            feat = _block.convblock(_block.conv0(x))
            tmp = _block.lastconv(feat)
            # output interpolate skipped. Keep stock slice/multiply semantics.
            _self.bypassed_interpolates += 1
            return tmp[:, :4] * scale, tmp[:, 4:5], tmp[:, 5:]

        self.block.forward = _forward

    def snapshot(self):
        return {
            "calls": int(self.calls),
            "bypassed_interpolates": int(self.bypassed_interpolates),
            "fallbacks": int(self.fallbacks),
            "patched": 1,
        }

    def close(self):
        try:
            if self._had_instance_attr:
                self.block.forward = self._instance_attr
            elif "forward" in getattr(self.block, "__dict__", {}):
                delattr(self.block, "forward")
        except Exception:
            pass



class _IFNetOuterCatPushdown:
    """FP16-only exact-output A/B: move IFNet outer cats after block downsampling.

    Current IFNet concatenates full-resolution component tensors and IFBlock then
    immediately bilinear-downsamples that concatenation for scales 16/8/4/2.
    Bilinear interpolation is channel-independent. On the production FP16 path,
    downsampling each component first is bit-identical in the regression preflight,
    while the subsequent cat copies only the much smaller downsampled tensors.

    Blocks 0..3 still execute through ``Module.__call__`` so forward hooks (most
    importantly the locked grid-cache invalidation hooks) retain their exact
    stage-boundary timing. Block4 stays on the normal locked scale=1 path.
    """

    def __init__(self, model: Any):
        self.model = model
        self.forwards = 0
        self.pushed_cats = 0
        self.component_interpolates = 0
        self.fallbacks = 0
        blocks = list(getattr(model, "blocks", []))
        scales = list(getattr(model, "scale_list", []))
        if len(blocks) != 5 or len(scales) != 5 or [float(x) for x in scales] != [16.0, 8.0, 4.0, 2.0, 1.0]:
            raise ValueError("outer cat pushdown requires current five-block IFNet scales [16,8,4,2,1]")
        if not callable(getattr(model, "warp", None)) or not callable(getattr(model, "_build_warp_grids", None)):
            raise ValueError("outer cat pushdown requires current IFNet warp/grid contract")
        try:
            dtype = model.get_dtype() if callable(getattr(model, "get_dtype", None)) else next(model.parameters()).dtype
        except Exception as exc:
            raise ValueError(f"could not determine IFNet dtype: {exc}")
        if dtype != torch.float16:
            raise ValueError(f"outer cat pushdown is FP16-only; model dtype is {dtype}")
        for bi, block in enumerate(blocks):
            for name in ("conv0", "convblock", "lastconv"):
                if not isinstance(getattr(block, name, None), torch.nn.Module):
                    raise ValueError(f"block{bi} missing expected {name} module")
        self.blocks = blocks
        self.scales = scales
        self._block_saved = []
        self._had_instance_attr = "forward" in getattr(model, "__dict__", {})
        self._instance_attr = model.__dict__.get("forward") if self._had_instance_attr else None
        self._original_forward = model.forward

        # Patch only blocks 0..3. They are still invoked through block(...), so
        # all existing module hooks fire normally. A tuple first argument is an
        # internal marker meaning "these are the pre-cat outer components".
        for block in blocks[:4]:
            had = "forward" in getattr(block, "__dict__", {})
            old_attr = block.__dict__.get("forward") if had else None
            original = block.forward
            self._block_saved.append((block, had, old_attr))

            def _block_forward(x, flow=None, scale=1, _block=block, _orig=original, _self=self):
                if not isinstance(x, tuple):
                    return _orig(x, flow, scale)
                inv = 1.0 / float(scale)
                resized = [torch.nn.functional.interpolate(part, scale_factor=inv, mode="bilinear") for part in x]
                _self.component_interpolates += len(x)
                x_small = torch.cat(tuple(resized), 1)
                _self.pushed_cats += 1
                if flow is not None:
                    flow_small = torch.nn.functional.interpolate(flow, scale_factor=inv, mode="bilinear").div_(scale)
                    x_small = torch.cat((x_small, flow_small), 1)
                feat = _block.convblock(_block.conv0(x_small))
                tmp = torch.nn.functional.interpolate(_block.lastconv(feat), scale_factor=scale, mode="bilinear")
                return tmp[:, :4] * scale, tmp[:, 4:5], tmp[:, 5:]

            block.forward = _block_forward

        def _forward(img0, img1, timestep=0.5, cache=None, _model=model, _self=self):
            if img0.dtype != torch.float16 or img1.dtype != torch.float16:
                _self.fallbacks += 1
                return _self._original_forward(img0, img1, timestep=timestep, cache=cache)
            _self.forwards += 1
            if not isinstance(timestep, torch.Tensor):
                timestep = torch.full((img0.shape[0], 1, img0.shape[2], img0.shape[3]), timestep, device=img0.device, dtype=img0.dtype)
            _model._build_warp_grids(img0.shape[2], img0.shape[3], img0.device)
            B = img0.shape[0]
            f0 = cache["img0"].expand(B, -1, -1, -1) if cache and "img0" in cache else _model.encode(img0)
            f1 = cache["img1"].expand(B, -1, -1, -1) if cache and "img1" in cache else _model.encode(img1)
            flow = mask = feat = None
            warped_img0, warped_img1 = img0, img1
            for i, block in enumerate(_self.blocks):
                scale = _self.scales[i]
                if i == 0:
                    flow, mask, feat = block((img0, img1, f0, f1, timestep), None, scale=scale)
                elif i < 4:
                    wf0 = _model.warp(f0, flow[:, :2])
                    wf1 = _model.warp(f1, flow[:, 2:4])
                    fd, mask, feat = block(
                        (warped_img0, warped_img1, wf0, wf1, timestep, mask, feat),
                        flow, scale=scale,
                    )
                    flow = flow.add_(fd)
                else:
                    fd, mask, feat = block(
                        torch.cat((warped_img0, warped_img1, _model.warp(f0, flow[:, :2]), _model.warp(f1, flow[:, 2:4]), timestep, mask, feat), 1),
                        flow, scale=scale,
                    )
                    flow = flow.add_(fd)
                warped_img0 = _model.warp(img0, flow[:, :2])
                warped_img1 = _model.warp(img1, flow[:, 2:4])
            return torch.lerp(warped_img1, warped_img0, torch.sigmoid(mask))

        model.forward = _forward

    def snapshot(self):
        return {
            "forwards": int(self.forwards),
            "pushed_cats": int(self.pushed_cats),
            "component_interpolates": int(self.component_interpolates),
            "fallbacks": int(self.fallbacks),
        }

    def close(self):
        try:
            if self._had_instance_attr:
                self.model.forward = self._instance_attr
            elif "forward" in getattr(self.model, "__dict__", {}):
                delattr(self.model, "forward")
        except Exception:
            pass
        for block, had, old_attr in reversed(self._block_saved):
            try:
                if had:
                    block.forward = old_attr
                elif "forward" in getattr(block, "__dict__", {}):
                    delattr(block, "forward")
            except Exception:
                pass
        self._block_saved = []


def iter_interpolated_sequence_chunks(
    interp_model: Any,
    images: torch.Tensor,
    multiplier: int,
    stereo_mode: str = "full_sbs",
    scene_cut: bool = True,
    scene_threshold: float = 0.22,
    cpu_output: str = "float16",
    timestep_batch: int = 1,
    pair_batch: int = 1,
    chunk_frames: int = 8,
    d2h_mode: str = "sync",
    h2d_mode: str = "sync",
    ifnet_profile: str = "off",
    feature_cache: str = "adjacent",
    warp_input_cache: str = "off",
    warp_grid_cache: str = "off",
    pinned_pool: Optional[_PinnedCpuBufferPool] = None,
    h2d_pinned_pool: Optional[_PinnedCpuBufferPool] = None,
    include_first: bool = True,
    final_tail: bool = False,
    stats: Optional[Dict[str, Any]] = None,
) -> _Iterator[torch.Tensor]:
    """Yield bounded CPU IMAGE chunks instead of materializing the expanded movie batch.

    DEV2.5 can pair-batch IFNet while retaining the async GPU->CPU transfer pipeline. In ``async_pinned``
    mode, predictions are copied on a separate ROCm/CUDA stream into pinned CPU
    buffers and synchronized only when enough ordered output exists to emit the
    next bounded chunk. ``sync`` retains the DEV2.2 transfer behavior for A/B tests.

    The stream represents a CFR timeline. On the final source batch ``final_tail``
    emits ``multiplier-1`` copies of the last real frame. This is intentional:
    N source frames at F fps occupy N/F seconds, therefore exact duration at M*F
    requires N*M output frames, not merely (N-1)*M+1 timestamp samples.
    """
    from comfy import model_management

    if images.ndim != 4 or images.shape[-1] != 3 or images.shape[0] < 1:
        raise ValueError(f"Expected non-empty BHWC RGB IMAGE, got {tuple(images.shape)}")
    if multiplier < 2 or multiplier > 16:
        raise ValueError("multiplier must be in 2..16")
    if stereo_mode not in ("split_eyes", "full_sbs"):
        raise ValueError("stereo_mode must be split_eyes or full_sbs")
    if cpu_output not in ("float16", "float32"):
        raise ValueError("cpu_output must be float16 or float32")
    if d2h_mode not in ("sync", "async_pinned"):
        raise ValueError("d2h_mode must be sync or async_pinned")
    if h2d_mode not in ("sync", "async_pinned"):
        raise ValueError("h2d_mode must be sync or async_pinned")
    if ifnet_profile not in ("off", "stages", "detail", "warp_detail", "block4_detail", "block4_res_inplace", "block4_scale1_bypass", "conv_family_detail", "outer_detail", "outer_cat_pushdown", "warp_postopt_detail"):
        raise ValueError("ifnet_profile must be off, stages, detail, warp_detail, block4_detail, block4_res_inplace, block4_scale1_bypass, conv_family_detail, or outer_detail, outer_cat_pushdown, or warp_postopt_detail")
    if feature_cache not in ("off", "adjacent"):
        raise ValueError("feature_cache must be off or adjacent")
    if warp_input_cache not in ("off", "fp32_reuse"):
        raise ValueError("warp_input_cache must be off or fp32_reuse")
    if warp_grid_cache not in ("off", "flow_reuse"):
        raise ValueError("warp_grid_cache must be off or flow_reuse")
    if warp_input_cache != "off" and warp_grid_cache != "off":
        raise ValueError("warp_input_cache and warp_grid_cache experimental modes cannot be combined")

    H, W = int(images.shape[1]), int(images.shape[2])
    if stereo_mode == "split_eyes" and W % 2:
        raise ValueError(f"split_eyes requires even SBS width, got {W}")

    chunk_frames = max(1, int(chunk_frames))
    out_dtype = torch.float16 if cpu_output == "float16" else torch.float32
    total_pairs = int(images.shape[0]) - 1
    n_mid = int(multiplier) - 1
    timesteps = [j / float(multiplier) for j in range(1, int(multiplier))]
    timestep_batch = max(1, min(int(timestep_batch), max(1, n_mid)))
    pair_batch = max(1, min(int(pair_batch), 4))

    if stats is None:
        stats = {}
    stats.setdefault("rife_seconds", 0.0)  # compatibility: inference + blocking D2H wait
    stats.setdefault("rife_inference_seconds", 0.0)  # IFNet + output postprocess GPU time
    stats.setdefault("ifnet_seconds", 0.0)
    stats.setdefault("rife_post_seconds", 0.0)
    stats.setdefault("rife_pack_seconds", 0.0)
    stats.setdefault("rife_forward_calls", 0)
    stats.setdefault("rife_pairs_submitted", 0)
    stats.setdefault("d2h_seconds", 0.0)   # GPU copy-engine activity; may overlap inference
    stats.setdefault("d2h_wait_seconds", 0.0)  # CPU wall time actually blocked on D2H
    stats.setdefault("scene_seconds", 0.0)
    # DEV2.4 splits the old ambiguous prepare bucket into CPU layout/staging,
    # actual H2D engine time, dependency wait, and post-upload dtype/padding work.
    stats.setdefault("prepare_seconds", 0.0)  # compatibility aggregate
    stats.setdefault("layout_seconds", 0.0)
    stats.setdefault("h2d_seconds", 0.0)
    stats.setdefault("h2d_wait_seconds", 0.0)
    stats.setdefault("gpu_prepare_seconds", 0.0)
    stats.setdefault("assemble_seconds", 0.0)
    stats.setdefault("scene_pairs", 0)
    stats.setdefault("chunks", 0)
    stats.setdefault("output_frames", 0)
    stats.setdefault("scene_cuts", 0)
    stats.setdefault("pinned_alloc_seconds", 0.0)
    stats.setdefault("pinned_allocations", 0)
    stats.setdefault("pinned_reuses", 0)
    stats.setdefault("pinned_peak_inflight", 0)
    stats.setdefault("h2d_pinned_alloc_seconds", 0.0)
    stats.setdefault("h2d_pinned_allocations", 0)
    stats.setdefault("h2d_pinned_reuses", 0)
    stats.setdefault("h2d_pinned_peak_inflight", 0)
    stats["d2h_mode_requested"] = str(d2h_mode)
    stats["d2h_mode_active"] = "sync"
    stats["h2d_mode_requested"] = str(h2d_mode)
    stats["h2d_mode_active"] = "sync"
    stats["pair_batch_requested"] = int(pair_batch)
    stats["pair_batch_active"] = int(pair_batch)
    stats["ifnet_profile_requested"] = str(ifnet_profile)
    stats["ifnet_profile_active"] = "off"
    stats["feature_cache_requested"] = str(feature_cache)
    stats["feature_cache_active"] = "off"
    stats.setdefault("feature_cache_extract_calls", 0)
    stats.setdefault("feature_cache_reuses", 0)
    stats.setdefault("feature_cache_pairs", 0)
    stats.setdefault("feature_cache_resets", 0)
    stats["warp_input_cache_requested"] = str(warp_input_cache)
    stats["warp_input_cache_active"] = "off"
    stats["warp_grid_cache_requested"] = str(warp_grid_cache)
    stats["warp_grid_cache_active"] = "off"
    stats.setdefault("warp_grid_builds", 0)
    stats.setdefault("warp_grid_reuses", 0)
    stats.setdefault("warp_grid_peak_entries", 0)
    stats.setdefault("warp_grid_forwards", 0)
    stats.setdefault("warp_grid_fallbacks", 0)
    stats.setdefault("warp_grid_invalidations", 0)
    stats["warp_grid_strategy"] = "off"
    stats.setdefault("warp_input_casts", 0)
    stats.setdefault("warp_input_reuses", 0)
    stats.setdefault("warp_input_peak_entries", 0)
    stats.setdefault("warp_input_forwards", 0)
    stats.setdefault("warp_input_fallbacks", 0)
    stats.setdefault("ifnet_stage_seconds", {})
    stats.setdefault("ifnet_stage_calls", {})
    stats.setdefault("ifnet_stage_unattributed_seconds", 0.0)
    stats.setdefault("ifnet_stage_profiled_seconds", 0.0)
    stats.setdefault("ifnet_stage_profiled_forwards", 0)
    stats["block4_res_inplace_requested"] = "inplace" if ifnet_profile == "block4_res_inplace" else "off"
    stats["block4_res_inplace_active"] = "off"
    stats.setdefault("block4_res_inplace_calls", 0)
    stats.setdefault("block4_res_inplace_patched", 0)
    stats["block4_scale1_requested"] = "locked"
    stats["block4_scale1_active"] = "off"
    stats.setdefault("block4_scale1_calls", 0)
    stats.setdefault("block4_scale1_bypassed_interpolates", 0)
    stats.setdefault("block4_scale1_fallbacks", 0)
    stats["outer_cat_pushdown_requested"] = "locked"
    stats["outer_cat_pushdown_active"] = "off"
    stats.setdefault("outer_cat_pushdown_forwards", 0)
    stats.setdefault("outer_cat_pushdown_pushed_cats", 0)
    stats.setdefault("outer_cat_pushdown_component_interpolates", 0)
    stats.setdefault("outer_cat_pushdown_fallbacks", 0)
    stats["warp_postopt_profile_active"] = "on" if ifnet_profile == "warp_postopt_detail" else "off"
    stats.setdefault("warp_postopt_seconds", {})
    stats.setdefault("warp_postopt_calls", {})
    stats.setdefault("warp_postopt_profiled_forwards", 0)

    # DEV2.4.1: resolve all scene decisions before any async H2D/D2H work is
    # launched. The detector itself is unchanged; only scheduling moved.
    if bool(scene_cut):
        scene_flags, scene_sec = _precompute_scene_flags(images, float(scene_threshold))
    else:
        scene_flags, scene_sec = [False] * total_pairs, 0.0
    stats["scene_seconds"] += scene_sec
    stats["scene_pairs"] += total_pairs

    buffered: _List[torch.Tensor] = []
    buffered_n = 0

    def add(piece: torch.Tensor):
        nonlocal buffered, buffered_n
        if piece is None or int(piece.shape[0]) == 0:
            return []
        if piece.device.type != "cpu" or piece.dtype != out_dtype:
            piece = piece.to(device="cpu", dtype=out_dtype)
        buffered.append(piece)
        buffered_n += int(piece.shape[0])
        ready = []
        while buffered_n >= chunk_frames:
            cat = torch.cat(buffered, dim=0)
            ready.append(cat[:chunk_frames].contiguous())
            rem = cat[chunk_frames:]
            buffered = [rem] if int(rem.shape[0]) else []
            buffered_n = int(rem.shape[0])
        return ready

    if include_first:
        for ready in add(images[0:1]):
            stats["chunks"] += 1
            stats["output_frames"] += int(ready.shape[0])
            yield ready

    inference_model = None
    device = None
    model_dtype = None
    align = 1
    copy_stream = None
    upload_stream = None
    async_active = False
    h2d_async_active = False
    if total_pairs > 0:
        device = interp_model.load_device
        model_dtype = interp_model.model_dtype()
        inference_model = interp_model.model
        align = int(getattr(inference_model, "pad_align", 1))
        activation_mem = inference_model.memory_used_forward(images.shape, model_dtype)
        model_management.load_models_gpu([interp_model], memory_required=activation_mem)
        if d2h_mode == "async_pinned" and getattr(device, "type", None) == "cuda":
            try:
                copy_stream = torch.cuda.Stream(device=device)
                if pinned_pool is None:
                    pinned_pool = _PinnedCpuBufferPool()
                async_active = True
                stats["d2h_mode_active"] = "async_pinned"
            except Exception as exc:
                stats["d2h_fallback_reason"] = str(exc)
                async_active = False
        if h2d_mode == "async_pinned" and getattr(device, "type", None) == "cuda":
            try:
                upload_stream = torch.cuda.Stream(device=device)
                if h2d_pinned_pool is None:
                    h2d_pinned_pool = _PinnedCpuBufferPool()
                h2d_async_active = True
                stats["h2d_mode_active"] = "async_pinned"
            except Exception as exc:
                stats["h2d_fallback_reason"] = str(exc)
                h2d_async_active = False

    # DEV2.10: exact-math A/B. Reuse the stock warp's fp32 source cast only
    # within one IFNet forward. The grid, sampler precision and cast-back remain
    # identical to current ComfyUI. Leave off by default until ROCm A/B proves a
    # stream-wall win.
    warp_input_optimizer = None
    if inference_model is not None and warp_input_cache == "fp32_reuse":
        try:
            warp_input_optimizer = _IFNetWarpInputCache(inference_model)
            stats["warp_input_cache_active"] = "fp32_reuse"
        except Exception as exc:
            stats["warp_input_cache_fallback_reason"] = str(exc)
            warp_input_optimizer = None

    # DEV2.11: exact-grid A/B. Reuse a sampling grid only while the backing
    # flow storage/version is unchanged.  This targets the repeated image->feature
    # stage-boundary grid construction without changing any sampler math.
    warp_grid_optimizer = None
    if inference_model is not None and warp_grid_cache == "flow_reuse":
        try:
            warp_grid_optimizer = _IFNetWarpGridCache(inference_model, profile_inner=(ifnet_profile == "warp_postopt_detail"))
            stats["warp_grid_cache_active"] = "flow_reuse"
            stats["warp_grid_strategy"] = str(warp_grid_optimizer.strategy)
        except Exception as exc:
            stats["warp_grid_cache_fallback_reason"] = str(exc)
            warp_grid_optimizer = None

    # DEV2.14 exact-output A/B: block4 ResConv residual addcmul writes into
    # its dead input tensor instead of allocating a separate residual result.
    block4_res_optimizer = None
    if inference_model is not None and ifnet_profile == "block4_res_inplace":
        try:
            block4_res_optimizer = _IFNetBlock4ResConvInplace(inference_model)
            stats["block4_res_inplace_active"] = "inplace"
        except Exception as exc:
            stats["block4_res_inplace_fallback_reason"] = str(exc)
            block4_res_optimizer = None

    # DEV2.16 production lock: DEV2.15 measured a clean exact-output win, so
    # block4's three scale=1 identity interpolates are bypassed on every IFNet run.
    # Unsupported/future models simply fall back without changing their forward.
    block4_scale1_optimizer = None
    if inference_model is not None:
        try:
            scales = list(getattr(inference_model, "scale_list"))
            compatible = callable(getattr(inference_model, "warp", None)) and len(scales) >= 5 and float(scales[4]) == 1.0
        except Exception:
            compatible = False
        if compatible:
            try:
                block4_scale1_optimizer = _IFNetBlock4Scale1Bypass(inference_model)
                stats["block4_scale1_active"] = "bypass"
            except Exception as exc:
                stats["block4_scale1_fallback_reason"] = str(exc)
                block4_scale1_optimizer = None
        else:
            stats["block4_scale1_fallback_reason"] = "model does not expose current IFNet scale_list/warp contract"

    # DEV2.19 production lock: DEV2.18 measured a clean exact-output win. Avoid four full-resolution outer cats by
    # bilinear-downsampling each component first for IFNet stages 0..3, then
    # concatenate at the block's reduced spatial resolution. Block4 remains on
    # the locked scale=1 path.
    outer_cat_optimizer = None
    if inference_model is not None:
        try:
            outer_cat_optimizer = _IFNetOuterCatPushdown(inference_model)
            stats["outer_cat_pushdown_active"] = "fp16"
        except Exception as exc:
            stats["outer_cat_pushdown_fallback_reason"] = str(exc)
            outer_cat_optimizer = None

    # DEV2.6/DEV2.8/DEV2.9 instrumentation only. "stages" times encode/blocks;
    # "detail" times complete image/feature warps; "warp_detail" preserves the
    # same warp math while timing its internal normalization/grid/casts/sample.
    stage_profiler = None
    if inference_model is not None and ifnet_profile in ("stages", "detail", "warp_detail", "block4_detail", "conv_family_detail", "outer_detail"):
        try:
            candidate = _IFNetStageProfiler(
                inference_model,
                profile_warp=(ifnet_profile == "detail"),
                profile_warp_inner=(ifnet_profile == "warp_detail"),
                profile_block4=(ifnet_profile == "block4_detail"),
                profile_conv_family=(ifnet_profile == "conv_family_detail"),
                profile_outer=(ifnet_profile == "outer_detail"),
            )
            if candidate.usable:
                stage_profiler = candidate
                stats["ifnet_profile_active"] = str(ifnet_profile)
                stats["ifnet_stage_names"] = list(candidate.stage_names)
            else:
                candidate.close()
                stats["ifnet_profile_fallback_reason"] = "No profiled IFNet child stages discovered"
        except Exception as exc:
            stats["ifnet_profile_fallback_reason"] = str(exc)
            stage_profiler = None

    # DEV2.7: optional adjacent-pair feature reuse. Current ComfyUI IFNet exposes
    # extract_features() and accepts cache={img0,img1}; img1 from temporal pair N
    # is mathematically identical to img0 for pair N+1.  Keep this A/B path
    # conservative: pair_batch=1 only, and reset at hard cuts/meta-batch boundaries.
    feature_cache_active = False
    if inference_model is not None and feature_cache == "adjacent":
        if pair_batch != 1:
            stats["feature_cache_fallback_reason"] = "adjacent cache requires pair_batch=1"
        elif not callable(getattr(inference_model, "extract_features", None)):
            stats["feature_cache_fallback_reason"] = "model has no extract_features()"
        else:
            feature_cache_active = True
            stats["feature_cache_active"] = "adjacent"

    def _event_elapsed_seconds(a, b) -> float:
        try:
            return float(a.elapsed_time(b)) / 1000.0
        except Exception:
            return 0.0

    def _source_cpu_shape(frame: torch.Tensor, streams: int, crop_w: int):
        return (int(streams), 3, H, int(crop_w))

    def _stage_source_cpu(frame: torch.Tensor, streams: int, crop_w: int):
        """Copy BHWC source pixels into a contiguous pinned BCHW staging buffer."""
        nonlocal h2d_pinned_pool
        if h2d_pinned_pool is None:
            h2d_pinned_pool = _PinnedCpuBufferPool()
        before = h2d_pinned_pool.snapshot()
        stage = h2d_pinned_pool.acquire(_source_cpu_shape(frame, streams, crop_w), frame.dtype)
        after = h2d_pinned_pool.snapshot()
        stats["h2d_pinned_alloc_seconds"] += max(0.0, after["alloc_seconds"] - before["alloc_seconds"])
        stats["h2d_pinned_allocations"] += max(0, after["allocations"] - before["allocations"])
        stats["h2d_pinned_reuses"] += max(0, after["reuses"] - before["reuses"])
        stats["h2d_pinned_peak_inflight"] = max(stats["h2d_pinned_peak_inflight"], after["peak_inflight"])
        tl = _time.perf_counter()
        if streams == 1:
            stage.copy_(frame[..., :3].movedim(-1, 1))
        else:
            half = int(crop_w)
            stage[0:1].copy_(frame[:, :, :half, :3].movedim(-1, 1))
            stage[1:2].copy_(frame[:, :, half:half * 2, :3].movedim(-1, 1))
        stats["layout_seconds"] += _time.perf_counter() - tl
        return stage

    def _submit_source_upload(frame: torch.Tensor, streams: int, crop_w: int):
        """Queue one source frame on the dedicated H2D stream and return a descriptor."""
        nonlocal h2d_async_active
        if not h2d_async_active:
            return None
        try:
            stage = _stage_source_cpu(frame, streams, crop_w)
            h2d_start = torch.cuda.Event(enable_timing=True)
            h2d_end = torch.cuda.Event(enable_timing=True)
            gpu_start = torch.cuda.Event(enable_timing=True)
            ready = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(upload_stream):
                h2d_start.record(upload_stream)
                gpu = stage.to(device=device, non_blocking=True)
                h2d_end.record(upload_stream)
                gpu_start.record(upload_stream)
                if gpu.dtype != model_dtype:
                    gpu = gpu.to(dtype=model_dtype)
                if align > 1:
                    from comfy.ldm.common_dit import pad_to_patch_size
                    gpu = pad_to_patch_size(gpu, (align, align), padding_mode="reflect")
                ready.record(upload_stream)
            return {
                "gpu": gpu,
                "stage": stage,
                "h2d_start": h2d_start,
                "h2d_end": h2d_end,
                "gpu_start": gpu_start,
                "ready": ready,
                "resolved": False,
            }
        except Exception as exc:
            try:
                if 'stage' in locals() and h2d_pinned_pool is not None:
                    h2d_pinned_pool.release(stage)
            except Exception:
                pass
            h2d_async_active = False
            stats["h2d_mode_active"] = "sync"
            stats["h2d_fallback_reason"] = str(exc)
            return None

    def _resolve_source_upload(desc):
        """Wait only for the unhidden tail of a prefetched upload, then recycle staging."""
        if desc is None:
            return None
        if desc.get("resolved", False):
            return desc["gpu"]
        tw = _time.perf_counter()
        desc["ready"].synchronize()
        wait_sec = _time.perf_counter() - tw
        stats["h2d_wait_seconds"] += wait_sec
        stats["h2d_seconds"] += _event_elapsed_seconds(desc["h2d_start"], desc["h2d_end"])
        stats["gpu_prepare_seconds"] += _event_elapsed_seconds(desc["gpu_start"], desc["ready"])
        if h2d_pinned_pool is not None:
            h2d_pinned_pool.release(desc.get("stage"))
        desc["stage"] = None
        desc["resolved"] = True
        return desc["gpu"]

    def _prepare_source_sync(frame: torch.Tensor, streams: int, crop_w: int):
        """DEV2.4 profiled version of the original blocking source preparation."""
        if streams == 1:
            views = [frame[..., :3].movedim(-1, 1)]
        else:
            half = int(crop_w)
            views = [
                frame[:, :, :half, :3].movedim(-1, 1),
                frame[:, :, half:half * 2, :3].movedim(-1, 1),
            ]
        tl = _time.perf_counter()
        # movedim is a view; contiguous/layout work occurs as part of the transfer.
        stats["layout_seconds"] += _time.perf_counter() - tl
        outs = []
        for view in views:
            if getattr(device, "type", None) == "cuda":
                stream = torch.cuda.current_stream(device=device)
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record(stream)
                tw = _time.perf_counter()
                out = view.to(device=device, dtype=model_dtype)
                wait_sec = _time.perf_counter() - tw
                e1.record(stream)
                e1.synchronize()
                stats["h2d_seconds"] += _event_elapsed_seconds(e0, e1)
                stats["h2d_wait_seconds"] += wait_sec
            else:
                tw = _time.perf_counter()
                out = view.to(device=device, dtype=model_dtype)
                sec = _time.perf_counter() - tw
                stats["h2d_seconds"] += sec
                stats["h2d_wait_seconds"] += sec
            if align > 1:
                from comfy.ldm.common_dit import pad_to_patch_size
                if getattr(device, "type", None) == "cuda":
                    stream = torch.cuda.current_stream(device=device)
                    g0 = torch.cuda.Event(enable_timing=True)
                    g1 = torch.cuda.Event(enable_timing=True)
                    g0.record(stream)
                    out = pad_to_patch_size(out, (align, align), padding_mode="reflect")
                    g1.record(stream)
                    g1.synchronize()
                    stats["gpu_prepare_seconds"] += _event_elapsed_seconds(g0, g1)
                else:
                    tg = _time.perf_counter()
                    out = pad_to_patch_size(out, (align, align), padding_mode="reflect")
                    stats["gpu_prepare_seconds"] += _time.perf_counter() - tg
            outs.append(out)
        return torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]

    def _account_rife_tracker(tracker):
        if tracker is None or tracker.get("accounted", False):
            return
        pack_sec = _event_elapsed_seconds(tracker.get("pack_start"), tracker.get("pack_end")) if tracker.get("pack_start") is not None else float(tracker.get("pack_seconds", 0.0))
        ifnet_sec = _event_elapsed_seconds(tracker.get("ifnet_start"), tracker.get("ifnet_end")) if tracker.get("ifnet_start") is not None else float(tracker.get("ifnet_seconds", 0.0))
        post_sec = _event_elapsed_seconds(tracker.get("ifnet_end"), tracker.get("post_end")) if tracker.get("ifnet_end") is not None else float(tracker.get("post_seconds", 0.0))
        stats["rife_pack_seconds"] += pack_sec
        stats["ifnet_seconds"] += ifnet_sec
        stats["rife_post_seconds"] += post_sec
        stats["rife_inference_seconds"] += ifnet_sec + post_sec
        stats["rife_seconds"] += ifnet_sec + post_sec
        stats["rife_forward_calls"] += 1
        stats["rife_pairs_submitted"] += int(tracker.get("pairs", 1))

        stage_trace = tracker.get("ifnet_stage_trace")
        if stage_trace is not None:
            stage_total = 0.0
            mode = str(stage_trace.get("mode", "cpu"))
            for stage, spans in stage_trace.get("events", {}).items():
                sec = 0.0
                for start, end in spans:
                    if mode == "cuda":
                        sec += _event_elapsed_seconds(start, end)
                    else:
                        sec += max(0.0, float(end) - float(start))
                stats["ifnet_stage_seconds"][stage] = float(stats["ifnet_stage_seconds"].get(stage, 0.0)) + sec
                stats["ifnet_stage_calls"][stage] = int(stats["ifnet_stage_calls"].get(stage, 0)) + len(spans)
                stage_name = str(stage)
                if stage_name == "encode" or (stage_name.startswith("block") and stage_name[5:].isdigit()) or stage_name.startswith("warp_") or stage_name.startswith("outer_"):
                    stage_total += sec
            stats["ifnet_stage_profiled_seconds"] += ifnet_sec
            stats["ifnet_stage_profiled_forwards"] += 1
            stats["ifnet_stage_unattributed_seconds"] += max(0.0, ifnet_sec - stage_total)


        warp_trace = tracker.get("warp_postopt_trace")
        if warp_trace is not None:
            mode = str(warp_trace.get("mode", "cpu"))
            for stage, spans in warp_trace.get("events", {}).items():
                sec = 0.0
                for start, end in spans:
                    if mode == "cuda":
                        sec += _event_elapsed_seconds(start, end)
                    else:
                        sec += max(0.0, float(end) - float(start))
                stats["warp_postopt_seconds"][stage] = float(stats["warp_postopt_seconds"].get(stage, 0.0)) + sec
                stats["warp_postopt_calls"][stage] = int(stats["warp_postopt_calls"].get(stage, 0)) + len(spans)
            stats["warp_postopt_profiled_forwards"] += 1

        tracker["accounted"] = True

    feature_next = None

    def _submit_prediction_group(tasks):
        """Run one IFNet forward for one or more temporal-pair tasks.

        DEV2.5 pair batching changes only IFNet's batch dimension. Each temporal
        pair is split back into its own existing D2H descriptor immediately after
        the forward, preserving ordering, bounded chunks, and pool lifetimes.

        DEV2.7 can precompute/cache IFNet head features for pair_batch=1. The
        extraction is deliberately inside the existing IFNet timing envelope, so
        benchmark comparisons include the real cost rather than hiding it.
        """
        nonlocal async_active, pinned_pool, feature_next
        if not tasks:
            return []
        is_gpu = getattr(device, "type", None) == "cuda"
        lengths = [int(task["a"].shape[0]) for task in tasks]

        if is_gpu:
            compute_stream = torch.cuda.current_stream(device=device)
            pack_start = pack_end = None
            if len(tasks) > 1:
                pack_start = torch.cuda.Event(enable_timing=True)
                pack_end = torch.cuda.Event(enable_timing=True)
                pack_start.record(compute_stream)
                a_all = torch.cat([task["a"] for task in tasks], dim=0)
                b_all = torch.cat([task["b"] for task in tasks], dim=0)
                t_all = torch.cat([task["t"] for task in tasks], dim=0)
                pack_end.record(compute_stream)
            else:
                a_all, b_all, t_all = tasks[0]["a"], tasks[0]["b"], tasks[0]["t"]

            ifnet_start = torch.cuda.Event(enable_timing=True)
            ifnet_end = torch.cuda.Event(enable_timing=True)
            post_end = torch.cuda.Event(enable_timing=True)
            if "ifnet_model_class" not in stats:
                stats["ifnet_model_class"] = type(inference_model).__name__
                first_param = next(inference_model.parameters(), None) if isinstance(inference_model, torch.nn.Module) else None
                stats["ifnet_model_dtype"] = str(first_param.dtype if first_param is not None else a_all.dtype)
                stats["ifnet_input_dtype"] = str(a_all.dtype)
                stats["ifnet_input_contiguous"] = bool(a_all.is_contiguous())
                stats["ifnet_input_channels_last"] = bool(a_all.is_contiguous(memory_format=torch.channels_last))
            ifnet_start.record(compute_stream)
            stage_trace = None
            if warp_input_optimizer is not None:
                warp_input_optimizer.begin_forward()
            if warp_grid_optimizer is not None:
                warp_grid_optimizer.begin_forward("cuda")
            if stage_profiler is not None:
                stage_profiler.begin("cuda")
            try:
                cache_arg = None
                if feature_cache_active and len(tasks) == 1:
                    task = tasks[0]
                    ctx = task.get("ctx")
                    base_cache = ctx.get("feature_cache_base") if ctx is not None else None
                    if base_cache is None:
                        base_a = task["base_a"]
                        base_b = task["base_b"]
                        if feature_next is None:
                            f0 = inference_model.extract_features(base_a)
                            stats["feature_cache_extract_calls"] += 1
                        else:
                            f0 = feature_next
                            stats["feature_cache_reuses"] += 1
                        f1 = inference_model.extract_features(base_b)
                        stats["feature_cache_extract_calls"] += 1
                        stats["feature_cache_pairs"] += 1
                        feature_next = f1
                        base_cache = {"img0": f0, "img1": f1}
                        if ctx is not None:
                            ctx["feature_cache_base"] = base_cache
                    count = int(task.get("count", 1))
                    if count > 1:
                        cache_arg = {
                            "img0": base_cache["img0"].repeat_interleave(count, dim=0),
                            "img1": base_cache["img1"].repeat_interleave(count, dim=0),
                        }
                    else:
                        cache_arg = base_cache
                try:
                    raw = inference_model(a_all, b_all, timestep=t_all, cache=cache_arg)
                except TypeError:
                    raw = inference_model(a_all, b_all, timestep=t_all)
            finally:
                if stage_profiler is not None:
                    stage_trace = stage_profiler.end()
                if warp_input_optimizer is not None:
                    warp_input_optimizer.end_forward()
                    snap = warp_input_optimizer.snapshot()
                    stats["warp_input_casts"] = int(snap["casts"])
                    stats["warp_input_reuses"] = int(snap["reuses"])
                    stats["warp_input_peak_entries"] = int(snap["peak_entries"])
                    stats["warp_input_forwards"] = int(snap["forwards"])
                    stats["warp_input_fallbacks"] = int(snap["fallbacks"])
                warp_post_trace = None
                if warp_grid_optimizer is not None:
                    warp_post_trace = warp_grid_optimizer.end_forward()
                    gsnap = warp_grid_optimizer.snapshot()
                    stats["warp_grid_builds"] = int(gsnap["builds"])
                    stats["warp_grid_reuses"] = int(gsnap["reuses"])
                    stats["warp_grid_peak_entries"] = int(gsnap["peak_entries"])
                    stats["warp_grid_forwards"] = int(gsnap["forwards"])
                    stats["warp_grid_fallbacks"] = int(gsnap["fallbacks"])
                    stats["warp_grid_invalidations"] = int(gsnap.get("invalidations", 0))
                    stats["warp_grid_strategy"] = str(gsnap.get("strategy", stats.get("warp_grid_strategy", "off")))
            ifnet_end.record(compute_stream)
            raw = raw.clamp_(0.0, 1.0)
            post_end.record(compute_stream)
            tracker = {
                "accounted": False, "pack_start": pack_start, "pack_end": pack_end,
                "ifnet_start": ifnet_start, "ifnet_end": ifnet_end, "post_end": post_end,
                "ifnet_stage_trace": stage_trace, "warp_postopt_trace": warp_post_trace, "pairs": len(tasks),
            }

            descs = []
            offset = 0
            if async_active:
                try:
                    for task, n in zip(tasks, lengths):
                        crop_w = int(task["crop_w"])
                        pred = raw[offset:offset+n, :, :H, :crop_w].detach()
                        offset += n
                        pool_before = pinned_pool.snapshot() if pinned_pool is not None else None
                        cpu_pred = pinned_pool.acquire(tuple(pred.shape), out_dtype)
                        pool_after = pinned_pool.snapshot()
                        if pool_before is not None:
                            stats["pinned_alloc_seconds"] += max(0.0, pool_after["alloc_seconds"] - pool_before["alloc_seconds"])
                            stats["pinned_allocations"] += max(0, pool_after["allocations"] - pool_before["allocations"])
                            stats["pinned_reuses"] += max(0, pool_after["reuses"] - pool_before["reuses"])
                            stats["pinned_peak_inflight"] = max(stats["pinned_peak_inflight"], pool_after["peak_inflight"])
                        d2h_start = torch.cuda.Event(enable_timing=True)
                        d2h_end = torch.cuda.Event(enable_timing=True)
                        copy_stream.wait_event(post_end)
                        with torch.cuda.stream(copy_stream):
                            d2h_start.record(copy_stream)
                            cpu_pred.copy_(pred, non_blocking=True)
                            d2h_end.record(copy_stream)
                        descs.append({
                            "cpu": cpu_pred, "gpu": pred, "rife_tracker": tracker,
                            "d2h_start": d2h_start, "d2h_end": d2h_end,
                            "streams": int(task["streams"]), "count": int(task["count"]),
                            "async": True, "pooled": True,
                        })
                    return descs
                except Exception as exc:
                    # A copy may already be queued when a later descriptor setup
                    # fails. Do not recycle that pinned destination until its DMA
                    # event has completed.
                    for desc in descs:
                        try:
                            desc.get("d2h_end").synchronize()
                        except Exception:
                            pass
                        try:
                            if pinned_pool is not None:
                                pinned_pool.release(desc.get("cpu"))
                        except Exception:
                            pass
                    async_active = False
                    stats["d2h_mode_active"] = "sync"
                    stats["d2h_fallback_reason"] = str(exc)
                    descs = []
                    offset = 0

            # Synchronous fallback / A-B mode. One model forward is still shared.
            post_end.synchronize()
            _account_rife_tracker(tracker)
            for task, n in zip(tasks, lengths):
                crop_w = int(task["crop_w"])
                pred = raw[offset:offset+n, :, :H, :crop_w].detach()
                offset += n
                d2h_start = torch.cuda.Event(enable_timing=True)
                d2h_start.record(compute_stream)
                wait0 = _time.perf_counter()
                cpu_pred = pred.to(device="cpu", dtype=out_dtype)
                wait_sec = _time.perf_counter() - wait0
                d2h_end = torch.cuda.Event(enable_timing=True)
                d2h_end.record(compute_stream)
                d2h_end.synchronize()
                stats["d2h_seconds"] += _event_elapsed_seconds(d2h_start, d2h_end)
                stats["d2h_wait_seconds"] += wait_sec
                descs.append({"cpu": cpu_pred, "streams": int(task["streams"]), "count": int(task["count"]), "async": False})
            return descs

        # CPU/CI path: preserve exact math while exercising pair-batch grouping.
        tp = _time.perf_counter()
        if len(tasks) > 1:
            a_all = torch.cat([task["a"] for task in tasks], dim=0)
            b_all = torch.cat([task["b"] for task in tasks], dim=0)
            t_all = torch.cat([task["t"] for task in tasks], dim=0)
        else:
            a_all, b_all, t_all = tasks[0]["a"], tasks[0]["b"], tasks[0]["t"]
        pack_sec = _time.perf_counter() - tp
        if "ifnet_model_class" not in stats:
            stats["ifnet_model_class"] = type(inference_model).__name__
            stats["ifnet_model_dtype"] = str(a_all.dtype)
            stats["ifnet_input_dtype"] = str(a_all.dtype)
            stats["ifnet_input_contiguous"] = bool(a_all.is_contiguous())
            stats["ifnet_input_channels_last"] = bool(a_all.is_contiguous(memory_format=torch.channels_last))
        t0 = _time.perf_counter()
        stage_trace = None
        if warp_input_optimizer is not None:
            warp_input_optimizer.begin_forward()
        if warp_grid_optimizer is not None:
            warp_grid_optimizer.begin_forward("cpu")
        if stage_profiler is not None:
            stage_profiler.begin("cpu")
        try:
            cache_arg = None
            if feature_cache_active and len(tasks) == 1:
                task = tasks[0]
                ctx = task.get("ctx")
                base_cache = ctx.get("feature_cache_base") if ctx is not None else None
                if base_cache is None:
                    base_a = task["base_a"]
                    base_b = task["base_b"]
                    if feature_next is None:
                        f0 = inference_model.extract_features(base_a)
                        stats["feature_cache_extract_calls"] += 1
                    else:
                        f0 = feature_next
                        stats["feature_cache_reuses"] += 1
                    f1 = inference_model.extract_features(base_b)
                    stats["feature_cache_extract_calls"] += 1
                    stats["feature_cache_pairs"] += 1
                    feature_next = f1
                    base_cache = {"img0": f0, "img1": f1}
                    if ctx is not None:
                        ctx["feature_cache_base"] = base_cache
                count = int(task.get("count", 1))
                if count > 1:
                    cache_arg = {
                        "img0": base_cache["img0"].repeat_interleave(count, dim=0),
                        "img1": base_cache["img1"].repeat_interleave(count, dim=0),
                    }
                else:
                    cache_arg = base_cache
            try:
                raw = inference_model(a_all, b_all, timestep=t_all, cache=cache_arg)
            except TypeError:
                raw = inference_model(a_all, b_all, timestep=t_all)
        finally:
            if stage_profiler is not None:
                stage_trace = stage_profiler.end()
            if warp_input_optimizer is not None:
                warp_input_optimizer.end_forward()
                snap = warp_input_optimizer.snapshot()
                stats["warp_input_casts"] = int(snap["casts"])
                stats["warp_input_reuses"] = int(snap["reuses"])
                stats["warp_input_peak_entries"] = int(snap["peak_entries"])
                stats["warp_input_forwards"] = int(snap["forwards"])
                stats["warp_input_fallbacks"] = int(snap["fallbacks"])
            warp_post_trace = None
            if warp_grid_optimizer is not None:
                warp_post_trace = warp_grid_optimizer.end_forward()
                gsnap = warp_grid_optimizer.snapshot()
                stats["warp_grid_builds"] = int(gsnap["builds"])
                stats["warp_grid_reuses"] = int(gsnap["reuses"])
                stats["warp_grid_peak_entries"] = int(gsnap["peak_entries"])
                stats["warp_grid_forwards"] = int(gsnap["forwards"])
                stats["warp_grid_fallbacks"] = int(gsnap["fallbacks"])
                stats["warp_grid_invalidations"] = int(gsnap.get("invalidations", 0))
                stats["warp_grid_strategy"] = str(gsnap.get("strategy", stats.get("warp_grid_strategy", "off")))
        ifnet_sec = _time.perf_counter() - t0
        tpost = _time.perf_counter()
        raw = raw.clamp_(0.0, 1.0)
        post_sec = _time.perf_counter() - tpost
        tracker = {
            "accounted": False, "pack_seconds": pack_sec, "ifnet_seconds": ifnet_sec,
            "post_seconds": post_sec, "ifnet_stage_trace": stage_trace, "warp_postopt_trace": warp_post_trace, "pairs": len(tasks),
        }
        _account_rife_tracker(tracker)
        descs = []
        offset = 0
        for task, n in zip(tasks, lengths):
            pred = raw[offset:offset+n, :, :H, :int(task["crop_w"])].detach()
            offset += n
            t1 = _time.perf_counter()
            cpu_pred = pred.to(device="cpu", dtype=out_dtype)
            d2h_sec = _time.perf_counter() - t1
            stats["d2h_seconds"] += d2h_sec
            stats["d2h_wait_seconds"] += d2h_sec
            descs.append({"cpu": cpu_pred, "streams": int(task["streams"]), "count": int(task["count"]), "async": False})
        return descs

    def _submit_prediction(a, b, t, streams, count, crop_w):
        return _submit_prediction_group([{
            "a": a, "b": b, "t": t, "streams": streams, "count": count, "crop_w": crop_w
        }])[0]

    def _resolve_prediction(desc):
        if desc.get("async", False):
            wait0 = _time.perf_counter()
            desc["d2h_end"].synchronize()
            wait_sec = _time.perf_counter() - wait0
            _account_rife_tracker(desc.get("rife_tracker"))
            d2h_sec = _event_elapsed_seconds(desc["d2h_start"], desc["d2h_end"])
            stats["d2h_seconds"] += d2h_sec
            stats["d2h_wait_seconds"] += wait_sec
            desc.pop("gpu", None)
        return desc["cpu"]

    def _resolve_pair(record):
        ta = _time.perf_counter()
        pooled_to_release = []
        if record.get("cut", False):
            mids = record["mids"]
        else:
            mids_by_t = []
            for desc in record["preds"]:
                pred = _resolve_prediction(desc)
                if desc.get("pooled", False):
                    pooled_to_release.append(pred)
                count = int(desc["count"])
                streams = int(desc["streams"])
                if streams == 1:
                    chunk = pred.movedim(1, -1)
                    mids_by_t.extend(chunk[k : k + 1] for k in range(count))
                else:
                    left = pred[:count].movedim(1, -1)
                    right = pred[count : count * 2].movedim(1, -1)
                    for k in range(count):
                        mids_by_t.append(torch.cat((left[k : k + 1], right[k : k + 1]), dim=2))
            # torch.cat copies the prediction data into fresh CPU storage.  Once
            # pair_piece below is built, the pinned D2H destinations are safe to
            # return to the pool for later temporal pairs/meta-batches.
            mids = torch.cat(mids_by_t, dim=0) if mids_by_t else record["src1"][:0].to(device="cpu", dtype=out_dtype)
        pair_piece = torch.cat((mids, record["src1"].to(device="cpu", dtype=out_dtype)), dim=0)
        for pooled in pooled_to_release:
            pinned_pool.release(pooled)
        stats["assemble_seconds"] += _time.perf_counter() - ta
        return pair_piece

    pending_pairs = []
    pending_frames = 0

    def _drain_until_chunk_possible(force=False):
        nonlocal pending_frames
        ready_all = []
        while pending_pairs and (force or buffered_n + pending_frames >= chunk_frames):
            rec = pending_pairs.pop(0)
            pending_frames -= int(rec["frames"])
            piece = _resolve_pair(rec)
            ready_all.extend(add(piece))
            if not force and ready_all:
                break
        return ready_all

    prepared_next = None
    source_uploads: Dict[int, Any] = {}

    def _streams_crop():
        if stereo_mode == "split_eyes":
            return 2, W // 2
        return 1, W

    def _ensure_upload(idx: int):
        if not h2d_async_active or idx < 0 or idx >= int(images.shape[0]):
            return None
        if idx not in source_uploads:
            streams0, crop0 = _streams_crop()
            source_uploads[idx] = _submit_source_upload(images[idx:idx + 1], streams0, crop0)
        return source_uploads.get(idx)

    def _take_uploaded(idx: int):
        desc = source_uploads.pop(idx, None)
        return _resolve_source_upload(desc) if desc is not None else None

    pair_contexts = []

    def _flush_pair_contexts():
        nonlocal pair_contexts
        if not pair_contexts:
            return []
        contexts = pair_contexts
        pair_contexts = []
        preds_per_pair = [[] for _ in contexts]
        j = 0
        while j < n_mid:
            count = min(timestep_batch, n_mid - j)
            ts = timesteps[j:j + count]
            tasks = []
            for ctx in contexts:
                p0, p1 = ctx["p0"], ctx["p1"]
                pH, pW = int(p0.shape[2]), int(p0.shape[3])
                streams = int(ctx["streams"])
                a = p0.repeat_interleave(count, dim=0)
                b = p1.repeat_interleave(count, dim=0)
                t = torch.tensor(ts, device=device, dtype=model_dtype).repeat(streams)
                t = t.view(-1, 1, 1, 1).expand(-1, 1, pH, pW)
                tasks.append({
                    "a": a, "b": b, "t": t, "streams": streams,
                    "count": count, "crop_w": int(ctx["crop_w"]),
                    "base_a": p0, "base_b": p1, "ctx": ctx,
                })
            descs = _submit_prediction_group(tasks)
            for idx, desc in enumerate(descs):
                preds_per_pair[idx].append(desc)
            j += count
        return [
            {"cut": False, "preds": preds_per_pair[idx], "src1": ctx["src1"], "frames": n_mid + 1}
            for idx, ctx in enumerate(contexts)
        ]

    def _queue_records(records):
        nonlocal pending_frames
        for rec in records:
            pending_pairs.append(rec)
            pending_frames += int(rec["frames"])
        return _drain_until_chunk_possible(force=False)

    try:
        with torch.inference_mode():
            for pair_idx in range(total_pairs):
                src0 = images[pair_idx : pair_idx + 1]
                src1 = images[pair_idx + 1 : pair_idx + 2]

                is_cut = bool(scene_flags[pair_idx])

                if is_cut:
                    # Preserve output order: submit any preceding non-cut batch
                    # before the hard-cut hold record.
                    for ready in _queue_records(_flush_pair_contexts()):
                        stats["chunks"] += 1
                        stats["output_frames"] += int(ready.shape[0])
                        yield ready
                    stats["scene_cuts"] += 1
                    if feature_cache_active and feature_next is not None:
                        feature_next = None
                        stats["feature_cache_resets"] += 1
                    cut_record = {
                        "cut": True,
                        "mids": _hold_intermediates(src0, n_mid, out_dtype),
                        "src1": src1,
                        "frames": n_mid + 1,
                    }
                    prepared_next = None
                    for ready in _queue_records([cut_record]):
                        stats["chunks"] += 1
                        stats["output_frames"] += int(ready.shape[0])
                        yield ready
                    continue

                streams, crop_w = _streams_crop()
                # Queue the *future* source before current RIFE. On pair N this
                # is frame N+2, so its H2D can overlap pair-N inference/D2H.
                if h2d_async_active:
                    if prepared_next is None:
                        _ensure_upload(pair_idx)
                    _ensure_upload(pair_idx + 1)
                    # With pair batching, look one batch window ahead so source
                    # uploads can remain hidden while the grouped IFNet runs.
                    _ensure_upload(pair_idx + max(2, pair_batch))

                    if prepared_next is None:
                        p0 = _take_uploaded(pair_idx)
                    else:
                        p0 = prepared_next
                    p1 = _take_uploaded(pair_idx + 1)
                    if p0 is None:
                        p0 = _prepare_source_sync(src0, streams, crop_w)
                    if p1 is None:
                        p1 = _prepare_source_sync(src1, streams, crop_w)
                else:
                    p0 = prepared_next if prepared_next is not None else _prepare_source_sync(src0, streams, crop_w)
                    p1 = _prepare_source_sync(src1, streams, crop_w)
                prepared_next = p1
                stats["prepare_seconds"] = (
                    stats.get("layout_seconds", 0.0)
                    + stats.get("h2d_wait_seconds", 0.0)
                    + stats.get("gpu_prepare_seconds", 0.0)
                )

                pair_contexts.append({
                    "p0": p0, "p1": p1, "streams": streams, "crop_w": crop_w, "src1": src1
                })
                if len(pair_contexts) >= pair_batch:
                    for ready in _queue_records(_flush_pair_contexts()):
                        stats["chunks"] += 1
                        stats["output_frames"] += int(ready.shape[0])
                        yield ready

            for ready in _queue_records(_flush_pair_contexts()):
                stats["chunks"] += 1
                stats["output_frames"] += int(ready.shape[0])
                yield ready

            for ready in _drain_until_chunk_possible(force=True):
                stats["chunks"] += 1
                stats["output_frames"] += int(ready.shape[0])
                yield ready
    finally:
        if block4_scale1_optimizer is not None:
            ssnap = block4_scale1_optimizer.snapshot()
            stats["block4_scale1_calls"] = int(ssnap["calls"])
            stats["block4_scale1_bypassed_interpolates"] = int(ssnap["bypassed_interpolates"])
            stats["block4_scale1_fallbacks"] = int(ssnap["fallbacks"])
            block4_scale1_optimizer.close()
        if block4_res_optimizer is not None:
            bsnap = block4_res_optimizer.snapshot()
            stats["block4_res_inplace_calls"] = int(bsnap["calls"])
            stats["block4_res_inplace_patched"] = int(bsnap["patched"])
            block4_res_optimizer.close()
        if outer_cat_optimizer is not None:
            osnap = outer_cat_optimizer.snapshot()
            stats["outer_cat_pushdown_forwards"] = int(osnap["forwards"])
            stats["outer_cat_pushdown_pushed_cats"] = int(osnap["pushed_cats"])
            stats["outer_cat_pushdown_component_interpolates"] = int(osnap["component_interpolates"])
            stats["outer_cat_pushdown_fallbacks"] = int(osnap["fallbacks"])
            outer_cat_optimizer.close()
        if stage_profiler is not None:
            stage_profiler.close()
        # Any speculative source prefetch that was never consumed (e.g. because
        # the following pair was a scene cut) must complete before its pinned
        # staging buffer can be recycled.
        for desc in list(source_uploads.values()):
            try:
                _resolve_source_upload(desc)
            except Exception:
                pass
        source_uploads.clear()
        stats["prepare_seconds"] = (
            stats.get("layout_seconds", 0.0)
            + stats.get("h2d_wait_seconds", 0.0)
            + stats.get("gpu_prepare_seconds", 0.0)
        )
        if warp_input_optimizer is not None:
            snap = warp_input_optimizer.snapshot()
            stats["warp_input_casts"] = int(snap["casts"])
            stats["warp_input_reuses"] = int(snap["reuses"])
            stats["warp_input_peak_entries"] = int(snap["peak_entries"])
            stats["warp_input_forwards"] = int(snap["forwards"])
            stats["warp_input_fallbacks"] = int(snap["fallbacks"])
            warp_input_optimizer.close()
        if warp_grid_optimizer is not None:
            gsnap = warp_grid_optimizer.snapshot()
            stats["warp_grid_builds"] = int(gsnap["builds"])
            stats["warp_grid_reuses"] = int(gsnap["reuses"])
            stats["warp_grid_peak_entries"] = int(gsnap["peak_entries"])
            stats["warp_grid_forwards"] = int(gsnap["forwards"])
            stats["warp_grid_fallbacks"] = int(gsnap["fallbacks"])
            stats["warp_grid_invalidations"] = int(gsnap.get("invalidations", 0))
            stats["warp_grid_strategy"] = str(gsnap.get("strategy", stats.get("warp_grid_strategy", "off")))
            warp_grid_optimizer.close()

    if final_tail:
        tail = images[-1:].to(device="cpu", dtype=out_dtype).expand(n_mid, -1, -1, -1).clone()
        for ready in add(tail):
            stats["chunks"] += 1
            stats["output_frames"] += int(ready.shape[0])
            yield ready

    if buffered_n:
        final_chunk = torch.cat(buffered, dim=0).contiguous()
        stats["chunks"] += 1
        stats["output_frames"] += int(final_chunk.shape[0])
        yield final_chunk


class LongVideoInterpolationStreamBatch:
    """Single meta-batch pull stream consumed by the streaming encoder node."""

    def __init__(
        self,
        interp_model: Any,
        images: torch.Tensor,
        state: LongVideoInterpolationState,
        state_key: Tuple[int, str],
        multiplier: int,
        source_fps: float,
        output_fps: float,
        stereo_mode: str,
        scene_cut: bool,
        scene_threshold: float,
        cpu_output: str,
        timestep_batch: int,
        pair_batch: int,
        chunk_frames: int,
        d2h_mode: str,
        h2d_mode: str,
        final: bool,
        bypass_reason: str = "source_at_target_rate",
        ifnet_profile: str = "off",
        feature_cache: str = "adjacent",
        warp_input_cache: str = "off",
        warp_grid_cache: str = "off",
    ):
        self.interp_model = interp_model
        self.images = images
        self.state = state
        self.state_key = state_key
        self.multiplier = int(multiplier)
        self.source_fps = float(source_fps)
        self.output_fps = float(output_fps)
        self.stereo_mode = str(stereo_mode)
        self.scene_cut = bool(scene_cut)
        self.scene_threshold = float(scene_threshold)
        self.cpu_output = str(cpu_output)
        self.timestep_batch = int(timestep_batch)
        self.pair_batch = int(pair_batch)
        self.chunk_frames = int(chunk_frames)
        self.d2h_mode = str(d2h_mode)
        self.h2d_mode = str(h2d_mode)
        self.ifnet_profile = str(ifnet_profile)
        self.feature_cache = str(feature_cache)
        self.warp_input_cache = str(warp_input_cache)
        self.warp_grid_cache = str(warp_grid_cache)
        self.final = bool(final)
        self.bypass_reason = str(bypass_reason)
        self.height = int(images.shape[1])
        self.width = int(images.shape[2])
        self.model_name = _model_label(interp_model)
        self.source_name = str(getattr(state, "source_name", "") or "")
        self.stats: Dict[str, Any] = {}
        self.consumed = False
        self.summary = ""

        has_prev = state.last_frame is not None
        base = int(images.shape[0]) * self.multiplier if has_prev else ((int(images.shape[0]) - 1) * self.multiplier + 1)
        if self.final:
            base += self.multiplier - 1
        self.planned_frames = int(base)

    def _iter_passthrough_chunks(self) -> _Iterator[torch.Tensor]:
        """Yield source frames unchanged for an already-target-rate stream."""
        emitted = 0
        chunks = 0
        d2h_seconds = 0.0
        target_dtype = torch.float16 if self.cpu_output == "float16" else torch.float32
        completed = False
        try:
            total = int(self.images.shape[0])
            for start in range(0, total, max(1, self.chunk_frames)):
                chunk = self.images[start:start + max(1, self.chunk_frames)]
                if chunk.device.type != "cpu" or chunk.dtype != target_dtype:
                    t0 = _time.perf_counter()
                    chunk = chunk.detach().to(device="cpu", dtype=target_dtype).contiguous()
                    d2h_seconds += _time.perf_counter() - t0
                else:
                    chunk = chunk.contiguous()
                emitted += int(chunk.shape[0])
                chunks += 1
                yield chunk
            completed = True
        finally:
            if completed:
                if emitted != self.planned_frames:
                    raise RuntimeError(
                        f"Streaming target-rate bypass accounting error: emitted {emitted}, "
                        f"planned {self.planned_frames}"
                    )
                self.state.input_frames += int(self.images.shape[0])
                self.state.output_frames += emitted
                self.state.last_frame = None
                if self.state.output_frames != self.state.input_frames:
                    raise RuntimeError(
                        f"Global target-rate bypass accounting error: output={self.state.output_frames}, "
                        f"input={self.state.input_frames}"
                    )

                self.stats.update({
                    "bypass_active": True,
                    "bypass_reason": self.bypass_reason,
                    "chunks": chunks,
                    "output_frames": emitted,
                    "rife_inference_seconds": 0.0,
                    "rife_seconds": 0.0,
                    "ifnet_seconds": 0.0,
                    "rife_post_seconds": 0.0,
                    "rife_pack_seconds": 0.0,
                    "rife_forward_calls": 0,
                    "rife_pairs_submitted": 0,
                    "pair_batch_active": 1,
                    "pair_batch_requested": self.pair_batch,
                    "feature_cache_active": "off",
                    "feature_cache_requested": self.feature_cache,
                    "d2h_seconds": d2h_seconds,
                    "d2h_wait_seconds": 0.0,
                    "d2h_mode_active": "passthrough",
                    "d2h_mode_requested": self.d2h_mode,
                    "h2d_seconds": 0.0,
                    "h2d_wait_seconds": 0.0,
                    "h2d_mode_active": "passthrough",
                    "h2d_mode_requested": self.h2d_mode,
                    "scene_seconds": 0.0,
                    "scene_pairs": 0,
                    "scene_cuts": 0,
                    "layout_seconds": 0.0,
                    "gpu_prepare_seconds": 0.0,
                    "assemble_seconds": 0.0,
                })
                duration_change = (
                    (self.state.output_frames / self.output_fps) /
                    (self.state.input_frames / self.source_fps) - 1.0
                ) * 100.0 if self.final and self.state.input_frames else 0.0
                self.summary = (
                    f"{self.model_name} BYPASS 1x ({self.bypass_reason}) | "
                    f"{self.source_fps:.9f}->{self.output_fps:.9f} fps | "
                    f"batch_out={emitted} chunks={chunks} | "
                    f"input={self.state.input_frames} output={self.state.output_frames} | "
                    f"final={self.final} duration_change={duration_change:+.9f}%"
                )
                print(
                    f"[LongVideo RIFE Bypass] reason={self.bypass_reason} source={self.source_fps:.9f} "
                    f"output={self.output_fps:.9f} frames={emitted} final={self.final}"
                )
                if self.final:
                    with _STATES_LOCK:
                        _STATES.pop(self.state_key, None)

    def iter_chunks(self) -> _Iterator[torch.Tensor]:
        if self.consumed:
            raise RuntimeError("LongVideo interpolation stream batch can only be consumed once")
        self.consumed = True

        if self.multiplier == 1:
            yield from self._iter_passthrough_chunks()
            return

        prior = self.state.last_frame
        include_first = prior is None
        if prior is not None:
            work = torch.cat((prior.to(dtype=self.images.dtype), self.images), dim=0)
        else:
            work = self.images

        emitted = 0
        completed = False
        if self.d2h_mode == "async_pinned" and self.state.pinned_pool is None:
            self.state.pinned_pool = _PinnedCpuBufferPool()
        if self.h2d_mode == "async_pinned" and self.state.h2d_pinned_pool is None:
            self.state.h2d_pinned_pool = _PinnedCpuBufferPool()
        try:
            for chunk in iter_interpolated_sequence_chunks(
                interp_model=self.interp_model,
                images=work,
                multiplier=self.multiplier,
                stereo_mode=self.stereo_mode,
                scene_cut=self.scene_cut,
                scene_threshold=self.scene_threshold,
                cpu_output=self.cpu_output,
                timestep_batch=self.timestep_batch,
                pair_batch=self.pair_batch,
                chunk_frames=self.chunk_frames,
                d2h_mode=self.d2h_mode,
                h2d_mode=self.h2d_mode,
                ifnet_profile=self.ifnet_profile,
                feature_cache=self.feature_cache,
                warp_input_cache=self.warp_input_cache,
                warp_grid_cache=self.warp_grid_cache,
                pinned_pool=self.state.pinned_pool,
                h2d_pinned_pool=self.state.h2d_pinned_pool,
                include_first=include_first,
                final_tail=self.final,
                stats=self.stats,
            ):
                emitted += int(chunk.shape[0])
                yield chunk
            completed = True
        finally:
            if completed:
                if emitted != self.planned_frames:
                    raise RuntimeError(
                        f"Streaming RIFE accounting error: emitted {emitted}, planned {self.planned_frames}"
                    )
                self.state.input_frames += int(self.images.shape[0])
                self.state.output_frames += emitted
                self.state.last_frame = None if self.final else self.images[-1:].detach().cpu().clone()

                if self.final:
                    expected_global = self.state.input_frames * self.multiplier
                else:
                    expected_global = self.state.input_frames * self.multiplier - (self.multiplier - 1)
                if self.state.output_frames != expected_global:
                    raise RuntimeError(
                        f"Global streaming RIFE accounting error: output={self.state.output_frames}, "
                        f"expected={expected_global}"
                    )

                duration_change = (
                    (self.state.output_frames / self.output_fps) /
                    (self.state.input_frames / self.source_fps) - 1.0
                ) * 100.0 if self.final and self.state.input_frames else 0.0
                self.summary = (
                    f"{self.model_name} {self.multiplier}x | "
                    f"{self.source_fps:.9f}->{self.output_fps:.9f} fps | "
                    f"batch_out={emitted} chunks={self.stats.get('chunks', 0)} | "
                    f"input={self.state.input_frames} output={self.state.output_frames} | "
                    f"final={self.final} duration_change={duration_change:+.9f}%"
                )

                # DEV2.10.1: always disclose the requested/active warp-cache state so
                # an install/version mismatch or model-compatibility fallback is visible
                # immediately in benchmark logs.
                if self.warp_input_cache != "off":
                    reason = str(self.stats.get("warp_input_cache_fallback_reason", "") or "none")
                    print(
                        f"[LongVideo IFNet Warp Input Cache Config] "
                        f"requested={self.warp_input_cache} "
                        f"active={self.stats.get('warp_input_cache_active', 'off')} "
                        f"model={self.stats.get('ifnet_model_class', 'unknown')} fallback={reason}"
                    )

                # DEV2.10: report exact-math warp source-cast reuse independently
                # of the companion encoder's generic stream profiler.
                if self.stats.get("warp_input_cache_active") == "fp32_reuse":
                    print(
                        f"[LongVideo IFNet Warp Input Cache] "
                        f"mode=fp32_reuse casts={int(self.stats.get('warp_input_casts', 0))} "
                        f"reuses={int(self.stats.get('warp_input_reuses', 0))} "
                        f"peak={int(self.stats.get('warp_input_peak_entries', 0))} "
                        f"forwards={int(self.stats.get('warp_input_forwards', 0))} "
                        f"fallbacks={int(self.stats.get('warp_input_fallbacks', 0))}"
                    )
                    totals = self.state.warp_input_cache_totals
                    if totals is None:
                        totals = {"casts": 0, "reuses": 0, "peak": 0, "forwards": 0, "fallbacks": 0}
                        self.state.warp_input_cache_totals = totals
                    totals["casts"] += int(self.stats.get("warp_input_casts", 0))
                    totals["reuses"] += int(self.stats.get("warp_input_reuses", 0))
                    totals["peak"] = max(int(totals.get("peak", 0)), int(self.stats.get("warp_input_peak_entries", 0)))
                    totals["forwards"] += int(self.stats.get("warp_input_forwards", 0))
                    totals["fallbacks"] += int(self.stats.get("warp_input_fallbacks", 0))
                    if self.final:
                        print(
                            f"[LongVideo IFNet Warp Input Cache TOTAL] "
                            f"mode=fp32_reuse casts={int(totals['casts'])} reuses={int(totals['reuses'])} "
                            f"peak={int(totals['peak'])} forwards={int(totals['forwards'])} "
                            f"fallbacks={int(totals['fallbacks'])}"
                        )

                # DEV2.11: report exact sampling-grid reuse across IFNet stages.
                if self.warp_grid_cache != "off":
                    reason = str(self.stats.get("warp_grid_cache_fallback_reason", "") or "none")
                    print(
                        f"[LongVideo IFNet Warp Grid Cache Config] "
                        f"requested={self.warp_grid_cache} "
                        f"active={self.stats.get('warp_grid_cache_active', 'off')} "
                        f"strategy={self.stats.get('warp_grid_strategy', 'off')} "
                        f"model={self.stats.get('ifnet_model_class', 'unknown')} fallback={reason}"
                    )
                if self.stats.get("warp_grid_cache_active") == "flow_reuse":
                    print(
                        f"[LongVideo IFNet Warp Grid Cache] "
                        f"mode=flow_reuse builds={int(self.stats.get('warp_grid_builds', 0))} "
                        f"reuses={int(self.stats.get('warp_grid_reuses', 0))} "
                        f"peak={int(self.stats.get('warp_grid_peak_entries', 0))} "
                        f"forwards={int(self.stats.get('warp_grid_forwards', 0))} "
                        f"fallbacks={int(self.stats.get('warp_grid_fallbacks', 0))}"
                    )
                    totals = self.state.warp_grid_cache_totals
                    if totals is None:
                        totals = {"builds": 0, "reuses": 0, "peak": 0, "forwards": 0, "fallbacks": 0}
                        self.state.warp_grid_cache_totals = totals
                    totals["builds"] += int(self.stats.get("warp_grid_builds", 0))
                    totals["reuses"] += int(self.stats.get("warp_grid_reuses", 0))
                    totals["peak"] = max(int(totals.get("peak", 0)), int(self.stats.get("warp_grid_peak_entries", 0)))
                    totals["forwards"] += int(self.stats.get("warp_grid_forwards", 0))
                    totals["fallbacks"] += int(self.stats.get("warp_grid_fallbacks", 0))
                    if self.final:
                        print(
                            f"[LongVideo IFNet Warp Grid Cache TOTAL] "
                            f"mode=flow_reuse builds={int(totals['builds'])} reuses={int(totals['reuses'])} "
                            f"peak={int(totals['peak'])} forwards={int(totals['forwards'])} "
                            f"fallbacks={int(totals['fallbacks'])}"
                        )

                # DEV2.16: block4 scale=1 bypass is now a locked production optimization.
                if self.stats.get("block4_scale1_requested") == "locked":
                    reason = str(self.stats.get("block4_scale1_fallback_reason", "") or "none")
                    print(
                        f"[LongVideo IFNet Block4 Scale1 Bypass Config] "
                        f"requested=locked active={self.stats.get('block4_scale1_active', 'off')} "
                        f"model={self.stats.get('ifnet_model_class', 'unknown')} fallback={reason}"
                    )
                    if self.stats.get("block4_scale1_active") == "bypass":
                        calls = int(self.stats.get("block4_scale1_calls", 0))
                        skipped = int(self.stats.get("block4_scale1_bypassed_interpolates", 0))
                        fallbacks = int(self.stats.get("block4_scale1_fallbacks", 0))
                        print(
                            f"[LongVideo IFNet Block4 Scale1 Bypass] "
                            f"calls={calls} skipped_interpolates={skipped} fallbacks={fallbacks}"
                        )
                        totals = self.state.block4_scale1_totals
                        if totals is None:
                            totals = {"calls": 0, "skipped": 0, "fallbacks": 0}
                            self.state.block4_scale1_totals = totals
                        totals["calls"] += calls
                        totals["skipped"] += skipped
                        totals["fallbacks"] += fallbacks
                        if self.final:
                            print(
                                f"[LongVideo IFNet Block4 Scale1 Bypass TOTAL] "
                                f"calls={int(totals['calls'])} skipped_interpolates={int(totals['skipped'])} "
                                f"fallbacks={int(totals['fallbacks'])}"
                            )

                # DEV2.14: exact-output in-place block4 residual A/B telemetry.
                if self.ifnet_profile == "block4_res_inplace":
                    reason = str(self.stats.get("block4_res_inplace_fallback_reason", "") or "none")
                    print(
                        f"[LongVideo IFNet Block4 ResConv Inplace Config] "
                        f"requested=inplace active={self.stats.get('block4_res_inplace_active', 'off')} "
                        f"model={self.stats.get('ifnet_model_class', 'unknown')} fallback={reason}"
                    )
                    if self.stats.get("block4_res_inplace_active") == "inplace":
                        calls = int(self.stats.get("block4_res_inplace_calls", 0))
                        patched = int(self.stats.get("block4_res_inplace_patched", 0))
                        print(
                            f"[LongVideo IFNet Block4 ResConv Inplace] "
                            f"calls={calls} patched={patched}"
                        )
                        totals = self.state.block4_res_inplace_totals
                        if totals is None:
                            totals = {"calls": 0, "patched": 0}
                            self.state.block4_res_inplace_totals = totals
                        totals["calls"] += calls
                        totals["patched"] = max(int(totals.get("patched", 0)), patched)
                        if self.final:
                            print(
                                f"[LongVideo IFNet Block4 ResConv Inplace TOTAL] "
                                f"calls={int(totals['calls'])} patched={int(totals['patched'])}"
                            )

                # DEV2.19: FP16 outer-cat pushdown is now a locked production optimization.
                if self.stats.get("outer_cat_pushdown_requested") == "locked":
                    reason = str(self.stats.get("outer_cat_pushdown_fallback_reason", "") or "none")
                    print(
                        f"[LongVideo IFNet Outer Cat Pushdown Config] "
                        f"requested=locked active={self.stats.get('outer_cat_pushdown_active', 'off')} "
                        f"model={self.stats.get('ifnet_model_class', 'unknown')} fallback={reason}"
                    )
                    if self.stats.get("outer_cat_pushdown_active") == "fp16":
                        fw = int(self.stats.get("outer_cat_pushdown_forwards", 0))
                        pc = int(self.stats.get("outer_cat_pushdown_pushed_cats", 0))
                        ci = int(self.stats.get("outer_cat_pushdown_component_interpolates", 0))
                        fb = int(self.stats.get("outer_cat_pushdown_fallbacks", 0))
                        print(
                            f"[LongVideo IFNet Outer Cat Pushdown] "
                            f"forwards={fw} pushed_cats={pc} component_interpolates={ci} fallbacks={fb}"
                        )
                        totals = self.state.outer_cat_pushdown_totals
                        if totals is None:
                            totals = {"forwards": 0, "pushed_cats": 0, "component_interpolates": 0, "fallbacks": 0}
                            self.state.outer_cat_pushdown_totals = totals
                        totals["forwards"] += fw
                        totals["pushed_cats"] += pc
                        totals["component_interpolates"] += ci
                        totals["fallbacks"] += fb
                        if self.final:
                            print(
                                f"[LongVideo IFNet Outer Cat Pushdown TOTAL] "
                                f"forwards={int(totals['forwards'])} pushed_cats={int(totals['pushed_cats'])} "
                                f"component_interpolates={int(totals['component_interpolates'])} "
                                f"fallbacks={int(totals['fallbacks'])}"
                            )

                # DEV2.19: post-optimization warp-inner profiler. This is
                # integrated into the accepted grid-cache warp so profiling does
                # not disable grid reuse or change sampling math.
                if self.ifnet_profile == "warp_postopt_detail":
                    ws = self.stats.get("warp_postopt_seconds", {})
                    wc = self.stats.get("warp_postopt_calls", {})
                    def _sum_part(part):
                        return float(ws.get(f"warp_post_image_{part}", 0.0)) + float(ws.get(f"warp_post_feature_{part}", 0.0)) + float(ws.get(f"warp_post_other_{part}", 0.0))
                    def _sum_calls(part):
                        return int(wc.get(f"warp_post_image_{part}", 0)) + int(wc.get(f"warp_post_feature_{part}", 0)) + int(wc.get(f"warp_post_other_{part}", 0))
                    norm = _sum_part("norm")
                    grid = _sum_part("grid")
                    cast_in = _sum_part("cast_in")
                    sample = _sum_part("sample")
                    cast_out = _sum_part("cast_out")
                    image = sum(float(ws.get(f"warp_post_image_{p}", 0.0)) for p in ("norm","grid","cast_in","sample","cast_out"))
                    feature = sum(float(ws.get(f"warp_post_feature_{p}", 0.0)) for p in ("norm","grid","cast_in","sample","cast_out"))
                    print(
                        f"[LongVideo IFNet Warp Postopt Profile] "
                        f"norm={norm:.3f}s/{_sum_calls('norm')}x grid={grid:.3f}s/{_sum_calls('grid')}x "
                        f"cast_in={cast_in:.3f}s/{_sum_calls('cast_in')}x sample={sample:.3f}s/{_sum_calls('sample')}x "
                        f"cast_out={cast_out:.3f}s/{_sum_calls('cast_out')}x "
                        f"image={image:.3f}s feature={feature:.3f}s "
                        f"profiled={int(self.stats.get('warp_postopt_profiled_forwards', 0))}fwd"
                    )
                    totals = self.state.warp_postopt_totals
                    if totals is None:
                        totals = {"seconds": {}, "calls": {}, "forwards": 0}
                        self.state.warp_postopt_totals = totals
                    for k, v in ws.items():
                        totals["seconds"][k] = float(totals["seconds"].get(k, 0.0)) + float(v)
                    for k, v in wc.items():
                        totals["calls"][k] = int(totals["calls"].get(k, 0)) + int(v)
                    totals["forwards"] += int(self.stats.get("warp_postopt_profiled_forwards", 0))
                    if self.final:
                        tws, twc = totals["seconds"], totals["calls"]
                        def _tsum(part):
                            return float(tws.get(f"warp_post_image_{part}", 0.0)) + float(tws.get(f"warp_post_feature_{part}", 0.0)) + float(tws.get(f"warp_post_other_{part}", 0.0))
                        def _tcall(part):
                            return int(twc.get(f"warp_post_image_{part}", 0)) + int(twc.get(f"warp_post_feature_{part}", 0)) + int(twc.get(f"warp_post_other_{part}", 0))
                        timage = sum(float(tws.get(f"warp_post_image_{p}", 0.0)) for p in ("norm","grid","cast_in","sample","cast_out"))
                        tfeature = sum(float(tws.get(f"warp_post_feature_{p}", 0.0)) for p in ("norm","grid","cast_in","sample","cast_out"))
                        print(
                            f"[LongVideo IFNet Warp Postopt Profile TOTAL] "
                            f"norm={_tsum('norm'):.3f}s/{_tcall('norm')}x grid={_tsum('grid'):.3f}s/{_tcall('grid')}x "
                            f"cast_in={_tsum('cast_in'):.3f}s/{_tcall('cast_in')}x sample={_tsum('sample'):.3f}s/{_tcall('sample')}x "
                            f"cast_out={_tsum('cast_out'):.3f}s/{_tcall('cast_out')}x "
                            f"image={timage:.3f}s feature={tfeature:.3f}s profiled={int(totals['forwards'])}fwd"
                        )

                # DEV2.17: decompose the optimized non-module IFNet path.
                # These spans are mutually exclusive with encode/block module
                # timings and preserve the currently installed grid-reuse warp.
                if self.stats.get("ifnet_profile_active") == "outer_detail":
                    stage_s = self.stats.get("ifnet_stage_seconds", {})
                    stage_c = self.stats.get("ifnet_stage_calls", {})
                    cat_initial = float(stage_s.get("outer_cat_initial", 0.0))
                    cat_refine = float(stage_s.get("outer_cat_refine", 0.0))
                    warp_feature = float(stage_s.get("outer_warp_feature", 0.0))
                    warp_image = float(stage_s.get("outer_warp_image", 0.0))
                    flow_add = float(stage_s.get("outer_flow_add", 0.0))
                    sigmoid = float(stage_s.get("outer_sigmoid", 0.0))
                    lerp = float(stage_s.get("outer_lerp", 0.0))
                    timestep = float(stage_s.get("outer_timestep", 0.0))
                    misc = float(self.stats.get("ifnet_stage_unattributed_seconds", 0.0))
                    print(
                        f"[LongVideo IFNet Outer Profile] "
                        f"cat={cat_initial + cat_refine:.3f}s/{int(stage_c.get('outer_cat_initial', 0)) + int(stage_c.get('outer_cat_refine', 0))}x "
                        f"feature_warp={warp_feature:.3f}s/{int(stage_c.get('outer_warp_feature', 0))}x "
                        f"image_warp={warp_image:.3f}s/{int(stage_c.get('outer_warp_image', 0))}x "
                        f"flow_add={flow_add:.3f}s/{int(stage_c.get('outer_flow_add', 0))}x "
                        f"sigmoid={sigmoid:.3f}s lerp={lerp:.3f}s timestep={timestep:.3f}s "
                        f"misc={misc:.3f}s profiled={int(self.stats.get('ifnet_stage_profiled_forwards', 0))}fwd"
                    )
                    totals = self.state.ifnet_outer_totals
                    if totals is None:
                        totals = {"seconds": {}, "calls": {}, "misc": 0.0, "forwards": 0}
                        self.state.ifnet_outer_totals = totals
                    for name, sec in stage_s.items():
                        totals["seconds"][name] = float(totals["seconds"].get(name, 0.0)) + float(sec)
                    for name, count in stage_c.items():
                        totals["calls"][name] = int(totals["calls"].get(name, 0)) + int(count)
                    totals["misc"] += misc
                    totals["forwards"] += int(self.stats.get("ifnet_stage_profiled_forwards", 0))
                    if self.final:
                        ts, tc = totals["seconds"], totals["calls"]
                        ci = float(ts.get("outer_cat_initial", 0.0))
                        cr = float(ts.get("outer_cat_refine", 0.0))
                        wf = float(ts.get("outer_warp_feature", 0.0))
                        wi = float(ts.get("outer_warp_image", 0.0))
                        fa = float(ts.get("outer_flow_add", 0.0))
                        sg = float(ts.get("outer_sigmoid", 0.0))
                        lp = float(ts.get("outer_lerp", 0.0))
                        tt = float(ts.get("outer_timestep", 0.0))
                        print(
                            f"[LongVideo IFNet Outer Profile TOTAL] "
                            f"cat={ci + cr:.3f}s/{int(tc.get('outer_cat_initial', 0)) + int(tc.get('outer_cat_refine', 0))}x "
                            f"feature_warp={wf:.3f}s/{int(tc.get('outer_warp_feature', 0))}x "
                            f"image_warp={wi:.3f}s/{int(tc.get('outer_warp_image', 0))}x "
                            f"flow_add={fa:.3f}s/{int(tc.get('outer_flow_add', 0))}x "
                            f"sigmoid={sg:.3f}s lerp={lp:.3f}s timestep={tt:.3f}s "
                            f"misc={float(totals.get('misc', 0.0)):.3f}s profiled={int(totals.get('forwards', 0))}fwd"
                        )

                # DEV2.16: refresh convolution-family hotspots under all locked
                # optimizations. Child hooks are nested beneath parent blocks and
                # therefore reported separately without affecting outer attribution.
                if self.stats.get("ifnet_profile_active") == "conv_family_detail":
                    stage_s = self.stats.get("ifnet_stage_seconds", {})
                    stage_c = self.stats.get("ifnet_stage_calls", {})
                    def _block_conv_parts(bi):
                        conv0 = float(stage_s.get(f"block{bi}_conv0_0", 0.0)) + float(stage_s.get(f"block{bi}_conv0_1", 0.0))
                        res = sum(float(stage_s.get(f"block{bi}_res{ri}_conv", 0.0)) for ri in range(8))
                        deconv = float(stage_s.get(f"block{bi}_deconv", 0.0))
                        pix = float(stage_s.get(f"block{bi}_pixelshuffle", 0.0))
                        return conv0, res, deconv, pix
                    parts = [_block_conv_parts(i) for i in range(5)]
                    conv0_total = sum(x[0] for x in parts)
                    res_total = sum(x[1] for x in parts)
                    deconv_total = sum(x[2] for x in parts)
                    pix_total = sum(x[3] for x in parts)
                    print(
                        f"[LongVideo IFNet Conv Family Profile] "
                        f"conv0={conv0_total:.3f}s resconv={res_total:.3f}s "
                        f"deconv={deconv_total:.3f}s pixelshuffle={pix_total:.3f}s "
                        + " ".join(
                            f"b{i}=({parts[i][0]:.3f}/{parts[i][1]:.3f}/{parts[i][2]:.3f})"
                            for i in range(5)
                        )
                        + f" profiled={int(self.stats.get('ifnet_stage_profiled_forwards', 0))}fwd"
                    )
                    totals = self.state.ifnet_conv_family_totals
                    if totals is None:
                        totals = {"seconds": {}, "calls": {}, "forwards": 0}
                        self.state.ifnet_conv_family_totals = totals
                    for name, sec in stage_s.items():
                        totals["seconds"][name] = float(totals["seconds"].get(name, 0.0)) + float(sec)
                    for name, count in stage_c.items():
                        totals["calls"][name] = int(totals["calls"].get(name, 0)) + int(count)
                    totals["forwards"] += int(self.stats.get("ifnet_stage_profiled_forwards", 0))
                    if self.final:
                        ts = totals["seconds"]
                        def _tparts(bi):
                            conv0 = float(ts.get(f"block{bi}_conv0_0", 0.0)) + float(ts.get(f"block{bi}_conv0_1", 0.0))
                            res = sum(float(ts.get(f"block{bi}_res{ri}_conv", 0.0)) for ri in range(8))
                            deconv = float(ts.get(f"block{bi}_deconv", 0.0))
                            pix = float(ts.get(f"block{bi}_pixelshuffle", 0.0))
                            return conv0, res, deconv, pix
                        tp = [_tparts(i) for i in range(5)]
                        print(
                            f"[LongVideo IFNet Conv Family Profile TOTAL] "
                            f"conv0={sum(x[0] for x in tp):.3f}s "
                            f"resconv={sum(x[1] for x in tp):.3f}s "
                            f"deconv={sum(x[2] for x in tp):.3f}s "
                            f"pixelshuffle={sum(x[3] for x in tp):.3f}s "
                            + " ".join(
                                f"b{i}=({tp[i][0]:.3f}/{tp[i][1]:.3f}/{tp[i][2]:.3f})"
                                for i in range(5)
                            )
                            + f" profiled={int(totals.get('forwards', 0))}fwd"
                        )

                # DEV2.12: block4 internal telemetry. Parent block4 is timed
                # together with its non-overlapping child modules; the remainder
                # is functional interpolate/cat/slice/scale bookkeeping.
                if self.stats.get("ifnet_profile_active") == "block4_detail":
                    stage_s = self.stats.get("ifnet_stage_seconds", {})
                    stage_c = self.stats.get("ifnet_stage_calls", {})
                    child_names = [
                        "block4_conv0_0", "block4_conv0_1",
                        *[f"block4_res{i}" for i in range(8)],
                        "block4_deconv", "block4_pixelshuffle",
                    ]
                    parent = float(stage_s.get("block4", 0.0))
                    child_sum = sum(float(stage_s.get(n, 0.0)) for n in child_names)
                    misc = max(0.0, parent - child_sum)
                    res_sum = sum(float(stage_s.get(f"block4_res{i}", 0.0)) for i in range(8))
                    res_calls = sum(int(stage_c.get(f"block4_res{i}", 0)) for i in range(8))
                    res_conv = sum(float(stage_s.get(f"block4_res{i}_conv", 0.0)) for i in range(8))
                    res_relu = sum(float(stage_s.get(f"block4_res{i}_relu", 0.0)) for i in range(8))
                    res_misc = max(0.0, res_sum - res_conv - res_relu)
                    print(
                        f"[LongVideo IFNet Block4 Profile] "
                        f"block4={parent:.3f}s/{int(stage_c.get('block4', 0))}x "
                        f"conv0={float(stage_s.get('block4_conv0_0', 0.0)) + float(stage_s.get('block4_conv0_1', 0.0)):.3f}s "
                        f"res8={res_sum:.3f}s/{res_calls}x "
                        f"res_conv={res_conv:.3f}s res_relu={res_relu:.3f}s res_misc={res_misc:.3f}s "
                        f"deconv={float(stage_s.get('block4_deconv', 0.0)):.3f}s "
                        f"pixelshuffle={float(stage_s.get('block4_pixelshuffle', 0.0)):.3f}s "
                        f"misc={misc:.3f}s profiled={int(self.stats.get('ifnet_stage_profiled_forwards', 0))}fwd"
                    )
                    totals = self.state.ifnet_block4_totals
                    if totals is None:
                        totals = {"seconds": {}, "calls": {}, "forwards": 0}
                        self.state.ifnet_block4_totals = totals
                    for name, sec in stage_s.items():
                        totals["seconds"][name] = float(totals["seconds"].get(name, 0.0)) + float(sec)
                    for name, count in stage_c.items():
                        totals["calls"][name] = int(totals["calls"].get(name, 0)) + int(count)
                    totals["forwards"] += int(self.stats.get("ifnet_stage_profiled_forwards", 0))
                    if self.final:
                        ts, tc = totals["seconds"], totals["calls"]
                        p = float(ts.get("block4", 0.0))
                        csum = sum(float(ts.get(n, 0.0)) for n in child_names)
                        m = max(0.0, p - csum)
                        rs = sum(float(ts.get(f"block4_res{i}", 0.0)) for i in range(8))
                        rc = sum(int(tc.get(f"block4_res{i}", 0)) for i in range(8))
                        rconv = sum(float(ts.get(f"block4_res{i}_conv", 0.0)) for i in range(8))
                        rrelu = sum(float(ts.get(f"block4_res{i}_relu", 0.0)) for i in range(8))
                        rmisc = max(0.0, rs - rconv - rrelu)
                        per_res = " ".join(f"r{i}={float(ts.get(f'block4_res{i}', 0.0)):.3f}s" for i in range(8))
                        print(
                            f"[LongVideo IFNet Block4 Profile TOTAL] "
                            f"block4={p:.3f}s/{int(tc.get('block4', 0))}x "
                            f"conv0={float(ts.get('block4_conv0_0', 0.0)) + float(ts.get('block4_conv0_1', 0.0)):.3f}s "
                            f"res8={rs:.3f}s/{rc}x "
                            f"res_conv={rconv:.3f}s res_relu={rrelu:.3f}s res_misc={rmisc:.3f}s "
                            f"deconv={float(ts.get('block4_deconv', 0.0)):.3f}s "
                            f"pixelshuffle={float(ts.get('block4_pixelshuffle', 0.0)):.3f}s "
                            f"misc={m:.3f}s {per_res} profiled={int(totals.get('forwards', 0))}fwd"
                        )

                # DEV2.9: inner warp telemetry. This is benchmark-only and
                # decomposes the exact current IFNet warp into norm/grid/fp32
                # input cast/grid_sample/cast-back without changing precision.
                if self.stats.get("ifnet_profile_active") == "warp_detail":
                    stage_s = self.stats.get("ifnet_stage_seconds", {})
                    stage_c = self.stats.get("ifnet_stage_calls", {})
                    def _sum_part(part):
                        return sum(float(stage_s.get(f"warp_{kind}_{part}", 0.0)) for kind in ("image", "feature", "other"))
                    def _sum_count(part):
                        return sum(int(stage_c.get(f"warp_{kind}_{part}", 0)) for kind in ("image", "feature", "other"))
                    norm = _sum_part("norm")
                    grid = _sum_part("grid")
                    cast_in = _sum_part("cast_in")
                    sample = _sum_part("sample")
                    cast_out = _sum_part("cast_out")
                    fallback = _sum_part("fallback")
                    misc = float(self.stats.get("ifnet_stage_unattributed_seconds", 0.0))
                    print(
                        f"[LongVideo IFNet Warp Inner Profile] "
                        f"norm={norm:.3f}s/{_sum_count('norm')}x "
                        f"grid={grid:.3f}s/{_sum_count('grid')}x "
                        f"cast_in={cast_in:.3f}s/{_sum_count('cast_in')}x "
                        f"sample={sample:.3f}s/{_sum_count('sample')}x "
                        f"cast_out={cast_out:.3f}s/{_sum_count('cast_out')}x "
                        f"fallback={fallback:.3f}s/{_sum_count('fallback')}x "
                        f"misc={misc:.3f}s profiled={int(self.stats.get('ifnet_stage_profiled_forwards', 0))}fwd"
                    )
                    totals = self.state.ifnet_detail_totals
                    if totals is None:
                        totals = {"seconds": {}, "calls": {}, "misc": 0.0, "forwards": 0}
                        self.state.ifnet_detail_totals = totals
                    for name, sec in stage_s.items():
                        totals["seconds"][name] = float(totals["seconds"].get(name, 0.0)) + float(sec)
                    for name, count in stage_c.items():
                        totals["calls"][name] = int(totals["calls"].get(name, 0)) + int(count)
                    totals["misc"] += misc
                    totals["forwards"] += int(self.stats.get("ifnet_stage_profiled_forwards", 0))
                    if self.final:
                        ts, tc = totals["seconds"], totals["calls"]
                        def _tsum(part):
                            return sum(float(ts.get(f"warp_{kind}_{part}", 0.0)) for kind in ("image", "feature", "other"))
                        def _tcount(part):
                            return sum(int(tc.get(f"warp_{kind}_{part}", 0)) for kind in ("image", "feature", "other"))
                        print(
                            f"[LongVideo IFNet Warp Inner Profile TOTAL] "
                            f"norm={_tsum('norm'):.3f}s/{_tcount('norm')}x "
                            f"grid={_tsum('grid'):.3f}s/{_tcount('grid')}x "
                            f"cast_in={_tsum('cast_in'):.3f}s/{_tcount('cast_in')}x "
                            f"sample={_tsum('sample'):.3f}s/{_tcount('sample')}x "
                            f"cast_out={_tsum('cast_out'):.3f}s/{_tcount('cast_out')}x "
                            f"fallback={_tsum('fallback'):.3f}s/{_tcount('fallback')}x "
                            f"misc={float(totals.get('misc', 0.0)):.3f}s profiled={int(totals.get('forwards', 0))}fwd"
                        )

                # DEV2.8: self-contained warp-detail telemetry. The companion
                # encoder already aggregates generic IFNet stage stats, but this
                # explicit line guarantees image/feature warp visibility even on
                # older encoder installs that only know encode/block0..4.
                if self.stats.get("ifnet_profile_active") == "detail":
                    stage_s = self.stats.get("ifnet_stage_seconds", {})
                    stage_c = self.stats.get("ifnet_stage_calls", {})
                    wi = float(stage_s.get("warp_image", 0.0))
                    wf = float(stage_s.get("warp_feature", 0.0))
                    wo = float(stage_s.get("warp_other", 0.0))
                    misc = float(self.stats.get("ifnet_stage_unattributed_seconds", 0.0))
                    print(
                        f"[LongVideo IFNet Detail Profile] "
                        f"warp_image={wi:.3f}s/{int(stage_c.get('warp_image', 0))}x "
                        f"warp_feature={wf:.3f}s/{int(stage_c.get('warp_feature', 0))}x "
                        f"warp_other={wo:.3f}s/{int(stage_c.get('warp_other', 0))}x "
                        f"misc={misc:.3f}s profiled={int(self.stats.get('ifnet_stage_profiled_forwards', 0))}fwd"
                    )
                    totals = self.state.ifnet_detail_totals
                    if totals is None:
                        totals = {"seconds": {}, "calls": {}, "misc": 0.0, "forwards": 0}
                        self.state.ifnet_detail_totals = totals
                    for name, sec in stage_s.items():
                        totals["seconds"][name] = float(totals["seconds"].get(name, 0.0)) + float(sec)
                    for name, count in stage_c.items():
                        totals["calls"][name] = int(totals["calls"].get(name, 0)) + int(count)
                    totals["misc"] += misc
                    totals["forwards"] += int(self.stats.get("ifnet_stage_profiled_forwards", 0))
                    if self.final:
                        ts, tc = totals["seconds"], totals["calls"]
                        print(
                            f"[LongVideo IFNet Detail Profile TOTAL] "
                            f"warp_image={float(ts.get('warp_image', 0.0)):.3f}s/{int(tc.get('warp_image', 0))}x "
                            f"warp_feature={float(ts.get('warp_feature', 0.0)):.3f}s/{int(tc.get('warp_feature', 0))}x "
                            f"warp_other={float(ts.get('warp_other', 0.0)):.3f}s/{int(tc.get('warp_other', 0))}x "
                            f"misc={float(totals.get('misc', 0.0)):.3f}s profiled={int(totals.get('forwards', 0))}fwd"
                        )

                if self.final:
                    with _STATES_LOCK:
                        _STATES.pop(self.state_key, None)
                    # No later meta-batch can reuse this run's pinned outputs.
                    if self.state.pinned_pool is not None:
                        self.state.pinned_pool.clear()
                        self.state.pinned_pool = None
                    if self.state.h2d_pinned_pool is not None:
                        self.state.h2d_pinned_pool.clear()
                        self.state.h2d_pinned_pool = None


class LV_LongVideoFrameInterpolationStream:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "interp_model": ("INTERP_MODEL",),
                "images": ("IMAGE",),
                "session": ("LV_SESSION",),
                "source_fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 240.0, "step": 0.001}),
                "target_fps": ("FLOAT", {"default": 120.0, "min": 2.0, "max": 480.0, "step": 0.001}),
                "stereo_mode": (["full_sbs", "split_eyes"], {"default": "full_sbs"}),
                "scene_cut": ("BOOLEAN", {"default": True}),
                "scene_threshold": ("FLOAT", {"default": 0.22, "min": 0.02, "max": 1.0, "step": 0.01}),
                "cpu_output": (["float16", "float32"], {"default": "float16"}),
                "timestep_batch": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "pair_batch": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "chunk_frames": ("INT", {"default": 8, "min": 1, "max": 32, "step": 1}),
                "d2h_mode": (["sync", "async_pinned"], {"default": "async_pinned"}),
                "h2d_mode": (["sync", "async_pinned"], {"default": "async_pinned"}),
                "ifnet_profile": (["off", "stages", "detail", "warp_detail", "block4_detail", "block4_res_inplace", "block4_scale1_bypass", "conv_family_detail", "outer_detail", "outer_cat_pushdown", "warp_postopt_detail"], {"default": "off"}),
                "feature_cache": (["off", "adjacent"], {"default": "adjacent"}),
                "warp_input_cache": (["off", "fp32_reuse", "grid_reuse"], {"default": "grid_reuse"}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("LV_INTERP_STREAM", "FLOAT", "STRING")
    RETURN_NAMES = ("stream", "output_fps", "status")
    FUNCTION = "make_stream"
    CATEGORY = "Long Video SBS"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def make_stream(
        self, interp_model, images, session, source_fps, target_fps, stereo_mode,
        scene_cut, scene_threshold, cpu_output, timestep_batch, pair_batch, chunk_frames, d2h_mode, h2d_mode, ifnet_profile="off", feature_cache="adjacent", warp_input_cache="grid_reuse",
        unique_id=None,
    ):
        manual_bypass = _manual_rife_bypass(interp_model)
        if manual_bypass:
            multiplier, output_fps = 1, float(source_fps)
        else:
            multiplier, output_fps = resolve_integer_multiplier(source_fps, target_fps)
        model_name = _model_label(interp_model)
        warp_experiment = str(warp_input_cache)
        if warp_experiment not in ("off", "fp32_reuse", "grid_reuse"):
            raise ValueError("warp_input_cache must be off, fp32_reuse, or grid_reuse")
        effective_warp_input_cache = "fp32_reuse" if warp_experiment == "fp32_reuse" else "off"
        effective_warp_grid_cache = "flow_reuse" if warp_experiment == "grid_reuse" else "off"
        signature = (
            model_name, float(source_fps), float(target_fps), int(multiplier), str(stereo_mode),
            bool(scene_cut), float(scene_threshold), str(cpu_output), int(timestep_batch), int(pair_batch),
            int(chunk_frames), str(d2h_mode), str(h2d_mode), str(ifnet_profile), str(feature_cache), warp_experiment,
            bool(manual_bypass),
        )
        key = _state_key(session, unique_id)
        with _STATES_LOCK:
            # Interrupted prompts can leave the previous session's retained frame
            # and pinned-buffer pool alive.  A different VDA session is a different
            # long-video run, even when the Comfy node id is unchanged.
            stale_keys = [k for k in _STATES if k[1] == str(unique_id) and k != key]
            for stale_key in stale_keys:
                stale = _STATES.pop(stale_key, None)
                if stale is not None and getattr(stale, "pinned_pool", None) is not None:
                    stale.pinned_pool.clear()
                    stale.pinned_pool = None
                if stale is not None and getattr(stale, "h2d_pinned_pool", None) is not None:
                    stale.h2d_pinned_pool.clear()
                    stale.h2d_pinned_pool = None
            state = _STATES.get(key)
            if state is None:
                state = LongVideoInterpolationState(
                    signature=signature,
                    source_name=str(getattr(session, "source_name", "") or "") or None,
                )
                _STATES[key] = state
            elif state.signature != signature:
                raise ValueError("RIFE streaming settings changed during an active long-video session")
            elif not state.source_name:
                state.source_name = str(getattr(session, "source_name", "") or "") or None

        final = bool(getattr(session, "finished", False))
        stream = LongVideoInterpolationStreamBatch(
            interp_model=interp_model,
            images=images,
            state=state,
            state_key=key,
            multiplier=multiplier,
            source_fps=float(source_fps),
            output_fps=float(output_fps),
            stereo_mode=str(stereo_mode),
            scene_cut=bool(scene_cut),
            scene_threshold=float(scene_threshold),
            cpu_output=str(cpu_output),
            timestep_batch=int(timestep_batch),
            pair_batch=int(pair_batch),
            chunk_frames=int(chunk_frames),
            d2h_mode=str(d2h_mode),
            h2d_mode=str(h2d_mode),
            ifnet_profile=str(ifnet_profile),
            feature_cache=str(feature_cache),
            warp_input_cache=effective_warp_input_cache,
            warp_grid_cache=effective_warp_grid_cache,
            final=final,
            bypass_reason="manual_disabled" if manual_bypass else "source_at_target_rate",
        )
        if multiplier == 1:
            reason = "manual enable_rife=false" if manual_bypass else "source already at target cadence"
            status = (
                f"STREAM {model_name} BYPASS 1x | {float(source_fps):.9f}->{output_fps:.9f} fps | "
                f"{reason} | planned={stream.planned_frames} | chunk={int(chunk_frames)} | final={final}"
            )
        else:
            status = (
                f"STREAM {model_name} {multiplier}x | {float(source_fps):.9f}->{output_fps:.9f} fps | "
                f"planned={stream.planned_frames} | pair_batch={int(pair_batch)} | chunk={int(chunk_frames)} | d2h={d2h_mode} | h2d={h2d_mode} | ifnet_profile={ifnet_profile} | feature_cache={feature_cache} | warp_experiment={warp_experiment} | stereo={stereo_mode} | final={final}"
            )
        return (stream, float(output_fps), status)
