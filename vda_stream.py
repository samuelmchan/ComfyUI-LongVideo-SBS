from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from .session import LongVideoSession
from .vda_backend import VDAModelHandle, transformed_hw


INFER_LEN = 32
OVERLAP = 10
KEYFRAMES = [0, 12, 24, 25, 26, 27, 28, 29, 30, 31]
INTERP_LEN = 8
STEP = INFER_LEN - OVERLAP  # 22
ALIGN_LEN = OVERLAP - INTERP_LEN  # 2
KF_ALIGN = KEYFRAMES[:ALIGN_LEN]  # [0, 12]

# One CPU worker is enough: steady 44-frame operation has at most one next
# VDA window whose 22 fresh frames can be prepared while the current B=1
# GPU forward is running. Keeping one worker avoids CPU oversubscription with
# OpenCV/FFmpeg while preserving deterministic per-frame preprocessing.
_VDA_PREPROCESS_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lv-vda-pre")
# Frame-level preprocessing pool. Two workers are the validated production default;
# 3/4 are exposed as optional CPU-tuning modes. Every task writes a distinct output
# slot, so worker count changes scheduling only, never resize/normalize math or order.
_VDA_FRAME_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="lv-vda-frame")


def compute_scale_and_shift(prediction, target, mask):
    prediction = prediction.astype(np.float32, copy=False)
    target = target.astype(np.float32, copy=False)
    mask = mask.astype(np.float32, copy=False)
    a00 = np.sum(mask * prediction * prediction)
    a01 = np.sum(mask * prediction)
    a11 = np.sum(mask)
    b0 = np.sum(mask * prediction * target)
    b1 = np.sum(mask * target)
    det = a00 * a11 - a01 * a01
    if det == 0:
        return 1.0, 0.0
    scale = (a11 * b0 - a01 * b1) / det
    shift = (-a01 * b0 + a00 * b1) / det
    return float(scale), float(shift)


def interpolate_frames(pre: Sequence[np.ndarray], post: Sequence[np.ndarray]):
    if len(pre) != len(post):
        raise ValueError("Interpolation lists must have equal length")
    if len(pre) <= 1:
        return [np.asarray(x, dtype=np.float32) for x in pre]
    step = 1.0 / (len(pre) - 1)
    out = []
    for i, (a, b) in enumerate(zip(pre, post)):
        w = i * step
        out.append(a * (1.0 - w) + b * w)
    return out


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _frame_to_model(frame: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    """Reference single-frame transform kept for regression/debugging.

    The production path uses `_frames_to_model_batch`, which performs the same
    OpenCV cubic resize and float32 ImageNet normalization but writes directly
    into one final NCHW batch allocation.
    """
    arr = frame.detach().cpu().numpy()
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    resized = cv2.resize(arr, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
    resized = (resized - _IMAGENET_MEAN) / _IMAGENET_STD
    chw = np.ascontiguousarray(resized.transpose(2, 0, 1), dtype=np.float32)
    return torch.from_numpy(chw)


def _frame_to_model_slot(
    frame: torch.Tensor, slot: np.ndarray, out_h: int, out_w: int
) -> None:
    """Resize/normalize one frame directly into its final NCHW batch slot."""
    arr = frame.detach().cpu().numpy()
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    resized = cv2.resize(arr, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
    np.subtract(resized, _IMAGENET_MEAN, out=resized)
    np.divide(resized, _IMAGENET_STD, out=resized)
    slot[...] = resized.transpose(2, 0, 1)


def _frames_to_model_batch(
    frames: Sequence[torch.Tensor], out_h: int, out_w: int, preprocess_workers: int = 1
) -> torch.Tensor:
    """Bit-identical batched CPU preprocessing with one final allocation.

    With preprocess_workers>1, independent per-frame OpenCV resize/normalize
    operations are dispatched to a persistent CPU pool. Each worker writes a
    distinct final batch slot, so ordering and numerical results remain identical
    to the single-worker path.
    """
    n = len(frames)
    if n == 0:
        return torch.empty((0, 3, out_h, out_w), dtype=torch.float32)

    batch = np.empty((n, 3, out_h, out_w), dtype=np.float32)
    requested = max(1, min(4, int(preprocess_workers)))
    workers = min(requested, n) if n >= 4 else 1
    if workers == 1:
        for i, frame in enumerate(frames):
            _frame_to_model_slot(frame, batch[i], out_h, out_w)
    else:
        futures = [
            _VDA_FRAME_EXECUTOR.submit(
                _frame_to_model_slot, frame, batch[i], out_h, out_w
            )
            for i, frame in enumerate(frames)
        ]
        for fut in futures:
            fut.result()

    return torch.from_numpy(batch)


def _autocast_context(device: torch.device, fp32: bool):
    # ROCm uses PyTorch's CUDA-compatible API namespace.
    device_type = "cuda" if device.type == "cuda" else device.type
    return torch.autocast(device_type=device_type, enabled=(not fp32 and device.type != "cpu"))


def _fill_window_input(
    dst: torch.Tensor,
    raw_window: Sequence[torch.Tensor],
    pre_keyframes: Optional[torch.Tensor],
    out_h: int,
    out_w: int,
    preprocess_workers: int = 1,
) -> None:
    """Fill one device-resident [32,3,H,W] VDA window exactly as upstream.

    `pre_keyframes` is either None for the first ever window or the ten
    preprocessed keyframes from the immediately preceding VDA input window.
    Only the 22 genuinely new RGB frames are CPU-preprocessed after startup.
    """
    if len(raw_window) != INFER_LEN:
        raise ValueError(f"Expected {INFER_LEN} frames, got {len(raw_window)}")

    if pre_keyframes is None:
        prepared = (
            _frames_to_model_batch(raw_window, out_h, out_w)
            if int(preprocess_workers) == 1
            else _frames_to_model_batch(raw_window, out_h, out_w, preprocess_workers)
        )
        dst.copy_(prepared, non_blocking=True)
        del prepared
        return

    pre = pre_keyframes
    if pre.ndim == 5:
        if pre.shape[0] != 1:
            raise RuntimeError(f"Expected one cached VDA prefix, got {tuple(pre.shape)}")
        pre = pre[0]
    expected_shape = (OVERLAP, 3, out_h, out_w)
    if tuple(pre.shape) != expected_shape:
        raise RuntimeError(
            "VDA input resolution changed mid-session; restart the workflow with fixed settings"
        )

    prepared_new = (
        _frames_to_model_batch(raw_window[OVERLAP:], out_h, out_w)
        if int(preprocess_workers) == 1
        else _frames_to_model_batch(raw_window[OVERLAP:], out_h, out_w, preprocess_workers)
    )
    dst[:OVERLAP].copy_(pre.to(device=dst.device, non_blocking=True), non_blocking=True)
    dst[OVERLAP:].copy_(prepared_new, non_blocking=True)
    del prepared_new



def _run_model(handle: VDAModelHandle, x: torch.Tensor):
    run = getattr(handle, "forward", None)
    if callable(run):
        return run(x)
    return handle.model(x)


def infer_window(
    handle: VDAModelHandle,
    raw_window: Sequence[torch.Tensor],
    session: LongVideoSession,
    input_size: int,
    fp32: bool,
    preprocess_workers: int = 1,
) -> np.ndarray:
    if len(raw_window) != INFER_LEN:
        raise ValueError(f"Expected {INFER_LEN} frames, got {len(raw_window)}")

    src_h, src_w = raw_window[0].shape[:2]
    out_h, out_w = transformed_hw(src_h, src_w, input_size)
    device = handle.ensure_device()

    cur_input = torch.empty(
        (1, INFER_LEN, 3, out_h, out_w),
        dtype=torch.float32,
        device=device,
    )
    _fill_window_input(
        cur_input[0], raw_window, session.pre_keyframes, out_h, out_w, preprocess_workers
    )

    with torch.inference_mode():
        with _autocast_context(device, fp32):
            depth = _run_model(handle, cur_input)

    # Keep the exact ten keyframes upstream VDA reuses on the next window.
    session.pre_keyframes = cur_input[:, KEYFRAMES].detach()

    depth_np = depth[0].float().cpu().numpy().astype(np.float32, copy=False)
    del depth, cur_input
    return depth_np


def infer_window_pipeline_pair(
    handle: VDAModelHandle,
    raw_window_a: Sequence[torch.Tensor],
    raw_window_b: Sequence[torch.Tensor],
    session: LongVideoSession,
    input_size: int,
    fp32: bool,
    preprocess_workers: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run two consecutive VDA windows as sequential B=1 forwards while
    overlapping window-B CPU preprocessing with window-A GPU inference.

    The model call shape and order are exactly the same as normal sequential
    inference. Only the independent OpenCV resize/normalize work for B's 22
    fresh frames is moved to one CPU worker while A is executing on the GPU.
    """
    if len(raw_window_a) != INFER_LEN or len(raw_window_b) != INFER_LEN:
        raise ValueError("Pipelined VDA inference requires two 32-frame windows")

    src_h, src_w = raw_window_a[0].shape[:2]
    if tuple(raw_window_b[0].shape[:2]) != (src_h, src_w):
        raise RuntimeError("VDA source resolution changed inside a pipelined pair")
    out_h, out_w = transformed_hw(src_h, src_w, input_size)
    device = handle.ensure_device()

    # Construct A exactly like infer_window().
    input_a = torch.empty(
        (1, INFER_LEN, 3, out_h, out_w),
        dtype=torch.float32,
        device=device,
    )
    _fill_window_input(
        input_a[0], raw_window_a, session.pre_keyframes, out_h, out_w, preprocess_workers
    )

    # B's temporal prefix comes from A's INPUT keyframes, exactly as in the
    # ordinary sequential path. The remaining 22 source frames are independent
    # CPU work, so start them before launching A's model forward.
    next_prefix = input_a[0, KEYFRAMES].detach()
    if int(preprocess_workers) == 1:
        prep_b_future = _VDA_PREPROCESS_EXECUTOR.submit(
            _frames_to_model_batch, raw_window_b[OVERLAP:], out_h, out_w
        )
    else:
        prep_b_future = _VDA_PREPROCESS_EXECUTOR.submit(
            _frames_to_model_batch, raw_window_b[OVERLAP:], out_h, out_w, preprocess_workers
        )

    with torch.inference_mode():
        with _autocast_context(device, fp32):
            depth_a = _run_model(handle, input_a)

    depth_a_np = depth_a[0].float().cpu().numpy().astype(np.float32, copy=False)
    del depth_a, input_a

    # By the time A's GPU work + D2H synchronization completes, B's CPU batch
    # should normally be ready. Even if it is not, waiting here preserves the
    # exact sequential order and model input.
    prepared_b = prep_b_future.result()
    input_b = torch.empty(
        (1, INFER_LEN, 3, out_h, out_w),
        dtype=torch.float32,
        device=device,
    )
    input_b[0, :OVERLAP].copy_(
        next_prefix.to(device=device, non_blocking=True), non_blocking=True
    )
    input_b[0, OVERLAP:].copy_(prepared_b, non_blocking=True)
    del prepared_b, next_prefix

    with torch.inference_mode():
        with _autocast_context(device, fp32):
            depth_b = _run_model(handle, input_b)

    session.pre_keyframes = input_b[:, KEYFRAMES].detach()
    depth_b_np = depth_b[0].float().cpu().numpy().astype(np.float32, copy=False)
    del depth_b, input_b
    return depth_a_np, depth_b_np


def infer_window_pair(
    handle: VDAModelHandle,
    raw_window_a: Sequence[torch.Tensor],
    raw_window_b: Sequence[torch.Tensor],
    session: LongVideoSession,
    input_size: int,
    fp32: bool,
    preprocess_workers: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run two consecutive VDA windows in one B=2 model forward.

    This preserves the official 32/10 temporal input construction. The second
    window's ten cached positions come directly from the first window's *input*
    keyframes, exactly as upstream sequential inference does. Temporal depth
    scale/shift alignment remains sequential in `process_stream_batch`.
    """
    if len(raw_window_a) != INFER_LEN or len(raw_window_b) != INFER_LEN:
        raise ValueError("Paired VDA inference requires two 32-frame windows")

    src_h, src_w = raw_window_a[0].shape[:2]
    if tuple(raw_window_b[0].shape[:2]) != (src_h, src_w):
        raise RuntimeError("VDA source resolution changed inside a paired window")
    out_h, out_w = transformed_hw(src_h, src_w, input_size)
    device = handle.ensure_device()

    pair_input = torch.empty(
        (2, INFER_LEN, 3, out_h, out_w),
        dtype=torch.float32,
        device=device,
    )

    # Window A uses the session prefix from the previous sequential window.
    _fill_window_input(
        pair_input[0], raw_window_a, session.pre_keyframes, out_h, out_w, preprocess_workers
    )

    # Window B's prefix is derived from window A's *input* tensor. This is known
    # before either depth prediction exists, so both model calls are independent
    # and can safely share one leading batch dimension.
    next_prefix = pair_input[0, KEYFRAMES].detach()
    _fill_window_input(
        pair_input[1], raw_window_b, next_prefix, out_h, out_w, preprocess_workers
    )

    with torch.inference_mode():
        with _autocast_context(device, fp32):
            depth = _run_model(handle, pair_input)

    if depth.ndim != 4 or depth.shape[0] != 2 or depth.shape[1] != INFER_LEN:
        raise RuntimeError(
            f"Unexpected paired VDA output shape: {tuple(depth.shape)}"
        )

    # After two consecutive windows, the next session prefix must come from B.
    session.pre_keyframes = pair_input[1:2, KEYFRAMES].detach()

    depth_np = depth.float().cpu().numpy().astype(np.float32, copy=False)
    out_a = depth_np[0]
    out_b = depth_np[1]
    del depth, pair_input
    return out_a, out_b


def _padded_window_at(
    buffer: List[torch.Tensor],
    offset: int,
    last_source_frame: torch.Tensor,
) -> Tuple[List[torch.Tensor], int]:
    available = max(0, len(buffer) - int(offset))
    actual = min(available, INFER_LEN)
    frames = list(buffer[offset : offset + actual])
    if actual == 0:
        frames = [last_source_frame]
    while len(frames) < INFER_LEN:
        frames.append(last_source_frame)
    return frames, actual

def _padded_window(
    buffer: List[torch.Tensor],
    last_source_frame: torch.Tensor,
) -> Tuple[List[torch.Tensor], int]:
    actual = min(len(buffer), INFER_LEN)
    frames = list(buffer[:actual])
    if actual == 0:
        frames = [last_source_frame]
        actual = 0
    while len(frames) < INFER_LEN:
        frames.append(last_source_frame)
    return frames, actual


def _actual_count(total: int, global_start: int, requested: int) -> int:
    return max(0, min(requested, total - global_start))


def _is_oom_error(exc: BaseException) -> bool:
    name = exc.__class__.__name__.lower()
    text = str(exc).lower()
    return (
        "outofmemory" in name
        or "out of memory" in text
        or "hip out of memory" in text
        or "cuda out of memory" in text
    )



def _coalesce_rgb_frames(frames: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    """Return zero-copy batched views for consecutive frame views when possible.

    Comfy/VHS RGB frames normally arrive as unbound views of one BHWC tensor.
    The historical path treated every frame as an independent source and then
    `torch.stack` copied 44 individual tensors.  This helper detects adjacent
    views that share storage/shape/stride and exposes them as one [N,H,W,C]
    as_strided view.  No pixel values are changed.
    """
    if not frames:
        return []

    out: List[torch.Tensor] = []
    i = 0
    n = len(frames)
    while i < n:
        first = frames[i]
        if first.ndim != 3 or first.device.type != "cpu":
            out.append(first.unsqueeze(0))
            i += 1
            continue

        try:
            storage_ptr = first.untyped_storage().data_ptr()
        except Exception:
            out.append(first.unsqueeze(0))
            i += 1
            continue

        shape = tuple(first.shape)
        stride = tuple(first.stride())
        dtype = first.dtype
        base_off = int(first.storage_offset())
        step = None
        j = i + 1
        while j < n:
            cur = frames[j]
            try:
                same = (
                    cur.ndim == 3
                    and cur.device.type == "cpu"
                    and cur.dtype == dtype
                    and tuple(cur.shape) == shape
                    and tuple(cur.stride()) == stride
                    and cur.untyped_storage().data_ptr() == storage_ptr
                )
            except Exception:
                same = False
            if not same:
                break
            delta = int(cur.storage_offset()) - int(frames[j - 1].storage_offset())
            if delta <= 0:
                break
            if step is None:
                step = delta
            elif delta != step:
                break
            j += 1

        count = j - i
        if count == 1 or step is None:
            out.append(first.unsqueeze(0))
        else:
            chunk = torch.as_strided(
                first,
                size=(count, *shape),
                stride=(step, *stride),
                storage_offset=base_off,
            )
            out.append(chunk)
        i = j
    return out


def _materialize_rgb_frames(frames: Sequence[torch.Tensor]) -> torch.Tensor:
    """Materialize ordered RGB frames with the fewest large CPU copies.

    If all frames are one contiguous storage run, return the zero-copy batch
    view directly. Otherwise concatenate a small number of coalesced chunks.
    This is value-identical to `torch.stack(frames, dim=0)`.
    """
    chunks = _coalesce_rgb_frames(frames)
    if not chunks:
        raise ValueError("Cannot materialize an empty RGB frame sequence")
    if len(chunks) == 1:
        only = chunks[0]
        return only if only.is_contiguous() else only.contiguous()
    return torch.cat(chunks, dim=0).contiguous()


def _own_rgb_frames(frames: Sequence[torch.Tensor]) -> torch.Tensor:
    """Create compact owned RGB storage while still coalescing sources.

    Persistent session state must NEVER keep a narrow view into an incoming VHS
    batch because that pins the full batch allocation. Even when all requested
    frames form one contiguous slice, clone that slice into independent storage.
    """
    chunks = _coalesce_rgb_frames(frames)
    if not chunks:
        raise ValueError("Cannot own an empty RGB frame sequence")
    if len(chunks) == 1:
        return chunks[0].clone().contiguous()
    return torch.cat(chunks, dim=0).contiguous()

def process_stream_batch(
    handle: VDAModelHandle,
    images: torch.Tensor,
    session: LongVideoSession,
    input_size: int,
    fp32: bool,
    is_final_input: bool,
    window_mode: str = "sequential",
    preprocess_workers: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Process one VHS input batch and return temporally-final RGB/depth frames.

    The function reproduces the official 32/10 VDA reuse and scale/shift
    alignment while delaying only the eight frames that the next window can
    retroactively blend. Returned RGB is delayed in lockstep with depth, so the
    stereo stage always receives matching frames.
    """

    if images.ndim != 4 or images.shape[-1] < 3:
        raise ValueError("Expected ComfyUI IMAGE tensor [B,H,W,C]")

    cpu_images = images.detach().cpu()[..., :3]
    for i in range(cpu_images.shape[0]):
        session.raw_buffer.append(cpu_images[i])
        session.last_source_frame = cpu_images[i]
    session.input_count += int(cpu_images.shape[0])

    if session.total_frames is None and is_final_input:
        session.total_frames = session.input_count

    total = session.total_frames
    if total is None:
        total = session.input_count if is_final_input else math.inf

    emitted_rgb: List[torch.Tensor] = []
    emitted_depth: List[np.ndarray] = []

    mode = str(window_mode or "sequential")
    if mode not in ("sequential", "pipeline_2", "pair_2"):
        raise ValueError(f"Unsupported VDA window_mode: {mode}")
    prefetched: List[Tuple[List[torch.Tensor], int, np.ndarray]] = []

    while True:
        start = session.next_window_start
        if math.isfinite(total) and start >= int(total):
            break

        enough_for_full = len(session.raw_buffer) >= INFER_LEN
        if not enough_for_full and not is_final_input:
            break
        if not enough_for_full and is_final_input and session.last_source_frame is None:
            break

        # If the final stream has <=22 total frames, upstream VDA runs only one
        # window. For larger streams every 22-frame start < total gets a window.
        if is_final_input and math.isfinite(total):
            if start >= int(total):
                break
        elif len(session.raw_buffer) < INFER_LEN:
            break

        if prefetched:
            raw_window, actual_buffered, depth = prefetched.pop(0)
        else:
            raw_window, actual_buffered = _padded_window(
                session.raw_buffer, session.last_source_frame
            )

            total_i_for_pair = int(total) if math.isfinite(total) else None
            second_start = start + STEP
            second_exists = (
                True if total_i_for_pair is None else second_start < total_i_for_pair
            )
            if is_final_input:
                second_ready = second_exists and (
                    len(session.raw_buffer) > STEP
                    or session.last_source_frame is not None
                )
            else:
                second_ready = second_exists and (
                    len(session.raw_buffer) >= INFER_LEN + STEP
                )

            pipeline_enabled = mode == "pipeline_2" and second_ready
            pair_enabled = (
                mode == "pair_2"
                and not bool(getattr(session, "vda_pair_disabled", False))
                and second_ready
            )

            if pipeline_enabled:
                raw_window_b, actual_buffered_b = _padded_window_at(
                    session.raw_buffer, STEP, session.last_source_frame
                )
                depth_a, depth_b = infer_window_pipeline_pair(
                    handle, raw_window, raw_window_b, session, input_size, fp32, preprocess_workers
                )
                depth = depth_a
                prefetched.append((raw_window_b, actual_buffered_b, depth_b))
                if not bool(getattr(session, "vda_pipeline_announced", False)):
                    print(
                        "[LongVideo VDA] pipelining next-window CPU preprocessing "
                        "under the current B=1 GPU forward"
                    )
                    session.vda_pipeline_announced = True
            elif pair_enabled:
                raw_window_b, actual_buffered_b = _padded_window_at(
                    session.raw_buffer, STEP, session.last_source_frame
                )
                try:
                    depth_a, depth_b = infer_window_pair(
                        handle, raw_window, raw_window_b, session, input_size, fp32, preprocess_workers
                    )
                    depth = depth_a
                    prefetched.append((raw_window_b, actual_buffered_b, depth_b))
                    if not bool(getattr(session, "vda_pair_announced", False)):
                        print(
                            "[LongVideo VDA] paired two consecutive 32-frame windows "
                            "in one B=2 model forward"
                        )
                        session.vda_pair_announced = True
                except BaseException as exc:
                    if not _is_oom_error(exc):
                        raise
                    session.vda_pair_disabled = True
                    print(
                        "[LongVideo VDA] pair_2 exceeded available VRAM; "
                        "falling back to sequential windows for this session"
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    depth = (
                        infer_window(handle, raw_window, session, input_size, fp32)
                        if int(preprocess_workers) == 1
                        else infer_window(handle, raw_window, session, input_size, fp32, preprocess_workers)
                    )
            else:
                depth = (
                    infer_window(handle, raw_window, session, input_size, fp32)
                    if int(preprocess_workers) == 1
                    else infer_window(handle, raw_window, session, input_size, fp32, preprocess_workers)
                )

        total_i = int(total) if math.isfinite(total) else None
        has_next = True
        if total_i is not None:
            has_next = (start + STEP) < total_i

        if session.window_index == 0:
            # Official first-window reference depths are indices 0 and 12.
            session.ref_align = [depth[KF_ALIGN[0]].copy(), depth[KF_ALIGN[1]].copy()]

            if not has_next:
                n = _actual_count(total_i, start, INFER_LEN) if total_i is not None else actual_buffered
                emitted_depth.extend([d.copy() for d in depth[:n]])
                emitted_rgb.extend(raw_window[:n])
            else:
                stable_n = 24
                if total_i is not None:
                    stable_n = min(stable_n, _actual_count(total_i, start, 24))
                emitted_depth.extend([d.copy() for d in depth[:stable_n]])
                emitted_rgb.extend(raw_window[:stable_n])

                pending_n = INTERP_LEN
                if total_i is not None:
                    pending_n = min(
                        pending_n,
                        _actual_count(total_i, start + 24, INTERP_LEN),
                    )
                session.pending_depth = [d.copy() for d in depth[24 : 24 + pending_n]]
                session.pending_rgb = [f for f in raw_window[24 : 24 + pending_n]]
        else:
            curr_align = [depth[0], depth[1]]
            ref = session.ref_align
            if len(ref) != ALIGN_LEN:
                raise RuntimeError("VDA alignment reference state is invalid")
            prediction = np.concatenate(curr_align, axis=0)
            target = np.concatenate(ref, axis=0)
            mask = np.ones_like(target, dtype=bool)
            scale, shift = compute_scale_and_shift(prediction, target, mask)
            aligned = depth * scale + shift
            np.maximum(aligned, 0, out=aligned)

            # The current window indices 2:10 are predictions for the previous
            # window's keyframes 24:32. Blend them into the delayed previous tail.
            if session.pending_depth:
                post_full = [aligned[i] for i in range(2, 10)]
                # Compute the official eight-frame weights, then keep only the
                # real (non-padding) pending frames for short final clips.
                padded_pre = list(session.pending_depth)
                while len(padded_pre) < INTERP_LEN:
                    padded_pre.append(padded_pre[-1])
                blended_full = interpolate_frames(padded_pre[:INTERP_LEN], post_full)
                real_pending = len(session.pending_depth)
                emitted_depth.extend([x.copy() for x in blended_full[:real_pending]])
                emitted_rgb.extend(session.pending_rgb[:real_pending])

            # Official ref_align keeps the very first reference and refreshes the
            # second reference with current index 12 after scale/shift.
            session.ref_align = [session.ref_align[0], aligned[12].copy()]

            # New source frames represented by this window are indices 10:32.
            if has_next:
                stable_new = 14  # 10:24; 24:32 must remain delayable
                if total_i is not None:
                    stable_new = min(
                        stable_new,
                        _actual_count(total_i, start + 10, 14),
                    )
                emitted_depth.extend([d.copy() for d in aligned[10 : 10 + stable_new]])
                emitted_rgb.extend(raw_window[10 : 10 + stable_new])

                pending_n = INTERP_LEN
                if total_i is not None:
                    pending_n = min(
                        pending_n,
                        _actual_count(total_i, start + 24, INTERP_LEN),
                    )
                session.pending_depth = [
                    d.copy() for d in aligned[24 : 24 + pending_n]
                ]
                session.pending_rgb = [
                    f for f in raw_window[24 : 24 + pending_n]
                ]
            else:
                new_n = (
                    _actual_count(total_i, start + 10, STEP)
                    if total_i is not None
                    else max(0, actual_buffered - 10)
                )
                emitted_depth.extend([d.copy() for d in aligned[10 : 10 + new_n]])
                emitted_rgb.extend(raw_window[10 : 10 + new_n])
                session.pending_depth = []
                session.pending_rgb = []

        session.window_index += 1
        session.next_window_start += STEP

        # Advance only actual source frames from the buffer. At EOF the padded
        # repeats are synthetic and never enter session.raw_buffer.
        consume = min(STEP, len(session.raw_buffer))
        if consume:
            del session.raw_buffer[:consume]

        if not has_next:
            break

        if (
            not is_final_input
            and len(session.raw_buffer) < INFER_LEN
            and not prefetched
        ):
            break

    # Compact persistent RGB state before returning to ComfyUI.
    #
    # Important ownership detail: `cpu_images[i]` is a view into the incoming
    # VHS batch tensor. Keeping even ONE such view in the session (notably
    # `last_source_frame`) pins the storage for the entire multi-frame batch.
    # At 4K x 44 float32 frames that can be several GiB.  Also, once VDA has a
    # cached temporal prefix, raw_window[0:10] is never consumed as source RGB:
    # infer_window replaces those positions with pre_keyframes, and the stream
    # emits only raw_window[10:32] for new RGB. The previous 8-frame tail that
    # still needs RGB is already tracked independently in pending_rgb.
    #
    # Therefore keep exact copies only for the still-live raw tail, represent
    # the dead ten-position prefix with aliases to one live frame (shape only),
    # and make last_source_frame alias owned compact storage whenever possible.
    if session.raw_buffer:
        if session.pre_keyframes is not None and len(session.raw_buffer) > OVERLAP:
            live_src = session.raw_buffer[OVERLAP:]

            # The delayed eight RGB frames and the still-live raw tail are
            # usually adjacent views into the same incoming VHS batch. Own them
            # in ONE compact allocation, then alias both session states into it.
            # This preserves the v1.6 memory fix while removing one large
            # allocation/copy from every steady meta-batch.
            pending_n = len(session.pending_rgb)
            persistent_src = list(session.pending_rgb) + list(live_src)
            persistent_owned = _own_rgb_frames(persistent_src)
            persistent_frames = list(persistent_owned.unbind(0))

            if pending_n:
                session.pending_rgb = persistent_frames[:pending_n]
            else:
                session.pending_rgb = []

            live_frames = persistent_frames[pending_n:]
            if live_frames:
                placeholder = live_frames[0]
                session.raw_buffer = [placeholder] * OVERLAP + live_frames
            else:
                # Defensive short/final-window case. The source prefix is dead
                # once pre_keyframes exists, so keep one owned frame only.
                fallback = persistent_frames[-1]
                session.raw_buffer = [fallback] * OVERLAP
        else:
            owned = _own_rgb_frames(session.raw_buffer)
            session.raw_buffer = list(owned.unbind(0))
            if session.pending_rgb:
                pending_owned = _own_rgb_frames(session.pending_rgb)
                session.pending_rgb = list(pending_owned.unbind(0))

        # The newest source frame is normally the last retained raw frame. Point
        # at that owned storage rather than retaining a view into the VHS batch.
        session.last_source_frame = session.raw_buffer[-1]
    elif session.last_source_frame is not None:
        # EOF/short-clip fallback: own exactly one frame, never the full batch.
        session.last_source_frame = session.last_source_frame.clone().contiguous()
        if session.pending_rgb:
            pending_owned = _own_rgb_frames(session.pending_rgb)
            session.pending_rgb = list(pending_owned.unbind(0))

    if is_final_input and session.total_frames is not None:
        if session.next_window_start >= session.total_frames or session.total_frames <= STEP:
            session.finished = True

    if emitted_rgb:
        # Coalesce chronological frame views into a few large storage runs. In
        # the steady 44-frame path this is normally previous pending RGB + one
        # current-batch run, avoiding 44-source `torch.stack` bookkeeping.
        rgb_out = _materialize_rgb_frames(emitted_rgb)
        depth_out = torch.from_numpy(np.stack(emitted_depth, axis=0)).contiguous()
        session.emitted_frames += int(rgb_out.shape[0])
    else:
        h, w = images.shape[1], images.shape[2]
        rgb_out = torch.empty((0, h, w, 3), dtype=images.dtype)
        # Depth dimensions are unknown until the first inference. Empty custom
        # depth is represented as [0,1,1].
        depth_out = torch.empty((0, 1, 1), dtype=torch.float32)

    return rgb_out, depth_out
