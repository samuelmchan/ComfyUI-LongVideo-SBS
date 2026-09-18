#!/usr/bin/env python3
"""Run a real 32-frame VDA Base inference through the custom node backend.

This intentionally uses vda_backend.load_vda_model(..., attention_backend='pytorch_sdpa')
so the test exercises the same ROCm-safe attention patch used by the ComfyUI node.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vda_backend import load_vda_model  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=["vits", "vitb", "vitl"], default="vitb")
    ap.add_argument("--size", type=int, default=518)
    ap.add_argument("--frames", type=int, default=32)
    args = ap.parse_args()

    print("Torch:", torch.__version__)
    print("HIP:", torch.version.hip)
    print("GPU available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise SystemExit("FAIL: no ROCm/CUDA-compatible PyTorch device")
    print("GPU:", torch.cuda.get_device_name(0))

    print(f"Loading VDA {args.encoder} through LongVideo-SBS backend...")
    handle = load_vda_model(
        encoder=args.encoder,
        attention_backend="pytorch_sdpa",
    )
    model = handle.model

    # Confirm both independent xFormers paths were patched.
    from video_depth_anything.dinov2_layers.attention import MemEffAttention
    from video_depth_anything.motion_module.motion_module import TemporalAttention

    dino_patched = bool(getattr(MemEffAttention, "_lv_sbs_sdpa_patched", False))
    temporal_patched = bool(getattr(TemporalAttention, "_lv_sbs_sdpa_patched", False))
    print("DINO PyTorch SDPA patch active:", dino_patched)
    print("Temporal PyTorch SDPA patch active:", temporal_patched)
    if not (dino_patched and temporal_patched):
        raise SystemExit("FAIL: both SDPA patches were not applied")

    dtype = torch.float16
    x = torch.randn(
        1, args.frames, 3, args.size, args.size,
        device=handle.device,
        dtype=dtype,
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(f"Starting inference: shape={tuple(x.shape)}, dtype={x.dtype}")
    t0 = time.perf_counter()

    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            y = model(x)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    print("Inference: PASS")
    print("Output shape:", tuple(y.shape))
    print("Elapsed seconds:", round(elapsed, 3))
    print("Current allocated GiB:", round(torch.cuda.memory_allocated() / 2**30, 3))
    print("Peak allocated GiB:", round(torch.cuda.max_memory_allocated() / 2**30, 3))
    print("Peak reserved GiB:", round(torch.cuda.max_memory_reserved() / 2**30, 3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
