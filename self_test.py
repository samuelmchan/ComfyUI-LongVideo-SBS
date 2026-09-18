#!/usr/bin/env python3
"""Environment/self-test for the LongVideo SBS node pack."""
from pathlib import Path
import os
import sys
import torch

print("Python:", sys.version.split()[0])
print("Torch:", torch.__version__)
print("torch.version.hip:", torch.version.hip)
print("torch.cuda.is_available():", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
    print("VRAM GiB:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))

root = Path(__file__).resolve().parent
repo = os.environ.get("VDA_REPO", "")
if repo:
    print("VDA_REPO:", repo)
else:
    comfy_root = root.parent.parent
    guess = comfy_root / "models" / "video_depth_anything" / "Video-Depth-Anything"
    print("VDA repo guess:", guess, "OK" if guess.exists() else "MISSING")

if not torch.cuda.is_available():
    raise SystemExit("FAIL: ROCm/CUDA-compatible PyTorch device is not visible")
if torch.version.hip is None:
    print("WARNING: torch.version.hip is None. This is not a ROCm PyTorch build.")
else:
    print("ROCm PyTorch detected: PASS")
