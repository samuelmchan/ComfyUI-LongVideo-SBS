#!/usr/bin/env python3
"""Preflight the merged LongVideo SBS stable workflow without mutating the environment."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
COMFY = HERE.parent.parent
MODELS = COMFY / "models"
CUSTOM = COMFY / "custom_nodes"

ok = True
warnings = []


def pass_(label, detail=""):
    print(f"[PASS] {label}" + (f": {detail}" if detail else ""))


def fail(label, detail=""):
    global ok
    ok = False
    print(f"[FAIL] {label}" + (f": {detail}" if detail else ""))


def warn(label, detail=""):
    warnings.append(label)
    print(f"[WARN] {label}" + (f": {detail}" if detail else ""))


def import_version(modname, attr="__version__", required=True):
    try:
        mod = __import__(modname)
        ver = getattr(mod, attr, "unknown")
        pass_(modname, str(ver))
        return mod
    except Exception as exc:
        (fail if required else warn)(modname, repr(exc))
        return None


print("LongVideo SBS v3.8.0 DEV3 dependency preflight")
print("ComfyUI root:", COMFY)
print("Python:", sys.version.split()[0], sys.executable)

# Shared runtime. Do not attempt to repair these here.
torch = import_version("torch")
if torch is not None:
    hip = getattr(getattr(torch, "version", None), "hip", None)
    if hip:
        pass_("ROCm/HIP runtime", str(hip))
    else:
        warn("ROCm/HIP runtime", "torch.version.hip is empty; the shipped Quest 3 profile was validated on AMD ROCm")
    try:
        if torch.cuda.is_available():
            pass_("GPU available", torch.cuda.get_device_name(0))
        else:
            fail("GPU available", "torch.cuda.is_available() is False (ROCm uses the torch.cuda API too)")
    except Exception as exc:
        fail("GPU query", repr(exc))

np = import_version("numpy")
if np is not None:
    try:
        if int(str(np.__version__).split('.')[0]) >= 2:
            warn("NumPy compatibility", f"{np.__version__}; the validated VDA install path used NumPy < 2")
    except Exception:
        pass
import_version("cv2")
import_version("tqdm")
import_version("einops")
import_version("easydict")

# ComfyUI core capability used by the RIFE loader.
if (COMFY / "folder_paths.py").is_file():
    pass_("ComfyUI root marker", "folder_paths.py")
else:
    fail("ComfyUI root marker", f"missing {COMFY/'folder_paths.py'}")
if (COMFY / "comfy_extras" / "nodes_frame_interpolation.py").is_file():
    pass_("ComfyUI frame-interpolation loader", "comfy_extras/nodes_frame_interpolation.py")
else:
    fail("ComfyUI frame-interpolation loader", "update ComfyUI; LV_RIFEModelLoader requires FrameInterpolationModelLoader")

# VideoHelperSuite is a hard dependency for the shipped workflow/meta-batch interface.
vhs_dirs = []
if CUSTOM.is_dir():
    for p in CUSTOM.iterdir():
        n = p.name.lower()
        if p.is_dir() and ("videohelpersuite" in n or "video-helper-suite" in n):
            vhs_dirs.append(p)
if vhs_dirs:
    pass_("ComfyUI-VideoHelperSuite", ", ".join(p.name for p in vhs_dirs))
else:
    fail("ComfyUI-VideoHelperSuite", "required for VHS_BatchManager/VHS video-info and loader compatibility")

legacy = CUSTOM / "ComfyUI-LongVideo-SBS-Upgrade"
if legacy.exists():
    fail("legacy split companion removed", f"delete {legacy}; the merged LongVideo SBS package already contains those LV_* nodes")
else:
    pass_("legacy split companion removed")

# FFmpeg/VAAPI. FFmpeg is required; VAAPI is required for the shipped hardware profile,
# but software encoder/decoder selections can be used as a fallback.
ffmpeg = shutil.which("ffmpeg")
ffprobe = shutil.which("ffprobe")
if ffmpeg:
    pass_("ffmpeg", ffmpeg)
else:
    fail("ffmpeg", "not found in PATH")
if ffprobe:
    pass_("ffprobe", ffprobe)
else:
    fail("ffprobe", "not found in PATH")
if ffmpeg:
    try:
        out = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10).stdout
        if "av1_vaapi" in out:
            pass_("FFmpeg AV1 VAAPI encoder", "av1_vaapi")
        else:
            warn("FFmpeg AV1 VAAPI encoder", "not listed; shipped workflow will need a software encoder or another FFmpeg build")
    except Exception as exc:
        warn("FFmpeg encoder query", repr(exc))
render = Path("/dev/dri/renderD128")
if render.exists():
    pass_("VAAPI render node", str(render))
else:
    warn("VAAPI render node", f"{render} missing; change the workflow device or use software decode/encode")

# VDA repository and the Large checkpoint used by the stable workflow.
vda_repos = [
    MODELS / "video_depth_anything" / "Video-Depth-Anything",
    MODELS / "Video-Depth-Anything",
    HERE / "vendor" / "Video-Depth-Anything",
]
vda = next((p for p in vda_repos if (p / "video_depth_anything" / "video_depth.py").is_file()), None)
if vda:
    pass_("Video Depth Anything repository", str(vda))
    ckpt = vda / "checkpoints" / "video_depth_anything_vitl.pth"
    if ckpt.is_file():
        pass_("VDA Large checkpoint", str(ckpt))
    else:
        alt = MODELS / "video_depth_anything" / "video_depth_anything_vitl.pth"
        if alt.is_file():
            pass_("VDA Large checkpoint", str(alt))
        else:
            fail("VDA Large checkpoint", "run install_vda.sh vitl")
else:
    fail("Video Depth Anything repository", "run install_vda.sh vitl or set VDA_REPO")

# RIFE 4.25 is the stable candidate. Accept common model locations.
rife_candidates = []
for d in [MODELS / "frame_interpolation", MODELS / "rife", MODELS / "vfi" / "rife"]:
    if d.is_dir():
        rife_candidates.extend(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in {'.pth','.pt','.safetensors'})
rife425 = [p for p in rife_candidates if "425" in p.name.lower() or "4.25" in p.name.lower()]
if rife425:
    pass_("RIFE 4.25 checkpoint", str(rife425[0]))
elif rife_candidates:
    warn("RIFE 4.25 checkpoint", "other RIFE models were found, but the stable benchmark uses 4.25")
else:
    fail("RIFE checkpoint", "place RIFE 4.25 in ComfyUI/models/frame_interpolation/")

print()
if ok:
    print("RESULT: dependency preflight PASS" + (f" with {len(warnings)} warning(s)" if warnings else ""))
    raise SystemExit(0)
print("RESULT: dependency preflight FAIL; correct the [FAIL] items before using the shipped stable workflow")
raise SystemExit(1)
