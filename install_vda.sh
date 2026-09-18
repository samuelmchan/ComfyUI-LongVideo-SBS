#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-vitl}"
case "$MODEL" in
  vits|vitb|vitl) ;;
  *) echo "Usage: $0 [vits|vitb|vitl]" >&2; exit 2 ;;
esac

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${COMFYUI_ROOT:-}" ]]; then
  COMFY="$COMFYUI_ROOT"
elif [[ -d "$HERE/../../models" ]]; then
  COMFY="$(cd "$HERE/../.." && pwd)"
else
  echo "Set COMFYUI_ROOT=/path/to/ComfyUI before running this script." >&2
  exit 1
fi

# Prefer an explicit override, then the two common ComfyUI venv names,
# then the currently active Python. This avoids silently installing into
# the wrong environment.
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY="$PYTHON_BIN"
elif [[ -x "$COMFY/venv/bin/python" ]]; then
  PY="$COMFY/venv/bin/python"
elif [[ -x "$COMFY/.venv/bin/python" ]]; then
  PY="$COMFY/.venv/bin/python"
else
  PY="$(command -v python || true)"
fi

if [[ -z "$PY" || ! -x "$PY" ]]; then
  echo "Could not find a usable Python interpreter." >&2
  echo "Set PYTHON_BIN=/path/to/ComfyUI/venv/bin/python and retry." >&2
  exit 1
fi

echo "Using Python: $PY"
"$PY" - <<'PYINFO'
import sys
print("Python:", sys.version.split()[0])
try:
    import torch
    print("PyTorch:", torch.__version__)
    print("ROCm/HIP:", torch.version.hip)
except Exception as exc:
    print("PyTorch check warning:", exc)
PYINFO

DEST="$COMFY/models/video_depth_anything"
REPO="$DEST/Video-Depth-Anything"
mkdir -p "$DEST"

if [[ ! -d "$REPO/.git" ]]; then
  git clone --depth 1 https://github.com/DepthAnything/Video-Depth-Anything.git "$REPO"
else
  git -C "$REPO" pull --ff-only
fi
mkdir -p "$REPO/checkpoints"

# Upstream VDA imports `utils.util` as a top-level package. In large ComfyUI
# environments another installed `utils` module can shadow VDA's sibling
# utils/ directory. Mark VDA's utils directory as an explicit package so the
# repo-local import wins when the repository is placed first on sys.path.
if [[ -d "$REPO/utils" ]]; then
  touch "$REPO/utils/__init__.py"
fi

case "$MODEL" in
  vits)
    URL="https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth"
    FILE="video_depth_anything_vits.pth"
    ;;
  vitb)
    URL="https://huggingface.co/depth-anything/Video-Depth-Anything-Base/resolve/main/video_depth_anything_vitb.pth"
    FILE="video_depth_anything_vitb.pth"
    ;;
  vitl)
    URL="https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth"
    FILE="video_depth_anything_vitl.pth"
    ;;
esac

if [[ ! -f "$REPO/checkpoints/$FILE" ]]; then
  curl -L --fail --retry 3 "$URL" -o "$REPO/checkpoints/$FILE"
fi

# Do not use VDA's old pinned torch/xformers stack, and do not mutate
# shared NumPy/OpenCV/tqdm packages in an existing ComfyUI environment.
# Validate the shared runtime first; install only VDA-specific lightweight deps.
"$PY" - <<'PYDEPS'
import sys
errors=[]
try:
    import numpy as np
    major=int(np.__version__.split('.')[0])
    print("NumPy:", np.__version__)
    if major >= 2:
        errors.append("NumPy >=2 detected; current Depth Anything 3 requires numpy<2")
except Exception as exc:
    errors.append(f"NumPy import failed: {exc}")
try:
    import cv2
    print("OpenCV:", cv2.__version__)
except Exception as exc:
    errors.append(f"OpenCV/cv2 import failed: {exc}")
try:
    import tqdm
    print("tqdm:", tqdm.__version__)
except Exception as exc:
    errors.append(f"tqdm import failed: {exc}")
if errors:
    print("\nShared ComfyUI dependency check failed:", file=sys.stderr)
    for e in errors:
        print(" -", e, file=sys.stderr)
    print("Fix the shared environment first; this installer will not change NumPy/OpenCV/tqdm automatically.", file=sys.stderr)
    raise SystemExit(1)
PYDEPS

"$PY" -m pip install 'einops>=0.7' 'easydict>=1.10'

echo
printf 'Installed VDA repo: %s\n' "$REPO"
printf 'Installed checkpoint: %s\n' "$REPO/checkpoints/$FILE"
echo "Shared NumPy/OpenCV/tqdm were left unchanged."
echo "Do NOT pip-install VDA's pinned torch/xformers requirements over your ROCm ComfyUI environment."
