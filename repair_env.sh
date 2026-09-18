#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${COMFYUI_ROOT:-}" ]]; then COMFY="$COMFYUI_ROOT";
elif [[ -d "$HERE/../../models" ]]; then COMFY="$(cd "$HERE/../.." && pwd)";
else echo "Set COMFYUI_ROOT=/path/to/ComfyUI" >&2; exit 1; fi
if [[ -n "${PYTHON_BIN:-}" ]]; then PY="$PYTHON_BIN";
elif [[ -x "$COMFY/venv/bin/python" ]]; then PY="$COMFY/venv/bin/python";
elif [[ -x "$COMFY/.venv/bin/python" ]]; then PY="$COMFY/.venv/bin/python";
else PY="$(command -v python || true)"; fi
[[ -n "$PY" && -x "$PY" ]] || { echo "Could not find ComfyUI Python" >&2; exit 1; }

echo "Repairing OpenCV/NumPy consistency with: $PY"
echo "Removing all mutually-exclusive OpenCV wheel variants first..."
"$PY" -m pip uninstall -y \
  opencv-python opencv-python-headless \
  opencv-contrib-python opencv-contrib-python-headless || true

echo "Installing the DA3-compatible shared versions..."
"$PY" -m pip install --force-reinstall \
  'numpy==1.26.4' \
  'opencv-python==4.11.0.86' \
  'tqdm==4.67.3'

echo
"$PY" - <<'PYINFO'
import numpy, cv2, tqdm
print("numpy:", numpy.__version__)
print("opencv:", cv2.__version__)
print("tqdm:", tqdm.__version__)
try:
    import torch
    print("torch:", torch.__version__)
    print("HIP:", torch.version.hip)
    print("GPU available:", torch.cuda.is_available())
except Exception as exc:
    print("torch check warning:", exc)
PYINFO

echo
echo "Dependency check:"
"$PY" -m pip check
