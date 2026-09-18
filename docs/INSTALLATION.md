# Installation

This project intentionally avoids managing the shared ComfyUI GPU stack. Do not replace a working PyTorch/ROCm environment just to install this node pack.

## 1. Install the custom node

From `ComfyUI/custom_nodes`:

```bash
git clone https://github.com/samuelmchan/ComfyUI-LongVideo-SBS.git
cd ComfyUI-LongVideo-SBS
```

If you previously used the old split package, remove the stale sibling before restarting:

```bash
rm -rf ../ComfyUI-LongVideo-SBS-Upgrade
```

## 2. Use the same Python environment as ComfyUI

Examples:

```bash
PY=/path/to/ComfyUI/venv/bin/python
# or
# PY=/path/to/ComfyUI/.venv/bin/python
```

Verify it:

```bash
"$PY" - <<'PY'
import sys, torch
print(sys.executable)
print("torch", torch.__version__)
print("hip", torch.version.hip)
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
PY
```

## 3. Install package-owned Python dependencies

```bash
"$PY" -m pip install -r /path/to/ComfyUI/custom_nodes/ComfyUI-LongVideo-SBS/requirements.txt
```

The requirements file owns only lightweight dependencies. It deliberately does not install or replace Torch, torchvision, xFormers, NumPy, OpenCV, or the GPU runtime.

## 4. Install ComfyUI-VideoHelperSuite

The production workflow uses VHS meta-batching and video-info types. Install/update ComfyUI-VideoHelperSuite through your normal ComfyUI Manager or git workflow.

## 5. Install Video Depth Anything Large

From the ComfyUI root:

```bash
COMFYUI_ROOT="$PWD" custom_nodes/ComfyUI-LongVideo-SBS/install_vda.sh vitl
```

Default location:

```text
ComfyUI/models/video_depth_anything/Video-Depth-Anything/
└── checkpoints/video_depth_anything_vitl.pth
```

You may also point the VDA loader at another repo/checkpoint explicitly.

## 6. Install RIFE

The production path is validated with a RIFE 4.25-compatible checkpoint. Preferred location:

```text
ComfyUI/models/frame_interpolation/
```

Recognized legacy locations include:

```text
ComfyUI/models/rife/
ComfyUI/models/vfi/rife/
```

`LV_RIFEModelLoader` in `auto` mode prefers 4.25 when available.

## 7. FFmpeg / VAAPI

Verify FFmpeg:

```bash
ffmpeg -version
ffprobe -version
```

For the shipped hardware profile, verify AV1 VAAPI and your render node:

```bash
ffmpeg -hide_banner -encoders | grep av1_vaapi
ls -l /dev/dri/renderD*
vainfo
```

The production workflow defaults to `/dev/dri/renderD128`. If your render node differs, change both the loader and encoder nodes.

Hardware decode can be disabled in the loader to use FFmpeg software decoding.

## 8. Preflight

From the repository root:

```bash
"$PY" dependency_check.py
```

Fix `[FAIL]` items before running the production workflow.

## 9. Load the workflow

Use:

```text
workflows/LongVideo-SBS-Quest3-Production.json
```

Select your source video in `LV_ProductionVideoLoader`; the distributed workflow uses `input.mp4` only as a neutral placeholder.

## 10. Expected startup line

After restarting ComfyUI:

```text
[LongVideo SBS] loaded v3.8.0 DEV3 (A/B defaults + explicit AV1 CQP)
```
