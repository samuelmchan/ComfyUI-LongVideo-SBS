# Arch Linux / ROCm notes

The reference development environment is Linux + AMD ROCm with VAAPI for decode/encode. These notes are intentionally conservative because ComfyUI environments are sensitive to Torch/ROCm version mismatches.

## Keep the working GPU stack intact

Do not let this custom node or Video Depth Anything replace your working:

- `torch`, `torchvision`, `torchaudio`;
- ROCm/HIP packages;
- NumPy/OpenCV;
- xFormers or attention packages used elsewhere in ComfyUI.

Install only this repository's `requirements.txt` into the Python environment that ComfyUI already uses.

## System video stack

You need a working FFmpeg + VAAPI stack. On Arch-derived systems, verify rather than blindly reinstalling:

```bash
ffmpeg -version
vainfo
ls -l /dev/dri/renderD*
ffmpeg -hide_banner -encoders | grep av1_vaapi
```

The production workflow defaults to:

```text
/dev/dri/renderD128
```

Change the workflow if your active VAAPI render node is different.

## ROCm/PyTorch check

From the ComfyUI Python environment:

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("hip", torch.version.hip)
print("available", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

## Decode troubleshooting

If a particular source shows decode corruption with VAAPI but software decode is clean, switch `Hardware Decode` off in `LV_ProductionVideoLoader`. The rest of the workflow can remain unchanged.
