# ComfyUI-LongVideo-SBS

Long-video **2D → depth → stereoscopic SBS → frame interpolation → encoded video** nodes for ComfyUI, built around Video Depth Anything, DIBR, RIFE, and streaming FFmpeg output.

The project is tuned for long-form VR video rather than short image batches: it keeps state across VHS meta-batches, uses bounded-memory streaming, preserves source cadence when interpolation is unnecessary, and can encode directly to 10-bit AV1 VAAPI.

> **Project status:** development release (`v3.8.0-dev3`). Primary validation is Linux + AMD ROCm/VAAPI. Other environments may work, but are not the reference configuration.

## Pipeline

```mermaid
flowchart LR
    A[Video Loader] --> B[Video Depth Anything]
    B --> C[Temporal Depth Normalize]
    C --> D[Edge-Aware Depth Smooth]
    D --> E[Stereo Comfort / Tuning]
    E --> F[Resolution-Normalized Stereo Controls]
    F --> G[Z-buffer DIBR]
    G --> H[Half-SBS Combine]
    H --> I[RIFE Streaming Interpolation]
    I --> J[Streaming FFmpeg Encoder]
```

## Highlights

- **Long-video streaming:** bounded batches instead of holding an entire video in VRAM/RAM.
- **Video Depth Anything Large** temporal depth inference.
- **Resolution-normalized stereo strength:** 3D strength is expressed as a percentage of eye-image width, so it scales with resolution.
- **Stereo comfort controls:** zero parallax, depth-range scaling, near/far compression, temporal stabilization, and optional edge feathering.
- **Z-buffer DIBR:** visibility-aware left/right rendering with hole filling.
- **RIFE 4.25 streaming:** state persists across meta-batches; interpolation can be manually disabled and automatically bypasses when source cadence is already effectively at target FPS.
- **Hardware/software decode toggle:** FFmpeg VAAPI or FFmpeg software decode.
- **Streaming encode:** AV1 VAAPI, x264, or x265 without buffering the whole result.
- **AV1 CQP + VBR:** explicit VAAPI CQP mode while preserving legacy workflows that used `constant_quality` for AV1.
- **Human-readable output names:** `output/halfsbs/<source> Half-SBS.mp4`.

## Current production workflow

Load:

```text
workflows/LongVideo-SBS-Quest3-Production.json
```

The current A/B-tested defaults are:

| Stage | Default |
|---|---|
| Batch size | 44 frames |
| Decode | Hardware decode ON (`/dev/dri/renderD128`) |
| VDA | `vitl`, input 518, FP16, `pipeline_2` |
| Depth normalize | 2nd–98th percentile, EMA 0.95 |
| Edge smoothing | strength 0.14, radius 4, edge protection 0.80 |
| Stereo strength | **1.40%** |
| Zero parallax | **0.00** |
| Depth range scale | **1.00** |
| Temporal disparity stability | **0.15** |
| Feathering | OFF |
| RIFE | ON, target 120 fps |
| Encoder | AV1 VAAPI, 10-bit |
| Rate control | **VBR 100 / 135 / 280 Mbps** |
| Output | `output/halfsbs/<source> Half-SBS.mp4` |

The workflow is shipped with a generic `input.mp4` placeholder; select your own source file in the loader.

## Stereo strength: what the percentage means

`stereo_strength_percent` is not an arbitrary slider. It is converted to maximum displacement per eye:

```text
max_disparity_eye_px = eye_width × stereo_strength_percent / 100
```

For a 1920-pixel-wide Half-SBS eye image:

```text
1.40% → 26.88 px/eye maximum disparity budget
```

This keeps relative stereo strength consistent as resolution changes.

## AV1 rate control: VBR vs CQP

The production workflow remains **VBR by default** because those settings are the current tested baseline.

For `av1_vaapi`, choose `cqp` to use fixed AV1 VAAPI quantization:

```text
-rc_mode CQP -global_quality <av1_qp>
```

`av1_qp` is AV1 VAAPI **q_idx / global_quality**, not the H.264/H.265 0–51 QP scale:

- lower = higher quality / larger files;
- higher = lower quality / smaller files;
- valid UI range: 1–255.

Legacy workflows using `constant_quality` with `av1_vaapi` still map to the same CQP path. CQP does **not** add `-b:v`, `-maxrate`, or `-bufsize`.

See [docs/ENCODING.md](docs/ENCODING.md).

## Installation

From your ComfyUI `custom_nodes` directory:

```bash
git clone https://github.com/samuelmchan/ComfyUI-LongVideo-SBS.git
cd ComfyUI-LongVideo-SBS
```

Install only the lightweight dependencies into the Python environment used by ComfyUI:

```bash
/path/to/ComfyUI/venv/bin/python -m pip install -r requirements.txt
```

Then install Video Depth Anything Large from the ComfyUI root:

```bash
COMFYUI_ROOT=/path/to/ComfyUI ./custom_nodes/ComfyUI-LongVideo-SBS/install_vda.sh vitl
```

You also need:

- a working ComfyUI GPU environment;
- ComfyUI-VideoHelperSuite;
- Video Depth Anything Large;
- a RIFE 4.25-compatible checkpoint in `ComfyUI/models/frame_interpolation/`;
- FFmpeg/ffprobe;
- for hardware decode/AV1 encode, working VAAPI and the correct `/dev/dri/renderD*` node.

Run the preflight check:

```bash
/path/to/ComfyUI/venv/bin/python dependency_check.py
```

Full details: [docs/INSTALLATION.md](docs/INSTALLATION.md). Arch/ROCm notes: [docs/ARCH_ROCM.md](docs/ARCH_ROCM.md).

## RIFE behavior

The current workflow targets 120 fps. If the source is already effectively at target cadence, the stream uses a 1× passthrough path instead of needlessly doubling frames. Near-120 rates such as 119.88 remain 119.88 rather than being relabeled to exactly 120.

You can also disable RIFE explicitly in `LV_RIFEModelLoader`.

## Output naming

With `filename_mode = source_name` and the production defaults:

```text
source:  espresso.mp4
output:  ComfyUI/output/halfsbs/espresso Half-SBS.mp4
```

Existing files are not overwritten; collisions receive a human-readable numeric suffix.

## Repository layout

```text
ComfyUI-LongVideo-SBS/
├── nodes.py
├── pipeline_support.py
├── frame_interpolation.py
├── stereo_renderer.py
├── video_loader.py
├── vda_backend.py
├── vda_stream.py
├── workflows/
│   └── LongVideo-SBS-Quest3-Production.json
├── docs/
├── tests/
└── scripts/
```

## Validation

Quick repository checks:

```bash
python -m compileall -q .
python scripts/validate_workflow.py
```

Inside a working ComfyUI/PyTorch environment:

```bash
PYTHONPATH=. python tests/test_dev3_cqp.py
PYTHONPATH=. python tests/test_frame_interpolation_state.py
```

The development package was also regression-tested against the historical loader, encoder, VDA, DIBR, and RIFE test chain before this repository cleanup.

## Notes

- The repository does **not** bundle VDA or RIFE checkpoints.
- It intentionally does not pin or replace PyTorch/ROCm, torchvision, xFormers, NumPy, or OpenCV. Those belong to the shared ComfyUI environment.
- AI super-resolution is intentionally outside this package; the expected production order is typically upscale first, then 2D→3D, then interpolation.

## License

ComfyUI-LongVideo-SBS is distributed under the **GNU General Public License v3.0 or later (GPL-3.0-or-later)**.

See [LICENSE](LICENSE) for the full license text.

Portions of the implementation are derived from or adapted from GPL-licensed ComfyUI components. Third-party software, models, checkpoints, and runtime dependencies retain their respective licenses.

See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for additional attribution and licensing information.

### Video Depth Anything model licensing

This repository does **not** distribute Video Depth Anything model weights.

At the time this documentation was written:

* **Video Depth Anything Small** weights are licensed under Apache-2.0.
* **Video Depth Anything Base** weights are licensed under CC BY-NC 4.0.
* **Video Depth Anything Large** weights are licensed under CC BY-NC 4.0.

The Base and Large model licenses therefore contain **non-commercial restrictions** that are separate from the GPL license covering this repository.

The license for ComfyUI-LongVideo-SBS does not override or expand the rights granted by third-party model licenses. Users are responsible for complying with the license applicable to the models and dependencies they use.
