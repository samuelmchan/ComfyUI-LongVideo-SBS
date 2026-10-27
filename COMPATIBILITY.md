# ComfyUI compatibility

This pack reaches into ComfyUI internals that are **not** a stable public API —
an upgrade can rename, move, or alias them and break the pack at import time
with an obscure error. `scripts/check_comfyui_compat.py` exists so that is
caught deliberately instead of at render time.

## Validated combinations

| Pack version | ComfyUI | Verified | Notes |
|---|---|---|---|
| `3.8.0-dev3` | 0.42.5 | 2026-10-26 | local `llmserver` + remote `dailydriver` |

Add a row whenever the pack is verified against a new ComfyUI release. This
table is the answer to "does this pack work on the ComfyUI I just installed?"

## What we depend on

| Module | Symbols |
|---|---|
| `comfy.ldm.common_dit` | `pad_to_patch_size` |
| `comfy.model_management` | `get_torch_device`, `load_models_gpu` |
| `comfy.model_patcher` | `CoreModelPatcher` (an alias of `ModelPatcher`) |
| `comfy.utils` | `load_torch_file` |
| `comfy_extras.nodes_frame_interpolation` | `FrameInterpolationModelLoader` |
| `folder_paths` | `get_annotated_filepath`, `get_filename_list`, `get_full_path`, `get_input_directory`, `get_output_directory`, `get_save_image_path`, `get_temp_directory`, `models_dir` |
| `nodes` | `NODE_CLASS_MAPPINGS` |

`comfy_extras.nodes_frame_interpolation` is the highest-risk entry: core extras
are reorganized more freely than `comfy/` itself. `folder_paths.models_dir` is
also read directly by the VDA backend; keep the `getattr(..., "models_dir",
None)` guard in `frame_interpolation.py` and the direct use in `vda_backend.py`
in mind when editing either.

## Running the check

```sh
# Static: parse a ComfyUI source tree. No torch, no GPU, seconds to run.
python scripts/check_comfyui_compat.py --comfy-root /path/to/ComfyUI

# Runtime: import the real modules in the ComfyUI venv (most accurate).
cd /path/to/ComfyUI
venv/bin/python custom_nodes/ComfyUI-LongVideo-SBS/scripts/check_comfyui_compat.py --runtime
```

Both print the ComfyUI version, every missing symbol, and every pack-owned node
type from the production workflow that is not registered. Exit code 0 = clean.

## When ComfyUI is upgraded (procedure)

Run this on **each** deployment, in order:

1. Upgrade ComfyUI (see the deployment runbook for the pinned install steps).
2. `venv/bin/python custom_nodes/ComfyUI-LongVideo-SBS/scripts/check_comfyui_compat.py --runtime`
3. Start ComfyUI and confirm the log has no `IMPORT FAILED` for this pack, and
   that `LV_*` nodes are present in `/object_info`.
4. If anything is missing: fix the pack here, re-run, and only then update the
   second host.
5. Add a row to the table above, bump `__version__` in `__init__.py` if behaviour
   changed, and tag the release.

## CI

Not wired into CI. ComfyUI upgrades here are deliberate and infrequent, on two
known deployments, and the `--runtime` check above is strictly stronger than
anything CI could run (real imports against the real environment, rather than
AST parsing). So the check is run manually as part of the upgrade procedure
instead of on every push.

If this repo starts taking outside contributions, the static check is a cheap
pre-merge gate — a `--depth 1` ComfyUI clone plus parsing, no torch, a few
seconds per leg. It is not a substitute for the runtime check after an upgrade.
