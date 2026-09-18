# Third-party components

This package integrates with, but does not bundle, these projects:

- Video Depth Anything — https://github.com/DepthAnything/Video-Depth-Anything
- ComfyUI VideoHelperSuite — https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite
- ComfyUI — https://github.com/Comfy-Org/ComfyUI
- RIFE model checkpoints / Practical-RIFE compatible weights — supplied separately by the user

The v3.5 DEV interpolation prototype uses ComfyUI core's frame-interpolation model detector/runtime interfaces. No RIFE checkpoint is bundled in this archive.

No VDA checkpoint is included in this archive. The install helper downloads the checkpoint selected by the user from the upstream Hugging Face repository.
