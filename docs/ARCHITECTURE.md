# Architecture

## Long-video design

The pipeline is built around VHS meta-batching. Stateful components retain the information needed across batches while keeping frame tensors bounded.

The main production path is:

1. `LV_ProductionVideoLoader` — FFmpeg hardware/software decode and source metadata.
2. `LV_VDAModelLoader` + `LV_VDAStream` — Video Depth Anything temporal inference.
3. `LV_DepthTemporalNormalize` — temporally stable percentile normalization.
4. `LV_EdgeAwareDepthSmooth` — controlled spatial depth smoothing.
5. `LV_DepthStereoTuning` — depth-range/compression/temporal/edge comfort controls.
6. `LV_ResolutionStereoControls` — converts percent-of-eye-width stereo strength to pixels.
7. `LV_StereoDIBR` — z-buffer left/right rendering and hole filling.
8. `LV_SBSCombine` — stereo packing.
9. `LV_LongVideoFrameInterpolationStream` — stateful RIFE interpolation or cadence-aware passthrough.
10. `LV_StreamingVideoEncoder` — bounded streaming encode and optional audio remux.

## Stereo controls

`zero_parallax` is the shared pivot for depth-range scaling and DIBR. Linked workflow widgets retain their positional serialized values to avoid ComfyUI widget-index drift.

`temporal_disparity_stability` smooths the tuned depth immediately before DIBR. It is depth-space stabilization rather than post-render disparity-map filtering.

## RIFE

The interpolation layer keeps temporal state across meta-batches. Production optimizations include adjacent feature reuse, warp-grid reuse, pair batching, and asynchronous pinned transfers where supported.

## Output

The streaming encoder consumes the interpolation stream directly. It does not require materializing the full output sequence in memory.
