# Changelog

## v3.8.0 DEV3 — A/B Defaults + Explicit AV1 CQP

- Promoted the latest A/B-tested workflow values:
  - stereo strength 1.40%;
  - zero parallax 0.00;
  - depth range scale 1.00;
  - temporal disparity stability 0.15;
  - feathering off;
  - hardware decode on;
  - AV1 VAAPI 10-bit VBR 100/135/280 Mbps.
- Added explicit `cqp` rate control for `av1_vaapi`.
- Preserved `av1_vaapi + constant_quality` as a legacy alias for CQP.
- AV1 CQP uses `-rc_mode CQP -global_quality <av1_qp>` and omits VBR bitrate flags.
- Added clearer q_idx/global_quality UI help.
- Cleaned stale linked zero-parallax placeholders in the production workflow.
- Removed unrelated RenderTime state and source-specific metadata from the public workflow.
- Cleaned repository layout and documentation for GitHub publication.

## Earlier development

The project evolved through the v3.3–v3.7 production/interpolation series and v3.8 comfort-control development. The public repository starts from the consolidated current implementation rather than carrying every intermediate workflow and release-note artifact in the top-level tree.
