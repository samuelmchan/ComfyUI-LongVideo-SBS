from __future__ import annotations

import json
from pathlib import Path

import pipeline_support as ps
from pipeline_support import (
    ENCODER_RATE_CONTROLS,
    LVDepthStereoTuning,
    LVResolutionStereoControls,
    LVStreamingVideoEncoder,
    _build_advanced_video_ffmpeg_args,
    _validate_advanced_encoder_settings,
)

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / "workflows" / "LongVideo-SBS-Quest3-Production.json"

# Node-schema defaults promoted from the user's A/B-tested workflow.
ctrl = LVResolutionStereoControls.INPUT_TYPES()["required"]
assert ctrl["stereo_strength_percent"][1]["default"] == 1.40
assert ctrl["zero_parallax"][1]["default"] == 0.00

tune = LVDepthStereoTuning.INPUT_TYPES()["required"]
assert tune["zero_parallax"][1]["default"] == 0.00
assert tune["depth_range_scale"][1]["default"] == 1.00
assert tune["temporal_disparity_stability"][1]["default"] == 0.15
assert tune["feather_enabled"][1]["default"] is False
assert tune["feather_strength"][1]["default"] == 0.15

enc = LVStreamingVideoEncoder.INPUT_TYPES()["required"]
assert "cqp" in ENCODER_RATE_CONTROLS
assert enc["rate_control"][1]["default"] == "vbr"
assert enc["av1_qp"][1]["default"] == 24
assert enc["bitrate_mbps"][1]["default"] == 100.0
assert enc["maxrate_mbps"][1]["default"] == 135.0
assert enc["bufsize_mbps"][1]["default"] == 280.0
assert "global_quality" in enc["av1_qp"][1]["tooltip"]

# Explicit AV1 CQP command: fixed q_idx, no VBR-only bitrate flags.
def cmd_for(rc: str):
    return _build_advanced_video_ffmpeg_args(
        "ffmpeg", 120.0, 3840, 2160, "/tmp/out.mp4",
        "av1_vaapi", "10bit", rc, 18.0, 68,
        100.0, 135.0, 280.0, "ultrafast", "/dev/dri/renderD128",
    )

for rc in ("cqp", "constant_quality"):
    cmd = cmd_for(rc)
    assert ["-rc_mode", "CQP"] == cmd[cmd.index("-rc_mode"):cmd.index("-rc_mode") + 2]
    assert cmd[cmd.index("-global_quality") + 1] == "68"
    assert "-b:v" not in cmd
    assert "-maxrate" not in cmd
    assert "-bufsize" not in cmd

vbr = cmd_for("vbr")
assert vbr[vbr.index("-rc_mode") + 1] == "VBR"
assert vbr[vbr.index("-b:v") + 1] == "100M"
assert vbr[vbr.index("-maxrate") + 1] == "135M"
assert vbr[vbr.index("-bufsize") + 1] == "280M"

# Software codecs must not silently accept the AV1-specific CQP mode.
orig_help = ps._ffmpeg_encoder_help
ps._ffmpeg_encoder_help = lambda *_args, **_kwargs: "Supported pixel formats: yuv420p yuv420p10le"
try:
    try:
        _validate_advanced_encoder_settings(
            "ffmpeg", "x264", "10bit", "cqp", 18.0, 68,
            100.0, 135.0, 280.0, "ultrafast", "/dev/dri/renderD128",
        )
        raise AssertionError("x264 cqp should be rejected")
    except ValueError as exc:
        assert "does not use the AV1 VAAPI CQP mode" in str(exc)
finally:
    ps._ffmpeg_encoder_help = orig_help

# Authoritative workflow regression + public sanitization.
wf = json.loads(WF.read_text())
assert wf["extra"]["longvideo_release"].startswith("v3.8.0 DEV3")
assert "render_time_report" not in wf["extra"]
assert not any(n["type"] == "RenderTime" for n in wf["nodes"])
loader = next(n for n in wf["nodes"] if n["type"] == "LV_ProductionVideoLoader")
assert loader["widgets_values_named"]["video"] == "input.mp4"
assert loader["widgets_values_named"]["hardware_decode"] is True
ctrl_node = next(n for n in wf["nodes"] if n["type"] == "LV_ResolutionStereoControls")
assert ctrl_node["widgets_values"] == ["half_sbs", 1.4, 0.0, 3]
tuning = next(n for n in wf["nodes"] if n["type"] == "LV_DepthStereoTuning")
assert tuning["widgets_values"] == [0.0, 1.0, 0.0, 0.0, 0.15, False, "depth_edges", 0.15, 1, 0.04, 0.5]
dibr = next(n for n in wf["nodes"] if n["type"] == "LV_StereoDIBR")
assert dibr["widgets_values"][3] == 0.0
assert dibr["widgets_values_named"]["zero_parallax"] == 0.0
encoder = next(n for n in wf["nodes"] if n["type"] == "LV_StreamingVideoEncoder")
assert encoder["widgets_values_named"]["rate_control"] == "vbr"
assert encoder["widgets_values_named"]["bitrate_mbps"] == 100
assert encoder["widgets_values_named"]["maxrate_mbps"] == 135
assert encoder["widgets_values_named"]["bufsize_mbps"] == 280
assert encoder["widgets_values_named"]["filename_suffix"] == " Half-SBS"

print("v3.8.0 DEV3 A/B defaults + explicit AV1 CQP regression: PASS")
