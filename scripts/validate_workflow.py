#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / "workflows" / "LongVideo-SBS-Quest3-Production.json"
wf = json.loads(WF.read_text())

assert not any(n.get("type") == "RenderTime" for n in wf["nodes"])
assert "render_time_report" not in wf.get("extra", {})

nodes = {n["type"]: n for n in wf["nodes"]}
loader = nodes["LV_ProductionVideoLoader"]["widgets_values_named"]
ctrl = nodes["LV_ResolutionStereoControls"]
tune = nodes["LV_DepthStereoTuning"]
dibr = nodes["LV_StereoDIBR"]
enc = nodes["LV_StreamingVideoEncoder"]["widgets_values_named"]

assert loader["video"] == "input.mp4"
assert loader["hardware_decode"] is True
assert ctrl["widgets_values"] == ["half_sbs", 1.4, 0.0, 3]
assert tune["widgets_values"] == [0.0, 1.0, 0.0, 0.0, 0.15, False, "depth_edges", 0.15, 1, 0.04, 0.5]
assert dibr["widgets_values"][3] == 0.0
assert enc["codec"] == "av1_vaapi"
assert enc["bit_depth"] == "10bit"
assert enc["rate_control"] == "vbr"
assert (enc["bitrate_mbps"], enc["maxrate_mbps"], enc["bufsize_mbps"]) == (100, 135, 280)
assert enc["filename_prefix"] == "halfsbs/"
assert enc["filename_suffix"] == " Half-SBS"

print("LongVideo production workflow: PASS")
