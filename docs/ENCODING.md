# Encoding

The streaming encoder supports `x264`, `x265`, and `av1_vaapi`.

## AV1 VAAPI

The production workflow uses:

```text
codec       = av1_vaapi
bit_depth   = 10bit
vaapi_device= /dev/dri/renderD128
```

### VBR

Current production default:

```text
rate_control = vbr
bitrate      = 100 Mbps
maxrate      = 135 Mbps
bufsize      = 280 Mbps
```

FFmpeg receives `-rc_mode VBR` plus the three bitrate controls.

### CQP

Select:

```text
rate_control = cqp
```

For AV1 VAAPI this becomes:

```text
-rc_mode CQP -global_quality <av1_qp>
```

The `av1_qp` field is AV1 VAAPI q_idx/global_quality on a 1–255 UI range. Lower values target higher quality and generally produce larger files. It is not directly comparable to the H.264/H.265 0–51 QP/CRF scale.

CQP intentionally omits:

```text
-b:v
-maxrate
-bufsize
```

For compatibility with older LongVideo workflows, `av1_vaapi + constant_quality` is treated as an alias for CQP.

## Software codecs

For x264/x265:

- `constant_quality` uses CRF;
- `lossless` uses the codec's lossless path;
- `vbr` uses bitrate/maxrate/bufsize;
- `cqp` is rejected because this UI's `cqp` mode is specifically the AV1 VAAPI q_idx path.

## Color path

The encoder accepts RGB frames from ComfyUI and converts to BT.709 limited-range YUV before encoding. AV1 10-bit uses `p010le` before VAAPI upload.
