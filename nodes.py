from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import torch

from .session import SESSIONS
from .stereo_renderer import normalize_depth_batch, render_stereo_batch
from .vda_backend import VDAModelHandle, get_default_device, load_vda_model
from .vda_stream import INFER_LEN, process_stream_batch
from .video_loader import VAAPIPreflightError, load_video_vaapi
from .frame_interpolation import LV_LongVideoFrameInterpolation, LV_LongVideoFrameInterpolationStream, LV_RIFEModelLoader


CATEGORY = "Long Video SBS"


def _prompt_requeue(prompt: Any, meta_batch: Any) -> int:
    try:
        batch_id = str(meta_batch.unique_id)
        node = prompt.get(batch_id, prompt.get(meta_batch.unique_id, {}))
        return int(node.get("inputs", {}).get("requeue", 0))
    except Exception:
        return 0


# Production video-loader decoder state. Keyed by the meta-batch object and node id so
# an automatic FFmpeg fallback remains on the same decoder for the entire stream.
_PRODUCTION_VIDEO_DECODERS = {}
_VIDEO_EXTENSIONS = {"webm", "mp4", "mkv", "gif", "mov"}


def _production_loader_key(meta_batch, unique_id):
    if meta_batch is None:
        return None
    return (id(meta_batch), str(unique_id))


def _production_loader_class(name: str):
    # Import ComfyUI's global node registry lazily so this pack does not depend on
    # VideoHelperSuite import order during startup.
    import nodes as comfy_nodes

    cls = comfy_nodes.NODE_CLASS_MAPPINGS.get(name)
    if cls is None:
        raise RuntimeError(
            f"Required VideoHelperSuite node {name!r} is not available. "
            "Update/enable ComfyUI-VideoHelperSuite and restart ComfyUI."
        )
    return cls


class LV_ProductionVideoLoader:
    """Production input loader with an explicit hardware/software FFmpeg toggle.

    Hardware mode uses the direct FFmpeg VAAPI path. Software mode uses the
    VideoHelperSuite FFmpeg loader and does not invoke OpenCV. The output order
    intentionally matches VHS_LoadVideo for IMAGE/frame_count/AUDIO/VHS_VIDEOINFO.
    """

    @classmethod
    def INPUT_TYPES(cls):
        import os
        import folder_paths

        input_dir = folder_paths.get_input_directory()
        files = []
        for name in os.listdir(input_dir):
            path = os.path.join(input_dir, name)
            if os.path.isfile(path) and name.rsplit(".", 1)[-1].lower() in _VIDEO_EXTENSIONS:
                files.append(name)
        return {
            "required": {
                "video": (sorted(files),),
                "hardware_decode": ("BOOLEAN", {"default": True}),
                "force_rate": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 240.0, "step": 0.01}),
                "custom_width": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 8}),
                "custom_height": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 8}),
                "frame_load_cap": ("INT", {"default": 0, "min": 0, "max": 2147483647, "step": 1}),
                "vaapi_device": ("STRING", {"default": "/dev/dri/renderD128"}),
                "prefetch_batches": ("INT", {"default": 1, "min": 0, "max": 1, "step": 1}),
            },
            "optional": {
                "meta_batch": ("VHS_BatchManager",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "AUDIO", "VHS_VIDEOINFO", "STRING")
    RETURN_NAMES = ("IMAGE", "frame_count", "audio", "video_info", "decoder_status")
    FUNCTION = "load_video"
    CATEGORY = CATEGORY

    @staticmethod
    def _common(video, force_rate, custom_width, custom_height, frame_load_cap, meta_batch, unique_id):
        return dict(
            video=video,
            force_rate=float(force_rate),
            custom_width=int(custom_width),
            custom_height=int(custom_height),
            frame_load_cap=int(frame_load_cap),
            meta_batch=meta_batch,
            vae=None,
            format="None",
            unique_id=unique_id,
        )

    def _run_vaapi(self, common, vaapi_device, prefetch_batches=1):
        return load_video_vaapi(common, vaapi_device, prefetch_batches=prefetch_batches)

    def _run_ffmpeg(self, common):
        ff_cls = _production_loader_class("VHS_LoadVideoFFmpeg")
        image, _mask, audio, video_info = ff_cls().load_video(
            **common,
            start_time=0.0,
        )
        frame_count = int(video_info.get("loaded_frame_count", image.shape[0]))
        return image, frame_count, audio, video_info

    def load_video(
        self, video, hardware_decode, force_rate, custom_width, custom_height,
        frame_load_cap, meta_batch=None, unique_id=None, vaapi_device="/dev/dri/renderD128",
        prefetch_batches=1,
    ):
        if video is None or not str(video).strip():
            raise ValueError(
                "LV_ProductionVideoLoader: no video selected. Choose a video in the video dropdown before running."
            )
        common = self._common(
            video, force_rate, custom_width, custom_height, frame_load_cap,
            meta_batch, unique_id,
        )
        if meta_batch is not None:
            try:
                setattr(meta_batch, "_longvideo_source_name", str(video))
            except Exception:
                pass

        key = _production_loader_key(meta_batch, unique_id)
        selected = "ffmpeg_vaapi" if bool(hardware_decode) else "ffmpeg"
        remembered = _PRODUCTION_VIDEO_DECODERS.get(key) if key is not None else None
        if remembered is not None and remembered != selected:
            # A decoder cannot be swapped after a VHS meta-batch stream has
            # started without risking duplicated/restarted frames.
            if meta_batch is not None and unique_id in getattr(meta_batch, "inputs", {}):
                selected = remembered

        try:
            if selected == "ffmpeg_vaapi":
                result = self._run_vaapi(common, vaapi_device, prefetch_batches)
                used = "hardware (ffmpeg_vaapi)"
                remembered_name = "ffmpeg_vaapi"
            else:
                result = self._run_ffmpeg(common)
                used = "software (ffmpeg)"
                remembered_name = "ffmpeg"

            if key is not None:
                if meta_batch is not None and unique_id in getattr(meta_batch, "inputs", {}):
                    _PRODUCTION_VIDEO_DECODERS[key] = remembered_name
                else:
                    _PRODUCTION_VIDEO_DECODERS.pop(key, None)

            image, frame_count, audio, video_info = result
            return image, frame_count, audio, video_info, used
        except Exception:
            if key is not None and (meta_batch is None or unique_id not in getattr(meta_batch, "inputs", {})):
                _PRODUCTION_VIDEO_DECODERS.pop(key, None)
            raise

    @classmethod
    def IS_CHANGED(cls, video, **kwargs):
        import os
        import folder_paths

        path = folder_paths.get_annotated_filepath(video)
        try:
            st = os.stat(path)
            return f"{path}:{st.st_size}:{st.st_mtime_ns}"
        except OSError:
            return float("nan")


class LV_VDAModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "encoder": (["vitl", "vitb", "vits"], {"default": "vitl"}),
                "attention_backend": (["pytorch_sdpa"], {"default": "pytorch_sdpa"}),
                "compile_mode": (["off"], {"default": "off"}),
                "repo_path": ("STRING", {"default": ""}),
                "checkpoint": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("LV_VDA_MODEL",)
    RETURN_NAMES = ("vda_model",)
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, encoder, attention_backend, compile_mode, repo_path, checkpoint):
        handle = load_vda_model(
            encoder=encoder,
            repo_path=repo_path,
            checkpoint=checkpoint,
            attention_backend=attention_backend,
            compile_mode=compile_mode,
        )
        return (handle,)


class LV_VDAStream:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vda_model": ("LV_VDA_MODEL",),
                "images": ("IMAGE",),
                "input_size": ("INT", {"default": 518, "min": 196, "max": 700, "step": 14}),
                "fp32": ("BOOLEAN", {"default": False}),
                "offload_model_when_finished": ("BOOLEAN", {"default": True}),
                "window_mode": (["pipeline_2", "sequential"], {"default": "pipeline_2"}),
                "preprocess_workers": ("INT", {"default": 2, "min": 1, "max": 2, "step": 1}),
            },
            "optional": {
                "meta_batch": ("VHS_BatchManager",),
                "video_info": ("VHS_VIDEOINFO",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
            },
        }

    RETURN_TYPES = ("IMAGE", "LV_DEPTH", "LV_SESSION", "STRING")
    RETURN_NAMES = ("synced_rgb", "depth", "session", "status")
    FUNCTION = "process"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def process(
        self,
        vda_model: VDAModelHandle,
        images,
        input_size,
        fp32,
        offload_model_when_finished,
        window_mode="pipeline_2",
        preprocess_workers=2,
        meta_batch=None,
        video_info=None,
        unique_id=None,
        prompt=None,
    ):
        if meta_batch is not None and int(meta_batch.frames_per_batch) < INFER_LEN:
            raise RuntimeError(
                "Long Video VDA requires VHS Batch Manager frames_per_batch >= 32. "
                "Use 32. Smaller batches can produce an empty first output, and VHS VideoCombine "
                "does not requeue an empty IMAGE batch."
            )

        meta_identity = id(meta_batch) if meta_batch is not None else id(images)
        key = (meta_identity, str(unique_id), vda_model.encoder)
        signature = (
            vda_model.encoder,
            vda_model.checkpoint,
            int(input_size),
            bool(fp32),
            str(window_mode),
            int(preprocess_workers),
            str(getattr(vda_model, "compile_mode", "off")),
        )
        requeue = _prompt_requeue(prompt, meta_batch) if meta_batch is not None else 0
        existing = SESSIONS.get(key)
        force_new = bool(requeue == 0 and existing is not None and existing.input_count > 0)
        session = SESSIONS.get_or_create(key, signature, force_new=force_new)

        if meta_batch is not None:
            session.set_total_from_meta_batch(meta_batch)
            source_name = getattr(meta_batch, "_longvideo_source_name", None)
            if source_name:
                session.source_name = str(source_name)
        if session.total_frames is None and video_info is not None and meta_batch is None:
            try:
                session.total_frames = int(video_info["loaded_frame_count"])
            except Exception:
                pass

        # The loader may not mark has_closed_inputs until it actually reaches EOF.
        # total_frames from VHS lets us determine the final input batch earlier.
        predicted_count = session.input_count + int(images.shape[0])
        final_by_count = (
            session.total_frames is not None and predicted_count >= session.total_frames
        )
        final_by_vhs = bool(getattr(meta_batch, "has_closed_inputs", False)) if meta_batch is not None else True
        is_final_input = final_by_count or final_by_vhs

        with session.lock:
            rgb, depth = process_stream_batch(
                handle=vda_model,
                images=images,
                session=session,
                input_size=int(input_size),
                fp32=bool(fp32),
                is_final_input=is_final_input,
                window_mode=str(window_mode),
                preprocess_workers=int(preprocess_workers),
            )

        if offload_model_when_finished and session.finished:
            vda_model.model.to("cpu")
            session.clear_gpu_state()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        status = (
            f"VDA windows={session.window_index} | input={session.input_count} | "
            f"emitted={session.emitted_frames} | total={session.total_frames} | "
            f"mode={window_mode} | pre_workers={int(preprocess_workers)} | "
            f"compile={getattr(vda_model, 'compile_mode', 'off')} | "
            f"finished={session.finished}"
        )
        return (rgb, depth, session, status)


class LV_DepthTemporalNormalize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rgb": ("IMAGE",),
                "depth": ("LV_DEPTH",),
                "session": ("LV_SESSION",),
                "low_percentile": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 20.0, "step": 0.5}),
                "high_percentile": ("FLOAT", {"default": 98.0, "min": 80.0, "max": 100.0, "step": 0.5}),
                "ema": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 0.999, "step": 0.01}),
                "scene_cut_threshold": ("FLOAT", {"default": 0.20, "min": 0.02, "max": 1.0, "step": 0.01}),
                "invert_depth": ("BOOLEAN", {"default": False}),
            }
        }
    RETURN_TYPES = ("LV_DEPTH",)
    RETURN_NAMES = ("normalized_depth",)
    FUNCTION = "normalize"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def normalize(
        self,
        rgb,
        depth,
        session,
        low_percentile,
        high_percentile,
        ema,
        scene_cut_threshold,
        invert_depth,
    ):
        if high_percentile <= low_percentile:
            raise ValueError("high_percentile must be greater than low_percentile")
        with session.lock:
            norm = normalize_depth_batch(
                rgb=rgb,
                depth=depth,
                session=session,
                low_pct=float(low_percentile),
                high_pct=float(high_percentile),
                ema=float(ema),
                scene_cut_threshold=float(scene_cut_threshold),
                invert_depth=bool(invert_depth),
            )
        return (norm,)


class LV_DepthPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"depth": ("LV_DEPTH",)}}
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preview",)
    FUNCTION = "preview"
    CATEGORY = CATEGORY

    def preview(self, depth):
        d = depth.float().clamp(0, 1).unsqueeze(-1).repeat(1, 1, 1, 3)
        return (d,)


class LV_StereoDIBR:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rgb": ("IMAGE",),
                "normalized_depth": ("LV_DEPTH",),
                "output_mode": (["half_sbs", "full_sbs"], {"default": "half_sbs"}),
                "renderer": (["zbuffer_async", "zbuffer"], {"default": "zbuffer_async"}),
                "max_disparity_eye_px": ("FLOAT", {"default": 32.0, "min": 0.0, "max": 160.0, "step": 1.0}),
                "zero_parallax": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
                "depth_gamma": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05}),
                "hole_fill": ("BOOLEAN", {"default": True}),
                "max_fill_distance": ("INT", {"default": 160, "min": 8, "max": 512, "step": 8}),
                "cpu_output": (["float16", "float32"], {"default": "float16"}),
            }
        }
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("left", "right", "left_holes", "right_holes")
    FUNCTION = "render"
    CATEGORY = CATEGORY

    def render(
        self,
        rgb,
        normalized_depth,
        output_mode,
        renderer,
        max_disparity_eye_px,
        zero_parallax,
        depth_gamma,
        hole_fill,
        max_fill_distance,
        cpu_output,
    ):
        device = get_default_device()
        cpu_dtype = torch.float16 if cpu_output == "float16" else torch.float32
        return render_stereo_batch(
            rgb=rgb,
            norm_depth=normalized_depth,
            device=device,
            output_mode=output_mode,
            max_disparity_eye_px=float(max_disparity_eye_px),
            zero_parallax=float(zero_parallax),
            depth_gamma=float(depth_gamma),
            renderer=renderer,
            hole_fill=bool(hole_fill),
            max_fill_distance=int(max_fill_distance),
            cpu_dtype=cpu_dtype,
        )


class LV_SBSCombine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "left": ("IMAGE",),
                "right": ("IMAGE",),
                "swap_eyes": ("BOOLEAN", {"default": False}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("sbs",)
    FUNCTION = "combine"
    CATEGORY = CATEGORY

    def combine(self, left, right, swap_eyes):
        if left.shape != right.shape:
            raise ValueError(f"Left/right shapes differ: {left.shape} vs {right.shape}")
        if swap_eyes:
            left, right = right, left
        return (torch.cat((left, right), dim=2),)


class LV_SessionInfo:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"session": ("LV_SESSION",)}}
    RETURN_TYPES = ("STRING", "INT", "INT", "BOOLEAN")
    RETURN_NAMES = ("status", "input_frames", "emitted_frames", "finished")
    FUNCTION = "info"
    CATEGORY = CATEGORY

    def info(self, session):
        status = (
            f"windows={session.window_index}, input={session.input_count}, "
            f"emitted={session.emitted_frames}, total={session.total_frames}, "
            f"buffer={len(session.raw_buffer)}, finished={session.finished}"
        )
        return (status, session.input_count, session.emitted_frames, session.finished)


NODE_CLASS_MAPPINGS = {
    "LV_ProductionVideoLoader": LV_ProductionVideoLoader,
    "LV_VDAModelLoader": LV_VDAModelLoader,
    "LV_VDAStream": LV_VDAStream,
    "LV_DepthTemporalNormalize": LV_DepthTemporalNormalize,
    "LV_DepthPreview": LV_DepthPreview,
    "LV_StereoDIBR": LV_StereoDIBR,
    "LV_SBSCombine": LV_SBSCombine,
    "LV_SessionInfo": LV_SessionInfo,
    "LV_RIFEModelLoader": LV_RIFEModelLoader,
    "LV_LongVideoFrameInterpolation": LV_LongVideoFrameInterpolation,
    "LV_LongVideoFrameInterpolationStream": LV_LongVideoFrameInterpolationStream,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LV_ProductionVideoLoader": "LongVideo • Production Video Loader (FFmpeg: Hardware / Software)",
    "LV_VDAModelLoader": "LongVideo • VDA Model Loader",
    "LV_VDAStream": "LongVideo • VDA Temporal Stream",
    "LV_DepthTemporalNormalize": "LongVideo • Stable Depth Normalize",
    "LV_DepthPreview": "LongVideo • Depth Preview",
    "LV_StereoDIBR": "LongVideo • Stereo DIBR (ROCm)",
    "LV_SBSCombine": "LongVideo • SBS Combine",
    "LV_SessionInfo": "LongVideo • Session Info",
    "LV_RIFEModelLoader": "LongVideo • RIFE Model Loader",
    "LV_LongVideoFrameInterpolation": "LongVideo • Frame Interpolation (DEV1 Legacy)",
    "LV_LongVideoFrameInterpolationStream": "LongVideo • Frame Interpolation Stream",
}
