from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}

CHECKPOINT_FILENAMES = {
    "vits": "video_depth_anything_vits.pth",
    "vitb": "video_depth_anything_vitb.pth",
    "vitl": "video_depth_anything_vitl.pth",
}


@dataclass
class VDAModelHandle:
    model: torch.nn.Module
    encoder: str
    repo_path: str
    checkpoint: str
    device: torch.device
    attention_backend: str
    name: str
    compile_mode: str = "off"
    compiled_model: Optional[torch.nn.Module] = None
    compile_failed: bool = False
    compile_announced: bool = False

    def ensure_device(self) -> torch.device:
        try:
            param = next(self.model.parameters())
            if param.device != self.device:
                self.model.to(self.device)
        except StopIteration:
            pass
        return self.device

    def forward(self, x: torch.Tensor):
        """Run VDA, using optional TorchInductor with one-shot eager fallback."""
        if self.compiled_model is None or self.compile_failed:
            return self.model(x)
        try:
            out = self.compiled_model(x)
            if not self.compile_announced:
                print(
                    f"[LongVideo VDA] torch.compile active | mode={self.compile_mode}"
                )
                self.compile_announced = True
            return out
        except Exception as exc:
            # Compilation/back-end failures must never strand a long render.
            # Disable the compiled wrapper for this loaded handle and retry the
            # identical input once through eager PyTorch.
            self.compile_failed = True
            self.compiled_model = None
            print(
                "[LongVideo VDA] torch.compile failed; falling back to eager "
                f"for this model handle: {type(exc).__name__}: {exc}"
            )
            return self.model(x)


def _comfy_models_dir() -> Optional[Path]:
    try:
        import folder_paths  # type: ignore

        return Path(folder_paths.models_dir)
    except Exception:
        return None


def find_repo(explicit: str = "") -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env = os.environ.get("VDA_REPO", "")
    if env:
        candidates.append(Path(env).expanduser())

    models_dir = _comfy_models_dir()
    if models_dir is not None:
        candidates.extend(
            [
                models_dir / "video_depth_anything" / "Video-Depth-Anything",
                models_dir / "Video-Depth-Anything",
            ]
        )

    # Standalone scripts (for example vda_smoke_test.py) may run before the
    # ComfyUI root has been added to sys.path, which makes importing
    # folder_paths fail. Derive the normal ComfyUI root from this node's
    # installed location as a second discovery path:
    #   <ComfyUI>/custom_nodes/ComfyUI-LongVideo-SBS/vda_backend.py
    here = Path(__file__).resolve().parent
    comfy_root = here.parent.parent
    candidates.extend(
        [
            comfy_root / "models" / "video_depth_anything" / "Video-Depth-Anything",
            comfy_root / "models" / "Video-Depth-Anything",
            here / "vendor" / "Video-Depth-Anything",
        ]
    )

    for candidate in candidates:
        marker = candidate / "video_depth_anything" / "video_depth.py"
        if marker.is_file():
            return candidate.resolve()

    pretty = "\n".join(f"  - {c}" for c in candidates)
    raise FileNotFoundError(
        "Video Depth Anything repository was not found. Run install_vda.sh or set VDA_REPO.\n"
        f"Checked:\n{pretty}"
    )


def find_checkpoint(repo: Path, encoder: str, explicit: str = "") -> Path:
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_file():
            return path.resolve()
        raise FileNotFoundError(f"VDA checkpoint does not exist: {path}")

    filename = CHECKPOINT_FILENAMES[encoder]
    candidates = [repo / "checkpoints" / filename]
    models_dir = _comfy_models_dir()
    if models_dir is not None:
        candidates.extend(
            [
                models_dir / "video_depth_anything" / filename,
                models_dir / filename,
            ]
        )

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Missing checkpoint {filename}. Run install_vda.sh {encoder} or place it in "
        f"{repo / 'checkpoints'}"
    )


def _safe_import_vda(repo: Path):
    """Import VDA without permanently stealing a generic top-level `utils` module.

    The upstream repository currently imports `utils.util` absolutely from
    video_depth_anything/video_depth.py. ComfyUI and other custom nodes can also
    have a top-level module named `utils`, so we temporarily isolate that import.
    """

    repo_str = str(repo)

    # Upstream keeps utils/ beside video_depth_anything/ and imports it as
    # `utils.util`. Because utils/ historically had no __init__.py, a regular
    # third-party package also named `utils` can override that namespace even
    # when the VDA repo is first on sys.path. Make it an explicit package in
    # memory/on disk before importing VDA.
    utils_dir = repo / "utils"
    if utils_dir.is_dir():
        try:
            (utils_dir / "__init__.py").touch(exist_ok=True)
        except OSError:
            pass

    old_utils = sys.modules.pop("utils", None)
    old_utils_util = sys.modules.pop("utils.util", None)
    sys.path.insert(0, repo_str)
    try:
        importlib.invalidate_caches()
        module = importlib.import_module("video_depth_anything.video_depth")
    finally:
        try:
            sys.path.remove(repo_str)
        except ValueError:
            pass
        # Keep VDA's bound function references, but restore the application's
        # prior generic `utils` namespace to reduce custom-node collisions.
        sys.modules.pop("utils", None)
        sys.modules.pop("utils.util", None)
        if old_utils is not None:
            sys.modules["utils"] = old_utils
        if old_utils_util is not None:
            sys.modules["utils.util"] = old_utils_util
    return module


def _patch_mem_eff_attention_to_sdpa() -> None:
    """Patch both VDA xFormers attention paths to PyTorch SDPA.

    VDA has *two independent* xFormers users:
      1. DINOv2 spatial self-attention (MemEffAttention).
      2. The temporal motion module (TemporalAttention/CrossAttention).

    On ROCm an incompatible CUDA xFormers wheel can still import successfully,
    which makes upstream VDA choose xFormers and then fail only when the first
    compiled operator is called. Patch both paths before model construction so
    they use torch.nn.functional.scaled_dot_product_attention instead.
    """

    # ---- DINOv2 spatial attention ----
    attention_mod = importlib.import_module(
        "video_depth_anything.dinov2_layers.attention"
    )
    cls = attention_mod.MemEffAttention

    if not getattr(cls, "_lv_sbs_sdpa_patched", False):
        def forward_sdpa(self, x, attn_bias=None):
            if attn_bias is not None:
                # VDA inference passes ordinary tensors, not nested-tensor biases.
                # Preserve a clear failure rather than silently changing semantics.
                raise RuntimeError(
                    "PyTorch-SDPA VDA patch does not support nested-tensor attn_bias"
                )
            b, n, c = x.shape
            qkv = self.qkv(x).reshape(
                b, n, 3, self.num_heads, c // self.num_heads
            )
            q, k, v = qkv.unbind(dim=2)
            # PyTorch SDPA expects B,H,L,D.
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )
            out = out.transpose(1, 2).contiguous().reshape(b, n, c)
            out = self.proj(out)
            out = self.proj_drop(out)
            return out

        cls.forward = forward_sdpa
        cls._lv_sbs_sdpa_patched = True

    # ---- Temporal motion attention ----
    # TemporalAttention.forward() deliberately takes the xFormers branch whenever
    # xFormers imported and head_dim is divisible by 8. At that point q/k/v are
    # already shaped B,L,H,D (e.g. 1369,32,8,48). Replace only the inherited
    # xFormers method so the surrounding VDA temporal/caching logic is untouched.
    motion_mod = importlib.import_module(
        "video_depth_anything.motion_module.motion_module"
    )
    temporal_cls = motion_mod.TemporalAttention

    if not getattr(temporal_cls, "_lv_sbs_sdpa_patched", False):
        def temporal_attention_sdpa(self, query, key, value, attention_mask):
            if attention_mask is not None:
                # Current VDA TemporalAttention asserts attention_mask is None.
                # Fail loudly if upstream changes that contract.
                raise RuntimeError(
                    "PyTorch-SDPA temporal VDA patch does not yet support attention_mask"
                )

            original_dtype = query.dtype
            if getattr(self, "upcast_efficient_attention", False):
                query = query.float()
                key = key.float()
                value = value.float()

            # Upstream xFormers path supplies B,L,H,D. SDPA expects B,H,L,D.
            q = query.permute(0, 2, 1, 3).contiguous()
            k = key.permute(0, 2, 1, 3).contiguous()
            v = value.permute(0, 2, 1, 3).contiguous()

            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )

            # Back to B,L,H,D, then use VDA's own head merge helper.
            out = out.permute(0, 2, 1, 3).contiguous()
            if getattr(self, "upcast_efficient_attention", False):
                out = out.to(original_dtype)
            return self.reshape_4d_to_heads(out)

        temporal_cls._memory_efficient_attention_xformers = temporal_attention_sdpa
        temporal_cls._lv_sbs_sdpa_patched = True


def get_default_device() -> torch.device:
    try:
        import comfy.model_management as mm  # type: ignore

        return mm.get_torch_device()
    except Exception:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")


def load_vda_model(
    encoder: str,
    repo_path: str = "",
    checkpoint: str = "",
    attention_backend: str = "pytorch_sdpa",
    compile_mode: str = "off",
) -> VDAModelHandle:
    if encoder not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported VDA encoder: {encoder}")

    repo = find_repo(repo_path)
    ckpt = find_checkpoint(repo, encoder, checkpoint)
    vda_module = _safe_import_vda(repo)

    if attention_backend == "pytorch_sdpa":
        _patch_mem_eff_attention_to_sdpa()

    VideoDepthAnything = vda_module.VideoDepthAnything
    model = VideoDepthAnything(**MODEL_CONFIGS[encoder], metric=False)
    state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    device = get_default_device()
    model = model.to(device).eval()

    mode = str(compile_mode or "off")
    allowed_compile_modes = {
        "off",
        "default",
        "reduce-overhead",
        "max-autotune",
        "max-autotune-no-cudagraphs",
    }
    if mode not in allowed_compile_modes:
        raise ValueError(f"Unsupported torch.compile mode: {mode}")

    compiled_model = None
    compile_failed = False
    if mode != "off":
        if not hasattr(torch, "compile"):
            print("[LongVideo VDA] torch.compile unavailable; using eager mode")
            compile_failed = True
        else:
            try:
                # Compilation is lazy: the first real VDA forward performs graph
                # capture/codegen/autotuning. Fixed 32-frame windows and fixed
                # transformed H/W are ideal for shape-specialized inference.
                compiled_model = torch.compile(
                    model,
                    mode=mode,
                    fullgraph=False,
                    dynamic=False,
                )
                print(
                    f"[LongVideo VDA] torch.compile armed | mode={mode} | "
                    "first forward performs compilation"
                )
            except Exception as exc:
                compile_failed = True
                print(
                    "[LongVideo VDA] unable to arm torch.compile; using eager: "
                    f"{type(exc).__name__}: {exc}"
                )

    return VDAModelHandle(
        model=model,
        encoder=encoder,
        repo_path=str(repo),
        checkpoint=str(ckpt),
        device=device,
        attention_backend=attention_backend,
        name=f"VDA-{encoder}",
        compile_mode=mode,
        compiled_model=compiled_model,
        compile_failed=compile_failed,
    )


def transformed_hw(src_h: int, src_w: int, input_size: int) -> Tuple[int, int]:
    """Match VDA's lower-bound aspect-preserving resize to a multiple of 14."""
    ratio = max(src_h, src_w) / max(1, min(src_h, src_w))
    adjusted = int(input_size)
    if ratio > 1.78:
        adjusted = int(adjusted * 1.777 / ratio)
        adjusted = max(14, round(adjusted / 14) * 14)

    scale_h = adjusted / src_h
    scale_w = adjusted / src_w
    scale = max(scale_h, scale_w)
    out_h = max(adjusted, int(round((src_h * scale) / 14.0) * 14))
    out_w = max(adjusted, int(round((src_w * scale) / 14.0) * 14))
    return out_h, out_w
