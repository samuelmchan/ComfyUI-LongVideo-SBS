from pathlib import Path

from .nodes import NODE_CLASS_MAPPINGS as CORE_NODE_CLASS_MAPPINGS
from .nodes import NODE_DISPLAY_NAME_MAPPINGS as CORE_NODE_DISPLAY_NAME_MAPPINGS
from .pipeline_support import NODE_CLASS_MAPPINGS as SUPPORT_NODE_CLASS_MAPPINGS
from .pipeline_support import NODE_DISPLAY_NAME_MAPPINGS as SUPPORT_NODE_DISPLAY_NAME_MAPPINGS

# v3.7 consolidates the former ComfyUI-LongVideo-SBS-Upgrade companion into this
# package. Preserve every existing LV_* node id so old workflows load unchanged.
_overlap = set(CORE_NODE_CLASS_MAPPINGS).intersection(SUPPORT_NODE_CLASS_MAPPINGS)
if _overlap:
    raise RuntimeError(f"LongVideo SBS duplicate merged node ids: {sorted(_overlap)}")

NODE_CLASS_MAPPINGS = {**CORE_NODE_CLASS_MAPPINGS, **SUPPORT_NODE_CLASS_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {
    **CORE_NODE_DISPLAY_NAME_MAPPINGS,
    **SUPPORT_NODE_DISPLAY_NAME_MAPPINGS,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
__version__ = "3.8.0-dev3"

# A stale split companion would register the same LV_* ids and make load order
# determine which implementation wins. Warn loudly so migration is deterministic.
_legacy_companion = Path(__file__).resolve().parent.parent / "ComfyUI-LongVideo-SBS-Upgrade"
if _legacy_companion.exists():
    print(
        "[LongVideo SBS] WARNING: legacy ComfyUI-LongVideo-SBS-Upgrade is still installed. "
        "v3.8.0 DEV3 already contains those nodes; remove the legacy folder and restart ComfyUI."
    )

print("[LongVideo SBS] loaded v3.8.0 DEV3 (A/B defaults + explicit AV1 CQP)")
