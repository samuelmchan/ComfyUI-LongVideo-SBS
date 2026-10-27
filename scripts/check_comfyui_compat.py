#!/usr/bin/env python3
"""Verify ComfyUI-LongVideo-SBS against a ComfyUI install or source checkout.

ComfyUI internals this node pack imports are not a stable public API: symbols
get renamed, moved between modules, or turned into aliases. Both break the pack
at import time with an obscure error, so check deliberately.

Two modes:

  static  (default)  Parse a ComfyUI *source tree*. No torch, no imports.
                     Use in CI against any ComfyUI tag or master.
  runtime            Import the real modules in the ComfyUI venv.
                     Use on a deployment right after upgrading ComfyUI.

Usage:
  python scripts/check_comfyui_compat.py --comfy-root /path/to/ComfyUI
  python scripts/check_comfyui_compat.py --runtime

Exit code is 0 only if every symbol resolves and every pack-owned node type
used by the production workflow is registered.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_WORKFLOW = ROOT / "workflows" / "LongVideo-SBS-Quest3-Production.json"

# Modules and attributes the pack depends on. Keep in sync with the imports in
# nodes.py, pipeline_support.py, frame_interpolation.py, vda_backend.py and
# video_loader.py.
API_SURFACE = {
    "comfy.ldm.common_dit": ["pad_to_patch_size"],
    "comfy.model_management": ["get_torch_device", "load_models_gpu"],
    "comfy.model_patcher": ["CoreModelPatcher"],
    "comfy.utils": ["load_torch_file"],
    "comfy_extras.nodes_frame_interpolation": ["FrameInterpolationModelLoader"],
    "folder_paths": [
        "get_annotated_filepath",
        "get_filename_list",
        "get_full_path",
        "get_input_directory",
        "get_output_directory",
        "get_save_image_path",
        "get_temp_directory",
        "models_dir",
    ],
    "nodes": ["NODE_CLASS_MAPPINGS"],
}

# Files that build the pack's NODE_CLASS_MAPPINGS dict literals.
PACK_NODE_SOURCES = ["nodes.py", "pipeline_support.py", "frame_interpolation.py"]


def module_path(comfy_root: Path, module: str) -> Path:
    if module.startswith("comfy_extras."):
        return comfy_root / "comfy_extras" / (module.split(".", 1)[1].replace(".", "/") + ".py")
    if module.startswith("comfy."):
        return comfy_root / "comfy" / (module.split(".", 1)[1].replace(".", "/") + ".py")
    return comfy_root / (module.replace(".", "/") + ".py")


def module_level_names(tree: ast.Module) -> set[str]:
    """Names bound at module scope, descending into if/try/with/for but not into
    function or class bodies. Covers `def x`, `class X`, `X = ...` and aliases
    like `CoreModelPatcher = ModelPatcher`."""
    names: set[str] = set()

    def visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(child.name)
                continue
            if isinstance(child, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
                visit(child)
                continue
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        names.add(target.id)
            elif isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                names.add(child.target.id)
            elif isinstance(child, (ast.Import, ast.ImportFrom)):
                for alias in child.names:
                    names.add(alias.asname or alias.name.split(".")[0])

    visit(tree)
    return names


def static_version(comfy_root: Path) -> str:
    version_file = comfy_root / "comfyui_version.py"
    if not version_file.is_file():
        return "unknown"
    for line in version_file.read_text(encoding="utf-8").splitlines():
        if line.startswith("__version__"):
            return line.split("=", 1)[1].strip().strip("\"'")
    return "unknown"


def check_static(comfy_root: Path) -> list[tuple[str, str, bool]]:
    cache: dict[Path, set[str] | None] = {}
    results = []
    for module, symbols in API_SURFACE.items():
        path = module_path(comfy_root, module)
        if path not in cache:
            if not path.is_file():
                cache[path] = None
            else:
                cache[path] = module_level_names(ast.parse(path.read_text(encoding="utf-8")))
        names = cache[path]
        for symbol in symbols:
            results.append((module, symbol, names is not None and symbol in names))
    return results


def check_runtime() -> list[tuple[str, str, bool]]:
    results = []
    for module, symbols in API_SURFACE.items():
        try:
            mod = importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001 - report any import failure
            for symbol in symbols:
                results.append((module, symbol, False))
            print(f"  ! import {module} failed: {exc}", file=sys.stderr)
            continue
        for symbol in symbols:
            results.append((module, symbol, hasattr(mod, symbol)))
    return results


def registered_pack_nodes() -> set[str]:
    """Node type names defined by the pack, read from its dict literals."""
    names: set[str] = set()
    for name in PACK_NODE_SOURCES:
        path = ROOT / name
        if not path.is_file():
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "NODE_CLASS_MAPPINGS" for t in node.targets
            ):
                if isinstance(node.value, ast.Dict):
                    for key in node.value.keys:
                        if isinstance(key, ast.Constant) and isinstance(key.value, str):
                            names.add(key.value)
    return names


def check_workflow_coverage() -> list[tuple[str, bool]]:
    """Every pack-owned node type in the production workflow must exist."""
    if not PRODUCTION_WORKFLOW.is_file():
        return []
    workflow = json.loads(PRODUCTION_WORKFLOW.read_text(encoding="utf-8"))
    used = {n["type"] for n in workflow.get("nodes", [])}
    pack_used = {t for t in used if t.startswith("LV_")}
    registered = registered_pack_nodes()
    return [(t, t in registered) for t in sorted(pack_used)]


def ensure_comfyui_on_path() -> Path:
    """Running a file under custom_nodes/ puts scripts/ on sys.path, not the
    ComfyUI root, so `comfyui_version` and `comfy` would not resolve."""
    for candidate in (Path.cwd(), *ROOT.parents):
        if (candidate / "comfyui_version.py").is_file():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return candidate
    raise SystemExit("could not locate ComfyUI; run from the ComfyUI root")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--comfy-root", type=Path, help="ComfyUI source tree (static mode)")
    parser.add_argument("--runtime", action="store_true", help="check the installed ComfyUI by importing it")
    args = parser.parse_args()

    if args.runtime and args.comfy_root:
        parser.error("choose --runtime or --comfy-root, not both")

    if args.runtime:
        mode = "runtime"
        comfy_root = ensure_comfyui_on_path()
        version = importlib.import_module("comfyui_version").__version__
        results = check_runtime()
    else:
        comfy_root = (args.comfy_root or Path.cwd()).resolve()
        if not (comfy_root / "comfyui_version.py").is_file():
            print(f"not a ComfyUI tree: {comfy_root}", file=sys.stderr)
            return 2
        mode = "static"
        version = static_version(comfy_root)
        results = check_static(comfy_root)

    print(f"ComfyUI {version} ({mode} check, {len(results)} symbols)")

    failures = 0
    for module, symbol, ok in results:
        if not ok:
            failures += 1
            print(f"  MISSING  {module}.{symbol}")

    coverage = check_workflow_coverage()
    missing_nodes = [name for name, ok in coverage if not ok]
    for name in missing_nodes:
        print(f"  MISSING  node type {name} (used by production workflow)")

    if failures or missing_nodes:
        print(f"FAIL: {failures} missing symbol(s), {len(missing_nodes)} missing node type(s)")
        return 1

    print(f"OK: all {len(results)} symbols resolve, {len(coverage)} workflow node types registered")
    return 0


if __name__ == "__main__":
    sys.exit(main())
