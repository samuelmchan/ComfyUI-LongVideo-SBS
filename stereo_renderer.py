from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def _scene_cut_and_norm(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    session,
    low_pct: float,
    high_pct: float,
    ema: float,
    scene_cut_threshold: float,
    invert_depth: bool,
) -> torch.Tensor:
    # Cheap scene thumbnail directly on CPU by strided sampling.
    stride_y = max(1, rgb.shape[0] // 48)
    stride_x = max(1, rgb.shape[1] // 48)
    thumb = rgb[::stride_y, ::stride_x, :3].float().contiguous()

    cut = False
    if session.prev_scene_thumb is not None:
        prev = session.prev_scene_thumb
        if prev.shape == thumb.shape:
            diff = torch.mean(torch.abs(thumb - prev)).item()
            cut = diff >= scene_cut_threshold
    session.prev_scene_thumb = thumb.clone()

    sample_stride = max(1, min(depth.shape[-2:]) // 192)
    sample = depth[::sample_stride, ::sample_stride].float().flatten()
    cur_low = torch.quantile(sample, low_pct / 100.0).item()
    cur_high = torch.quantile(sample, high_pct / 100.0).item()
    if cur_high <= cur_low + 1e-8:
        cur_high = cur_low + 1e-6

    if cut or session.norm_low is None or session.norm_high is None:
        session.norm_low = cur_low
        session.norm_high = cur_high
    else:
        session.norm_low = ema * session.norm_low + (1.0 - ema) * cur_low
        session.norm_high = ema * session.norm_high + (1.0 - ema) * cur_high

    norm = (depth.float() - session.norm_low) / max(
        session.norm_high - session.norm_low, 1e-6
    )
    norm = norm.clamp(0.0, 1.0)
    if invert_depth:
        norm = 1.0 - norm
    return norm


def normalize_depth_batch(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    session,
    low_pct: float = 2.0,
    high_pct: float = 98.0,
    ema: float = 0.95,
    scene_cut_threshold: float = 0.20,
    invert_depth: bool = False,
) -> torch.Tensor:
    if len(rgb) != len(depth):
        raise ValueError("RGB/depth batch lengths differ")
    out = []
    for i in range(len(depth)):
        out.append(
            _scene_cut_and_norm(
                rgb[i],
                depth[i],
                session,
                low_pct,
                high_pct,
                ema,
                scene_cut_threshold,
                invert_depth,
            )
        )
    return torch.stack(out, dim=0) if out else torch.empty_like(depth)


def _grid_sample_eye(
    image: torch.Tensor,
    norm_depth: torch.Tensor,
    separation_px: float,
    zero_parallax: float,
    direction: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # image [C,H,W], depth [H,W]. Larger normalized depth is nearer.
    c, h, w = image.shape
    disparity = (norm_depth - zero_parallax) * separation_px
    shift = direction * disparity * 0.5

    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=image.device, dtype=image.dtype),
        torch.linspace(-1.0, 1.0, w, device=image.device, dtype=image.dtype),
        indexing="ij",
    )
    xshift = 2.0 * shift.to(image.dtype) / max(w - 1, 1)
    grid = torch.stack((xx - xshift, yy), dim=-1).unsqueeze(0)
    result = F.grid_sample(
        image.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )[0]
    holes = torch.zeros((h, w), device=image.device, dtype=torch.bool)
    return result, holes


def _forward_warp_one(
    image: torch.Tensor,
    norm_depth: torch.Tensor,
    separation_px: float,
    zero_parallax: float,
    direction: float,
    z_epsilon: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Horizontal forward splat with a nearest-depth z-buffer.

    Uses only scatter_reduce/scatter_add, so there is no custom CUDA kernel and
    the same code path works on ROCm/HIP.
    """

    c, h, w = image.shape
    device = image.device
    dtype = image.dtype

    disparity = (norm_depth - zero_parallax) * separation_px
    shift = direction * disparity * 0.5
    x = torch.arange(w, device=device, dtype=torch.float32).view(1, w)
    target = x + shift.float()
    x0 = torch.floor(target).to(torch.int64)
    x1 = x0 + 1
    w1 = target - x0.float()
    w0 = 1.0 - w1
    row = (torch.arange(h, device=device, dtype=torch.int64) * w).view(h, 1)

    depth_flat = norm_depth.float().reshape(-1)
    color_flat = image.reshape(c, -1)
    zbuf = torch.full((h * w,), -float("inf"), device=device, dtype=torch.float32)

    candidates = []
    for xi, wi in ((x0, w0), (x1, w1)):
        valid = (xi >= 0) & (xi < w) & (wi > 0)
        src = torch.nonzero(valid.reshape(-1), as_tuple=False).squeeze(1)
        if src.numel() == 0:
            continue
        idx = (row + xi.clamp(0, w - 1)).reshape(-1)[src]
        z = depth_flat[src]
        zbuf.scatter_reduce_(0, idx, z, reduce="amax", include_self=True)
        candidates.append((src, idx, wi.reshape(-1)[src]))

    out = torch.zeros((c, h * w), device=device, dtype=dtype)
    weight = torch.zeros((h * w,), device=device, dtype=torch.float32)
    for src, idx, wi in candidates:
        z = depth_flat[src]
        wins = z >= (zbuf[idx] - z_epsilon)
        if not torch.any(wins):
            continue
        srcw = src[wins]
        idxw = idx[wins]
        ww = wi[wins].float()
        vals = color_flat[:, srcw] * ww.to(dtype).unsqueeze(0)
        out.scatter_add_(1, idxw.unsqueeze(0).expand(c, -1), vals)
        weight.scatter_add_(0, idxw, ww)

    valid_out = weight > 1e-8
    out[:, valid_out] /= weight[valid_out].to(dtype).unsqueeze(0)
    holes = ~valid_out.reshape(h, w)
    warped_depth = zbuf.reshape(h, w)
    warped_depth = torch.where(
        torch.isfinite(warped_depth), warped_depth, torch.ones_like(warped_depth)
    )
    return out.reshape(c, h, w), holes, warped_depth



def _forward_warp_pair(
    image: torch.Tensor,
    norm_depth: torch.Tensor,
    separation_px: float,
    zero_parallax: float,
    z_epsilon: float = 1e-5,
):
    """Render both eyes with the exact reference splat while sharing setup.

    Scatter/reduce remains eye-local (important on ROCm); only the common
    disparity/coordinate/flatten setup is reused.
    """
    c, h, w = image.shape
    device = image.device
    dtype = image.dtype

    disparity = (norm_depth - zero_parallax) * separation_px
    base_shift = disparity * 0.5
    x = torch.arange(w, device=device, dtype=torch.float32).view(1, w)
    row = (torch.arange(h, device=device, dtype=torch.int64) * w).view(h, 1)
    depth_flat = norm_depth.float().reshape(-1)
    color_flat = image.reshape(c, -1)

    def one(shift):
        target = x + shift.float()
        x0 = torch.floor(target).to(torch.int64)
        x1 = x0 + 1
        w1 = target - x0.float()
        w0 = 1.0 - w1
        zbuf = torch.full((h * w,), -float("inf"), device=device, dtype=torch.float32)
        candidates = []
        for xi, wi in ((x0, w0), (x1, w1)):
            valid = (xi >= 0) & (xi < w) & (wi > 0)
            src = torch.nonzero(valid.reshape(-1), as_tuple=False).squeeze(1)
            if src.numel() == 0:
                continue
            idx = (row + xi.clamp(0, w - 1)).reshape(-1)[src]
            z = depth_flat[src]
            zbuf.scatter_reduce_(0, idx, z, reduce="amax", include_self=True)
            candidates.append((src, idx, wi.reshape(-1)[src]))

        out = torch.zeros((c, h * w), device=device, dtype=dtype)
        weight = torch.zeros((h * w,), device=device, dtype=torch.float32)
        for src, idx, wi in candidates:
            z = depth_flat[src]
            wins = z >= (zbuf[idx] - z_epsilon)
            if not torch.any(wins):
                continue
            srcw = src[wins]
            idxw = idx[wins]
            ww = wi[wins].float()
            vals = color_flat[:, srcw] * ww.to(dtype).unsqueeze(0)
            out.scatter_add_(1, idxw.unsqueeze(0).expand(c, -1), vals)
            weight.scatter_add_(0, idxw, ww)

        valid_out = weight > 1e-8
        out[:, valid_out] /= weight[valid_out].to(dtype).unsqueeze(0)
        holes = ~valid_out.reshape(h, w)
        warped_depth = zbuf.reshape(h, w)
        warped_depth = torch.where(
            torch.isfinite(warped_depth), warped_depth, torch.ones_like(warped_depth)
        )
        return out.reshape(c, h, w), holes, warped_depth

    left = one(base_shift)
    right = one(-base_shift)
    return left, right


_COORD_CACHE = {}


def _warp_coords_cached(device: torch.device, h: int, w: int):
    """Static coordinates reused by the experimental fused z-buffer path."""
    key = (device.type, device.index, int(h), int(w))
    cached = _COORD_CACHE.get(key)
    if cached is not None:
        return cached
    x = torch.arange(w, device=device, dtype=torch.float32).view(1, w)
    row_offsets = (torch.arange(h, device=device, dtype=torch.int64) * w)
    row_flat = row_offsets.repeat_interleave(w)
    if len(_COORD_CACHE) >= 8:
        _COORD_CACHE.clear()
    _COORD_CACHE[key] = (x, row_flat)
    return x, row_flat


def _forward_warp_one_fused(
    image: torch.Tensor,
    norm_depth: torch.Tensor,
    separation_px: float,
    zero_parallax: float,
    direction: float,
    z_epsilon: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact fused variant of the production z-buffer splat.

    It preserves the current two-tap horizontal splat and near-equal-depth
    weighted blending, but combines x0/x1 candidates into one depth reduction
    and packs RGB+weight into one scatter-add. Output is intended to be
    numerically equivalent to _forward_warp_one apart from last-bit FP order.
    """
    c, h, w = image.shape
    device = image.device
    dtype = image.dtype
    n = h * w

    disparity = (norm_depth - zero_parallax) * separation_px
    shift = direction * disparity * 0.5
    x, row_flat = _warp_coords_cached(device, h, w)
    target = x + shift.float()
    x0 = torch.floor(target).to(torch.int64)
    x1 = x0 + 1
    w1 = target - x0.float()
    w0 = 1.0 - w1

    depth_flat = norm_depth.float().reshape(-1)
    color_flat = image.reshape(c, -1)
    src_parts = []
    idx_parts = []
    weight_parts = []

    for xi, wi in ((x0, w0), (x1, w1)):
        valid_flat = ((xi >= 0) & (xi < w) & (wi > 0)).reshape(-1)
        src = torch.nonzero(valid_flat, as_tuple=False).squeeze(1)
        if src.numel() == 0:
            continue
        xi_flat = xi.reshape(-1)
        src_parts.append(src)
        idx_parts.append(row_flat[src] + xi_flat[src])
        weight_parts.append(wi.reshape(-1)[src].float())

    if not src_parts:
        return (
            torch.zeros((c, h, w), device=device, dtype=dtype),
            torch.ones((h, w), device=device, dtype=torch.bool),
            torch.ones((h, w), device=device, dtype=torch.float32),
        )

    src_all = torch.cat(src_parts, dim=0)
    idx_all = torch.cat(idx_parts, dim=0)
    ww_all = torch.cat(weight_parts, dim=0)
    z_all = depth_flat[src_all]

    zbuf = torch.full((n,), -float("inf"), device=device, dtype=torch.float32)
    zbuf.scatter_reduce_(0, idx_all, z_all, reduce="amax", include_self=True)

    wins = z_all >= (zbuf[idx_all] - z_epsilon)
    srcw = src_all[wins]
    idxw = idx_all[wins]
    ww = ww_all[wins]

    accum = torch.zeros((c + 1, n), device=device, dtype=torch.float32)
    if srcw.numel() > 0:
        vals_rgb = color_flat[:, srcw].float() * ww.unsqueeze(0)
        vals = torch.cat((vals_rgb, ww.unsqueeze(0)), dim=0)
        accum.scatter_add_(1, idxw.unsqueeze(0).expand(c + 1, -1), vals)

    weight = accum[c]
    valid_out = weight > 1e-8
    out = accum[:c]
    out[:, valid_out] /= weight[valid_out].unsqueeze(0)
    out = out.to(dtype=dtype)
    holes = ~valid_out.reshape(h, w)
    warped_depth = zbuf.reshape(h, w)
    warped_depth = torch.where(
        torch.isfinite(warped_depth), warped_depth, torch.ones_like(warped_depth)
    )
    return out.reshape(c, h, w), holes, warped_depth

def _gather_width(image: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    c, h, w = image.shape
    idx = indices.clamp(0, w - 1).unsqueeze(0).expand(c, -1, -1)
    return torch.gather(image, 2, idx)


def _gather_depth(depth: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    h, w = depth.shape
    return torch.gather(depth, 1, indices.clamp(0, w - 1))


def fill_holes_background(
    image: torch.Tensor,
    holes: torch.Tensor,
    warped_depth: torch.Tensor,
    max_distance: int = 160,
) -> torch.Tensor:
    """Fill horizontal disocclusions from the farther valid side.

    Stereo disocclusions are predominantly horizontal. For every hole, find the
    nearest valid sample on the left and right in one pass and prefer the farther
    (smaller normalized-depth) candidate. This avoids the repeated average-pool
    smearing used by simpler DIBR implementations.
    """

    if not torch.any(holes):
        return image

    h, w = holes.shape
    device = holes.device
    valid = ~holes
    x = torch.arange(w, device=device, dtype=torch.int64).view(1, w).expand(h, -1)

    left_seed = torch.where(valid, x, torch.full_like(x, -1))
    left_idx = torch.cummax(left_seed, dim=1).values

    valid_rev = torch.flip(valid, dims=[1])
    xr = torch.arange(w, device=device, dtype=torch.int64).view(1, w).expand(h, -1)
    right_seed_rev = torch.where(valid_rev, xr, torch.full_like(xr, -1))
    right_rev = torch.cummax(right_seed_rev, dim=1).values
    right_idx = torch.where(
        right_rev >= 0,
        (w - 1) - torch.flip(right_rev, dims=[1]),
        torch.full_like(right_rev, -1),
    )

    left_ok = left_idx >= 0
    right_ok = right_idx >= 0
    left_dist = torch.where(left_ok, x - left_idx, torch.full_like(x, 10**9))
    right_dist = torch.where(right_ok, right_idx - x, torch.full_like(x, 10**9))
    left_ok &= left_dist <= max_distance
    right_ok &= right_dist <= max_distance

    left_img = _gather_width(image, left_idx)
    right_img = _gather_width(image, right_idx)
    left_depth = _gather_depth(warped_depth, left_idx)
    right_depth = _gather_depth(warped_depth, right_idx)

    # Smaller normalized depth is farther/background. If depths are nearly equal,
    # prefer the geometrically nearer source pixel.
    prefer_left = left_depth < right_depth
    similar = torch.abs(left_depth - right_depth) < 0.02
    prefer_left = torch.where(similar, left_dist <= right_dist, prefer_left)
    prefer_left = (prefer_left & left_ok) | (~right_ok & left_ok)
    has_candidate = left_ok | right_ok

    chosen = torch.where(prefer_left.unsqueeze(0), left_img, right_img)
    fill_mask = holes & has_candidate
    result = torch.where(fill_mask.unsqueeze(0), chosen, image)

    # Any very large border holes left after the bounded background fill receive
    # nearest-border replication rather than black pixels.
    remaining = holes & ~has_candidate
    if torch.any(remaining):
        nearest_left = left_dist <= right_dist
        fallback = torch.where(nearest_left.unsqueeze(0), left_img, right_img)
        result = torch.where(remaining.unsqueeze(0), fallback, result)
    return result



def _forward_warp_batch(
    image: torch.Tensor,
    norm_depth: torch.Tensor,
    separation_px: float,
    zero_parallax: float,
    direction: float,
    z_epsilon: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batch-exact form of :func:`_forward_warp_one`.

    image is [B,C,H,W] and norm_depth is [B,H,W].  Each frame receives a
    disjoint flattened z-buffer index range, so the scatter reductions retain
    exactly the same per-frame visibility/blending rules while amortizing
    Python dispatch, host/device transfers, interpolation, and GPU kernel
    launches across a small frame chunk.
    """
    b, c, h, w = image.shape
    device = image.device
    dtype = image.dtype
    frame_n = h * w

    disparity = (norm_depth - zero_parallax) * separation_px
    shift = direction * disparity * 0.5
    x = torch.arange(w, device=device, dtype=torch.float32).view(1, 1, w)
    target = x + shift.float()
    x0 = torch.floor(target).to(torch.int64)
    x1 = x0 + 1
    w1 = target - x0.float()
    w0 = 1.0 - w1

    row = (torch.arange(h, device=device, dtype=torch.int64) * w).view(1, h, 1)
    batch_offset = (
        torch.arange(b, device=device, dtype=torch.int64) * frame_n
    ).view(b, 1, 1)
    base = row + batch_offset

    depth_flat = norm_depth.float().reshape(-1)
    color_flat = image.permute(1, 0, 2, 3).reshape(c, -1)
    zbuf = torch.full(
        (b * frame_n,), -float("inf"), device=device, dtype=torch.float32
    )

    candidates = []
    for xi, wi in ((x0, w0), (x1, w1)):
        valid = (xi >= 0) & (xi < w) & (wi > 0)
        src = torch.nonzero(valid.reshape(-1), as_tuple=False).squeeze(1)
        if src.numel() == 0:
            continue
        idx = (base + xi.clamp(0, w - 1)).reshape(-1)[src]
        z = depth_flat[src]
        zbuf.scatter_reduce_(0, idx, z, reduce="amax", include_self=True)
        candidates.append((src, idx, wi.reshape(-1)[src]))

    out = torch.zeros((c, b * frame_n), device=device, dtype=dtype)
    weight = torch.zeros((b * frame_n,), device=device, dtype=torch.float32)
    for src, idx, wi in candidates:
        z = depth_flat[src]
        wins = z >= (zbuf[idx] - z_epsilon)
        if not torch.any(wins):
            continue
        srcw = src[wins]
        idxw = idx[wins]
        ww = wi[wins].float()
        vals = color_flat[:, srcw] * ww.to(dtype).unsqueeze(0)
        out.scatter_add_(1, idxw.unsqueeze(0).expand(c, -1), vals)
        weight.scatter_add_(0, idxw, ww)

    valid_out = weight > 1e-8
    out[:, valid_out] /= weight[valid_out].to(dtype).unsqueeze(0)
    holes = (~valid_out).reshape(b, h, w)
    warped_depth = zbuf.reshape(b, h, w)
    warped_depth = torch.where(
        torch.isfinite(warped_depth), warped_depth, torch.ones_like(warped_depth)
    )
    out = out.reshape(c, b, h, w).permute(1, 0, 2, 3)
    return out, holes, warped_depth


def _gather_width_batch(image: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    b, c, h, w = image.shape
    idx = indices.clamp(0, w - 1).unsqueeze(1).expand(-1, c, -1, -1)
    return torch.gather(image, 3, idx)


def _gather_depth_batch(depth: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return torch.gather(depth, 2, indices.clamp(0, depth.shape[-1] - 1))


def fill_holes_background_batch(
    image: torch.Tensor,
    holes: torch.Tensor,
    warped_depth: torch.Tensor,
    max_distance: int = 160,
) -> torch.Tensor:
    """Batch-exact form of :func:`fill_holes_background`."""
    if not torch.any(holes):
        return image

    b, c, h, w = image.shape
    device = holes.device
    valid = ~holes
    x = torch.arange(w, device=device, dtype=torch.int64).view(1, 1, w).expand(b, h, -1)

    left_seed = torch.where(valid, x, torch.full_like(x, -1))
    left_idx = torch.cummax(left_seed, dim=2).values

    valid_rev = torch.flip(valid, dims=[2])
    xr = torch.arange(w, device=device, dtype=torch.int64).view(1, 1, w).expand(b, h, -1)
    right_seed_rev = torch.where(valid_rev, xr, torch.full_like(xr, -1))
    right_rev = torch.cummax(right_seed_rev, dim=2).values
    right_idx = torch.where(
        right_rev >= 0,
        (w - 1) - torch.flip(right_rev, dims=[2]),
        torch.full_like(right_rev, -1),
    )

    left_ok = left_idx >= 0
    right_ok = right_idx >= 0
    left_dist = torch.where(left_ok, x - left_idx, torch.full_like(x, 10**9))
    right_dist = torch.where(right_ok, right_idx - x, torch.full_like(x, 10**9))
    left_ok &= left_dist <= max_distance
    right_ok &= right_dist <= max_distance

    left_img = _gather_width_batch(image, left_idx)
    right_img = _gather_width_batch(image, right_idx)
    left_depth = _gather_depth_batch(warped_depth, left_idx)
    right_depth = _gather_depth_batch(warped_depth, right_idx)

    prefer_left = left_depth < right_depth
    similar = torch.abs(left_depth - right_depth) < 0.02
    prefer_left = torch.where(similar, left_dist <= right_dist, prefer_left)
    prefer_left = (prefer_left & left_ok) | (~right_ok & left_ok)
    has_candidate = left_ok | right_ok

    chosen = torch.where(prefer_left.unsqueeze(1), left_img, right_img)
    fill_mask = holes & has_candidate
    result = torch.where(fill_mask.unsqueeze(1), chosen, image)

    remaining = holes & ~has_candidate
    if torch.any(remaining):
        nearest_left = left_dist <= right_dist
        fallback = torch.where(nearest_left.unsqueeze(1), left_img, right_img)
        result = torch.where(remaining.unsqueeze(1), fallback, result)
    return result


def _render_zbuffer_batched(
    rgb: torch.Tensor,
    norm_depth: torch.Tensor,
    device: torch.device,
    output_mode: str,
    max_disparity_eye_px: float,
    zero_parallax: float,
    depth_gamma: float,
    hole_fill: bool,
    max_fill_distance: int,
    cpu_dtype: torch.dtype,
    chunk_size: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Render the reference z-buffer algorithm in small frame batches.

    A two-frame chunk is deliberately conservative for 4K on a 24 GB GPU while
    VDA Large remains resident.  If ROCm reports OOM, the caller retries with
    chunk_size=1, preserving output rather than failing the whole long-video run.
    """
    left_chunks = []
    right_chunks = []
    lmask_chunks = []
    rmask_chunks = []

    for start in range(0, len(rgb), chunk_size):
        end = min(len(rgb), start + chunk_size)
        frames = rgb[start:end].permute(0, 3, 1, 2).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        b, c, h, w = frames.shape
        eye_w = w // 2 if output_mode == "half_sbs" else w
        separation_full = max_disparity_eye_px * (w / eye_w)

        d = norm_depth[start:end].unsqueeze(1).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        d = F.interpolate(d, size=(h, w), mode="bilinear", align_corners=False)[:, 0]
        d = d.clamp(0, 1).pow(depth_gamma)

        left, lholes, ldepth = _forward_warp_batch(
            frames, d, separation_full, zero_parallax, +1.0
        )
        right, rholes, rdepth = _forward_warp_batch(
            frames, d, separation_full, zero_parallax, -1.0
        )
        if hole_fill:
            left = fill_holes_background_batch(
                left, lholes, ldepth, max_fill_distance
            )
            right = fill_holes_background_batch(
                right, rholes, rdepth, max_fill_distance
            )

        if eye_w != w:
            left = F.interpolate(
                left, size=(h, eye_w), mode="bilinear",
                align_corners=False, antialias=True
            )
            right = F.interpolate(
                right, size=(h, eye_w), mode="bilinear",
                align_corners=False, antialias=True
            )
            lholes = F.interpolate(
                lholes.float().unsqueeze(1), size=(h, eye_w), mode="nearest"
            )[:, 0] > 0.5
            rholes = F.interpolate(
                rholes.float().unsqueeze(1), size=(h, eye_w), mode="nearest"
            )[:, 0] > 0.5

        left_chunks.append(
            left.clamp(0, 1).permute(0, 2, 3, 1).to("cpu", dtype=cpu_dtype)
        )
        right_chunks.append(
            right.clamp(0, 1).permute(0, 2, 3, 1).to("cpu", dtype=cpu_dtype)
        )
        lmask_chunks.append(lholes.to("cpu", dtype=torch.float32))
        rmask_chunks.append(rholes.to("cpu", dtype=torch.float32))

        del frames, d, left, right, lholes, rholes, ldepth, rdepth

    return (
        torch.cat(left_chunks, dim=0),
        torch.cat(right_chunks, dim=0),
        torch.cat(lmask_chunks, dim=0),
        torch.cat(rmask_chunks, dim=0),
    )


def _render_zbuffer_pair(
    rgb: torch.Tensor,
    norm_depth: torch.Tensor,
    device: torch.device,
    output_mode: str,
    max_disparity_eye_px: float,
    zero_parallax: float,
    depth_gamma: float,
    hole_fill: bool,
    max_fill_distance: int,
    cpu_dtype: torch.dtype,
):
    """Reference z-buffer with eye-pair fusion only for regular ops.

    The expensive scatter/reduce remains separate per eye. Hole filling,
    half-SBS resize, mask resize, and D2H copies are paired.
    """
    left_out, right_out, left_masks, right_masks = [], [], [], []
    for i in range(len(rgb)):
        frame = rgb[i].permute(2, 0, 1).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        h, w = frame.shape[-2:]
        eye_w = w // 2 if output_mode == "half_sbs" else w
        separation_full = max_disparity_eye_px * (w / eye_w)

        d = norm_depth[i].unsqueeze(0).unsqueeze(0).to(
            device=device, dtype=torch.float32
        )
        d = F.interpolate(d, size=(h, w), mode="bilinear", align_corners=False)[0, 0]
        d = d.clamp(0, 1).pow(depth_gamma)

        (left, lholes, ldepth), (right, rholes, rdepth) = _forward_warp_pair(
            frame, d, separation_full, zero_parallax
        )

        eyes = torch.stack((left, right), dim=0)
        holes = torch.stack((lholes, rholes), dim=0)
        depths = torch.stack((ldepth, rdepth), dim=0)
        if hole_fill:
            eyes = fill_holes_background_batch(
                eyes, holes, depths, max_fill_distance
            )

        if eye_w != w:
            eyes = F.interpolate(
                eyes, size=(h, eye_w), mode="bilinear",
                align_corners=False, antialias=True
            )
            holes = F.interpolate(
                holes.float().unsqueeze(1), size=(h, eye_w), mode="nearest"
            )[:, 0] > 0.5

        eyes_cpu = eyes.clamp(0, 1).permute(0, 2, 3, 1).to(
            "cpu", dtype=cpu_dtype
        )
        holes_cpu = holes.to("cpu", dtype=torch.float32)
        left_out.append(eyes_cpu[0])
        right_out.append(eyes_cpu[1])
        left_masks.append(holes_cpu[0])
        right_masks.append(holes_cpu[1])

        del frame, d, left, right, lholes, rholes, ldepth, rdepth, eyes, holes, depths

    return (
        torch.stack(left_out, dim=0),
        torch.stack(right_out, dim=0),
        torch.stack(left_masks, dim=0),
        torch.stack(right_masks, dim=0),
    )




def _render_zbuffer_async_output(
    rgb: torch.Tensor,
    norm_depth: torch.Tensor,
    device: torch.device,
    output_mode: str,
    max_disparity_eye_px: float,
    zero_parallax: float,
    depth_gamma: float,
    hole_fill: bool,
    max_fill_distance: int,
    cpu_dtype: torch.dtype,
):
    """Reference z-buffer with overlapped pinned D2H output.

    Rendering math is intentionally identical to ``zbuffer``.  The only
    difference is output transport: final eye tensors are copied into their
    preallocated CPU batch slots, avoiding the per-frame CPU allocations and
    final ``torch.stack``.  On ROCm/CUDA a dedicated copy stream lets D2H for
    frame N overlap compute for frame N+1.
    """
    b = len(rgb)
    h = int(rgb.shape[1])
    w = int(rgb.shape[2])
    eye_w = w // 2 if output_mode == "half_sbs" else w

    use_async = device.type == "cuda" and torch.cuda.is_available()
    try:
        left_out = torch.empty(
            (b, h, eye_w, 3), dtype=cpu_dtype, device="cpu", pin_memory=use_async
        )
        right_out = torch.empty(
            (b, h, eye_w, 3), dtype=cpu_dtype, device="cpu", pin_memory=use_async
        )
        # Keep masks compact during transfer; convert once after the batch.
        left_masks_bool = torch.empty(
            (b, h, eye_w), dtype=torch.bool, device="cpu", pin_memory=use_async
        )
        right_masks_bool = torch.empty(
            (b, h, eye_w), dtype=torch.bool, device="cpu", pin_memory=use_async
        )
    except RuntimeError:
        # Some environments disable host registration/pinned allocations.
        # Preserve correctness by using the same direct-preallocation path
        # without asynchronous copies.
        use_async = False
        left_out = torch.empty((b, h, eye_w, 3), dtype=cpu_dtype)
        right_out = torch.empty((b, h, eye_w, 3), dtype=cpu_dtype)
        left_masks_bool = torch.empty((b, h, eye_w), dtype=torch.bool)
        right_masks_bool = torch.empty((b, h, eye_w), dtype=torch.bool)

    copy_stream = torch.cuda.Stream(device=device) if use_async else None
    compute_stream = torch.cuda.current_stream(device) if use_async else None

    for i in range(b):
        frame = rgb[i].permute(2, 0, 1).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        separation_full = max_disparity_eye_px * (w / eye_w)

        d = norm_depth[i].unsqueeze(0).unsqueeze(0).to(
            device=device, dtype=torch.float32
        )
        d = F.interpolate(d, size=(h, w), mode="bilinear", align_corners=False)[0, 0]
        d = d.clamp(0, 1).pow(depth_gamma)

        left, lholes, ldepth = _forward_warp_one(
            frame, d, separation_full, zero_parallax, +1.0
        )
        right, rholes, rdepth = _forward_warp_one(
            frame, d, separation_full, zero_parallax, -1.0
        )
        if hole_fill:
            left = fill_holes_background(left, lholes, ldepth, max_fill_distance)
            right = fill_holes_background(right, rholes, rdepth, max_fill_distance)

        if eye_w != w:
            left = F.interpolate(
                left.unsqueeze(0), size=(h, eye_w), mode="bilinear",
                align_corners=False, antialias=True
            )[0]
            right = F.interpolate(
                right.unsqueeze(0), size=(h, eye_w), mode="bilinear",
                align_corners=False, antialias=True
            )[0]
            lholes = F.interpolate(
                lholes.float().view(1, 1, h, w), size=(h, eye_w), mode="nearest"
            )[0, 0] > 0.5
            rholes = F.interpolate(
                rholes.float().view(1, 1, h, w), size=(h, eye_w), mode="nearest"
            )[0, 0] > 0.5

        # Match the reference output cast exactly, but pack HWC contiguously on
        # the GPU so the host transfer itself is a simple contiguous DMA.
        left_src = left.clamp(0, 1).to(dtype=cpu_dtype).permute(1, 2, 0).contiguous()
        right_src = right.clamp(0, 1).to(dtype=cpu_dtype).permute(1, 2, 0).contiguous()

        if use_async:
            ready = torch.cuda.Event()
            ready.record(compute_stream)
            with torch.cuda.stream(copy_stream):
                copy_stream.wait_event(ready)
                left_out[i].copy_(left_src, non_blocking=True)
                right_out[i].copy_(right_src, non_blocking=True)
                left_masks_bool[i].copy_(lholes, non_blocking=True)
                right_masks_bool[i].copy_(rholes, non_blocking=True)
            # Tell the caching allocator these storages remain in use by the
            # copy stream even though Python drops the references this loop.
            left_src.record_stream(copy_stream)
            right_src.record_stream(copy_stream)
            lholes.record_stream(copy_stream)
            rholes.record_stream(copy_stream)
        else:
            left_out[i].copy_(left_src.to("cpu"))
            right_out[i].copy_(right_src.to("cpu"))
            left_masks_bool[i].copy_(lholes.to("cpu"))
            right_masks_bool[i].copy_(rholes.to("cpu"))

        del frame, d, left, right, lholes, rholes, ldepth, rdepth, left_src, right_src

    if use_async:
        copy_stream.synchronize()

    # Reference zbuffer returns float masks.  Convert once per full batch rather
    # than allocating/stacking 44 separate CPU mask tensors.
    return left_out, right_out, left_masks_bool.float(), right_masks_bool.float()

def _render_zbuffer_profile(
    rgb: torch.Tensor,
    norm_depth: torch.Tensor,
    device: torch.device,
    output_mode: str,
    max_disparity_eye_px: float,
    zero_parallax: float,
    depth_gamma: float,
    hole_fill: bool,
    max_fill_distance: int,
    cpu_dtype: torch.dtype,
):
    """Diagnostic-only reference z-buffer with per-stage timing.

    The rendering operations intentionally mirror the production ``zbuffer``
    path.  On ROCm/CUDA, torch.cuda.Event is used so reported stage times are
    device times rather than Python dispatch times.  The final output is kept
    identical to the reference renderer; the extra synchronization makes this
    mode unsuitable for production throughput benchmarking.
    """
    import time

    left_out = []
    right_out = []
    left_masks = []
    right_masks = []
    stage_pairs = {}
    cpu_stage = {}

    use_events = device.type == "cuda" and torch.cuda.is_available()

    def begin(name):
        if use_events:
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            a.record()
            return (name, a, b, None)
        return (name, None, None, time.perf_counter())

    def end(token):
        name, a, b, t0 = token
        if use_events:
            b.record()
            stage_pairs.setdefault(name, []).append((a, b))
        else:
            cpu_stage[name] = cpu_stage.get(name, 0.0) + (time.perf_counter() - t0) * 1000.0

    if use_events:
        torch.cuda.synchronize(device)
    wall0 = time.perf_counter()

    for i in range(len(rgb)):
        tok = begin("rgb_h2d")
        frame = rgb[i].permute(2, 0, 1).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        end(tok)

        h, w = frame.shape[-2:]
        eye_w = w // 2 if output_mode == "half_sbs" else w
        separation_full = max_disparity_eye_px * (w / eye_w)

        tok = begin("depth_prepare")
        d = norm_depth[i].unsqueeze(0).unsqueeze(0).to(
            device=device, dtype=torch.float32
        )
        d = F.interpolate(d, size=(h, w), mode="bilinear", align_corners=False)[0, 0]
        d = d.clamp(0, 1).pow(depth_gamma)
        end(tok)

        tok = begin("warp_left")
        left, lholes, ldepth = _forward_warp_one(
            frame, d, separation_full, zero_parallax, +1.0
        )
        end(tok)

        tok = begin("warp_right")
        right, rholes, rdepth = _forward_warp_one(
            frame, d, separation_full, zero_parallax, -1.0
        )
        end(tok)

        if hole_fill:
            tok = begin("fill_left")
            left = fill_holes_background(left, lholes, ldepth, max_fill_distance)
            end(tok)
            tok = begin("fill_right")
            right = fill_holes_background(right, rholes, rdepth, max_fill_distance)
            end(tok)

        if eye_w != w:
            tok = begin("halfsbs_rgb_resize")
            left = F.interpolate(
                left.unsqueeze(0), size=(h, eye_w), mode="bilinear",
                align_corners=False, antialias=True
            )[0]
            right = F.interpolate(
                right.unsqueeze(0), size=(h, eye_w), mode="bilinear",
                align_corners=False, antialias=True
            )[0]
            end(tok)

            tok = begin("halfsbs_mask_resize")
            lholes = F.interpolate(
                lholes.float().view(1, 1, h, w), size=(h, eye_w), mode="nearest"
            )[0, 0] > 0.5
            rholes = F.interpolate(
                rholes.float().view(1, 1, h, w), size=(h, eye_w), mode="nearest"
            )[0, 0] > 0.5
            end(tok)

        tok = begin("rgb_d2h")
        left_cpu = left.clamp(0, 1).permute(1, 2, 0).to("cpu", dtype=cpu_dtype)
        right_cpu = right.clamp(0, 1).permute(1, 2, 0).to("cpu", dtype=cpu_dtype)
        end(tok)

        tok = begin("mask_d2h")
        lmask_cpu = lholes.to("cpu")
        rmask_cpu = rholes.to("cpu")
        end(tok)

        left_out.append(left_cpu)
        right_out.append(right_cpu)
        left_masks.append(lmask_cpu)
        right_masks.append(rmask_cpu)

        del frame, d, left, right, lholes, rholes, ldepth, rdepth

    if use_events:
        torch.cuda.synchronize(device)
        stage_ms = {
            name: sum(a.elapsed_time(b) for a, b in pairs)
            for name, pairs in stage_pairs.items()
        }
    else:
        stage_ms = cpu_stage

    stack0 = time.perf_counter()
    result = (
        torch.stack(left_out, dim=0),
        torch.stack(right_out, dim=0),
        torch.stack(left_masks, dim=0).float(),
        torch.stack(right_masks, dim=0).float(),
    )
    stack_ms = (time.perf_counter() - stack0) * 1000.0
    wall_ms = (time.perf_counter() - wall0) * 1000.0

    stage_ms["cpu_output_stack"] = stack_ms
    ordered = [
        "rgb_h2d", "depth_prepare", "warp_left", "warp_right",
        "fill_left", "fill_right", "halfsbs_rgb_resize",
        "halfsbs_mask_resize", "rgb_d2h", "mask_d2h",
        "cpu_output_stack",
    ]
    accounted = sum(stage_ms.get(k, 0.0) for k in ordered)
    print("[LongVideo DIBR Profile] reference zbuffer")
    print(
        f"[LongVideo DIBR Profile] frames={len(rgb)} size={rgb.shape[2]}x{rgb.shape[1]} "
        f"output_mode={output_mode} device={device} wall={wall_ms:.1f}ms"
    )
    for name in ordered:
        ms = stage_ms.get(name, 0.0)
        pct = (100.0 * ms / accounted) if accounted > 0 else 0.0
        print(
            f"[LongVideo DIBR Profile] {name:22s} {ms:9.1f} ms  "
            f"{pct:5.1f}%  ({ms/max(1,len(rgb)):.2f} ms/frame)"
        )
    print(
        f"[LongVideo DIBR Profile] accounted={accounted:.1f}ms "
        f"wall={wall_ms:.1f}ms"
    )
    return result

def render_stereo_batch(
    rgb: torch.Tensor,
    norm_depth: torch.Tensor,
    device: torch.device,
    output_mode: str = "half_sbs",
    max_disparity_eye_px: float = 36.0,
    zero_parallax: float = 0.15,
    depth_gamma: float = 1.0,
    renderer: str = "zbuffer",
    hole_fill: bool = True,
    max_fill_distance: int = 160,
    cpu_dtype: torch.dtype = torch.float16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(rgb) != len(norm_depth):
        raise ValueError("RGB and normalized depth lengths differ")
    if len(rgb) == 0:
        return (
            torch.empty((0, 1, 1, 3)),
            torch.empty((0, 1, 1, 3)),
            torch.empty((0, 1, 1)),
            torch.empty((0, 1, 1)),
        )

    if renderer == "zbuffer_async":
        return _render_zbuffer_async_output(
            rgb=rgb, norm_depth=norm_depth, device=device, output_mode=output_mode,
            max_disparity_eye_px=max_disparity_eye_px, zero_parallax=zero_parallax,
            depth_gamma=depth_gamma, hole_fill=hole_fill,
            max_fill_distance=max_fill_distance, cpu_dtype=cpu_dtype,
        )

    if renderer == "zbuffer_profile":
        return _render_zbuffer_profile(
            rgb=rgb, norm_depth=norm_depth, device=device, output_mode=output_mode,
            max_disparity_eye_px=max_disparity_eye_px, zero_parallax=zero_parallax,
            depth_gamma=depth_gamma, hole_fill=hole_fill,
            max_fill_distance=max_fill_distance, cpu_dtype=cpu_dtype,
        )

    if renderer == "zbuffer_pair":
        return _render_zbuffer_pair(
            rgb=rgb, norm_depth=norm_depth, device=device, output_mode=output_mode,
            max_disparity_eye_px=max_disparity_eye_px, zero_parallax=zero_parallax,
            depth_gamma=depth_gamma, hole_fill=hole_fill,
            max_fill_distance=max_fill_distance, cpu_dtype=cpu_dtype,
        )

    if renderer == "zbuffer_batched":
        try:
            return _render_zbuffer_batched(
                rgb=rgb,
                norm_depth=norm_depth,
                device=device,
                output_mode=output_mode,
                max_disparity_eye_px=max_disparity_eye_px,
                zero_parallax=zero_parallax,
                depth_gamma=depth_gamma,
                hole_fill=hole_fill,
                max_fill_distance=max_fill_distance,
                cpu_dtype=cpu_dtype,
                chunk_size=2,
            )
        except torch.OutOfMemoryError:
            # Conservative long-video fallback. This is the same batched math
            # with a single frame per chunk, so quality/output semantics remain
            # unchanged if the resident VDA model leaves insufficient headroom.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return _render_zbuffer_batched(
                rgb=rgb,
                norm_depth=norm_depth,
                device=device,
                output_mode=output_mode,
                max_disparity_eye_px=max_disparity_eye_px,
                zero_parallax=zero_parallax,
                depth_gamma=depth_gamma,
                hole_fill=hole_fill,
                max_fill_distance=max_fill_distance,
                cpu_dtype=cpu_dtype,
                chunk_size=1,
            )

    left_out = []
    right_out = []
    left_masks = []
    right_masks = []

    for i in range(len(rgb)):
        frame = rgb[i].permute(2, 0, 1).to(device=device, dtype=torch.float32, non_blocking=True)
        h, w = frame.shape[-2:]
        eye_w = w // 2 if output_mode == "half_sbs" else w
        # User disparity is defined in final stored eye pixels. Render at source
        # width, then downsample horizontally for half-SBS.
        separation_full = max_disparity_eye_px * (w / eye_w)

        d = norm_depth[i].unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)
        d = F.interpolate(d, size=(h, w), mode="bilinear", align_corners=False)[0, 0]
        d = d.clamp(0, 1).pow(depth_gamma)

        if renderer == "grid_sample":
            left, lholes = _grid_sample_eye(frame, d, separation_full, zero_parallax, +1.0)
            right, rholes = _grid_sample_eye(frame, d, separation_full, zero_parallax, -1.0)
        else:
            warp_fn = _forward_warp_one_fused if renderer == "zbuffer_fused" else _forward_warp_one
            left, lholes, ldepth = warp_fn(frame, d, separation_full, zero_parallax, +1.0)
            right, rholes, rdepth = warp_fn(frame, d, separation_full, zero_parallax, -1.0)
            if hole_fill:
                left = fill_holes_background(left, lholes, ldepth, max_fill_distance)
                right = fill_holes_background(right, rholes, rdepth, max_fill_distance)

        if eye_w != w:
            left = F.interpolate(
                left.unsqueeze(0), size=(h, eye_w), mode="bilinear", align_corners=False, antialias=True
            )[0]
            right = F.interpolate(
                right.unsqueeze(0), size=(h, eye_w), mode="bilinear", align_corners=False, antialias=True
            )[0]
            lholes = F.interpolate(lholes.float().view(1, 1, h, w), size=(h, eye_w), mode="nearest")[0, 0] > 0.5
            rholes = F.interpolate(rholes.float().view(1, 1, h, w), size=(h, eye_w), mode="nearest")[0, 0] > 0.5

        left_out.append(left.clamp(0, 1).permute(1, 2, 0).to("cpu", dtype=cpu_dtype))
        right_out.append(right.clamp(0, 1).permute(1, 2, 0).to("cpu", dtype=cpu_dtype))
        left_masks.append(lholes.to("cpu"))
        right_masks.append(rholes.to("cpu"))

        del frame, d, left, right

    # Keep the ROCm/CUDA caching allocator warm across Comfy meta-batches.
    # `torch.cuda.empty_cache()` does not free live tensors; it releases cached
    # blocks back to the driver and can force synchronization/reallocation on the
    # next batch. ComfyUI/PyTorch can reclaim cached blocks automatically when a
    # later allocation actually needs them.

    return (
        torch.stack(left_out, dim=0),
        torch.stack(right_out, dim=0),
        torch.stack(left_masks, dim=0).float(),
        torch.stack(right_masks, dim=0).float(),
    )
