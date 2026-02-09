# ISVS/src/match_metrics.py
from __future__ import annotations

from typing import Iterable, Dict, Optional
import numpy as np
import torch

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None


def _grid_centers_from_indices(
    indices: np.ndarray,
    gw: int,
    gh: int,
    shown_size: tuple[int, int],
) -> np.ndarray:
    """Return (N,2) centers in pixel coords for patch indices."""
    W, H = shown_size
    qy = indices // gw
    qx = indices % gw
    xs = (qx + 0.5) * (W / gw)
    ys = (qy + 0.5) * (H / gh)
    return np.stack([xs, ys], axis=1).astype(np.float32)


def _compute_nn_stats(
    Ft: torch.Tensor,
    Fc: torch.Tensor,
    block_size: int = 2048,
):
    """
    Compute:
      - top1/top2 similarity per target patch
      - best observed index per target patch
      - best target index per observed patch
    using blockwise matmul to control memory.
    """
    Ft = Ft.to(torch.float32)
    Fc = Fc.to(torch.float32)
    Nt = Ft.shape[0]
    Nc = Fc.shape[0]

    top1 = torch.empty(Nt, dtype=torch.float32)
    top2 = torch.empty(Nt, dtype=torch.float32)
    best_obs_idx = torch.empty(Nt, dtype=torch.long)

    if Nc < 2:
        top2.fill_(float("nan"))

    best_val_obs = torch.full((Nc,), -1e9, dtype=torch.float32)
    best_idx_obs = torch.full((Nc,), -1, dtype=torch.long)

    for start in range(0, Nt, block_size):
        end = min(start + block_size, Nt)
        Ft_blk = Ft[start:end]  # (B, D)
        sim = Fc @ Ft_blk.T     # (Nc, B)

        if Nc >= 2:
            vals, inds = torch.topk(sim, k=2, dim=0)
            top1[start:end] = vals[0]
            top2[start:end] = vals[1]
            best_obs_idx[start:end] = inds[0]
        else:
            vals, inds = torch.topk(sim, k=1, dim=0)
            top1[start:end] = vals[0]
            best_obs_idx[start:end] = inds[0]

        row_vals, row_inds = torch.max(sim, dim=1)  # (Nc,)
        improved = row_vals > best_val_obs
        if improved.any():
            best_val_obs[improved] = row_vals[improved]
            best_idx_obs[improved] = row_inds[improved] + start

    return top1, top2, best_obs_idx, best_idx_obs


def compute_correspondence_metrics(
    gf_t,
    gf_c,
    picks: Optional[Iterable[tuple[int, int, int]]] = None,
    block_size: int = 2048,
    ransac_reproj_thresh: float = 3.0,
    ransac_max_iters: int = 2000,
    ransac_confidence: float = 0.995,
) -> Dict[str, float]:
    """
    Compute aggregate correspondence metrics:
      - mutual_nn_ratio
      - top1_top2_gap
      - ransac_inlier_ratio
    """
    Ft = gf_t.patch_feat
    Fc = gf_c.patch_feat
    Nt = Ft.shape[0]
    Nc = Fc.shape[0]

    if picks is None:
        target_indices = np.arange(Nt, dtype=np.int64)
    else:
        target_indices = np.array([idx for (_, _, idx) in picks], dtype=np.int64)

    if Nt == 0 or Nc == 0 or target_indices.size == 0:
        return {
            "mutual_nn_ratio": float("nan"),
            "top1_top2_gap": float("nan"),
            "ransac_inlier_ratio": float("nan"),
        }

    top1, top2, best_obs_idx, best_idx_obs = _compute_nn_stats(
        Ft, Fc, block_size=block_size
    )

    target_idx_t = torch.from_numpy(target_indices.astype(np.int64))
    obs_idx_for_target = best_obs_idx[target_idx_t]
    back_target = best_idx_obs[obs_idx_for_target]
    mutual = (back_target == target_idx_t).float().mean().item()

    gap = (top1 - top2).detach().cpu().numpy()
    gap_sel = gap[target_indices]
    gap_mean = float(np.nanmean(gap_sel)) if gap_sel.size > 0 else float("nan")

    # RANSAC inlier ratio (homography in pixel coords)
    ransac_ratio = float("nan")
    if cv2 is not None and target_indices.size >= 4:
        obs_indices = obs_idx_for_target.detach().cpu().numpy().astype(np.int64)
        src_pts = _grid_centers_from_indices(
            target_indices, gf_t.gw, gf_t.gh, gf_t.shown_size
        )
        dst_pts = _grid_centers_from_indices(
            obs_indices, gf_c.gw, gf_c.gh, gf_c.shown_size
        )
        H, mask = cv2.findHomography(
            src_pts,
            dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=float(ransac_reproj_thresh),
            maxIters=int(ransac_max_iters),
            confidence=float(ransac_confidence),
        )
        if mask is not None:
            inlier_count = int(mask.ravel().sum())
            ransac_ratio = inlier_count / float(mask.size)

    return {
        "mutual_nn_ratio": float(mutual),
        "top1_top2_gap": float(gap_mean),
        "ransac_inlier_ratio": float(ransac_ratio),
    }

