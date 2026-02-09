# ISVS/src/mask_to_picks.py
from typing import List, Tuple
import numpy as np
from PIL import Image

Idx = Tuple[int, int, int]  # (qy, qx, idx)

def mask_to_picks(mask_img: Image.Image, gf_t, min_covered_ratio: float = 0.0):
    """
    gf_t: GridFeatures (gh, gw, shown_size=(W,H))
    パッチ矩形は (W/gw, H/gh) で決める。モデルのpatch_sizeに依存しない。
    """
    gh, gw = gf_t.gh, gf_t.gw
    shown_w, shown_h = gf_t.shown_size
    px_x = shown_w / gw
    px_y = shown_h / gh

    mask_resized = mask_img.resize((shown_w, shown_h), Image.NEAREST)
    mask_np = np.array(mask_resized)
    if mask_np.ndim == 3:
        mask_bin = (mask_np.sum(axis=2) > 0).astype(np.uint8)
    else:
        mask_bin = (mask_np > 0).astype(np.uint8)

    picks = []
    for qy in range(gh):
        for qx in range(gw):
            x0 = int(round(qx * px_x)); x1 = int(round((qx + 1) * px_x))
            y0 = int(round(qy * px_y)); y1 = int(round((qy + 1) * px_y))
            patch_region = mask_bin[y0:y1, x0:x1]
            if patch_region.size == 0:
                continue
            if patch_region.mean() >= min_covered_ratio and patch_region.any():
                idx = qy * gw + qx
                picks.append((qy, qx, idx))
    return picks
