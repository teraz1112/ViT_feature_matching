# ISVS/src/similarity.py
from typing import Tuple
import numpy as np
import torch
from PIL import Image

def minmax01(x: np.ndarray):
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)

def cosine_sim_map(query: torch.Tensor,  # (D,)
                   obs_feats: torch.Tensor,  # (N,D), L2済
                   gh: int, gw: int,
                   upsample_size: Tuple[int, int]) -> np.ndarray:
    """
    クエリ1本と観測パッチのコサイン類似度→(gh,gw)→upsample to (ShownH,ShownW).
    """
    with torch.no_grad():
        sim = (obs_feats @ query)  # (N,)
    sim_map = sim.detach().cpu().numpy().reshape(gh, gw)
    sim_map = minmax01(sim_map)
    # upsample
    ShownW, ShownH = upsample_size
    pil = Image.fromarray((sim_map * 255).astype(np.uint8))
    up = pil.resize((ShownW, ShownH), Image.BICUBIC)
    return np.asarray(up)  # (ShownH,ShownW) uint8
