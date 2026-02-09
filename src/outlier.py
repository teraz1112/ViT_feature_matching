# ISVS/src/outlier.py
from __future__ import annotations
from typing import List, Dict, Tuple
import numpy as np

def mad_inlier_mask_from_rows(
    rows: List[Dict],
    k: float = 3.5,
    min_keep: int = 8,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    rows（collect_flow_on_observed_recordsの出力）に含まれる dx,dy から
    Median + MAD で外れ値を除去する inlier mask を返す。

    判定は r_i = ||(dx_i,dy_i) - median(dx,dy)|| の偏差に対して行う。
    """
    n = len(rows)
    if n == 0:
        return np.zeros((0,), dtype=bool)
    if n < min_keep:
        return np.ones((n,), dtype=bool)

    dx = np.asarray([r["dx"] for r in rows], dtype=np.float64)
    dy = np.asarray([r["dy"] for r in rows], dtype=np.float64)

    med_dx = np.median(dx)
    med_dy = np.median(dy)

    r = np.sqrt((dx - med_dx) ** 2 + (dy - med_dy) ** 2)          # 距離
    med_r = np.median(r)
    mad = np.median(np.abs(r - med_r))

    if mad < eps:
        # ほぼ全点が同じ -> 外れ値判定しない
        return np.ones((n,), dtype=bool)

    sigma = 1.4826 * mad                                         # robust scale
    thr = k * sigma
    mask = np.abs(r - med_r) <= thr                               # 偏差で判定

    # 全落ち防止
    if mask.sum() < max(2, min_keep // 2):
        return np.ones((n,), dtype=bool)

    return mask
