# ISVS/src/correspondence.py
from typing import List, Dict, Tuple, Optional
import torch

def _grid_center_xy(W, H, gw, gh, qx, qy):
    px_x = W / gw; px_y = H / gh
    return (qx + 0.5) * px_x, (qy + 0.5) * px_y

def compute_correspondences_oneway(gf_t, gf_c, picks) -> List[Dict]:
    Ft, Fc = gf_t.patch_feat, gf_c.patch_feat
    gh_t, gw_t, (Wt, Ht) = gf_t.gh, gf_t.gw, gf_t.shown_size
    gh_c, gw_c, (Wc, Hc) = gf_c.gh, gf_c.gw, gf_c.shown_size

    arrows = []
    for (qy_src, qx_src, idx_src) in picks:
        qvec = Ft[idx_src]
        sim = Fc @ qvec                            # (Nc,)
        j = int(torch.argmax(sim).item())
        best = float(sim[j].item())
        qy_dst, qx_dst = divmod(j, gw_c)
        x_src, y_src = _grid_center_xy(Wt, Ht, gw_t, gh_t, qx_src, qy_src)
        x_dst, y_dst = _grid_center_xy(Wc, Hc, gw_c, gh_c, qx_dst, qy_dst)
        arrows.append({"src_xy_tgt": (x_src, y_src),
                       "dst_xy_obs": (x_dst, y_dst),
                       "score": best,
                       "grid_src": (qy_src, qx_src)})
    return arrows

def compute_correspondences_mutual(gf_t, gf_c, picks, min_sim: Optional[float]=None) -> List[Dict]:
    Ft, Fc = gf_t.patch_feat, gf_c.patch_feat
    gh_t, gw_t, (Wt, Ht) = gf_t.gh, gf_t.gw, gf_t.shown_size
    gh_c, gw_c, (Wc, Hc) = gf_c.gh, gf_c.gw, gf_c.shown_size
    S = Fc @ Ft.T                                   # (Nc, Nt)

    arrows = []
    for (qy_src, qx_src, idx_src) in picks:
        col = S[:, idx_src]                         # (Nc,)
        j = int(torch.argmax(col).item())
        s_ij = float(col[j].item())
        row = S[j, :]                               # (Nt,)
        i_back = int(torch.argmax(row).item())
        if i_back == idx_src and (min_sim is None or s_ij >= min_sim):
            qy_dst, qx_dst = divmod(j, gw_c)
            x_src, y_src = _grid_center_xy(Wt, Ht, gw_t, gh_t, qx_src, qy_src)
            x_dst, y_dst = _grid_center_xy(Wc, Hc, gw_c, gh_c, qx_dst, qy_dst)
            arrows.append({"src_xy_tgt": (x_src, y_src),
                           "dst_xy_obs": (x_dst, y_dst),
                           "score": s_ij,
                           "grid_src": (qy_src, qx_src)})
    return arrows

def compute_correspondences_mutual_topk(
    gf_t, gf_c, picks, K: int, min_sim: Optional[float]=None
) -> List[Dict]:
    """
    順方向: i -> j*(i) を一意に決定（列最大）
    逆方向: 各 j*(i) の行の Top-K 生成インデックス集合 N_K(j*)
    採用条件: i ∈ N_K(j*) かつ (min_sim is None or S[j*,i] >= min_sim)
    """
    Ft, Fc = gf_t.patch_feat, gf_c.patch_feat
    gh_t, gw_t, (Wt, Ht) = gf_t.gh, gf_t.gw, gf_t.shown_size
    gh_c, gw_c, (Wc, Hc) = gf_c.gh, gf_c.gw, gf_c.shown_size

    S = Fc @ Ft.T                                   # (Nc, Nt)

    arrows = []
    K = max(1, int(K))
    for (qy_src, qx_src, idx_src) in picks:
        # forward
        col = S[:, idx_src]                         # (Nc,)
        j = int(torch.argmax(col).item())
        s_ij = float(col[j].item())

        # backward Top-K
        row = S[j, :]                               # (Nt,)
        # torch.topk は値とインデックスを返す（降順）
        topk_vals, topk_inds = torch.topk(row, k=min(K, row.numel()))
        topk_set = set(int(x) for x in topk_inds.tolist())

        if (idx_src in topk_set) and (min_sim is None or s_ij >= min_sim):
            qy_dst, qx_dst = divmod(j, gw_c)
            x_src, y_src = _grid_center_xy(Wt, Ht, gw_t, gh_t, qx_src, qy_src)
            x_dst, y_dst = _grid_center_xy(Wc, Hc, gw_c, gh_c, qx_dst, qy_dst)
            arrows.append({"src_xy_tgt": (x_src, y_src),
                           "dst_xy_obs": (x_dst, y_dst),
                           "score": s_ij,
                           "grid_src": (qy_src, qx_src)})
    return arrows

def compute_correspondences(
    gf_t, gf_c, picks, mode: str="oneway",
    min_sim: Optional[float]=None, topk: Optional[int]=None
):
    if mode == "mutual_topk":
        return compute_correspondences_mutual_topk(gf_t, gf_c, picks, K=(topk or 1), min_sim=min_sim)
    if mode == "mutual":
        return compute_correspondences_mutual(gf_t, gf_c, picks, min_sim=min_sim)
    return compute_correspondences_oneway(gf_t, gf_c, picks)


def filter_correspondences_for_display(
    arrows: List[Dict],
    stride: int = 3,
    min_score: float = None,
) -> List[Dict]:
    """
    可視化用に矢印を間引く。
    - stride: qy,qxベースのグリッド間引き
    - min_score: 類似度しきい値 (Noneなら無視)

    Returns reduced_arrows (List[Dict]) with the same dict format.
    """
    reduced: List[Dict] = []
    for a in arrows:
        qy, qx = a["grid_src"]
        if stride is not None:
            if (qy % stride) != 0 or (qx % stride) != 0:
                continue
        if min_score is not None and a["score"] < min_score:
            continue
        reduced.append(a)
    return reduced
