# ISVS/src/pca_maps.py
import numpy as np
from typing import Tuple, List, Optional
from PIL import Image
from sklearn.decomposition import PCA
import torch

def _minmax01(x: np.ndarray):
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-8:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)

def _minmax01_with_bounds(x: np.ndarray, mn: float, mx: float):
    if mx - mn < 1e-8:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)

def fit_pca_rgb_transform(
    patch_feats: List[torch.Tensor],
    n_components: int = 3,
):
    """
    複数画像のパッチ特徴をまとめてPCAをfitし、
    その全体でのRGB正規化用min/maxを返す。
    """
    X = np.concatenate([f.detach().cpu().numpy() for f in patch_feats], axis=0)
    pca = PCA(n_components=n_components)
    rgb_all = pca.fit_transform(X)
    mins = rgb_all.min(axis=0)
    maxs = rgb_all.max(axis=0)
    return pca, mins, maxs

def pca_rgb_from_patch_features_with_transform(
    patch_feat: torch.Tensor,  # (N, D) torch, L2済
    gh: int, gw: int,
    upsample_size: Tuple[int, int],  # (ShownW, ShownH)
    pca: PCA,
    mins: Optional[np.ndarray] = None,
    maxs: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    既存PCAにtransformしてRGB化。
    mins/maxsが指定されていればそれを使って色域を揃える。
    """
    X = patch_feat.detach().cpu().numpy()
    rgb3 = pca.transform(X)
    if mins is None or maxs is None:
        mins = rgb3.min(axis=0)
        maxs = rgb3.max(axis=0)
    for c in range(3):
        rgb3[:, c] = _minmax01_with_bounds(rgb3[:, c], mins[c], maxs[c])
    rgb_map = rgb3.reshape(gh, gw, 3)

    ShownW, ShownH = upsample_size
    pil = Image.fromarray((rgb_map * 255).astype(np.uint8))
    up = pil.resize((ShownW, ShownH), Image.BICUBIC)
    return np.asarray(up)

def pca_rgb_from_patch_features_pair(
    patch_feat_a: torch.Tensor,
    gh_a: int, gw_a: int,
    upsample_size_a: Tuple[int, int],
    patch_feat_b: torch.Tensor,
    gh_b: int, gw_b: int,
    upsample_size_b: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    2枚分を同一PCA空間+同一min/maxでRGB化して返す。
    """
    pca, mins, maxs = fit_pca_rgb_transform([patch_feat_a, patch_feat_b], n_components=3)
    rgb_a = pca_rgb_from_patch_features_with_transform(
        patch_feat_a, gh_a, gw_a, upsample_size_a, pca, mins, maxs
    )
    rgb_b = pca_rgb_from_patch_features_with_transform(
        patch_feat_b, gh_b, gw_b, upsample_size_b, pca, mins, maxs
    )
    return rgb_a, rgb_b

def pca_rgb_from_patch_features(
    patch_feat: torch.Tensor,  # (N, D) torch, L2済
    gh: int, gw: int,
    upsample_size: Tuple[int, int],  # (ShownW, ShownH)
) -> np.ndarray:
    """
    (N,D)→PCA(3)→(gh,gw,3)→(ShownH,ShownW,3 uint8)
    PCAは画像ごとにfit（グローバルfitが必要なら別途）。
    """
    X = patch_feat.detach().cpu().numpy()  # (N,D)
    pca = PCA(n_components=3)
    rgb3 = pca.fit_transform(X)            # (N,3)
    for c in range(3):
        rgb3[:, c] = _minmax01(rgb3[:, c])
    rgb_map = rgb3.reshape(gh, gw, 3)

    ShownW, ShownH = upsample_size
    pil = Image.fromarray((rgb_map * 255).astype(np.uint8))
    up = pil.resize((ShownW, ShownH), Image.BICUBIC)
    return np.asarray(up)  # (ShownH,ShownW,3) uint8
