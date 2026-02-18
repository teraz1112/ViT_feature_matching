from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace as dc_replace
from typing import Optional


@dataclass
class PairConfig:
    run_name: str
    out_dir: str
    crop_mode: str
    encoder: str
    model_name: str
    vit_patch: int
    long_side: int
    pad_to_multiple: bool
    enable_dino_pca: bool
    draw_query_box: bool
    multi_combine: str
    show_on_screen: bool
    flow_stride: int
    flow_min_score: Optional[float]
    save_flow_figure: bool
    match_mode: str
    mutual_min_sim: float
    mutual_topk_K: int
    feat_mode: str
    feat_layer: int
    feat_last_k: int
    ours_pca_dim: Optional[int] = None
    ours_shallow_layer: int = 3
    ours_clip_feat_mode: str = "layer"
    ours_clip_feat_layer: int = 6
    ours_clip_feat_last_k: int = 4
    ours_use_dino_deep: bool = True
    ours_use_dino_shallow: bool = True
    ours_use_clip: bool = True
    mad_k: Optional[float] = 1.5
    classic_max_features: int = 5000
    classic_ratio: float = 0.85
    classic_max_matches: int = 1000
    classic_use_cross_check: bool = False
    classic_use_mask: bool = True
    classic_sift_contrast: float = 0.01
    classic_sift_edge: float = 10.0
    classic_sift_sigma: float = 1.6
    classic_orb_fast_threshold: int = 5
    classic_orb_scale_factor: float = 1.2
    classic_orb_nlevels: int = 8
    classic_orb_wta_k: int = 2


_SCENARIOS: dict[str, PairConfig] = {
    "dino": PairConfig(
        run_name="dino_run", 
        out_dir="outputs",
        crop_mode="scale_then_center_crop",
        encoder="dino",
        model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        vit_patch=16,
        long_side=1500,
        pad_to_multiple=True,
        enable_dino_pca=True,
        draw_query_box=False,
        multi_combine="mean",
        show_on_screen=False,
        flow_stride=2,
        flow_min_score=None,
        save_flow_figure=True,
        match_mode="oneway",
        mutual_min_sim=0.0,
        mutual_topk_K=10,
        feat_mode="last",
        feat_layer=1,
        feat_last_k=4,
    ),
    "clip": PairConfig(
        run_name="clip_run",
        out_dir="outputs",
        crop_mode="scale_then_center_crop",
        encoder="clip",
        model_name="openai/clip-vit-large-patch14",
        vit_patch=14,
        long_side=1500,
        pad_to_multiple=True,
        enable_dino_pca=True,
        draw_query_box=False,
        multi_combine="mean",
        show_on_screen=False,
        flow_stride=2,
        flow_min_score=None,
        save_flow_figure=True,
        match_mode="oneway",
        mutual_min_sim=0.0,
        mutual_topk_K=5,
        feat_mode="layer",
        feat_layer=6,
        feat_last_k=4,
    ),
    "ours": PairConfig(
        run_name="ours_run",
        out_dir="outputs",
        crop_mode="scale_then_center_crop",
        encoder="ours",
        model_name="facebook/dinov3-vitb16-pretrain-lvd1689m|openai/clip-vit-large-patch14",
        vit_patch=16,
        long_side=1500,
        pad_to_multiple=True,
        enable_dino_pca=True,
        draw_query_box=False,
        multi_combine="mean",
        show_on_screen=False,
        flow_stride=2,
        flow_min_score=None,
        save_flow_figure=True,
        match_mode="oneway",
        mutual_min_sim=0.0,
        mutual_topk_K=30,
        feat_mode="last",
        feat_layer=9,
        feat_last_k=4,
        ours_shallow_layer=6,
        ours_clip_feat_mode="layer",
        ours_clip_feat_layer=6,
        ours_clip_feat_last_k=4,
        ours_pca_dim=None,
        mad_k=1.5,
        ours_use_dino_deep=True,
        ours_use_dino_shallow=False,
        ours_use_clip=True,
    ),
    "sift": PairConfig(
        run_name="sift_run",
        out_dir="outputs",
        crop_mode="scale_then_center_crop",
        encoder="sift",
        model_name="opencv-sift",
        vit_patch=16,
        long_side=0,
        pad_to_multiple=False,
        enable_dino_pca=False,
        draw_query_box=False,
        multi_combine="mean",
        show_on_screen=False,
        flow_stride=1,
        flow_min_score=None,
        save_flow_figure=True,
        match_mode="oneway",
        mutual_min_sim=0.0,
        mutual_topk_K=10,
        feat_mode="last",
        feat_layer=-1,
        feat_last_k=1,
        classic_max_features=5000,
        classic_ratio=0.85,
        classic_max_matches=1000,
        classic_use_cross_check=False,
    ),
    "orb": PairConfig(
        run_name="orb_run",
        out_dir="outputs",
        crop_mode="scale_then_center_crop",
        encoder="orb",
        model_name="opencv-orb",
        vit_patch=16,
        long_side=0,
        pad_to_multiple=False,
        enable_dino_pca=False,
        draw_query_box=False,
        multi_combine="mean",
        show_on_screen=False,
        flow_stride=1,
        flow_min_score=None,
        save_flow_figure=True,
        match_mode="oneway",
        mutual_min_sim=0.0,
        mutual_topk_K=10,
        feat_mode="last",
        feat_layer=-1,
        feat_last_k=1,
        classic_max_features=5000,
        classic_ratio=0.8,
        classic_max_matches=1000,
        classic_use_cross_check=False,
    ),
}


def get_config(scenario: str) -> PairConfig:
    key = "sift" if scenario == "shift" else scenario
    if key not in _SCENARIOS:
        raise KeyError(f"Unknown scenario '{scenario}'")
    return _SCENARIOS[key]


def list_scenarios() -> list[str]:
    return list(_SCENARIOS.keys())


def clone_with_overrides(cfg: PairConfig, **kwargs) -> PairConfig:
    return dc_replace(cfg, **kwargs)
