from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image

from config import PairConfig, clone_with_overrides, get_config, list_scenarios
from src.correspondence import compute_correspondences, filter_correspondences_for_display
from src.cropper import crop_to_match
from src.dashboard import (
    collect_flow_on_observed_records,
    save_dashboard,
    save_flow_figure,
    save_flow_lines_figure,
    save_flow_on_observed_figure,
    save_similarity_overlay_figure,
)
from src.io_tools import ensure_outdir, load_image, save_image, save_metrics_as_csv, save_rows_as_csv
from src.mask_to_picks import mask_to_picks
from src.match_metrics import compute_correspondence_metrics
from src.pca_maps import pca_rgb_from_patch_features_pair
from src.similarity import cosine_sim_map

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _load_yaml(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _combine_queries(feats: torch.Tensor, indices: list[int], how: str) -> torch.Tensor:
    stack = feats[indices]
    if how == "median":
        q = stack.median(dim=0).values
    elif how == "max":
        q = stack.max(dim=0).values
    else:
        q = stack.mean(dim=0)
    return torch.nn.functional.normalize(q, dim=-1)


def _np_rgb_to_pil(arr: np.ndarray) -> Image.Image:
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(arr)


def _fit_pca_obs_from_target_mask(Ft: torch.Tensor, idx_list: list[int], n: int):
    X = Ft[idx_list].to(torch.float32)
    K, D = X.shape
    if K < 2:
        raise RuntimeError(f"PCA fit needs >=2 samples, got K={K}")
    mu = X.mean(dim=0, keepdim=True)
    Xc = X - mu
    q_max = min(K - 1, D)
    n_eff = min(int(n), q_max)
    q = max(n_eff, min(32, q_max))
    _, _, V = torch.pca_lowrank(Xc, q=q, center=False)
    W = V[:, :n_eff].contiguous()
    return W, mu.squeeze(0), n_eff


def _project_to_pca_obs(F: torch.Tensor, W: torch.Tensor, mu: torch.Tensor):
    Z = (F.to(torch.float32) - mu) @ W
    return torch.nn.functional.normalize(Z, dim=-1)


def _pca_reduce_pair(Ft: torch.Tensor, Fc: torch.Tensor, n: int):
    X = torch.cat([Ft, Fc], dim=0).to(torch.float32)
    mu = X.mean(dim=0, keepdim=True)
    Xc = X - mu
    D = Xc.shape[1]
    n = int(n)
    if n <= 0 or n >= D:
        return torch.nn.functional.normalize(Ft, dim=-1), torch.nn.functional.normalize(Fc, dim=-1), None, None
    q = min(max(n, 32), min(Xc.shape[0] - 1, D))
    _, _, V = torch.pca_lowrank(Xc, q=q, center=False)
    W = V[:, :n]
    Ft_r = torch.nn.functional.normalize((Ft - mu) @ W, dim=-1)
    Fc_r = torch.nn.functional.normalize((Fc - mu) @ W, dim=-1)
    return Ft_r, Fc_r, W, mu.squeeze(0)


def _grab_observed_from_camera() -> Image.Image:
    try:
        from pypylon import pylon  # type: ignore
    except Exception as e:  # pragma: no cover - depends on local camera setup
        raise RuntimeError("pypylon is required when --use-camera is set.") from e

    tl_factory = pylon.TlFactory.GetInstance()
    devices = tl_factory.EnumerateDevices()
    if not devices:
        raise RuntimeError("Basler camera not found.")
    camera = pylon.InstantCamera(tl_factory.CreateDevice(devices[0]))
    camera.Open()
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    camera.StartGrabbingMax(1)
    grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    try:
        if not grab_result.GrabSucceeded():
            raise RuntimeError("Failed to grab a frame from camera.")
        image = converter.Convert(grab_result)
        frame_bgr = image.Array
        frame_rgb = frame_bgr[..., ::-1].copy()
        return Image.fromarray(frame_rgb)
    finally:
        grab_result.Release()
        camera.Close()


def _list_image_files(dir_path: Path) -> list[Path]:
    if not dir_path.exists():
        return []
    return sorted(p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _make_full_mask(target_img: Image.Image) -> Image.Image:
    return Image.new("RGB", target_img.size, color=(255, 255, 255))


def _ensure_cv2():
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for SIFT/ORB matching.")


def _mask_to_cv2(mask_img: Image.Image | None, size: tuple[int, int]) -> np.ndarray | None:
    if mask_img is None:
        return None
    if mask_img.size != size:
        mask_img = mask_img.resize(size, Image.NEAREST)
    m = np.asarray(mask_img.convert("L"))
    return (m > 0).astype(np.uint8) * 255


def _detect_and_match_classic(
    img_tgt: Image.Image,
    img_obs: Image.Image,
    mask_img: Image.Image | None,
    method: str,
    max_features: int | None,
    ratio: float | None,
    cross_check: bool,
    use_mask: bool = True,
    sift_contrast: float = 0.01,
    sift_edge: float = 10.0,
    sift_sigma: float = 1.6,
    orb_fast_threshold: int = 5,
    orb_scale_factor: float = 1.2,
    orb_nlevels: int = 8,
    orb_wta_k: int = 2,
):
    _ensure_cv2()
    gray_t = np.asarray(img_tgt.convert("L"))
    gray_o = np.asarray(img_obs.convert("L"))
    mask = _mask_to_cv2(mask_img, img_tgt.size) if use_mask else None

    method = method.lower()
    if method in ("sift", "shift"):
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError("cv2.SIFT_create is unavailable. Install opencv-contrib-python.")
        nfeat = int(max_features) if max_features else 0
        detector = cv2.SIFT_create(
            nfeatures=nfeat,
            contrastThreshold=float(sift_contrast),
            edgeThreshold=float(sift_edge),
            sigma=float(sift_sigma),
        )
        norm = cv2.NORM_L2
    elif method == "orb":
        if not hasattr(cv2, "ORB_create"):
            raise RuntimeError("cv2.ORB_create is unavailable.")
        nfeat = int(max_features) if max_features else 2000
        detector = cv2.ORB_create(
            nfeatures=nfeat,
            scaleFactor=float(orb_scale_factor),
            nlevels=int(orb_nlevels),
            WTA_K=int(orb_wta_k),
            fastThreshold=int(orb_fast_threshold),
        )
        norm = cv2.NORM_HAMMING
    else:
        raise ValueError(f"Unknown classic method: {method}")

    kp_t, des_t = detector.detectAndCompute(gray_t, mask)
    kp_o, des_o = detector.detectAndCompute(gray_o, None)
    if des_t is None or des_o is None or len(kp_t) == 0 or len(kp_o) == 0:
        return [], [], []

    if cross_check:
        matcher = cv2.BFMatcher(norm, crossCheck=True)
        good = matcher.match(des_t, des_o)
    else:
        matcher = cv2.BFMatcher(norm, crossCheck=False)
        knn = matcher.knnMatch(des_t, des_o, k=2)
        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if ratio is None or m.distance < ratio * n.distance:
                good.append(m)

    good.sort(key=lambda m: m.distance)
    return kp_t, kp_o, good


def _matches_to_flow_arrows(kp_t, kp_o, matches, max_matches: int | None):
    if max_matches is not None:
        matches = matches[: int(max_matches)]
    arrows = []
    for i, m in enumerate(matches):
        x1, y1 = kp_t[m.queryIdx].pt
        x2, y2 = kp_o[m.trainIdx].pt
        score = 1.0 / (float(m.distance) + 1e-6)
        arrows.append(
            {
                "src_xy_tgt": (float(x1), float(y1)),
                "dst_xy_obs": (float(x2), float(y2)),
                "score": float(score),
                "grid_src": (i, 0),
            }
        )
    return arrows


def _classic_ransac_inlier_ratio(
    kp_t,
    kp_o,
    matches,
    ransac_reproj_thresh: float = 3.0,
    ransac_max_iters: int = 2000,
    ransac_confidence: float = 0.995,
) -> float:
    if cv2 is None or matches is None or len(matches) < 4:
        return float("nan")
    src_pts = np.float32([kp_t[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_o[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(
        src_pts,
        dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(ransac_reproj_thresh),
        maxIters=int(ransac_max_iters),
        confidence=float(ransac_confidence),
    )
    if mask is None:
        return float("nan")
    return float(mask.ravel().sum() / float(mask.size))


def run_pair_pipeline(
    target_path: Path,
    observed_path: Path | None,
    cfg: PairConfig,
    mask_path: Path | None = None,
    out_dir: Path | None = None,
    use_camera: bool = False,
) -> dict[str, Path | None]:
    if out_dir is not None:
        cfg = clone_with_overrides(cfg, out_dir=str(out_dir))
    ensure_outdir(Path(cfg.out_dir))

    if not target_path.exists():
        raise FileNotFoundError(f"Target image not found: {target_path}")
    if not use_camera and (observed_path is None or not observed_path.exists()):
        raise FileNotFoundError(f"Observed image not found: {observed_path}")
    if mask_path is not None and not mask_path.exists():
        raise FileNotFoundError(f"Mask image not found: {mask_path}")

    img_tgt = load_image(target_path)
    target_saved = save_image(img_tgt, Path(cfg.out_dir) / f"{cfg.run_name}_target.png")
    img_obs = _grab_observed_from_camera() if use_camera else load_image(observed_path)
    img_mask = _make_full_mask(img_tgt) if mask_path is None else load_image(mask_path)

    cropped, meta = crop_to_match(target_img=img_tgt, observed_img=img_obs, mode=cfg.crop_mode)
    cropped_path = Path(cfg.out_dir) / f"{cfg.run_name}_observed_cropped.png"
    save_image(cropped, cropped_path)

    classic_mode = cfg.encoder.lower() in ("sift", "orb", "shift")
    if classic_mode:
        kp_t, kp_c, matches = _detect_and_match_classic(
            img_tgt,
            cropped,
            img_mask,
            method=cfg.encoder,
            max_features=getattr(cfg, "classic_max_features", 2000),
            ratio=getattr(cfg, "classic_ratio", 0.75),
            cross_check=bool(getattr(cfg, "classic_use_cross_check", False)),
            use_mask=bool(getattr(cfg, "classic_use_mask", True)),
            sift_contrast=float(getattr(cfg, "classic_sift_contrast", 0.01)),
            sift_edge=float(getattr(cfg, "classic_sift_edge", 10.0)),
            sift_sigma=float(getattr(cfg, "classic_sift_sigma", 1.6)),
            orb_fast_threshold=int(getattr(cfg, "classic_orb_fast_threshold", 5)),
            orb_scale_factor=float(getattr(cfg, "classic_orb_scale_factor", 1.2)),
            orb_nlevels=int(getattr(cfg, "classic_orb_nlevels", 8)),
            orb_wta_k=int(getattr(cfg, "classic_orb_wta_k", 2)),
        )
        flow_arrows_full = _matches_to_flow_arrows(kp_t, kp_c, matches, max_matches=None)
        flow_arrows_display = flow_arrows_full
        if cfg.flow_min_score is not None:
            flow_arrows_display = [a for a in flow_arrows_display if a["score"] >= cfg.flow_min_score]
        max_disp = getattr(cfg, "classic_max_matches", None)
        if max_disp is not None:
            flow_arrows_display = flow_arrows_display[: int(max_disp)]

        metrics = {
            "mutual_nn_ratio": float("nan"),
            "top1_top2_gap": float("nan"),
            "ransac_inlier_ratio": _classic_ransac_inlier_ratio(kp_t, kp_c, matches),
        }
        pca_target = None
        pca_cropped = None
        sim_overlay = None
        query_boxes = None
        tgt_disp = img_tgt
        crp_disp = cropped
    else:
        from src.encoders import get_feature_extractor

        extractor = get_feature_extractor(
            cfg.encoder,
            cfg.model_name,
            feat_mode=cfg.feat_mode,
            feat_layer=cfg.feat_layer,
            feat_last_k=cfg.feat_last_k,
        )
        if cfg.encoder.lower() == "ours":
            extractor.shallow_layer = int(getattr(cfg, "ours_shallow_layer", 3))
            extractor.clip_feat_mode = getattr(cfg, "ours_clip_feat_mode", "layer")
            extractor.clip_feat_layer = int(getattr(cfg, "ours_clip_feat_layer", -1))
            extractor.clip_feat_last_k = int(getattr(cfg, "ours_clip_feat_last_k", 4))
            extractor.use_dino_deep = bool(getattr(cfg, "ours_use_dino_deep", True))
            extractor.use_dino_shallow = bool(getattr(cfg, "ours_use_dino_shallow", True))
            extractor.use_clip = bool(getattr(cfg, "ours_use_clip", True))

        gf_t = extractor.extract(img=img_tgt, vit_patch=cfg.vit_patch, long_side=cfg.long_side, pad_to_multiple=cfg.pad_to_multiple)
        gf_c = extractor.extract(img=cropped, vit_patch=cfg.vit_patch, long_side=cfg.long_side, pad_to_multiple=cfg.pad_to_multiple)

        Ft_raw = gf_t.patch_feat
        Fc_raw = gf_c.patch_feat
        if cfg.encoder.lower() == "ours" and getattr(cfg, "ours_pca_dim", None) is not None:
            n = int(cfg.ours_pca_dim)
            Ft_r, Fc_r, _, _ = _pca_reduce_pair(Ft_raw, Fc_raw, n=n)
            gf_t.patch_feat = Ft_r
            gf_c.patch_feat = Fc_r

        picks = mask_to_picks(mask_img=img_mask, gf_t=gf_t, min_covered_ratio=0.0)
        if len(picks) == 0:
            qy_mid, qx_mid = gf_t.gh // 2, gf_t.gw // 2
            picks = [(qy_mid, qx_mid, qy_mid * gf_t.gw + qx_mid)]

        idx_list = [idx for (_, _, idx) in picks]
        if cfg.encoder.lower() == "ours" and getattr(cfg, "ours_pca_dim", None) is not None:
            n = int(cfg.ours_pca_dim)
            W, mu, _ = _fit_pca_obs_from_target_mask(Ft_raw, idx_list, n=n)
            gf_t.patch_feat = _project_to_pca_obs(Ft_raw, W, mu)
            gf_c.patch_feat = _project_to_pca_obs(Fc_raw, W, mu)

        query_vec = _combine_queries(gf_t.patch_feat, idx_list, cfg.multi_combine)
        sim_overlay = cosine_sim_map(query=query_vec, obs_feats=gf_c.patch_feat, gh=gf_c.gh, gw=gf_c.gw, upsample_size=gf_c.shown_size)

        Wt, Ht = gf_t.shown_size
        px_x_t = Wt / gf_t.gw
        px_y_t = Ht / gf_t.gh
        query_boxes = (
            [
                (qx * px_x_t, qy * px_y_t, (qx + 1) * px_x_t, (qy + 1) * px_y_t)
                for (qy, qx, _) in picks
            ]
            if cfg.draw_query_box
            else None
        )

        flow_arrows_full = compute_correspondences(
            gf_t=gf_t,
            gf_c=gf_c,
            picks=picks,
            mode=cfg.match_mode,
            min_sim=cfg.mutual_min_sim,
            topk=cfg.mutual_topk_K,
        )
        flow_arrows_display = filter_correspondences_for_display(flow_arrows_full, stride=cfg.flow_stride, min_score=cfg.flow_min_score)
        metrics = compute_correspondence_metrics(gf_t=gf_t, gf_c=gf_c, picks=picks)

        pca_target = pca_cropped = None
        if cfg.enable_dino_pca:
            pca_target, pca_cropped = pca_rgb_from_patch_features_pair(
                gf_t.patch_feat,
                gf_t.gh,
                gf_t.gw,
                gf_t.shown_size,
                gf_c.patch_feat,
                gf_c.gh,
                gf_c.gw,
                gf_c.shown_size,
            )
        tgt_disp = img_tgt if img_tgt.size == gf_t.shown_size else img_tgt.resize(gf_t.shown_size)
        crp_disp = cropped if cropped.size == gf_c.shown_size else cropped.resize(gf_c.shown_size)

    dash_path = Path(cfg.out_dir) / f"{cfg.run_name}_dashboard.png"
    save_dashboard(
        target_img=tgt_disp,
        observed_img=img_obs,
        cropped_img=crp_disp,
        meta=meta,
        save_path=dash_path,
        title=f"Pair Correspondence Dashboard | {cfg.run_name}",
        show_on_screen=cfg.show_on_screen,
        pca_target=pca_target,
        pca_cropped=pca_cropped,
        sim_on_cropped=sim_overlay,
        draw_query_box_on_target=bool(query_boxes),
        query_boxes_on_target=query_boxes,
        flow_arrows_display=flow_arrows_display,
        flow_arrows_full=flow_arrows_full,
    )

    flow_only_path = None
    flow_lines_path = None
    if cfg.save_flow_figure:
        flow_only_path = Path(cfg.out_dir) / f"{cfg.run_name}_flow.png"
        save_flow_figure(target_img=tgt_disp, cropped_img=crp_disp, flow_arrows=flow_arrows_display, save_path=flow_only_path)
        flow_lines_path = Path(cfg.out_dir) / f"{cfg.run_name}_flow_lines.png"
        save_flow_lines_figure(target_img=tgt_disp, cropped_img=crp_disp, flow_arrows=flow_arrows_display, save_path=flow_lines_path)

    pca_tgt_path = None
    if pca_target is not None:
        pca_tgt_path = Path(cfg.out_dir) / f"{cfg.run_name}_pca_target.png"
        _np_rgb_to_pil(pca_target).save(pca_tgt_path)

    pca_crp_path = None
    if pca_cropped is not None:
        pca_crp_path = Path(cfg.out_dir) / f"{cfg.run_name}_pca_cropped.png"
        _np_rgb_to_pil(pca_cropped).save(pca_crp_path)

    sim_path = None
    if sim_overlay is not None:
        sim_path = Path(cfg.out_dir) / f"{cfg.run_name}_similarity.png"
        save_similarity_overlay_figure(sim_on_cropped=sim_overlay, save_path=sim_path)

    cropped_only_path = Path(cfg.out_dir) / f"{cfg.run_name}_cropped.png"
    crp_disp.save(cropped_only_path)
    flow_obs_path = Path(cfg.out_dir) / f"{cfg.run_name}_flow_on_observed.png"
    save_flow_on_observed_figure(target_img=tgt_disp, cropped_img=crp_disp, flow_arrows=flow_arrows_display, save_path=flow_obs_path)

    flow_obs_csv = Path(cfg.out_dir) / f"{cfg.run_name}_flow_on_observed.csv"
    rows = collect_flow_on_observed_records(target_img=tgt_disp, cropped_img=crp_disp, flow_arrows=flow_arrows_display)
    save_rows_as_csv(rows, flow_obs_csv, header=["score", "dx", "dy"], sort_by="score", descending=True)

    scores = [r.get("score", None) for r in rows if r.get("score", None) is not None]
    metrics["max_score"] = float(max(scores)) if scores else float("nan")
    metrics["min_score"] = float(min(scores)) if scores else float("nan")

    metrics_csv = Path(cfg.out_dir) / f"{cfg.run_name}_flow_on_observed_summary.csv"
    save_metrics_as_csv(
        metrics,
        metrics_csv,
        header=["mutual_nn_ratio", "top1_top2_gap", "ransac_inlier_ratio", "max_score", "min_score"],
    )

    return {
        "target": target_saved,
        "cropped": cropped_path,
        "dashboard": dash_path,
        "flow": flow_only_path,
        "flow_lines": flow_lines_path,
        "pca_target": pca_tgt_path,
        "pca_cropped": pca_crp_path,
        "similarity": sim_path,
        "flow_observed": flow_obs_path,
        "flow_csv": flow_obs_csv,
        "metrics_csv": metrics_csv,
    }


def run_vit_dataset(vit_root: Path, scenarios: list[str], out_root: Path | None, show_on_screen: bool = False):
    output_prefixes = ("dino_run_", "clip_run_", "ours_run_", "sift_run_", "orb_run_")
    if not vit_root.exists():
        raise FileNotFoundError(f"ViT root not found: {vit_root}")
    for obj_dir in sorted(p for p in vit_root.iterdir() if p.is_dir()):
        gt_dir = obj_dir / "GT"
        gt_images = _list_image_files(gt_dir)
        if not gt_images:
            print(f"[WARN] GT image not found in: {gt_dir}")
            continue
        gt_image = gt_images[0]
        gt_mask = gt_dir / f"{obj_dir.name}_GT_mask.png"
        if not gt_mask.exists():
            gt_mask = None

        dg_dirs = [p for p in obj_dir.iterdir() if p.is_dir() and p.name != "GT"]
        for dg_dir in sorted(dg_dirs):
            dg_images = sorted(
                p
                for p in dg_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.lower().startswith(output_prefixes)
            )
            for dg_image in dg_images:
                if out_root is None:
                    out_dir = dg_image.parent
                else:
                    out_dir = out_root / dg_image.parent.relative_to(vit_root)
                for scenario in scenarios:
                    cfg = clone_with_overrides(get_config(scenario), out_dir=str(out_dir), show_on_screen=show_on_screen)
                    print(f"[INFO] {obj_dir.name} | {dg_image.relative_to(obj_dir)} | {scenario}")
                    run_pair_pipeline(
                        target_path=gt_image,
                        observed_path=dg_image,
                        mask_path=gt_mask,
                        cfg=cfg,
                        out_dir=out_dir,
                        use_camera=False,
                    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pair correspondence estimation and visualization.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_pair = sub.add_parser("pair", help="Run one target/observed pair.")
    p_pair.add_argument("--config", type=Path, default=None, help="Optional YAML config.")
    p_pair.add_argument("--scenario", type=str, default=None, choices=list_scenarios())
    p_pair.add_argument("--target", type=Path, default=None)
    p_pair.add_argument("--observed", type=Path, default=None)
    p_pair.add_argument("--mask", type=Path, default=None)
    p_pair.add_argument("--out-dir", type=Path, default=None)
    p_pair.add_argument("--use-camera", action="store_true")
    p_pair.add_argument("--show-on-screen", action="store_true")

    p_batch = sub.add_parser("batch", help="Run batch for data/ViT style directory.")
    p_batch.add_argument("--config", type=Path, default=None, help="Optional YAML config.")
    p_batch.add_argument("--vit-root", type=Path, default=None)
    p_batch.add_argument("--scenarios", type=str, default=None, help="Comma-separated scenarios.")
    p_batch.add_argument("--out-root", type=Path, default=None)
    p_batch.add_argument("--show-on-screen", action="store_true")
    return parser


def _handle_pair(args):
    cfg_file = _load_yaml(args.config)
    pair_cfg = cfg_file.get("pair", {})

    scenario = args.scenario or pair_cfg.get("scenario", "dino")
    cfg = get_config(scenario)

    target = args.target or (Path(pair_cfg["target"]) if "target" in pair_cfg else None)
    observed = args.observed or (Path(pair_cfg["observed"]) if "observed" in pair_cfg else None)
    mask = args.mask if args.mask is not None else (Path(pair_cfg["mask"]) if pair_cfg.get("mask") else None)
    out_dir = args.out_dir or (Path(pair_cfg["out_dir"]) if pair_cfg.get("out_dir") else Path(cfg.out_dir))
    use_camera = bool(args.use_camera or pair_cfg.get("use_camera", False))
    show_on_screen = bool(args.show_on_screen or pair_cfg.get("show_on_screen", False))

    if target is None:
        raise ValueError("--target is required for pair mode.")
    if not use_camera and observed is None:
        raise ValueError("--observed is required unless --use-camera is set.")

    cfg = replace(cfg, out_dir=str(out_dir), show_on_screen=show_on_screen)
    outputs = run_pair_pipeline(
        target_path=target,
        observed_path=observed,
        mask_path=mask,
        cfg=cfg,
        out_dir=out_dir,
        use_camera=use_camera,
    )
    print("[OK] pair completed")
    for key, value in outputs.items():
        print(f"  {key}: {value}")


def _handle_batch(args):
    cfg_file = _load_yaml(args.config)
    batch_cfg = cfg_file.get("batch", {})
    vit_root = args.vit_root or (Path(batch_cfg["vit_root"]) if batch_cfg.get("vit_root") else None)
    if vit_root is None:
        raise ValueError("--vit-root is required for batch mode.")
    scenario_text = args.scenarios or batch_cfg.get("scenarios", "dino,clip,ours,sift,orb")
    scenarios = [x.strip() for x in scenario_text.split(",") if x.strip()]
    out_root = args.out_root if args.out_root is not None else (Path(batch_cfg["out_root"]) if batch_cfg.get("out_root") else None)
    show_on_screen = bool(args.show_on_screen or batch_cfg.get("show_on_screen", False))
    run_vit_dataset(vit_root=vit_root, scenarios=scenarios, out_root=out_root, show_on_screen=show_on_screen)
    print("[OK] batch completed")


def main():
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "pair":
        _handle_pair(args)
    elif args.command == "batch":
        _handle_batch(args)
    else:  # pragma: no cover
        raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
