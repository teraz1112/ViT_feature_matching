# ISVS/src/dashboard.py
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
from pathlib import Path
import csv
from src.io_tools import unique_path

def _pil_to_np(img: Image.Image) -> np.ndarray:
    return np.asarray(img)

def _pair_colors(n: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, 4))
    cmap = plt.get_cmap("hsv")
    return cmap(np.linspace(0, 1, n, endpoint=False))

def _colors_by_y(starts: List[Tuple[float, float]], cmap_name: str = "turbo") -> np.ndarray:
    if not starts:
        return np.zeros((0, 4))
    ys = np.array([p[1] for p in starts], dtype=np.float32)
    y_min = float(ys.min())
    y_max = float(ys.max())
    if y_max - y_min < 1e-6:
        t = np.zeros_like(ys)
    else:
        t = (ys - y_min) / (y_max - y_min)
    cmap = plt.get_cmap(cmap_name)
    return cmap(t)

def _draw_box(img: Image.Image, box: Tuple[int, int, int, int], color=(255, 0, 0), width=3) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle(box, outline=color, width=width)
    return out

def _make_alignment_canvas_and_vectors(
    target_img: Image.Image,
    cropped_img: Image.Image,
    flow_arrows: Optional[List[Dict]],
):
    """
    target_imgとcropped_imgを横連結したRGBキャンバスと、
    そのキャンバス座標上での対応ベクトル始点と終点を返す。
    """
    tgt_np = _pil_to_np(target_img)
    obs_np = _pil_to_np(cropped_img)
    canvas_np = np.concatenate([tgt_np, obs_np], axis=1)

    if not flow_arrows:
        return canvas_np, [], []

    w_tgt = target_img.size[0]

    starts = []
    ends = []
    for a in flow_arrows:
        xs, ys = a["src_xy_tgt"]  # 左画像(ターゲット)座標
        xd, yd = a["dst_xy_obs"]  # 右画像(観測cropped)座標
        starts.append((xs, ys))
        ends.append((xd + w_tgt, yd))  # 観測は右にオフセット

    return canvas_np, starts, ends

def _make_observed_flow_canvas(
    target_img: Image.Image,
    cropped_img: Image.Image,
    flow_arrows: Optional[List[Dict]],
):
    """
    観測側(cropped_img)をベースに、ターゲット上のパッチ座標(src_xy_tgt)を
    観測側座標系にスケールマップして、対応先(dst_xy_obs)と結ぶためのペアを返す。

    returns:
      canvas_np: np.array(H,W,3) 観測画像
      pairs: list of (sx, sy, dx, dy)
        sx,sy : ターゲット側パッチ位置を観測側に射影した座標
        dx,dy : 観測側のマッチ先パッチ座標
    """

    obs_np = _pil_to_np(cropped_img)
    canvas_np = obs_np

    pairs = []
    if not flow_arrows:
        return canvas_np, pairs

    tgt_w, tgt_h = target_img.size  # 注意: PILは (W,H)
    obs_w, obs_h = cropped_img.size

    scale_x = obs_w / tgt_w
    scale_y = obs_h / tgt_h

    for a in flow_arrows:
        (src_x, src_y) = a["src_xy_tgt"]  # target座標
        (dst_x, dst_y) = a["dst_xy_obs"]  # observed座標

        # ターゲット座標を観測側に写す
        sx = src_x * scale_x
        sy = src_y * scale_y

        dx = dst_x
        dy = dst_y

        pairs.append((sx, sy, dx, dy))

    return canvas_np, pairs


def save_similarity_overlay_figure(
    sim_on_cropped,
    save_path: Path,
):
    """
    cropped画像とのオーバーレイをせず、
    類似度ヒートマップのみをカラーマップで保存。
    """
    fig = plt.figure(figsize=(5, 5))
    ax = plt.subplot(1, 1, 1)
    if sim_on_cropped is not None:
        ax.imshow(sim_on_cropped, cmap="jet", interpolation="bilinear")
    ax.axis("off")
    plt.subplots_adjust(0, 0, 1, 1)
    save_path = unique_path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)




def save_flow_on_observed_figure(
    target_img: Image.Image,
    cropped_img: Image.Image,
    flow_arrows: Optional[List[Dict]],
    save_path: Path,
):
    canvas_np, pairs = _make_observed_flow_canvas(target_img, cropped_img, flow_arrows)

    fig = plt.figure(figsize=(5, 5))
    ax = plt.subplot(1, 1, 1)
    ax.imshow(canvas_np)

    if len(pairs) > 0:
        for (sx, sy, dx, dy) in pairs:
            ax.arrow(sx, sy, dx - sx, dy - sy, color="lime", linewidth=1.5,
                     head_width=4, head_length=4, alpha=0.9, length_includes_head=True)
        sx_list = [p[0] for p in pairs]
        sy_list = [p[1] for p in pairs]
        dx_list = [p[2] for p in pairs]
        dy_list = [p[3] for p in pairs]
        ax.scatter(sx_list, sy_list, c="white", s=10, alpha=0.8, edgecolors="black", linewidths=0.5)
        ax.scatter(dx_list, dy_list, c="red", s=10, alpha=0.8, edgecolors="black", linewidths=0.5)

    ax.axis("off")
    plt.subplots_adjust(0, 0, 1, 1)
    save_path = unique_path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)





def save_flow_figure(
    target_img: Image.Image,
    cropped_img: Image.Image,
    flow_arrows: Optional[List[Dict]],
    save_path: Path,
):
    canvas_np, starts, ends = _make_alignment_canvas_and_vectors(target_img, cropped_img, flow_arrows)

    fig = plt.figure(figsize=(10, 5))
    ax = plt.subplot(1, 1, 1)
    ax.imshow(canvas_np)

    if len(starts) > 0:
        colors = _colors_by_y(starts)
        for i, (p0, p1) in enumerate(zip(starts, ends)):
            xs, ys = p0
            xe, ye = p1
            c = colors[i]
            ax.scatter([xs], [ys], s=16, c=[c], edgecolors="black", linewidths=0.4, alpha=0.9)
            ax.scatter([xe], [ye], s=16, c=[c], edgecolors="black", linewidths=0.4, alpha=0.9)

    ax.axis("off")
    plt.subplots_adjust(0, 0, 1, 1)
    save_path = unique_path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def save_flow_lines_figure(
    target_img: Image.Image,
    cropped_img: Image.Image,
    flow_arrows: Optional[List[Dict]],
    save_path: Path,
):
    canvas_np, starts, ends = _make_alignment_canvas_and_vectors(target_img, cropped_img, flow_arrows)

    fig = plt.figure(figsize=(10, 5))
    ax = plt.subplot(1, 1, 1)
    ax.imshow(canvas_np)

    if len(starts) > 0:
        for (p0, p1) in zip(starts, ends):
            xs, ys = p0
            xe, ye = p1
            ax.plot([xs, xe], [ys, ye], color="cyan", linewidth=1.0, alpha=0.7)
            ax.scatter([xs], [ys], c="white", s=10)
            ax.scatter([xe], [ye], c="red", s=10)

    ax.axis("off")
    plt.subplots_adjust(0, 0, 1, 1)
    save_path = unique_path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)



def save_dashboard(
    target_img: Image.Image,
    observed_img: Image.Image,
    cropped_img: Image.Image,
    meta: Dict,
    save_path: Path,
    title: str = "ISVS Dashboard",
    show_on_screen: bool = False,
    pca_target: Optional[np.ndarray] = None,
    pca_cropped: Optional[np.ndarray] = None,
    sim_on_cropped: Optional[np.ndarray] = None,
    draw_query_box_on_target: bool = False,
    query_box_on_target: Optional[Tuple[int,int,int,int]] = None,
    query_boxes_on_target: Optional[List[Tuple[int,int,int,int]]] = None,
    flow_arrows_display: Optional[List[Dict]] = None,
    flow_arrows_full: Optional[List[Dict]] = None,
):
    """
    3x3レイアウト:
      Row1: [1]Target / [2]Observed / [3]ScaledPreview
      Row2: [4]Cropped(+Similarity) / [5]PCA(Target) / [6]PCA(Cropped)
      Row3: [7]Flow arrows (sparse) / [8]Flow lines (sparse) / [9](unused)

    flow_arrows_display: 間引き後（疎）の矢印
    flow_arrows_full:    全対応（現状は使わないがログ的に保持）
    """
    # Row1 preview準備
    scaled_preview = None
    if meta.get("scaled_size") is not None and meta.get("crop_box_on_scaled") is not None:
        scaled_w, scaled_h = meta["scaled_size"]
        scaled = observed_img.resize((scaled_w, scaled_h), Image.BICUBIC)
        draw = ImageDraw.Draw(scaled)
        draw.rectangle(meta["crop_box_on_scaled"], outline=(255, 0, 0), width=3)
        scaled_preview = scaled

    # Row3用キャンバスを作成（疎なflowで）
    alignment_canvas_np, flow_starts, flow_ends = _make_alignment_canvas_and_vectors(
        target_img,
        cropped_img,
        flow_arrows_display,
    )

    fig = plt.figure(figsize=(14, 13))
    fig.suptitle(title)

    # Row1 Col1: Target
    ax1 = plt.subplot(3,3,1)
    timg_disp = target_img
    if draw_query_box_on_target:
        if query_box_on_target is not None:
            timg_disp = _draw_box(timg_disp, query_box_on_target,
                                  color=(255, 64, 0), width=3)
        if query_boxes_on_target:
            tmp_img = timg_disp
            for box in query_boxes_on_target:
                tmp_img = _draw_box(tmp_img, box,
                                    color=(255,128,0), width=2)
            timg_disp = tmp_img
    ax1.imshow(_pil_to_np(timg_disp))
    ax1.set_title(f"Target ({target_img.size[0]}x{target_img.size[1]})")
    ax1.axis("off")

    # Row1 Col2: Observed original
    ax2 = plt.subplot(3,3,2)
    ow, oh = observed_img.size
    ax2.imshow(_pil_to_np(observed_img))
    ax2.set_title(f"Observed original ({ow}x{oh})")
    ax2.axis("off")

    # Row1 Col3: Scaled preview
    ax3 = plt.subplot(3,3,3)
    if scaled_preview is not None:
        sw, sh = meta.get("scaled_size", (None, None))
        box = meta.get("crop_box_on_scaled")
        ax3.imshow(_pil_to_np(scaled_preview))
        ax3.set_title(f"Scaled ({sw}x{sh}) + Box {box}")
    else:
        ax3.text(0.5, 0.5,
                 "No scaled preview\n(mode=center_crop_only)",
                 ha="center", va="center")
    ax3.axis("off")

    # Row2 Col1: Cropped (+Similarity)
    ax4 = plt.subplot(3,3,4)
    ax4.imshow(_pil_to_np(cropped_img))
    if sim_on_cropped is not None:
        ax4.imshow(sim_on_cropped,
                   cmap="jet",
                   alpha=0.45,
                   interpolation="bilinear")
    cw, ch = cropped_img.size
    subtitle4 = f"Cropped ({cw}x{ch})"
    if sim_on_cropped is not None:
        subtitle4 += " + Similarity"
    ax4.set_title(subtitle4)
    ax4.axis("off")

    # Row2 Col2: PCA target
    ax5 = plt.subplot(3,3,5)
    if pca_target is not None:
        ax5.imshow(pca_target)
        ax5.set_title("DINOv3 PCA RGB (Target)")
    else:
        ax5.text(0.5, 0.5, "No PCA (Target)",
                 ha="center", va="center")
    ax5.axis("off")

    # Row2 Col3: PCA cropped
    ax6 = plt.subplot(3,3,6)
    if pca_cropped is not None:
        ax6.imshow(pca_cropped)
        ax6.set_title("DINOv3 PCA RGB (Cropped)")
    else:
        ax6.text(0.5, 0.5, "No PCA (Cropped)",
                 ha="center", va="center")
    ax6.axis("off")

    # Row3 Col1: Flow with arrows (sparse)
    ax7 = plt.subplot(3,3,7)
    ax7.imshow(alignment_canvas_np)
    if len(flow_starts) > 0:
        colors = _colors_by_y(flow_starts)
        for i, (p0, p1) in enumerate(zip(flow_starts, flow_ends)):
            xs, ys = p0
            xe, ye = p1
            c = colors[i]
            ax7.scatter([xs], [ys], s=16, c=[c], edgecolors="black", linewidths=0.4, alpha=0.9)
            ax7.scatter([xe], [ye], s=16, c=[c], edgecolors="black", linewidths=0.4, alpha=0.9)
    ax7.set_title("Semantic correspondence points (sparse)")
    ax7.axis("off")

    # Row3 Col2: Flow with lines/points (sparse)
    ax8 = plt.subplot(3,3,8)
    ax8.imshow(alignment_canvas_np)
    if len(flow_starts) > 0:
        for (p0, p1) in zip(flow_starts, flow_ends):
            xs, ys = p0
            xe, ye = p1
            ax8.plot([xs, xe], [ys, ye],
                     color="cyan", linewidth=1.0, alpha=0.7)
            ax8.scatter([xs], [ys], c="white", s=10)
            ax8.scatter([xe], [ye], c="red",   s=10)
    ax8.set_title("Semantic correspondence flow (sparse, lines)")
    ax8.axis("off")

    # Row3 Col3: 
    ax9 = plt.subplot(3,3,9)
    obs_canvas_np, obs_pairs = _make_observed_flow_canvas(
        target_img,
        cropped_img,
        flow_arrows_display,
    )
    ax9.imshow(obs_canvas_np)
    if len(obs_pairs) > 0:
        sx_list = []
        sy_list = []
        dx_list = []
        dy_list = []
        for (sx, sy, dx, dy) in obs_pairs:
            ax9.arrow(
                sx, sy,
                dx - sx, dy - sy,
                color="lime",
                linewidth=1.5,
                head_width=4,
                head_length=4,
                length_includes_head=True,
                alpha=0.9,
            )
            sx_list.append(sx); sy_list.append(sy)
            dx_list.append(dx); dy_list.append(dy)

        ax9.scatter(sx_list, sy_list,
                    c="white", s=10,
                    alpha=0.8, edgecolors="black", linewidths=0.5)
        ax9.scatter(dx_list, dy_list,
                    c="red",   s=10,
                    alpha=0.8, edgecolors="black", linewidths=0.5)

    ax9.set_title("Flow on observed (sparse)")
    ax9.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = unique_path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)

    if show_on_screen:
        plt.show()

    plt.close(fig)

def collect_flow_on_observed_records(
    target_img: Image.Image,
    cropped_img: Image.Image,
    flow_arrows: List[Dict],
) -> List[Dict]:
    """
    観測画像座標系での誤差ベクトル記録を作る。
    返す各行はCSVにそのまま書けるdict。
    """
    rows: List[Dict] = []
    if not flow_arrows:
        return rows

    tgt_w, tgt_h = target_img.size  # (W,H)
    obs_w, obs_h = cropped_img.size

    sx_scale = obs_w / tgt_w
    sy_scale = obs_h / tgt_h

    for a in flow_arrows:
        src_x, src_y = a["src_xy_tgt"]   # target座標（表示解像度）
        dst_x, dst_y = a["dst_xy_obs"]   # observed座標（表示解像度）

        sx = src_x * sx_scale            # 観測側に写像したsrc
        sy = src_y * sy_scale

        qy, qx = a.get("grid_src", (None, None))
        score = a.get("score", None)

        rows.append({
            "qy": qy,
            "qx": qx,
            "src_x_tgt": float(src_x),
            "src_y_tgt": float(src_y),
            "src_x_obs_mapped": float(sx),
            "src_y_obs_mapped": float(sy),
            "dst_x_obs": float(dst_x),
            "dst_y_obs": float(dst_y),
            "dx": float(dst_x - sx),
            "dy": float(dst_y - sy),
            "score": (float(score) if score is not None else None),
            "obs_w": obs_w,
            "obs_h": obs_h,
            "tgt_w": tgt_w,
            "tgt_h": tgt_h,
        })
    return rows
