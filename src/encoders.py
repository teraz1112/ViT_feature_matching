# ISVS/src/encoders.py
from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn.functional as F
import numpy as np
import math
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from transformers import CLIPImageProcessor, CLIPVisionModel


@dataclass
class GridFeatures:
    patch_feat: torch.Tensor   # (N, D) L2-normalized (torch.float32, device=cpu or cuda)
    gh: int
    gw: int
    shown_size: Tuple[int, int]  # (W, H)

def _select_tokens_from_hidden_states(
    hidden_states,                  # Tuple[Tensor], 各 (B, T, D)
    num_prefix_tokens: int,         # 先頭の除外トークン数 (CLS + register 等)
    feat_mode: str, feat_layer: int, feat_last_k: int
):
    """
    指定層のパッチトークン（先頭 num_prefix_tokens を除外）を返す。
    - "last":        hidden_states[-1][:, num_prefix_tokens:, :]
    - "layer":       hidden_states[idx][:, num_prefix_tokens:, :]
    - "avg_last_k":  mean(hidden_states[-k:])[:, num_prefix_tokens:, :]
    返り値: (1, N, D)
    """
    assert len(hidden_states) >= 1
    if feat_mode == "last":
        hs = hidden_states[-1]
    elif feat_mode == "layer":
        idx = feat_layer if feat_layer >= 0 else (len(hidden_states) + feat_layer)
        hs = hidden_states[idx]
    elif feat_mode == "avg_last_k":
        k = max(1, min(feat_last_k, len(hidden_states)))
        hs = sum(hidden_states[-k:]) / float(k)
    else:
        raise ValueError(f"Unknown feat_mode: {feat_mode}")
    return hs[:, num_prefix_tokens:, :]


def _minmax01(x: np.ndarray):
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-8:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)

def _resize_long_side(img: Image.Image, long_side: int) -> Image.Image:
    if long_side <= 0: return img.copy()
    w, h = img.size
    if max(w, h) == long_side: return img.copy()
    if h >= w:
        new_h = long_side
        new_w = round(w * (long_side / h))
    else:
        new_w = long_side
        new_h = round(h * (long_side / w))
    return img.resize((new_w, new_h), Image.BICUBIC)

def _pad_to_multiple(img: Image.Image, m: int) -> Image.Image:
    w, h = img.size
    new_w = ((w + m - 1)//m)*m
    new_h = ((h + m - 1)//m)*m
    if (new_w, new_h)==(w,h): return img.copy()
    canvas = Image.new("RGB", (new_w, new_h))
    canvas.paste(img, (0,0))
    return canvas

class _BaseExtractor:
    def __init__(self, model_name: str, device: str|None=None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = None
        self.model = None

    def _prep_image(self, img: Image.Image, vit_patch: int, long_side: int, pad_to_multiple: bool):
        img_r = _resize_long_side(img, long_side)
        img_x = _pad_to_multiple(img_r, vit_patch) if pad_to_multiple else img_r
        return img_x, img_r.size, img_x.size  # img_x, shown(size of img_r), actual(size of img_x)

    def extract(self, img: Image.Image, vit_patch: int, long_side: int, pad_to_multiple: bool) -> GridFeatures:
        raise NotImplementedError

class DINOv3Extractor(_BaseExtractor):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        # DINOv3には register token があるモデルがある
        self.num_register = getattr(self.model.config, "num_register_tokens", 0)

    def extract(self, img: Image.Image, vit_patch: int, long_side: int, pad_to_multiple: bool) -> GridFeatures:
        img_x, shown_wh, _ = self._prep_image(img, vit_patch, long_side, pad_to_multiple)
        inputs = self.processor(images=img_x, return_tensors="pt",
                                do_resize=False, do_center_crop=False).to(self.device)
        with torch.inference_mode():
            out = self.model(**inputs, output_hidden_states=True)

        # ★ register token を考慮
        num_register = int(getattr(self.model.config, "num_register_tokens", 0))
        num_prefix = 1 + num_register  # CLS + register

        patch_tokens = _select_tokens_from_hidden_states(
            out.hidden_states,
            num_prefix_tokens=num_prefix,
            feat_mode=self.feat_mode,
            feat_layer=self.feat_layer,
            feat_last_k=self.feat_last_k,
        )                              # (1, N, D)

        N = patch_tokens.shape[1]
        Hin, Win = inputs["pixel_values"].shape[-2:]
        gh, gw = Hin // vit_patch, Win // vit_patch

        # ★ 安全化（万一 prefix 処理が合わなくても reshape 可能に）
        if N > gh * gw:
            patch_tokens = patch_tokens[:, : gh * gw, :]
            N = gh * gw
        elif N < gh * gw:
            raise RuntimeError(f"[DINO] tokens不足: N={N}, gh*gw={gh*gw}（前処理サイズ/patchを確認）")

        feats = torch.nn.functional.normalize(patch_tokens[0], dim=-1).detach().cpu()
        return GridFeatures(patch_feat=feats, gh=gh, gw=gw, shown_size=shown_wh)

class CLIPVisionExtractor(_BaseExtractor):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.processor = CLIPImageProcessor.from_pretrained(model_name)
        self.model = CLIPVisionModel.from_pretrained(model_name, use_safetensors=True).to(self.device).eval()

    def extract(self, img: Image.Image, vit_patch: int, long_side: int, pad_to_multiple: bool) -> GridFeatures:
        patch_size = int(getattr(self.model.config, "patch_size", vit_patch or 32))
        img_x, shown_wh, _ = self._prep_image(img, patch_size, long_side, pad_to_multiple)
        inputs = self.processor(images=img_x, return_tensors="pt",
                                do_resize=False, do_center_crop=False).to(self.device)
        with torch.inference_mode():
            out = self.model(**inputs, interpolate_pos_encoding=True, output_hidden_states=True)

        num_prefix = 1  # CLIPはCLSのみ
        patch_tokens = _select_tokens_from_hidden_states(
            out.hidden_states,
            num_prefix_tokens=num_prefix,
            feat_mode=self.feat_mode,
            feat_layer=self.feat_layer,
            feat_last_k=self.feat_last_k,
        )                                   # (1, N, D)
        N = patch_tokens.shape[1]
        Hin, Win = inputs["pixel_values"].shape[-2:]
        gh, gw = Hin // patch_size, Win // patch_size
        if gh * gw != N:
            g = int(round(N ** 0.5)); gh = gw = g
        if N > gh * gw:
            patch_tokens = patch_tokens[:, : gh * gw, :]
            N = gh * gw
        elif N < gh * gw:
            raise RuntimeError(f"[CLIP] tokens不足: N={N}, gh*gw={gh*gw}")
        feats = torch.nn.functional.normalize(patch_tokens[0], dim=-1).detach().cpu()
        return GridFeatures(patch_feat=feats, gh=gh, gw=gw, shown_size=shown_wh)

class OursFusionExtractor(_BaseExtractor):
    """
    Ours = DINOv3(deep) + DINOv3(shallow layer) + CLIP(last)
    - 基準グリッドは DINO 側 (gh,gw)
    - CLIP 特徴は (gh,gw) に bilinear 補間してから concat
    """
    def __init__(self, dino_model_name: str, clip_model_name: str):
        super().__init__(model_name=f"{dino_model_name}|{clip_model_name}")
        self.dino_model_name = dino_model_name
        self.clip_model_name = clip_model_name

        # DINO
        self.dino_processor = AutoImageProcessor.from_pretrained(dino_model_name)
        self.dino_model = AutoModel.from_pretrained(dino_model_name).to(self.device).eval()
        self.num_register = int(getattr(self.dino_model.config, "num_register_tokens", 0))

        # CLIP
        self.clip_processor = CLIPImageProcessor.from_pretrained(clip_model_name)
        self.clip_model = CLIPVisionModel.from_pretrained(clip_model_name, use_safetensors=True).to(self.device).eval()

        # 外から注入される前提（get_feature_extractor で設定）
        self.feat_mode = "last"
        self.feat_layer = -1
        self.feat_last_k = 4
        self.shallow_layer = 3  # hidden_states index（外から上書き可）
        self.clip_feat_mode = "layer"
        self.clip_feat_layer = -1
        self.clip_feat_last_k = 4
        self.use_dino_deep = True
        self.use_dino_shallow = True
        self.use_clip = True

    def extract(self, img: Image.Image, vit_patch: int, long_side: int, pad_to_multiple: bool) -> GridFeatures:
        # ---------- DINO forward（1回） ----------
        img_x_dino, shown_wh, _ = self._prep_image(img, vit_patch, long_side, pad_to_multiple)
        din_in = self.dino_processor(images=img_x_dino, return_tensors="pt",
                                     do_resize=False, do_center_crop=False).to(self.device)
        with torch.inference_mode():
            din_out = self.dino_model(**din_in, output_hidden_states=True)

        num_prefix = 1 + self.num_register  # CLS + register

        deep_tokens = _select_tokens_from_hidden_states(
            din_out.hidden_states,
            num_prefix_tokens=num_prefix,
            feat_mode=self.feat_mode,
            feat_layer=self.feat_layer,
            feat_last_k=self.feat_last_k,
        )  # (1, N, Dd)

        shallow_tokens = _select_tokens_from_hidden_states(
            din_out.hidden_states,
            num_prefix_tokens=num_prefix,
            feat_mode="layer",
            feat_layer=self.shallow_layer,
            feat_last_k=1,
        )  # (1, N, Ds)

        Hin, Win = din_in["pixel_values"].shape[-2:]
        gh, gw = Hin // vit_patch, Win // vit_patch
        Nref = gh * gw

        deep = deep_tokens[0, :Nref, :]
        shallow = shallow_tokens[0, :Nref, :]
        deep = torch.nn.functional.normalize(deep, dim=-1)
        shallow = torch.nn.functional.normalize(shallow, dim=-1)

        # ---------- CLIP forward ----------
        patch_size = int(getattr(self.clip_model.config, "patch_size", 14))
        img_x_clip, shown_wh_clip, _ = self._prep_image(img, patch_size, long_side, pad_to_multiple)
        clip_in = self.clip_processor(images=img_x_clip, return_tensors="pt",
                                      do_resize=False, do_center_crop=False).to(self.device)
        with torch.inference_mode():
            clip_out = self.clip_model(**clip_in, interpolate_pos_encoding=True, output_hidden_states=True)

        clip_tokens = _select_tokens_from_hidden_states(
            clip_out.hidden_states,
            num_prefix_tokens=1,              # CLIPはCLSのみ除外
            feat_mode=self.clip_feat_mode,
            feat_layer=self.clip_feat_layer,
            feat_last_k=self.clip_feat_last_k,
        )[0]  # (Nc, Dc)

        Hin2, Win2 = clip_in["pixel_values"].shape[-2:]
        gh2, gw2 = Hin2 // patch_size, Win2 // patch_size
        N2 = gh2 * gw2
        clip = clip_tokens[:N2, :]
        clip = torch.nn.functional.normalize(clip, dim=-1)

        # ---------- CLIP を (gh,gw) に補間 ----------
        Dc = clip.shape[1]
        clip_map = clip.reshape(gh2, gw2, Dc).permute(2, 0, 1).unsqueeze(0)  # (1, Dc, gh2, gw2)
        clip_res = F.interpolate(clip_map, size=(gh, gw), mode="bilinear", align_corners=False)
        clip_res = clip_res.squeeze(0).permute(1, 2, 0).reshape(Nref, Dc)    # (Nref, Dc)
        clip_res = torch.nn.functional.normalize(clip_res, dim=-1)

        # ---------- concat ----------
        if (not self.use_dino_deep) and (not self.use_dino_shallow):
            raise RuntimeError("Ours: DINO deep と shallow が両方OFFです（マスク量子化の前提が崩れます）。")

        parts = []
        dims = []

        if self.use_dino_deep:
            parts.append(deep)
            dims.append(("dino_deep", deep.shape[1]))

        if self.use_dino_shallow:
            parts.append(shallow)
            dims.append(("dino_shallow", shallow.shape[1]))

        if self.use_clip:
            parts.append(clip_res)
            dims.append(("clip", clip_res.shape[1]))

        fused = torch.cat(parts, dim=-1)
        fused = torch.nn.functional.normalize(fused, dim=-1).detach().cpu()

        # 任意：デバッグログ
        # print("[INFO] Ours parts:", dims, "-> fused_dim", fused.shape[1])

        return GridFeatures(patch_feat=fused, gh=gh, gw=gw, shown_size=shown_wh)


def get_feature_extractor(encoder: str, model_name: str, feat_mode="last", feat_layer=-1, feat_last_k=4):
    enc = encoder.lower()
    if enc == "dino":
        ex = DINOv3Extractor(model_name)
    elif enc == "clip":
        ex = CLIPVisionExtractor(model_name)
    elif enc == "ours":
        # model_name は "DINO|CLIP"
        if "|" not in model_name:
            raise ValueError("encoder='ours' の model_name は 'DINO|CLIP' 形式にしてください。")
        dino_name, clip_name = model_name.split("|", 1)
        ex = OursFusionExtractor(dino_name.strip(), clip_name.strip())
    else:
        raise ValueError(f"Unknown encoder: {encoder}")

    # 抽出層設定を注入（ours の deep 側にも効く）
    ex.feat_mode   = feat_mode
    ex.feat_layer  = feat_layer
    ex.feat_last_k = feat_last_k
    return ex


