# ISVS/src/cropper.py
from typing import Dict, Tuple
from PIL import Image
import math

def _scale_to_cover(img: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    """短辺基準で「はみ出させて」等倍率拡大（cover）。"""
    tw, th = target_size
    w, h = img.size
    scale = max(tw / w, th / h)
    new_w = math.ceil(w * scale)
    new_h = math.ceil(h * scale)
    return img.resize((new_w, new_h), Image.BICUBIC)

def _center_crop(img: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    tw, th = target_size
    w, h = img.size
    if w < tw or h < th:
        raise ValueError(f"Center-crop failed: observed smaller than target (obs={w}x{h}, tgt={tw}x{th})")
    left = (w - tw) // 2
    top  = (h - th) // 2
    right = left + tw
    bottom = top + th
    return img.crop((left, top, right, bottom)), (left, top, right, bottom)

def crop_to_match(target_img: Image.Image,
                  observed_img: Image.Image,
                  mode: str = "scale_then_center_crop"):
    """
    観測画像を目標画像のサイズに合わせる。
    mode:
      - "scale_then_center_crop": 目標サイズ以上へ拡大→中心クロップ（推奨）
      - "center_crop_only": リサイズせずに中心クロップ（観測が小さいと失敗）
    Returns:
      cropped_img (PIL.Image), meta (dict)
    """
    tw, th = target_img.size
    meta: Dict = {
        "target_size": (tw, th),
        "observed_size": observed_img.size,
        "mode": mode,
        "scaled_size": None,
        "crop_box_on_scaled": None
    }

    if mode == "scale_then_center_crop":
        scaled = _scale_to_cover(observed_img, (tw, th))
        meta["scaled_size"] = scaled.size
        cropped, box = _center_crop(scaled, (tw, th))
        meta["crop_box_on_scaled"] = box
        return cropped, meta

    elif mode == "center_crop_only":
        cropped, box = _center_crop(observed_img, (tw, th))
        meta["crop_box_on_scaled"] = box  # スケールなしでも箱情報は返す
        return cropped, meta

    else:
        raise ValueError(f"Unknown mode: {mode}")
