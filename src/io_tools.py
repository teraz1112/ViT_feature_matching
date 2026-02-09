# ISVS/src/io_tools.py
from pathlib import Path
from PIL import Image
import os
from pathlib import Path
import csv

def unique_path(path: Path) -> Path:
    path = Path(path)
    if not path.exists():
        return path
    parent = path.parent
    stem = path.stem
    suffix = path.suffix
    i = 1
    while True:
        cand = parent / f"{stem}_{i}{suffix}"
        if not cand.exists():
            return cand
        i += 1

def ensure_outdir(out_dir: Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

def load_image(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    img = Image.open(path).convert("RGB")
    return img

def save_image(img: Image.Image, path: Path, overwrite: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if (not overwrite) and path.exists():
        path = unique_path(path)
    img.save(path)
    return path

def save_rows_as_csv(rows, path, header=None, sort_by=None, descending=False, overwrite: bool = False) -> Path:
    """
    rows: List[Dict]（collect_flow_on_observed_recordsの戻り値）
    path: str | Path
    header: Optional[List[str]]
    sort_by: Optional[str]
    descending: bool
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if (not overwrite) and path.exists():
        path = unique_path(path)

    header = header or [
        "qy","qx",
        "src_x_tgt","src_y_tgt",
        "src_x_obs_mapped","src_y_obs_mapped",
        "dst_x_obs","dst_y_obs",
        "dx","dy","score",
        "obs_w","obs_h","tgt_w","tgt_h",
    ]

    rows = list(rows or [])
    if sort_by is not None:
        def _sort_key(r):
            v = r.get(sort_by, None)
            return float(v) if v is not None else float("-inf")
        rows.sort(key=_sort_key, reverse=bool(descending))

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, None) for k in header})
    return path

def save_metrics_as_csv(metrics: dict, path, header=None, overwrite: bool = False) -> Path:
    """
    metrics: Dict[str, Any]
    path: str | Path
    header: Optional[List[str]]
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if (not overwrite) and path.exists():
        path = unique_path(path)

    header = header or list(metrics.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerow({k: metrics.get(k, None) for k in header})
    return path
