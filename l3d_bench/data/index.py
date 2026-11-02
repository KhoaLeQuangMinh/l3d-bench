"""Build index.csv: one row per L3D frame, the single source every later step reads.

Expected layout per official split root (as on Kaggle):
    <root>/images/Patient_21_0019200.jpg
    <root>/labels/Patient_21_0019200.json
    <root>/depth_AdelaiDepth/Patient_21_0019200.png   (optional; used by D2GPLand and BCRNet)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

FRAME_RE = re.compile(r"^(Patient_\d+)_(\d+)$")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
OPTIONAL_DIRS = {"depth": ("depth_AdelaiDepth", ".png")}
COLUMNS = ["frame_id", "patient", "frame_no", "split", "height", "width", "image_path", "label_path",
           "depth_path"]


def parse_frame_id(stem: str) -> tuple[str, int]:
    m = FRAME_RE.match(stem)
    if not m:
        raise ValueError(f"unexpected frame name {stem!r} (want Patient_<n>_<frame>)")
    return m.group(1), int(m.group(2))


def build_index(roots: dict[str, str | Path]) -> pd.DataFrame:
    """roots: {"train": path, "val": path, "test": path} -> DataFrame sorted by split, patient, frame."""
    rows = []
    for split, root in roots.items():
        root = Path(root)
        images = sorted(p for p in (root / "images").iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not images:
            raise FileNotFoundError(f"no images under {root / 'images'}")
        for img in images:
            stem = img.stem
            patient, frame_no = parse_frame_id(stem)
            label = root / "labels" / f"{stem}.json"
            if not label.exists():
                raise FileNotFoundError(f"missing label for {img}")
            with open(label) as f:
                meta = json.load(f)
            depth_dir, depth_ext = OPTIONAL_DIRS["depth"]
            depth = root / depth_dir / f"{stem}{depth_ext}"
            rows.append({
                "frame_id": stem, "patient": patient, "frame_no": frame_no, "split": split,
                "height": int(meta["imageHeight"]), "width": int(meta["imageWidth"]),
                "image_path": str(img), "label_path": str(label),
                "depth_path": str(depth) if depth.exists() else "",
            })
    df = pd.DataFrame(rows, columns=COLUMNS)
    order = {s: i for i, s in enumerate(["train", "val", "test"])}
    df["_o"] = df["split"].map(order).fillna(9)
    df = df.sort_values(["_o", "patient", "frame_no"]).drop(columns="_o").reset_index(drop=True)
    check_index(df)
    return df


def check_index(df: pd.DataFrame) -> None:
    """Hard invariants: unique frame ids, and no patient in two official splits."""
    dup = df["frame_id"][df["frame_id"].duplicated()]
    if len(dup):
        raise ValueError(f"duplicate frame ids: {list(dup)[:5]}")
    per_patient = df.groupby("patient")["split"].nunique()
    leaked = per_patient[per_patient > 1]
    if len(leaked):
        raise ValueError(f"patients in more than one split: {list(leaked.index)}")


def summary(df: pd.DataFrame) -> dict:
    out = {"frames": df.groupby("split").size().to_dict(),
           "patients": df.groupby("split")["patient"].nunique().to_dict(),
           "n_patients": int(df["patient"].nunique()),
           "non_1080p_frames": df.loc[df["height"] != 1080, ["frame_id", "height", "width"]]
                                 .to_dict("records")}
    top = df.groupby(["split", "patient"]).size().reset_index(name="n")
    out["top_patient_share"] = {
        s: round(float(g["n"].max() / g["n"].sum()), 4) for s, g in top.groupby("split")}
    return out


def load_index(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"frame_id": str}, keep_default_na=False)
    check_index(df)
    return df
