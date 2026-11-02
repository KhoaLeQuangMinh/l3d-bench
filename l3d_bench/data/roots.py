"""Build symlinked data roots so each baseline repo sees our splits without code changes.

Two roots per run:
  training root: Train = train patients; Val = Test = selection patients (inner-val, or official val).
                 Whatever a repo selects on (val or, in TopoNet/D2GPLand/A2ONet's case, test) is
                 therefore never the scored set.
  eval root:     Test = held-out patients (fold held-out, or official test); Train/Val empty.

Folder-name style per repo (read from their get_split code):
  upper: Train/ Val/ Test/      D2GPLand, BCRNet
  lower: train/ val/ test/      TopoNet (any other folder name is read as val!), A2ONet
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

STYLES = {"upper": ("Train", "Val", "Test"), "lower": ("train", "val", "test")}
SUBDIRS = {"images": "image_path", "labels": "label_path", "depth_AdelaiDepth": "depth_path"}
EXTRA_SUBDIRS = ("sam", "s_bezier")  # BCRNet's precomputed inputs, if present next to images/


def _link_frames(index: pd.DataFrame, patients, dest: Path) -> int:
    rows = index[index["patient"].isin(set(patients))]
    for sub in list(SUBDIRS) + list(EXTRA_SUBDIRS):
        (dest / sub).mkdir(parents=True, exist_ok=True)
    for _, r in rows.iterrows():
        for sub, col in SUBDIRS.items():
            src = r[col]
            if src:
                _symlink(Path(src), dest / sub / Path(src).name)
        img_root = Path(r["image_path"]).parent.parent
        for sub in EXTRA_SUBDIRS:
            for cand in (img_root / sub).glob(f"{r['frame_id']}.*") if (img_root / sub).exists() else []:
                _symlink(cand, dest / sub / cand.name)
    return len(rows)


def _symlink(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    os.symlink(src.resolve() if src.exists() else src, dst)


def build_roots(index: pd.DataFrame, split_spec: dict, out_dir: str | Path, style: str = "upper") -> dict:
    """split_spec = {'train': [...], 'inner_val': [...], 'heldout': [...]} -> paths of both roots."""
    tr_name, va_name, te_name = STYLES[style]
    out = Path(out_dir)
    train_root, eval_root = out / "train_root", out / "eval_root"
    counts = {
        "train": _link_frames(index, split_spec["train"], train_root / tr_name),
        "select": _link_frames(index, split_spec["inner_val"], train_root / va_name),
        "heldout": _link_frames(index, split_spec["heldout"], eval_root / te_name),
    }
    _link_frames(index, split_spec["inner_val"], train_root / te_name)  # Test = selection set, on purpose
    for name in (tr_name, va_name):
        (eval_root / name / "images").mkdir(parents=True, exist_ok=True)
        (eval_root / name / "labels").mkdir(parents=True, exist_ok=True)
    return {"train_root": str(train_root), "eval_root": str(eval_root), "counts": counts}
