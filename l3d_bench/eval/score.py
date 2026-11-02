"""Score a folder of predicted label maps against ground truth rendered from the official polylines.

Prediction files: <pred_dir>/<frame_id>.png, single-channel uint8 label map (0..3), any size
(resized to 1024 with nearest-neighbour if needed). Every method - ours and every baseline - is scored
by this one function.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ..data.annotations import load_annotation
from ..data.masks import gt_for_scoring
from .metrics import score_frame


def load_prediction(path: str | Path, size: int = 1024) -> np.ndarray:
    pred = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if pred is None:
        raise FileNotFoundError(path)
    if pred.ndim == 3:
        raise ValueError(f"{path}: expected a single-channel label map, got {pred.shape}")
    if pred.shape != (size, size):
        pred = cv2.resize(pred, (size, size), interpolation=cv2.INTER_NEAREST)
    if pred.max() > 3:
        raise ValueError(f"{path}: label values must be 0..3, found max {pred.max()}")
    return pred


def score_folder(pred_dir: str | Path, index: pd.DataFrame, frame_ids, poses: dict | None = None,
                 cfg_metric: dict | None = None) -> pd.DataFrame:
    m = cfg_metric or {"line_width_at_1080": 35, "score_size": 1024}
    rows = index.set_index("frame_id")
    out = []
    for fid in frame_ids:
        r = rows.loc[fid]
        gt = gt_for_scoring(load_annotation(r["label_path"]), m["line_width_at_1080"], m["score_size"])
        pred = load_prediction(Path(pred_dir) / f"{fid}.png", m["score_size"])
        row = {"frame_id": fid, "patient": r["patient"], "pose": (poses or {}).get(fid, "")}
        row.update(score_frame(pred, gt))
        out.append(row)
    return pd.DataFrame(out)
