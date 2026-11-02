"""PyTorch dataset for our models (M1-M3). Baselines use their own loaders, unchanged.

Each item: image float (3, S, S) in [0, 1], label long (S, S) with 0..3, junctions (4, 2) in [0, 1],
junction_vis (4,), pose (0 canonical, 1 raised, -1 unknown), frame_id.
Augmentation (train only): horizontal flip, scale + small shift, colour jitter. No vertical flip.
Image, label and junctions always receive the same geometric transform.
"""
from __future__ import annotations

import random

import cv2
import numpy as np
import pandas as pd

from .annotations import load_annotation
from .junctions import derive_junctions, hflip_junctions
from .masks import render_label_map

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # data utilities stay usable without torch
    torch, Dataset = None, object

POSE_ID = {"canonical": 0, "raised": 1}


def _affine(size: int, scale: float, tx: float, ty: float) -> np.ndarray:
    c = size / 2.0
    return np.array([[scale, 0, (1 - scale) * c + tx], [0, scale, (1 - scale) * c + ty]], dtype=np.float32)


def _color_jitter(img: np.ndarray, amount: float, rng: random.Random) -> np.ndarray:
    out = img.astype(np.float32)
    b = 1 + rng.uniform(-amount, amount)
    c = 1 + rng.uniform(-amount, amount)
    s = 1 + rng.uniform(-amount, amount)
    out = out * b
    mean = out.mean()
    out = (out - mean) * c + mean
    gray = out.mean(axis=2, keepdims=True)
    out = (out - gray) * s + gray
    return np.clip(out, 0, 255).astype(np.uint8)


class L3DDataset(Dataset):
    def __init__(self, index: pd.DataFrame, frame_ids, cfg: dict, train: bool = False,
                 poses: dict[str, str] | None = None, seed: int = 0):
        rows = index.set_index("frame_id")
        self.rows = [rows.loc[f] for f in frame_ids]
        self.frame_ids = list(frame_ids)
        self.size = int(cfg["data"]["input_size"])
        self.width = int(cfg["metric"]["line_width_at_1080"])
        self.jthr = float(cfg["junctions"]["connect_threshold_at_1080"])
        self.aug = cfg["augment"]
        if self.aug.get("vertical_flip"):
            raise ValueError("vertical flips are not allowed (they would fake raised-liver views)")
        self.train = train
        self.poses = poses or {}
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def pose_list(self) -> list[str]:
        return [self.poses.get(f, "unknown") for f in self.frame_ids]

    def load(self, i: int):
        row, fid = self.rows[i], self.frame_ids[i]
        ann = load_annotation(row["label_path"])
        img = cv2.cvtColor(cv2.imread(row["image_path"]), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(render_label_map(ann, self.width), (self.size, self.size),
                           interpolation=cv2.INTER_NEAREST)
        coords, vis = derive_junctions(ann, self.jthr)
        return fid, img, label, coords, vis

    def augment(self, img, label, coords, vis):
        a, rng, s = self.aug, self.rng, self.size
        if rng.random() < a.get("hflip_p", 0.0):
            img, label = img[:, ::-1].copy(), label[:, ::-1].copy()
            coords, vis = hflip_junctions(coords, vis)
        lo, hi = a.get("scale_range", [1.0, 1.0])
        scale = rng.uniform(lo, hi)
        t = a.get("translate_frac", 0.0) * s
        m = _affine(s, scale, rng.uniform(-t, t), rng.uniform(-t, t))
        img = cv2.warpAffine(img, m, (s, s), flags=cv2.INTER_LINEAR, borderValue=0)
        label = cv2.warpAffine(label, m, (s, s), flags=cv2.INTER_NEAREST, borderValue=0)
        px = coords * s
        px = px @ m[:, :2].T + m[:, 2]
        coords = px / s
        inside = ((coords >= 0) & (coords <= 1)).all(axis=1)
        vis = vis * inside.astype(np.float32)
        coords = np.where(vis[:, None] > 0, coords, 0.0).astype(np.float32)
        if a.get("color_jitter", 0) > 0:
            img = _color_jitter(img, a["color_jitter"], rng)
        return img, label, coords, vis

    def __getitem__(self, i: int):
        fid, img, label, coords, vis = self.load(i)
        if self.train:
            img, label, coords, vis = self.augment(img, label, coords, vis)
        item = {
            "image": img.transpose(2, 0, 1).astype(np.float32) / 255.0,
            "label": label.astype(np.int64),
            "junctions": coords.astype(np.float32),
            "junction_vis": vis.astype(np.float32),
            "pose": POSE_ID.get(self.poses.get(fid, ""), -1),
            "frame_id": fid,
        }
        if torch is not None:
            for k in ("image", "label", "junctions", "junction_vis"):
                item[k] = torch.from_numpy(np.ascontiguousarray(item[k]))
        return item
