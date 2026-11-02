"""Pose labels: one per frame, 'canonical' or 'raised', made by hand (see docs/pose_definition.md)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

POSES = ("canonical", "raised")


def load_pose_labels(path: str | Path) -> dict[str, str]:
    """CSV with columns frame_id, pose, labeler -> {frame_id: pose}. Rejects unknown values and duplicates."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in ("frame_id", "pose"):
        if col not in df.columns:
            raise ValueError(f"{path}: missing column {col!r}")
    bad = df.loc[~df["pose"].isin(POSES), "pose"].unique()
    if len(bad):
        raise ValueError(f"{path}: unknown pose values {list(bad)}")
    dup = df["frame_id"][df["frame_id"].duplicated()]
    if len(dup):
        raise ValueError(f"{path}: frames labeled twice (merge the overlap set first): {list(dup)[:5]}")
    return dict(zip(df["frame_id"], df["pose"]))


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Agreement between two labelers on the same frames, corrected for chance."""
    if len(a) != len(b) or not a:
        raise ValueError("need two equally long, non-empty label lists")
    cats = sorted(set(a) | set(b))
    n = len(a)
    p_obs = sum(x == y for x, y in zip(a, b)) / n
    p_exp = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    if p_exp == 1.0:
        return 1.0
    return (p_obs - p_exp) / (1.0 - p_exp)


def raised_share(frame_ids, poses: dict[str, str]) -> float:
    ids = list(frame_ids)
    if not ids:
        return 0.0
    return sum(poses[f] == "raised" for f in ids) / len(ids)
