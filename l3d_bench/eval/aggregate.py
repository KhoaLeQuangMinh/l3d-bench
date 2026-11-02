"""Summaries of per-frame scores: overall mean (as the papers report), per patient, per pose, and
patient-level bootstrap confidence intervals (resampling patients, not frames)."""
from __future__ import annotations

import numpy as np
import pandas as pd

METRICS = ("dice", "iou", "assd", "dice_ridge", "dice_silhouette", "dice_ligament")


def summarize(frames: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> dict:
    """frames: one row per scored frame with columns frame_id, patient, pose (optional) + metric columns."""
    out = {"n_frames": len(frames), "n_patients": int(frames["patient"].nunique())}
    cols = [m for m in METRICS if m in frames.columns]
    out["mean"] = {m: float(np.nanmean(frames[m])) for m in cols}
    out["per_patient"] = frames.groupby("patient")[cols].mean().round(4).to_dict("index")
    if "pose" in frames.columns and frames["pose"].isin(["canonical", "raised"]).any():
        out["per_pose"] = {p: {m: float(np.nanmean(g[m])) for m in cols} | {"n_frames": len(g)}
                           for p, g in frames.groupby("pose") if p in ("canonical", "raised")}
    if out["n_patients"] >= 5 and n_boot > 0:
        out["ci95"] = patient_bootstrap(frames, cols, n_boot, seed)
    return out


def patient_bootstrap(frames: pd.DataFrame, cols, n_boot: int = 1000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    groups = {p: g[list(cols)].to_numpy(dtype=float) for p, g in frames.groupby("patient")}
    names = list(groups)
    stats = []
    for _ in range(n_boot):
        pick = rng.choice(len(names), size=len(names), replace=True)
        stack = np.concatenate([groups[names[i]] for i in pick], axis=0)
        stats.append(np.nanmean(stack, axis=0))
    stats = np.asarray(stats)
    lo, hi = np.percentile(stats, [2.5, 97.5], axis=0)
    return {c: [float(lo[i]), float(hi[i])] for i, c in enumerate(cols)}


def summarize_folds(fold_means: list[dict]) -> dict:
    """Mean and std across folds of each fold's overall mean."""
    keys = fold_means[0].keys()
    return {k: {"mean": float(np.mean([f[k] for f in fold_means])),
                "std": float(np.std([f[k] for f in fold_means], ddof=1))} for k in keys}
