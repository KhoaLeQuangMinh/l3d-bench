"""Patient-disjoint, pose-stratified 5-fold split of the 35 pool patients, plus an inner-val per fold.

Each fold i has three disjoint patient sets:
    heldout   (~7 patients)  scored for fold i
    inner_val (3 patients)   used only to pick the best epoch
    train     (the rest)     used for training
Test patients of the official split never appear. A patient never spans two sets of the same fold.

Assignment is greedy: patients sorted by frame count (largest first) go to the fold that most reduces
imbalance in frame count and raised-pose share. Inner-val patients are chosen exhaustively among each
fold's training patients to match the pool's raised share and a target frame count.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import random
from pathlib import Path

import pandas as pd


def _patient_table(index: pd.DataFrame, poses: dict[str, str] | None) -> pd.DataFrame:
    pool = index[index["split"].isin(["train", "val"])]
    g = pool.groupby("patient")["frame_id"].apply(list).reset_index(name="frames")
    g["n"] = g["frames"].apply(len)
    if poses is None:
        g["raised"] = 0
    else:
        missing = [f for fl in g["frames"] for f in fl if f not in poses]
        if missing:
            raise ValueError(f"{len(missing)} pool frames have no pose label, e.g. {missing[:3]}")
        g["raised"] = g["frames"].apply(lambda fl: sum(poses[f] == "raised" for f in fl))
    return g


def _imbalance(sizes, raised, target_n, target_r):
    cost = 0.0
    for n, r in zip(sizes, raised):
        cost += ((n - target_n) / target_n) ** 2
        share = r / n if n else 0.0
        cost += 4.0 * (share - target_r) ** 2
    return cost


def make_folds(index: pd.DataFrame, poses: dict[str, str] | None = None, k: int = 5,
               n_inner: int = 3, seed: int = 0) -> dict:
    rng = random.Random(seed)
    tab = _patient_table(index, poses)
    total_n, total_r = int(tab["n"].sum()), int(tab["raised"].sum())
    target_n, target_r = total_n / k, (total_r / total_n if total_n else 0.0)

    rows = tab.sample(frac=1.0, random_state=seed).sort_values("n", ascending=False, kind="stable")
    members = [[] for _ in range(k)]
    sizes, raised = [0] * k, [0] * k
    for _, row in rows.iterrows():
        best, best_cost = None, None
        order = list(range(k))
        rng.shuffle(order)
        for f in order:
            s, r = sizes.copy(), raised.copy()
            s[f] += row["n"]
            r[f] += row["raised"]
            c = _imbalance(s, r, target_n, target_r) + 0.01 * len(members[f])
            if best_cost is None or c < best_cost:
                best, best_cost = f, c
        members[best].append(row["patient"])
        sizes[best] += row["n"]
        raised[best] += row["raised"]

    lookup = tab.set_index("patient")
    folds = []
    for i in range(k):
        held = sorted(members[i])
        rest = sorted(p for j in range(k) if j != i for p in members[j])
        inner_target = target_n * n_inner / max(1, round(len(tab) / k))  # ~ frames of n_inner typical patients
        best_combo, best_cost = None, None
        for combo in itertools.combinations(rest, n_inner):
            n = int(lookup.loc[list(combo), "n"].sum())
            r = int(lookup.loc[list(combo), "raised"].sum())
            share = r / n if n else 0.0
            c = ((n - inner_target) / inner_target) ** 2 + 4.0 * (share - target_r) ** 2
            if best_cost is None or c < best_cost - 1e-12:
                best_combo, best_cost = combo, c
        inner = sorted(best_combo)
        train = sorted(p for p in rest if p not in inner)
        folds.append({"fold": i, "heldout": held, "inner_val": inner, "train": train,
                      "frames": {"heldout": int(lookup.loc[held, "n"].sum()),
                                 "inner_val": int(lookup.loc[inner, "n"].sum()),
                                 "train": int(lookup.loc[train, "n"].sum())},
                      "raised": {"heldout": int(lookup.loc[held, "raised"].sum()),
                                 "inner_val": int(lookup.loc[inner, "raised"].sum()),
                                 "train": int(lookup.loc[train, "raised"].sum())}})
    spec = {"k": k, "n_inner": n_inner, "seed": seed, "pose_stratified": poses is not None,
            "pool_patients": len(tab), "pool_frames": total_n, "pool_raised": total_r, "folds": folds}
    check_folds(spec, index)
    return spec


def official_split(index: pd.DataFrame) -> dict:
    """The published split in the same format: train / val (selection) / test (scored)."""
    pats = {s: sorted(index.loc[index["split"] == s, "patient"].unique()) for s in ("train", "val", "test")}
    return {"fold": "official", "train": pats["train"], "inner_val": pats["val"], "heldout": pats["test"]}


def check_folds(spec: dict, index: pd.DataFrame) -> None:
    test_patients = set(index.loc[index["split"] == "test", "patient"])
    pool = set(index.loc[index["split"].isin(["train", "val"]), "patient"])
    seen_heldout = []
    for f in spec["folds"]:
        sets = [set(f["train"]), set(f["inner_val"]), set(f["heldout"])]
        if any(a & b for a, b in itertools.combinations(sets, 2)):
            raise AssertionError(f"fold {f['fold']}: patient sets overlap")
        if set().union(*sets) != pool:
            raise AssertionError(f"fold {f['fold']}: does not cover the pool exactly")
        if set().union(*sets) & test_patients:
            raise AssertionError(f"fold {f['fold']}: contains official test patients")
        seen_heldout += f["heldout"]
    if sorted(seen_heldout) != sorted(pool):
        raise AssertionError("each pool patient must be held out exactly once")


def save_folds(spec: dict, path: str | Path) -> str:
    text = json.dumps(spec, indent=2, sort_keys=True)
    Path(path).write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def load_folds(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def split_patients(spec: dict, split: str) -> dict:
    """split = 'f0'..'f4' -> {'train','inner_val','heldout'} patient lists."""
    i = int(split.lstrip("f"))
    f = spec["folds"][i]
    return {"train": f["train"], "inner_val": f["inner_val"], "heldout": f["heldout"]}
