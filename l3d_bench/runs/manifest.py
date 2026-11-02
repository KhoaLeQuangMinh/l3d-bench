"""The run manifest: one row per training run; run ids follow your UNETR++ notebook's style.

run_id = <method>__<variant>__<split>__s<seed>, e.g. m2__junction__f3__s0, toponet__official__official__s0
split  = 'official' or 'f0'..'f4'
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

METHODS = {  # method -> variant tag
    "m1": "plain", "m2": "junction", "m3": "posebal",
    "toponet": "official", "a2onet": "official", "d2gpland": "official", "bcrnet": "official",
}
SPLITS = ["official", "f0", "f1", "f2", "f3", "f4"]
RUN_RE = re.compile(r"^(?P<method>[a-z0-9]+)__(?P<variant>[a-z0-9]+)__(?P<split>official|f[0-4])__s(?P<seed>\d+)$")


def run_id(method: str, split: str, seed: int) -> str:
    return f"{method}__{METHODS[method]}__{split}__s{seed}"


def parse_run_id(rid: str) -> dict:
    m = RUN_RE.match(rid)
    if not m:
        raise ValueError(f"bad run id {rid!r}")
    d = m.groupdict()
    d["seed"] = int(d["seed"])
    if d["method"] not in METHODS or METHODS[d["method"]] != d["variant"]:
        raise ValueError(f"unknown method/variant in {rid!r}")
    return d


def make_manifest(seeds=(0,), owners: list[str] | None = None, methods=None) -> pd.DataFrame:
    """Owners are assigned by split, so a slow account delays one split, not one method."""
    methods = list(methods or METHODS)
    rows = []
    for seed in seeds:
        for si, split in enumerate(SPLITS):
            owner = owners[si % len(owners)] if owners else ""
            for m in methods:
                rows.append({"run_id": run_id(m, split, seed), "method": m, "split": split, "seed": seed,
                             "owner": owner, "status": "todo"})
    return pd.DataFrame(rows)


def load_manifest(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False)
    for rid in df["run_id"]:
        parse_run_id(rid)
    if df["run_id"].duplicated().any():
        raise ValueError("duplicate run ids in manifest")
    return df
