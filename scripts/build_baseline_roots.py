"""Create the symlinked training and eval roots a baseline repo needs for one split.

    python scripts/build_baseline_roots.py --method toponet --split f2 --out /kaggle/working/roots
    -> /kaggle/working/roots/toponet/f2/train_root  (train / val=inner-val / test=inner-val)
       /kaggle/working/roots/toponet/f2/eval_root   (test = held-out)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from l3d_bench.config import REPO_ROOT, load_config
from l3d_bench.data.folds import load_folds, official_split, split_patients
from l3d_bench.data.index import load_index
from l3d_bench.data.roots import build_roots

STYLE = {"toponet": "lower", "a2onet": "lower", "d2gpland": "upper", "bcrnet": "upper"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=sorted(STYLE))
    ap.add_argument("--split", required=True, help="official or f0..f4")
    ap.add_argument("--out", default="/kaggle/working/roots")
    args = ap.parse_args()
    cfg = load_config()
    index = load_index(REPO_ROOT / cfg["data"]["index_file"])
    spec = official_split(index) if args.split == "official" else \
        split_patients(load_folds(REPO_ROOT / cfg["data"]["folds_file"]), args.split)
    res = build_roots(index, spec, Path(args.out) / args.method / args.split, style=STYLE[args.method])
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
