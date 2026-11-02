"""Generate data/splits/folds_5.json (needs pose labels for the 35 pool patients) and official.json.

    python scripts/make_folds.py --seed 0
    python scripts/make_folds.py --no-pose      # frame-count balance only (for development, not the study)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from l3d_bench.config import REPO_ROOT, load_config
from l3d_bench.data.folds import make_folds, official_split, save_folds
from l3d_bench.data.index import load_index
from l3d_bench.data.pose import load_pose_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n_inner", type=int, default=3)
    ap.add_argument("--no-pose", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    index = load_index(REPO_ROOT / cfg["data"]["index_file"])
    poses = None if args.no_pose else load_pose_labels(REPO_ROOT / cfg["data"]["pose_file"])
    spec = make_folds(index, poses, k=args.k, n_inner=args.n_inner, seed=args.seed)
    out = REPO_ROOT / cfg["data"]["folds_file"]
    out.parent.mkdir(parents=True, exist_ok=True)
    digest = save_folds(spec, out)
    (out.parent / "official.json").write_text(json.dumps(official_split(index), indent=2))
    for f in spec["folds"]:
        print(f"fold {f['fold']}: heldout {f['frames']['heldout']} frames ({f['raised']['heldout']} raised), "
              f"inner-val {f['inner_val']} ({f['frames']['inner_val']} frames)")
    print(f"{out}  sha256={digest}  pose_stratified={spec['pose_stratified']}")


if __name__ == "__main__":
    main()
