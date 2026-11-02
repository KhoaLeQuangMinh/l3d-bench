"""Write runs/manifest.csv (42 rows per seed: 7 methods x official + 5 folds), owners assigned by split.

    python scripts/make_manifest.py --seeds 0 --owners khoa,an,binh,chi,dung,giang
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from l3d_bench.config import REPO_ROOT
from l3d_bench.runs.manifest import make_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--owners", default="")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    owners = [o for o in args.owners.split(",") if o] or None
    m = make_manifest(seeds=seeds, owners=owners)
    out = REPO_ROOT / "runs" / "manifest.csv"
    m.to_csv(out, index=False)
    print(f"{len(m)} runs -> {out}")


if __name__ == "__main__":
    main()
