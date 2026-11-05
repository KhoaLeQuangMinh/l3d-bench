"""BCRNet input: SAM ViT-B features for every L3D frame (the BCRNet runner does this by itself).

    pip install git+https://github.com/facebookresearch/segment-anything.git
    python scripts/precompute_sam_features.py --checkpoint sam_vit_b_01ec64.pth --out /kaggle/working/sam

Preprocessing (frame squashed to 1024 x 1024, as BCRNet's model sees it; SAM normalisation): see
l3d_bench/baselines/bcrnet_prep.py.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from l3d_bench.baselines.bcrnet_prep import read_frames, sam_features, write_frames
from l3d_bench.config import load_config
from l3d_bench.data.index import build_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    index = build_index(load_config()["data"]["roots"])
    with tempfile.TemporaryDirectory() as d:
        frames = read_frames(write_frames(Path(d) / "frames.tsv", index, index["frame_id"]))
    print(sam_features(frames, args.checkpoint, Path(args.out), args.device), "new feature files")


if __name__ == "__main__":
    main()
