"""BCRNet input: SAM ViT-B image-encoder features (256 x 64 x 64) per frame -> <out>/<frame_id>.npy.

BCRNet's paper says a frozen SAM-ViT-B encoder provides the features; its code reads them from
<split>/sam/<frame_id>.npy, and the commented block in backbone/Unet.py shows the call
(sam.image_encoder(imgs) -> B x 256 x 64 x 64). The preprocessing is not stated in the paper, so we use
SAM's own (ResizeLongestSide to 1024, pixel mean/std normalisation, pad to 1024 x 1024). Documented in
docs/baselines.md as an assumption.

    pip install git+https://github.com/facebookresearch/segment-anything.git
    python scripts/precompute_sam_features.py --checkpoint sam_vit_b_01ec64.pth --out /kaggle/working/sam
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from l3d_bench.config import REPO_ROOT, load_config
from l3d_bench.data.index import load_index


def main():
    import torch
    from segment_anything import sam_model_registry
    from segment_anything.utils.transforms import ResizeLongestSide

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = load_config()
    index = load_index(REPO_ROOT / cfg["data"]["index_file"])
    sam = sam_model_registry["vit_b"](checkpoint=args.checkpoint).to(args.device).eval()
    resize = ResizeLongestSide(sam.image_encoder.img_size)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for i, r in enumerate(index.itertuples()):
            dst = out / f"{r.frame_id}.npy"
            if dst.exists():
                continue
            img = cv2.cvtColor(cv2.imread(r.image_path), cv2.COLOR_BGR2RGB)
            x = torch.as_tensor(resize.apply_image(img), device=args.device).permute(2, 0, 1)[None].float()
            feat = sam.image_encoder(sam.preprocess(x))[0]          # 256 x 64 x 64
            np.save(dst, feat.cpu().numpy().astype(np.float32))
            if i % 100 == 0:
                print(f"{i}/{len(index)}")


if __name__ == "__main__":
    main()
