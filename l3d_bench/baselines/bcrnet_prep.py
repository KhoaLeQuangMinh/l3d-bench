"""BCRNet's precomputed inputs, per frame: SAM ViT-B image-encoder features and Bezier targets.

    python -m l3d_bench.baselines.bcrnet_prep sam     --frames frames.tsv --checkpoint sam_vit_b_01ec64.pth --out <dir>/sam
    python -m l3d_bench.baselines.bcrnet_prep beziers --frames frames.tsv --repo <BCRNet clone> --out <dir>/s_bezier

frames.tsv: one line per frame, "<frame_id>\t<image_path>\t<label_path>".

SAM features (<frame_id>.npy, float32 256 x 64 x 64). BCRNet reads them from <split>/sam/; the code that made
them is not released. The commented block in backbone/Unet.py feeds the model's own image batch to
sam.image_encoder, and the model's image is the frame squashed to 1024 x 1024 (cv2.resize, no aspect ratio
kept). We do the same, so each 64 x 64 SAM cell lines up with the CNN features it is added to, and use SAM's
own normalisation (pixel mean/std on 0-255 RGB). An assumption, documented in baselines/bcrnet/README.md.

Bezier targets (<frame_id>.npz): BCRNet's own utils/preprocess.py generate_beziers with its defaults
(degree 5, 25 curve points, 1080 x 1920 mask canvas, 30 px lines), run on the L3D LabelMe JSON.
Frames that already have an output file are skipped, so a resumed session only pays for missing ones.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path


def read_frames(path: str | Path) -> list[tuple[str, str, str]]:
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            fid, img, lab = line.split("\t")
            rows.append((fid, img, lab))
    return rows


def write_frames(path: str | Path, index, frame_ids) -> Path:
    rows = index.set_index("frame_id")
    Path(path).write_text("".join(f"{f}\t{rows.loc[f, 'image_path']}\t{rows.loc[f, 'label_path']}\n"
                                  for f in frame_ids))
    return Path(path)


def sam_input(img_bgr, size: int = 1024):
    """Frame -> the RGB 1024 x 1024 uint8 array BCRNet's model sees (squashed, as utils/bezier_dataset.py)."""
    import cv2
    return cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), (size, size))


def sam_features(frames, checkpoint: str, out: Path, device: str = "cuda", log=print) -> int:
    import cv2
    import numpy as np
    import torch
    from segment_anything import sam_model_registry

    out.mkdir(parents=True, exist_ok=True)
    todo = [(f, img) for f, img, _ in frames if not (out / f"{f}.npy").exists()]
    if not todo:
        return 0
    sam = sam_model_registry["vit_b"](checkpoint=checkpoint).to(device).eval()
    with torch.no_grad():
        for i, (fid, img) in enumerate(todo):
            x = torch.as_tensor(sam_input(cv2.imread(img)), device=device).permute(2, 0, 1)[None].float()
            feat = sam.image_encoder(sam.preprocess(x))[0]                  # 256 x 64 x 64
            tmp = out / f"{fid}.tmp.npy"
            np.save(tmp, feat.float().cpu().numpy())
            os.replace(tmp, out / f"{fid}.npy")
            if i % 200 == 0:
                log(f"SAM features {i}/{len(todo)}")
    return len(todo)


def bezier_targets(frames, repo: Path, out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    todo = [(f, lab) for f, _, lab in frames if not (out / f"{f}.npz").exists()]
    if not todo:
        return 0
    sys.path.insert(0, str(Path(repo) / "utils"))                   # preprocess.py does "from bezier import ..."
    from preprocess import generate_beziers
    with tempfile.TemporaryDirectory() as src:
        for fid, lab in todo:                                       # its glob reads every *.json in one folder
            os.symlink(lab, Path(src) / f"{fid}.json")
        generate_beziers(src, str(out))
    missing = [f for f, _ in todo if not (out / f"{f}.npz").exists()]
    if missing:
        raise RuntimeError(f"generate_beziers wrote no target for {missing[:5]}")
    return len(todo)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["sam", "beziers"])
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint")
    ap.add_argument("--repo")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    frames = read_frames(a.frames)
    if a.what == "sam":
        n = sam_features(frames, a.checkpoint, Path(a.out), a.device)
    else:
        n = bezier_targets(frames, Path(a.repo), Path(a.out))
    print(f"[bcrnet_prep] {a.what}: {n} new of {len(frames)} frames -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
