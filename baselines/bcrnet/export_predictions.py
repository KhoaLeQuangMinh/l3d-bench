"""Run a BCRNet checkpoint on a list of frames: PNG label maps for our scorer, plus BCRNet's own metric.

Runs inside the BCRNet repo (cwd) with its BezierDataset (squashed 1024 x 1024 image, AdelaiDepth map, SAM
features, Bezier targets) and TransformerPureDetector. BCRNet predicts, per landmark, point sequences
('ctrl_points', already scaled to the frame's native pixel size); its own metric (test.py metrix) draws them
as 30 px polylines on a 1080 x 1920 canvas. For our scorer we draw the same polylines on the native canvas
with the protocol's thickness int(35 * H / 1080), then resize to 1024 x 1024 with nearest neighbour.
Classes: BCRNet's landmark order is silhouette, ligament, ridge -> our ids 2, 3, 1, drawn in that order
(ridge last wins overlaps; overlaps between classes are rare and tiny).

--mode eval: model.eval() (our benchmark, patch 01 does the same in test.py).
--mode train: the released test.py's behaviour (no model.eval(): BatchNorm uses batch statistics); only for
the 'native' reproduction number.

    python export_predictions.py --model_path epoch-60.pt --list frames.txt --precomputed <dir> \
        --out preds/ --native_json native.json --mode eval
"""
import argparse
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.getcwd())
from adet.modeling.bezier_detection import TransformerPureDetector  # noqa: E402
from utils.bezier_dataset import BezierDataset, collate_fun  # noqa: E402
from utils.config_utils import load_config  # noqa: E402

OUR_CLASS = (2, 3, 1)          # BCRNet landmark index 0 silhouette, 1 ligament, 2 ridge


def render(results_img, height: int, width: int, size: int = 1024, width_at_1080: int = 35) -> np.ndarray:
    canvas = np.zeros((height, width), dtype=np.uint8)
    thickness = int(width_at_1080 * height / 1080)
    for m, lm in enumerate(results_img):
        for curve in lm["ctrl_points"].cpu().numpy():
            for i in range(1, len(curve)):
                p1 = tuple(map(int, curve[i - 1]))
                p2 = tuple(map(int, curve[i]))
                cv2.line(canvas, p1, p2, OUR_CLASS[m], thickness)
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_NEAREST)


def build_root(images, precomputed: Path, root: Path) -> None:
    """BezierDataset reads <root>/{images,depth_AdelaiDepth,sam,s_bezier}/<id>.*"""
    for sub in ("images", "depth_AdelaiDepth", "sam", "s_bezier"):
        (root / sub).mkdir(parents=True)
    for img in map(Path, images):
        fid = img.stem
        os.symlink(img, root / "images" / f"{fid}.jpg")
        os.symlink(img.parent.parent / "depth_AdelaiDepth" / f"{fid}.png", root / "depth_AdelaiDepth" / f"{fid}.png")
        os.symlink(precomputed / "sam" / f"{fid}.npy", root / "sam" / f"{fid}.npy")
        os.symlink(precomputed / "s_bezier" / f"{fid}.npz", root / "s_bezier" / f"{fid}.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--list", required=True)
    ap.add_argument("--precomputed", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--native_json", required=True)
    ap.add_argument("--mode", choices=["eval", "train"], default="eval")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    images = [line.strip() for line in open(a.list) if line.strip()]
    os.makedirs(a.out, exist_ok=True)
    spec = importlib.util.spec_from_file_location("bcrnet_test", os.path.join(os.getcwd(), "test.py"))
    bcr_test = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bcr_test)

    cfg = load_config(a.config)
    cfg.MODEL.DEVICE = a.device
    model = TransformerPureDetector(cfg).to(a.device)
    model.load_state_dict(torch.load(a.model_path, map_location=a.device, weights_only=False)["model"])
    if a.mode == "eval":
        model.eval()
    else:
        model.train()
    rows = []
    with tempfile.TemporaryDirectory() as tmp, torch.no_grad():
        build_root(images, Path(a.precomputed), Path(tmp) / "root")
        ds = BezierDataset(str(Path(tmp) / "root"), device=a.device)
        for batch in DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fun):
            _, _, _, targets, info = batch
            results = model(batch)
            h, w = (int(v) for v in info[0]["img_size"])
            fid = info[0]["item_name"]
            cv2.imwrite(os.path.join(a.out, fid + ".png"), render(results[0], h, w))
            iou, dice = bcr_test.metrix(results, targets, info, save_path=None)
            rows.append({"frame_id": fid, "dice": float(dice), "iou": float(iou),
                         "n_curves": [int(lm["ctrl_points"].shape[0]) for lm in results[0]]})
    if len(rows) != len(images):
        raise RuntimeError(f"exported {len(rows)} of {len(images)} frames")
    summary = {k: float(np.mean([r[k] for r in rows])) for k in ("dice", "iou")}
    json.dump({"metric": f"BCRNet test.py metrix (its own 30 px GT), model.{a.mode}()", "mean": summary,
               "frames": rows}, open(a.native_json, "w"), indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
