"""Run an A2ONet checkpoint on a list of frames: PNG label maps for our scorer, plus A2ONet's own metric.

Runs inside the A2ONet repo (cwd) and mirrors its test.py: model built with pretrained_resnet=False, weights
loaded strictly, fusion alpha --alpha (test.py: 1.0; see specs.A2ONet.checkpoints), its own LandmarkDataset (1024 x 1024), argmax of outputs['mask_out'].
A2ONet's metric (a2onet/metrics.py compute_metrics) is the protocol we copied, so both numbers must agree.

    python export_predictions.py --model_path best_val_path.pth --sam sam.pth --depth vitb.pth --list f.txt --out preds/
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.getcwd())
from a2onet import A2ONet  # noqa: E402
from a2onet.metrics import compute_metrics  # noqa: E402
from data.dataset import LandmarkDataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--sam", required=True)
    ap.add_argument("--depth", required=True)
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--native_json", required=True)
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="curve-mask fusion alpha; test.py uses 1.0, train.py evaluates with the alpha of the epoch")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = [line.strip() for line in open(a.list) if line.strip()]
    os.makedirs(a.out, exist_ok=True)
    model = A2ONet(num_classes=4, K=20, sam_checkpoint=a.sam, depth_checkpoint=a.depth,
                   pretrained_dir=os.path.dirname(a.sam), pretrained_resnet=False)
    state = torch.load(a.model_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    model.set_fusion_alpha(a.alpha)
    loader = DataLoader(LandmarkDataset(files, img_size=1024, num_classes=4), batch_size=1, shuffle=False)
    rows = []
    with torch.no_grad():
        for image, mask, name in loader:
            outputs = model(image.to(device))
            pred = torch.argmax(outputs["mask_out"], dim=1)[0].cpu().numpy().astype(np.uint8)
            gt = torch.argmax(mask, dim=1)[0].numpy()
            stem = os.path.splitext(os.path.basename(name[0]))[0]
            cv2.imwrite(os.path.join(a.out, stem + ".png"), pred)
            dice, iou, dist = compute_metrics(pred, gt, 4)
            rows.append({"frame_id": stem, "dice": float(dice), "iou": float(iou), "assd": float(dist)})
    summary = {k: float(np.mean([r[k] for r in rows])) for k in ("dice", "iou", "assd")}
    json.dump({"metric": "A2ONet a2onet/metrics.py", "fusion_alpha": a.alpha, "mean": summary, "frames": rows}, open(a.native_json, "w"), indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
