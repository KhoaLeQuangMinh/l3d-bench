"""Run a TopoNet checkpoint on a list of frames: PNG label maps for our scorer, plus TopoNet's own metric.

Runs from inside the TopoNet repo (cwd), with its own dataset class, so preprocessing is exactly the released
test.py: cv2 resize to 1024, ToTensor, argmax over 4 classes. Its native metric (pooled Dice; medpy ASSD on
the 3 class masks, 80 when the prediction is empty, as test.py; and the per-class surface-distance ASSD of its
train.py validation loop as assd_trainpy) is computed on its own ground truth (35 px fixed,
4K canvas chosen by path substring) and written to --native_json, to compare with the paper.

    python export_predictions.py --model_path best_val_model.pth --depth_path vitb.pth --list frames.txt --out preds/
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
from medpy.metric import assd
import surface_distance
from surface_distance import metrics as sd_metrics
from torch.utils.data import DataLoader

sys.path.insert(0, os.getcwd())
from models.TopoNet import TopoNet  # noqa: E402
from utils.dataset import LandmarkDataset  # noqa: E402
from utils.metrics import evaluation  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--depth_path", required=True)
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--native_json", required=True)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = [line.strip() for line in open(a.list) if line.strip()]
    os.makedirs(a.out, exist_ok=True)
    model = TopoNet(1024, 1024, depth_path=a.depth_path).to(device)
    model.load_state_dict(torch.load(a.model_path, map_location=device))
    model.eval()
    loader = DataLoader(LandmarkDataset(files, transform=None, mode="test"), batch_size=1, shuffle=False)
    rows = []
    with torch.no_grad():
        for X, depth, y, name in loader:
            output, _ = model(X.to(device))
            pred_map = torch.argmax(torch.softmax(output, dim=1), dim=1)[0].cpu().numpy().astype(np.uint8)
            gt_map = torch.argmax(y, dim=1)[0].numpy()
            stem = os.path.splitext(os.path.basename(name[0]))[0]
            cv2.imwrite(os.path.join(a.out, stem + ".png"), pred_map)
            pred = np.array([pred_map == i for i in range(4)]).astype(np.uint8)
            gt = np.array([gt_map == i for i in range(4)]).astype(np.uint8)
            iou, dice = evaluation(pred[1:].flatten(), gt[1:].flatten())
            if np.count_nonzero(pred[1:]) == 0:
                dist = 80.0
            elif np.count_nonzero(gt[1:]) == 0:   # test.py would crash here (medpy needs both non-empty)
                dist = float("nan")
            else:
                dist = float(assd(pred[1:], gt[1:]))
            # ASSD as train.py's validation loop computes it: surface-distance per class, mean of the 3 classes,
            # 80 if the prediction is empty or the mean is >= 500 (a class missing on either side gives inf)
            if np.count_nonzero(pred[1:]) == 0:
                dist_train = 80.0
            else:
                per_class = []
                for i in range(3):
                    sd = sd_metrics.compute_surface_distances(gt[i + 1].astype(bool), pred[i + 1].astype(bool), (1.0, 1.0))
                    per_class.append(surface_distance.compute_average_surface_distance(sd)[1])
                dist_train = float(np.mean(per_class)) if np.mean(per_class) < 500 else 80.0
            rows.append({"frame_id": stem, "dice": float(dice), "iou": float(iou), "assd": dist,
                         "assd_trainpy": dist_train})
    summary = {k: float(np.nanmean([r[k] for r in rows])) for k in ("dice", "iou", "assd", "assd_trainpy")}
    json.dump({"metric": "TopoNet test.py (its own GT)", "mean": summary, "frames": rows}, open(a.native_json, "w"), indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
