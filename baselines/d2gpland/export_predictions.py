"""Run a D2GPLand checkpoint (model + edge prototypes) on a list of frames: PNG label maps for our scorer,
plus D2GPLand's own metric.

Runs inside the D2GPLand repo (cwd; SAM is read from sam_path/sam_vit_b_01ec64.pth there) and mirrors its
test.py: its LandmarkDataset with ToTensor (image, AdelaiDepth map, GT with 30 px lines and bilinear resize),
argmax of the softmax over 4 classes. Its test.py reports pooled Dice and IoU only (no ASSD).

    python export_predictions.py --model_path best_val_model_path.pth --prototype_path best_val_prototype_path.pth \
        --list frames.txt --out preds/ --native_json native.json
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

sys.path.insert(0, os.getcwd())
from models.D2GPLand import D2GPLand, Edge_Prototypes  # noqa: E402
from utils.dataset import LandmarkDataset  # noqa: E402
from utils.metrics import evaluation  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--prototype_path", required=True)
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--native_json", required=True)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = [line.strip() for line in open(a.list) if line.strip()]
    os.makedirs(a.out, exist_ok=True)
    model = D2GPLand(1024, 1024).to(device)
    model.load_state_dict(torch.load(a.model_path, map_location=device)["model"])
    protos = Edge_Prototypes(num_classes=3, feat_dim=256).to(device)
    protos.load_state_dict(torch.load(a.prototype_path, map_location=device)["model"])
    model.eval()
    protos.eval()
    loader = DataLoader(LandmarkDataset(files, transform=T.Compose([T.ToTensor()]), mode="test"), batch_size=1,
                        shuffle=False)
    rows = []
    with torch.no_grad():
        for X, depth, y, name in loader:
            output, feature, edge_out = model(X.to(device), depth.to(device), protos())
            pred_map = torch.argmax(torch.softmax(output, dim=1), dim=1)[0].cpu().numpy().astype(np.uint8)
            gt_map = torch.argmax(y, dim=1)[0].numpy()
            stem = os.path.splitext(os.path.basename(name[0]))[0]
            cv2.imwrite(os.path.join(a.out, stem + ".png"), pred_map)
            pred = np.array([pred_map == i for i in range(4)]).astype(np.uint8)
            gt = np.array([gt_map == i for i in range(4)]).astype(np.uint8)
            iou, dice = evaluation(pred[1:].flatten(), gt[1:].flatten())
            rows.append({"frame_id": stem, "dice": float(dice), "iou": float(iou)})
    summary = {k: float(np.mean([r[k] for r in rows])) for k in ("dice", "iou")}
    json.dump({"metric": "D2GPLand test.py (its own GT)", "mean": summary, "frames": rows}, open(a.native_json, "w"),
              indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
