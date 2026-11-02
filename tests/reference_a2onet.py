"""Verbatim copies of A2ONet's label rendering and metric (hyperiondk115/A2ONet @ 765ada1), kept only as a
test oracle: our implementation must give identical numbers. Source files: data/dataset.py, a2onet/metrics.py.
"""
import json

import cv2
import numpy as np
from medpy.metric.binary import assd

LINE_WIDTH = 35
LABEL_IDS = {'r': 1, 's': 2, 'l': 3}
EMPTY_ASSD = 80.0


def _draw_polyline(canvas, points, color, thickness):
    for i in range(1, len(points)):
        pt1 = tuple(int(v) for v in points[i - 1])
        pt2 = tuple(int(v) for v in points[i])
        cv2.line(canvas, pt1, pt2, color, thickness)


def render_json(json_path, width=LINE_WIDTH):
    with open(json_path, 'r') as f:
        data = json.load(f)
    h, w = data.get('imageHeight'), data.get('imageWidth')
    canvas = np.zeros((int(h), int(w)), dtype=np.uint8)
    thickness = int(width * canvas.shape[0] / 1080)
    for shape in data['shapes']:
        color = LABEL_IDS.get(shape['label'][:1].lower(), 0)
        _draw_polyline(canvas, shape['points'], color, thickness)
    return canvas


def load_mask_label(json_path, size):
    label = render_json(json_path)
    return cv2.resize(label, (size, size), interpolation=cv2.INTER_NEAREST)


def compute_metrics(pred, gt, num_classes=4, smooth=1e-5):
    pred_fg = np.stack([pred == c for c in range(1, num_classes)])
    gt_fg = np.stack([gt == c for c in range(1, num_classes)])

    inter = np.logical_and(pred_fg, gt_fg).sum()
    dice = (2 * inter + smooth) / (pred_fg.sum() + gt_fg.sum() + smooth)
    iou = dice / (2 - dice)

    if pred_fg.any() and gt_fg.any():
        dist = float(assd(pred_fg, gt_fg))
    else:
        dist = EMPTY_ASSD
    return float(dice), float(iou), dist
