"""Turn polylines into label maps, reproducing A2ONet's `render_json` exactly.

A2ONet (data/dataset.py): canvas of size (imageHeight, imageWidth), thickness int(35 * H / 1080),
each shape drawn with cv2.line between consecutive int-cast points, colour = class id (0 for unknown
labels), later shapes overwrite earlier ones. The label map is then resized to 1024 x 1024 with
nearest-neighbour interpolation.
"""
from __future__ import annotations

import cv2
import numpy as np

from .annotations import Annotation


def line_thickness(height: int, width_at_1080: int = 35) -> int:
    return int(width_at_1080 * height / 1080)


def render_label_map(ann: Annotation, width_at_1080: int = 35) -> np.ndarray:
    """One label per pixel (0 background, 1 ridge, 2 silhouette, 3 ligament) at native size."""
    canvas = np.zeros((ann.height, ann.width), dtype=np.uint8)
    thickness = line_thickness(ann.height, width_at_1080)
    for shape in ann.shapes:
        pts = shape.points
        for i in range(1, len(pts)):
            p1 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
            p2 = (int(pts[i][0]), int(pts[i][1]))
            cv2.line(canvas, p1, p2, int(shape.class_id), thickness)
    return canvas


def render_class_masks(ann: Annotation, width_at_1080: int = 35) -> np.ndarray:
    """(3, H, W) binary masks drawn independently per class (no overwriting); used for the overlap check."""
    out = np.zeros((3, ann.height, ann.width), dtype=np.uint8)
    thickness = line_thickness(ann.height, width_at_1080)
    for shape in ann.shapes:
        if shape.class_id == 0:
            continue
        pts = shape.points
        for i in range(1, len(pts)):
            p1 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
            p2 = (int(pts[i][0]), int(pts[i][1]))
            cv2.line(out[shape.class_id - 1], p1, p2, 1, thickness)
    return out


def resize_label_map(label: np.ndarray, size: int = 1024) -> np.ndarray:
    return cv2.resize(label, (size, size), interpolation=cv2.INTER_NEAREST)


def gt_for_scoring(ann: Annotation, width_at_1080: int = 35, size: int = 1024) -> np.ndarray:
    """Ground truth exactly as the published metric sees it."""
    return resize_label_map(render_label_map(ann, width_at_1080), size)


def overlap_fraction(ann: Annotation, width_at_1080: int = 35) -> tuple[int, int]:
    """(pixels covered by >= 2 classes, pixels covered by >= 1 class), at native size."""
    cover = render_class_masks(ann, width_at_1080).sum(axis=0)
    return int((cover >= 2).sum()), int((cover >= 1).sum())
