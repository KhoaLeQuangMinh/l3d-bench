"""Read L3D LabelMe annotations.

L3D stores polylines in LabelMe JSON (`shapes[*].label`, `shapes[*].points`) with the image size in
`imageHeight` / `imageWidth`. The class is the first letter of the label, exactly as every baseline
reads it: r = ridge, s = silhouette, l = ligament. Shape order is kept, because when masks are drawn
into one label map the shape drawn last wins (that is how the published metric works).
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CLASS_NAMES = {1: "ridge", 2: "silhouette", 3: "ligament"}
FIRST_LETTER_TO_CLASS = {"r": 1, "s": 2, "l": 3}


@dataclass
class Shape:
    raw_label: str
    class_id: int          # 0 if the label is not r/s/l (A2ONet draws such shapes with value 0)
    points: np.ndarray     # (N, 2) float32, native pixel coordinates (x, y)


@dataclass
class Annotation:
    height: int
    width: int
    shapes: list[Shape] = field(default_factory=list)

    def polylines(self, class_id: int) -> list[np.ndarray]:
        return [s.points for s in self.shapes if s.class_id == class_id and len(s.points) >= 2]


def class_of(label: str) -> int:
    return FIRST_LETTER_TO_CLASS.get(label.strip()[:1].lower(), 0)


def load_annotation(json_path: str | Path, strict: bool = False) -> Annotation:
    """Parse one LabelMe JSON. With strict=True, an unknown label raises instead of mapping to 0."""
    with open(json_path) as f:
        data = json.load(f)
    height, width = data.get("imageHeight"), data.get("imageWidth")
    if height is None or width is None:
        raise ValueError(f"{json_path}: imageHeight/imageWidth missing")
    shapes = []
    for shape in data.get("shapes", []):
        label = str(shape.get("label", ""))
        cid = class_of(label)
        if strict and cid == 0:
            raise ValueError(f"{json_path}: unknown label {label!r}")
        pts = np.asarray(shape.get("points", []), dtype=np.float32).reshape(-1, 2)
        shapes.append(Shape(raw_label=label, class_id=cid, points=pts))
    return Annotation(height=int(height), width=int(width), shapes=shapes)


def label_inventory(json_paths) -> Counter:
    """Count every distinct raw label string (Phase 1 check for typos such as 'rigde')."""
    counts: Counter = Counter()
    for p in json_paths:
        with open(p) as f:
            for shape in json.load(f).get("shapes", []):
                counts[str(shape.get("label", ""))] += 1
    return counts
