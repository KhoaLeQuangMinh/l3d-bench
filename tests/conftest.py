"""A tiny synthetic L3D on disk: same folder layout and LabelMe format as the real data, small images."""
from __future__ import annotations

import json
import zlib
from pathlib import Path

import cv2
import numpy as np
import pytest

# patient -> (official split, n frames, (H, W)); Patient_32 mimics the 4K patient
PATIENTS = {
    "Patient_1": ("train", 5, (108, 192)), "Patient_2": ("train", 4, (108, 192)),
    "Patient_3": ("train", 6, (108, 192)), "Patient_4": ("train", 3, (108, 192)),
    "Patient_5": ("train", 4, (108, 192)), "Patient_32": ("train", 3, (216, 384)),
    "Patient_40": ("val", 6, (108, 192)), "Patient_18": ("val", 2, (108, 192)),
    "Patient_41": ("test", 4, (108, 192)), "Patient_21": ("test", 2, (108, 192)),
}
SPLIT_DIR = {"train": "Train", "val": "Val", "test": "Test"}


def make_shapes(h: int, w: int, raised: bool, rng: np.random.Generator):
    """Silhouette across the top, ridge across the bottom, ligament joining them (y-flipped if raised)."""
    def y(v):  # fraction of height
        v = 1 - v if raised else v
        return float(v * h)
    jitter = float(rng.uniform(-0.03, 0.03) * w)
    sil = [[0.15 * w + jitter, y(0.25)], [0.5 * w, y(0.15)], [0.85 * w + jitter, y(0.25)]]
    ridge = [[0.15 * w + jitter, y(0.27)], [0.5 * w, y(0.75)], [0.85 * w + jitter, y(0.27)]]
    lig = [[0.5 * w, y(0.16)], [0.5 * w, y(0.73)]]
    return [{"label": "s", "points": sil}, {"label": "r", "points": ridge}, {"label": "l", "points": lig}]


def write_frame(root: Path, stem: str, h: int, w: int, shapes, depth: bool = True):
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    img = np.random.default_rng(zlib.crc32(stem.encode())).integers(0, 255, (h, w, 3), dtype=np.uint8)
    cv2.imwrite(str(root / "images" / f"{stem}.jpg"), img)
    (root / "labels" / f"{stem}.json").write_text(json.dumps(
        {"shapes": shapes, "imageHeight": h, "imageWidth": w, "imagePath": f"{stem}.jpg"}))
    if depth:
        (root / "depth_AdelaiDepth").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(root / "depth_AdelaiDepth" / f"{stem}.png"), img[:, :, 0])


@pytest.fixture(scope="session")
def fake_l3d(tmp_path_factory):
    base = tmp_path_factory.mktemp("l3d")
    rng = np.random.default_rng(0)
    poses = {}
    for patient, (split, n, (h, w)) in PATIENTS.items():
        root = base / f"l3d-{split}" / SPLIT_DIR[split]
        for i in range(n):
            stem = f"{patient}_{(i + 1) * 60:07d}"
            raised = (i == 0 and patient in ("Patient_40", "Patient_1", "Patient_41")) or \
                     (patient == "Patient_3" and i < 2)
            write_frame(root, stem, h, w, make_shapes(h, w, raised, rng))
            poses[stem] = "raised" if raised else "canonical"
    roots = {s: base / f"l3d-{s}" / SPLIT_DIR[s] for s in ("train", "val", "test")}
    return {"base": base, "roots": roots, "poses": poses}


@pytest.fixture(scope="session")
def index(fake_l3d):
    from l3d_bench.data.index import build_index
    return build_index(fake_l3d["roots"])


@pytest.fixture
def cfg():
    from l3d_bench.config import load_config
    return load_config(overrides={"data": {"input_size": 64}})
