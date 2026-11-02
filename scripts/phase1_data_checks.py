"""Phase 1 / gate G1 on the real L3D: index, label inventory, class-overlap share, overlays, report.

    python scripts/phase1_data_checks.py --out reports/g1            # Kaggle paths from configs/base.yaml
    python scripts/phase1_data_checks.py --train ... --val ... --test ... --out reports/g1

Writes: data/index.csv, <out>/g1_report.json, <out>/overlays/*.jpg (20 random frames + every non-1080p frame).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from l3d_bench.config import REPO_ROOT, load_config
from l3d_bench.data.annotations import label_inventory, load_annotation
from l3d_bench.data.index import build_index, summary
from l3d_bench.data.masks import overlap_fraction, render_label_map

COLORS = {1: (0, 0, 255), 2: (0, 255, 0), 3: (255, 0, 0)}  # BGR: ridge red, silhouette green, ligament blue


def overlay(image_path: str, label: np.ndarray) -> np.ndarray:
    img = cv2.imread(image_path)
    if img.shape[:2] != label.shape:
        img = cv2.resize(img, (label.shape[1], label.shape[0]))
    out = img.copy()
    for c, col in COLORS.items():
        out[label == c] = col
    return cv2.addWeighted(img, 0.5, out, 0.5, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train"), ap.add_argument("--val"), ap.add_argument("--test")
    ap.add_argument("--out", default="reports/g1")
    ap.add_argument("--n_overlays", type=int, default=20)
    args = ap.parse_args()
    cfg = load_config()
    roots = {s: getattr(args, s) or cfg["data"]["roots"][s] for s in ("train", "val", "test")}
    out = Path(args.out)
    (out / "overlays").mkdir(parents=True, exist_ok=True)

    index = build_index(roots)
    index_path = REPO_ROOT / cfg["data"]["index_file"]
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(index_path, index=False)

    report = {"roots": {k: str(v) for k, v in roots.items()}, "index": summary(index)}
    report["label_inventory"] = dict(label_inventory(index["label_path"]).most_common())

    both = any_ = 0
    per_frame = []
    for _, r in index.iterrows():
        b, a = overlap_fraction(load_annotation(r["label_path"]), cfg["metric"]["line_width_at_1080"])
        both, any_ = both + b, any_ + a
        per_frame.append((b / a if a else 0.0, r["frame_id"]))
    report["class_overlap_share"] = round(both / any_, 5) if any_ else 0.0
    report["class_overlap_top_frames"] = [f for _, f in sorted(per_frame, reverse=True)[:10]]

    rng = random.Random(0)
    picks = rng.sample(list(index["frame_id"]), min(args.n_overlays, len(index)))
    picks += [f for f in index.loc[index["height"] != 1080, "frame_id"] if f not in picks]
    rows = index.set_index("frame_id")
    for fid in picks:
        lab = render_label_map(load_annotation(rows.loc[fid, "label_path"]), cfg["metric"]["line_width_at_1080"])
        cv2.imwrite(str(out / "overlays" / f"{fid}.jpg"), overlay(rows.loc[fid, "image_path"], lab))
    report["overlays"] = picks

    (out / "g1_report.json").write_text(json.dumps(report, indent=2))
    s = report["index"]
    print(json.dumps({"frames": s["frames"], "patients": s["patients"], "n_patients": s["n_patients"],
                      "non_1080p": len(s["non_1080p_frames"]), "labels": report["label_inventory"],
                      "class_overlap_share": report["class_overlap_share"]}, indent=2))
    print(f"index -> {index_path}\nreport -> {out / 'g1_report.json'}")


if __name__ == "__main__":
    main()
