"""Import Khoa's Oct 2026 orientation audit (Surgical AI/data/user_audit_results.csv) as pose labels.

The audit has one row per L3D frame: status normal / flipped for the 1043 Train + Val frames, 'unreviewed' for the
109 Test frames (kept untouched on purpose). Its `status` column is the reviewed label: it disagrees with the
automatic `heuristic_flipped` column on 136 frames. Mapping: normal -> canonical, flipped -> raised.
Test frames are not written: folds and M3 only use the Train + Val pool.

    python scripts/import_pose_audit.py --audit "/path/to/user_audit_results.csv"
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from l3d_bench.config import REPO_ROOT, load_config

MAP = {"normal": "canonical", "flipped": "raised"}


def convert(audit_path: str | Path, labeler: str = "khoa_audit_2026-10-07") -> list[dict]:
    rows = list(csv.DictReader(open(audit_path)))
    out, seen = [], set()
    for r in rows:
        if r["id"] + ".jpg" != r["filename"]:
            raise ValueError(f"id/filename mismatch: {r['id']}")
        if r["split"] == "Test":
            if r["status"] != "unreviewed":
                raise ValueError(f"test frame {r['id']} has a label; test must stay unlabeled here")
            continue
        if r["status"] not in MAP:
            raise ValueError(f"{r['id']}: unexpected status {r['status']!r}")
        if r["id"] in seen:
            raise ValueError(f"duplicate {r['id']}")
        seen.add(r["id"])
        out.append({"frame_id": r["id"], "pose": MAP[r["status"]], "labeler": labeler, "split": r["split"].lower()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", required=True)
    ap.add_argument("--out", default=str(REPO_ROOT / load_config()["data"]["pose_file"]))
    a = ap.parse_args()
    rows = convert(a.audit)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame_id", "pose", "labeler"])
        w.writeheader()
        w.writerows({k: r[k] for k in ("frame_id", "pose", "labeler")} for r in rows)
    print(len(rows), "frames ->", a.out, dict(Counter((r["split"], r["pose"]) for r in rows)))


if __name__ == "__main__":
    main()
