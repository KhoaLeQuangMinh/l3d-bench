"""CLI for one baseline run. See runner.py for what happens.

    python -m l3d_bench.baselines.run --method toponet --split official
    python -m l3d_bench.baselines.run --method toponet --split official --released     # G2
    python -m l3d_bench.baselines.run --method a2onet --split official --pilot         # 8 frames, few epochs
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from ..config import load_config
from .runner import make_ctx, run_baseline
from .specs import SPECS


def main(argv=None) -> int:
    t0 = time.monotonic()
    kaggle = Path("/kaggle/working").exists()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, choices=sorted(SPECS))
    ap.add_argument("--split", default="official")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--released", action="store_true", help="score the authors' released weights (no training)")
    ap.add_argument("--pilot", action="store_true", default=os.environ.get("PILOT_MODE") == "1")
    ap.add_argument("--out", default="/kaggle/working/runs" if kaggle else "runs")
    ap.add_argument("--work", default="/kaggle/working/bl" if kaggle else "bl_work")
    ap.add_argument("--no-drive", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config()
    drive = None
    if not a.no_drive:
        from ..train.drive import DriveSync
        d = cfg["drive"]
        drive = DriveSync.from_kaggle(secret_name=d["secret_name"], remote=d["remote"], base_dir=d["base_dir"])
    ctx = make_ctx(a.method, a.split, a.seed, a.out, a.work, cfg, pilot=a.pilot, released=a.released, drive=drive)
    state = run_baseline(SPECS[a.method], ctx, t0=t0, released=a.released, force=a.force)
    print(json.dumps(state, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
