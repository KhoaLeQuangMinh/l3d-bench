"""Gate G3, part 1: can M1 memorise 10 training frames? (300 steps, no augmentation)

    python scripts/overfit_check.py                     # real L3D, pretrained Swin-T, GPU
    python scripts/overfit_check.py --steps 600

Passes when the mean A2ONet Dice on the same 10 frames reaches --target (default 0.8; thin 35 px lines
rarely reach exactly 1 because of the boundary pixels). If it does not, the loss, labels or learning rate
are wrong, and no full run should start.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def overfit(cfg: dict, index, n_frames: int = 10, steps: int = 300, device: str | None = None,
            log_every: int = 50, log=print) -> list[dict]:
    import torch
    from torch.utils.data import DataLoader

    from l3d_bench.data.dataset import L3DDataset
    from l3d_bench.eval.metrics import a2onet_dice_torch
    from l3d_bench.models.build import build_model
    from l3d_bench.train.seed import seed_everything

    seed_everything(0)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = bool(cfg["train"].get("amp")) and dev.type == "cuda"
    pool = list(index.loc[index["split"] == "train", "frame_id"])
    ids = pool[:: max(1, len(pool) // n_frames)][:n_frames]          # spread over patients
    ds = L3DDataset(index, ids, cfg, train=False)                       # no augmentation
    model = build_model(cfg).to(dev)
    tc = cfg["train"]
    opt = torch.optim.AdamW(model.param_groups(tc["lr_backbone"], tc["lr_head"], tc["weight_decay"]))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    loader = DataLoader(ds, batch_size=tc["batch_size"], shuffle=True, generator=torch.Generator().manual_seed(0))
    history, step = [], 0

    def evaluate() -> float:
        model.eval()
        scores = []
        with torch.no_grad():
            for b in DataLoader(ds, batch_size=1):
                with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                    pred = model.predict(b["image"].to(dev))
                scores.append(a2onet_dice_torch(pred.long(), b["label"].to(dev)).cpu())
        model.train()
        return float(torch.cat(scores).mean())

    t = time.time()
    model.train()
    while step < steps:
        for b in loader:
            extra = {}
            if getattr(model, "uses_junctions", False):
                extra = {"junctions": b["junctions"].to(dev), "junction_vis": b["junction_vis"].to(dev)}
            with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                loss = model(b["image"].to(dev), b["label"].to(dev), **extra).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.get("clip_grad_norm", 1e9))
            scaler.step(opt)
            scaler.update()
            step += 1
            if step % log_every == 0 or step == steps:
                row = {"step": step, "loss": round(loss.item(), 4), "dice": round(evaluate(), 4),
                       "minutes": round((time.time() - t) / 60, 2)}
                history.append(row)
                log(json.dumps(row))
            if step >= steps:
                break
    return history


def main():
    from l3d_bench.config import REPO_ROOT, load_config
    from l3d_bench.data.index import build_index

    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--target", type=float, default=0.8)
    ap.add_argument("--method", default="m1", choices=["m1", "m2"])
    args = ap.parse_args()
    cfg = load_config(REPO_ROOT / "configs" / f"{args.method}.yaml")
    hist = overfit(cfg, build_index(cfg["data"]["roots"]), args.frames, args.steps)
    final = hist[-1]["dice"]
    print(f"\noverfit Dice {final:.4f} after {args.steps} steps -> {'PASS' if final >= args.target else 'FAIL'}"
          f" (target {args.target})")
    return 0 if final >= args.target else 1


if __name__ == "__main__":
    sys.exit(main())
