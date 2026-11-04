"""Entry point for one study run (M1-M3). Baselines run their own code; see baselines/<name>/.

    python -m l3d_bench.run --run-id m1__plain__official__s0
    python -m l3d_bench.run --run-id m1__plain__official__s0 --pilot          # 2 epochs on a few frames
    python -m l3d_bench.run --run-id m1__plain__f2__s0 --set train.epochs=40

Re-running the same command continues an unfinished run (from the local folder or from Drive) and does
nothing for a finished one (marker DONE on Drive), unless --force.

Result folder <out>/<run_id>/ (mirrored to gdrive:l3d-bench/results/<run_id>/):
    run_card.json            what ran: config, commit, seed, split, GPU, versions, best epoch, status
    history.csv, train.log   per-epoch loss, selection Dice, learning rates, minutes
    checkpoint_latest.pth    resume state (every epoch)
    best.pth                 weights of the best epoch on the selection set
    select_frames.csv / select_metrics.json      scores on the selection set (val / inner-val)
    heldout_frames.csv / heldout_metrics.json    scores on the scored set (test / held-out fold)
    pred_select.zip, pred_heldout.zip            PNG label maps
    DONE
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

from .config import REPO_ROOT, deep_merge, load_config
from .data.folds import load_folds, official_split, split_patients
from .data.index import build_index
from .runs.manifest import parse_run_id

OUR_METHODS = ("m1", "m2", "m3")
PILOT = {"train": 8, "select": 4, "heldout": 4, "epochs": 2}


def parse_overrides(pairs: list[str]) -> dict:
    out: dict = {}
    for pair in pairs or []:
        key, _, value = pair.partition("=")
        if not key or not _:
            raise ValueError(f"--set expects key.sub=value, got {pair!r}")
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)
    return out


def pilot_subset(ids: dict, poses: dict | None) -> dict:
    """A few frames per set, spread over patients; the training subset holds raised and canonical frames when pose
    labels exist (M3's sampler needs both)."""
    out = {}
    for k, v in ids.items():
        n = PILOT[k]
        if k == "train" and poses:
            raised = [f for f in v if poses.get(f) == "raised"]
            canon = [f for f in v if poses.get(f) == "canonical"]
            if raised and canon:
                h = n // 2
                out[k] = raised[:: max(1, len(raised) // h)][:h] + canon[:: max(1, len(canon) // (n - h))][: n - h]
                continue
        out[k] = v[:: max(1, len(v) // n)][:n]
    return out


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def resolve_split(split: str, index, cfg: dict) -> tuple[dict, dict]:
    """-> ({'train','select','heldout'}: frame id lists, info about the split)."""
    if split == "official":
        pats = official_split(index)
        info = {"split": "official", "select_set": "official val", "scored_set": "official test"}
    else:
        path = REPO_ROOT / cfg["data"]["folds_file"]
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing: fold runs need the committed folds file "
                                    "(scripts/make_folds.py, after pose labelling)")
        pats = split_patients(load_folds(path), split)
        info = {"split": split, "select_set": "inner-val", "scored_set": f"held-out fold {split}",
                "folds_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    ids = {}
    for key, src in (("train", "train"), ("select", "inner_val"), ("heldout", "heldout")):
        ids[key] = list(index.loc[index["patient"].isin(pats[src]), "frame_id"])
        info[f"{key}_patients"] = sorted(pats[src])
        info[f"{key}_frames"] = len(ids[key])
    if set(info["train_patients"]) & (set(info["select_patients"]) | set(info["heldout_patients"])):
        raise AssertionError("patient leakage between training and evaluation sets")
    return ids, info


def environment() -> dict:
    env = {"python": platform.python_version()}
    try:
        import torch
        import transformers
        env.update(torch=torch.__version__, transformers=transformers.__version__,
                   gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    except ImportError:
        pass
    return env


def load_poses(cfg: dict, required: bool) -> dict | None:
    from .data.pose import load_pose_labels
    path = REPO_ROOT / cfg["data"]["pose_file"]
    if path.exists():
        return load_pose_labels(path)
    if required:
        raise FileNotFoundError(f"{path} is missing: M3's pose-balanced sampler needs pose labels")
    return None


def run(run_id: str, cfg: dict, out_root: str | Path, *, drive=None, index=None, poses=None, pilot=False,
        force=False, t0: float | None = None, device: str | None = None) -> dict:
    import torch

    from .data.dataset import L3DDataset
    from .eval.aggregate import summarize
    from .eval.export import export_predictions
    from .eval.score import score_folder
    from .models.build import build_model
    from .train.loop import Trainer
    from .train.seed import seed_everything

    t0 = time.monotonic() if t0 is None else t0
    meta = parse_run_id(run_id)
    if meta["method"] not in OUR_METHODS:
        raise ValueError(f"{run_id}: baselines run their own code (baselines/{meta['method']}/README.md)")
    sync_id = f"_pilot__{run_id}" if pilot else run_id
    run_dir = Path(out_root) / sync_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if not force and ((run_dir / "DONE").exists() or (drive is not None and drive.exists(sync_id, "DONE"))):
        print(f"[{sync_id}] already finished (DONE marker); nothing to do. Use --force to redo it.")
        return {"status": "already_done"}

    if pilot:
        cfg = deep_merge(cfg, {"train": {"epochs": PILOT["epochs"]}})
    seed_everything(meta["seed"])
    if index is None:
        index = build_index(cfg["data"]["roots"])
    if poses is None:
        poses = load_poses(cfg, required=cfg.get("sampler") == "pose_balanced")
    ids, split_info = resolve_split(meta["split"], index, cfg)
    if pilot:
        ids = pilot_subset(ids, poses)
        split_info.update({f"{k}_frames": len(v) for k, v in ids.items()})

    prev = {}
    if (run_dir / "run_card.json").exists():
        prev = json.loads((run_dir / "run_card.json").read_text())
    elif drive is not None and drive.enabled and drive.exists(sync_id, "run_card.json"):
        drive.pull_file(sync_id, "run_card.json", run_dir)
        prev = json.loads((run_dir / "run_card.json").read_text())
    sessions = prev.get("sessions", []) + [{"started": time.strftime("%Y-%m-%d %H:%M"), "commit": git_commit()}]
    card = {"run_id": run_id, **meta, "status": "running", "pilot": pilot, "commit": git_commit(), "sessions": sessions,
            "config": cfg, **split_info, "environment": environment(),
            "selection_rule": "best epoch by mean A2ONet Dice on the selection set; earliest wins ties"}
    card_path = run_dir / "run_card.json"

    def write_card(**updates):
        card.update(updates)
        card["hours_this_session"] = round((time.monotonic() - t0) / 3600, 3)
        card_path.write_text(json.dumps(card, indent=2, default=str))
        if drive is not None and drive.enabled:
            drive.push_file(card_path, sync_id)

    write_card()
    model = build_model(cfg)
    train_ds = L3DDataset(index, ids["train"], cfg, train=True, poses=poses, seed=meta["seed"])
    select_ds = L3DDataset(index, ids["select"], cfg, train=False, poses=poses)
    trainer = Trainer(cfg, model, train_ds, select_ds, run_dir, run_id=sync_id, seed=meta["seed"], drive=drive,
                      device=device, t0=t0)
    state = trainer.fit()
    if state["status"] == "paused":
        write_card(status="paused", done_epochs=state["done_epochs"], best=state["best"])
        return state
    if trainer.hours() + float(cfg["train"].get("final_reserve_hours", 0)) > float(cfg["train"]["max_session_hours"]):
        trainer.log("training finished but too little time is left to export; re-run to export and score")
        write_card(status="paused", done_epochs=state["done_epochs"], best=state["best"])
        return {**state, "status": "paused"}

    best = trainer.load_best()
    trainer.log(f"best epoch {best['epoch']} (select Dice {best['select_dice']:.4f}); exporting and scoring")
    results = {}
    for name in ("select", "heldout"):
        ds = L3DDataset(index, ids[name], cfg, train=False, poses=poses)
        pred_dir = run_dir / f"pred_{name}"
        export_predictions(model, ds, pred_dir, trainer.device, amp=trainer.amp,
                           batch_size=cfg["train"].get("eval_batch_size", 1))
        frames = score_folder(pred_dir, index, ids[name], poses, cfg["metric"])
        frames.to_csv(run_dir / f"{name}_frames.csv", index=False)
        summary = summarize(frames)
        (run_dir / f"{name}_metrics.json").write_text(json.dumps(summary, indent=2))
        shutil.make_archive(str(run_dir / f"pred_{name}"), "zip", pred_dir)
        results[name] = summary["mean"]
        trainer.log(f"{name}: " + "  ".join(f"{k} {v:.4f}" for k, v in summary["mean"].items()))
        if drive is not None and drive.enabled:
            for f in (f"{name}_frames.csv", f"{name}_metrics.json", f"pred_{name}.zip"):
                drive.push_file(run_dir / f, sync_id)

    gap = abs(results["select"]["dice"] - best["select_dice"])
    if gap > 0.005:  # same frames, same Dice: only AMP noise may differ
        trainer.log(f"WARNING: selection Dice during training {best['select_dice']:.4f} vs final scoring "
                    f"{results['select']['dice']:.4f}")
    write_card(status="done", done_epochs=state["done_epochs"], best_epoch=best["epoch"],
               best_select_dice_training=best["select_dice"], select=results["select"],
               heldout=results["heldout"], minutes_per_epoch=[h["epoch_minutes"] for h in trainer.history])
    (run_dir / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    if drive is not None and drive.enabled:
        trainer._push("train.log", "history.csv", "DONE")
        trainer.wait_uploads()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"status": "done", "best": best, **results}


def main(argv=None) -> int:
    t0 = time.monotonic()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out", default="/kaggle/working/runs" if Path("/kaggle/working").exists() else "runs")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. train.epochs=40")
    ap.add_argument("--pilot", action="store_true", default=os.environ.get("PILOT_MODE") == "1")
    ap.add_argument("--no-drive", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    meta = parse_run_id(args.run_id)
    cfg = load_config(REPO_ROOT / "configs" / f"{meta['method']}.yaml", parse_overrides(args.set))
    drive = None
    if not args.no_drive:
        from .train.drive import DriveSync
        d = cfg["drive"]
        drive = DriveSync.from_kaggle(secret_name=d["secret_name"], remote=d["remote"], base_dir=d["base_dir"])
        if not drive.enabled:
            print("[drive] disabled: results stay in", args.out)
    state = run(args.run_id, cfg, args.out, drive=drive, pilot=args.pilot, force=args.force, t0=t0)
    print(json.dumps({k: v for k, v in state.items()}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
