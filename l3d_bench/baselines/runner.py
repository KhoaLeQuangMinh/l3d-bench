"""Run one official baseline (TopoNet, A2ONet, D2GPLand, BCRNet) for one split, from its own repo, end to end.

    python -m l3d_bench.baselines.run --method toponet --split official            # train + score
    python -m l3d_bench.baselines.run --method toponet --split official --released # G2: released weights only

Steps (re-running the same command continues where it stopped, like run.py):
  1. clone the repo at the pinned commit, apply baselines/<method>/patches/*.diff (documented diffs only)
  2. install what the repo needs, download pretrained weights (cached in the work folder)
  3. link a data root in the repo's folder style:
       official: Train = train, Val = official val, Test = official test (the repo's own procedure)
       fold fK:  Train = train patients, Val = Test = inner-val (whatever it selects on is never scored)
  4. train with the authors' recipe; the resume patch saves resume.pth every epoch and exits with code 3
     before Kaggle's limit; checkpoints and logs are uploaded to Drive in the background
  5. export PNG label maps with the val-selected checkpoint for the selection set and the scored set, and,
     on the official split, also with the repo's own (test-selected) checkpoint: that is the reproduction
     number, and the gap between the two measures test-selection inflation
  6. score everything with l3d_bench.eval (our protocol) and keep the repo's own metric next to it
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..config import REPO_ROOT
from ..data.index import build_index
from ..eval.aggregate import summarize
from ..eval.score import score_folder

PAUSED_EXIT = 3


@dataclass
class Ctx:
    method: str
    split: str
    seed: int
    sync_id: str
    run_dir: Path            # checkpoints, logs, predictions, metrics (mirrored to Drive)
    work: Path               # repo clone, weights, data roots (not synced)
    cfg: dict
    index: object
    ids: dict
    pilot: bool = False
    drive: object = None
    repo: Path = field(init=False)
    weights: Path = field(init=False)
    root: Path = field(init=False)

    def __post_init__(self):
        self.repo = self.work / f"{self.method}-src"
        self.weights = self.work / "weights"
        self.root = self.work / "roots" / self.sync_id
        self.weights.mkdir(parents=True, exist_ok=True)

    def log(self, msg: str) -> None:
        line = f"[{self.sync_id}] {msg}"
        print(line, flush=True)
        with open(self.run_dir / "runner.log", "a") as f:
            f.write(line + "\n")


def sh(cmd, cwd=None, env=None, check=True) -> int:
    """Run a shell command, streaming its output."""
    print("$", cmd if isinstance(cmd, str) else " ".join(map(str, cmd)), flush=True)
    res = subprocess.run(cmd, cwd=cwd, env=env, shell=isinstance(cmd, str))
    if check and res.returncode != 0:
        raise RuntimeError(f"command failed ({res.returncode}): {cmd}")
    return res.returncode


def pip(*pkgs: str) -> None:
    sh([sys.executable, "-m", "pip", "install", "-q", *pkgs])


# ---- repo and data ---------------------------------------------------------------------------
def clone_and_patch(ctx: Ctx, spec) -> list[dict]:
    """Clone at the pinned commit and apply our documented patches (idempotent)."""
    marker = ctx.repo / ".l3d_patched"
    patches = sorted((REPO_ROOT / "baselines" / spec.name / "patches").glob("*.diff"))
    info = [{"file": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()[:16]} for p in patches]
    if marker.exists() and json.loads(marker.read_text()) == info:
        return info
    if ctx.repo.exists():
        shutil.rmtree(ctx.repo)
    sh(["git", "clone", "-q", spec.repo_url, str(ctx.repo)])
    sh(["git", "-C", str(ctx.repo), "checkout", "-q", spec.commit])
    for p in patches:
        sh(["git", "-C", str(ctx.repo), "apply", "--whitespace=nowarn", str(p)])
    marker.write_text(json.dumps(info))
    return info


def link_root(ctx: Ctx, style: str) -> Path:
    """Symlinked data root with Train/Val/Test (or lower case) for this split, rebuilt every time."""
    names = {"upper": ("Train", "Val", "Test"), "lower": ("train", "val", "test")}[style]
    if ctx.split == "official":
        sets = (ctx.ids["train"], ctx.ids["select"], ctx.ids["heldout"])
    else:
        sets = (ctx.ids["train"], ctx.ids["select"], ctx.ids["select"])
    if ctx.root.exists():
        shutil.rmtree(ctx.root)
    rows = ctx.index.set_index("frame_id")
    for name, ids in zip(names, sets):
        for sub in ("images", "labels", "depth_AdelaiDepth", "sam", "s_bezier"):
            (ctx.root / name / sub).mkdir(parents=True, exist_ok=True)
        for fid in ids:
            r = rows.loc[fid]
            img = Path(r["image_path"])
            os.symlink(img, ctx.root / name / "images" / img.name)
            os.symlink(r["label_path"], ctx.root / name / "labels" / Path(r["label_path"]).name)
            if r["depth_path"]:
                os.symlink(r["depth_path"], ctx.root / name / "depth_AdelaiDepth" / Path(r["depth_path"]).name)
            for sub in ("sam", "s_bezier"):                     # BCRNet's precomputed inputs
                src = ctx.work / "precomputed" / sub
                for cand in src.glob(f"{fid}.*") if src.exists() else []:
                    os.symlink(cand, ctx.root / name / sub / cand.name)
    return ctx.root


def write_list(ctx: Ctx, name: str, ids) -> Path:
    rows = ctx.index.set_index("frame_id")
    path = ctx.run_dir / f"frames_{name}.txt"
    path.write_text("\n".join(rows.loc[f, "image_path"] for f in ids) + "\n")
    return path


# ---- Drive -----------------------------------------------------------------------------------
class Watcher:
    """Uploads the repo's checkpoints and logs to Drive whenever they change (in the background)."""

    def __init__(self, ctx: Ctx, names: list[str], every: float = 120.0):
        self.ctx, self.names, self.every = ctx, names, every
        self.seen: dict[str, float] = {}
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def scan(self) -> None:
        d = self.ctx.drive
        if d is None or not d.enabled:
            return
        for n in self.names:                       # in list order: best_* before resume.pth
            p = self.ctx.run_dir / n
            if p.exists() and self.seen.get(n) != p.stat().st_mtime:
                self.seen[n] = p.stat().st_mtime
                d.push_file_async(p, self.ctx.sync_id)

    def _loop(self):
        while not self._stop.wait(self.every):
            self.scan()

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self.scan()
        if self.ctx.drive is not None and self.ctx.drive.enabled:
            self.ctx.drive.wait()


def pull_state(ctx: Ctx, names: list[str]) -> None:
    d = ctx.drive
    if d is None or not d.enabled or (ctx.run_dir / "resume.pth").exists():
        return
    if d.exists(ctx.sync_id, "resume.pth") or d.exists(ctx.sync_id, "TRAINING_DONE"):
        ctx.log("pulling training state from Drive")
        for n in names:
            if d.exists(ctx.sync_id, n):
                d.pull_file(ctx.sync_id, n, ctx.run_dir)


def push(ctx: Ctx, *names: str) -> None:
    if ctx.drive is not None and ctx.drive.enabled:
        for n in names:
            if (ctx.run_dir / n).exists():
                ctx.drive.push_file(ctx.run_dir / n, ctx.sync_id)


# ---- training and scoring --------------------------------------------------------------------
def train(ctx: Ctx, spec, stop_at: float) -> str:
    if (ctx.run_dir / "TRAINING_DONE").exists():
        return "trained"
    cmd, cwd, extra_env = spec.train_cmd(ctx)
    env = {**os.environ, **extra_env, "L3D_STOP_AT": str(stop_at), "PYTHONUNBUFFERED": "1",
           "L3D_SAVE_DIR": str(ctx.run_dir), "WANDB_MODE": "offline"}
    if ctx.pilot:
        env["L3D_MAX_EPOCHS"] = str(spec.pilot_epochs)
    ctx.log("training: " + " ".join(map(str, cmd)))
    with Watcher(ctx, spec.sync_files + ["train.log", "epochs.jsonl"]):
        with open(ctx.run_dir / "train.log", "a") as log:
            proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1)
            for line in proc.stdout:
                if "it/s]" in line or "s/it]" in line:   # tqdm noise stays out of the log
                    continue
                print(line, end="", flush=True)
                log.write(line)
            code = proc.wait()
    if code == PAUSED_EXIT:
        return "paused"
    if code != 0:
        raise RuntimeError(f"{spec.name} training exited with code {code}; see train.log")
    if not (ctx.run_dir / "TRAINING_DONE").exists():
        raise RuntimeError("training ended without TRAINING_DONE")
    return "trained"


def export_and_score(ctx: Ctx, spec, ckpt: Path, set_name: str, tag: str) -> dict:
    ids = ctx.ids[set_name]
    out = ctx.run_dir / f"pred_{tag}_{set_name}"
    native = ctx.run_dir / f"repometric_{tag}_{set_name}.json"
    sh(spec.export_cmd(ctx, ckpt, write_list(ctx, f"{tag}_{set_name}", ids), out, native), cwd=ctx.repo)
    frames = score_folder(out, ctx.index, ids, None, ctx.cfg["metric"])
    frames.to_csv(ctx.run_dir / f"{tag}_{set_name}_frames.csv", index=False)
    summary = summarize(frames)
    summary["native"] = json.loads(native.read_text())["mean"] if native.exists() else None
    (ctx.run_dir / f"{tag}_{set_name}_metrics.json").write_text(json.dumps(summary, indent=2))
    shutil.make_archive(str(out), "zip", out)
    push(ctx, f"{tag}_{set_name}_frames.csv", f"{tag}_{set_name}_metrics.json", f"pred_{tag}_{set_name}.zip",
         native.name)
    ctx.log(f"{tag} on {set_name}: ours " + "  ".join(f"{k} {v:.4f}" for k, v in summary["mean"].items())
            + (f" | repo metric {summary['native']}" if summary["native"] else ""))
    return {"ours": summary["mean"], "native": summary["native"]}


def run_baseline(spec, ctx: Ctx, *, t0: float, released: bool = False, force: bool = False) -> dict:
    ctx.run_dir.mkdir(parents=True, exist_ok=True)
    d = ctx.drive
    if not force and ((ctx.run_dir / "DONE").exists() or (d is not None and d.enabled and d.exists(ctx.sync_id, "DONE"))):
        print(f"[{ctx.sync_id}] already finished (DONE); nothing to do. Use --force to redo it.")
        return {"status": "already_done"}
    budget = float(ctx.cfg["train"]["max_session_hours"])
    reserve = float(spec.reserve_hours)
    card = {"run_id": ctx.sync_id, "method": spec.name, "split": ctx.split, "seed": ctx.seed, "pilot": ctx.pilot,
            "repo": spec.repo_url, "repo_commit": spec.commit, "recipe": spec.recipe, "released_weights": released,
            "frames": {k: len(v) for k, v in ctx.ids.items()}, "status": "running",
            "batch_size_override": os.environ.get("L3D_BATCH_SIZE")}

    def write_card(**kw):
        card.update(kw, hours_this_session=round((time.monotonic() - t0) / 3600, 3))
        (ctx.run_dir / "run_card.json").write_text(json.dumps(card, indent=2, default=str))
        push(ctx, "run_card.json")

    card["patches"] = clone_and_patch(ctx, spec)
    spec.setup(ctx, released=released)
    link_root(ctx, spec.style)
    write_card()
    results = {}
    if released:
        ckpts = {"released": spec.released_checkpoint(ctx)}
    else:
        pull_state(ctx, spec.sync_files + ["TRAINING_DONE", "epochs.jsonl", "train.log"])
        stop_at = time.time() + (budget - reserve) * 3600 - (time.monotonic() - t0)
        state = train(ctx, spec, stop_at)
        push(ctx, "TRAINING_DONE", "train.log", "epochs.jsonl")
        if state == "paused":
            write_card(status="paused")
            ctx.log("paused before the session limit; run the same notebook again to continue")
            return {"status": "paused"}
        ckpts = spec.checkpoints(ctx)
        card["epochs_log"] = "epochs.jsonl"
    for tag, ckpt in ckpts.items():
        if not ckpt.exists():
            raise FileNotFoundError(f"{tag} checkpoint {ckpt} is missing")
        sets = ("select", "heldout") if tag in ("val", "released") else ("heldout",)
        for s in sets:
            results[f"{tag}_{s}"] = export_and_score(ctx, spec, ckpt, s, tag)
    write_card(status="done", results=results,
               note="'val' = selected on the val / inner-val folder (benchmark); 'native' = the repo's own "
                    "selection, on official test (reproduction only)")
    (ctx.run_dir / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    push(ctx, "DONE", "runner.log")
    return {"status": "done", **results}


def make_ctx(method, split, seed, out_root, work_root, cfg, *, index=None, pilot=False, released=False, drive=None):
    from ..run import PILOT, resolve_split
    if index is None:
        index = build_index(cfg["data"]["roots"])
    ids, _ = resolve_split(split, index, cfg)
    if pilot:
        ids = {k: v[: PILOT[k]] for k, v in ids.items()}
    if released:
        sync_id = f"_g2__{method}__released"
    else:
        sync_id = f"{method}__official__{split}__s{seed}"
    if pilot:
        sync_id = f"_pilot__{sync_id}"
    return Ctx(method=method, split=split, seed=seed, sync_id=sync_id, run_dir=Path(out_root) / sync_id,
               work=Path(work_root), cfg=cfg, index=index, ids=ids, pilot=pilot, drive=drive)
