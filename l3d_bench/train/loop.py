"""Training loop shared by M1-M3: AMP, gradient accumulation, best-epoch selection, resume, Drive sync, time budget.

Selection: after every epoch the model predicts the selection set (official val, or the fold's inner-val)
and the epoch with the highest mean A2ONet Dice is kept as best.pth (earliest epoch wins ties). The scored
set is never touched here.

Resume: checkpoint_latest.pth (model, optimizer, scheduler, scaler, every RNG, history, best) is written
after every epoch and pushed to Drive; a new session pulls it and continues at the next epoch. Shuffling and
augmentation depend only on (seed, epoch), so a resumed run sees the same batches as an uninterrupted one.

Time budget: before starting an epoch the loop checks that it can finish it (1.15 x the slowest epoch so
far) within train.max_session_hours, and otherwise stops cleanly with status "paused".
"""
from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler

from ..data.sampler import PoseBalancedSampler
from ..eval.metrics import a2onet_dice_torch
from .checkpoint import load_checkpoint, save_checkpoint

LATEST, BEST, HISTORY, LOG = "checkpoint_latest.pth", "best.pth", "history.csv", "train.log"


def _worker_init(_):
    import cv2
    cv2.setNumThreads(0)


class Trainer:
    def __init__(self, cfg: dict, model, train_ds, select_ds, run_dir: str | Path, *, run_id: str = "",
                 seed: int = 0, drive=None, device: str | None = None, t0: float | None = None,
                 clock=time.monotonic, log=print):
        self.cfg, self.tc = cfg, cfg["train"]
        self.model, self.train_ds, self.select_ds = model, train_ds, select_ds
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.run_id, self.seed, self.drive = run_id, seed, drive
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.amp = bool(self.tc.get("amp")) and self.device.type == "cuda"
        self.clock, self.t0 = clock, (clock() if t0 is None else t0)
        self._log = log
        self.epochs = int(self.tc["epochs"])

        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(model.param_groups(self.tc["lr_backbone"], self.tc["lr_head"],
                                                              self.tc["weight_decay"]))
        if self.tc.get("scheduler", "cosine") != "cosine":
            raise ValueError(f"unknown scheduler {self.tc['scheduler']!r}")
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs,
                                                                    eta_min=self.tc.get("eta_min", 0.0))
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.history: list[dict] = []
        self.best = {"epoch": 0, "select_dice": -1.0}
        self.done_epochs = 0

    # ---- helpers -------------------------------------------------------------------------------
    def log(self, msg: str) -> None:
        line = f"[{self.run_id}] {msg}"
        self._log(line)
        with open(self.dir / LOG, "a") as f:
            f.write(line + "\n")

    def hours(self) -> float:
        return (self.clock() - self.t0) / 3600.0

    def _push(self, *names: str) -> None:
        if self.drive is not None and self.drive.enabled:
            for n in names:
                self.drive.push_file(self.dir / n, self.run_id)

    def _train_loader(self, epoch: int) -> DataLoader:
        self.train_ds.set_epoch(epoch)
        kind = self.cfg.get("sampler", "uniform")
        if kind == "uniform":
            g = torch.Generator().manual_seed(self.seed * 10_007 + epoch)
            sampler = RandomSampler(self.train_ds, generator=g)
        elif kind == "pose_balanced":
            sampler = PoseBalancedSampler(self.train_ds.pose_list(), seed=self.seed)
            sampler.set_epoch(epoch)
        else:
            raise ValueError(f"unknown sampler {kind!r}")
        return DataLoader(self.train_ds, batch_size=self.tc["batch_size"], sampler=sampler, drop_last=False,
                          num_workers=self.tc.get("num_workers", 0), worker_init_fn=_worker_init,
                          pin_memory=self.device.type == "cuda")

    def _eval_loader(self, ds) -> DataLoader:
        return DataLoader(ds, batch_size=self.tc.get("eval_batch_size", 1), shuffle=False,
                          num_workers=self.tc.get("num_workers", 0), worker_init_fn=_worker_init)

    # ---- resume --------------------------------------------------------------------------------
    def try_resume(self) -> bool:
        latest = self.dir / LATEST
        if not latest.exists() and self.drive is not None and self.drive.enabled:
            if self.drive.exists(self.run_id, LATEST):
                self.log("pulling checkpoint from Drive")
                for n in (LATEST, BEST, HISTORY, LOG):
                    self.drive.pull_file(self.run_id, n, self.dir)
        if not latest.exists():
            return False
        info = load_checkpoint(latest, model=self.model, optimizer=self.optimizer, scheduler=self.scheduler,
                               scaler=self.scaler, map_location=self.device)
        self.done_epochs = info["epoch"]
        self.best = info["best"] or self.best
        self.history = info["extra"].get("history", [])
        if self.best["epoch"] > 0 and not (self.dir / BEST).exists():
            raise RuntimeError(f"{LATEST} says best epoch {self.best['epoch']} but {BEST} is missing")
        self.log(f"resumed after epoch {self.done_epochs}; best so far epoch {self.best['epoch']} "
                 f"(select Dice {self.best['select_dice']:.4f})")
        return True

    # ---- one epoch -----------------------------------------------------------------------------
    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        accum = int(self.tc.get("accumulation", 1))
        loader = self._train_loader(epoch)
        total, n, skipped = 0.0, 0, 0
        self.optimizer.zero_grad(set_to_none=True)
        steps = len(loader)
        for i, batch in enumerate(loader):
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.amp):
                loss = self.model(images, labels).loss
            if not torch.isfinite(loss):
                skipped += 1
                continue
            self.scaler.scale(loss / accum).backward()
            total, n = total + float(loss.detach()), n + 1
            if (i + 1) % accum == 0 or (i + 1) == steps:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.tc.get("clip_grad_norm", 1e9))
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
        if skipped:
            self.log(f"epoch {epoch}: skipped {skipped} batches with non-finite loss")
        if n == 0:
            raise RuntimeError(f"epoch {epoch}: every batch had a non-finite loss")
        return total / n

    @torch.no_grad()
    def select_dice(self) -> float:
        self.model.eval()
        scores = []
        for batch in self._eval_loader(self.select_ds):
            images = batch["image"].to(self.device)
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.amp):
                pred = self.model.predict(images)
            scores.append(a2onet_dice_torch(pred.long(), batch["label"].to(self.device)).cpu())
        return float(torch.cat(scores).mean())

    def _write_history(self) -> None:
        with open(self.dir / HISTORY, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.history[0]))
            w.writeheader()
            w.writerows(self.history)

    # ---- main ----------------------------------------------------------------------------------
    def fit(self) -> dict:
        self.try_resume()
        budget = float(self.tc["max_session_hours"])
        slowest = max((h["epoch_minutes"] for h in self.history), default=0.0) / 60.0
        for epoch in range(self.done_epochs + 1, self.epochs + 1):
            if slowest and self.hours() + 1.15 * slowest > budget:
                self.log(f"time budget: {self.hours():.2f} h used, an epoch takes ~{slowest:.2f} h; "
                         f"pausing before epoch {epoch}. Re-run to continue.")
                return {"status": "paused", "done_epochs": self.done_epochs, "best": self.best}
            t = self.clock()
            loss = self.train_epoch(epoch)
            dice = self.select_dice()
            self.scheduler.step()
            minutes = (self.clock() - t) / 60.0
            slowest = max(slowest, minutes / 60.0)
            improved = dice > self.best["select_dice"] + 1e-9
            if improved:
                self.best = {"epoch": epoch, "select_dice": dice}
                tmp = self.dir / (BEST + ".tmp")
                torch.save({"model": self.model.state_dict(), "epoch": epoch, "select_dice": dice}, tmp)
                tmp.replace(self.dir / BEST)
            lrs = {g.get("name", str(k)): g["lr"] for k, g in enumerate(self.optimizer.param_groups)}
            self.history.append({"epoch": epoch, "train_loss": round(loss, 5), "select_dice": round(dice, 5),
                                 "best_epoch": self.best["epoch"], "lr_backbone": lrs.get("backbone"),
                                 "lr_head": lrs.get("head"), "epoch_minutes": round(minutes, 2),
                                 "hours_used": round(self.hours(), 3)})
            self.done_epochs = epoch
            self._write_history()
            save_checkpoint(self.dir / LATEST, model=self.model, optimizer=self.optimizer, epoch=epoch,
                            scheduler=self.scheduler, scaler=self.scaler, best=self.best,
                            extra={"history": self.history, "run_id": self.run_id})
            self.log(f"epoch {epoch}/{self.epochs}  loss {loss:.4f}  select Dice {dice:.4f}"
                     f"{'  *best*' if improved else ''}  ({minutes:.1f} min, {self.hours():.2f} h used)")
            if improved:
                self._push(BEST)  # before LATEST, so Drive never has a LATEST pointing at a missing best
            if epoch % int(self.tc.get("checkpoint_every", 1)) == 0 or epoch == self.epochs:
                self._push(LATEST)
            self._push(HISTORY, LOG)
            if math.isnan(dice):
                raise RuntimeError("selection Dice is NaN")
        return {"status": "trained", "done_epochs": self.done_epochs, "best": self.best}

    def load_best(self) -> dict:
        ckpt = torch.load(self.dir / BEST, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        return {"epoch": ckpt["epoch"], "select_dice": ckpt["select_dice"]}
