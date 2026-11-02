"""Save / resume full training state, including every RNG, so a resumed run continues exactly."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def _rng_state() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def save_checkpoint(path: str | Path, *, model, optimizer, epoch: int, scheduler=None, scaler=None,
                    best: dict | None = None, extra: dict | None = None) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "best": best or {},
        "extra": extra or {},
        "rng": _rng_state(),
    }, tmp)
    tmp.replace(path)  # atomic: a session cut mid-write never leaves a broken checkpoint


def load_checkpoint(path: str | Path, *, model, optimizer=None, scheduler=None, scaler=None,
                    map_location="cpu") -> dict:
    """Restores everything in place; returns {'epoch', 'best', 'extra'}. Next epoch = epoch + 1."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    _set_rng_state(ckpt["rng"])
    return {"epoch": ckpt["epoch"], "best": ckpt.get("best", {}), "extra": ckpt.get("extra", {})}
