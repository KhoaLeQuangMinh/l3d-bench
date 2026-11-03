"""config -> model. One model class with switches, so a gain can only come from the switch being tested."""
from __future__ import annotations

from .mask2former import Mask2FormerSegmenter


def build_model(cfg: dict):
    m = cfg["model"]
    if m.get("junction"):
        from .junction import Mask2FormerJunction
        return Mask2FormerJunction(m["backbone"], gate_init=m.get("junction_gate_init", 0.1),
                                   w_coord=m.get("junction_loss_coord", 5.0), w_vis=m.get("junction_loss_vis", 1.0))
    return Mask2FormerSegmenter(m["backbone"])
