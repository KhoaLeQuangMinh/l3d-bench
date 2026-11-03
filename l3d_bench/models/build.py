"""config -> model. One model class with switches, so a gain can only come from the switch being tested."""
from __future__ import annotations

from .mask2former import Mask2FormerSegmenter


def build_model(cfg: dict):
    m = cfg["model"]
    if m.get("junction"):
        raise NotImplementedError("M2 (junction steering) is Phase 4; models/junction.py is not written yet")
    return Mask2FormerSegmenter(m["backbone"])
