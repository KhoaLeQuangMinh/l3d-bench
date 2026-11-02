"""Config loading: base.yaml deep-merged with a model config (m1/m2/m3) and optional overrides."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(model_cfg: str | Path | None = None, overrides: dict[str, Any] | None = None,
                base: str | Path = REPO_ROOT / "configs" / "base.yaml") -> dict:
    with open(base) as f:
        cfg = yaml.safe_load(f)
    if model_cfg is not None:
        with open(model_cfg) as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg
