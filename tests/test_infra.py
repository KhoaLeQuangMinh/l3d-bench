import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from l3d_bench.data.dataset import L3DDataset
from l3d_bench.data.folds import make_folds, official_split
from l3d_bench.data.roots import build_roots
from l3d_bench.runs.manifest import SPLITS, make_manifest, parse_run_id, run_id
from l3d_bench.train.checkpoint import load_checkpoint, save_checkpoint
from l3d_bench.train.drive import DriveSync, format_rclone_config


# ---- dataset ------------------------------------------------------------------------------------
def test_dataset_shapes(index, cfg, fake_l3d):
    ids = list(index["frame_id"][:4])
    ds = L3DDataset(index, ids, cfg, train=False, poses=fake_l3d["poses"])
    item = ds[0]
    assert item["image"].shape == (3, 64, 64) and item["label"].shape == (64, 64)
    assert item["label"].dtype == torch.int64 and set(item["label"].unique().tolist()) <= {0, 1, 2, 3}
    assert item["junctions"].shape == (4, 2) and item["pose"] in (0, 1)


def test_hflip_moves_image_label_and_junctions_together(index, cfg):
    cfg["augment"].update({"hflip_p": 1.0, "scale_range": [1.0, 1.0], "translate_frac": 0.0, "color_jitter": 0})
    fid = index["frame_id"].iloc[0]
    plain = L3DDataset(index, [fid], cfg, train=False)[0]
    flipped = L3DDataset(index, [fid], cfg, train=True)[0]
    assert torch.equal(flipped["label"], plain["label"].flip(-1))
    assert torch.allclose(flipped["image"], plain["image"].flip(-1))
    assert torch.allclose(flipped["junctions"][2, 0], 1 - plain["junctions"][3, 0], atol=1e-5)


def test_vertical_flip_refused(index, cfg):
    cfg["augment"]["vertical_flip"] = True
    with pytest.raises(ValueError):
        L3DDataset(index, list(index["frame_id"][:1]), cfg)


# ---- baseline roots -----------------------------------------------------------------------------
@pytest.mark.parametrize("style,names", [("upper", ("Train", "Val", "Test")), ("lower", ("train", "val", "test"))])
def test_build_roots(index, fake_l3d, tmp_path, style, names):
    spec = make_folds(index, fake_l3d["poses"], k=4, n_inner=2)["folds"][0]
    out = build_roots(index, spec, tmp_path / style, style=style)
    tr, ev = Path(out["train_root"]), Path(out["eval_root"])
    assert sorted(os.listdir(tr)) == sorted(names)                 # TopoNet reads any other folder as val
    sel = sorted(p.name for p in (tr / names[1] / "images").iterdir())
    assert sel == sorted(p.name for p in (tr / names[2] / "images").iterdir())  # Test = selection set
    held = {p.name.rsplit("_", 1)[0] for p in (ev / names[2] / "images").iterdir()}
    assert held == set(spec["heldout"])
    assert (tr / names[0] / "depth_AdelaiDepth").exists()
    assert out["counts"]["heldout"] == len(list((ev / names[2] / "labels").iterdir()))


def test_official_roots(index, tmp_path):
    out = build_roots(index, official_split(index), tmp_path, style="upper")
    assert out["counts"] == {"train": 25, "select": 8, "heldout": 6}


# ---- manifest -----------------------------------------------------------------------------------
def test_manifest_42_per_seed():
    m = make_manifest(seeds=(0,), owners=[f"u{i}" for i in range(10)])
    assert len(m) == 42 and m["run_id"].is_unique
    assert set(m["split"]) == set(SPLITS)
    assert m.groupby("split")["owner"].nunique().max() == 1
    assert parse_run_id(run_id("m2", "f3", 1)) == {"method": "m2", "variant": "junction", "split": "f3", "seed": 1}
    with pytest.raises(ValueError):
        parse_run_id("m2__plain__f3__s1")


# ---- checkpoint ---------------------------------------------------------------------------------
def _train(model, opt, steps, seed_data):
    g = torch.Generator().manual_seed(seed_data)
    losses = []
    for _ in range(steps):
        x = torch.randn(8, 4, generator=g)
        noise = torch.rand(1).item()  # consumes the global RNG, which resume must restore
        loss = ((model(x) - noise) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def test_resume_equals_straight_run(tmp_path):
    torch.manual_seed(0)
    m1 = torch.nn.Linear(4, 1)
    o1 = torch.optim.Adam(m1.parameters(), lr=0.1)
    straight = _train(m1, o1, 4, 1)

    torch.manual_seed(0)
    m2 = torch.nn.Linear(4, 1)
    o2 = torch.optim.Adam(m2.parameters(), lr=0.1)
    first = _train(m2, o2, 2, 1)
    save_checkpoint(tmp_path / "ck.pth", model=m2, optimizer=o2, epoch=1)
    torch.manual_seed(123)  # a new session starts with different global RNG state
    m3 = torch.nn.Linear(4, 1)
    o3 = torch.optim.Adam(m3.parameters(), lr=0.1)
    info = load_checkpoint(tmp_path / "ck.pth", model=m3, optimizer=o3)
    g = torch.Generator().manual_seed(1)
    for _ in range(2):  # advance the data generator past the first two steps
        torch.randn(8, 4, generator=g)
    rest = []
    for _ in range(2):
        x = torch.randn(8, 4, generator=g)
        noise = torch.rand(1).item()
        loss = ((m3(x) - noise) ** 2).mean()
        o3.zero_grad(); loss.backward(); o3.step()
        rest.append(loss.item())
    assert info["epoch"] == 1
    assert np.allclose(first + rest, straight)


# ---- drive --------------------------------------------------------------------------------------
def test_format_flattened_secret():
    raw = 'type = drive scope = drive token = {"a":1} client_id = abc'
    text = format_rclone_config(raw)
    assert text.splitlines()[0] == "[gdrive]"
    assert any(l.startswith("token") for l in text.splitlines())
    assert text.count("\n") >= 4


def test_retry_then_success():
    calls = []

    def runner(cmd, **kw):
        calls.append(cmd)
        return SimpleNamespace(returncode=0 if len(calls) >= 3 else 1, stderr="429 rate limit", stdout="")
    sync = DriveSync(enabled=True, runner=runner, sleep=lambda s: None)
    assert sync._exec(["copy", "a", "b"])
    assert len(calls) == 3
    assert sync.remote_path("m1__plain__f0__s0", "x.pth") == "gdrive:l3d-bench/results/m1__plain__f0__s0/x.pth"


def test_disabled_is_noop(tmp_path):
    sync = DriveSync(enabled=False, runner=lambda *a, **k: 1 / 0)
    f = tmp_path / "f.txt"
    f.write_text("x")
    assert sync.push_file(f, "r") is False and sync.exists("r", "f.txt") is False


def test_config_from_secret(tmp_path):
    target = tmp_path / "home" / "rclone.conf"
    flat = 'type = drive scope = drive token = {"access_token":"x"}'   # Kaggle secrets may arrive flattened
    sync = DriveSync.from_kaggle(get_secret=lambda name: flat if name == "RCLONE_CONFIG_GDRIVE" else None,
                                 targets=(str(target),))
    assert sync.enabled and target.read_text().startswith("[gdrive]\n")
    assert (target.stat().st_mode & 0o777) == 0o600


def test_missing_secret_disables(tmp_path, monkeypatch):
    monkeypatch.setattr(DriveSync, "_detect", staticmethod(lambda: False))
    sync = DriveSync.from_kaggle(get_secret=lambda name: "", targets=(str(tmp_path / "c.conf"),))
    assert not sync.enabled and not (tmp_path / "c.conf").exists()
