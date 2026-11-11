"""Baseline runner pieces that need no repo clone: data roots, run ids, patches present."""
import os
from pathlib import Path

from l3d_bench.baselines.runner import link_root, make_ctx
from l3d_bench.baselines.specs import SPECS
from l3d_bench.config import REPO_ROOT, load_config


def _names(d):
    return sorted(p.name.rsplit("_", 1)[0] for p in (d / "images").iterdir())


def test_official_root_is_native(index, tmp_path):
    ctx = make_ctx("toponet", "official", 0, tmp_path / "runs", tmp_path / "w", load_config(), index=index)
    root = link_root(ctx, "lower")
    assert sorted(os.listdir(root)) == ["test", "train", "val"]          # anything else TopoNet reads as val
    assert set(_names(root / "test")) == {"Patient_41", "Patient_21"}
    assert set(_names(root / "val")) == {"Patient_40", "Patient_18"}
    img = next((root / "train" / "images").iterdir())
    assert img.is_symlink() and (root / "train" / "labels" / (img.stem + ".json")).exists()
    assert (root / "train" / "depth_AdelaiDepth" / (img.stem + ".png")).exists()
    assert ctx.sync_id == "toponet__official__official__s0"


def test_fold_root_selects_on_inner_val(index, fake_l3d, tmp_path):
    from l3d_bench.data.folds import make_folds, save_folds
    path = tmp_path / "folds.json"
    spec = make_folds(index, fake_l3d["poses"], k=4, n_inner=2)
    save_folds(spec, path)
    cfg = load_config(overrides={"data": {"folds_file": str(path)}})
    ctx = make_ctx("d2gpland", "f2", 0, tmp_path / "runs", tmp_path / "w", cfg, index=index, pilot=True)
    root = link_root(ctx, "upper")
    assert sorted(os.listdir(root)) == ["Test", "Train", "Val"]
    assert _names(root / "Val") == _names(root / "Test")                  # selection set in both
    assert set(_names(root / "Val")) <= set(spec["folds"][2]["inner_val"])
    assert not set(_names(root / "Train")) & set(spec["folds"][2]["heldout"])
    assert ctx.sync_id == "_pilot__d2gpland__official__f2__s0"


def test_released_run_id(index, tmp_path):
    ctx = make_ctx("toponet", "official", 0, tmp_path, tmp_path, load_config(), index=index, released=True)
    assert ctx.sync_id == "_g2__toponet__released"


def test_every_spec_has_patches_and_export():
    for name, spec in SPECS.items():
        assert list((REPO_ROOT / "baselines" / name / "patches").glob("*.diff")), name
        assert Path(spec.export_script()).exists(), name
        assert len(spec.commit) == 40


def test_bcrnet_frames_roundtrip_and_sam_input(index, tmp_path):
    import numpy as np
    from l3d_bench.baselines.bcrnet_prep import read_frames, sam_input, write_frames
    ids = list(index["frame_id"][:3])
    rows = read_frames(write_frames(tmp_path / "f.tsv", index, ids))
    assert [r[0] for r in rows] == ids and all(Path(r[1]).exists() and Path(r[2]).exists() for r in rows)
    x = sam_input(np.zeros((108, 192, 3), np.uint8))
    assert x.shape == (1024, 1024, 3)                                      # squashed, like BCRNet's input


def test_bcrnet_selection_picks_best_then_earliest(index, tmp_path, monkeypatch):
    import json
    from l3d_bench.baselines import specs
    ctx = make_ctx("bcrnet", "official", 0, tmp_path / "runs", tmp_path / "w", load_config(), index=index)
    ctx.run_dir.mkdir(parents=True)
    for e in (10, 20, 30):
        (ctx.run_dir / f"epoch-{e}.pt").write_text("x")
    (ctx.run_dir / "selection.json").write_text(json.dumps({
        "epoch-10.pt": {"epoch": 10, "select_dice": 0.5}, "epoch-20.pt": {"epoch": 20, "select_dice": 0.7},
        "epoch-30.pt": {"epoch": 30, "select_dice": 0.7}}))
    monkeypatch.setattr(specs, "sh", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no re-scoring")))
    ck = specs.BCRNet().checkpoints(ctx)
    assert ck["val"].name == "epoch-20.pt" and ck["native"].name == "epoch-30.pt"


def test_a2onet_scores_each_checkpoint_with_its_selection_alpha(index, tmp_path):
    import json
    from l3d_bench.baselines.specs import A2ONet
    ctx = make_ctx("a2onet", "official", 0, tmp_path / "runs", tmp_path / "w", load_config(), index=index)
    ctx.run_dir.mkdir(parents=True)
    r = lambda d: {"dice": d, "iou": 0, "assd": 0}
    ctx.run_dir.joinpath("metrics.json").write_text(json.dumps([
        {"epoch": 5, "val": r(0.60), "test": r(0.61)}, {"epoch": 10, "val": r(0.66), "test": r(0.60)},
        {"epoch": 15, "val": r(0.64), "test": r(0.65)}, {"epoch": 20, "val": r(0.66), "test": r(0.62)}]))
    spec = A2ONet()
    ck = spec.checkpoints(ctx)
    assert set(ck) == {"val", "val_testpy", "native"}
    assert spec.alphas == {"val": 0.0, "val_testpy": 1.0, "native": 0.5}     # val: epoch 10 (first max); test: 15
    ctx.current_tag = "native"
    spec.sam = spec.depth = tmp_path
    cmd = spec.export_cmd(ctx, ck["native"], tmp_path / "l.txt", tmp_path / "o", tmp_path / "n.json")
    assert cmd[cmd.index("--alpha") + 1] == "0.5"
