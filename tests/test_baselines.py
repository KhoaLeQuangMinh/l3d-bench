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
