"""M1 model, training loop, resume, Drive round-trip and the run entry point, on a tiny random Mask2Former
(no download) and the synthetic L3D from conftest.py."""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from l3d_bench.config import REPO_ROOT, load_config
from l3d_bench.data.dataset import L3DDataset
from l3d_bench.data.folds import make_folds, save_folds
from l3d_bench.eval.metrics import a2onet_dice_torch, a2onet_metrics
from l3d_bench.models.build import build_model
from l3d_bench.models.mask2former import label_to_targets
from l3d_bench.run import parse_overrides, resolve_split, run
from l3d_bench.train.loop import Trainer
from l3d_bench.train.seed import seed_everything

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from overfit_check import overfit


def tiny_cfg(method="m1", **train):
    t = {"epochs": 3, "num_workers": 0, "batch_size": 2, "accumulation": 2}
    t.update(train)
    return load_config(REPO_ROOT / "configs" / f"{method}.yaml",
                       {"data": {"input_size": 64}, "metric": {"score_size": 64},
                        "model": {"backbone": "tiny-random"}, "train": t})


class FakeDrive:
    """Stands in for rclone: a folder per run id."""
    enabled = True

    def __init__(self, root):
        self.root = Path(root)

    def push_file(self, local, run_id):
        dst = self.root / run_id
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy(local, dst / Path(local).name)
        return True

    def exists(self, run_id, name):
        return (self.root / run_id / name).exists()

    def pull_file(self, run_id, name, local_dir):
        src = self.root / run_id / name
        if src.exists():
            shutil.copy(src, Path(local_dir) / name)
        return src.exists()


# ---- model --------------------------------------------------------------------------------------
def test_targets_include_background():
    lab = torch.zeros(2, 8, 8, dtype=torch.long)
    lab[0, 2:4] = 1
    lab[0, 6] = 3
    masks, classes = label_to_targets(lab)
    assert classes[0].tolist() == [0, 1, 3] and classes[1].tolist() == [0]
    assert masks[0].shape == (3, 8, 8) and torch.equal(masks[0].sum(0), torch.ones(8, 8))


def test_model_forward_predict_and_groups():
    model = build_model(tiny_cfg())
    x = torch.rand(2, 3, 64, 64)
    y = torch.zeros(2, 64, 64, dtype=torch.long)
    y[:, 20:24] = 2
    out = model(x, y)
    assert torch.isfinite(out.loss)
    assert out.class_queries_logits.shape[-1] == 5                     # 4 classes + no-object
    scores = model.semantic_scores(out, (64, 64))
    assert scores.shape == (2, 4, 64, 64)
    pred = model.eval().predict(x)
    assert pred.shape == (2, 64, 64) and pred.dtype == torch.uint8 and int(pred.max()) <= 3
    groups = model.param_groups(1e-5, 1e-4, 1e-4)
    ids = [{id(p) for p in g["params"]} for g in groups]
    assert ids[0] and ids[1] and not (ids[0] & ids[1])
    assert len(ids[0]) + len(ids[1]) == len(list(model.parameters()))


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        out.update(_flat(v, f"{prefix}{k}.") if isinstance(v, dict) else {f"{prefix}{k}": v})
    return out


def test_configs_differ_only_in_their_switch():
    m1, m2, m3 = (_flat(load_config(REPO_ROOT / "configs" / f"{m}.yaml")) for m in ("m1", "m2", "m3"))
    assert {k for k in m1 if m1[k] != m2[k]} == {"model.junction"}
    assert {k for k in m1 if m1[k] != m3[k]} == {"sampler"}


def test_m2_head_steers_queries_and_trains():
    torch.manual_seed(0)
    m2 = build_model(tiny_cfg("m2"))
    assert abs(float(m2.steering.gate) - 0.1) < 1e-7
    assert not any("anchor_head" in n or "steering" in n for n, _ in build_model(tiny_cfg()).named_parameters())
    x = torch.rand(2, 3, 64, 64)
    y = torch.zeros(2, 64, 64, dtype=torch.long)
    y[:, 20:24] = 1
    j = torch.rand(2, 4, 2)
    v = torch.tensor([[1.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    out = m2(x, y, j, v)
    assert torch.isfinite(out.loss) and out.junction_coords.shape == (2, 4, 2)
    assert torch.allclose(out.loss, out.mask_loss + 5 * out.coord_loss + 1 * out.vis_loss, atol=1e-4)
    out.loss.backward()
    assert m2.steering.gate.grad is not None and m2.anchor_head.queries.grad.abs().sum() > 0
    m2.eval()
    with torch.no_grad():
        a = m2(x).masks_queries_logits
        m2.steering.gate.fill_(0.0)
        b = m2(x).masks_queries_logits
    assert not torch.allclose(a, b)                         # the steered queries really reach the decoder
    with pytest.raises(ValueError):
        m2.train()(x, y)                                    # training without junction targets is an error


def test_gpu_dice_equals_protocol():
    rng = np.random.default_rng(0)
    pred = rng.integers(0, 4, (5, 32, 32))
    gt = rng.integers(0, 4, (5, 32, 32))
    gt[3] = 0                                                              # empty GT frame
    pred[4] = 0                                                            # empty prediction
    fast = a2onet_dice_torch(torch.from_numpy(pred), torch.from_numpy(gt)).numpy()
    ref = [a2onet_metrics(p, g)[0] for p, g in zip(pred, gt)]
    assert np.allclose(fast, ref, atol=1e-12)


# ---- loop ---------------------------------------------------------------------------------------
class SessionTrainer(Trainer):
    """'Hours used' = epochs trained in this session, so a budget of 1.5 pauses after 2 epochs."""
    session_epochs = 0

    def train_epoch(self, epoch):
        self.session_epochs += 1
        return super().train_epoch(epoch)

    def hours(self):
        return float(self.session_epochs)


def _trainer(index, poses, cfg, run_dir, budget):
    seed_everything(0)
    cfg = {**cfg, "train": {**cfg["train"], "max_session_hours": budget}}
    tr_ids = list(index.loc[index["split"] == "train", "frame_id"])
    va_ids = list(index.loc[index["split"] == "val", "frame_id"])
    model = build_model(cfg)
    return SessionTrainer(cfg, model, L3DDataset(index, tr_ids, cfg, train=True, poses=poses),
                          L3DDataset(index, va_ids, cfg, poses=poses), run_dir, run_id="t", seed=0,
                          device="cpu", log=lambda *_: None)


def test_resume_matches_straight_run(index, fake_l3d, tmp_path):
    cfg = tiny_cfg(epochs=3)
    straight = _trainer(index, fake_l3d["poses"], cfg, tmp_path / "a", budget=100).fit()
    assert straight["status"] == "trained"
    first = _trainer(index, fake_l3d["poses"], cfg, tmp_path / "b", budget=1.5).fit()
    assert first == {"status": "paused", "done_epochs": 2, "best": first["best"]}
    second = _trainer(index, fake_l3d["poses"], cfg, tmp_path / "b", budget=100)
    assert second.fit()["status"] == "trained"
    a = (tmp_path / "a" / "history.csv").read_text().splitlines()
    b = (tmp_path / "b" / "history.csv").read_text().splitlines()
    cols = a[0].split(",")
    keep = [cols.index(c) for c in ("epoch", "train_loss", "select_dice", "best_epoch", "lr_head")]
    assert [[r.split(",")[i] for i in keep] for r in a] == [[r.split(",")[i] for i in keep] for r in b]


def test_pose_balanced_sampler_runs(index, fake_l3d, tmp_path):
    cfg = tiny_cfg("m3", epochs=1)
    assert _trainer(index, fake_l3d["poses"], cfg, tmp_path, budget=100).fit()["status"] == "trained"


# ---- run entry point ----------------------------------------------------------------------------
def test_run_official_end_to_end_with_drive_resume(index, fake_l3d, tmp_path):
    drive = FakeDrive(tmp_path / "drive")
    rid = "m1__plain__official__s0"
    # session 1: with no epoch timed yet one epoch runs, then the tiny budget pauses the run
    st = run(rid, tiny_cfg(epochs=3, max_session_hours=1e-9), tmp_path / "kaggle_a", drive=drive,
             index=index, poses=fake_l3d["poses"], device="cpu")
    assert st["status"] == "paused" and st["done_epochs"] == 1
    # session 2 on another machine: pulls the checkpoint from Drive, finishes, exports, scores
    st = run(rid, tiny_cfg(epochs=3), tmp_path / "kaggle_b", drive=drive, index=index,
             poses=fake_l3d["poses"], device="cpu")
    assert st["status"] == "done"
    d = tmp_path / "drive" / rid
    for f in ("run_card.json", "history.csv", "best.pth", "checkpoint_latest.pth", "select_metrics.json",
              "heldout_metrics.json", "heldout_frames.csv", "pred_heldout.zip", "pred_select.zip", "DONE"):
        assert (d / f).exists(), f
    card = json.loads((d / "run_card.json").read_text())
    assert card["status"] == "done" and card["done_epochs"] == 3
    assert card["scored_set"] == "official test" and card["heldout_frames"] == 6
    assert len((d / "history.csv").read_text().splitlines()) == 4
    # selection Dice from training equals the protocol scorer on the same frames (input = score size)
    assert abs(card["best_select_dice_training"] - card["select"]["dice"]) < 1e-9
    frames = (d / "heldout_frames.csv").read_text()
    assert "Patient_41" in frames and "Patient_40" not in frames
    # a fourth call does nothing
    assert run(rid, tiny_cfg(), tmp_path / "kaggle_d", drive=drive, index=index)["status"] == "already_done"


def test_pilot_uses_separate_folder(index, tmp_path):
    st = run("m1__plain__official__s1", tiny_cfg(epochs=5), tmp_path, index=index, pilot=True, device="cpu")
    assert st["status"] == "done"
    card = json.loads((tmp_path / "_pilot__m1__plain__official__s1" / "run_card.json").read_text())
    assert card["pilot"] and card["done_epochs"] == 2 and card["train_frames"] == 8


def test_fold_split_resolution(index, fake_l3d, tmp_path):
    spec = make_folds(index, fake_l3d["poses"], k=4, n_inner=2)
    path = tmp_path / "folds.json"
    save_folds(spec, path)
    cfg = tiny_cfg()
    cfg["data"]["folds_file"] = str(path)                  # absolute path wins over REPO_ROOT
    ids, info = resolve_split("f1", index, cfg)
    assert info["heldout_patients"] == spec["folds"][1]["heldout"]
    assert not set(ids["train"]) & set(ids["select"]) and not set(ids["train"]) & set(ids["heldout"])
    test_frames = set(index.loc[index["split"] == "test", "frame_id"])
    assert not test_frames & (set(ids["train"]) | set(ids["select"]) | set(ids["heldout"]))
    assert len(info["folds_sha256"]) == 64


def test_baseline_ids_rejected(index, tmp_path):
    with pytest.raises(ValueError):
        run("toponet__official__official__s0", tiny_cfg(), tmp_path, index=index)


def test_parse_overrides():
    assert parse_overrides(["train.epochs=40", "model.backbone=x", "train.amp=false"]) == \
        {"train": {"epochs": 40, "amp": False}, "model": {"backbone": "x"}}


def test_overfit_loss_falls(index):
    cfg = tiny_cfg(lr_head=1e-3, lr_backbone=1e-3)
    hist = overfit(cfg, index, n_frames=4, steps=60, device="cpu", log_every=20, log=lambda *_: None)
    assert hist[-1]["loss"] < hist[0]["loss"]


def test_m2_run_end_to_end(index, tmp_path):
    st = run("m2__junction__official__s0", tiny_cfg("m2", epochs=2), tmp_path, index=index, device="cpu")
    assert st["status"] == "done"


def test_m3_pilot_has_both_poses(index, fake_l3d, tmp_path):
    st = run("m3__posebal__official__s0", tiny_cfg("m3", epochs=2), tmp_path, index=index,
             poses=fake_l3d["poses"], pilot=True, device="cpu")
    assert st["status"] == "done"


def test_base_config_matches_freeze():
    import yaml
    frozen = yaml.safe_load((REPO_ROOT / "configs" / "frozen.yaml").read_text())
    assert load_config() == frozen, "configs/base.yaml changed after the study-v1 freeze"


def test_manifest_ids_valid():
    from l3d_bench.runs.manifest import load_manifest
    m = load_manifest(REPO_ROOT / "runs" / "manifest.csv")
    folds = m[m["split"] != "official"]
    ours = folds["method"].isin(["m1", "m2", "m3"])
    assert set(folds.loc[ours, "commit"]) == {"study-v1"}
    assert set(folds.loc[~ours, "commit"]) == {"study-v1.1"}   # study-v1 + BCRNet runner + A2ONet scoring fix
