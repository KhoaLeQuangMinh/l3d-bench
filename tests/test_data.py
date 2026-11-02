import json

import numpy as np
import pytest

from l3d_bench.data.annotations import Annotation, Shape, class_of, label_inventory, load_annotation
from l3d_bench.data.folds import check_folds, make_folds, official_split, save_folds, split_patients
from l3d_bench.data.index import build_index, parse_frame_id, summary
from l3d_bench.data.junctions import derive_junctions, hflip_junctions
from l3d_bench.data.masks import gt_for_scoring, line_thickness, overlap_fraction, render_label_map
from l3d_bench.data.pose import cohen_kappa, load_pose_labels, raised_share
from l3d_bench.data.sampler import PoseBalancedSampler
from tests import reference_a2onet as ref


# ---- annotations ---------------------------------------------------------------------------------
def test_class_mapping_by_first_letter():
    assert class_of("ridge") == 1 and class_of("Rigde") == 1
    assert class_of("s") == 2 and class_of("silhouette") == 2
    assert class_of("l") == 3 and class_of("Ligament") == 3
    assert class_of("falciform") == 0  # A2ONet draws unknown labels with value 0


def test_unknown_label_strict(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"shapes": [{"label": "falciform", "points": [[0, 0], [5, 5]]}],
                             "imageHeight": 10, "imageWidth": 10}))
    assert load_annotation(p).shapes[0].class_id == 0
    with pytest.raises(ValueError):
        load_annotation(p, strict=True)


def test_inventory(index):
    inv = label_inventory(index["label_path"])
    assert set(inv) == {"r", "s", "l"}


# ---- masks: identical to A2ONet ------------------------------------------------------------------
def test_thickness_rule():
    assert line_thickness(1080) == 35 and line_thickness(2160) == 70 and line_thickness(108) == 3


def test_render_matches_a2onet(index):
    for path in index["label_path"]:
        ours = render_label_map(load_annotation(path))
        assert np.array_equal(ours, ref.render_json(path))
        assert np.array_equal(gt_for_scoring(load_annotation(path), size=64), ref.load_mask_label(path, 64))


def test_last_shape_wins():
    ann = Annotation(20, 20, [Shape("r", 1, np.array([[0, 10], [19, 10]], np.float32)),
                              Shape("s", 2, np.array([[10, 0], [10, 19]], np.float32))])
    lab = render_label_map(ann, width_at_1080=60)
    assert lab[10, 10] == 2


def test_overlap_fraction_counts_crossings():
    ann = Annotation(20, 20, [Shape("r", 1, np.array([[0, 10], [19, 10]], np.float32)),
                              Shape("s", 2, np.array([[10, 0], [10, 19]], np.float32))])
    both, any_ = overlap_fraction(ann, width_at_1080=60)
    assert 0 < both < any_


# ---- index --------------------------------------------------------------------------------------
def test_index_counts(index):
    s = summary(index)
    assert s["frames"] == {"train": 25, "val": 8, "test": 6}
    assert s["n_patients"] == 10
    assert len(s["non_1080p_frames"]) == 39  # every synthetic frame is smaller than 1080p
    assert s["top_patient_share"]["val"] == 0.75
    assert (index["depth_path"] != "").all()


def test_parse_frame_id():
    assert parse_frame_id("Patient_21_0019200") == ("Patient_21", 19200)
    with pytest.raises(ValueError):
        parse_frame_id("frame_001")


def test_index_detects_leak(fake_l3d, tmp_path):
    import shutil
    leak = tmp_path / "Val"
    shutil.copytree(fake_l3d["roots"]["val"], leak)
    src = fake_l3d["roots"]["train"]
    for sub, ext in (("images", ".jpg"), ("labels", ".json")):
        shutil.copy(src / sub / f"Patient_1_0000060{ext}", leak / sub / f"Patient_1_0009999{ext}")
    with pytest.raises(ValueError, match="more than one split"):
        build_index({"train": src, "val": leak})


# ---- folds --------------------------------------------------------------------------------------
def test_folds_invariants_and_determinism(index, fake_l3d):
    spec = make_folds(index, fake_l3d["poses"], k=4, n_inner=2, seed=0)
    check_folds(spec, index)
    assert make_folds(index, fake_l3d["poses"], k=4, n_inner=2, seed=0) == spec
    for f in spec["folds"]:
        assert len(f["inner_val"]) == 2
        assert not set(f["heldout"]) & {"Patient_41", "Patient_21"}


def test_folds_without_pose_balances_frames(index):
    spec = make_folds(index, None, k=4, n_inner=2, seed=1)
    sizes = [f["frames"]["heldout"] for f in spec["folds"]]
    assert max(sizes) - min(sizes) <= 6


def test_missing_pose_label_is_an_error(index, fake_l3d):
    poses = dict(fake_l3d["poses"])
    poses.pop(next(iter(poses)))
    with pytest.raises(ValueError, match="no pose label"):
        make_folds(index, poses, k=4, n_inner=2)


def test_official_split_and_save(index, tmp_path, fake_l3d):
    off = official_split(index)
    assert off["heldout"] == ["Patient_21", "Patient_41"]
    spec = make_folds(index, fake_l3d["poses"], k=4, n_inner=2)
    digest = save_folds(spec, tmp_path / "folds.json")
    assert len(digest) == 64
    assert set(split_patients(spec, "f2")) == {"train", "inner_val", "heldout"}


# ---- pose ---------------------------------------------------------------------------------------
def test_pose_loader_and_kappa(tmp_path):
    p = tmp_path / "pose.csv"
    p.write_text("frame_id,pose,labeler\na,canonical,x\nb,raised,y\n")
    assert load_pose_labels(p) == {"a": "canonical", "b": "raised"}
    p.write_text("frame_id,pose,labeler\na,lifted,x\n")
    with pytest.raises(ValueError):
        load_pose_labels(p)
    assert cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    assert abs(cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"])) < 1e-9
    assert raised_share(["a", "b"], {"a": "raised", "b": "canonical"}) == 0.5


# ---- junctions ----------------------------------------------------------------------------------
def _liver(dx=0.0, raised=False, h=1080, w=1920):
    from tests.conftest import make_shapes
    shapes = make_shapes(h, w, raised, np.random.default_rng(0))
    for s in shapes:
        for pt in s["points"]:
            pt[0] += dx * w
    return Annotation(h, w, [Shape(s["label"], class_of(s["label"]), np.asarray(s["points"], np.float32))
                             for s in shapes])


def test_junctions_found_and_ordered():
    c, v = derive_junctions(_liver())
    assert v.tolist() == [1, 1, 1, 1]
    assert c[2, 0] < c[3, 0]                       # lateral_a is the left one
    assert abs(c[0, 1] - 0.155) < 0.03 and abs(c[1, 1] - 0.74) < 0.03


def test_junctions_survive_shift_that_broke_old_rule():
    # shift the liver right: both lateral points end up right of 45% width; the old rule merged them
    c, v = derive_junctions(_liver(dx=0.3))
    assert v[2] == 1 and v[3] == 1 and c[2, 0] < c[3, 0]


def test_junctions_raised_liver():
    c, v = derive_junctions(_liver(raised=True))
    assert v.tolist() == [1, 1, 1, 1]
    assert c[0, 1] > c[1, 1]  # ligament-silhouette point is now below the ligament-ridge point


def test_missing_ligament_gives_invisible_not_guessed():
    ann = _liver()
    ann.shapes = [s for s in ann.shapes if s.class_id != 3]
    c, v = derive_junctions(ann)
    assert v[0] == 0 and v[1] == 0 and (c[:2] == 0).all()


def test_hflip_swaps_laterals():
    c, v = derive_junctions(_liver())
    c2, v2 = hflip_junctions(c, v)
    assert np.allclose(c2[2, 0], 1 - c[3, 0]) and np.allclose(c2[3, 0], 1 - c[2, 0])


# ---- sampler ------------------------------------------------------------------------------------
def test_pose_balanced_sampler_ratio():
    poses = ["raised"] * 10 + ["canonical"] * 90
    s = PoseBalancedSampler(poses, seed=3, num_samples=20000)
    idx = list(s)
    share = sum(poses[i] == "raised" for i in idx) / len(idx)
    assert abs(share - 0.5) < 0.02
    assert len(PoseBalancedSampler(poses)) == 100
    with pytest.raises(ValueError):
        PoseBalancedSampler(["raised", "unknown"])
