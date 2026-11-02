import math

import cv2
import numpy as np
import pandas as pd
import pytest

from l3d_bench.data.annotations import load_annotation
from l3d_bench.data.masks import gt_for_scoring
from l3d_bench.eval.aggregate import summarize, summarize_folds
from l3d_bench.eval.metrics import EMPTY_ASSD, a2onet_metrics, per_class_dice, score_frame
from l3d_bench.eval.score import score_folder
from tests import reference_a2onet as ref


def _bar(shift=0, cls=1, size=64):
    lab = np.zeros((size, size), np.uint8)
    lab[20:24, 10 + shift:50 + shift] = cls
    return lab


def test_perfect_and_empty():
    gt = _bar()
    d, i, a = a2onet_metrics(gt, gt)
    assert d == pytest.approx(1.0) and i == pytest.approx(1.0) and a == 0.0
    d, i, a = a2onet_metrics(np.zeros_like(gt), gt)
    assert d < 1e-6 and a == EMPTY_ASSD
    assert score_frame(np.zeros_like(gt), gt)["assd_empty_penalty"] == 1.0


def test_shift_gives_assd_close_to_shift():
    gt = np.zeros((64, 64), np.uint8)
    gt[10:50, 20:24] = 1
    pred = np.roll(gt, 3, axis=1)
    _, _, a = a2onet_metrics(pred, gt)
    assert 1.0 < a <= 3.0


def test_matches_a2onet_reference_on_random_pairs():
    rng = np.random.default_rng(0)
    for _ in range(25):
        gt = np.zeros((64, 64), np.uint8)
        pred = np.zeros((64, 64), np.uint8)
        for lab in (gt, pred):
            for c in (1, 2, 3):
                if rng.random() < 0.8:
                    x0, y0 = rng.integers(0, 50, 2)
                    lab[y0:y0 + rng.integers(2, 14), x0:x0 + rng.integers(2, 14)] = c
        assert a2onet_metrics(pred, gt) == pytest.approx(ref.compute_metrics(pred, gt))


def test_dice_is_pooled_not_averaged():
    gt = np.zeros((32, 32), np.uint8)
    gt[0:10, 0:10] = 1          # big ridge
    gt[20:22, 20:22] = 3        # tiny ligament
    pred = gt.copy()
    pred[20:22, 20:22] = 0      # miss the ligament completely
    d, _, _ = a2onet_metrics(pred, gt)
    pc = per_class_dice(pred, gt)
    assert d > 0.95                               # pooled Dice barely notices
    assert pc["dice_ligament"] == 0.0             # per-class Dice does
    assert math.isnan(pc["dice_silhouette"])      # absent from both


def test_score_folder_end_to_end(index, tmp_path):
    test = index[index["split"] == "test"]
    for _, r in test.iterrows():  # perfect predictions, saved at a different size to test resizing
        gt = gt_for_scoring(load_annotation(r["label_path"]), size=1024)
        cv2.imwrite(str(tmp_path / f"{r['frame_id']}.png"), cv2.resize(gt, (512, 512), interpolation=cv2.INTER_NEAREST))
    df = score_folder(tmp_path, index, list(test["frame_id"]))
    assert len(df) == 6 and df["dice"].min() > 0.98


def test_score_folder_rejects_rgb(index, tmp_path):
    fid = index["frame_id"].iloc[0]
    cv2.imwrite(str(tmp_path / f"{fid}.png"), np.zeros((8, 8, 3), np.uint8))
    with pytest.raises(ValueError):
        score_folder(tmp_path, index, [fid])


def test_summarize_and_bootstrap():
    rows = []
    for p in range(6):
        for f in range(3):
            rows.append({"frame_id": f"{p}_{f}", "patient": f"P{p}", "pose": "raised" if f == 0 else "canonical",
                         "dice": 0.5 + 0.05 * p, "iou": 0.4, "assd": 10.0})
    s = summarize(pd.DataFrame(rows), n_boot=200)
    assert s["mean"]["dice"] == pytest.approx(np.mean([0.5 + 0.05 * p for p in range(6)]))
    assert s["per_pose"]["raised"]["n_frames"] == 6
    lo, hi = s["ci95"]["dice"]
    assert lo < s["mean"]["dice"] < hi
    f = summarize_folds([{"dice": 0.6}, {"dice": 0.7}])
    assert f["dice"]["mean"] == pytest.approx(0.65)
