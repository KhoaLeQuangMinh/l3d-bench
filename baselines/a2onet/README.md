# A2ONet (MICCAI 2026) — https://github.com/hyperiondk115/A2ONet @ 765ada123848c1656cfa8fc120157a093fb69b13

Recipe as released: `python train.py --data_path <train_root> --save_dir <out> --epochs 100 --batch_size <fits>`
(Adam lr 1e-4, wd 1e-5, cosine to 1e-6, seed 3407; README default is 5 GPUs x batch 1 = global batch 5).
Needs `path/sam_vit_b_01ec64.pth` and `path/depth_anything_v2_vitb.pth`.
Data root style: **lower** (`train/ val/ test/`).

Checkpoint selection: saves `best_path.pth` by **test** Dice (the file its README evaluates) and
`best_val_path.pth` by val Dice. We use `best_val_path.pth`; with our roots, its test folder is also the
selection set. Its metric (a2onet/metrics.py) is our scoring protocol.

Batch: batch 2 on one T4 runs out of memory (pilot, 3 Nov), so we use the authors' own DDP path on Kaggle's
two T4s: batch 1 per GPU with SyncBatchNorm, as in the paper (which used 5 GPUs). Global batch 2 instead of 5,
the same global batch as the README's single-GPU command. L3D_GPUS=1 falls back to one GPU at batch 1. Evaluation every 5 epochs, as released (so selection is among epochs 0, 5, ..., 99).
Export mirrors test.py (pretrained_resnet=False), except for the fusion alpha (below).

**Fusion alpha (found 10 Nov, official run).** train.py ramps the curve-mask fusion in: alpha = 0 up to epoch
10, then +0.1 per epoch, 1.0 from epoch 20, and its every-5-epoch evaluation uses the alpha of that epoch. On
the official split both best_val_path.pth and best_path.pth are epoch 10 (val 66.12, test 64.97 in its log),
i.e. from the warm-up, when the fusion branch was switched off. test.py always sets alpha 1.0, so scoring
these files as released (what our first export did) gave test 62.14 for a checkpoint whose own log says
64.97: the released code selects and tests the same weights under two different models. We now score each
checkpoint with the alpha it was selected under ('val', 'native'; recorded in the run card) and keep the
test.py way as 'val_testpy' (alpha 1.0, held-out only). Both are reported. Also worth reporting: no
fully-fused epoch (20-99) beat the warm-up epochs on val in this run (best 64.7 at epoch 40 vs 66.1).

Patches (patches/):
- 01_resume.diff — infrastructure only: resume.pth after every epoch, clean exit (code 3) before Kaggle's time
  limit (under DDP rank 0 decides and broadcasts, all ranks exit 0 and a PAUSED marker is written),
  TRAINING_DONE marker, L3D_MAX_EPOCHS for the pilot. Its own best_val_path.pth / best_path.pth unchanged.

Check from the smoke test: on identical predictions, A2ONet's compute_metrics and our scorer give the same
Dice, IoU and ASSD to every printed digit (it is the protocol we copied).

How it runs: `python -m l3d_bench.baselines.run --method a2onet --split <official|fK>` (notebook:
`notebooks/baseline_template.ipynb`). Official split: the repo sees the real Train/Val/Test and keeps its own
test-selected checkpoint (reproduction number) plus a val-selected one (benchmark number). Folds: Val = Test =
inner-val, so nothing it selects on is ever scored. Predictions are exported by `export_predictions.py` (run
inside the repo with its own dataset class and model) and scored by `l3d_bench.eval`; the repo's own metric is
kept next to ours (`repometric_*.json`).
