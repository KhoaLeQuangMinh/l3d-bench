# A2ONet (MICCAI 2026) — https://github.com/hyperiondk115/A2ONet @ 765ada123848c1656cfa8fc120157a093fb69b13

Recipe as released: `python train.py --data_path <train_root> --save_dir <out> --epochs 100 --batch_size <fits>`
(Adam lr 1e-4, wd 1e-5, cosine to 1e-6, seed 3407; README default is 5 GPUs x batch 1 = global batch 5).
Needs `path/sam_vit_b_01ec64.pth` and `path/depth_anything_v2_vitb.pth`.
Data root style: **lower** (`train/ val/ test/`).

Checkpoint selection: saves `best_path.pth` by **test** Dice (the file its README evaluates) and
`best_val_path.pth` by val Dice. We use `best_val_path.pth`; with our roots, its test folder is also the
selection set. Its metric (a2onet/metrics.py) is our scoring protocol.

Deviation noted: single Kaggle GPU, so the global batch of 5 is not matched; we use the README's single-GPU
command (--batch_size 2). Evaluation every 5 epochs, as released (so selection is among epochs 0, 5, ..., 99).
Export mirrors test.py (pretrained_resnet=False, fusion alpha 1.0).

Patches (patches/):
- 01_resume.diff — infrastructure only: resume.pth after every epoch, clean exit (code 3) before Kaggle's time
  limit, TRAINING_DONE marker, L3D_MAX_EPOCHS for the pilot. Its own best_val_path.pth / best_path.pth unchanged.

Check from the smoke test: on identical predictions, A2ONet's compute_metrics and our scorer give the same
Dice, IoU and ASSD to every printed digit (it is the protocol we copied).

How it runs: `python -m l3d_bench.baselines.run --method a2onet --split <official|fK>` (notebook:
`notebooks/baseline_template.ipynb`). Official split: the repo sees the real Train/Val/Test and keeps its own
test-selected checkpoint (reproduction number) plus a val-selected one (benchmark number). Folds: Val = Test =
inner-val, so nothing it selects on is ever scored. Predictions are exported by `export_predictions.py` (run
inside the repo with its own dataset class and model) and scored by `l3d_bench.eval`; the repo's own metric is
kept next to ours (`repometric_*.json`).
