# TopoNet (MICCAI 2025) — https://github.com/cuiruize/TopoNet @ be25da521d5a0fb7dd1a06990c8e1532c6597cb1

Recipe as released: `python train.py --data_path <train_root> --depth_path depth_anything_v2_vitb.pth --save_path <out>/`
(batch 4, Adam lr 1e-4, wd 3e-5, 200 epochs, Betti loss after epoch 5).
Data root style: **lower** (`train/ val/ test/`); any other folder name is read as val.

Checkpoint selection: `train.py` validates on `test_file` (its test split) and keeps `best_model_path.pth`.
Our training root points `test/` at the selection set (inner-val or official val), so it selects there.

Known quirks kept as released: `scheduler.step()` sits after the epoch loop (LR stays at 1e-4);
4K canvases chosen by path substring (`_31`, `_36`, `_25`, `_29`), which misses Patient 32.

Patches (patches/):
- 01_resume_val_selection.diff — infrastructure only, training unchanged: resume.pth after every epoch and a
  clean exit (code 3) before Kaggle's time limit; an extra checkpoint `best_val_model.pth` chosen by Dice on the
  val folder with the repo's own evaluation code; `epochs.jsonl` log; L3D_BATCH_SIZE / L3D_MAX_EPOCHS (pilot)
  read from the environment. The released `best_model_path.pth` (selected on test) is still written.

Setup: Betti-Matching-3D built from source at 50d7379 (Sept 2024; its 2026 commit changed the API) into
`betti_match/Betti_Matching/betti_build`, the import path train.py expects; Depth Anything V2 ViT-B from Hugging
Face; ImageNet ResNet-34 downloaded by the code; surface-distance from GitHub.

Cost warning: the Betti loss runs on the CPU (on 1024 x 1024 maps, ~1 s per class map measured here), and the
recipe is 200 epochs. Expect several Kaggle sessions per run; the pilot measures minutes per epoch.

Ground-truth quirks kept as released (they affect TopoNet's own training targets and its own metric, never
our scoring): 4K canvas chosen by path substring (`_31`, `_36`, `_25`, `_29`), so Patients 30, 32 and 39 (also
4K) are drawn on a 1080p canvas and clipped, and 1080p frames whose name contains those substrings (e.g.
unpadded frame numbers of Patients 40-42, 49) are drawn on a 4K canvas.

How it runs: `python -m l3d_bench.baselines.run --method toponet --split <official|fK>` (notebook:
`notebooks/baseline_template.ipynb`). Official split: the repo sees the real Train/Val/Test and keeps its own
test-selected checkpoint (reproduction number) plus a val-selected one (benchmark number). Folds: Val = Test =
inner-val, so nothing it selects on is ever scored. Predictions are exported by `export_predictions.py` (run
inside the repo with its own dataset class and model) and scored by `l3d_bench.eval`; the repo's own metric is
kept next to ours (`repometric_*.json`).

G2: `--released` scores the released weights (Google Drive, via gdown) on official val and test.
