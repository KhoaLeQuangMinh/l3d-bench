# D2GPLand (MICCAI 2024) — https://github.com/PJLallen/D2GPLand @ 5768f86063bb6f9d28d769ebd8256d1513f70447

Recipe as released: `python train.py --data_path <train_root> --batch_size 4 --lr 1e-4 --decay_lr 1e-6 --epoch 60`
Needs SAM-b weights and AdelaiDepth maps in `<split>/depth_AdelaiDepth/` (present in the Kaggle L3D copy).
Data root style: **upper** (`Train/ Val/ Test/`).

Checkpoint selection: validates on `test_file` and saves `best_model_path.pth` + `best_prototype_path.pth`
to the working directory. Our training root points `Test/` at the selection set.

Known quirks kept as released: scheduler stepped once after training; the dataset class overwrites the
passed transform with ToTensor (no augmentation); GT label map resized with bilinear interpolation;
lines 30 px; same 4K path-substring rule as TopoNet.

Patches (patches/):
- 01_fix_get_split.diff — `get_split` compared folder names with `is` (always False for os.listdir
  strings, verified), so Train and Test came back empty and every folder was read as Val; it also returned
  bare file names. Fixed to `==` and full paths. Needed to run at all.
- 02_resume_val_selection.diff — infrastructure only: checkpoints go to L3D_SAVE_DIR instead of the working
  directory; resume.pth every epoch and clean exit before Kaggle's limit; extra checkpoints
  `best_val_model_path.pth` / `best_val_prototype_path.pth` chosen by Dice on the val folder with the repo's own
  evaluation code; `epochs.jsonl`; L3D_MAX_EPOCHS for the pilot.

- 03_memory_only.diff — memory only, same values and gradients: the frozen SAM encoder sits behind a trainable
  1x1 conv, so training stores all SAM activations for 4 images (out of memory on a 16 GB T4, pilot 3 Nov). SAM
  now runs one image at a time with activation checkpointing during training; the contrastive loss resizes one
  mask channel and repeats it instead of resizing 256 identical copies. Checked: identical outputs and
  gradients in float64 (float32 differs only by rounding), identical mask values.

Batch size must stay 4: the contrastive loss hard-codes ref_labels for 4 images (12 = 3 classes x 4). It also
built a (4, 256, 1024, 1024) float mask per class (~4.3 GB); patch 03 removes that.
SAM-b is read from the fixed path `sam_path/sam_vit_b_01ec64.pth` inside the repo (linked by the runner).

How it runs: `python -m l3d_bench.baselines.run --method d2gpland --split <official|fK>` (notebook:
`notebooks/baseline_template.ipynb`). Official split: the repo sees the real Train/Val/Test and keeps its own
test-selected checkpoint (reproduction number) plus a val-selected one (benchmark number). Folds: Val = Test =
inner-val, so nothing it selects on is ever scored. Predictions are exported by `export_predictions.py` (run
inside the repo with its own dataset class and model) and scored by `l3d_bench.eval`; the repo's own metric is
kept next to ours (`repometric_*.json`).
