# A2ONet (MICCAI 2026) — https://github.com/hyperiondk115/A2ONet @ 765ada123848c1656cfa8fc120157a093fb69b13

Recipe as released: `python train.py --data_path <train_root> --save_dir <out> --epochs 100 --batch_size <fits>`
(Adam lr 1e-4, wd 1e-5, cosine to 1e-6, seed 3407; README default is 5 GPUs x batch 1 = global batch 5).
Needs `path/sam_vit_b_01ec64.pth` and `path/depth_anything_v2_vitb.pth`.
Data root style: **lower** (`train/ val/ test/`).

Checkpoint selection: saves `best_path.pth` by **test** Dice (the file its README evaluates) and
`best_val_path.pth` by val Dice. We use `best_val_path.pth`; with our roots, its test folder is also the
selection set. Its metric (a2onet/metrics.py) is our scoring protocol.

Deviation noted: single Kaggle GPU, so the global batch of 5 is not matched.
Patches: none needed to run.
