# TopoNet (MICCAI 2025) — https://github.com/cuiruize/TopoNet @ be25da521d5a0fb7dd1a06990c8e1532c6597cb1

Recipe as released: `python train.py --data_path <train_root> --depth_path depth_anything_v2_vitb.pth --save_path <out>/`
(batch 4, Adam lr 1e-4, wd 3e-5, 200 epochs, Betti loss after epoch 5).
Data root style: **lower** (`train/ val/ test/`); any other folder name is read as val.

Checkpoint selection: `train.py` validates on `test_file` (its test split) and keeps `best_model_path.pth`.
Our training root points `test/` at the selection set (inner-val or official val), so it selects there.

Known quirks kept as released: `scheduler.step()` sits after the epoch loop (LR stays at 1e-4);
4K canvases chosen by path substring (`_31`, `_36`, `_25`, `_29`), which misses Patient 32.

Patches: none needed to run.
