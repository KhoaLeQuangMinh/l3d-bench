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
