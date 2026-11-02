# l3d-bench

A fair re-evaluation of laparoscopic liver landmark detection on L3D: one scoring protocol, patient-
disjoint pose-stratified folds, every published method retrained under the same splits.

## Status

| Phase | Content | State |
| --- | --- | --- |
| 0 Setup | repo, configs, Drive sync (rclone, from the UNETR++ notebook), checkpoint/resume, run manifest | code done, needs G0 on Kaggle |
| 1 Data | annotations, index, label maps (A2ONet-exact), folds + inner-val, pose labels, junction targets, dataset, M3 sampler | code done, needs G1 on real L3D |
| 2 Metric | A2ONet-exact Dice / IoU / ASSD + per-class Dice, folder scorer, patient bootstrap | code done, needs G2 (TopoNet weights) |
| 3 M1 | Mask2Former wrapper, training loop, export | next |
| 4 M2, M3 | junction head + steering, freeze | after 3 |

## Layout

```
configs/            base.yaml + m1/m2/m3 (only the switch differs)
l3d_bench/data/     annotations, index, masks, folds, pose, junctions, dataset, sampler, roots (baseline symlinks)
l3d_bench/eval/     metrics (the protocol), score (folder of PNG label maps), aggregate
l3d_bench/train/    checkpoint, drive, seed
l3d_bench/runs/     manifest (run ids: <method>__<variant>__<split>__s<seed>)
scripts/            phase1_data_checks, make_folds, make_manifest, build_baseline_roots, precompute_sam_features
baselines/<name>/   pinned commit, recipe, patches/*.diff (only fixes that make code run or score correctly)
docs/               protocol, pose_definition, gates
notebooks/          Kaggle templates
tests/              synthetic L3D fixtures; reference copy of A2ONet's metric as test oracle
```

## First steps on Kaggle (G0 + G1)

1. Attach `khoatrytopublish/l3d-train`, `l3d-val`, `l3d-test`. Under Add-ons > Secrets, add `RCLONE_CONFIG_GDRIVE` (the full `[gdrive]` rclone config of the lab Google account) and switch it on for the notebook. Never put that config in the repo or a notebook: this repo is public.
2. Run `notebooks/phase1_data_checks.ipynb`. It writes `data/index.csv` and `reports/g1/` and syncs them to Drive.
3. Check the overlays by eye and fill in `docs/gates.md`.

## Local

```
pip install -e ".[dev]" torch
pytest -q
```
