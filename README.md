# l3d-bench

A fair re-evaluation of laparoscopic liver landmark detection on L3D: one scoring protocol, patient-
disjoint pose-stratified folds, every published method retrained under the same splits.

## Status

| Phase | Content | State |
| --- | --- | --- |
| 0 Setup | repo, configs, Drive sync (rclone, from the UNETR++ notebook), checkpoint/resume, run manifest | code done, needs G0 on Kaggle |
| 1 Data | annotations, index, label maps (A2ONet-exact), folds + inner-val, pose labels, junction targets, dataset, M3 sampler | code done, needs G1 on real L3D |
| 2 Metric | A2ONet-exact Dice / IoU / ASSD + per-class Dice, folder scorer, patient bootstrap | code done, needs G2 (TopoNet weights) |
| 3 M1 | Mask2Former wrapper, training loop (resume, Drive, time budget), export, `run.py`, overfit check | code done, needs G3 on Kaggle |
| 4 M2, M3 | junction head + steering, freeze | after 3 |

## Layout

```
configs/            base.yaml + m1/m2/m3 (only the switch differs)
l3d_bench/data/     annotations, index, masks, folds, pose, junctions, dataset, sampler, roots (baseline symlinks)
l3d_bench/eval/     metrics (the protocol), score (folder of PNG label maps), aggregate
l3d_bench/models/   mask2former (M1), build (config -> model)
l3d_bench/train/    loop, checkpoint, drive, seed
l3d_bench/run.py    one study run: --run-id <method>__<variant>__<split>__s<seed>
l3d_bench/runs/     manifest (run ids: <method>__<variant>__<split>__s<seed>)
scripts/            phase1_data_checks, overfit_check, make_folds, make_manifest, build_baseline_roots, precompute_sam_features
baselines/<name>/   pinned commit, recipe, patches/*.diff (only fixes that make code run or score correctly)
docs/               protocol, pose_definition, gates
notebooks/          Kaggle templates
tests/              synthetic L3D fixtures; reference copy of A2ONet's metric as test oracle
```

## First steps on Kaggle (G0 + G1)

1. Attach `khoatrytopublish/l3d-train`, `l3d-val`, `l3d-test`. Under Add-ons > Secrets, add `RCLONE_CONFIG_GDRIVE` (the full `[gdrive]` rclone config of the lab Google account) and switch it on for the notebook. Never put that config in the repo or a notebook: this repo is public.
2. Run `notebooks/phase1_data_checks.ipynb`. It writes `data/index.csv` and `reports/g1/` and syncs them to Drive.
3. Check the overlays by eye and fill in `docs/gates.md`.

## Study runs (M1-M3)

Use `notebooks/run_template.ipynb`: set `RUN_ID`, Save & Run All. It runs

```
python -m l3d_bench.run --run-id m1__plain__official__s0            # --pilot for a 2-epoch smoke run
```

which trains, keeps the best epoch on the selection set (official val or the fold's inner-val), then exports
and scores the selection set and the scored set (official test or the held-out fold) with the shared metric.
Everything goes to `gdrive:l3d-bench/results/<run_id>/` every epoch. If Kaggle's time limit stops the run,
running the same notebook again continues from Drive; a finished run (DONE on Drive) is skipped.

G3 order on Kaggle: `overfit_check.py` passes (Dice >= 0.8 on 10 memorised frames) -> `--pilot` run finishes
-> full M1 on the official split (gives hours per epoch and the validation curve).

## Local

```
pip install -e ".[dev]" torch
pytest -q
```
