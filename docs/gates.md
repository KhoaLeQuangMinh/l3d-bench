# Gate log

| Gate | Check | Date | Outcome |
| --- | --- | --- | --- |
| G0 Setup | Dummy run on a lab account: clone, read one L3D image, write to Drive, resume in a new session | | |
| G1 Data | `scripts/phase1_data_checks.py`: counts vs D2GPLand (921/122/109, 39 patients), label inventory, class-overlap share, overlays checked by eye | 2026-11-03 | **Pass** (commit 6464254). 921/122/109 frames, 32/3/4 patients, no patient in two splits. Labels: silhouette 2318, rigde 1661 (typo, mapped to ridge), ligament 1101, ridge 390; no unknown labels. Class overlap 0.106% of labelled pixels. 121 frames 3840x2160 (patients 25, 29, 30, 31, 32, 36, 39). Top-patient share: test 77.1%, val 82.8%. Patients 40, 41, 42, 49 use unpadded frame numbers (parsed as int). Overlays (20 random, all 4K, P40/P41, top-overlap frames) checked by Khoa: masks align with anatomy, colours correct. |
| G2 Metric | Tests pass; TopoNet released weights scored with our metric under TopoNet's settings; gap to 65.19 / 50.56 / 28.07 explained | 2026-11-03 | **Dice/IoU reproduced; ASSD open.** Released weights, official test, TopoNet's own GT + test.py metric: Dice 64.79 / IoU 50.36 / ASSD 17.04 (paper 65.19 / 50.56 / 28.07). Same predictions, our protocol: 66.37 / 51.89 / 17.51; on val: ours 67.28 vs TopoNet's own 59.88. The gap is TopoNet's GT rendering (fixed 35 px; 4K canvas by path substring, which clips Patients 30/32/39 and mis-sizes unpadded 1080p frames of P40/41/42/49): to confirm per frame. ASSD 17 vs 28: hypothesis = paper used train.py's per-class surface-distance ASSD (80 when a class is missing); export now reports it as assd_trainpy, re-run --released to check. |
| G3 Pipeline | M1 overfits 10 images (`scripts/overfit_check.py`, Dice >= 0.8); `--pilot` run completes and resumes from Drive; full M1 on the official split; hours per run known | | |
| G4 Switches | M2, M3 one-epoch smoke runs; config diff shows only their switch | | |
| G5 Freeze | `configs/frozen.yaml` committed, tag `study-v1` | | |

## Notes for the paper (from G1)

- Test is 77% one patient, val 83% another: the official split measures mostly two livers.
- 10.5% of frames are 4K, concentrated in seven training patients; report 4K share per fold.
- Raw label typo `rigde` is the majority ridge spelling (1661 vs 390).
- Frame-number format differs for patients 40, 41, 42, 49 (5-6 digits vs 7), hinting at a separate extraction pipeline for the val/test-dominant patients.
- TopoNet's own evaluation code mis-renders ground truth for some frames (see G2): on official val its own Dice for the
  released weights is 59.9 vs 67.3 under the correct GT. The same quirk shapes its training targets.
