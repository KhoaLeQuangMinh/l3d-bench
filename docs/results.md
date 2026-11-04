# Results log (official split, seed 0)

Our protocol (A2ONet's metric, correct GT), percent. Test is scored once per method and seed.

| Method | Selected on | Best epoch | Val Dice | Test Dice | Test IoU | Test ASSD | Ridge | Silhouette | Ligament | Run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M1 Mask2Former | val | 21 / 50 | 71.87 | 70.86 | 56.63 | 9.62 | 70.0 | 72.8 | 62.0 | m1__plain__official__s0 |
| M2 + junction steering | val | 18 / 50 | 71.30 | 70.11 | 55.83 | 11.14 | 68.5 | 72.2 | 60.4 | m2__junction__official__s0 |
| TopoNet, released weights | test (authors) | - | 67.28 | 66.37 | 51.89 | 17.51 | 65.3 | 69.2 | 54.9 | _g2__toponet__released |

Notes
- M2 vs M1, one seed: -0.75 Dice, +1.5 px ASSD on test; the val curves overlap (M1 0.70-0.72, M2 0.70-0.71 after
  epoch 6). Not a conclusion until seeds 1-2 are in.
- TopoNet: not retrained (decision 4 Nov, see baselines/toponet/README.md); released weights, trained and selected by
  the authors (on test); no fold results; its own GT/metric gives 64.79 / 50.36 / 17.04.
- A2ONet (running): val Dice 0.584 (ep 0), 0.643 (5), 0.661 (10), 0.648 (15), 0.579 (20, fusion alpha reaches 1 and
  the loss spikes), 0.614 (25); paused after epoch 25 at 9.4 h.

## Seeds (official test)

| Method | Seeds | Dice | IoU | ASSD | Ridge | Silhouette | Ligament | Best epochs (val Dice) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M1 | 0, 1 | 70.83 ± 0.05 | 56.70 ± 0.09 | 10.93 ± 1.86 | 69.70 ± 0.42 | 73.35 ± 0.78 | 61.45 ± 0.78 | 21 (71.87), 9 (71.81) |
| M2 | 0, 1 | 70.83 ± 1.01 | 56.67 ± 1.19 | 11.09 ± 0.08 | 69.35 ± 1.20 | 72.90 ± 0.99 | 61.20 ± 1.13 | 18 (71.30), 23 (71.93) |

- Seed 1 runs: m1__plain__official__s1 (test 70.79 / 56.76 / 12.25), m2__junction__official__s1 (71.54 / 57.51 / 11.03).
- After two seeds M1 and M2 are tied on test Dice (70.83 vs 70.83); M2's seed spread (1.4 Dice) is larger than any
  M1-M2 gap. Val curves are flat after ~epoch 8, so the selected epoch varies a lot between seeds (9 vs 21).
