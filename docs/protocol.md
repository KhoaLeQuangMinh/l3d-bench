# Evaluation protocol (one protocol for every number)

Copied from A2ONet's released code (`data/dataset.py`, `a2onet/metrics.py`, commit 765ada1);
TopoNet and D2GPLand compute the same Dice. Implemented in `l3d_bench/eval/metrics.py`, verified
against a verbatim copy of A2ONet's functions in `tests/reference_a2onet.py`.

```
for each scored frame:
    GT    = zeros(imageHeight, imageWidth)
    for shape in labels.shapes (file order):          # later shapes overwrite earlier ones
        class = {r:1, s:2, l:3}[first letter of label] (0 otherwise)
        draw shape.points as cv2.line, thickness int(35 * imageHeight / 1080)
    GT    = resize(GT, 1024 x 1024, nearest)
    PRED  = model label map at 1024 x 1024 (argmax over background + 3 classes)
    P, G  = stack(PRED == c for c in 1..3), stack(GT == c for c in 1..3)       # (3, 1024, 1024)
    Dice  = (2 |P & G| + 1e-5) / (|P| + |G| + 1e-5)      # pooled over the three classes
    IoU   = Dice / (2 - Dice)                             # as the papers report it
    ASSD  = medpy.assd(P, G) if P and G non-empty else 80
report: mean over frames of Dice, IoU, ASSD
secondary: per-class Dice (NaN when a class is absent from both), per patient, per pose,
           patient-level bootstrap 95% CI (folds only; the test set has 4 patients)
```

Checkpoint selection: every method selects on a set it is never scored on
(official: 122 val frames -> scored on 109 test; folds: inner-val -> scored on held-out).
The test set is scored once per method and seed.
