# Pose labels

**Current labels (4 Nov 2026, confirmed by Khoa: every frame was judged by eye, and `flipped` means the liver is
lifted and the tissue under it exposed, i.e. our `raised`):** `data/pose_labels.csv` = Khoa's orientation audit of 7 Oct 2026
(`Surgical AI/data/user_audit_results.csv`, imported by `scripts/import_pose_audit.py`): the 1043 Train + Val frames,
`normal` -> `canonical` (950), `flipped` -> `raised` (93, 8.9%). The 109 Test frames were deliberately left unlabeled
and stay out of this file. The audit's reviewed `status` differs from its automatic `heuristic_flipped` column on 136
frames (116 heuristic-flipped frames judged normal, 20 flipped frames the heuristic missed).
Raised frames are concentrated: Patient_20 29, Patient_12 20, Patient_40 15, Patient_38 6, Patient_39 5, others <= 5.

Still to do for the paper: a second labeler on the 100 frames in `data/pose_agreement_frames.txt` (30 raised + 70
canonical, random, seed 0; shown in shuffled order, without the first labels) for Cohen's kappa, and test-frame labels for per-pose test
results, made blind to every model output, after the study runs are frozen.

## Labeling guide (for the agreement check and the test frames)

Label every L3D frame as one of two classes. Look at the image only; do not open the old
`user_audit_results.csv` or any model output.

| Label | Definition (to refine) |
| --- | --- |
| `canonical` | The liver rests in its usual position: the anterior ridge lies in the lower part of the view and the tissue under the liver is not lifted. |
| `raised` | The liver is lifted by an instrument (tissue under the liver pulled up / exposed), so the ridge and the silhouette change their usual vertical order or the ridge moves to the upper part of the view. |

Rules
- Unsure -> pick the closer class and add the frame id to `unsure.txt`; these are reviewed together.
- Add 4 example images per class here (frame ids + screenshots) before labeling starts.

Output: `data/pose_labels.csv` with columns `frame_id,pose,labeler`.
Agreement: 100 frames are labeled by two people; report Cohen's kappa (`l3d_bench.data.pose.cohen_kappa`).
