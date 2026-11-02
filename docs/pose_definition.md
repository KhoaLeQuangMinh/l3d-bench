# Pose labeling guide (draft — Khoa to finalize with example images)

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
