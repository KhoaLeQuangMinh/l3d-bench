# BCRNet (MICCAI 2025) — https://github.com/jinlab-imvr/BCRNet @ 5bf74f941f7e7961b095fb731b0d73ee0b545442

Recipe (paper: arXiv 2506.15279, implementation details; README command): 60 epochs, batch 4, Adam lr 1e-5,
wd 1e-4. train.py has no defaults for epochs and batch, so the runner passes them:
`python train.py --data_path <root>/ --epoch 60 --bs 4 --save_path <run_dir>` (wandb offline).
Data root style: **upper** (`Train/ Val/ Test/`).

    python -m l3d_bench.baselines.run --method bcrnet --split official     # or f0..f4; --pilot first

## What the runner does (l3d_bench/baselines/specs.py, class BCRNet)

1. Installs fvcore, shapely, timm, wandb, ninja, segment-anything (detectron2 is **not** needed: nothing on the
   train or test path imports it; only adet/config and the unused visualiser do).
2. Builds the multi-scale deformable attention op `adet._C` (build_ext.py): JIT-compiles adet/layers/csrc
   with torch.utils.cpp_extension and checks forward and backward against the PyTorch reference
   (`ms_deform_attn_core_pytorch`, same file) in float64. If the build or the check fails, patch 02 makes
   MSDeformAttn use the PyTorch reference: the same computation, slower. The run card's `launch` field says
   which one ran.
3. Precomputes, for every frame of the run (bcrnet_prep.py; existing files are skipped):
   - `s_bezier/<id>.npz`: BCRNet's own `utils/preprocess.py generate_beziers` with its defaults (degree 5,
     25 curve points, 1080 x 1920 mask canvas, 30 px), on the L3D LabelMe JSON.
   - `sam/<id>.npy`: SAM ViT-B image-encoder features, 256 x 64 x 64. **Assumption** (the script that made
     them is not released): the input is the frame squashed to 1024 x 1024, exactly the image BCRNet's model
     sees (the commented code in backbone/Unet.py feeds the model's image batch to `sam.image_encoder`), with
     SAM's own normalisation. Squashing keeps each SAM cell aligned with the CNN feature it is added to.
   About 4 MB per frame, stored in /tmp on Kaggle (/kaggle/working is capped at 20 GB).
4. Trains with the recipe; patch 02 adds resume.pth every epoch and the session time guard.
5. Selection (BCRNet's code selects nothing; it saves `epoch-N.pt` every 10 epochs and logs val Dice to
   wandb): every saved checkpoint is scored on the selection set (official val / inner-val) with our metric,
   model.eval(); the best is kept, earliest on ties (`selection.json`). Official split only: 'native' = the
   last checkpoint run as the released test.py runs it (train mode), for the reproduction number.
6. Export (export_predictions.py): BCRNet predicts point sequences per landmark (`ctrl_points`, in native
   pixels). We draw them as polylines on the native canvas with the protocol thickness int(35 * H / 1080),
   classes silhouette -> 2, ligament -> 3, ridge -> 1 (drawn in that order), and resize to 1024 x 1024
   (nearest). Its own metric (test.py `metrix`) is stored next to ours (`repometric_*.json`).

## Patches (patches/)

- 01_test_in_eval_mode.diff — test.py ran the model in train mode (BatchNorm uses batch statistics at test
  time). Changed to eval(). Affects scoring only.
- 02_resume_extension_storage.diff —
  - train.py: resume.pth (model, optimizer, epoch, RNG) after every epoch, stop before the session limit
    (exit 3), `L3D_MAX_EPOCHS` for pilots, also save `epoch-N.pt` at the last epoch (pilots stop early),
    TRAINING_DONE marker, print the validation Dice it logs to wandb. Training math unchanged.
  - adet/layers/ms_deform_attn.py: `_C` optional, PyTorch reference used when it is missing or on CPU.
  - csrc: deprecated ATen calls that no longer compile with current PyTorch (`.type().is_cuda()` ->
    `.is_cuda()`, `.data<T>()` -> `.data_ptr<T>()`, dispatch on `scalar_type()`). Same kernels.
  - utils/preprocess.py: landmark masks stored as uint8 0/1 with np.savez_compressed (50 MB -> ~0.1 MB per
    frame); values identical (the dataset casts them to float).

## Quirks of the released code (kept, reported)

- Normalisation: images are scaled to 0-1, then normalised with PIXEL_MEAN/STD given in 0-255 units
  (123.675 / 58.395 ...), so the RGB input lies in about [-2.12, -2.10]. Kept as released.
- Validation in train.py runs with the model in train mode (`# model.eval()` is commented out).
- Its metric draws predictions scaled to the native size on a 1080 x 1920 canvas, so the 4K frames
  (Patient 10) are misaligned in its own numbers. Our scorer uses the native canvas.
- No seeding anywhere in train.py: runs are not bit-reproducible.
- The DeepSolo weights named in config.yaml are never loaded. ImageNet ResNet-50 is downloaded by the code.

## Checks done before the first Kaggle run (CPU sandbox, fake L3D frames)

- Fresh clone at the pinned commit + patches 01-03 apply cleanly.
- Pilot flow end to end: forced pause after epoch 1, resume from resume.pth, epoch 2, selection over the
  saved checkpoints, export + scoring (val on select and held-out, native on held-out), DONE.
  Also resumed correctly after a crash in the middle of writing resume.pth (atomic tmp + replace).
- Export geometry: rendering BCRNet's own Bezier targets (curve points x native size) through
  export_predictions.render gives per-class Dice 0.91-0.97 against our GT on every frame tried, so the
  coordinates and the class mapping are right.
- Not checkable without a GPU: the CUDA build of adet._C (build_ext.py checks it against the reference on
  Kaggle) and GPU memory at batch 4 (the pilot shows it). Checkpoint sizes: epoch-N.pt 357 MB, resume.pth
  1.06 GB, so about 3.2 GB on Drive per run.

## Patch 03

- 03_fix_loss_schedule.diff — `1 - torch.sigmoid((e - 10) / 2)` passes a Python float to torch.sigmoid,
  which raises TypeError in every PyTorch version, so the released train.py stops at epoch 1. Wrapped the
  value in torch.tensor; the schedule is the one written.

## Patch 04

- 04_fix_bezier_sampler_degree.diff — the model predicts degree-5 Bezier proposals (config BEZIER_DEGREE: 5,
  6 control points) but TransformerPureDetector builds its BezierSampler with the default degree 4. The
  sampler only runs, in inference(), on proposals scoring above INFERENCE_TH_TEST (0.35), so training
  crashes as soon as the model gets confident: on Kaggle at step 139 of epoch 1 ("Expected size ... [1, 5]
  but got: [1, 6]"). The sampled points feed only the visualisation and an unused Chamfer value, never the
  loss or the predictions we score, so passing degree=BEZIER_DEGREE changes nothing else.
  Checked on CPU with the score threshold forced to 0 (every query kept): inference in eval and train mode
  and the export all run.
- First Kaggle run: adet._C built and matched the PyTorch reference (max abs err 1.4e-14); batch 4 fits a
  T4 at about 2.55 s/step, 231 steps/epoch (about 10 min/epoch, about 10-11 h for 60 epochs + validation).
