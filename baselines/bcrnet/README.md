# BCRNet (MICCAI 2025) — https://github.com/jinlab-imvr/BCRNet @ 5bf74f941f7e7961b095fb731b0d73ee0b545442

Recipe (paper: arXiv 2506.15279, Sec. implementation details): 60 epochs, batch 4, Adam lr 1e-5, wd 1e-4.
`WANDB_MODE=offline python train.py --data_path <train_root> --epoch 60 --bs 4 --save_path <out>`
Data root style: **upper** (`Train/ Val/ Test/`).

Inputs per split folder: `images/`, `depth_AdelaiDepth/` (in the Kaggle copy), `sam/<id>.npy`
(scripts/precompute_sam_features.py, SAM ViT-B, 256x64x64), `s_bezier/<id>.npz` (their utils/preprocess.py,
generate_beziers). DeepSolo weights in config.yaml are never loaded by the code; not needed.
ImageNet ResNet-50 is downloaded by the code.

Checkpoint selection: none in the code (saves every 10 epochs, logs val Dice to wandb). We score each
saved checkpoint on the selection set (Val = inner-val / official val) and keep the best.

Patches (patches/):
- 01_test_in_eval_mode.diff — test.py ran the model in train mode (BatchNorm uses batch statistics at
  test time). Changed to eval(). Affects scoring correctness only.

Runner status: not written yet. Extra setup it needs on Kaggle: detectron2, and compiling `adet._C` (the
multi-scale deformable attention CUDA op in adet/layers/csrc; the repo has no setup.py for it), plus the SAM
features (scripts/precompute_sam_features.py) and Bezier targets (utils/preprocess.py) for every frame.
It has native resume (--model_path epoch-N.pt, saved every 10 epochs).
