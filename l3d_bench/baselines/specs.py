"""What is specific to each baseline: repo, recipe, setup, training command, checkpoints, export.

Every training command uses the repo's own defaults (its published recipe); only paths are passed.
TopoNet's and D2GPLand's argparse options have no type=, so passing numbers on the command line would turn
them into strings and crash: their defaults are the recipe, and we leave them alone.
"""
from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

from ..config import REPO_ROOT
from .runner import Ctx, pip, sh

SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"


def depth_anything_vitb(ctx: Ctx) -> Path:
    dst = ctx.weights / "depth_anything_v2_vitb.pth"
    if not dst.exists():
        from huggingface_hub import hf_hub_download
        src = hf_hub_download("depth-anything/Depth-Anything-V2-Base", "depth_anything_v2_vitb.pth")
        dst.symlink_to(src)
    return dst


def sam_vitb(ctx: Ctx) -> Path:
    dst = ctx.weights / "sam_vit_b_01ec64.pth"
    if not dst.exists():
        tmp = dst.with_suffix(".part")
        urllib.request.urlretrieve(SAM_URL, tmp)
        tmp.rename(dst)
    return dst


def once(ctx: Ctx, key: str) -> bool:
    """True the first time per work folder (so setup runs once per Kaggle session)."""
    marker = ctx.work / f".setup_{key}"
    if marker.exists():
        return False
    marker.touch()
    return True


class Spec:
    name = ""
    repo_url = ""
    commit = ""
    style = "upper"
    recipe = ""
    pilot_epochs = 2
    reserve_hours = 1.5              # final uploads (multi-GB checkpoints: up to ~1 h on a slow Drive) + export
    sync_files: list[str] = []

    def export_script(self) -> str:
        return str(REPO_ROOT / "baselines" / self.name / "export_predictions.py")


class TopoNet(Spec):
    name = "toponet"
    repo_url = "https://github.com/cuiruize/TopoNet.git"
    commit = "be25da521d5a0fb7dd1a06990c8e1532c6597cb1"
    style = "lower"
    recipe = "train.py defaults: 200 epochs, batch 4, Adam 1e-4, wd 3e-5, Betti loss after epoch 5"
    pilot_epochs = 7                 # the Betti / clDice loss starts after epoch 5
    sync_files = ["best_val_model.pth", "best_model_path.pth", "resume.pth"]
    released_gdrive_id = "1dfvMlagKPOO3P3QswYns8JO5tLTj798r"
    betti_commit = "50d7379"         # Betti-Matching-3D, Sept 2024 (TopoNet's era; the 2026 commit changed the API)

    def setup(self, ctx: Ctx, released: bool = False) -> None:
        if once(ctx, "toponet_pip"):
            pip("pytorch-metric-learning", "einops", "medpy", "pybind11", "gdown",
                "git+https://github.com/google-deepmind/surface-distance.git")
        bm = ctx.repo / "betti_match" / "Betti_Matching"
        if not list((bm / "betti_build").glob("betti_matching*.so")):
            if not bm.exists():
                sh(["git", "clone", "-q", "https://github.com/nstucki/Betti-Matching-3D.git", str(bm)])
            sh(["git", "-C", str(bm), "checkout", "-q", self.betti_commit])
            cmake_dir = sh_out([sys.executable, "-m", "pybind11", "--cmakedir"])
            sh(["cmake", "-S", str(bm), "-B", str(bm / "betti_build"), f"-Dpybind11_DIR={cmake_dir}",
                f"-DPYTHON_EXECUTABLE={sys.executable}", "-Wno-dev"])
            sh(["cmake", "--build", str(bm / "betti_build"), "-j4"])
        self.depth = depth_anything_vitb(ctx)

    def train_cmd(self, ctx: Ctx):
        return ([sys.executable, "train.py", "--data_path", f"{ctx.root}/", "--depth_path", str(self.depth),
                 "--save_path", f"{ctx.run_dir}/"], ctx.repo, {})

    def checkpoints(self, ctx: Ctx) -> dict:
        ck = {"val": ctx.run_dir / "best_val_model.pth"}
        if ctx.split == "official":
            ck["native"] = ctx.run_dir / "best_model_path.pth"
        return ck

    def released_checkpoint(self, ctx: Ctx) -> Path:
        dst = ctx.weights / "toponet_released.pth"
        if not dst.exists():
            sh([sys.executable, "-m", "gdown", "--fuzzy", "-O", str(dst),
                f"https://drive.google.com/file/d/{self.released_gdrive_id}/view"])
        return dst

    def export_cmd(self, ctx: Ctx, ckpt: Path, frame_list: Path, out: Path, native: Path):
        return [sys.executable, self.export_script(), "--model_path", str(ckpt), "--depth_path", str(self.depth),
                "--list", str(frame_list), "--out", str(out), "--native_json", str(native)]


class A2ONet(Spec):
    name = "a2onet"
    repo_url = "https://github.com/hyperiondk115/A2ONet.git"
    commit = "765ada123848c1656cfa8fc120157a093fb69b13"
    style = "lower"
    recipe = ("DDP on Kaggle's 2 T4s, batch 1 per GPU with SyncBatchNorm (paper: 5 GPUs x batch 1); train.py "
              "defaults: 100 epochs, Adam 1e-4, wd 1e-5, cosine to 1e-6, eval every 5 epochs, seed 3407")
    pilot_epochs = 2
    sync_files = ["best_val_path.pth", "best_path.pth", "metrics.json", "train_log.txt", "resume.pth"]

    def setup(self, ctx: Ctx, released: bool = False) -> None:
        if released:
            raise ValueError("A2ONet has no released weights")
        if once(ctx, "a2onet_pip"):
            pip("medpy")
        self.sam, self.depth = sam_vitb(ctx), depth_anything_vitb(ctx)

    def train_cmd(self, ctx: Ctx):
        # The paper trained with DDP, batch 1 per GPU (5 GPUs, SyncBatchNorm). Batch 2 on one T4 runs out of
        # memory, so on Kaggle's T4 x2 we run the authors' own DDP path: 2 GPUs x batch 1 = global batch 2,
        # the same global batch as the README's single-GPU command. L3D_GPUS=1 falls back to one GPU, batch 1.
        import torch
        gpus = int(os.environ.get("L3D_GPUS", torch.cuda.device_count() or 1))
        launch = ([sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={gpus}"]
                  if gpus > 1 else [sys.executable])
        self.launch_info = {"gpus": gpus, "batch_per_gpu": int(os.environ.get("L3D_BATCH_SIZE", "1"))}
        return (launch + ["train.py", "--data_path", str(ctx.root), "--save_dir", str(ctx.run_dir),
                 "--batch_size", os.environ.get("L3D_BATCH_SIZE", "1"), "--num_workers", "2",
                 "--seed", str(3407 + ctx.seed),
                 "--sam_checkpoint", str(self.sam), "--depth_checkpoint", str(self.depth),
                 "--pretrained_dir", str(ctx.weights)], ctx.repo, {})

    def checkpoints(self, ctx: Ctx) -> dict:
        ck = {"val": ctx.run_dir / "best_val_path.pth"}
        if ctx.split == "official":
            ck["native"] = ctx.run_dir / "best_path.pth"
        return ck

    def export_cmd(self, ctx: Ctx, ckpt: Path, frame_list: Path, out: Path, native: Path):
        return [sys.executable, self.export_script(), "--model_path", str(ckpt), "--sam", str(self.sam),
                "--depth", str(self.depth), "--list", str(frame_list), "--out", str(out), "--native_json", str(native)]


class D2GPLand(Spec):
    name = "d2gpland"
    repo_url = "https://github.com/PJLallen/D2GPLand.git"
    commit = "5768f86063bb6f9d28d769ebd8256d1513f70447"
    style = "upper"
    recipe = "train.py defaults: 60 epochs, batch 4, Adam 1e-4, wd 3e-5 (scheduler stepped once, as released)"
    pilot_epochs = 2
    sync_files = ["best_val_model_path.pth", "best_val_prototype_path.pth", "best_model_path.pth",
                  "best_prototype_path.pth", "resume.pth"]

    def setup(self, ctx: Ctx, released: bool = False) -> None:
        if released:
            raise ValueError("D2GPLand's released prototype weights are not linked; train it instead")
        if once(ctx, "d2gpland_pip"):
            pip("pytorch-metric-learning", "medpy")
        link = ctx.repo / "sam_path" / "sam_vit_b_01ec64.pth"     # the model reads SAM from this fixed path
        link.parent.mkdir(exist_ok=True)
        if not link.exists():
            link.symlink_to(sam_vitb(ctx))

    def train_cmd(self, ctx: Ctx):
        return [sys.executable, "train.py", "--data_path", f"{ctx.root}/"], ctx.repo, {}

    def checkpoints(self, ctx: Ctx) -> dict:
        ck = {"val": ctx.run_dir / "best_val_model_path.pth"}
        if ctx.split == "official":
            ck["native"] = ctx.run_dir / "best_model_path.pth"
        return ck

    def export_cmd(self, ctx: Ctx, ckpt: Path, frame_list: Path, out: Path, native: Path):
        proto = ckpt.with_name(ckpt.name.replace("model_path", "prototype_path"))
        return [sys.executable, self.export_script(), "--model_path", str(ckpt), "--prototype_path", str(proto),
                "--list", str(frame_list), "--out", str(out), "--native_json", str(native)]


def sh_out(cmd) -> str:
    import subprocess
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()


SPECS = {s.name: s for s in (TopoNet(), A2ONet(), D2GPLand())}
