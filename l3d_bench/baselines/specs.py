"""What is specific to each baseline: repo, recipe, setup, training command, checkpoints, export.

Every training command uses the repo's own defaults (its published recipe); only paths are passed.
TopoNet's and D2GPLand's argparse options have no type=, so passing numbers on the command line would turn
them into strings and crash: their defaults are the recipe, and we leave them alone.
"""
from __future__ import annotations

import os
import shutil
import sys
import time
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
        """A2ONet ramps its curve-mask fusion in during training (alpha 0 up to epoch 10, then +0.1 per epoch, 1.0
        from epoch 20) and evaluates every 5 epochs with the alpha of that epoch, so best_val_path.pth /
        best_path.pth can come from the warm-up, when the fusion branch was switched off. Its test.py always
        sets alpha 1.0, i.e. scores such a checkpoint differently from how it was selected. We score each
        checkpoint with the alpha it was selected under ('val', 'native'), and also the val checkpoint as
        test.py would ('val_testpy', alpha 1.0, held-out only)."""
        ck = {"val": ctx.run_dir / "best_val_path.pth", "val_testpy": ctx.run_dir / "best_val_path.pth"}
        if ctx.split == "official":
            ck["native"] = ctx.run_dir / "best_path.pth"
        self.alphas = {tag: self.selected_alpha(ctx, tag) for tag in ck}
        self.launch_info = {**(getattr(self, "launch_info", None) or {}), "fusion_alpha": self.alphas}
        ctx.log(f"A2ONet fusion alpha used for scoring: {self.alphas}")
        return ck

    @staticmethod
    def fusion_alpha(epoch: int, start: int = 10, warmup: int = 10) -> float:
        """a2onet/losses.py A2OCriterion.fusion_alpha with its defaults (train.py passes none)."""
        return 0.0 if epoch < start else float(min(1.0, (epoch - start) / float(warmup)))

    def selected_alpha(self, ctx: Ctx, tag: str) -> float:
        if tag == "val_testpy":
            return 1.0
        import json
        hist = json.loads((ctx.run_dir / "metrics.json").read_text())
        key = "test" if tag == "native" else ("val" if hist[0].get("val") else "test")
        best = max(hist, key=lambda h: (h[key]["dice"], -h["epoch"]))   # train.py keeps the first strict improvement
        return self.fusion_alpha(int(best["epoch"]))

    def export_cmd(self, ctx: Ctx, ckpt: Path, frame_list: Path, out: Path, native: Path):
        alpha = getattr(self, "alphas", {}).get(ctx.current_tag, 1.0)
        return [sys.executable, self.export_script(), "--model_path", str(ckpt), "--sam", str(self.sam),
                "--depth", str(self.depth), "--list", str(frame_list), "--out", str(out), "--native_json", str(native),
                "--alpha", str(alpha)]


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


class BCRNet(Spec):
    name = "bcrnet"
    repo_url = "https://github.com/jinlab-imvr/BCRNet.git"
    commit = "5bf74f941f7e7961b095fb731b0d73ee0b545442"
    style = "upper"
    recipe = "paper / README: 60 epochs, batch 4, Adam 1e-5, wd 1e-4 (train.py has no defaults; passed on the CLI)"
    epochs, batch = 60, 4
    pilot_epochs = 2
    reserve_hours = 2.0              # + scoring the 6 saved checkpoints on the selection set
    sync_files = [f"epoch-{e}.pt" for e in range(10, 61, 10)] + ["epoch-2.pt", "selection.json", "resume.pth"]

    def setup(self, ctx: Ctx, released: bool = False) -> None:
        import json
        if released:
            raise ValueError("BCRNet has no released weights")
        if once(ctx, "bcrnet_pip"):
            pip("fvcore", "shapely", "timm", "wandb", "ninja", "git+https://github.com/facebookresearch/segment-anything.git")
        if not (ctx.repo / ".l3d_ext.json").exists():          # once per session (the clone lives in the work folder)
            out = sh_out([sys.executable, str(REPO_ROOT / "baselines" / "bcrnet" / "build_ext.py")], cwd=ctx.repo)
            (ctx.repo / ".l3d_ext.json").write_text(out.splitlines()[-1])
            if "error" in json.loads(out.splitlines()[-1]):
                ctx.log("adet._C build output (tail):\n" + out[-3000:])
        self.launch_info = json.loads((ctx.repo / ".l3d_ext.json").read_text())
        ctx.log(f"adet._C: {self.launch_info}")
        self.prep = self.precompute(ctx)

    def precompute(self, ctx: Ctx) -> Path:
        """SAM features + Bezier targets for every frame of this run. ~4 MB of SAM features per frame, so they
        go to local scratch (/tmp on Kaggle; /kaggle/working is capped at 20 GB) and the work folder links to it."""
        from .bcrnet_prep import write_frames
        target = Path(os.environ.get("L3D_PRECOMP_DIR", "/tmp/l3d_bcrnet_precomputed" if Path("/kaggle").exists()
                                     else str(ctx.work / "precomputed_data")))
        target.mkdir(parents=True, exist_ok=True)
        link = ctx.work / "precomputed"
        if not link.exists():
            link.symlink_to(target, target_is_directory=True)
        ids = list(dict.fromkeys(ctx.ids["train"] + ctx.ids["select"] + ctx.ids["heldout"]))
        frames = write_frames(ctx.work / f"bcrnet_frames_{ctx.sync_id}.tsv", ctx.index, ids)
        t = time.time()
        sh([sys.executable, "-m", "l3d_bench.baselines.bcrnet_prep", "beziers", "--frames", str(frames),
            "--repo", str(ctx.repo), "--out", str(link / "s_bezier")], cwd=REPO_ROOT)
        sh([sys.executable, "-m", "l3d_bench.baselines.bcrnet_prep", "sam", "--frames", str(frames),
            "--checkpoint", str(sam_vitb(ctx)), "--out", str(link / "sam"),
            "--device", "cuda" if _cuda() else "cpu"], cwd=REPO_ROOT)
        ctx.log(f"BCRNet inputs ready for {len(ids)} frames ({(time.time() - t) / 60:.1f} min)")
        return link

    def train_cmd(self, ctx: Ctx):
        return ([sys.executable, "train.py", "--data_path", f"{ctx.root}/", "--epoch", str(self.epochs),
                 "--bs", os.environ.get("L3D_BATCH_SIZE", str(self.batch)), "--save_path", str(ctx.run_dir)],
                ctx.repo, {})

    def checkpoints(self, ctx: Ctx) -> dict:
        """BCRNet's code selects nothing: it saves epoch-N.pt every 10 epochs (and, with patch 02, at the last
        epoch) and logs val Dice to wandb. We score every saved checkpoint on the selection set with our
        metric (model.eval()) and keep the best, earliest on ties. On the official split the 'native'
        reproduction number is the last checkpoint run as the released test.py runs it (train mode)."""
        import json

        from ..eval.score import score_folder
        from .runner import write_list
        cands = sorted(ctx.run_dir.glob("epoch-*.pt"), key=lambda p: int(p.stem.split("-")[1]))
        if not cands:
            raise FileNotFoundError(f"no epoch-*.pt in {ctx.run_dir}")
        path = ctx.run_dir / "selection.json"
        sel = json.loads(path.read_text()) if path.exists() else {}
        lst = write_list(ctx, "selection", ctx.ids["select"])
        for ck in cands:
            if ck.name in sel:
                continue
            out = ctx.work / "selection_preds" / ck.stem
            if out.exists():
                shutil.rmtree(out)
            sh(self._export(ctx, ck, lst, out, ctx.work / f"selection_{ck.stem}_native.json", "eval"), cwd=ctx.repo)
            frames = score_folder(out, ctx.index, ctx.ids["select"], None, ctx.cfg["metric"])
            sel[ck.name] = {"epoch": int(ck.stem.split("-")[1]), "select_dice": float(frames["dice"].mean())}
            path.write_text(json.dumps(sel, indent=2))
            ctx.log(f"selection: {ck.name} select Dice {sel[ck.name]['select_dice']:.4f}")
        if ctx.drive is not None and ctx.drive.enabled:
            ctx.drive.push_file(path, ctx.sync_id)
        best = max(cands, key=lambda p: (round(sel[p.name]["select_dice"], 9), -sel[p.name]["epoch"]))
        ctx.log(f"selected {best.name} (select Dice {sel[best.name]['select_dice']:.4f})")
        ck = {"val": best}
        if ctx.split == "official":
            ck["native"] = cands[-1]
        return ck

    def _export(self, ctx: Ctx, ckpt: Path, frame_list: Path, out: Path, native: Path, mode: str):
        return [sys.executable, self.export_script(), "--model_path", str(ckpt), "--list", str(frame_list),
                "--precomputed", str(ctx.work / "precomputed"), "--out", str(out), "--native_json", str(native),
                "--mode", mode]

    def export_cmd(self, ctx: Ctx, ckpt: Path, frame_list: Path, out: Path, native: Path):
        return self._export(ctx, ckpt, frame_list, out, native, "train" if ctx.current_tag == "native" else "eval")


def _cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def sh_out(cmd, cwd=None) -> str:
    import subprocess
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if res.returncode != 0:
        raise RuntimeError(f"command failed ({res.returncode}): {cmd}\n{res.stderr[-2000:]}")
    return res.stdout.strip()


SPECS = {s.name: s for s in (TopoNet(), A2ONet(), D2GPLand(), BCRNet())}
