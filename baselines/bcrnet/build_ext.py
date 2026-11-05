"""Build BCRNet's multi-scale deformable attention op (adet._C) and check it against the PyTorch reference.

The repo ships the CUDA sources (adet/layers/csrc, from AdelaiDet / Deformable DETR) but no setup.py. We JIT-
compile them with torch.utils.cpp_extension, copy the module to adet/_C.so, then compare its forward and
backward with ms_deform_attn_core_pytorch (the reference implementation in the same file) on random inputs.
If the build or the check fails, adet/_C.so is removed and our patch 02 makes MSDeformAttn use the PyTorch
reference instead: the same computation, slower. The result is printed as one JSON line.

    cd <BCRNet clone> && python build_ext.py
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time

CHECK = r"""
import torch, json
from adet import _C
from adet.layers.ms_deform_attn import _MSDeformAttnFunction, ms_deform_attn_core_pytorch
torch.manual_seed(0)
dev = "cuda"
shapes = torch.as_tensor([[32, 32], [16, 16], [8, 8], [4, 4]], dtype=torch.long, device=dev)
start = torch.cat([shapes.new_zeros(1), (shapes[:, 0] * shapes[:, 1]).cumsum(0)[:-1]])
S = int((shapes[:, 0] * shapes[:, 1]).sum())
N, M, D, Lq, L, P = 2, 8, 32, 50, 4, 4
value = torch.rand(N, S, M, D, device=dev, dtype=torch.double, requires_grad=True)
loc = torch.rand(N, Lq, M, L, P, 2, device=dev, dtype=torch.double, requires_grad=True)
w = torch.rand(N, Lq, M, L, P, device=dev, dtype=torch.double) + 1e-5
w = (w / w.sum(-1, keepdim=True).sum(-2, keepdim=True)).requires_grad_()
out_c = _MSDeformAttnFunction.apply(value, shapes, start, loc, w, 64)
out_r = ms_deform_attn_core_pytorch(value, shapes, loc, w)
g = torch.rand_like(out_c)
gc = torch.autograd.grad(out_c, (value, loc, w), g)
gr = torch.autograd.grad(out_r, (value, loc, w), g)
err = max([float((out_c - out_r).abs().max())] + [float((a - b).abs().max()) for a, b in zip(gc, gr)])
print(json.dumps({"max_abs_err": err}))
"""


def main() -> int:
    repo = os.getcwd()
    csrc = os.path.join(repo, "adet", "layers", "csrc")
    dst = os.path.join(repo, "adet", "_C.so")
    t = time.time()
    info = {"adet_C": "fallback (PyTorch reference)"}
    try:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME, load
        if not torch.cuda.is_available() or CUDA_HOME is None:
            raise RuntimeError(f"no GPU or no CUDA toolkit (CUDA_HOME={CUDA_HOME})")
        major, minor = torch.cuda.get_device_capability()
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
        build = os.path.join(repo, "build_adet_C")
        os.makedirs(build, exist_ok=True)
        sources = [os.path.join(csrc, "vision.cpp"), os.path.join(csrc, "cuda_version.cu")] + \
            sorted(glob.glob(os.path.join(csrc, "DeformAttn", "*.cpp"))) + \
            sorted(glob.glob(os.path.join(csrc, "DeformAttn", "*.cu")))
        mod = load(name="_C", sources=sources, extra_include_paths=[csrc], build_directory=build,
                   extra_cflags=["-O3", "-DWITH_CUDA"],
                   extra_cuda_cflags=["-O3", "-DWITH_CUDA", "-DCUDA_HAS_FP16=1", "-D__CUDA_NO_HALF_OPERATORS__",
                                      "-D__CUDA_NO_HALF_CONVERSIONS__", "-D__CUDA_NO_HALF2_OPERATORS__"],
                   with_cuda=True, verbose=False)
        shutil.copy(mod.__file__, dst)
        res = subprocess.run([sys.executable, "-c", CHECK], cwd=repo, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError("check failed: " + res.stderr.strip().splitlines()[-1])
        err = json.loads(res.stdout.strip().splitlines()[-1])["max_abs_err"]
        info["max_abs_err_vs_reference"] = err
        if err > 1e-6:
            raise RuntimeError(f"CUDA op differs from the PyTorch reference (max abs err {err:.2e})")
        info["adet_C"] = "built (CUDA)"
    except Exception as e:  # noqa: BLE001
        if os.path.exists(dst):
            os.remove(dst)
        info["error"] = f"{type(e).__name__}: {e}"[:500]
    info["minutes"] = round((time.time() - t) / 60, 1)
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    sys.exit(main())
