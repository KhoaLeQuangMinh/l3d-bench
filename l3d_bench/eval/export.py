"""Model -> one PNG label map per frame (uint8, 0..3, at the model input size, 1024).

The PNGs are what score.py reads, the same format every baseline's export script writes.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def export_predictions(model, dataset, out_dir: str | Path, device, amp: bool = False, batch_size: int = 1,
                       num_workers: int = 0) -> list[str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.eval()
    device = torch.device(device)
    written = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    for batch in loader:
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            pred = model.predict(batch["image"].to(device)).cpu().numpy()
        for fid, p in zip(batch["frame_id"], pred):
            cv2.imwrite(str(out / f"{fid}.png"), p)
            written.append(fid)
    return written
