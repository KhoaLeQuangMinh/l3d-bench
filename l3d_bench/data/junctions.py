"""Four curve-intersection points derived by rule from the official polylines (M2 training targets only).

Never hand-placed, never scored, never given anatomical names in the paper.

    0 lig_sil   ligament endpoint nearest the silhouette (pair midpoint)
    1 lig_ridge ligament endpoint nearest the ridge (pair midpoint)
    2 lateral_a ridge/silhouette meeting point with the smaller x of the two lateral points
    3 lateral_b the other lateral point

Fixes over the old junction_extractor.py: works in native pixels with an isotropic threshold scaled by
H / 1080 (the old one scaled x and y differently), and splits the two lateral points by their order
relative to each other instead of a fixed cut at 45% of the image width (which failed whenever the
liver was shifted or raised). A horizontal flip swaps lateral_a and lateral_b (handled in dataset.py).
"""
from __future__ import annotations

import numpy as np

from .annotations import Annotation

RIDGE, SIL, LIG = 1, 2, 3
NAMES = ["lig_sil", "lig_ridge", "lateral_a", "lateral_b"]


def _densify(poly: np.ndarray, step: float = 2.0) -> np.ndarray:
    pts = [poly[0]]
    for a, b in zip(poly[:-1], poly[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        for t in np.linspace(0, 1, n + 1)[1:]:
            pts.append(a + t * (b - a))
    return np.asarray(pts, dtype=np.float32)


def _nearest(endpoints, curves):
    """Best (distance, midpoint) between any endpoint and any point on any curve."""
    best = (np.inf, None)
    for ep in endpoints:
        for c in curves:
            d = np.linalg.norm(c - ep, axis=1)
            j = int(np.argmin(d))
            if d[j] < best[0]:
                best = (float(d[j]), (ep + c[j]) / 2.0)
    return best


def derive_junctions(ann: Annotation, threshold_at_1080: float = 150.0):
    """Returns coords (4, 2) in [0, 1] (x / W, y / H) and visibility (4,) in {0, 1}."""
    thr = threshold_at_1080 * ann.height / 1080.0
    ridge = [_densify(p) for p in ann.polylines(RIDGE)]
    sil = [_densify(p) for p in ann.polylines(SIL)]
    lig = ann.polylines(LIG)
    coords = np.zeros((4, 2), dtype=np.float32)
    vis = np.zeros(4, dtype=np.float32)

    lig_ends = [p[i] for p in lig for i in (0, -1)]
    for slot, curves in ((0, sil), (1, ridge)):
        if lig_ends and curves:
            d, mid = _nearest(lig_ends, curves)
            if d <= thr:
                coords[slot], vis[slot] = mid, 1.0

    lateral = []
    if ridge and sil:
        ridge_ends = [p[i] for p in ann.polylines(RIDGE) for i in (0, -1)]
        sil_ends = [p[i] for p in ann.polylines(SIL) for i in (0, -1)]
        cand = []
        for ep in ridge_ends:
            d, mid = _nearest([ep], sil)
            if d <= thr:
                cand.append((d, mid))
        for ep in sil_ends:
            d, mid = _nearest([ep], ridge)
            if d <= thr:
                cand.append((d, mid))
        cand.sort(key=lambda t: t[0])
        for d, mid in cand:  # keep the two best, distinct points
            if all(np.linalg.norm(mid - m) > thr for m in lateral):
                lateral.append(mid)
            if len(lateral) == 2:
                break
    lateral.sort(key=lambda m: float(m[0]))
    if len(lateral) == 2:
        coords[2], coords[3], vis[2], vis[3] = lateral[0], lateral[1], 1.0, 1.0
    elif len(lateral) == 1:
        ref_x = coords[1][0] if vis[1] else ann.width / 2.0
        slot = 2 if lateral[0][0] < ref_x else 3
        coords[slot], vis[slot] = lateral[0], 1.0

    coords[:, 0] /= ann.width
    coords[:, 1] /= ann.height
    return np.clip(coords, 0.0, 1.0), vis


def hflip_junctions(coords: np.ndarray, vis: np.ndarray):
    """Mirror x and swap the two lateral slots."""
    c, v = coords.copy(), vis.copy()
    c[:, 0] = np.where(v > 0, 1.0 - c[:, 0], 0.0)
    c[[2, 3]] = c[[3, 2]]
    v[[2, 3]] = v[[3, 2]]
    return c, v
