"""M2: M1 + junction steering. The only change from M1 is this module; data, loss on masks, schedule and
selection are identical.

    anchor head   4 learnable queries, 2 decoder layers (self-attention among the 4, cross-attention to the
                  stride-16 pixel-decoder features, FFN) -> (x, y) = sigmoid(MLP), visibility = MLP logit
    steering      Mask2Former's 100 query features cross-attend to the 4 anchor tokens before the masked-
                  attention decoder:  Q <- LayerNorm(Q + alpha * dQ),  alpha learnable, starts at 0.1
    loss          Mask2Former loss + 5.0 * SmoothL1(coords, visible anchors only) + 1.0 * BCE(visibility)

Ported from your Exp 5 (JunctionAnchorHead, JunctionQuerySteering, loss weights). Two changes:
  * targets are the rule-derived curve-intersection points (data/junctions.py), never hand-placed anchors;
  * no anatomical coordinate priors in the head (Exp 5 initialised each slot at a hand-chosen position);
    every slot starts at the image centre and is learned.

Implementation: hooks on the Hugging Face modules, so Mask2Former's own forward, Hungarian loss and
auxiliary losses run unchanged. A pre-hook on the transformer module runs the anchor head on the
pixel-decoder features; a pre-hook on the decoder replaces its input query features by the steered ones.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .mask2former import Mask2FormerSegmenter

STRIDE16 = 1  # multi_scale_features from the pixel decoder are ordered 1/32, 1/16, 1/8


class JunctionDecoderLayer(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * dim, dim),
                                 nn.Dropout(dropout))
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, q, context):
        x = self.norm1(q)
        q = q + self.drop(self.self_attn(x, x, x, need_weights=False)[0])
        x = self.norm2(q)
        q = q + self.drop(self.cross_attn(x, context, context, need_weights=False)[0])
        return q + self.ffn(self.norm3(q))


class JunctionAnchorHead(nn.Module):
    def __init__(self, dim: int = 256, heads: int = 8, layers: int = 2, num_anchors: int = 4):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_anchors, dim) * 0.02)
        self.layers = nn.ModuleList(JunctionDecoderLayer(dim, heads) for _ in range(layers))
        self.coord = nn.Sequential(nn.Linear(dim, 128), nn.GELU(), nn.Linear(128, 2))
        self.vis = nn.Sequential(nn.Linear(dim, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, feat):
        """feat (B, C, H, W) -> coords (B, 4, 2) in [0, 1] as (x, y), vis logits (B, 4), tokens (B, 4, C)."""
        context = feat.flatten(2).transpose(1, 2)
        q = self.queries.expand(feat.shape[0], -1, -1)
        for layer in self.layers:
            q = layer(q, context)
        return self.coord(q).sigmoid(), self.vis(q).squeeze(-1), q


class QuerySteering(nn.Module):
    def __init__(self, dim: int = 256, heads: int = 8, gate_init: float = 0.1, dropout: float = 0.1):
        super().__init__()
        self.proj_q, self.proj_k, self.proj_v = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, queries, anchors):
        """queries (B, Q, C), anchors (B, 4, C) -> steered (B, Q, C)."""
        dq = self.attn(self.proj_q(queries), self.proj_k(anchors), self.proj_v(anchors), need_weights=False)[0]
        return self.norm(queries + self.gate * self.drop(dq))


class Mask2FormerJunction(Mask2FormerSegmenter):
    uses_junctions = True

    def __init__(self, backbone: str, gate_init: float = 0.1, w_coord: float = 5.0, w_vis: float = 1.0):
        super().__init__(backbone)
        cfg = self.m2f.config
        dim, heads = cfg.hidden_dim, cfg.num_attention_heads
        if cfg.mask_feature_size != dim:
            raise ValueError("anchor head assumes pixel-decoder width == decoder width")
        self.anchor_head = JunctionAnchorHead(dim, heads)
        self.steering = QuerySteering(dim, heads, gate_init)
        self.w_coord, self.w_vis = w_coord, w_vis
        self._anchor = None
        tm = self.m2f.model.transformer_module
        tm.register_forward_pre_hook(self._run_anchor_head, with_kwargs=True)
        tm.decoder.register_forward_pre_hook(self._steer_queries, with_kwargs=True)

    # ---- hooks ---------------------------------------------------------------------------------
    def _run_anchor_head(self, module, args, kwargs):
        feats = kwargs["multi_scale_features"] if "multi_scale_features" in kwargs else args[0]
        self._anchor = self.anchor_head(feats[STRIDE16])
        return None

    def _steer_queries(self, module, args, kwargs):
        q = kwargs["inputs_embeds"]                                # (Q, B, C)
        anchors = self._anchor[2].to(q.dtype)
        kwargs["inputs_embeds"] = self.steering(q.transpose(0, 1), anchors).transpose(0, 1)
        return args, kwargs

    # ---- forward -------------------------------------------------------------------------------
    def forward(self, images, labels=None, junctions=None, junction_vis=None):
        out = super().forward(images, labels)
        coords, vis_logits, _ = self._anchor
        out.junction_coords, out.junction_vis_logits = coords, vis_logits
        self._anchor = None
        if labels is not None:
            if junctions is None or junction_vis is None:
                raise ValueError("M2 training needs junction targets")
            vis_loss = F.binary_cross_entropy_with_logits(vis_logits.float(), junction_vis.float())
            keep = junction_vis > 0.5
            coord_loss = (F.smooth_l1_loss(coords.float()[keep], junctions.float()[keep]) if keep.any()
                          else coords.sum() * 0.0)
            out.mask_loss = out.loss.detach()
            out.coord_loss, out.vis_loss = coord_loss.detach(), vis_loss.detach()
            out.loss = out.loss + self.w_coord * coord_loss + self.w_vis * vis_loss
        return out
