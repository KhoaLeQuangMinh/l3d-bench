"""M1: Hugging Face Mask2Former (Swin-T, ADE20k semantic weights) for 3 landmark classes.

Classes. The model predicts 4 classes: 0 background, 1 ridge, 2 silhouette, 3 ligament (plus Mask2Former's
own "no object"). Background is a real target mask, so the prediction is a plain argmax over the 4
semantic maps and no confidence threshold has to be tuned. (The old Exp 8 code left class 0 untrained and
cut pixels below 0.25 to background; that threshold is gone.)

Loss. Mask2Former's own Hungarian loss, unchanged from the Hugging Face defaults: class CE (weight 2,
no-object 0.1) + mask BCE (5) + mask Dice (5) on 12544 importance-sampled points, on every decoder layer.

Inference. Standard Mask2Former semantic inference:
    S[c, h, w] = sum_q softmax(class_q)[c] * sigmoid(mask_q[h, w])     for c in 0..3 (no-object dropped)
    label[h, w] = argmax_c S[c, h, w]
Masks come out at 1/4 of the input size and are upsampled bilinearly to the input size (1024).

Inputs are float images in [0, 1]; ImageNet normalisation (the checkpoint's preprocessor) happens here.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

NUM_LABELS = 4
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
BACKBONE_PREFIX = "model.pixel_level_module.encoder."
TINY = "tiny-random"  # test-only: a small randomly initialised model, no download


def _tiny_config():
    from transformers import Mask2FormerConfig, SwinConfig
    swin = SwinConfig(image_size=64, embed_dim=16, depths=[1, 1, 1, 1], num_heads=[1, 1, 2, 2], window_size=4,
                      out_features=["stage1", "stage2", "stage3", "stage4"])
    return Mask2FormerConfig(backbone_config=swin, num_labels=NUM_LABELS, feature_size=32, mask_feature_size=32,
                             hidden_dim=32, encoder_layers=1, decoder_layers=3, num_queries=8,
                             encoder_feedforward_dim=64, dim_feedforward=64, num_attention_heads=2,
                             train_num_points=256, use_pretrained_backbone=False)


def label_to_targets(labels: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """(B, H, W) label maps -> per image: binary masks (K, H, W) float and class ids (K,) for the classes present."""
    masks, classes = [], []
    for lab in labels:
        present = torch.unique(lab)
        masks.append(torch.stack([(lab == c) for c in present]).float())
        classes.append(present.long())
    return masks, classes


class Mask2FormerSegmenter(nn.Module):
    def __init__(self, backbone: str = "facebook/mask2former-swin-tiny-ade-semantic"):
        super().__init__()
        from transformers import Mask2FormerForUniversalSegmentation
        id2label = {0: "background", 1: "ridge", 2: "silhouette", 3: "ligament"}
        if backbone == TINY:
            self.m2f = Mask2FormerForUniversalSegmentation(_tiny_config())
        else:
            self.m2f = Mask2FormerForUniversalSegmentation.from_pretrained(
                backbone, num_labels=NUM_LABELS, id2label=id2label, label2id={v: k for k, v in id2label.items()},
                ignore_mismatched_sizes=True)  # only the class head is re-initialised (151 -> 4 + 1)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def param_groups(self, lr_backbone: float, lr_head: float, weight_decay: float) -> list[dict]:
        backbone, head = [], []
        for name, p in self.named_parameters():
            if p.requires_grad:
                (backbone if name.startswith("m2f." + BACKBONE_PREFIX) else head).append(p)
        return [{"params": backbone, "lr": lr_backbone, "weight_decay": weight_decay, "name": "backbone"},
                {"params": head, "lr": lr_head, "weight_decay": weight_decay, "name": "head"}]

    def forward(self, images: torch.Tensor, labels: torch.Tensor | None = None):
        """images (B, 3, H, W) in [0, 1]; labels (B, H, W) long or None. Returns the HF output (with .loss)."""
        x = (images - self.mean) / self.std
        if labels is None:
            return self.m2f(pixel_values=x)
        masks, classes = label_to_targets(labels)
        return self.m2f(pixel_values=x, mask_labels=masks, class_labels=classes)

    @staticmethod
    def semantic_scores(outputs, size: tuple[int, int]) -> torch.Tensor:
        """(B, 4, H, W) semantic scores, computed in float32."""
        cls = outputs.class_queries_logits.float().softmax(-1)[..., :-1]              # (B, Q, 4)
        masks = F.interpolate(outputs.masks_queries_logits.float(), size=size, mode="bilinear",
                              align_corners=False).sigmoid()                            # (B, Q, H, W)
        return torch.einsum("bqc,bqhw->bchw", cls, masks)

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) -> (B, H, W) uint8 label maps at the input size."""
        out = self.forward(images)
        return self.semantic_scores(out, images.shape[-2:]).argmax(1).to(torch.uint8)
