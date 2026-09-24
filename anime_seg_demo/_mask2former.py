"""Mask2Former forward/postprocessing reused from AnimeSeg."""
from typing import Dict
import torch
from torch import nn
import torch.nn.functional as F
from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation


class Mask2FormerAnimeSegModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_classes = config["num_labels"]
        self.model = Mask2FormerForUniversalSegmentation(Mask2FormerConfig.from_dict(config))

    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        h, w = pixel_values.shape[-2:]
        outputs = self.model(pixel_values=pixel_values)

        cls_logits = outputs.class_queries_logits
        mask_logits = outputs.masks_queries_logits

        cls_probs = F.softmax(cls_logits, dim=-1)[..., : self.num_classes]
        up_mask_logits = F.interpolate(mask_logits, size=(h, w), mode="bilinear", align_corners=False)
        up_mask_probs = up_mask_logits.sigmoid()

        sem_prob = torch.einsum("bqc,bqhw->bchw", cls_probs, up_mask_probs)
        sem_prob = sem_prob / sem_prob.sum(dim=1, keepdim=True).clamp(min=1e-6)
        sem_logits = torch.log(sem_prob.clamp(min=1e-6))

        return {
            "semantic_logits": sem_logits,
            "query_mask_logits": up_mask_logits,
            "query_part_logits": cls_logits,
        }
