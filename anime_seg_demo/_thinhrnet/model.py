from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F
import timm

from .ocr import OCRModule
from .pointrend import PointRendHead, point_sample, sample_uncertain_points


class ThinHRNetOCR(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        pretrained: bool = True,
        ocr_mid_channels: int = 256,
        ocr_key_channels: int = 128,
        use_pointrend: bool = True,
        pointrend_points: int = 4096,
        pointrend_topk_ratio: float = 0.75,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            pretrained=pretrained,
        )
        channels = self.backbone.feature_info.channels()
        high_channels = channels[0]
        mid_channels = channels[1]

        self.high_proj = nn.Sequential(
            nn.Conv2d(high_channels, ocr_mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(ocr_mid_channels),
            nn.ReLU(inplace=True),
        )
        self.mid_proj = nn.Sequential(
            nn.Conv2d(mid_channels, ocr_mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(ocr_mid_channels),
            nn.ReLU(inplace=True),
        )

        self.aux_head = nn.Conv2d(ocr_mid_channels, num_classes, kernel_size=1)
        self.ocr = OCRModule(ocr_mid_channels, ocr_key_channels, ocr_mid_channels, num_classes)
        self.classifier = nn.Conv2d(ocr_mid_channels, num_classes, kernel_size=1)

        self.use_pointrend = use_pointrend
        self.pointrend_points = pointrend_points
        self.pointrend_topk_ratio = pointrend_topk_ratio
        if use_pointrend:
            self.pointrend_head = PointRendHead(ocr_mid_channels + num_classes, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.backbone(x)
        high = self.high_proj(feats[0])
        mid = self.mid_proj(feats[1])

        aux_logits = self.aux_head(mid)
        aux_probs = F.interpolate(aux_logits, size=high.shape[-2:], mode="bilinear", align_corners=False)
        ocr_out = self.ocr(high, aux_probs)
        logits = self.classifier(ocr_out)

        # Ensure float32 for numerical stability
        logits = logits.float()
        aux_logits = aux_logits.float()

        outputs: Dict[str, torch.Tensor] = {
            "logits": logits,
            "aux_logits": aux_logits,
            "features": high,
        }

        if self.use_pointrend:
            probs = torch.softmax(logits, dim=1)
            points, point_indices = sample_uncertain_points(
                probs, self.pointrend_points, self.pointrend_topk_ratio
            )
            point_feats = point_sample(high, points)
            point_logits = point_sample(logits, points)
            point_inputs = torch.cat([point_feats, point_logits], dim=1)
            refined_logits = self.pointrend_head(point_inputs)
            outputs.update(
                {
                    "point_logits": refined_logits.float(),
                    "point_indices": point_indices,
                    "point_coords": points,
                }
            )

        return outputs
