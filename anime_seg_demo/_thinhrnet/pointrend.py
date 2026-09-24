from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


def _points_to_grid(points: torch.Tensor, height: int, width: int) -> torch.Tensor:
    x = (points[..., 0] / (width - 1)) * 2 - 1
    y = (points[..., 1] / (height - 1)) * 2 - 1
    grid = torch.stack([x, y], dim=-1)
    return grid


def sample_uncertain_points(probs: torch.Tensor, num_points: int, topk_ratio: float) -> Tuple[torch.Tensor, torch.Tensor]:
    b, c, h, w = probs.shape
    uncertainty = 1.0 - probs.max(dim=1).values
    flat = uncertainty.view(b, -1)

    topk = int(num_points / topk_ratio)
    topk = min(topk, flat.shape[1])
    _, topk_idx = torch.topk(flat, k=topk, dim=1)

    rand_idx = torch.randint(0, topk, (b, num_points), device=probs.device)
    point_idx = torch.gather(topk_idx, 1, rand_idx)

    y = (point_idx // w).float()
    x = (point_idx % w).float()
    points = torch.stack([x, y], dim=-1)
    return points, point_idx


def point_sample(input_tensor: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    b, c, h, w = input_tensor.shape
    grid = _points_to_grid(points, h, w).unsqueeze(2)
    sampled = F.grid_sample(input_tensor, grid, align_corners=True)
    return sampled.squeeze(3)


class PointRendHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, hidden_channels: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, num_classes, kernel_size=1),
        )

    def forward(self, point_features: torch.Tensor) -> torch.Tensor:
        return self.mlp(point_features)


def scatter_refined_logits(
    coarse_logits: torch.Tensor,
    point_logits: torch.Tensor,
    point_indices: torch.Tensor,
) -> torch.Tensor:
    b, c, h, w = coarse_logits.shape
    refined = coarse_logits.view(b, c, -1).clone()
    idx = point_indices.unsqueeze(1).expand(-1, c, -1)
    refined.scatter_(2, idx, point_logits)
    return refined.view(b, c, h, w)
