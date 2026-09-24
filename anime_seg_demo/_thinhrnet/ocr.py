from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class SpatialGather(nn.Module):
    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, feats: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feats.shape
        _, k, _, _ = probs.shape
        feats = feats.view(b, c, -1)
        probs = probs.view(b, k, -1)
        probs = F.softmax(self.scale * probs, dim=2)
        context = torch.bmm(feats, probs.transpose(1, 2))
        context = context.unsqueeze(3)
        return context


class ObjectAttentionBlock(nn.Module):
    def __init__(self, in_channels: int, key_channels: int, out_channels: int) -> None:
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, key_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, key_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.out_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.scale = key_channels ** -0.5

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        query = self.query_conv(x).view(b, -1, h * w).permute(0, 2, 1)
        key = self.key_conv(context).view(b, -1, context.shape[2] * context.shape[3])
        value = self.value_conv(context).view(b, -1, context.shape[2] * context.shape[3])

        sim_map = torch.bmm(query, key) * self.scale
        attn = F.softmax(sim_map, dim=-1)
        out = torch.bmm(value, attn.permute(0, 2, 1))
        out = out.view(b, -1, h, w)
        return self.out_conv(out)


class OCRModule(nn.Module):
    def __init__(self, in_channels: int, key_channels: int, out_channels: int, num_classes: int) -> None:
        super().__init__()
        self.gather = SpatialGather()
        self.attention = ObjectAttentionBlock(in_channels, key_channels, out_channels)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.num_classes = num_classes

    def forward(self, feats: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        context = self.gather(feats, probs)
        context = self.attention(feats, context)
        out = torch.cat([feats, context], dim=1)
        return self.conv(out)
