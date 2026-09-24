"""Historical MobileNetV2/AFS U-Net runtime."""
import torch
from torch import nn
import torchvision

CLASS_NAMES = ('background', 'skin', 'face', 'hair', 'eye_left', 'eye_right', 'eyebrow_left', 'eyebrow_right', 'nose', 'mouth', 'clothes', 'accessory')

class AFSModel(nn.Module):
    """MobileNetV2 encoder with the historical five-stage AFS decoder."""

    def __init__(self, num_classes: int = len(CLASS_NAMES)) -> None:
        super().__init__()
        if num_classes != len(CLASS_NAMES):
            raise ValueError(f"this archive requires {len(CLASS_NAMES)} classes")
        mobile_blocks = torchvision.models.mobilenet_v2(weights=None).features

        self.en_block0 = nn.Sequential(mobile_blocks[0], mobile_blocks[1])
        self.en_block1 = nn.Sequential(mobile_blocks[2], mobile_blocks[3])
        self.en_block2 = nn.Sequential(
            mobile_blocks[4], mobile_blocks[5], mobile_blocks[6]
        )
        self.en_block3 = nn.Sequential(
            mobile_blocks[7],
            mobile_blocks[8],
            mobile_blocks[9],
            mobile_blocks[10],
            mobile_blocks[11],
            mobile_blocks[12],
            mobile_blocks[13],
        )
        self.en_block4 = nn.Sequential(
            mobile_blocks[14], mobile_blocks[15], mobile_blocks[16]
        )

        self.de_block4 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(160, 96, kernel_size=3, padding=1),
            nn.InstanceNorm2d(96),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.2),
        )
        self.de_block3 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(96 * 2, 32, kernel_size=3, padding=1),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.2),
        )
        self.de_block2 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(32 * 2, 24, kernel_size=3, padding=1),
            nn.InstanceNorm2d(24),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.2),
        )
        self.de_block1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(24 * 2, 16, kernel_size=3, padding=1),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.2),
        )
        self.de_block0 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(16 * 2, num_classes, kernel_size=3, padding=1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        e0 = self.en_block0(image)
        e1 = self.en_block1(e0)
        e2 = self.en_block2(e1)
        e3 = self.en_block3(e2)
        e4 = self.en_block4(e3)

        d4 = self.de_block4(e4)
        d3 = self.de_block3(torch.cat((d4, e3), dim=1))
        d2 = self.de_block2(torch.cat((d3, e2), dim=1))
        d1 = self.de_block1(torch.cat((d2, e1), dim=1))
        return self.de_block0(torch.cat((d1, e0), dim=1))
