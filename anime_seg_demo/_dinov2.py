"""DINOv2/U-Net++ inference math reused from AnimeSeg."""
from typing import List, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from transformers import Dinov2Config, Dinov2Model

class ChannelAttention(nn.Module):
    """Channel Attention Module (from CBAM)."""

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, padding=0),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SpatialAttention(nn.Module):
    """Spatial Attention Module (from CBAM), useful for boundary localization."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(x_cat))
        return x * out

class ConvBlock(nn.Module):
    """Standard Conv Block: Conv -> BN -> ReLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.gn = nn.GroupNorm(32, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        return x

class UpsampleBlock(nn.Module):
    """Upsampling block: Bilinear + Conv (no ConvTranspose2d artifacts)."""

    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = ConvBlock(in_channels, out_channels, kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bilinear upsampling
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        # Convolution
        x = self.conv(x)
        return x

class PyramidFeatureFusion(nn.Module):
    """Fuse multi-scale features using pyramid structure."""

    def __init__(self, in_channels_list: List[int], out_channels: int):
        super().__init__()
        # Reduce all inputs to out_channels
        self.reduces = nn.ModuleList([
            nn.Conv2d(c, out_channels, kernel_size=1, padding=0) for c in in_channels_list
        ])
        # Fusion conv
        self.fusion = ConvBlock(out_channels * len(in_channels_list), out_channels)

    def forward(self, features: List[torch.Tensor], target_size: Tuple[int, int]) -> torch.Tensor:
        """
        Args:
            features: List of feature tensors at different resolutions
            target_size: Target size to resize all features to

        Returns:
            Fused feature tensor at target_size
        """
        reduced = []
        for feat, reduce in zip(features, self.reduces):
            # Resize to target size
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            feat = reduce(feat)
            reduced.append(feat)

        # Concatenate and fuse
        x = torch.cat(reduced, dim=1)
        x = self.fusion(x)
        return x

class UNetPlusPlusDecoder(nn.Module):
    """
    U-Net++ style decoder with skip connections and pyramid fusion.

    Supports HRNet-style multi-scale outputs.
    All skip connections are preserved (no resolution loss).
    """

    def __init__(
        self,
        num_classes: int,
        backbone_embed_dim: int = 768,
        decoder_channels: int = 256,
        use_attention: bool = True,
    ):
        super().__init__()

        self.embed_dim = backbone_embed_dim
        self.decoder_channels = decoder_channels

        # ========== Feature Reduction ==========
        # Reduce each encoder layer to decoder_channels
        self.reduce_layers = nn.ModuleDict({
            'l0': nn.Sequential(
                nn.Conv2d(backbone_embed_dim, decoder_channels, kernel_size=1),
                nn.GroupNorm(32, decoder_channels),
                nn.ReLU(inplace=True),
            ),
            'l6': nn.Sequential(
                nn.Conv2d(backbone_embed_dim, decoder_channels, kernel_size=1),
                nn.GroupNorm(32, decoder_channels),
                nn.ReLU(inplace=True),
            ),
            'l9': nn.Sequential(
                nn.Conv2d(backbone_embed_dim, decoder_channels, kernel_size=1),
                nn.GroupNorm(32, decoder_channels),
                nn.ReLU(inplace=True),
            ),
            'l11': nn.Sequential(
                nn.Conv2d(backbone_embed_dim, decoder_channels, kernel_size=1),
                nn.GroupNorm(32, decoder_channels),
                nn.ReLU(inplace=True),
            ),
        })

        # ========== Multi-scale Fusion at Base ==========
        self.base_fusion = PyramidFeatureFusion(
            in_channels_list=[decoder_channels] * 4,
            out_channels=decoder_channels,
        )

        # ========== Decoder Levels (36x36 -> 512x512) ==========
        # Level 1: 36x36 -> 72x72
        self.dec_level1 = nn.ModuleDict({
            'upsample': UpsampleBlock(decoder_channels, decoder_channels, scale_factor=2),
            'skip_conv': ConvBlock(decoder_channels, decoder_channels),
            'fusion': ConvBlock(decoder_channels * 2, decoder_channels),
            'attn': ChannelAttention(decoder_channels) if use_attention else nn.Identity(),
        })

        # Level 2: 72x72 -> 144x144
        self.dec_level2 = nn.ModuleDict({
            'upsample': UpsampleBlock(decoder_channels, decoder_channels, scale_factor=2),
            'skip_conv': ConvBlock(decoder_channels, decoder_channels),
            'fusion': ConvBlock(decoder_channels * 2, decoder_channels),
            'attn': ChannelAttention(decoder_channels) if use_attention else nn.Identity(),
        })

        # Level 3: 144x144 -> 288x288
        self.dec_level3 = nn.ModuleDict({
            'upsample': UpsampleBlock(decoder_channels, decoder_channels, scale_factor=2),
            'skip_conv': ConvBlock(decoder_channels, decoder_channels),
            'fusion': ConvBlock(decoder_channels * 2, decoder_channels),
            'attn': ChannelAttention(decoder_channels) if use_attention else nn.Identity(),
        })

        # Level 4: 288x288 -> 512x512
        self.dec_level4 = nn.ModuleDict({
            'upsample': UpsampleBlock(decoder_channels, decoder_channels, scale_factor=2),
            'spatial_attn': SpatialAttention() if use_attention else nn.Identity(),
        })

        # ========== Output Head ==========
        self.head = nn.Sequential(
            ConvBlock(decoder_channels, decoder_channels // 2),
            ConvBlock(decoder_channels // 2, 64),
            nn.Conv2d(64, num_classes, kernel_size=1, padding=0),
        )

    def forward(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            hidden_states: List of 4 feature tensors [l0, l6, l9, l11]
                          Each shape: (B, num_tokens, 768)

        Returns:
            logits: (B, num_classes, 512, 512)
        """
        # ========== Reshape and Reduce ==========
        def reshape_features(features: torch.Tensor) -> torch.Tensor:
            """(B, num_tokens, C) -> (B, C, 36, 36)"""
            B, N, C = features.shape
            # Remove class token
            features_no_cls = features[:, 1:, :]  # (B, 1296, C)
            H = W = int((features_no_cls.shape[1]) ** 0.5)  # 36x36
            return features_no_cls.transpose(1, 2).reshape(B, C, H, W)

        # Reshape all 4 encoder layers to (B, 768, 36, 36)
        l0_spatial = reshape_features(hidden_states[0])
        l6_spatial = reshape_features(hidden_states[1])
        l9_spatial = reshape_features(hidden_states[2])
        l11_spatial = reshape_features(hidden_states[3])

        # Reduce to decoder_channels
        l0 = self.reduce_layers['l0'](l0_spatial)      # (B, 256, 36, 36)
        l6 = self.reduce_layers['l6'](l6_spatial)      # (B, 256, 36, 36)
        l9 = self.reduce_layers['l9'](l9_spatial)      # (B, 256, 36, 36)
        l11 = self.reduce_layers['l11'](l11_spatial)   # (B, 256, 36, 36)

        # ========== Multi-scale Fusion at Base ==========
        x = self.base_fusion([l0, l6, l9, l11], target_size=(36, 36))  # (B, 256, 36, 36)

        # ========== Level 1: 36x36 -> 72x72 ==========
        x = self.dec_level1['upsample'](x)  # (B, 256, 72, 72)
        # Skip connection with l6 (if needed, resize l6)
        skip = F.interpolate(l6, size=(72, 72), mode='bilinear', align_corners=False) if l6.shape[-2:] != (72, 72) else l6
        skip = self.dec_level1['skip_conv'](skip)
        x = torch.cat([x, skip], dim=1)
        x = self.dec_level1['fusion'](x)  # (B, 256, 72, 72)
        x = self.dec_level1['attn'](x)

        # ========== Level 2: 72x72 -> 144x144 ==========
        x = self.dec_level2['upsample'](x)  # (B, 256, 144, 144)
        # Skip connection with l9
        skip = F.interpolate(l9, size=(144, 144), mode='bilinear', align_corners=False) if l9.shape[-2:] != (144, 144) else l9
        skip = self.dec_level2['skip_conv'](skip)
        x = torch.cat([x, skip], dim=1)
        x = self.dec_level2['fusion'](x)  # (B, 256, 144, 144)
        x = self.dec_level2['attn'](x)

        # ========== Level 3: 144x144 -> 288x288 ==========
        x = self.dec_level3['upsample'](x)  # (B, 256, 288, 288)
        # Skip connection with l0
        skip = F.interpolate(l0, size=(288, 288), mode='bilinear', align_corners=False) if l0.shape[-2:] != (288, 288) else l0
        skip = self.dec_level3['skip_conv'](skip)
        x = torch.cat([x, skip], dim=1)
        x = self.dec_level3['fusion'](x)  # (B, 256, 288, 288)
        x = self.dec_level3['attn'](x)

        # ========== Level 4: 288x288 -> 512x512 ==========
        x = self.dec_level4['upsample'](x)  # (B, 256, 512, 512)
        x = self.dec_level4['spatial_attn'](x)

        # ========== Output Head ==========
        logits = self.head(x)  # (B, num_classes, 512, 512)

        return logits

class DINOv2ForSegmentation(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_classes = config["num_labels"]
        self.backbone = Dinov2Model(Dinov2Config.from_dict(config["backbone_config"]))
        self.decoder = UNetPlusPlusDecoder(
            num_classes=self.num_classes,
            backbone_embed_dim=self.backbone.config.hidden_size,
            **config["decoder_config"],
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, 3, H, W) normalized to [0, 1]

        Returns:
            logits: (B, num_classes, 512, 512)
        """
        # Get backbone features
        if hasattr(self.backbone, 'base_model'):
            outputs = self.backbone.base_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
            )
        else:
            outputs = self.backbone(
                pixel_values=pixel_values,
                output_hidden_states=True,
            )

        hidden_states = outputs.hidden_states
        num_layers = len(hidden_states)

        # Select key layers
        if num_layers == 12:
            indices = [0, 6, 9, 11]
        elif num_layers == 24:
            indices = [0, 12, 18, 23]
        elif num_layers == 40:
            indices = [0, 20, 30, 39]
        else:
            indices = [0, num_layers // 2, (num_layers * 3) // 4, num_layers - 1]

        selected_states = [hidden_states[i] for i in indices]

        # Decoder
        logits = self.decoder(selected_states)

        return logits
