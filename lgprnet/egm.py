"""Edge Guidance Module (EGM). See THIRD_PARTY.md for source credits."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import Conv1x1, ConvBNReLU


def _sobel_filters(in_channels, out_channels=1):
    kernel_x = torch.tensor(
        [[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32,
    )
    kernel_y = torch.tensor(
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32,
    )
    conv_x = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
    conv_y = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
    conv_x.weight = nn.Parameter(
        kernel_x.reshape(1, 1, 3, 3).repeat(out_channels, in_channels, 1, 1),
        requires_grad=False,
    )
    conv_y.weight = nn.Parameter(
        kernel_y.reshape(1, 1, 3, 3).repeat(out_channels, in_channels, 1, 1),
        requires_grad=False,
    )
    return (
        nn.Sequential(conv_x, nn.BatchNorm2d(out_channels)),
        nn.Sequential(conv_y, nn.BatchNorm2d(out_channels)),
    )


class SpatialAttention(nn.Module):
    """Spatial recalibration using channel-wise maximum responses."""

    def __init__(self, kernel_size=7):
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError("kernel_size must be 3 or 7")
        self.conv1 = nn.Conv2d(
            1, 1, kernel_size, padding=kernel_size // 2, bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        spatial_map = torch.max(x, dim=1, keepdim=True).values
        return self.sigmoid(self.conv1(spatial_map))


class EGM(nn.Module):
    """Generate boundary logits from shallow CNN and deep Transformer features.

    x1: [N, 256, H/4, W/4]; t4: [N, 512, H/32, W/32].
    Apply sigmoid to the output to obtain the boundary prior B.
    """

    def __init__(self):
        super().__init__()
        self.reduce1 = Conv1x1(256, 64)
        self.sobel_x, self.sobel_y = _sobel_filters(256)
        self.reduce2 = Conv1x1(512, 256)
        self.conv = ConvBNReLU(256, 256)
        self.conv_out = nn.Conv2d(256, 1, 1)
        self.block = nn.Sequential(
            ConvBNReLU(320, 256),
            ConvBNReLU(256, 256),
        )
        self.sa = SpatialAttention()

    def forward(self, x1, t4):
        gx = self.sobel_x(x1)
        gy = self.sobel_y(x1)
        structural_feature = torch.sigmoid(torch.sqrt(gx.square() + gy.square())) * x1
        structural_feature = self.reduce1(structural_feature)
        semantic_feature = self.reduce2(t4)
        semantic_feature = F.interpolate(
            semantic_feature, size=x1.shape[-2:],
            mode="bilinear", align_corners=False,
        )
        fused = self.block(torch.cat((semantic_feature, structural_feature), dim=1))
        attended = self.sa(fused) * fused
        return self.conv_out(self.conv(attended) + fused)
