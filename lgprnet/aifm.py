"""Attention-based Interactive Fusion Module (AIFM)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import SGA
from .common import Conv1x1, ConvBNReLU


class AIFM(nn.Module):
    """Refine features with region and boundary priors.

    Inputs are the current-level feature, high-level feature, boundary prior B,
    and current region prior. Priors are probabilities, not raw logits. This
    module returns a fused feature; the prediction head is separate.
    """

    def __init__(self, high_channels, current_channels, out_channels):
        super().__init__()
        self.high_projection = Conv1x1(high_channels, out_channels)
        self.current_projection = Conv1x1(current_channels, out_channels)
        self.boundary_guidance = SGA(dim=out_channels, num_heads=8)
        self.region_guidance = SGA(dim=out_channels, num_heads=8)
        self.fusion = ConvBNReLU(2 * out_channels, out_channels)

    def forward(self, current_feature, high_feature, boundary_prior, region_prior):
        size = current_feature.shape[-2:]
        if high_feature.shape[-2:] != size:
            high_feature = F.interpolate(
                high_feature, size=size, mode="bilinear", align_corners=False,
            )
        if boundary_prior.shape[-2:] != size:
            boundary_prior = F.interpolate(
                boundary_prior, size=size, mode="bilinear", align_corners=False,
            )
        if region_prior.shape[-2:] != size:
            region_prior = F.interpolate(
                region_prior, size=size, mode="bilinear", align_corners=False,
            )
        high_feature = self.high_projection(high_feature)
        current_feature = self.current_projection(current_feature)
        region_feature = self.region_guidance(high_feature, region_prior)
        boundary_feature = self.boundary_guidance(current_feature, boundary_prior)
        return self.fusion(torch.cat((region_feature, boundary_feature), dim=1))
