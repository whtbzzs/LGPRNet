"""Network assembly for supplied encoders and final DGRM implementations.

This partial release does not supply pretrained encoders or DGRM. The caller
must provide them explicitly; no substitute for the final DGRM is included.
"""

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aifm import AIFM
from .common import Conv1x1, ConvBNReLU
from .egm import EGM


class LGPRNetOutputs(NamedTuple):
    """Full-resolution region logits and boundary probability prior.

    P1 is coarse; P2/P3 are intermediate; P4 is final. Use sigmoid on P4
    for a segmentation probability map. B is already a probability map.
    """

    P1: torch.Tensor
    P2: torch.Tensor
    P3: torch.Tensor
    P4: torch.Tensor
    B: torch.Tensor


class LGPRNet(nn.Module):
    """Assemble EGM, four supplied DGRMs, and three AIFM stages.

    cnn_encoder returns x1..x4 with channels (256, 512, 1024, 2048).
    transformer_encoder returns t1..t4 with channels (64, 128, 320, 512).
    dgrm_stages contains four final modules taking (xi, ti) and returning
    ci with channels (64, 128, 320, 512), respectively. This interface does
    not certify equivalence to the privately trained model/checkpoints.
    """

    def __init__(self, cnn_encoder, transformer_encoder, dgrm_stages):
        super().__init__()
        dgrm_stages = list(dgrm_stages)
        if len(dgrm_stages) != 4:
            raise ValueError("LGPRNet requires four supplied DGRM stages")
        self.cnn_encoder = cnn_encoder
        self.transformer_encoder = transformer_encoder
        self.dgrm_stages = nn.ModuleList(dgrm_stages)
        self.egm = EGM()
        self.reduce1 = Conv1x1(64, 64)
        self.reduce2 = Conv1x1(128, 128)
        self.reduce3 = Conv1x1(320, 256)
        self.reduce4 = Conv1x1(512, 256)
        self.aifm1 = AIFM(256, 256, 256)
        self.aifm2 = AIFM(256, 128, 128)
        self.aifm3 = AIFM(128, 64, 64)
        self.predictor_p2 = nn.Conv2d(256, 1, 1)
        self.predictor_p3 = nn.Conv2d(128, 1, 1)
        self.predictor_p4 = nn.Conv2d(64, 1, 1)
        self.region_predictor = nn.Sequential(
            ConvBNReLU(1024, 256),
            ConvBNReLU(256, 256),
            nn.Conv2d(256, 1, 1),
        )

    def forward(self, image):
        x = tuple(self.cnn_encoder(image))
        t = tuple(self.transformer_encoder(image))
        if len(x) != 4 or len(t) != 4:
            raise ValueError("Each encoder must return four stage features")
        boundary_logits = self.egm(x[0], t[3])
        boundary_prior = boundary_logits.sigmoid()
        c1, c2, c3, c4 = (
            stage(local, global_feature)
            for stage, local, global_feature in zip(self.dgrm_stages, x, t)
        )
        size = c4.shape[-2:]
        aligned = [
            F.interpolate(c, size=size, mode="bilinear", align_corners=False)
            for c in (c1, c2, c3)
        ]
        p1 = self.region_predictor(torch.cat((c4, *aligned), dim=1))
        c1 = self.reduce1(c1)
        c2 = self.reduce2(c2)
        c3 = self.reduce3(c3)
        c4 = self.reduce4(c4)
        feature = self.aifm1(c3, c4, boundary_prior, p1.sigmoid())
        p2 = self.predictor_p2(feature)
        feature = self.aifm2(c2, feature, boundary_prior, p2.sigmoid())
        p3 = self.predictor_p3(feature)
        feature = self.aifm3(c1, feature, boundary_prior, p3.sigmoid())
        p4 = self.predictor_p4(feature)
        outputs = [
            F.interpolate(
                prediction, size=image.shape[-2:],
                mode="bilinear", align_corners=False,
            )
            for prediction in (p1, p2, p3, p4, boundary_prior)
        ]
        return LGPRNetOutputs(*outputs)
