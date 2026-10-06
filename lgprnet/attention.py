"""Prior-guided attention and gated depthwise feed-forward blocks.

Layer normalization and GDFN follow the Restormer formulation. Source credits
and its MIT license are retained in THIRD_PARTY.md and licenses/.
"""

import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class BiasFreeLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        if len(normalized_shape) != 1:
            raise ValueError("Layer normalization expects one channel dimension")
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        variance = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(variance + 1e-5) * self.weight


class WithBiasLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        if len(normalized_shape) != 1:
            raise ValueError("Layer normalization expects one channel dimension")
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        variance = x.var(-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(variance + 1e-5) * self.weight + self.bias


class LayerNorm2d(nn.Module):
    def __init__(self, dim, norm_type="WithBias"):
        super().__init__()
        if norm_type == "BiasFree":
            self.body = BiasFreeLayerNorm(dim)
        elif norm_type == "WithBias":
            self.body = WithBiasLayerNorm(dim)
        else:
            raise ValueError("norm_type must be 'BiasFree' or 'WithBias'")

    def forward(self, x):
        height, width = x.shape[-2:]
        tokens = rearrange(x, "b c h w -> b (h w) c")
        return rearrange(self.body(tokens), "b (h w) c -> b c h w", h=height, w=width)


class GDFN(nn.Module):
    """Gated Depthwise Feed-Forward Network used inside SGA."""

    def __init__(self, dim, ffn_expansion_factor=4, bias=False):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2, hidden_features * 2, 3,
            padding=1, groups=hidden_features * 2, bias=bias,
        )
        self.project_out = nn.Conv2d(hidden_features, dim, 1, bias=bias)

    def forward(self, x):
        x1, x2 = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class GuidedSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False):
        super().__init__()
        if num_heads < 1 or dim % num_heads != 0:
            raise ValueError("dim must be divisible by a positive num_heads")
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv_0 = nn.Conv2d(dim, dim, 1, bias=bias)
        self.qkv_1 = nn.Conv2d(dim, dim, 1, bias=bias)
        self.qkv_2 = nn.Conv2d(dim, dim, 1, bias=bias)
        self.qkv1conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=bias)
        self.qkv2conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=bias)
        self.qkv3conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x, prior=None):
        _, _, height, width = x.shape
        q = self.qkv1conv(self.qkv_0(x))
        k = self.qkv2conv(self.qkv_1(x))
        v = self.qkv3conv(self.qkv_2(x))
        if prior is not None:
            q = q * prior
            k = k * prior
        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attention = ((q @ k.transpose(-2, -1)) * self.temperature).softmax(dim=-1)
        attended = rearrange(
            attention @ v, "b head c (h w) -> b (head c) h w",
            head=self.num_heads, h=height, w=width,
        )
        return self.project_out(attended)


class SGA(nn.Module):
    """Shared guided attention used by SGA-R and SGA-B.

    The branches have the same structure and separate weights. They differ
    in the guiding prior and the feature level supplied by AIFM.
    """

    def __init__(self, dim=128, num_heads=8, ffn_expansion_factor=4,
                 bias=False, norm_type="WithBias"):
        super().__init__()
        self.norm1 = LayerNorm2d(dim, norm_type)
        self.attn = GuidedSelfAttention(dim, num_heads, bias)
        self.norm2 = LayerNorm2d(dim, norm_type)
        self.ffn = GDFN(dim, ffn_expansion_factor, bias)

    def forward(self, x, prior=None):
        x = x + self.attn(self.norm1(x), prior)
        return x + self.ffn(self.norm2(x))
