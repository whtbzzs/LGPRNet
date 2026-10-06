"""Check component shapes with synthetic inputs, without pretrained weights."""

import argparse

import torch

from lgprnet import AIFM, EGM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(0)
    egm = EGM().to(device).eval()
    stages = [
        ("AIFM-1", AIFM(256, 256, 256), 256, 256),
        ("AIFM-2", AIFM(256, 128, 128), 128, 256),
        ("AIFM-3", AIFM(128, 64, 64), 64, 128),
    ]
    with torch.inference_mode():
        boundary_logits = egm(
            torch.randn(1, 256, 16, 16, device=device),
            torch.randn(1, 512, 2, 2, device=device),
        )
        assert boundary_logits.shape == (1, 1, 16, 16)
        assert torch.isfinite(boundary_logits).all()
        print("EGM boundary logits:", tuple(boundary_logits.shape))
        for name, module, current_channels, high_channels in stages:
            module = module.to(device).eval()
            feature = module(
                torch.randn(1, current_channels, 8, 10, device=device),
                torch.randn(1, high_channels, 4, 5, device=device),
                boundary_logits.sigmoid(),
                torch.rand(1, 1, 2, 3, device=device),
            )
            assert feature.shape == (1, current_channels, 8, 10)
            assert torch.isfinite(feature).all()
            print(name, "fused feature:", tuple(feature.shape))
    print("Component checks passed. This is not a trained-model evaluation.")


if __name__ == "__main__":
    main()
