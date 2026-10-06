# LGPRNet

**LGPRNet: Local-Global Prior Refinement Network for Camouflaged Object Detection**

LGPRNet coordinates local CNN features, global Transformer representations,
boundary guidance, and progressively updated region priors for camouflaged
object detection.

## Method overview

LGPRNet combines local-global representation refinement with progressive
region-boundary guidance. Its main components are:

| Component | Description |
| --- | --- |
| EGM | Edge Guidance Module; constructs a target-related boundary prior |
| DGRM | Dual-Guided Refinement Module; refines local and global features |
| AIFM | Attention-based Interactive Fusion Module; fuses region- and boundary-guided features |

AIFM uses SGA in its region-guided and boundary-guided branches, with a Gated
Depthwise Feed-Forward Network (GDFN) for feature refinement. Prediction names
follow the manuscript: P1 is the coarse region prediction and P4 is the final
segmentation prediction.

## Setup

Install PyTorch appropriate for your device, followed by the dependencies:

```bash
pip install -r requirements.txt
```

Run the component example from the repository root:

```bash
python -m examples.check_components --device cpu
```

This example uses random tensors to check EGM and the three AIFM stages.
It does not generate predictions for benchmark images.

## Verification

The component example was run on CPU with Python 3.11, PyTorch 2.14.1+cpu,
and einops 0.8.2. Checks cover output shapes, finite values, and AIFM input
gradients. The network assembly's prediction order and output sizes were
also checked using synthetic inputs.

## Components

```python
from lgprnet import EGM, AIFM

egm = EGM()
boundary_logits = egm(x1, t4)
boundary_prior = boundary_logits.sigmoid()

aifm1 = AIFM(high_channels=256, current_channels=256, out_channels=256)
refined_feature = aifm1(c3, c4, boundary_prior, region_prior)
```

`x1` has 256 channels and `t4` has 512 channels. EGM outputs one-channel
boundary logits at the spatial resolution of `x1`. AIFM receives probability
priors in [0, 1], aligns their spatial resolutions, and returns a fused feature.
The region-guided and boundary-guided SGA branches have the same structure
but separate parameters. Prediction heads are separate from AIFM.

## Network assembly

`LGPRNet` accepts the CNN encoder, Transformer encoder, and four DGRM stages
as constructor arguments:

```python
from lgprnet import LGPRNet

# Initialize with the configured encoders and refinement stages.
model = LGPRNet(cnn_encoder, transformer_encoder, dgrm_stages)
outputs = model(image)
probability_map = outputs.P4.sigmoid()
```

The CNN must return four stage features with
channels `(256, 512, 1024, 2048)`; the Transformer must return four features
with channels `(64, 128, 320, 512)`. Each supplied DGRM takes the corresponding
CNN and Transformer features and returns a feature with Transformer-stage
channels and matching spatial resolution.

The region predictions are ordered from coarse to fine:

| Name | Role |
| --- | --- |
| P1 | Coarse region prediction; its sigmoid supplies the initial region prior |
| P2 | Updated prediction from AIFM-1 |
| P3 | Updated prediction from AIFM-2 |
| P4 | Final prediction from AIFM-3 |
| B | Shared boundary probability prior from EGM |

P1--P4 are logits resized to the input resolution; B is already a probability
map and must not receive another sigmoid.
Within decoding, priors are computed from the native-resolution logits.
The selected final output is `outputs.P4`, not the first returned tensor.
Binary thresholding is not part of the network.

## Benchmarks

The relevant benchmarks are CAMO, COD10K, and NC4K; CHAMELEON is used only
in the manuscript's PR and F-measure curve comparisons.

## Acknowledgements

The implementation incorporates established code and attention mechanisms.
See [THIRD_PARTY.md](THIRD_PARTY.md) for source credits and license information.
