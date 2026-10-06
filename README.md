# LGPRNet

**LGPRNet: Local-Global Prior Refinement Network for Camouflaged Object Detection**

LGPRNet coordinates local CNN features, global Transformer representations,
boundary guidance, and progressively updated region priors for camouflaged
object detection.

## Release scope

This repository provides selected components and network assembly code.
It is a partial source release, not a complete reproduction package for the
reported experiments.

| Component | Description | Availability |
| --- | --- | --- |
| EGM | Edge Guidance Module; constructs a target-related boundary prior | Included |
| DGRM | Dual-Guided Refinement Module; refines local and global features | Not included in this release |
| AIFM | Attention-based Interactive Fusion Module; fuses region- and boundary-guided features | Included |
| SGA | Shared guided attention used in the region and boundary branches | Included |
| GDFN | Gated Depthwise Feed-Forward Network inside SGA | Included |
| LGPRNet | Assembly of supplied encoders, supplied DGRM stages, EGM, and AIFM | Included; requires external components |
| Training, evaluation, and trained weights | Complete experiment reproduction | Not included in this release |

The code was prepared from an available local implementation. It has not
been validated against the final trained model or its checkpoints. Module
names and prediction order follow the current manuscript. No published
performance, parameter count, FLOPs, or FPS is claimed for this partial release.

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
and einops 0.8.2. Numerical comparisons against the available local source
matched exactly for EGM and SGA, and for AIFM after removing its historical
post-fusion enhancement block. The network assembly matched the retained
operations using test-only encoder/refiner fixtures; output sizes, prediction
order, and AIFM input gradients were also checked. These checks establish
implementation consistency for the released operations, not equivalence
to the final trained network or reproduction of benchmark results.

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

`LGPRNet` requires explicitly supplied encoders and four final DGRM modules:

```python
from lgprnet import LGPRNet

# These objects must be supplied from your own final implementation.
model = LGPRNet(cnn_encoder, transformer_encoder, dgrm_stages)
outputs = model(image)
probability_map = outputs.P4.sigmoid()
```

This example describes the interface; the three external arguments are not
provided by this repository. The CNN must return four stage features with
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

The assembly retains the available local prediction blocks and aligns outputs
to the input size explicitly. Renamed state-dictionary keys are not compatible
with historical checkpoints without a separately verified conversion.

## Data and acknowledgements

Datasets, pretrained backbone weights, and prediction maps are not redistributed.
The relevant benchmarks are CAMO, COD10K, and NC4K; CHAMELEON is used only
in the manuscript's PR and F-measure curve comparisons.

The local implementation incorporates established code and mechanisms.
SGA and its feed-forward layers are not claimed as newly invented structures.
See [THIRD_PARTY.md](THIRD_PARTY.md) for source credits and license information.
