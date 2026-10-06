# Source credits

This release was prepared from the local research implementation. Renaming
classes for consistency with the manuscript does not change their provenance.

- The local project base is [CTF-Net](https://github.com/zcc0616/CTF-Net),
  by Dongdong Zhang, Chunping Wang, Huiying Wang, Qiang Fu, and Zhaorui Li.
  The released EGM and retained network operations derive from that local
  project. Upstream paper: https://doi.org/10.1016/j.cviu.2025.104431.
- Layer normalization and the gated depthwise feed-forward formulation follow
  [Restormer](https://github.com/swz30/Restormer), by Syed Waqas Zamir and
  contributors. Its MIT license is retained in
  [licenses/Restormer-MIT.txt](licenses/Restormer-MIT.txt). Guided attention
  retains the prior-modulated query/key formulation present in the local code.
- The encoder interfaces correspond to
  [Res2Net](https://github.com/Res2Net/Res2Net-PretrainedModels) and
  [PVTv2](https://github.com/whai362/PVT). Their code and weights are not
  bundled with this release; obtain them from their authors and follow their
  respective licenses.

No new blanket license is assigned to inherited source code. The local
project does not include a license for all inherited components; public
redistribution terms for those components must be established with their
rights holders. This notice does not grant permissions beyond existing
upstream licenses.
