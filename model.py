"""GRAM-style network for MNIST, following "Modular Pretraining Enables Access
Control" (gradient-routed auxiliary modules) scaled down to an MLP classifier.

Architecture: shared embed -> 2 residual GRAM blocks -> shared head.
Each GRAM block: h_out = core_mlp(h) + m5 * aux5(h) + m7 * aux7(h), added to
the residual stream. Aux modules are small (~9% of MLP params), mirroring the
paper's asymmetric core/aux sizing.

Parameter partitions: 'core' (embed + core MLPs + head), 'aux5', 'aux7'.
"""

import torch
import torch.nn as nn

AUX_DIGITS = (5, 7)  # aux module 0 <-> digit 5, aux module 1 <-> digit 7
N_AUX = len(AUX_DIGITS)


def mlp(d_in, d_hidden, d_out):
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_out))


class GramBlock(nn.Module):
    def __init__(self, dim=256, core_hidden=512, aux_hidden=64):
        super().__init__()
        self.core = mlp(dim, core_hidden, dim)
        self.aux = nn.ModuleList([mlp(dim, aux_hidden, dim) for _ in range(N_AUX)])

    def forward(self, h, mask):
        out = self.core(h)
        for m, module in zip(mask, self.aux):
            if m:  # m may be a float in (0,1] for continuous ablation
                out = out + m * module(h)
        return out


class GramNet(nn.Module):
    def __init__(self, dim=256, core_hidden=512, aux_hidden=64, n_blocks=2):
        super().__init__()
        self.embed = nn.Sequential(nn.Flatten(), nn.Linear(784, dim), nn.ReLU())
        self.blocks = nn.ModuleList(
            [GramBlock(dim, core_hidden, aux_hidden) for _ in range(n_blocks)]
        )
        self.head = nn.Linear(dim, 10)

    def forward(self, x, mask=(1, 1)):
        """mask: length-N_AUX activation indicator (module i active iff mask[i])."""
        h = self.embed(x)
        for block in self.blocks:
            h = h + block(h, mask)
        return self.head(h)


def partition_params(model):
    """Split parameters into the GRAM partitions: core / aux5 / aux7."""
    parts = {"core": [], "aux5": [], "aux7": []}
    for name, p in model.named_parameters():
        if ".aux.0." in name:
            parts["aux5"].append(p)
        elif ".aux.1." in name:
            parts["aux7"].append(p)
        else:
            parts["core"].append(p)
    return parts


PROFILES = {
    "full": (1, 1),       # core + module5 + module7
    "ablate5": (0, 1),    # 5-capability removed
    "ablate7": (1, 0),    # 7-capability removed
    "core_only": (0, 0),  # both removed
}
