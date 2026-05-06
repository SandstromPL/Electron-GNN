import torch
import torch.nn as nn


class SpectrumMLP(nn.Module):
    """
    MLP that maps fixed-size Coulomb eigenvalue features to a dense
    Lorentzian spectrum grid. Softplus output ensures non-negative intensities.

    input_dim  : max_atoms Coulomb eigenvalues (padded)
    output_dim : N_GRID spectrum points (default 512)
    """

    def __init__(self, input_dim=16, hidden_dims=(256, 512, 512, 256), output_dim=512):
        super().__init__()
        dims = [input_dim] + list(hidden_dims) + [output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.GELU())
            else:
                layers.append(nn.Softplus())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
