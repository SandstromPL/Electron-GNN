import os
import glob
import sys
import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from baseline.features import coulomb_matrix_eigenvalues, MAX_ATOMS

OMEGA_MIN = 0.01
OMEGA_MAX = 5.0
N_GRID = 512
GAMMA = 0.015
OMEGA_GRID = np.linspace(OMEGA_MIN, OMEGA_MAX, N_GRID, dtype=np.float32)


def lorentzian_spectrum(freqs, amps, omega_grid=OMEGA_GRID, gamma=GAMMA):
    """Reconstruct Lorentzian spectrum from discrete peaks."""
    S = np.zeros(len(omega_grid), dtype=np.float32)
    for w, b in zip(freqs, amps):
        S += b * gamma / ((omega_grid - w) ** 2 + gamma ** 2)
    return S


def baseline_collate(batch):
    """
    Custom collate: stack fixed-size tensors (features, spectrum),
    keep variable-length peak tensors as lists.
    """
    return {
        "features": torch.stack([b["features"] for b in batch]),
        "spectrum": torch.stack([b["spectrum"] for b in batch]),
        "frequencies": [b["frequencies"] for b in batch],
        "amplitudes": [b["amplitudes"] for b in batch],
        "atomic_numbers": [b["atomic_numbers"] for b in batch],
        "positions": [b["positions"] for b in batch],
        "name": [b["name"] for b in batch],
    }


class BaselineDataset(Dataset):
    """
    Loads processed .pt files from the Electron-GNN pipeline.
    Returns Coulomb eigenvalue features and target Lorentzian spectrum grid.
    """

    def __init__(self, processed_dir, max_atoms=MAX_ATOMS):
        self.files = sorted(glob.glob(os.path.join(processed_dir, "*.pt")))
        self.max_atoms = max_atoms
        if not self.files:
            raise FileNotFoundError(f"No .pt files found in {processed_dir}")

    def __len__(self):
        return len(self.files)

    def get_name(self, idx):
        return os.path.basename(self.files[idx]).replace("_targets.pt", "")

    def __getitem__(self, idx):
        d = torch.load(self.files[idx], weights_only=False)

        feats = coulomb_matrix_eigenvalues(
            d["atomic_numbers"].numpy(),
            d["positions"].numpy(),
            max_atoms=self.max_atoms,
        )

        freqs = d["frequencies"].numpy()
        amps = d["amplitudes_x"].numpy()
        spectrum = lorentzian_spectrum(freqs, amps)

        return {
            "features": torch.tensor(feats),
            "spectrum": torch.tensor(spectrum),
            "frequencies": d["frequencies"],
            "amplitudes": d["amplitudes_x"],
            "atomic_numbers": d["atomic_numbers"],
            "positions": d["positions"],
            "name": self.get_name(idx),
        }
