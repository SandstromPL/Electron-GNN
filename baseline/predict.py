import os
import sys
import numpy as np
import torch
from scipy.signal import find_peaks
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from baseline.dataset import OMEGA_GRID, lorentzian_spectrum
from baseline.model import SpectrumMLP


def load_model(ckpt_path):
    """Load model from checkpoint, using saved config."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {"input_dim": 16, "hidden_dims": [256, 512, 512, 256], "output_dim": 512})
    model = SpectrumMLP(
        input_dim=cfg["input_dim"],
        hidden_dims=tuple(cfg["hidden_dims"]),
        output_dim=cfg["output_dim"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def predict_spectrum(model, features_tensor):
    """Run forward pass, return numpy spectrum array."""
    with torch.no_grad():
        out = model(features_tensor.unsqueeze(0)).squeeze(0)
    return out.numpy()


def extract_peaks_from_spectrum(spectrum, omega_grid=OMEGA_GRID, min_height_frac=0.02, min_distance=8):
    """
    Extract peaks from a dense predicted spectrum.
    Returns (frequencies, amplitudes) as numpy arrays.

    min_height_frac : fraction of max intensity to use as height threshold
    min_distance    : minimum separation in grid points between peaks
    """
    if spectrum.max() < 1e-12:
        return np.array([]), np.array([])

    height = spectrum.max() * min_height_frac
    prominence = height * 0.3

    peak_idx, _ = find_peaks(spectrum, height=height, distance=min_distance, prominence=prominence)

    if len(peak_idx) == 0:
        return np.array([]), np.array([])

    return omega_grid[peak_idx], spectrum[peak_idx]


def spectral_overlap(S_pred, S_true):
    """Cosine similarity between two spectrum vectors."""
    denom = np.linalg.norm(S_pred) * np.linalg.norm(S_true) + 1e-12
    return float(np.dot(S_pred, S_true) / denom)


def matched_freq_mae(pred_freqs, true_freqs):
    """Mean absolute error after optimal (Hungarian) frequency matching."""
    if len(pred_freqs) == 0 or len(true_freqs) == 0:
        return float("nan")
    cost = np.abs(pred_freqs[:, None] - true_freqs[None, :])
    row, col = linear_sum_assignment(cost)
    return float(cost[row, col].mean())
