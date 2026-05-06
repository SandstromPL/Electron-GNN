"""
Standalone evaluation script for the Coulomb-MLP baseline.

Usage (from Electron-GNN root):
    python -m baseline.evaluate --data_dir data/processed --ckpt baseline/checkpoints/best_model.pth
"""
import os
import sys
import argparse

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from baseline.dataset import BaselineDataset, OMEGA_GRID
from baseline.predict import load_model, predict_spectrum, extract_peaks_from_spectrum, spectral_overlap, matched_freq_mae


def evaluate(args):
    dataset = BaselineDataset(args.data_dir)
    model = load_model(args.ckpt)

    print(f"\n{'Molecule':<16} {'Overlap':>9} {'Freq MAE':>10} {'Pred Peaks':>12} {'True Peaks':>12}")
    print("-" * 65)

    overlaps, maes = [], []

    for i in range(len(dataset)):
        sample = dataset[i]
        name = sample["name"]

        pred_spectrum = predict_spectrum(model, sample["features"])
        true_spectrum = sample["spectrum"].numpy()
        true_freqs = sample["frequencies"].numpy()

        pred_freqs, _ = extract_peaks_from_spectrum(
            pred_spectrum, OMEGA_GRID,
            min_height_frac=args.min_height_frac,
            min_distance=args.min_distance,
        )

        ov = spectral_overlap(pred_spectrum, true_spectrum)
        mae = matched_freq_mae(pred_freqs, true_freqs)

        overlaps.append(ov)
        if not np.isnan(mae):
            maes.append(mae)

        mae_str = f"{mae:.4f}" if not np.isnan(mae) else "  N/A"
        print(f"{name:<16} {ov:>9.4f} {mae_str:>10} {len(pred_freqs):>12} {len(true_freqs):>12}")

    print("-" * 65)
    print(f"{'Average':<16} {np.mean(overlaps):>9.4f} {np.mean(maes) if maes else float('nan'):>10.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/processed")
    p.add_argument("--ckpt", default="baseline/checkpoints/best_model.pth")
    p.add_argument("--min_height_frac", type=float, default=0.02)
    p.add_argument("--min_distance", type=int, default=8)
    evaluate(p.parse_args())
