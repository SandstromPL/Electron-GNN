"""
Train the Coulomb-MLP baseline.

Usage (from Electron-GNN root):
    python -m baseline.train --data_dir data/processed --epochs 1000
"""
import os
import sys
import json
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from baseline.dataset import BaselineDataset, baseline_collate
from baseline.model import SpectrumMLP


def log_mse_loss(pred, target, eps=1e-9):
    """
    MSE in log space — gives equal weight to small and large peaks.
    More physically meaningful than raw MSE for sparse spectra.
    """
    return nn.functional.mse_loss(torch.log(pred + eps), torch.log(target + eps))


def train(args):
    os.makedirs(args.save_dir, exist_ok=True)

    dataset = BaselineDataset(args.data_dir, max_atoms=args.max_atoms)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=baseline_collate)

    model = SpectrumMLP(
        input_dim=args.max_atoms,
        hidden_dims=tuple(args.hidden_dims),
        output_dim=512,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")
    print(f"Dataset: {len(dataset)} molecules")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    log = {
        "epochs": [],
        "loss": [],
        "config": {
            "data_dir": args.data_dir,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "max_atoms": args.max_atoms,
            "hidden_dims": args.hidden_dims,
        },
    }

    best_loss = float("inf")
    ckpt_path = os.path.join(args.save_dir, "best_model.pth")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in loader:
            feats = batch["features"]
            spectra = batch["spectrum"]

            pred = model(feats)
            loss = log_mse_loss(pred, spectra)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        log["epochs"].append(epoch)
        log["loss"].append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": {
                        "input_dim": args.max_atoms,
                        "hidden_dims": args.hidden_dims,
                        "output_dim": 512,
                    },
                },
                ckpt_path,
            )

        if epoch % max(1, args.epochs // 20) == 0 or epoch == 1:
            print(f"Epoch {epoch:5d}/{args.epochs}  loss={avg_loss:.6f}  best={best_loss:.6f}")

    log_path = os.path.join(args.save_dir, "train_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nDone. Best loss: {best_loss:.6f}")
    print(f"Checkpoint : {ckpt_path}")
    print(f"Log        : {log_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/processed")
    p.add_argument("--save_dir", default="baseline/checkpoints")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max_atoms", type=int, default=16)
    p.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 512, 512, 256])
    train(p.parse_args())
