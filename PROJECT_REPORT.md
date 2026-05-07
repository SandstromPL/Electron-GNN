# Comprehensive Project Report: Electron-GNN

**Date:** May 6, 2026
**Project:** ML-Accelerated Quantum Spectroscopy via Graph Neural Networks (Electron-GNN)
**Status:** In Progress (V3 Hybrid Workflow)

---

## 1. Executive Summary

Electron-GNN is a machine learning pipeline designed to accelerate quantum spectroscopy. It predicts absorption spectra directly from molecular geometry by circumventing computationally expensive Real-Time Time-Dependent Density Functional Theory (RT-TDDFT) simulations. 

The pipeline works by extracting physical target peaks (frequencies $\omega_k$ and amplitudes $B_k$) from short RT-TDDFT dipole response logs using a signal processing pipeline (Padé + K-Means + LASSO). It then uses an equivariant Graph Neural Network (GNN) to map atomic coordinates and species to these unordered peak sets. Finally, the predicted sets are analytically reconstructed into a continuous Lorentzian spectrum.

Currently, the project operates on a **Version 3 (V3) Two-Tower Hybrid** architecture. This hybrid approach combines a legacy V1 model (which serves as a strong frequency prior) with a newer V2 GATv2+Transformer model (which excels at amplitude and peak-count prediction) to achieve the best overall spectral overlap, particularly given the current data-limited regime.

---

## 2. Core Scientific Theory & Pipeline

### 2.1 The Physics Motivation
A standard RT-TDDFT simulation involves applying a Dirac-delta electric field perturbation to a molecule and calculating the induced dipole moment $\mu(t)$ over time. The absorption spectrum is derived from the Fourier transform of this dipole response. However, achieving high frequency resolution requires extremely long simulation times, which scales terribly, $O(N_e^3)$, with the number of electrons.

Electron-GNN bypasses this "Curse of Resolution" by predicting the underlying mathematical poles (frequencies) and oscillator strengths (amplitudes) directly from the 3D molecular structure.

### 2.2 Data Generation (The Hauge Pipeline)
Instead of training the ML model on raw time-series data, the project extracts clean physical parameters:
1. **Padé Approximant:** Finds the roots (poles) of the z-transform of the short dipole signal to perfectly locate the transition frequencies ($\omega_k$).
2. **K-Means Clustering:** Filters out numerical ghost poles, keeping only true physical frequencies.
3. **LASSO Regression:** Constrains amplitudes ($B_k$) to be positive and solves for them, creating a perfectly sparse target set: $\{\omega_k, B_k\}$.

*Implementation:* `scripts/parser.py` and `scripts/extract_peaks.py` (which heavily relies on an external `HyQD/absorption-spectrum` library).

---

## 3. Machine Learning Architecture

The ML task is challenging because it requires mapping an $E(3)$ invariant graph (the molecule) to an unordered set of predictions (the peaks) of variable size $K$.

### 3.1 The Amplitude Tower (V2 Architecture)
The core model (`models/mace_net.py`) is designed as a set-predictor.
* **Encoder:** A multi-layer Graph Attention Network (GATv2) processes node embeddings (one-hot element encoding) and edge features (distances/relative vectors). It pools the graph into a global context.
* **Decoder:** A DETR-style Transformer Decoder. It uses $K_{max}$ (e.g., 64) learned queries to attend to the dense node features. 
* **Heads:** The refined slots are passed through specialized MLPs to predict:
  * `prob`: Probability that the slot contains a real peak (Existence Mask).
  * `freq`: The transition frequency ($\omega_k$).
  * `amp`: The amplitude vector magnitude ($B_k$).
  * `count`: A global head predicting the total number of valid peaks (cardinality).

### 3.2 The Frequency Prior (V1 Architecture)
The older V1 model (`models/mace_net_v1.py`) is a simpler architecture that predicts fixed-size slots directly from global graph pooling. Despite its simplicity, it was found to learn frequency locations much better on the tiny dataset than the complex V2 decoder.

### 3.3 The V3 Hybrid Combiner
Because the dataset is incredibly small (only Ammonia and Water), the complex V2 model struggles to generalize frequency locations, while V1 struggles with amplitudes. The V3 pipeline (`utils/hybrid_inference.py`) solves this:
1. It takes the highly accurate frequency predictions from V1.
2. It takes the highly accurate amplitude and peak-count predictions from V2.
3. It performs a frequency-aware assignment (with confidence penalties) to stitch them together into a final peak set.

### 3.4 Losses and Training Objective
The training loop (`train/losses.py`) handles the variable-sized target sets using two main losses:
1. **Bipartite (Hungarian) Matching Loss:** Optimally aligns the $K_{max}$ unordered predictions with the variable-sized true target set. It applies Smooth L1 loss on matched frequencies and log-scaled amplitudes, and BCE loss on the existence probability mask to push unmatched slots to zero.
2. **Auto-Differential Spectrum Regularizer:** A physics-informed loss that analytically reconstructs the time-domain signal $\sum B \sin(\omega t)$ and the frequency-domain Lorentzian spectrum inside PyTorch, penalizing the difference between the predicted and true continuous spectra.

---

## 4. Repository Structure & Key Components

*   **`data/`**: Contains raw RT-TDDFT logs (`raw/`) and extracted PyTorch Geometric targets (`processed/*.pt`). *Current bottleneck: Only contains NH3 and H2O.*
*   **`models/`**: 
    *   `mace_net.py`: V2/V3 Amplitude tower with GATv2 and set decoder.
    *   `mace_net_v1.py`: V1 legacy frequency tower.
    *   `molecule_graph.py`: Converts XYZ coordinates to PyG Data graphs.
*   **`train/`**: 
    *   `dataset.py`: PyG dataset loader.
    *   `losses.py`: Hungarian matching and spectrum reconstruction losses.
    *   `train_v3_two_tower.py`: The complex training loop that handles warming up the frequency prior, training the amplitude tower, and early stopping.
*   **`utils/` & `scripts/`**: 
    *   `hybrid_inference.py`: Logic to combine V1 and V2 outputs.
    *   `evaluate_two_tower.py`: Benchmarks V1 vs V2 vs Hybrid.
    *   Plotting and diagnostic utilities.
*   **`dashboard/`**: `app.py` is a comprehensive Streamlit dashboard for real-time model comparison, dataset inspection, and 3D visualization.
*   **`checkpoints/`**: Stores the production weights (`best_model_v1.pth`, `best_model.pth`).
*   **`docs/`**: Extensive physics theory, architectural reasoning, and version release notes.
*   **`volumetric_viz/`**: Experimental features for 3D isosurface rendering of electron density predictions (currently being integrated inline).

---

## 5. Current Performance & Status

Based on the latest evaluation (`results/v3_retrain_eval_summary.txt`), the Hybrid V3 pipeline is the most stable and performant model for generating complete spectra.

**Average Metrics:**
*   **V1:** Freq MAE 0.037, Amp MAE 2.12e-3, Overlap 0.487
*   **V2:** Freq MAE 0.519, Amp MAE 8.66e-5, Overlap 0.537
*   **Hybrid (V3):** Freq MAE 0.130, Amp MAE 1.17e-4, **Overlap 0.569**

**Why V3?**
The V3 amplitude tower alone occasionally collapses its peak count predictions leading to flat spectra. The hybrid combiner acts as a failsafe, utilizing V1's stable frequencies to maintain the structural integrity of the spectrum while benefiting from V2's precise amplitude scaling.

**Dashboard Quality Gate:**
Because amplitude retraining on a 2-molecule dataset is highly unstable, the dashboard now implements an automatic quality gate: it dynamically scores available amplitude checkpoints against a baseline overlap score and automatically selects the safest one for inference.

---

## 6. Known Constraints & Bottlenecks

1.  **Data Scarcity (CRITICAL):** The entire pipeline is architecturally sound but is severely bottlenecked by data. Having only 2 molecules means the model has almost zero capacity for chemical generalization. Amplitude calibration is extremely fragile.
2.  **Capacity Caps:** The V1 frequency tower is currently locked to a $K_{max} = 50$, which artificially truncates predictions for larger molecules (like Water, which has 55 peaks). The hybrid decoder allows some "overflow" from the amplitude tower to mitigate this, but a V1 tower retrain with $K_{max} = 64+$ is needed.
3.  **High-Capacity Retraining Risks:** Retraining the frequency tower on small data causes catastrophic forgetting. It requires strict regularization, warmup freezing, and teacher-student losses (all of which have been implemented in `train_v3_two_tower.py` but await more data).

---

## 7. Recommended Next Steps

1.  **Massive Dataset Generation:** Scale the input dataset from 2 molecules to a larger benchmark set containing diverse chemistries and conformers.
2.  **Vector-Aware Labels:** Add full 3D polarization targets (x, y, z axes) rather than just single-axis extraction to fully utilize the Equivariant nature of the intended architecture.
3.  **Retrain Frequency Prior:** Once data is available, retrain the V1 tower with $K_{max} \ge 64$ to remove the artificial bottleneck.
4.  **Validation Splits:** Implement strict train/test splits (currently disabled or overlapping due to tiny dataset size) to accurately measure generalization.
5.  **Uncertainty Quantification:** Add variance heads to the amplitude tower to allow the network to express low confidence on unseen chemical moieties.
