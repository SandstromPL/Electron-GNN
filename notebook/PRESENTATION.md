# Presentation: Electron Density Temporal Prediction using GNNs

## Slide-by-Slide Content for Academic Presentation

---

## Slide 1: Title Slide

**Title:** Electron Density Temporal Prediction using Graph Neural Networks

**Subtitle:** ML-Accelerated Quantum Dynamics for the Ammonia (NH₃) Molecule

**Author:** [Your Name]
**Course:** UGQ301
**Date:** May 2026

---

## Slide 2: The Problem — Why This Matters

### Quantum Simulations Are Slow
- RT-TDDFT simulates electron dynamics by solving Kohn-Sham equations at every timestep
- Each timestep requires diagonalizing large Hamiltonian matrices — **computationally prohibitive** for long timescales or large molecules

### The Core Question
> Can a neural network learn the time evolution operator of electron density, replacing expensive quantum steps with cheap forward passes?

### Why Ammonia (NH₃)?
- Simple 8-electron system — a tractable proof-of-concept
- Pyramidal geometry with rich spatial structure
- Well-studied, making physical intuition available

**Figure:** Show 3D molecular reconstruction (fig_05.png)

---

## Slide 3: The Task

### Input → Output Mapping

| | Description |
|---|-------------|
| **Input** | Electron density at 10,540 spatial grid points for **5 consecutive timesteps** (t−4 ... t) |
| **Output** | Electron density at 10,540 grid points for the **next timestep** (t+1) |
| **Data** | 401 RT-TDDFT snapshots, t=0 to t=2000, step=5 |

### Why This Is Hard

| Challenge | Magnitude |
|-----------|-----------|
| Density dynamic range | **17 orders of magnitude** (10⁻¹⁷ to 196) |
| Temporal change scale | ~**0.001%** of full range (σ = 6.34 × 10⁻⁵) |
| Dataset size | Only **316 training samples** (tiny for DL) |
| Physical constraint | Total electrons must be **conserved** exactly |

**Figure:** Show dataset insights (fig_01.png) — highlight skewness and delta scale

---

## Slide 4: Data Preprocessing Pipeline

### Why Raw Data Can't Be Used Directly

1. **17 orders of magnitude range** → activations would saturate/underflow
2. **Tiny deltas (σ=6.34×10⁻⁵)** → gradients vanish, model can't learn

### Three-Step Transform

```
Step 1: log1p(ρ)           Step 2: Δρ = ρ(t+1) − ρ(t)      Step 3: z = (Δρ − μ) / σ
─────────────────         ─────────────────────────        ──────────────────────
Compresses range from      Predict the change, not          Scales targets 15,774× 
[10⁻¹⁷, 196] → [0, 5.29]   the absolute value               to σ=1.0 (learnable!)
```

### Key Preprocessing Numbers

| Metric | Before | After |
|--------|--------|-------|
| Skewness | 4.08 | 3.34 |
| Target range | ±4.74 × 10⁻⁴ | ±7.49 |
| Target std | 6.34 × 10⁻⁵ | 1.0000 |
| Learnable? | ❌ No | ✅ Yes |

---

## Slide 5: Graph Construction — Dual Edge Types

### The Graph

- **10,540 nodes** = spatial grid points
- **184,018 edges** = connections between nodes
- Average degree: **17.5**

### Two Complementary Edge Types

| Edge Type | Method | k | What It Captures |
|-----------|--------|---|-------------------|
| **Spatial kNN** | Euclidean distance in 3D | 10 | Physical locality — nearby points influence each other |
| **Feature kNN** | Cosine similarity on temporal signatures | 8 | Behavioral similarity — points with similar dynamics connect |

### Why Dual Edges?

- Spatial edges enforce **physical smoothness** (density doesn't jump discontinuously)
- Feature edges capture **functional relationships** (e.g., symmetry-related points)
- Together: **184,018 edges** providing rich message-passing pathways

**Figure:** Show graph schematic or node degree distribution

---

## Slide 6: Model Architecture — TemporalGAT

### Graph Attention Network (GAT)
Unlike simple message-passing, GAT learns **which neighbors to listen to** via multi-head self-attention.

```
Input (N, 5)  →  InputProj (5→128)  →  4× GATBlock  →  OutputHead (128→1)  →  Output (N,)
```

### Layer-by-Layer Breakdown

| Layer | Type | Details |
|-------|------|---------|
| Input Projection | Linear + LayerNorm + GELU | 5 → 128 dimensions |
| GAT Block 1 | GATConv + LayerNorm + GELU + Residual | 4 heads, 128D |
| GAT Block 2 | GATConv + LayerNorm + GELU + Residual | 4 heads, 128D |
| GAT Block 3 | GATConv + LayerNorm + GELU + Residual | 4 heads, 128D |
| GAT Block 4 | GATConv + LayerNorm + GELU + Residual | 1 head, 128D |
| Output Head | Linear(128→64→32→1) + GELU | Scalar per node |

### Design Philosophy

- **79,489 parameters** — deliberately small for 316 training samples
- Residual connections prevent vanishing gradients in deep GNNs
- LayerNorm stabilizes training with graph-structured data
- GELU activation chosen over ReLU for smoother gradients

**Figure:** Show architecture diagram

---

## Slide 7: Physics-Aware Loss Function

### Standard ML Loss = Not Good Enough
MSE alone doesn't know about electrons — the model might predict density values that look plausible but violate physics.

### Three-Component Composite Loss

```
L_total = L_weighted_mse + 0.20 × L_conservation + 0.02 × L_smoothness
```

| Component | Formula | Physical Meaning | Weight |
|-----------|---------|------------------|--------|
| **Weighted MSE** | Σ wᵢ(yᵢ − ŷᵢ)² / Σ wᵢ | Error at each grid point, de-emphasized for vacuum | 1.0 |
| **Electron Conservation** | (Σρ̂ − Σρ_true)² / (Σρ_true)² | Total electron count must be preserved | 0.20 |
| **Graph Smoothness** | mean((ŷ_src − ŷ_dst)²) over edges | Density must vary smoothly in space | 0.02 |

### Soft Node Weighting

- **Active nodes** (60.8%): weights proportional to temporal variance [0.16–3.40]
- **Static nodes** (39.2%): floor weight of **0.12** (not zero — preserves graph connectivity)
- Avoids breaking the graph structure while reducing noise from vacuum regions

---

## Slide 8: Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Optimizer | AdamW | Weight decay decoupled from LR |
| Learning rate | 3 × 10⁻⁴ | Moderate — small dataset, needs careful steps |
| Weight decay | 1 × 10⁻⁴ | Light regularization for small model |
| Scheduler | CosineAnnealingWarmRestarts (T₀=20) | Helps escape local minima |
| Batch size | 2 | Small — each sample has 10,540 nodes × 5 timesteps |
| Gradient clipping | max norm = 1.0 | Prevents instability from attention mechanism |
| Early stopping | patience = 20 epochs | Prevents overfitting on small dataset |
| Mixed precision | AMP (float16 forward, float32 backward) | ~2× speedup on RTX A6000 |

### Dataset Split (Chronological — NO Shuffle)

| Split | Samples | Percentage |
|-------|---------|------------|
| Train | 316 | 80% |
| Val | 39 | 10% |
| Test | 41 | 10% |

> Chronological split prevents "peeking into the future" — tests true temporal generalization.

---

## Slide 9: Training Results

### Training Curves

| Metric | Value |
|--------|-------|
| Total epochs | 100 |
| Training time | 541s (9.0 minutes) |
| Best epoch | 88 |
| Best val loss | 2.645541 |
| Train-val gap | ~6% (mild underfitting) |

**Figure:** Show training curves (fig_02.png) — highlight loss convergence and val/train ratio

### Key Observations

- Loss improves slowly but steadily — no signs of overfitting
- Cosine annealing restart cycles visible in the loss oscillations
- Mass penalty terms are near-zero (1×10⁻¹⁴) — MSE dominates
- Smoothness penalty is small but non-zero (0.00002–0.00020)

---

## Slide 10: Test Results — The Central Paradox

### R² ≈ 0 at Delta Level, R² = 1.0 at Density Level

| Metric Space | R² | MAE |
|-------------|-----|-----|
| Scaled delta (model output) | **0.000413** ✗ | 0.500 |
| Raw delta (active nodes) | **0.000420** ✗ | 5.20 × 10⁻⁵ |
| **Abs density (active nodes)** | **1.000000** ✓ | **1.51 × 10⁻⁴** |

### Why This Happens

```
True density changes  ≈  0.001%  of absolute density value
Model's delta error   ≈  0.0001%  of absolute density value
─────────────────────────────────────────────────────────────
Delta-level R² is low because the model predicts near-zero changes
Absolute R² is high because "no change" is already ~99.999% correct
```

> **The model has learned that "do nothing" is an excellent baseline for this near-equilibrium system.**

---

## Slide 11: Test Results — Absolute Density (The Good News)

### Active Nodes (6,411 grid points with real electron density)

| Metric | Value | Interpretation |
|--------|-------|---------------|
| R² | **1.000000** | Perfect linear correlation |
| MAE | 1.51 × 10⁻⁴ | Tiny absolute error |
| RMSE | 4.85 × 10⁻⁴ | Even worst cases are small |
| NRMSE | **0.000248%** | Normalized error is negligible |
| MAPE | **0.0458%** | < 0.05% average percentage error |
| Within 0.5% | **100.00%** | Every single active node! |
| Within 1.0% | **100.00%** | Every single active node! |
| Within 5.0% | **100.00%** | Every single active node! |

### Electron Conservation

| Metric | Value |
|--------|-------|
| Mean mass error | **0.000060%** |
| Median mass error | 0.000061% |
| P95 mass error | 0.000064% |

> Out of 106,097 electrons, the model is off by ~0.06 electrons on average — essentially perfect.

**Figure:** Show test evaluation grid (fig_03.png) — highlight R² bars and within-X% chart

---

## Slide 12: Model Performance Breakdown

### Five Evaluation Spaces — Complete Transparency

| Space | All Nodes MAE | Active Nodes MAE | R² | Verdict |
|-------|--------------|-----------------|-----|---------|
| Scaled delta | 0.500 | — | 0.0004 | Delta prediction needs work |
| Raw delta | 3.17 × 10⁻⁵ | 5.20 × 10⁻⁵ | 0.0004 | Small absolute error |
| Abs density | 9.63 × 10⁻⁵ | 1.51 × 10⁻⁴ | 1.0 | **Excellent reconstruction** |
| Electron mass | — | — | — | **0.00006% error** |

### Active vs Static Nodes

- **Static (39.2%):** Near-zero density, small relative errors → trivial to predict
- **Active (60.8%):** Real electron density, harder → but still **100% within 0.5%**
- MAPE drops from 451% (all nodes) to **0.046%** (active only) — because static nodes have near-zero denominators

---

## Slide 13: Visual Evidence I — Density Reconstruction

**Figure:** Show 3D reconstruction (fig_05.png)

### What to Notice

- Left panel: true RT-TDDFT density structure
- Right panel: GNN-predicted density structure
- Pyramidal NH₃ geometry clearly visible in both
- Density patterns are **visually indistinguishable**
- Confirms R² = 1.0 is not just a numerical artifact

---

## Slide 14: Visual Evidence II — Error Analysis

**Figure:** Show 3D error map (fig_06.png) and test evaluation grid (fig_03.png)

### Spatial Error Map (fig_06)

- Red = model over-predicts, Blue = under-predicts, White = near-zero error
- Predominantly white/light colors → errors are small everywhere
- Highest errors around molecular core (higher absolute density → larger absolute errors)

### Attention Analysis (fig_04)

- Attention weights are broadly distributed, not concentrated on few nodes
- No strong correlation between node activity and attention received
- Consistent with the model's "identity+perturbation" behavior pattern

---

## Slide 15: Key Innovations

| # | Innovation | Impact |
|---|-----------|--------|
| 1 | `log1p` transform | Handles 17-orders-of-magnitude without clipping |
| 2 | Delta standardization (×15,774) | Makes targets numerically learnable |
| 3 | Dual edge types (spatial + feature) | Complementary inductive biases |
| 4 | Soft static node weighting | Preserves graph structure |
| 5 | Physics-aware composite loss | Enforces conservation + smoothness |
| 6 | Chronological data split | True temporal generalization test |
| 7 | Float64 preprocessing | Prevents precision loss at σ = 10⁻⁵ |
| 8 | Multi-space evaluation | Complete transparency |

---

## Slide 16: Limitations & Critical Analysis

### 1. The Persistence Baseline Problem
> A model that outputs "no change" (ρ_predicted = ρ_last) would also achieve R² ≈ 1.0

- The current model is essentially a learned near-identity operator
- Absolute density R² = 1.0 is expected, not surprising

### 2. Delta Prediction Is Weak
- R² ≈ 0.0004 at delta level — model doesn't capture temporal dynamics
- Predicted deltas (±1 × 10⁻⁶) are ~500× narrower than true deltas (±5 × 10⁻⁴)

### 3. Single Molecule
- Only tested on NH₃ — no evidence of transfer learning
- Other molecules may have richer dynamics requiring real delta prediction

### 4. Tiny Dataset (316 samples)
- Prevents deeper architectures or larger models
- Risk of memorization rather than generalization

### 5. No Baseline Comparison
- Persistence model, linear regression, or simple GNN variants not evaluated
- Hard to quantify value-add of physics constraints and dual edges

### 6. Single-Step Only
- Autoregressive multi-step rollouts unexplored
- Error accumulation over long trajectories unknown

---

## Slide 17: Future Directions

| Direction | Why |
|-----------|-----|
| **Autoregressive rollout** | Test if the model can propagate its own predictions over multiple steps |
| **Multi-molecule training** | Generalize beyond NH₃ — test on H₂O, CH₄, etc. |
| **Direct multi-horizon prediction** | Predict t+1, t+2, ..., t+k simultaneously |
| **Baseline comparisons** | Quantify value of each component (edges, physics loss, soft weights) |
| **Hyperparameter optimization** | Bayesian sweep over model size, loss weights, learning rate |
| **Larger models / more data** | If dataset grows, explore MPNN, Transformer, or equivariant architectures |
| **Improve delta-level prediction** | Curriculum learning, delta-specific losses, or adversarial training |

---

## Slide 18: Summary

### What We Built
A complete pipeline for learning electron density dynamics from RT-TDDFT data using Graph Attention Networks.

### What Works
- ✅ Absolute density reconstruction is near-perfect (R² = 1.0, 100% within 0.5%)
- ✅ Electron conservation is enforced to 0.00006%
- ✅ Physics-aware loss provides stable training
- ✅ Dual-edge graph captures both spatial and behavioral relationships

### What Needs Work
- ❌ Delta-level prediction is essentially zero (R² ≈ 0.0004)
- ❌ Model acts as near-identity operator — need to capture real dynamics
- ❌ Only tested on single molecule with tiny dataset

### The Bottom Line
> The pipeline infrastructure is solid — preprocessing, graph construction, physics constraints, and evaluation all work correctly. The model successfully reconstructs absolute density but has not yet learned the true temporal dynamics. This is a foundation to build upon, not an endpoint.

---

## Slide 19: Questions?

### Backup Slides Available
- Detailed metric tables
- Training log analysis
- Architecture alternatives considered
- Data preprocessing ablation discussion

---

## Figures to Include in Slides

| Slide | Figure | File |
|-------|--------|------|
| 2 | Molecular 3D reconstruction | `report_figures/fig_05.png` |
| 3 | Dataset insights (skewness, delta scale) | `report_figures/fig_01.png` |
| 9 | Training curves | `report_figures/fig_02.png` |
| 11 | Test evaluation grid | `report_figures/fig_03.png` |
| 13 | 3D density reconstruction | `report_figures/fig_05.png` |
| 14 | 3D error map + attention analysis | `report_figures/fig_06.png` + `report_figures/fig_04.png` |

---

## Key Numbers for Quick Reference (Cheat Sheet)

| What | Number |
|------|--------|
| Grid points | 10,540 |
| Timesteps | 401 |
| Training samples | 316 |
| Model params | 79,489 |
| Graph edges | 184,018 |
| Density range | 10⁻¹⁷ – 196 |
| Delta σ | 6.34 × 10⁻⁵ |
| Standardization factor | 15,774× |
| Active nodes | 60.8% |
| Training time | 9.0 min |
| Abs density R² (active) | 1.000000 |
| Abs density MAPE (active) | 0.046% |
| Within 0.5% (active) | 100% |
| Mass conservation error | 0.00006% |
| Delta R² | 0.000413 |

---

*Presentation content prepared from `gnn_work_improved.ipynb`*
*All figures extracted from notebook outputs*
