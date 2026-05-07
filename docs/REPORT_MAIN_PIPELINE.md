# Electron-GNN Main Pipeline Report (Full Project Overview)

**Date:** 2026-05-06  
**Status:** V3 two-tower hybrid is the active production path.  
**Scope:** End-to-end pipeline from RT-TDDFT outputs to hybrid inference, with all numeric outputs and evaluation results.

---

## 1. Idea and Goal

The project accelerates quantum absorption spectroscopy by predicting discrete transition parameters directly from molecular geometry, bypassing long RT-TDDFT simulations. Instead of regressing a dense spectrum, the model predicts unordered transition sets and reconstructs the spectrum analytically:

$$
\text{Structure} \rightarrow \{(\omega_k, B_k, p_k)\}_{k=1}^{K_{max}} \rightarrow S(\omega)
$$

Key motivations:
- RT-TDDFT is accurate but expensive (long time-propagation at high resolution).
- Predicting sparse transition sets is more physically meaningful and data-efficient.
- The active V3 hybrid combines a strong frequency prior (V1) with a stronger amplitude learner (V2).

---

## 2. End-to-End Flow (Raw Data to Hybrid Inference)

### Step 0: Raw simulation artifacts
- Inputs: ReSpect `.out` and `.xyz` files per molecule.
- Parses time steps and dipole traces from lines containing `Step EAS:`.

### Step 1: Dipole and geometry parsing
- Time grid and dipole traces per axis:
  $$\mu_a(t_n), \; a \in \{x,y,z\}$$
- Geometry from the `[Atoms]` block yields atomic numbers and 3D coordinates.

### Step 2: Peak extraction (Padé + clustering + LASSO)
- Pipeline: Padé root finding, K-Means cleanup, positive LASSO.
- Cutoff frequency: `cutoff_freq = 4.0` a.u.
- Active peak threshold: $B_k > 1\times 10^{-8}$.
- Output per molecule:
  - `frequencies` ($\omega_k$)
  - `amplitudes_x` ($B_k$ for x-polarized excitation)

### Step 3: Processed dataset
- Stored in `data/processed/*.pt`.
- Current dataset size: **2 molecules**.
  - **Ammonia (NH3):** 39 peaks
  - **Water (H2O):** 55 peaks

### Step 4: Graph construction (PyTorch Geometric)
- Node features: one-hot for `[H, C, N, O, F]` (5 dims).
- Directed edges for all pairs within cutoff radius:
  $$\|\mathbf{r}_i - \mathbf{r}_j\|_2 \le r_c, \; r_c = 5.0$$
- Edge features (4 dims):
  $$[d_{ij}, \Delta x_{ij}, \Delta y_{ij}, \Delta z_{ij}]$$

### Step 5: Variable-length target collation
- Each graph stores `y_freq`, `y_amp`, and `num_peaks`.
- Batching flattens targets; `num_peaks` is used to split per-graph targets.

### Step 6: Model components

#### 6.1 V1 frequency tower (legacy prior)
- Class: `SpectralEquivariantGNNV1`
- Fixed slot output size: **K_max = 50**
- Outputs per sample:
  - `prob` (existence probability per slot)
  - `freq` (frequencies)
  - `amp` (legacy amplitudes, not used in hybrid)

#### 6.2 V2 amplitude tower (set decoder)
- Class: `SpectralEquivariantGNN`
- Default slot size: **K_max = 64**
- Encoder: **GATv2** with residuals
  - default `num_layers = 4`, `num_heads = 4`, `hidden_dim = 128`, `dropout = 0.0`
- Decoder: Transformer set decoder with **2 layers** and learned queries
- Heads:
  - `prob` (slot existence)
  - `freq`
  - `amp`
  - `count` (global peak count)
- Amplitude scaling: `amp_scale = 1e-3`

### Step 7: Losses and training objective (V2 and V3)

#### 7.1 Bipartite matching loss (V2 amplitude tower)
- Hungarian cost:
  $$C_{ij} = 10|\omega_i - \omega_j^{(t)}| + |B_i - B_j^{(t)}|$$
- Smooth L1 with $\beta = 0.02$ for frequency and log-amplitude.
- Log amplitude scale: `amp_log_scale = 1e4`.
- Additional terms:
  - Existence BCE (logits if present)
  - Unmatched amplitude penalty
  - Sum consistency
  - Count loss
- Weighted total (per graph):
  $$8L_\omega + 8L_B + 1.2L_{prob} + 1.0L_{unmatched} + 6.0L_{sum} + 0.5L_{count}$$

#### 7.2 Auto-differential spectrum regularizer
- Time-domain reconstruction:
  - `t_max = 400`, `dt = 0.2`
- Spectrum reconstruction:
  - grid size = 512
  - Lorentzian broadening: `gamma = 0.015`
  - log loss scale: `log1p(5e3 * S)`
- Total spectrum loss:
  $$L_{signal} + 0.5L_{spec} + 0.5L_{area}$$

#### 7.3 Training loss combination
- Default:
  $$L = L_{bipartite} + \lambda_{spec} L_{spectrum},\; \lambda_{spec} = 0.3$$
- Gradient clip: `1.0`.

#### 7.4 Frequency tower loss (V1 retraining option)
- Weighted loss:
  $$12L_\omega + 1.0L_{prob} + 0.3L_{count} + 0.02L_{unmatched}$$
- Teacher regularization available for safe high-capacity retraining.

### Step 8: Hybrid decode (V3)
- Decode thresholds (production):
  - `prob_threshold = 0.65`
  - `fallback_topk = 5`
  - `min_freq_separation = 0.005`
  - `allow_amp_overflow = true`
  - `amp_conf_penalty = 0.05`
- Cardinality selection prefers amplitude tower `count` head.
- Amplitudes are assigned by frequency-aware Hungarian matching.

### Step 9: Spectrum reconstruction and evaluation
- Lorentzian reconstruction uses `gamma = 0.015`.
- Overlap metric:
  - frequency grid: `omega in [0.01, 5.0]`, 1024 points
  - overlap = cosine similarity of spectra
- Other metrics:
  - frequency MAE
  - amplitude MAE
  - predicted peak count vs true count

---

## 3. Results in Depth (All Numeric Outputs)

### 3.1 Baseline evaluation (best_model.pth)

**Ammonia (NH3)**

| Model | Freq MAE | Amp MAE | Overlap | Pred Peaks | True Peaks |
|---|---:|---:|---:|---:|---:|
| V1 | 0.03000 | 1.947729e-03 | 0.5240 | 41 | 39 |
| V2 | 0.71727 | 1.083328e-04 | 0.5434 | 39 | 39 |
| Hybrid | 0.03560 | 1.419743e-04 | 0.6472 | 39 | 39 |

**Water (H2O)**

| Model | Freq MAE | Amp MAE | Overlap | Pred Peaks | True Peaks |
|---|---:|---:|---:|---:|---:|
| V1 | 0.04456 | 2.305660e-03 | 0.4510 | 41 | 55 |
| V2 | 0.32185 | 6.498991e-05 | 0.5311 | 56 | 55 |
| Hybrid | 0.22634 | 9.328221e-05 | 0.4924 | 56 | 55 |

**Averages**

| Model | Avg Freq MAE | Avg Amp MAE | Avg Overlap |
|---|---:|---:|---:|
| V1 | 0.03728 | 2.126695e-03 | 0.4875 |
| V2 | 0.51956 | 8.666136e-05 | 0.5373 |
| Hybrid | 0.13097 | 1.176282e-04 | 0.5698 |

**Interpretation**
- V1 is the strongest frequency prior on tiny data.
- V2 is the strongest amplitude learner (lowest amp MAE).
- Hybrid gives the best average overlap, which is the main end metric.

### 3.2 Retrained amplitude checkpoint (v3_amp_tower.pth)

**Ammonia (NH3)**

| Model | Freq MAE | Amp MAE | Overlap | Pred Peaks | True Peaks |
|---|---:|---:|---:|---:|---:|
| V1 | 0.03000 | 1.947729e-03 | 0.5240 | 41 | 39 |
| V2 | 0.88691 | 7.368602e-05 | 0.4298 | 39 | 39 |
| Hybrid | 0.03560 | 1.845578e-04 | 0.5498 | 39 | 39 |

**Water (H2O)**

| Model | Freq MAE | Amp MAE | Overlap | Pred Peaks | True Peaks |
|---|---:|---:|---:|---:|---:|
| V1 | 0.04456 | 2.305660e-03 | 0.4510 | 41 | 55 |
| V2 | 0.50220 | 6.502168e-05 | 0.5474 | 55 | 55 |
| Hybrid | 0.29987 | 6.598743e-05 | 0.4123 | 55 | 55 |

**Averages**

| Model | Avg Freq MAE | Avg Amp MAE | Avg Overlap |
|---|---:|---:|---:|
| V1 | 0.03728 | 2.126695e-03 | 0.4875 |
| V2 | 0.69456 | 6.935385e-05 | 0.4886 |
| Hybrid | 0.16774 | 1.252726e-04 | 0.4811 |

**Interpretation**
- Retrained amplitude tower lowered amp MAE but reduced overlap.
- Hybrid overlap dropped, so baseline amp checkpoint is preferred.

### 3.3 Overflow-disabled hybrid control

**Ammonia (NH3)**

| Model | Freq MAE | Amp MAE | Overlap | Pred Peaks | True Peaks |
|---|---:|---:|---:|---:|---:|
| V1 | 0.03000 | 1.947729e-03 | 0.5240 | 41 | 39 |
| V2 | 0.71727 | 1.083328e-04 | 0.5434 | 39 | 39 |
| Hybrid | 0.03560 | 1.419743e-04 | 0.6472 | 39 | 39 |

**Water (H2O)**

| Model | Freq MAE | Amp MAE | Overlap | Pred Peaks | True Peaks |
|---|---:|---:|---:|---:|---:|
| V1 | 0.04456 | 2.305660e-03 | 0.4510 | 41 | 55 |
| V2 | 0.32185 | 6.498991e-05 | 0.5311 | 56 | 55 |
| Hybrid | 0.04180 | 1.033901e-04 | 0.4666 | 50 | 55 |

**Averages**

| Model | Avg Freq MAE | Avg Amp MAE | Avg Overlap |
|---|---:|---:|---:|
| V1 | 0.03728 | 2.126695e-03 | 0.4875 |
| V2 | 0.51956 | 8.666136e-05 | 0.5373 |
| Hybrid | 0.03870 | 1.226822e-04 | 0.5569 |

**Interpretation**
- Disabling overflow reduces predicted peak count for water (50 vs 55).
- Hybrid overlap remains strong but still trails the baseline hybrid with overflow.

### 3.4 Production selection
- Recommended mode: **V3 hybrid two-tower**.
- Decode defaults from `results/model_selection.json`:
  - `prob_threshold = 0.65`
  - `fallback_topk = 8`
  - `allow_amp_overflow = true`
  - `min_freq_separation = 0.005`

---

## 4. Current Constraints (Numeric)

- Dataset size: **2 molecules** (NH3, H2O).
- Peak counts: **39** and **55**.
- V1 slot cap: **K_max = 50** (truncates water without overflow).
- V2 slot cap: **K_max = 64**.
- Small-data instability: amplitude retraining reduces overlap on current benchmarks.

---

## 5. Summary (What This Approach Achieves)

- End-to-end pipeline from RT-TDDFT outputs to ML-predicted spectra is fully implemented.
- Hybrid inference produces the strongest overall spectral overlap on the tiny dataset.
- The project is ready for scale-up, and the main bottleneck is dataset size, not tooling.
