"""
Baseline Observatory — Coulomb MLP Dashboard.

Run from Electron-GNN root:
    streamlit run baseline/dashboard.py
"""
import os
import sys
import json

import numpy as np
import pandas as pd
import torch
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from baseline.dataset import BaselineDataset, OMEGA_GRID, lorentzian_spectrum
from baseline.model import SpectrumMLP
from baseline.features import coulomb_matrix_eigenvalues, MAX_ATOMS
from baseline.predict import (
    load_model,
    predict_spectrum,
    extract_peaks_from_spectrum,
    spectral_overlap,
    matched_freq_mae,
)

DATA_DIR = os.path.join(ROOT, "data", "processed")
CKPT_DIR = os.path.join(ROOT, "baseline", "checkpoints")
CKPT_PATH = os.path.join(CKPT_DIR, "best_model.pth")
LOG_PATH = os.path.join(CKPT_DIR, "train_log.json")

ELEM_MAP = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Baseline Observatory", layout="wide", page_icon="⚗")


def apply_theme():
    st.markdown(
        """
<style>
:root {
  --bg: #111014; --panel: #1a1820; --ink: #f2e6cf;
  --muted: #b9a98b; --accent: #c9973a; --line: #2a2630;
  --blue: #7eb8d4;
}
html, body, [class*="css"] {
  font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
  color: var(--ink);
}
.stApp {
  background:
    radial-gradient(1200px 500px at 10% -10%, #2a221a 0%, transparent 60%),
    radial-gradient(1000px 450px at 90% 0%, #2f1f1b 0%, transparent 55%),
    var(--bg);
}
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #1f1d25 0%, #18161d 100%);
  border-right: 1px solid var(--line);
}
h1, h2, h3 { letter-spacing: 0.2px; }
.block-card {
  background: rgba(22,20,28,0.88); border: 1px solid var(--line);
  border-radius: 12px; padding: 0.9rem 1rem;
}
.small-note { color: var(--muted); font-size: 0.92rem; }
div[data-testid="metric-container"] {
  background: rgba(18,17,24,0.9); border: 1px solid var(--line);
  border-radius: 10px; padding: 0.5rem 0.8rem;
}
.stDataFrame { background: rgba(18,17,24,0.9); }
.stTabs [data-baseweb="tab"] { color: var(--muted); }
.stTabs [aria-selected="true"] { color: var(--ink); border-bottom: 2px solid var(--accent); }
</style>""",
        unsafe_allow_html=True,
    )


PLOT_BG = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(18,17,24,0.9)",
    font=dict(color="#f2e6cf", family="Georgia, serif", size=12),
    xaxis=dict(gridcolor="#2a2630", linecolor="#2a2630", zerolinecolor="#2a2630"),
    yaxis=dict(gridcolor="#2a2630", linecolor="#2a2630", zerolinecolor="#2a2630"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#2a2630"),
    margin=dict(l=50, r=20, t=40, b=40),
)

COLOR_TRUE = "#c9973a"
COLOR_PRED = "#7eb8d4"
COLOR_PEAK_TRUE = "#f2e6cf"
COLOR_PEAK_PRED = "#7eb8d4"


# ── Cached loaders ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_dataset():
    return BaselineDataset(DATA_DIR)


@st.cache_resource
def get_model():
    if not os.path.exists(CKPT_PATH):
        return None
    return load_model(CKPT_PATH)


# ── Helpers ────────────────────────────────────────────────────────────────────
def spectrum_fig(omega, true_s, pred_s=None, true_freqs=None, pred_freqs=None,
                 pred_amps=None, title="Spectrum"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=omega, y=true_s, name="True",
        line=dict(color=COLOR_TRUE, width=2)
    ))
    if pred_s is not None:
        fig.add_trace(go.Scatter(
            x=omega, y=pred_s, name="Predicted (MLP)",
            line=dict(color=COLOR_PRED, width=1.5, dash="dash")
        ))
    if true_freqs is not None and len(true_freqs):
        y_marks = [float(true_s[np.argmin(np.abs(omega - w))]) for w in true_freqs]
        fig.add_trace(go.Scatter(
            x=true_freqs, y=y_marks,
            mode="markers", name=f"True peaks ({len(true_freqs)})",
            marker=dict(color=COLOR_PEAK_TRUE, size=6, symbol="circle-open", line=dict(width=1.5))
        ))
    if pred_freqs is not None and len(pred_freqs):
        fig.add_trace(go.Scatter(
            x=pred_freqs, y=pred_amps if pred_amps is not None else np.ones_like(pred_freqs),
            mode="markers", name=f"Pred peaks ({len(pred_freqs)})",
            marker=dict(color=COLOR_PEAK_PRED, size=8, symbol="triangle-up")
        ))
    fig.update_layout(
        title=title,
        xaxis_title="Frequency (a.u.)",
        yaxis_title="Intensity",
        height=380,
        **PLOT_BG,
    )
    return fig


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    apply_theme()

    st.title("⚗ Baseline Observatory")
    st.caption("Coulomb Matrix Eigenvalues  ·  MLP Spectrum Regression  ·  Direct Peak Extraction")

    dataset = get_dataset()
    model = get_model()
    names = [dataset.get_name(i) for i in range(len(dataset))]

    tabs = st.tabs(["Overview", "Data", "Training", "Inference", "Compare All"])

    # ── Overview ───────────────────────────────────────────────────────────────
    with tabs[0]:
        st.header("Approach")

        c1, c2, c3 = st.columns(3)
        c1.metric("Molecules", len(dataset))
        total_peaks = sum(len(dataset[i]["frequencies"]) for i in range(len(dataset)))
        c2.metric("Total Peaks", total_peaks)
        c3.metric("Model Ready", "✓" if model else "✗ (train first)")

        st.markdown("---")

        col_l, col_r = st.columns(2)

        with col_l:
            st.subheader("Pipeline")
            st.markdown("""
```
Atomic structure
  ↓
Coulomb matrix  M_ij = Z_i Z_j / |r_i - r_j|
  ↓
Sorted eigenvalues (16-dim, padded)
  → permutation invariant
  → rotation + translation invariant
  ↓
MLP  16 → 256 → 512 → 512 → 256 → 512
  → LayerNorm + GELU hidden layers
  → Softplus output (non-negative)
  ↓
Dense spectrum S(ω)  [512 pts, 0.01–5.0 a.u.]
  ↓
scipy.signal.find_peaks
  ↓
{ω_k, B_k} transition set
```
""")

        with col_r:
            st.subheader("Why this design")
            st.markdown("""
**No set-prediction problem.**
The GNN approach predicts an unordered variable-length set of peaks,
requiring Hungarian matching losses and slot probabilities.
This baseline sidesteps that entirely by regressing the spectrum
as a fixed-length vector.

**Coulomb eigenvalues are provably invariant.**
Any rotation or permutation of atoms produces the same feature vector.
No need for equivariant layers or message passing.

**When to use GNN instead.**
Once you have 50+ molecules, especially larger ones (>10 atoms),
the GNN locality prior becomes genuinely useful. This baseline
stays better for small molecules at small data scale.
""")

        st.markdown("---")
        st.subheader("Commands")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Train**")
            st.code("python -m baseline.train --data_dir data/processed --epochs 1000", language="bash")
        with c2:
            st.markdown("**Evaluate**")
            st.code("python -m baseline.evaluate --data_dir data/processed", language="bash")

    # ── Data ──────────────────────────────────────────────────────────────────
    with tabs[1]:
        st.header("Dataset Explorer")

        sel_data = st.selectbox("Molecule", names, key="data_sel")
        idx = names.index(sel_data)
        sample = dataset[idx]

        col_l, col_r = st.columns([1, 2])

        with col_l:
            st.subheader("Atomic Structure")
            z_arr = sample["atomic_numbers"].numpy()
            pos_arr = sample["positions"].numpy()
            df_atoms = pd.DataFrame({
                "Element": [ELEM_MAP.get(int(z), str(z)) for z in z_arr],
                "Z": z_arr.tolist(),
                "x (Bohr)": pos_arr[:, 0].round(4).tolist(),
                "y (Bohr)": pos_arr[:, 1].round(4).tolist(),
                "z (Bohr)": pos_arr[:, 2].round(4).tolist(),
            })
            st.dataframe(df_atoms, use_container_width=True, hide_index=True)

            st.subheader("Coulomb Eigenvalues")
            feats = coulomb_matrix_eigenvalues(z_arr, pos_arr)
            nonzero_feats = feats[feats > 1e-6]
            fig_feats = go.Figure(go.Bar(
                x=[f"λ{i + 1}" for i in range(len(nonzero_feats))],
                y=nonzero_feats.tolist(),
                marker_color=COLOR_TRUE,
            ))
            fig_feats.update_layout(
                title="Input features (sorted eigenvalues)",
                height=220,
                xaxis_title="Index",
                yaxis_title="Eigenvalue",
                **PLOT_BG,
            )
            st.plotly_chart(fig_feats, use_container_width=True)

        with col_r:
            st.subheader("True Spectrum")
            true_s = sample["spectrum"].numpy()
            true_freqs = sample["frequencies"].numpy()
            true_amps = sample["amplitudes"].numpy()

            fig_true = spectrum_fig(
                OMEGA_GRID, true_s, true_freqs=true_freqs,
                title=f"{sel_data} — True Lorentzian Spectrum",
            )
            st.plotly_chart(fig_true, use_container_width=True)

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("True Peaks", len(true_freqs))
            mc2.metric("ω min", f"{true_freqs.min():.3f} a.u.")
            mc3.metric("ω max", f"{true_freqs.max():.3f} a.u.")

            st.subheader("Peak Table (first 20)")
            df_peaks = pd.DataFrame({
                "ω_k (a.u.)": true_freqs[:20].round(5).tolist(),
                "B_k (amplitude)": true_amps[:20].tolist(),
            })
            st.dataframe(df_peaks, use_container_width=True, hide_index=True)

    # ── Training ──────────────────────────────────────────────────────────────
    with tabs[2]:
        st.header("Training")

        if os.path.exists(LOG_PATH):
            with open(LOG_PATH) as f:
                log = json.load(f)

            col_l, col_r = st.columns([3, 1])

            with col_l:
                fig_loss = go.Figure(go.Scatter(
                    x=log["epochs"],
                    y=log["loss"],
                    line=dict(color=COLOR_TRUE, width=1.5),
                    name="Log-MSE Loss",
                ))
                fig_loss.update_layout(
                    title="Training Loss",
                    xaxis_title="Epoch",
                    yaxis_title="Log-MSE",
                    height=350,
                    **PLOT_BG,
                )
                st.plotly_chart(fig_loss, use_container_width=True)

            with col_r:
                st.subheader("Run Config")
                cfg = log.get("config", {})
                for k, v in cfg.items():
                    st.text(f"{k}: {v}")
                st.markdown("---")
                st.metric("Final Loss", f"{log['loss'][-1]:.6f}")
                st.metric("Best Loss", f"{min(log['loss']):.6f}")
                st.metric("Epochs Run", len(log["epochs"]))
        else:
            st.info("No training log found.")
            st.code("python -m baseline.train --data_dir data/processed --epochs 1000", language="bash")

        st.markdown("---")
        st.subheader("Model Architecture")
        m_display = SpectrumMLP(input_dim=MAX_ATOMS)
        n_params = sum(p.numel() for p in m_display.parameters())
        st.code(str(m_display), language="text")
        st.metric("Parameters", f"{n_params:,}")

    # ── Inference ─────────────────────────────────────────────────────────────
    with tabs[3]:
        st.header("Inference")

        if model is None:
            st.warning("No trained checkpoint found at `baseline/checkpoints/best_model.pth`.")
            st.code("python -m baseline.train --data_dir data/processed --epochs 1000", language="bash")
            st.stop()

        col_ctrl, col_main = st.columns([1, 3])

        with col_ctrl:
            sel_inf = st.selectbox("Molecule", names, key="inf_sel")
            st.markdown("**Peak detection**")
            min_h = st.slider("Height threshold (fraction of max)", 0.005, 0.20, 0.02, 0.005)
            min_d = st.slider("Min separation (grid points)", 2, 30, 8)
            show_residual = st.checkbox("Show residual (pred − true)", value=False)

        idx = names.index(sel_inf)
        sample = dataset[idx]

        pred_s = predict_spectrum(model, sample["features"])
        true_s = sample["spectrum"].numpy()
        true_freqs = sample["frequencies"].numpy()

        pred_freqs, pred_amps = extract_peaks_from_spectrum(
            pred_s, OMEGA_GRID, min_height_frac=min_h, min_distance=min_d
        )

        overlap = spectral_overlap(pred_s, true_s)
        freq_mae = matched_freq_mae(pred_freqs, true_freqs)

        with col_main:
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Spectral Overlap", f"{overlap:.4f}")
            mc2.metric("Freq MAE (matched)", f"{freq_mae:.4f}" if not np.isnan(freq_mae) else "N/A")
            mc3.metric("Pred Peaks", len(pred_freqs))
            mc4.metric("True Peaks", len(true_freqs))

            fig_inf = spectrum_fig(
                OMEGA_GRID, true_s, pred_s=pred_s,
                true_freqs=true_freqs,
                pred_freqs=pred_freqs, pred_amps=pred_amps,
                title=f"{sel_inf} — Predicted vs True",
            )
            st.plotly_chart(fig_inf, use_container_width=True)

            if show_residual:
                residual = pred_s - true_s
                fig_res = go.Figure(go.Scatter(
                    x=OMEGA_GRID, y=residual,
                    line=dict(color="#e07070", width=1.2), name="Residual"
                ))
                fig_res.add_hline(y=0, line_color="#2a2630")
                fig_res.update_layout(
                    title="Residual (Predicted − True)",
                    xaxis_title="Frequency (a.u.)",
                    yaxis_title="ΔS(ω)",
                    height=220,
                    **PLOT_BG,
                )
                st.plotly_chart(fig_res, use_container_width=True)

            if len(pred_freqs) > 0 and len(true_freqs) > 0:
                st.subheader("Matched Peaks")
                from scipy.optimize import linear_sum_assignment
                cost = np.abs(pred_freqs[:, None] - true_freqs[None, :])
                row_idx, col_idx = linear_sum_assignment(cost)
                df_match = pd.DataFrame({
                    "Pred ω (a.u.)": pred_freqs[row_idx].round(5).tolist(),
                    "True ω (a.u.)": true_freqs[col_idx].round(5).tolist(),
                    "|Δω|": np.abs(pred_freqs[row_idx] - true_freqs[col_idx]).round(5).tolist(),
                })
                st.dataframe(df_match, use_container_width=True, hide_index=True)

    # ── Compare All ───────────────────────────────────────────────────────────
    with tabs[4]:
        st.header("All Molecules Summary")

        if model is None:
            st.warning("Train the model first to see inference results.")
            st.stop()

        st.subheader("Inference parameters")
        c1, c2 = st.columns(2)
        cmp_min_h = c1.slider("Height threshold", 0.005, 0.20, 0.02, 0.005, key="cmp_h")
        cmp_min_d = c2.slider("Min separation (pts)", 2, 30, 8, key="cmp_d")

        rows = []
        figs = []

        for i, name in enumerate(names):
            s = dataset[i]
            ps = predict_spectrum(model, s["features"])
            ts = s["spectrum"].numpy()
            tf = s["frequencies"].numpy()
            pf, pa = extract_peaks_from_spectrum(ps, OMEGA_GRID, cmp_min_h, cmp_min_d)

            ov = spectral_overlap(ps, ts)
            mae = matched_freq_mae(pf, tf)
            rows.append({
                "Molecule": name,
                "Overlap": round(ov, 4),
                "Freq MAE": round(mae, 4) if not np.isnan(mae) else None,
                "Pred Peaks": len(pf),
                "True Peaks": len(tf),
            })
            figs.append((name, ts, ps, tf, pf, pa))

        df_all = pd.DataFrame(rows)
        st.dataframe(df_all, use_container_width=True, hide_index=True)

        st.subheader("Spectra")
        cols = st.columns(min(len(figs), 3))
        for k, (name, ts, ps, tf, pf, pa) in enumerate(figs):
            with cols[k % len(cols)]:
                fig = spectrum_fig(
                    OMEGA_GRID, ts, pred_s=ps,
                    true_freqs=tf, pred_freqs=pf, pred_amps=pa,
                    title=name,
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
