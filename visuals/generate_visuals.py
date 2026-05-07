#!/usr/bin/env python3
"""
generate_visuals.py
Generate 8 original professional SVG diagrams for the Electron-GNN project.

Run:
    python visuals/generate_visuals.py
    # or from repo root:
    .venv/bin/python visuals/generate_visuals.py

Outputs (same directory as this script):
    01_pipeline_architecture.svg
    02_gnn_model_internals.svg
    03_two_tower_hybrid.svg
    04_lorentzian_spectrum.svg
    05_pade_complex_plane.svg
    06_molecular_graph.svg
    07_hungarian_matching.svg
    08_loss_decomposition.svg
"""

import math, os
import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))

# ── Colour tokens ──────────────────────────────────────────────────────────
BG  = "#0d1117"; BG2 = "#161b22"; BG3 = "#21262d"
BD  = "#30363d"; TX  = "#e6edf3"; TX2 = "#8b949e"; TX3 = "#484f58"
BLU = "#58a6ff"; BLU2 = "#1f6feb"
GRN = "#3fb950"; GRN2 = "#196c2e"
RED = "#f85149"; RED2 = "#b91c1c"
ORG = "#e3b341"; ORG2 = "#9e6a03"
PUR = "#a371f7"
CYN = "#76e3ea"; CYN2 = "#0e7490"
PNK = "#ff7b72"


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ── Shared SVG defs (gradients + markers + filters) ───────────────────────
DEFS = (
    "<defs>\n"
    "<style>text{font-family:'Inter','Segoe UI',Arial,sans-serif}"
    ".mono{font-family:'JetBrains Mono','Fira Code','Courier New',monospace}"
    "</style>\n"
    f'<linearGradient id="gbg" x1="0" y1="0" x2="1" y2="1">'
    f'<stop offset="0%" stop-color="{BG}"/><stop offset="100%" stop-color="{BG2}"/></linearGradient>\n'
    f'<linearGradient id="gblue" x1="0" y1="0" x2="0" y2="1">'
    f'<stop offset="0%" stop-color="{BLU2}"/><stop offset="100%" stop-color="#0d419d"/></linearGradient>\n'
    f'<linearGradient id="ggreen" x1="0" y1="0" x2="0" y2="1">'
    f'<stop offset="0%" stop-color="#2ea043"/><stop offset="100%" stop-color="#196c2e"/></linearGradient>\n'
    f'<linearGradient id="gpurple" x1="0" y1="0" x2="0" y2="1">'
    f'<stop offset="0%" stop-color="#8957e5"/><stop offset="100%" stop-color="#553098"/></linearGradient>\n'
    f'<linearGradient id="gorange" x1="0" y1="0" x2="0" y2="1">'
    f'<stop offset="0%" stop-color="{ORG2}"/><stop offset="100%" stop-color="#5a3e00"/></linearGradient>\n'
    f'<linearGradient id="gred" x1="0" y1="0" x2="0" y2="1">'
    f'<stop offset="0%" stop-color="{RED2}"/><stop offset="100%" stop-color="#7f1d1d"/></linearGradient>\n'
    f'<linearGradient id="gcyan" x1="0" y1="0" x2="0" y2="1">'
    f'<stop offset="0%" stop-color="{CYN2}"/><stop offset="100%" stop-color="#164e63"/></linearGradient>\n'
    f'<linearGradient id="gteal" x1="0" y1="0" x2="0" y2="1">'
    f'<stop offset="0%" stop-color="#0f766e"/><stop offset="100%" stop-color="#134e4a"/></linearGradient>\n'
    f'<linearGradient id="gdark" x1="0" y1="0" x2="0" y2="1">'
    f'<stop offset="0%" stop-color="{BG2}"/><stop offset="100%" stop-color="{BG}"/></linearGradient>\n'
    f'<marker id="ablue"   markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0,10 3.5,0 7" fill="{BLU}"/></marker>\n'
    f'<marker id="agreen"  markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0,10 3.5,0 7" fill="{GRN}"/></marker>\n'
    f'<marker id="aorange" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0,10 3.5,0 7" fill="{ORG}"/></marker>\n'
    f'<marker id="ared"    markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0,10 3.5,0 7" fill="{RED}"/></marker>\n'
    f'<marker id="apurple" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0,10 3.5,0 7" fill="{PUR}"/></marker>\n'
    f'<marker id="acyan"   markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0,10 3.5,0 7" fill="{CYN}"/></marker>\n'
    f'<marker id="adim"    markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0,10 3.5,0 7" fill="{TX2}"/></marker>\n'
    '<filter id="sh" x="-20%" y="-20%" width="140%" height="140%">'
    '<feDropShadow dx="0" dy="3" stdDeviation="5" flood-opacity="0.6"/></filter>\n'
    '<filter id="glow" x="-30%" y="-30%" width="160%" height="160%">'
    '<feGaussianBlur in="SourceAlpha" stdDeviation="5" result="b"/>'
    '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>\n'
    "</defs>\n"
)


def O(w, h):
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">\n'
        f'{DEFS}'
        f'<rect width="{w}" height="{h}" fill="url(#gbg)"/>\n'
    )

def X(): return "</svg>\n"

# ── Primitives ─────────────────────────────────────────────────────────────
def R(x, y, w, h, fill=BG2, stroke=BD, rx=8, sw=1.5, op=1, sh=False):
    e = ' filter="url(#sh)"' if sh else ""
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" opacity="{op}"{e}/>\n'

def T(x, y, s, sz=13, fill=TX, a="middle", w="normal"):
    return f'<text x="{x}" y="{y}" font-size="{sz}" fill="{fill}" text-anchor="{a}" font-weight="{w}">{esc(s)}</text>\n'

def TM(x, y, s, sz=11, fill=TX2):
    return f'<text x="{x}" y="{y}" font-size="{sz}" fill="{fill}" text-anchor="middle" class="mono">{esc(s)}</text>\n'

def L(x1, y1, x2, y2, color=BD, sw=1.5, mk="", dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    m = f' marker-end="url(#a{mk})"' if mk else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{d}{m}/>\n'

def C(cx, cy, rad, fill=BLU, stroke="none", sw=0, op=1):
    return f'<circle cx="{cx}" cy="{cy}" r="{rad}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" opacity="{op}"/>\n'

def PL(points, stroke=BLU, sw=2, fill="none", op=1):
    pts = " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
    return f'<polyline points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" stroke-linejoin="round" stroke-linecap="round" opacity="{op}"/>\n'

def ARROWPATH(d, stroke=BLU, sw=2, mk="blue"):
    return f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{sw}" marker-end="url(#a{mk})"/>\n'


def save(fn, content):
    path = os.path.join(DIR, fn)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    kb = len(content) / 1024
    print(f"  [OK]  {fn}  ({kb:.1f} KB)")


# ══════════════════════════════════════════════════════════════════════════════
# SVG 01 — Pipeline Architecture
# ══════════════════════════════════════════════════════════════════════════════
def make_01_pipeline():
    W, H = 1450, 660
    s = O(W, H)

    # Header bar
    s += R(0, 0, W, 88, fill=BG2, stroke=BD, rx=0, sw=0)
    s += L(0, 88, W, 88, BD, sw=1)
    s += T(W // 2, 38, "End-to-End Pipeline: RT-TDDFT  →  GNN Training  →  Spectrum Prediction", sz=21, w="700")
    s += T(W // 2, 68, "Electron-GNN  ·  Quantum spectroscopy accelerated by equivariant graph neural networks", sz=12, fill=TX2)

    cw, ch = 165, 215
    gap = 48
    xs = [15 + i * (cw + gap) for i in range(7)]
    cy = 128

    # Phase panels
    phase_data = [
        (xs[0] - 8,  cy - 28, 3 * (cw + gap) - gap + 16, ch + 66, ORG, "gorange", "⚛  Phase A — Quantum Physics"),
        (xs[3] - 8,  cy - 28, 2 * (cw + gap) - gap + 16, ch + 66, BLU, "gblue",   "🤖  Phase B — Machine Learning"),
        (xs[5] - 8,  cy - 28, 2 * (cw + gap) - gap + 16, ch + 66, GRN, "ggreen",  "📈  Phase C — Inference & Output"),
    ]
    for px, py, pw, ph, border, grad, plabel in phase_data:
        s += R(px, py, pw, ph, fill=f"url(#{grad})", stroke=border, rx=14, sw=1.5, op=0.13)
        s += T(px + pw // 2, py + 19, plabel, sz=10, fill=border, w="700")

    stages = [
        ("gorange", "⚡", "RT-TDDFT Run",      "ReSpect v3+",            "Laser pulse excites electrons", ".out  .xyz files"),
        ("gorange", "📡", "Signal Parser",     "scripts/parser.py",      "Extract μ(t) dipole trajectory", "times[], dipole[], pos[]"),
        ("gorange", "🔬", "Peak Extraction",   "Padé + K-Means + LASSO", "Find quantum transitions {ω,B}", "data/processed/*.pt"),
        ("gblue",   "🕸", "Graph Builder",     "molecule_graph.py",      "Atoms + bonds → PyG Data", "Data(x, edge_index, edge_attr)"),
        ("gpurple", "🧠", "Two-Tower GNN",     "train_v3_two_tower.py",  "V1 freq tower + V2 amp tower", "checkpoints/*.pth"),
        ("gcyan",   "⚙", "Hybrid Inference",  "utils/hybrid_inference", "V1 freq + V2 amp → Hungarian", "{ω_k, B_k} peak set"),
        ("ggreen",  "📊", "Dashboard",         "dashboard/app.py",       "Streamlit comparison + reports", "Lorentzian spectrum"),
    ]

    arrow_y = cy + ch // 2 + 12

    for i, (x0, (grad, icon, title, sub, desc, out)) in enumerate(zip(xs, stages)):
        cx_ = x0 + cw // 2
        s += R(x0, cy, cw, ch, fill=f"url(#{grad})", stroke=BD, rx=12, sw=1.5, sh=True)
        # Icon background circle
        s += C(cx_, cy + 34, 20, fill="rgba(0,0,0,0.3)")
        s += T(cx_, cy + 40, icon, sz=20)
        s += T(cx_, cy + 68, title, sz=13, w="700")
        s += TM(cx_, cy + 86, sub, sz=9, fill=TX3)
        s += T(cx_, cy + 107, desc, sz=10, fill=TX2)
        s += L(x0 + 14, cy + 120, x0 + cw - 14, cy + 120, BD, sw=0.5)
        s += T(cx_, cy + 133, "Output:", sz=8, fill=TX3)
        s += TM(cx_, cy + 151, out, sz=9, fill=CYN)

        if i < 6:
            arr_cols  = [ORG, ORG, ORG, BLU, PUR, CYN]
            arr_marks = ["orange", "orange", "orange", "blue", "purple", "cyan"]
            nx = xs[i + 1]
            s += L(x0 + cw + 2, arrow_y, nx - 2, arrow_y, arr_cols[i], sw=2.5, mk=arr_marks[i])

    # Data labels on arrows
    flow_labels = ["μ(t), r_i, Z_i", "{ω_k, B_k}", ".pt tensors", "PyG batch", ".pth ckpts", "{ω, B} peaks"]
    for i, label in enumerate(flow_labels):
        mid_x = xs[i] + cw + gap // 2
        s += T(mid_x, arrow_y + 18, label, sz=8, fill=TX3)

    # Bottom legend
    ly = cy + ch + 55
    s += R(15, ly, 1420, 55, fill=BG2, stroke=BD, rx=8, sw=1)
    legend_items = [
        (ORG, "ReSpect (.out/.xyz)  — Raw quantum chemistry output"),
        (BLU, "PyG Data (.pt)  — Serialised molecular graphs + spectral targets"),
        (PUR, "PyTorch (.pth)  — Trained model checkpoints"),
        (GRN, "Plotly / Streamlit  — Interactive dashboard and report plots"),
    ]
    s += T(40, ly + 20, "Data types:", sz=10, fill=TX2, a="start")
    for j, (col, label) in enumerate(legend_items):
        lx = 140 + j * 330
        s += C(lx, ly + 16, 5, fill=col)
        s += T(lx + 12, ly + 21, label, sz=9, fill=TX2, a="start")

    s += T(W // 2, H - 12, "github.com/Galabavamsi/Electron-GNN  ·  RT-TDDFT + Equivariant GNN", sz=9, fill=TX3)
    s += X()
    return s


# ══════════════════════════════════════════════════════════════════════════════
# SVG 02 — GNN Model Architecture
# ══════════════════════════════════════════════════════════════════════════════
def make_02_gnn_architecture():
    W, H = 1120, 760
    s = O(W, H)

    s += R(0, 0, W, 78, fill=BG2, stroke=BD, rx=0, sw=0)
    s += L(0, 78, W, 78, BD, sw=1)
    s += T(W // 2, 35, "SpectralEquivariantGNN  —  V2 / V3 Amplitude Tower Architecture", sz=19, w="700")
    s += T(W // 2, 60, "GATv2 encoder + DETR-style set decoder  ·  Input: molecular graph  →  Output: {prob, freq, amp, count} per slot", sz=11, fill=TX2)

    # ── Left: Encoder ──────────────────────────────────────────────────────
    EX, EW = 35, 455   # encoder column x, width

    # 1. Input
    s += R(EX, 98, EW, 60, fill="url(#gdark)", stroke=BD, rx=8, sw=1.5)
    s += T(EX + EW // 2, 122, "Input: Molecular Graph", sz=13, w="700")
    s += TM(EX + EW // 2, 143, "x: (N,5)  edge_index: (2,E)  edge_attr: (E,4)  batch: (N,)", sz=10)

    # 2. Node emb + Edge emb side by side
    NX, NW, NH = EX, 215, 82
    s += R(NX, 185, NW, NH, fill="url(#gblue)", stroke=BLU, rx=8, sw=1.5)
    s += T(NX + NW // 2, 208, "Node Embedding", sz=12, w="700")
    s += TM(NX + NW // 2, 226, "Linear(5→128)", sz=10)
    s += TM(NX + NW // 2, 242, "SiLU  ·  LayerNorm", sz=10)
    s += TM(NX + NW // 2, 258, "→ h: (N, 128)", sz=10, fill=CYN)

    EX2 = NX + NW + 15
    NW2 = EX + EW - EX2
    s += R(EX2, 185, NW2, NH, fill="url(#gpurple)", stroke=PUR, rx=8, sw=1.5)
    s += T(EX2 + NW2 // 2, 208, "Edge Embedding", sz=12, w="700")
    s += TM(EX2 + NW2 // 2, 226, "Linear(4→128) + SiLU", sz=10)
    s += TM(EX2 + NW2 // 2, 242, "Linear + LayerNorm", sz=10)
    s += TM(EX2 + NW2 // 2, 258, "→ e: (E, 128)", sz=10, fill=PUR)

    # 3. GATv2 × 4 block
    GAY = 295
    s += R(EX, GAY, EW, 165, fill="url(#ggreen)", stroke=GRN, rx=10, sw=1.5)
    s += T(EX + EW // 2, GAY + 22, "GATv2Conv  ×  4  layers  (with Residual + GELU)", sz=13, w="700")
    # Individual layer boxes
    for li in range(4):
        lx = EX + 10 + li * 109
        s += R(lx, GAY + 35, 98, 50, fill=BG3, stroke=GRN2, rx=6, sw=1)
        s += T(lx + 49, GAY + 56, f"Layer {li + 1}", sz=11, w="600")
        s += TM(lx + 49, GAY + 75, "Attn 4 heads", sz=9, fill=TX2)
    s += T(EX + EW // 2, GAY + 110, "in: (N,128)  +  edge (E,128)  →  out: (N,128)  +  residual", sz=10, fill=TX2)
    s += TM(EX + EW // 2, GAY + 130, "h = GELU(GATv2(h, edge_index, e)) + h_res", sz=10, fill=GRN)
    s += T(EX + EW // 2, GAY + 148, "LayerNorm applied after each layer", sz=9, fill=TX3)

    # 4. Global Pooling
    GPY = 487
    s += R(EX, GPY, EW, 75, fill="url(#gorange)", stroke=ORG, rx=8, sw=1.5)
    s += T(EX + EW // 2, GPY + 22, "Global Pooling  →  Context Vector", sz=13, w="700")
    s += TM(EX + EW // 2, GPY + 43, "mean_pool(h, batch)  ||  max_pool(h, batch)  →  cat: (B, 256)", sz=10)
    s += TM(EX + EW // 2, GPY + 60, "Linear(256→128) + SiLU + LN  →  context: (B, 128)", sz=10, fill=ORG)

    # Encoder vertical arrows
    s += L(EX + EW // 2, 158, EX + EW // 2, 183, BLU, sw=2, mk="blue")
    s += L(EX + EW // 2, 267, EX + EW // 2, 293, GRN, sw=2, mk="green")
    s += L(EX + EW // 2, 460, EX + EW // 2, 485, ORG, sw=2, mk="orange")

    # ── Right: Decoder ─────────────────────────────────────────────────────
    DX, DW = 610, 475

    # 5. K Queries
    s += R(DX, 185, DW, 75, fill="url(#gcyan)", stroke=CYN, rx=8, sw=1.5)
    s += T(DX + DW // 2, 210, "Learnable Query Tokens", sz=13, w="700")
    s += TM(DX + DW // 2, 230, "query_embed: nn.Parameter  →  shape: (K_max=64, 128)", sz=10)
    s += TM(DX + DW // 2, 249, "Expanded to: (B, 64, 128)  per batch", sz=10, fill=CYN)

    # 6. TransformerDecoder
    TDY = 288
    s += R(DX, TDY, DW, 95, fill="url(#gblue)", stroke=BLU, rx=8, sw=1.5)
    s += T(DX + DW // 2, TDY + 22, "TransformerDecoder  (2 layers, 4 heads)", sz=13, w="700")
    s += TM(DX + DW // 2, TDY + 43, "tgt=queries  ·  memory=dense_nodes  (from GATv2)", sz=10)
    s += TM(DX + DW // 2, TDY + 60, "memory_key_padding_mask = ~node_mask", sz=10)
    s += TM(DX + DW // 2, TDY + 76, "slot_features: (B, 64, 128)  cross-attends over graph nodes", sz=10, fill=BLU)

    # 7. Slot Refinement
    SRY = 410
    s += R(DX, SRY, DW, 75, fill="url(#gpurple)", stroke=PUR, rx=8, sw=1.5)
    s += T(DX + DW // 2, SRY + 22, "Slot Refinement", sz=13, w="700")
    s += TM(DX + DW // 2, SRY + 43, "cat(slot_features, context expanded)  →  (B,64,256)", sz=10)
    s += TM(DX + DW // 2, SRY + 60, "Linear(256→128) + GELU  →  (B, K_max, 128)", sz=10, fill=PUR)

    # 8. Output Heads (4 boxes)
    OHY = 515
    s += R(DX, OHY, DW, 165, fill=BG3, stroke=BD, rx=10, sw=1.5)
    s += T(DX + DW // 2, OHY + 20, "Output Heads  (applied independently to each slot)", sz=12, w="700")
    head_data = [
        (BLU,  "head_prob",  "Linear(128→1)", "sigmoid → p ∈ (0,1)",   "Slot existence probability"),
        (GRN,  "head_freq",  "MLP(128→1)",    "softplus + 1e-5 → ω",    "Transition frequency [a.u.]"),
        (ORG,  "head_amp",   "MLP(128→1)",    "softplus × amp_scale",   "Dipole amplitude B_k [a.u.]"),
        (RED,  "head_count", "MLP(128→1)*",   "softplus → N̂",           "Predicted peak count  *global"),
    ]
    for j, (col, hname, arch, output, meaning) in enumerate(head_data):
        hy = OHY + 35 + j * 30
        s += R(DX + 10, hy, 120, 23, fill=col, stroke="none", rx=4, sw=0, op=0.2)
        s += T(DX + 70, hy + 15, hname, sz=10, fill=col, w="600")
        s += TM(DX + 200, hy + 15, arch, sz=9, fill=TX2)
        s += T(DX + 330, hy + 15, output, sz=9, fill=col, a="middle")
        s += T(DX + DW - 10, hy + 15, meaning, sz=9, fill=TX3, a="end")

    # Decoder vertical arrows
    s += L(DX + DW // 2, 260, DX + DW // 2, 286, CYN, sw=2, mk="cyan")
    s += L(DX + DW // 2, 383, DX + DW // 2, 408, PUR, sw=2, mk="purple")
    s += L(DX + DW // 2, 485, DX + DW // 2, 513, ORG, sw=2, mk="orange")

    # ── Cross-column connections ────────────────────────────────────────────
    # GATv2 → TransformerDecoder (cross-attention, horizontal)
    cc_y1 = GAY + 165 // 2   # mid of GATv2 block
    cc_y2 = TDY + 95 // 2    # mid of TransformerDecoder
    s += L(EX + EW, cc_y1, DX, cc_y2, GRN, sw=2, mk="green", dash="6,3")
    s += T((EX + EW + DX) // 2, (cc_y1 + cc_y2) // 2 - 8, "node memory", sz=9, fill=GRN)

    # Pooling → Slot Refine (context)
    cp_y1 = GPY + 37
    cp_y2 = SRY + 37
    mid_x  = (EX + EW + DX) // 2
    s += ARROWPATH(f"M{EX+EW},{cp_y1} H{mid_x} V{cp_y2} H{DX}", stroke=ORG, sw=2, mk="orange")
    s += T(mid_x, cp_y1 - 8, "context (B,128)", sz=9, fill=ORG)

    s += T(W // 2, H - 14, "K_max = 64 slots  ·  hidden_dim = 128  ·  4 GATv2 layers  ·  2 decoder layers  ·  4 attention heads", sz=9, fill=TX3)
    s += X()
    return s


# ══════════════════════════════════════════════════════════════════════════════
# SVG 03 — Two-Tower Hybrid Inference
# ══════════════════════════════════════════════════════════════════════════════
def make_03_two_tower():
    W, H = 1080, 680
    s = O(W, H)

    s += R(0, 0, W, 78, fill=BG2, stroke=BD, rx=0, sw=0)
    s += L(0, 78, W, 78, BD, sw=1)
    s += T(W // 2, 35, "Two-Tower Hybrid Inference Stack", sz=20, w="700")
    s += T(W // 2, 60, "V1 frequency prior  +  V2 amplitude tower  →  Hungarian-matched hybrid output", sz=11, fill=TX2)

    # Shared input box at top center
    s += R(390, 95, 300, 52, fill="url(#gdark)", stroke=BD, rx=8, sw=1.5)
    s += T(540, 117, "Input: Molecular Graph", sz=13, w="700")
    s += TM(540, 136, "x:(N,5)  edge_index:(2,E)  edge_attr:(E,4)", sz=10)

    # Arrows from input to each tower
    s += L(490, 147, 310, 175, ORG, sw=2, mk="orange")
    s += L(590, 147, 770, 175, BLU, sw=2, mk="blue")

    TW, TH = 390, 350
    T1X, T2X = 30, 660
    TY = 178

    # V1 Tower outline
    s += R(T1X, TY, TW, TH, fill="url(#gorange)", stroke=ORG, rx=12, sw=2, op=0.18)
    s += R(T1X, TY, TW, TH, fill="none", stroke=ORG, rx=12, sw=2)
    s += R(T1X + 10, TY + 8, TW - 20, 30, fill=ORG, stroke="none", rx=6)
    s += T(T1X + TW // 2, TY + 28, "V1  —  Frequency Tower", sz=13, w="700", fill=BG)

    v1_blocks = [
        ("SpectralEquivariantGNNV1", "mace_net_v1.py", BG2, TX),
        ("Backbone: Linear GNN\n+ global_add_pool", "", BG3, TX2),
        ("K_max = 50  fixed output slots", "Simple scalar heads", BG3, TX2),
        ("head_freq: (B,50) → ω_k  [Softplus]", "", BG3, GRN),
        ("head_prob: (B,50) → p_k  [Sigmoid]", "", BG3, BLU),
    ]
    for bi, (text1, text2, bg, col) in enumerate(v1_blocks):
        by = TY + 48 + bi * 55
        s += R(T1X + 14, by, TW - 28, 44, fill=bg, stroke=BD, rx=6, sw=1)
        s += T(T1X + TW // 2, by + 18, text1, sz=11, fill=col, w="600")
        if text2:
            s += TM(T1X + TW // 2, by + 34, text2, sz=9, fill=TX3)

    # V2 Tower outline
    s += R(T2X, TY, TW, TH, fill="url(#gblue)", stroke=BLU, rx=12, sw=2, op=0.18)
    s += R(T2X, TY, TW, TH, fill="none", stroke=BLU, rx=12, sw=2)
    s += R(T2X + 10, TY + 8, TW - 20, 30, fill=BLU, stroke="none", rx=6)
    s += T(T2X + TW // 2, TY + 28, "V2  —  Amplitude Tower", sz=13, w="700", fill=BG)

    v2_blocks = [
        ("SpectralEquivariantGNN", "mace_net.py  (GATv2)", BG2, TX),
        ("GATv2Conv × 4  +  Set Decoder", "Transformer: 2L, 4H", BG3, TX2),
        ("K_max = 64  learnable query slots", "Cardinality-aware", BG3, TX2),
        ("head_amp: (B,64) → B_k  [Softplus × scale]", "", BG3, ORG),
        ("head_count: (B,) → N̂  [Softplus]", "count head", BG3, RED),
    ]
    for bi, (text1, text2, bg, col) in enumerate(v2_blocks):
        by = TY + 48 + bi * 55
        s += R(T2X + 14, by, TW - 28, 44, fill=bg, stroke=BD, rx=6, sw=1)
        s += T(T2X + TW // 2, by + 18, text1, sz=11, fill=col, w="600")
        if text2:
            s += TM(T2X + TW // 2, by + 34, text2, sz=9, fill=TX3)

    # Combiner
    CY = 558
    s += R(120, CY, 840, 78, fill="url(#gteal)", stroke=CYN, rx=12, sw=2)
    s += T(540, CY + 22, "Hybrid Combiner  (utils/hybrid_inference.py)", sz=14, w="700")
    steps = [
        "① N̂ from V2 count head",
        "② Top-N freq slots from V1 (by prob)",
        "③ Overflow freq from V2 if N̂ > K_max_v1",
        "④ Hungarian: match V2 amps to selected freqs",
        "⑤ Output: {ω_k, B_k, p_k}",
    ]
    for si, step in enumerate(steps):
        sx = 135 + si * 163
        s += T(sx, CY + 46, step, sz=9, fill=TX, a="start")
    s += T(540, CY + 65, "Final hybrid peak set — best of V1 frequencies + V2 amplitudes", sz=10, fill=CYN)

    # Arrows from towers to combiner
    s += L(T1X + TW // 2, TY + TH, T1X + TW // 2, CY - 2, ORG, sw=2.5, mk="orange")
    s += L(T2X + TW // 2, TY + TH, T2X + TW // 2, CY - 2, BLU, sw=2.5, mk="blue")

    # Output arrow
    s += L(540, CY + 78, 540, H - 30, GRN, sw=2.5, mk="green")
    s += R(390, H - 28, 300, 22, fill=GRN2, stroke=GRN, rx=6, sw=1)
    s += T(540, H - 13, "Reconstructed Absorption Spectrum: S(ω) = Σ B_k · γ / ((ω−ω_k)² + γ²)", sz=10, fill=TX)

    # Legend badges
    s += C(30, H - 20, 6, fill=ORG)
    s += T(44, H - 15, "Frequency tower  (V1)", sz=9, fill=TX2, a="start")
    s += C(200, H - 20, 6, fill=BLU)
    s += T(214, H - 15, "Amplitude tower  (V2)", sz=9, fill=TX2, a="start")
    s += C(380, H - 20, 6, fill=CYN)
    s += T(394, H - 15, "Hybrid combiner", sz=9, fill=TX2, a="start")

    s += X()
    return s


# ══════════════════════════════════════════════════════════════════════════════
# SVG 04 — Lorentzian Spectrum Physics
# ══════════════════════════════════════════════════════════════════════════════
def make_04_lorentzian():
    W, H = 960, 580
    s = O(W, H)

    s += R(0, 0, W, 75, fill=BG2, stroke=BD, rx=0, sw=0)
    s += L(0, 75, W, 75, BD, sw=1)
    s += T(W // 2, 33, "Lorentzian Spectral Reconstruction", sz=20, w="700")
    s += T(W // 2, 57, "S(ω) = Σₖ  Bₖ · γ / ((ω − ωₖ)² + γ²)    where  γ = 0.04 a.u.  (broadening)", sz=12, fill=TX2)

    # Plot area
    PX1, PX2 = 90, 880   # x range in SVG
    PY1, PY2 = 510, 95   # y range (PY1=bottom/zero, PY2=top/max)
    S_MAX = 28.0
    OM_MAX = 4.0

    def ox(om): return PX1 + (om / OM_MAX) * (PX2 - PX1)
    def oy(sv): return PY1 + (sv / S_MAX) * (PY2 - PY1)   # PY2 < PY1 so inverts

    # Plot background + border
    s += R(PX1 - 5, PY2 - 10, (PX2 - PX1) + 15, (PY1 - PY2) + 20,
           fill=BG2, stroke=BD, rx=6, sw=1)

    # Grid lines (horizontal)
    for sv in [0, 5, 10, 15, 20, 25]:
        gy = oy(sv)
        s += L(PX1, gy, PX2, gy, BD, sw=0.7, dash="4,4")
        s += T(PX1 - 8, gy + 5, str(sv), sz=9, fill=TX3, a="end")

    # Grid lines (vertical)
    for om in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
        gx = ox(om)
        s += L(gx, PY2, gx, PY1, BD, sw=0.7, dash="4,4")
        s += T(gx, PY1 + 18, f"{om:.1f}", sz=9, fill=TX3)

    # Axis lines
    s += L(PX1, PY1, PX2, PY1, TX2, sw=2)   # x-axis
    s += L(PX1, PY2, PX1, PY1, TX2, sw=2)   # y-axis

    # Axis labels
    s += T((PX1 + PX2) // 2, PY1 + 38, "ω  (atomic units)", sz=12, fill=TX2)
    s += T(PX1 - 55, (PY1 + PY2) // 2, "S(ω)", sz=12, fill=TX2)

    # Peaks
    peak_params = [
        (0.80, 0.72, 0.040, BLU,  "ω₁"),
        (1.62, 1.00, 0.040, GRN,  "ω₂"),
        (2.85, 0.52, 0.040, ORG,  "ω₃"),
    ]

    omega = np.linspace(0.001, OM_MAX, 1200)
    S_total = np.zeros_like(omega)

    for om_k, B_k, gam, col, lbl in peak_params:
        Sk = B_k * gam / ((omega - om_k) ** 2 + gam ** 2)
        S_total += Sk

        # Individual peak as dashed polyline
        pts = [(ox(om), oy(sv)) for om, sv in zip(omega, Sk) if PY2 < oy(sv) < PY1 + 5]
        if pts:
            s += PL(pts, stroke=col, sw=1.5, op=0.5)
            # Clip to plot area using SVG rect
            # peak annotation
            peak_x = ox(om_k)
            peak_y = oy(B_k / gam)
            s += C(peak_x, peak_y, 4, fill=col, stroke=TX, sw=1)
            s += T(peak_x, peak_y - 14, lbl, sz=11, fill=col, w="700")
            s += T(peak_x, peak_y - 28, f"ω={om_k:.2f}  B={B_k:.2f}", sz=9, fill=col)

    # Total spectrum (solid bright line)
    pts_total = [(ox(om), oy(sv)) for om, sv in zip(omega, S_total)]
    s += PL(pts_total, stroke=TX, sw=2.5, op=1.0)

    # Annotations: gamma bracket at ω₂ peak
    w2 = 1.62; g = 0.040; B2 = 1.00
    S_half = (B2 / g) / 2
    # Half-width at half-max: omega = w2 ± gamma
    x_fwhm_l = ox(w2 - g)
    x_fwhm_r = ox(w2 + g)
    y_half = oy(S_half)
    s += L(x_fwhm_l, y_half, x_fwhm_r, y_half, CYN, sw=1.5)
    s += L(x_fwhm_l, y_half - 5, x_fwhm_l, y_half + 5, CYN, sw=1.5)
    s += L(x_fwhm_r, y_half - 5, x_fwhm_r, y_half + 5, CYN, sw=1.5)
    s += T((x_fwhm_l + x_fwhm_r) // 2, y_half - 10, "2γ = FWHM", sz=9, fill=CYN)

    # Legend
    legend_items = [
        (BLU, "ω₁  individual peak  (dashed)"),
        (GRN, "ω₂  individual peak  (dashed)"),
        (ORG, "ω₃  individual peak  (dashed)"),
        (TX,  "S(ω)  total spectrum  (solid)"),
    ]
    LLX, LLY = PX2 - 260, PY2 + 10
    s += R(LLX - 10, LLY - 5, 270, len(legend_items) * 22 + 15,
           fill=BG, stroke=BD, rx=6, sw=1, op=0.9)
    for j, (col, label) in enumerate(legend_items):
        ly = LLY + 12 + j * 22
        if col == TX:
            s += L(LLX, ly - 4, LLX + 24, ly - 4, col, sw=2.5)
        else:
            s += L(LLX, ly - 4, LLX + 24, ly - 4, col, sw=1.5, dash="5,3")
        s += T(LLX + 30, ly, label, sz=10, fill=col, a="start")

    s += T(W // 2, H - 12,
           "Physical Lorentzians: each peak ωₖ has amplitude Bₖ (from LASSO) and broadening γ (instrument)",
           sz=9, fill=TX3)
    s += X()
    return s


# ══════════════════════════════════════════════════════════════════════════════
# SVG 05 — Padé Approximant Complex Plane
# ══════════════════════════════════════════════════════════════════════════════
def make_05_pade():
    W, H = 800, 800
    s = O(W, H)

    s += R(0, 0, W, 72, fill=BG2, stroke=BD, rx=0, sw=0)
    s += L(0, 72, W, 72, BD, sw=1)
    s += T(W // 2, 32, "Padé Approximant Poles in the Complex Plane", sz=19, w="700")
    s += T(W // 2, 56, "Physical resonances live on the upper half unit circle  ·  z = exp(iωΔt)", sz=11, fill=TX2)

    CX, CY, RC = 400, 430, 265

    # Upper half highlight
    # Draw as a filled semicircle arc
    # Arc from angle 0 to π (upper half of unit circle)
    pts_arc = []
    for a in np.linspace(0, math.pi, 80):
        px = CX + RC * math.cos(a)
        py = CY - RC * math.sin(a)   # SVG y inverted
        pts_arc.append((px, py))
    pts_arc.append((CX + RC, CY))   # close back to start
    pts_arc.append((CX - RC, CY))
    # Fill upper half
    arc_d = f"M {CX + RC:.1f} {CY:.1f} "
    arc_d += " ".join(f"L {px:.1f} {py:.1f}" for px, py in pts_arc[:80])
    arc_d += " Z"
    s += f'<path d="{arc_d}" fill="{BLU}" opacity="0.06" stroke="none"/>\n'
    s += T(CX, CY - RC // 2 - 30, "Physical domain", sz=10, fill=BLU, a="middle")
    s += T(CX, CY - RC // 2 - 14, "Im(z) > 0  ·  |z| ≈ 1", sz=9, fill=TX3)

    # Full unit circle
    s += f'<circle cx="{CX}" cy="{CY}" r="{RC}" fill="none" stroke="{BD}" stroke-width="1.5" stroke-dasharray="6,3"/>\n'
    s += f'<circle cx="{CX}" cy="{CY}" r="{RC}" fill="none" stroke="{BLU}" stroke-width="2" stroke-dasharray="none" opacity="0.4"/>\n'

    # Coordinate axes
    s += L(CX - RC - 30, CY, CX + RC + 30, CY, TX2, sw=1.5, mk="dim")   # Re axis
    s += L(CX, CY + RC + 30, CX, CY - RC - 30, TX2, sw=1.5, mk="dim")   # Im axis
    s += T(CX + RC + 42, CY + 5, "Re(z)", sz=11, fill=TX2)
    s += T(CX + 10, CY - RC - 38, "Im(z)", sz=11, fill=TX2, a="start")
    s += T(CX + 8, CY + 18, "0", sz=10, fill=TX3)
    s += T(CX + RC + 6, CY + 18, "+1", sz=9, fill=TX3)
    s += T(CX - RC - 14, CY + 18, "−1", sz=9, fill=TX3)
    s += T(CX + 8, CY - RC + 10, "+i", sz=9, fill=TX3)

    # ── Noise poles (scattered, near unit circle) ──────────────────────────
    rng = np.random.default_rng(seed=42)
    n_noise = 70
    angles_n = rng.uniform(0, 2 * math.pi, n_noise)
    radii_n  = rng.normal(1.0, 0.13, n_noise)
    for an, rn in zip(angles_n, radii_n):
        px = CX + RC * rn * math.cos(an)
        py = CY - RC * rn * math.sin(an)
        if 40 < px < W - 40 and 80 < py < H - 40:
            s += C(px, py, 3, fill=BG3, stroke=TX3, sw=1, op=0.7)

    # ── Physical poles (cluster centroids on upper semicircle) ─────────────
    phys_omegas = [0.48, 0.98, 1.50, 2.02, 2.54]
    phys_colors = [BLU, GRN, ORG, RED, PUR]
    phys_labels = ["ω₁", "ω₂", "ω₃", "ω₄", "ω₅"]

    for om_k, col, lbl in zip(phys_omegas, phys_colors, phys_labels):
        re = math.cos(om_k)
        im = math.sin(om_k)
        px = CX + RC * re
        py = CY - RC * im

        # Cluster halo
        s += C(px, py, 26, fill=col, stroke="none", op=0.12)
        s += C(px, py, 26, fill="none", stroke=col, sw=1, op=0.4)

        # Cluster scatter (small dots near centroid)
        rng2 = np.random.default_rng(seed=int(om_k * 100))
        sc_re = rng2.normal(re, 0.04, 5)
        sc_im = rng2.normal(im, 0.04, 5)
        for src, sim in zip(sc_re, sc_im):
            spx = CX + RC * src
            spy = CY - RC * sim
            s += C(spx, spy, 3.5, fill=col, op=0.5)

        # Centroid marker
        s += C(px, py, 7, fill=col, stroke=TX, sw=1.5)
        # Label above
        lx = px + (px - CX) * 0.22
        ly = py - (im * 22 + 15)
        s += T(lx, ly, lbl, sz=11, fill=col, w="700")
        s += T(lx, ly + 14, f"ω={om_k:.2f}", sz=8, fill=TX3)

    # |z|=1 label
    s += T(CX + RC * math.cos(math.pi / 4) + 18,
           CY - RC * math.sin(math.pi / 4) - 18,
           "|z| = 1", sz=10, fill=BD, a="start")

    # Legend
    s += R(22, H - 120, 360, 110, fill=BG2, stroke=BD, rx=8, sw=1)
    s += T(30, H - 100, "Legend:", sz=11, fill=TX, a="start", w="600")
    s += C(38, H - 80, 4, fill=BG3, stroke=TX3, sw=1)
    s += T(50, H - 75, "Noisy Padé poles  (mathematical artefacts)", sz=10, fill=TX2, a="start")
    s += C(38, H - 58, 7, fill=BLU, stroke=TX, sw=1)
    s += T(50, H - 53, "Physical pole  (cluster centroid on unit circle)", sz=10, fill=TX2, a="start")
    s += R(28, H - 40, 20, 12, fill=BLU, stroke="none", rx=3, op=0.25)
    s += T(55, H - 32, "K-means cluster region", sz=10, fill=TX2, a="start")

    s += T(W // 2, H - 12,
           "Step: Padé roots → K-means cluster upper half-circle → LASSO selects non-zero amplitudes",
           sz=9, fill=TX3)
    s += X()
    return s


# ══════════════════════════════════════════════════════════════════════════════
# SVG 06 — Molecule → Graph Construction
# ══════════════════════════════════════════════════════════════════════════════
def make_06_molecular_graph():
    W, H = 1140, 620
    s = O(W, H)

    s += R(0, 0, W, 75, fill=BG2, stroke=BD, rx=0, sw=0)
    s += L(0, 75, W, 75, BD, sw=1)
    s += T(W // 2, 34, "Molecular Graph Construction  —  NH₃ example", sz=20, w="700")
    s += T(W // 2, 58, "Atom positions (a.u.) and bonds → PyTorch Geometric Data object with node and edge features", sz=11, fill=TX2)

    # ── Divider ────────────────────────────────────────────────────────────
    s += L(550, 85, 550, H - 10, BD, sw=1.5, dash="5,4")
    s += T(275, 94, "Ball-and-Stick Geometry  (atomic units)", sz=11, fill=TX2)
    s += T(845, 94, "PyG Data Graph Representation", sz=11, fill=TX2)

    # ── Left: NH3 molecule ─────────────────────────────────────────────────
    # NH3 in simple screen coordinates (approximate 3D perspective)
    # N at center, 3 H arranged around it
    NP = (260, 290)    # N position
    HP = [
        (260, 430),    # H1 below
        (395, 195),    # H2 upper right
        (125, 195),    # H3 upper left
    ]
    atom_col = {"N": BLU,  "H": "#a0a0a0"}
    atom_rad = {"N": 28,   "H": 16}

    # Draw bonds first (below atoms)
    for hpos in HP:
        s += L(NP[0], NP[1], hpos[0], hpos[1], "#606060", sw=8)
    # Bond labels (distance)
    for i, hpos in enumerate(HP):
        mx = (NP[0] + hpos[0]) // 2
        my = (NP[1] + hpos[1]) // 2
        s += T(mx + 18, my, "1.91 a.u.", sz=8, fill=TX3)

    # Draw atoms
    for sym, pos, r_at in [("N", NP, 28)] + [("H", h, 16) for h in HP]:
        s += C(pos[0], pos[1], r_at + 4, fill="rgba(0,0,0,0.3)")
        s += C(pos[0], pos[1], r_at, fill=atom_col[sym], stroke=TX2, sw=1.5)
        s += T(pos[0], pos[1] + 5, sym, sz=12, fill=TX, w="700")

    # Atom coordinate labels
    coord_data = [
        ("N", NP, "(0.00, 0.00, 0.00)"),
        ("H₁", HP[0], "(0.00, −1.77, −0.72)"),
        ("H₂", HP[1], "(1.53, 0.89, −0.72)"),
        ("H₃", HP[2], "(−1.53, 0.89, −0.72)"),
    ]
    for sym, pos, coord in coord_data:
        dx = 45 if pos[0] > NP[0] else -45 if pos[0] < NP[0] else 0
        dy = 30 if pos[1] > NP[1] else -30
        tx = pos[0] + dx
        ty = pos[1] + dy
        s += T(tx, ty, f"{sym}: {coord}", sz=8, fill=TX3)

    # Axis indicator (small)
    s += L(70, 530, 120, 530, TX3, sw=1, mk="dim")
    s += L(70, 530, 70, 480, TX3, sw=1, mk="dim")
    s += T(126, 534, "x", sz=9, fill=TX3, a="start")
    s += T(74, 474, "y", sz=9, fill=TX3, a="start")
    s += T(70, 565, "NH₃  ·  cutoff = 5.0 a.u.", sz=9, fill=TX3)

    # ── Center arrow ───────────────────────────────────────────────────────
    s += ARROWPATH(f"M 460 310 C 510 310 510 310 535 310", stroke=CYN, sw=3, mk="cyan")
    s += T(497, 298, "build_molecule", sz=8, fill=CYN)
    s += T(497, 326, "_graph()", sz=8, fill=CYN)

    # ── Right: Graph representation ────────────────────────────────────────
    # Node positions (mirror of left panel)
    GNP  = (820, 290)
    GHP  = [(820, 430), (955, 195), (685, 195)]
    edge_color = "#505050"

    # Graph edges (thinner, gray)
    for ghp in GHP:
        s += L(GNP[0], GNP[1], ghp[0], ghp[1], edge_color, sw=3)

    # Node circles (styled differently — show as rings with feature info)
    # N node
    s += C(GNP[0], GNP[1], 32, fill="none", stroke=BLU, sw=2.5)
    s += C(GNP[0], GNP[1], 22, fill=BLU, op=0.2)
    s += T(GNP[0], GNP[1] + 5, "N", sz=13, fill=BLU, w="700")

    # H nodes
    for ghp in GHP:
        s += C(ghp[0], ghp[1], 22, fill="none", stroke=TX2, sw=2)
        s += C(ghp[0], ghp[1], 14, fill=TX2, op=0.2)
        s += T(ghp[0], ghp[1] + 5, "H", sz=12, fill=TX2, w="700")

    # Node feature annotations
    s += R(GNP[0] + 40, GNP[1] - 48, 160, 95, fill=BG2, stroke=BLU, rx=6, sw=1)
    s += T(GNP[0] + 120, GNP[1] - 35, "Node features x[N]:", sz=9, fill=TX2)
    s += TM(GNP[0] + 120, GNP[1] - 18, "one-hot(N) = [0,0,1,0,0]", sz=9, fill=BLU)
    s += T(GNP[0] + 120, GNP[1] + 0, "H/C/N/O/F encoding", sz=8, fill=TX3)
    s += T(GNP[0] + 120, GNP[1] + 18, "shape: (N_atoms, 5)", sz=8, fill=TX3)
    s += T(GNP[0] + 120, GNP[1] + 34, "  N=4  for NH₃", sz=8, fill=TX3)

    # Edge feature box (on one edge)
    emx = (GNP[0] + GHP[1][0]) // 2 + 15
    emy = (GNP[1] + GHP[1][1]) // 2
    s += R(emx + 10, emy - 45, 175, 90, fill=BG2, stroke=ORG, rx=6, sw=1)
    s += T(emx + 98, emy - 32, "Edge features:", sz=9, fill=TX2)
    s += TM(emx + 98, emy - 14, "edge_attr[E, 4]", sz=9, fill=ORG)
    s += T(emx + 98, emy + 4,  "[dist, Δx, Δy, Δz]", sz=9, fill=TX2)
    s += T(emx + 98, emy + 22, "1.91, Δx, Δy, Δz", sz=8, fill=TX3)
    s += T(emx + 98, emy + 38, "E = N×(N−1) = 12 edges", sz=8, fill=TX3)

    # Fully-connected note
    s += T(845, H - 40, "Fully connected within cutoff radius (no self-loops)", sz=9, fill=TX3)
    s += T(845, H - 24, "PyG heterogeneous batching supports variable N and E per molecule", sz=9, fill=TX3)

    s += X()
    return s


# ══════════════════════════════════════════════════════════════════════════════
# SVG 07 — Hungarian Bipartite Matching
# ══════════════════════════════════════════════════════════════════════════════
def make_07_hungarian():
    W, H = 1000, 610
    s = O(W, H)

    s += R(0, 0, W, 75, fill=BG2, stroke=BD, rx=0, sw=0)
    s += L(0, 75, W, 75, BD, sw=1)
    s += T(W // 2, 32, "Bipartite Matching Loss  —  Hungarian Algorithm", sz=19, w="700")
    s += T(W // 2, 57,
           "cost(i,j) = 10|ω_pred[i] − ω_true[j]| + |B_pred[i] − B_true[j]|    →    linear_sum_assignment (scipy)",
           sz=11, fill=TX2)

    # ── Predicted slots (left) ─────────────────────────────────────────────
    SX1, SY0, SW, SH, SGAP = 30, 100, 220, 38, 8
    s += T(SX1 + SW // 2, SY0 - 10, "Predicted Slots  (K_max = 64)", sz=11, fill=BLU, w="700")

    pred_slots = [
        (0.92, 0.82, 0.0018, True,  BLU),   # matched
        (0.87, 1.63, 0.0010, True,  GRN),
        (0.79, 2.86, 0.0006, True,  ORG),
        (0.71, 0.40, 0.0004, True,  PUR),
        (0.63, 1.20, 0.0003, True,  CYN),
        (0.41, 2.10, 0.0001, False, TX3),   # unmatched
        (0.28, 3.50, 0.0001, False, TX3),
        (0.12, 0.95, 0.0000, False, TX3),
    ]

    slot_centers = []
    for i, (prob, freq, amp, matched, col) in enumerate(pred_slots):
        sy = SY0 + i * (SH + SGAP)
        slot_centers.append((SX1 + SW, sy + SH // 2))
        fill = f"url(#g{'blue' if matched else 'dark'})"
        s += R(SX1, sy, SW, SH, fill=fill, stroke=col if matched else BD, rx=5, sw=1.2)
        # Probability bar
        bar_w = int(prob * 50)
        s += R(SX1 + 4, sy + 4, bar_w, SH - 8, fill=col if matched else TX3,
               stroke="none", rx=3, sw=0, op=0.4)
        s += T(SX1 + 60, sy + SH // 2 + 5, f"p={prob:.2f}", sz=9, fill=col if matched else TX3, a="start")
        s += TM(SX1 + 140, sy + SH // 2 + 5, f"ω={freq:.2f}", sz=9, fill=TX if matched else TX3)
        s += TM(SX1 + 200, sy + SH // 2 + 5, f"B={amp:.4f}", sz=8, fill=TX2 if matched else TX3)

    # Unmatched label
    s += T(SX1 + SW // 2, SY0 + 5 * (SH + SGAP) + SH + 5,
           "⋯  unmatched slots  (K_max − N_true remaining)", sz=8, fill=TX3)

    # ── True peaks (right) ────────────────────────────────────────────────
    TX1 = 760; TW2 = 200
    s += T(TX1 + TW2 // 2, SY0 - 10, "True Peaks  (N_true = 5)", sz=11, fill=GRN, w="700")

    true_peaks = [
        (0.80, 0.0019),
        (1.60, 0.0011),
        (2.90, 0.0006),
        (0.42, 0.0004),
        (1.18, 0.0003),
    ]

    true_centers = []
    for i, (freq, amp) in enumerate(true_peaks):
        ty = SY0 + i * (SH + SGAP) + (SH + SGAP) // 4
        true_centers.append((TX1, ty + SH // 2))
        s += R(TX1, ty, TW2, SH, fill="url(#ggreen)", stroke=GRN, rx=5, sw=1.2)
        s += TM(TX1 + 60, ty + SH // 2 + 5, f"ω={freq:.2f}", sz=10, fill=TX)
        s += TM(TX1 + 152, ty + SH // 2 + 5, f"B={amp:.4f}", sz=9, fill=TX2)

    # ── Matching lines ────────────────────────────────────────────────────
    # Optimal matching: slot 0→peak0, 1→1, 2→2, 3→3, 4→4
    match_colors = [BLU, GRN, ORG, PUR, CYN]
    for i, col in enumerate(match_colors):
        sx_c, sy_c = slot_centers[i]
        tx_c, ty_c = true_centers[i]
        # Curved path through center
        mx = (sx_c + tx_c) // 2
        s += f'<path d="M {sx_c} {sy_c} C {mx} {sy_c}, {mx} {ty_c}, {tx_c} {ty_c}" fill="none" stroke="{col}" stroke-width="1.8" stroke-opacity="0.7"/>\n'

    # Unmatched slots annotation (dashed to X)
    for i in range(5, 8):
        sx_c, sy_c = slot_centers[i]
        s += L(sx_c, sy_c, sx_c + 80, sy_c, TX3, sw=1, dash="4,4")
        s += T(sx_c + 95, sy_c + 4, "×  suppressed", sz=8, fill=TX3)

    # ── Cost matrix schematic ─────────────────────────────────────────────
    CMD_X, CMD_Y, CMW, CMH = 285, 100, 440, 235
    s += R(CMD_X, CMD_Y, CMW, CMH, fill=BG2, stroke=BD, rx=8, sw=1.5)
    s += T(CMD_X + CMW // 2, CMD_Y + 18, "Cost Matrix  C[K_max × N_true]", sz=11, fill=TX2, w="600")

    # Mini grid
    NR, NC = 6, 5
    cell = 32
    gx0 = CMD_X + (CMW - NC * cell) // 2 - 15
    gy0 = CMD_Y + 32

    # Column headers (true peaks)
    for j in range(NC):
        s += T(gx0 + j * cell + cell // 2, gy0 + 8, f"T{j+1}", sz=8, fill=GRN)

    # Row headers + cells
    match_map = {(0,0), (1,1), (2,2), (3,3), (4,4)}
    np_rng = np.random.default_rng(44)
    for i in range(NR):
        s += T(gx0 - 8, gy0 + 18 + i * cell, f"P{i+1}", sz=8, fill=BLU if i < 5 else TX3, a="end")
        for j in range(NC):
            cx_c = gx0 + j * cell
            cy_c = gy0 + 12 + i * cell
            is_match = (i, j) in match_map
            cost_val = 0.05 + np_rng.uniform(0, 0.2) if not is_match else np_rng.uniform(0, 0.04)
            intensity = min(1.0, cost_val / 0.25)
            # color from low cost (green) to high cost (red)
            if is_match:
                fill = "url(#ggreen)"
                sc = GRN
            else:
                r_i = int(200 * intensity); g_i = int(100 * (1 - intensity))
                fill = f"rgb({r_i+30},{g_i+30},{30})"
                sc = BD
            s += R(cx_c + 2, cy_c, cell - 3, cell - 3, fill=fill, stroke=sc, rx=3, sw=0.8, op=0.7)
            s += T(cx_c + cell // 2, cy_c + cell // 2 + 4, f"{cost_val:.2f}", sz=7, fill=TX)

    s += T(CMD_X + CMW // 2, CMD_Y + CMH - 12,
           "Green diagonal = optimal assignment  ·  scipy.optimize.linear_sum_assignment", sz=9, fill=TX3)

    # ── Loss terms shown below ─────────────────────────────────────────────
    LY = 365
    s += R(30, LY, W - 60, 145, fill=BG2, stroke=BD, rx=8, sw=1.5)
    s += T(W // 2, LY + 20, "Loss Computation After Matching", sz=12, fill=TX, w="700")
    loss_items = [
        (BLU, "L_freq",     "8.0 × SmoothL1( ω_pred[matched], ω_true )"),
        (ORG, "L_amp",      "8.0 × SmoothL1( log(B_pred[matched]), log(B_true) )"),
        (GRN, "L_prob",     "1.2 × BCE( p_pred, target_mask )  where target_mask[matched]=1"),
        (RED, "L_unmatched","1.0 × mean( B_pred[unmatched]² )  — suppress noisy slots"),
        (PUR, "L_sum",      "6.0 × SmoothL1( ΣB_pred, ΣB_true )  — total amplitude conservation"),
        (CYN, "L_count",    "0.5 × SmoothL1( count_head, N_true )  — cardinality supervision"),
    ]
    for j, (col, name, formula) in enumerate(loss_items):
        ly = LY + 38 + j * 20
        row = j // 2; col_idx = j % 2
        lx_base = 50 + col_idx * 460
        ly_base = LY + 38 + row * 30
        s += C(lx_base, ly_base + 4, 4, fill=col)
        s += T(lx_base + 12, ly_base + 9, f"{name}: {formula}", sz=9, fill=TX2, a="start")

    s += T(W // 2, H - 15,
           "Total bipartite loss = sum of all 6 terms  ·  The physical spectrum loss (λ=0.3) is added separately",
           sz=9, fill=TX3)
    s += X()
    return s


# ══════════════════════════════════════════════════════════════════════════════
# SVG 08 — Loss Function Decomposition
# ══════════════════════════════════════════════════════════════════════════════
def make_08_loss():
    W, H = 1040, 660
    s = O(W, H)

    s += R(0, 0, W, 75, fill=BG2, stroke=BD, rx=0, sw=0)
    s += L(0, 75, W, 75, BD, sw=1)
    s += T(W // 2, 32, "Multi-Term Loss Function Decomposition", sz=20, w="700")
    s += T(W // 2, 57,
           "L_total = L_bipartite + λ_spec · L_spectrum    where λ_spec = 0.3",
           sz=12, fill=TX2)

    # ── Bipartite matching loss terms ──────────────────────────────────────
    terms = [
        (8.0,  BLU,  "L_freq",      "Frequency Match",
         "SmoothL1(ω_pred[matched] − ω_true)",
         "Primary frequency alignment. Hungarian-matched pairs only.",
         "β = 0.02"),
        (8.0,  ORG,  "L_amp",       "Amplitude Match (Log-scale)",
         "SmoothL1(log1p(1e4·B_pred) − log1p(1e4·B_true))",
         "Log-scale suppresses large amplitude magnitudes. Stabilises training.",
         "β = 0.02  ·  scale = 1e4"),
        (6.0,  PUR,  "L_sum",       "Total Amplitude Conservation",
         "SmoothL1( ΣB_pred − ΣB_true )",
         "Ensures global spectral area (integrated oscillator strength) is preserved.",
         "β = 0.01"),
        (1.2,  GRN,  "L_prob",      "Existence Probability  (BCE)",
         "BinaryCrossEntropy( p_pred, target_mask )",
         "target_mask[matched_slots]=1, rest=0. Trains the slot existence head.",
         "Applied to all K_max slots"),
        (1.0,  RED,  "L_unmatched", "Unmatched Slot Suppression",
         "mean( B_pred[unmatched_slots]² )",
         "Drives non-assigned slot amplitudes toward zero. Prevents noisy predictions.",
         "L2 penalty on amplitude"),
        (0.5,  CYN,  "L_count",     "Cardinality Regression",
         "SmoothL1( count_head − N_true )",
         "Trains the global count head to predict number of peaks. Guides top-K decode.",
         "β = 1.0  ·  global pooling"),
    ]

    total_weight = sum(w for w, *_ in terms)
    BAR_MAX_W = 330
    ROW_H = 82
    START_Y = 92

    for i, (weight, col, key, name, formula, desc, note) in enumerate(terms):
        ry = START_Y + i * ROW_H
        # Row background
        s += R(18, ry, W - 36, ROW_H - 4, fill=BG2, stroke=BD, rx=7, sw=1, sh=False)
        # Left: colour badge + weight
        s += R(22, ry + 8, 55, ROW_H - 20, fill=col, stroke="none", rx=5, op=0.2)
        s += T(49, ry + ROW_H // 2 - 4, f"×{weight:.1f}", sz=14, fill=col, w="700")
        # Name
        s += T(90, ry + ROW_H // 2 - 10, name, sz=12, fill=col, a="start", w="700")
        s += T(90, ry + ROW_H // 2 + 7, key, sz=9, fill=TX3, a="start")
        # Formula (monospace)
        s += f'<text x="92" y="{ry + ROW_H // 2 + 23}" font-size="9" fill="{TX2}" text-anchor="start" class="mono">{esc(formula)}</text>\n'
        # Description
        s += T(92, ry + ROW_H // 2 + 37, desc, sz=9, fill=TX3, a="start")
        # Bar chart
        bx = W - BAR_MAX_W - 50
        bar_w = int((weight / total_weight) * BAR_MAX_W)
        s += R(bx, ry + 18, BAR_MAX_W, 20, fill=BG3, stroke=BD, rx=4, sw=0.5)
        s += R(bx, ry + 18, bar_w, 20, fill=col, stroke="none", rx=4, op=0.85)
        s += T(bx + BAR_MAX_W + 8, ry + 32, f"{weight / total_weight * 100:.0f}%", sz=10, fill=col, a="start", w="600")
        # Note
        s += T(bx + bar_w // 2, ry + 56, note, sz=8, fill=TX3)

    # ── Physics spectrum regularizer ────────────────────────────────────────
    SPY = START_Y + 6 * ROW_H + 10
    s += R(18, SPY, W - 36, 80, fill="url(#gcyan)", stroke=CYN, rx=8, sw=1.5, op=0.15)
    s += R(18, SPY, W - 36, 80, fill="none", stroke=CYN, rx=8, sw=1.5)
    s += T(W // 2, SPY + 20, "L_spectrum  —  Physics Regularizer  (weighted by λ_spec = 0.3)", sz=12, fill=CYN, w="700")
    spec_terms = [
        "L_signal: MSE[ Σ Bₖ sin(ωₖ t) · (pred vs true) ]  —  Time-domain dipole consistency",
        "L_spec:   MSE[ log(1 + 5000·S_pred(ω) ) − log(1 + 5000·S_true(ω)) ]  —  Log-Lorentzian spectrum",
        "L_area:   SmoothL1( Σ S_pred(ω) − Σ S_true(ω) )  —  Spectral area conservation",
    ]
    for j, st in enumerate(spec_terms):
        s += f'<text x="32" y="{SPY + 38 + j * 16}" font-size="9" fill="{TX2}" text-anchor="start" class="mono">{esc(st)}</text>\n'

    s += T(W // 2, H - 14,
           f"Total bipartite weight = {total_weight:.1f}  ·  Effective: L_total = L_bipartite/batch + 0.3·L_spectrum/batch",
           sz=9, fill=TX3)
    s += X()
    return s


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\nGenerating Electron-GNN SVG visuals...\n")
    saves = [
        ("01_pipeline_architecture.svg",  make_01_pipeline),
        ("02_gnn_model_internals.svg",    make_02_gnn_architecture),
        ("03_two_tower_hybrid.svg",       make_03_two_tower),
        ("04_lorentzian_spectrum.svg",    make_04_lorentzian),
        ("05_pade_complex_plane.svg",     make_05_pade),
        ("06_molecular_graph.svg",        make_06_molecular_graph),
        ("07_hungarian_matching.svg",     make_07_hungarian),
        ("08_loss_decomposition.svg",     make_08_loss),
    ]
    for fn, gen in saves:
        save(fn, gen())

    print(f"\nAll 8 SVGs written to: {DIR}\n")
