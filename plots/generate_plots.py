#!/usr/bin/env python3
"""Generate publication-quality plots for epistemic probe replication experiment."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import norm

# ─── Load data ───────────────────────────────────────────────────────────────
with open("/home/clawboy/research-epistemic-awareness/results/probe_results_full.json") as f:
    data = json.load(f)

# Ordered layers
layer_names = ["layer_5", "layer_10", "layer_15", "layer_20", "layer_25", "layer_30", "layer_31"]
layer_nums = [5, 10, 15, 20, 25, 30, 31]

# Extract arrays
tf_auroc   = [data["layers"][l]["training_free"]["auroc"] for l in layer_names]
lr_auroc   = [data["layers"][l]["logistic_regression"]["auroc_mean"] for l in layer_names]
lr_std     = [data["layers"][l]["logistic_regression"]["auroc_std"] for l in layer_names]
fact_auroc = [data["layers"][l]["factual_only"]["auroc_mean"] for l in layer_names]
fact_std   = [data["layers"][l]["factual_only"]["auroc_std"] for l in layer_names]
reas_auroc = [data["layers"][l]["reasoning_only"]["auroc_mean"] for l in layer_names]
reas_std   = [data["layers"][l]["reasoning_only"]["auroc_std"] for l in layer_names]

# ─── Global style ───────────────────────────────────────────────────────────
BG_DARK   = "#1a1a2e"
BG_AXES   = "#16213e"
GRID_CLR  = "#ffffff"
TXT_CLR   = "#e0e0e0"
BLUE      = "#4fc3f7"
ORANGE    = "#ffb74d"
GREEN     = "#81c784"
RED       = "#ef5350"
GRAY      = "#90a4ae"
ACCENT    = "#e94560"

plt.rcParams.update({
    "figure.facecolor": BG_DARK,
    "axes.facecolor": BG_AXES,
    "axes.edgecolor": "#444466",
    "text.color": TXT_CLR,
    "axes.labelcolor": TXT_CLR,
    "xtick.color": TXT_CLR,
    "ytick.color": TXT_CLR,
    "font.family": "sans-serif",
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 15,
    "legend.fontsize": 12,
    "legend.framealpha": 0.6,
    "legend.facecolor": BG_DARK,
    "legend.edgecolor": "#444466",
})

OUT = "/home/clawboy/research-epistemic-awareness/plots"

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 1 — AUROC by Layer
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6), dpi=200)

x = np.arange(len(layer_nums))
width = 0.35

bars_tf = ax.bar(x - width/2, tf_auroc, width, label="Training-Free",
                 color=BLUE, edgecolor="none", zorder=3)
bars_lr = ax.bar(x + width/2, lr_auroc, width, yerr=lr_std,
                 label="Logistic Regression", color=ORANGE,
                 edgecolor="none", capsize=4, error_kw={"ecolor": "#ffcc80", "capthick": 1.5},
                 zorder=3)

# Value labels on bars
for bar in bars_tf:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.008, f"{h:.3f}",
            ha="center", va="bottom", fontsize=10, color=BLUE, fontweight="bold")
for bar, val in zip(bars_lr, lr_auroc):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.025, f"{val:.3f}",
            ha="center", va="bottom", fontsize=10, color=ORANGE, fontweight="bold")

ax.axhline(0.5, color=ACCENT, ls="--", lw=1.5, label="Random baseline", zorder=2)
ax.set_xticks(x)
ax.set_xticklabels([f"Layer {n}" for n in layer_nums])
ax.set_ylim(0.5, 1.0)
ax.set_ylabel("AUROC")
ax.set_title("Correctness Signal Across Model Depth")
ax.legend(loc="lower right")
ax.grid(axis="y", color=GRID_CLR, alpha=0.08, zorder=1)
ax.set_axisbelow(True)

plt.tight_layout()
fig.savefig(f"{OUT}/01_auroc_by_layer.png")
plt.close(fig)
print("✓ Plot 1 saved")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 2 — Factual vs Reasoning
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6), dpi=200)

width = 0.25

bars_all  = ax.bar(x - width, lr_auroc, width, yerr=lr_std, label="All Tasks",
                   color=GRAY, edgecolor="none", capsize=3,
                   error_kw={"ecolor": "#b0bec5", "capthick": 1.2}, zorder=3)
bars_fact = ax.bar(x, fact_auroc, width, yerr=fact_std, label="Factual (MMLU)",
                   color=GREEN, edgecolor="none", capsize=3,
                   error_kw={"ecolor": "#a5d6a7", "capthick": 1.2}, zorder=3)
bars_reas = ax.bar(x + width, reas_auroc, width, yerr=reas_std, label="Reasoning (GSM8K)",
                   color=RED, edgecolor="none", capsize=3,
                   error_kw={"ecolor": "#ef9a9a", "capthick": 1.2}, zorder=3)

ax.axhline(0.5, color=ACCENT, ls="--", lw=1.5, label="Random baseline", zorder=2)
ax.set_xticks(x)
ax.set_xticklabels([f"Layer {n}" for n in layer_nums])
ax.set_ylim(0.3, 1.0)
ax.set_ylabel("AUROC")
ax.set_title("The Reasoning Gap: Factual vs Mathematical Tasks")
ax.legend(loc="lower right")
ax.grid(axis="y", color=GRID_CLR, alpha=0.08, zorder=1)
ax.set_axisbelow(True)

plt.tight_layout()
fig.savefig(f"{OUT}/02_factual_vs_reasoning.png")
plt.close(fig)
print("✓ Plot 2 saved")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 3 — TF vs LR scatter
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 8), dpi=200)

ax.scatter(tf_auroc, lr_auroc, s=160, c=BLUE, edgecolors=ORANGE,
           linewidths=2, zorder=4)

for i, ln in enumerate(layer_nums):
    ax.annotate(f"L{ln}", (tf_auroc[i], lr_auroc[i]),
                textcoords="offset points", xytext=(10, 8),
                fontsize=13, fontweight="bold", color=TXT_CLR)

# Diagonal
lo, hi = 0.70, 0.95
ax.plot([lo, hi], [lo, hi], "--", color=ACCENT, lw=1.5, alpha=0.7, label="y = x")

ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_xlabel("Training-Free AUROC")
ax.set_ylabel("Logistic Regression AUROC (mean)")
ax.set_title("Training-Free Probes Nearly Match Logistic Regression")
ax.set_aspect("equal")
ax.legend(loc="upper left")
ax.grid(color=GRID_CLR, alpha=0.08, zorder=1)
ax.set_axisbelow(True)

plt.tight_layout()
fig.savefig(f"{OUT}/03_tf_vs_lr_scatter.png")
plt.close(fig)
print("✓ Plot 3 saved")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 4 — Activation space schematic
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6), dpi=200)

# Generate two overlapping gaussians
x_range = np.linspace(-4, 6, 600)
# "Wrong answers" — wider, shifted right
wrong_mu, wrong_sig = 1.5, 1.6
wrong_pdf = norm.pdf(x_range, wrong_mu, wrong_sig)
# "Correct answers" — tighter, shifted left
correct_mu, correct_sig = -0.3, 1.0
correct_pdf = norm.pdf(x_range, correct_mu, correct_sig)

# Scale to look nice
scale = 1.0
ax.fill_between(x_range, wrong_pdf * scale, alpha=0.45, color=RED, zorder=2)
ax.plot(x_range, wrong_pdf * scale, color=RED, lw=2.5, zorder=3)

ax.fill_between(x_range, correct_pdf * scale, alpha=0.50, color=GREEN, zorder=2)
ax.plot(x_range, correct_pdf * scale, color=GREEN, lw=2.5, zorder=3)

# Decision boundary
boundary = 0.6
ax.axvline(boundary, color=ACCENT, ls="--", lw=2.5, zorder=4)
ax.text(boundary + 0.12, 0.34, "Decision\nBoundary", fontsize=13,
        color=ACCENT, fontweight="bold", va="top")

# Labels for distributions
ax.text(correct_mu - 0.1, norm.pdf(correct_mu, correct_mu, correct_sig) * scale + 0.03,
        "Correct\nanswers", fontsize=14, color=GREEN, fontweight="bold",
        ha="center", va="bottom")
ax.text(wrong_mu + 0.1, norm.pdf(wrong_mu, wrong_mu, wrong_sig) * scale + 0.02,
        "Wrong\nanswers", fontsize=14, color=RED, fontweight="bold",
        ha="center", va="bottom")

# Axis labels
ax.set_xlabel("Probe Score (1-D projection of activation space)", fontsize=15)
ax.set_ylabel("Density", fontsize=15)
ax.set_title("Why Probing Works: Correct and Wrong Answers Live in Different Regions",
             fontsize=16, pad=12)
ax.set_xlim(-4, 6)
ax.set_ylim(0, 0.48)

# Subtle grid
ax.grid(axis="both", color=GRID_CLR, alpha=0.06, zorder=1)
ax.set_axisbelow(True)

plt.tight_layout()
fig.savefig(f"{OUT}/04_activation_space_schematic.png")
plt.close(fig)
print("✓ Plot 4 saved")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 5 — Summary dashboard (2×2)
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=200)

# ── Subplot (0,0): AUROC by Layer ──
ax = axes[0, 0]
ax.bar(x - width/2, tf_auroc, width, label="Training-Free", color=BLUE, edgecolor="none", zorder=3)
ax.bar(x + width/2, lr_auroc, width, yerr=lr_std, label="Logistic Reg.", color=ORANGE,
       edgecolor="none", capsize=3, error_kw={"ecolor": "#ffcc80"}, zorder=3)
ax.axhline(0.5, color=ACCENT, ls="--", lw=1.2, zorder=2)
ax.set_xticks(x)
ax.set_xticklabels([f"L{n}" for n in layer_nums], fontsize=11)
ax.set_ylim(0.5, 1.0)
ax.set_ylabel("AUROC", fontsize=13)
ax.set_title("AUROC by Layer", fontsize=15, fontweight="bold")
ax.legend(fontsize=10, loc="lower right")
ax.grid(axis="y", color=GRID_CLR, alpha=0.08, zorder=1)
ax.set_axisbelow(True)

# ── Subplot (0,1): Factual vs Reasoning ──
ax = axes[0, 1]
ax.bar(x - width, lr_auroc, width, yerr=lr_std, label="All", color=GRAY,
       edgecolor="none", capsize=2, error_kw={"ecolor": "#b0bec5"}, zorder=3)
ax.bar(x, fact_auroc, width, yerr=fact_std, label="Factual", color=GREEN,
       edgecolor="none", capsize=2, error_kw={"ecolor": "#a5d6a7"}, zorder=3)
ax.bar(x + width, reas_auroc, width, yerr=reas_std, label="Reasoning", color=RED,
       edgecolor="none", capsize=2, error_kw={"ecolor": "#ef9a9a"}, zorder=3)
ax.axhline(0.5, color=ACCENT, ls="--", lw=1.2, zorder=2)
ax.set_xticks(x)
ax.set_xticklabels([f"L{n}" for n in layer_nums], fontsize=11)
ax.set_ylim(0.3, 1.0)
ax.set_ylabel("AUROC", fontsize=13)
ax.set_title("Factual vs Reasoning", fontsize=15, fontweight="bold")
ax.legend(fontsize=10, loc="lower right")
ax.grid(axis="y", color=GRID_CLR, alpha=0.08, zorder=1)
ax.set_axisbelow(True)

# ── Subplot (1,0): TF vs LR scatter ──
ax = axes[1, 0]
ax.scatter(tf_auroc, lr_auroc, s=120, c=BLUE, edgecolors=ORANGE, linewidths=1.5, zorder=4)
for i, ln in enumerate(layer_nums):
    ax.annotate(f"L{ln}", (tf_auroc[i], lr_auroc[i]),
                textcoords="offset points", xytext=(8, 6),
                fontsize=11, fontweight="bold", color=TXT_CLR)
ax.plot([0.72, 0.94], [0.72, 0.94], "--", color=ACCENT, lw=1.2, alpha=0.7)
ax.set_xlim(0.72, 0.94)
ax.set_ylim(0.72, 0.94)
ax.set_xlabel("Training-Free AUROC", fontsize=13)
ax.set_ylabel("LR AUROC", fontsize=13)
ax.set_title("TF vs LR Correlation", fontsize=15, fontweight="bold")
ax.set_aspect("equal")
ax.grid(color=GRID_CLR, alpha=0.08, zorder=1)
ax.set_axisbelow(True)

# ── Subplot (1,1): Activation schematic ──
ax = axes[1, 1]
ax.fill_between(x_range, wrong_pdf * scale, alpha=0.40, color=RED, zorder=2)
ax.plot(x_range, wrong_pdf * scale, color=RED, lw=2, zorder=3)
ax.fill_between(x_range, correct_pdf * scale, alpha=0.45, color=GREEN, zorder=2)
ax.plot(x_range, correct_pdf * scale, color=GREEN, lw=2, zorder=3)
ax.axvline(boundary, color=ACCENT, ls="--", lw=2, zorder=4)
ax.text(correct_mu, norm.pdf(correct_mu, correct_mu, correct_sig) * scale + 0.02,
        "Correct", fontsize=12, color=GREEN, fontweight="bold", ha="center")
ax.text(wrong_mu, norm.pdf(wrong_mu, wrong_mu, wrong_sig) * scale + 0.015,
        "Wrong", fontsize=12, color=RED, fontweight="bold", ha="center")
ax.set_xlim(-4, 6)
ax.set_ylim(0, 0.48)
ax.set_xlabel("Probe Score", fontsize=13)
ax.set_ylabel("Density", fontsize=13)
ax.set_title("Activation Space", fontsize=15, fontweight="bold")
ax.grid(color=GRID_CLR, alpha=0.06, zorder=1)
ax.set_axisbelow(True)

# ── Big title ──
fig.suptitle("Epistemic Probing Results — Qwen3.5-4B (656 questions)",
             fontsize=22, fontweight="bold", color=TXT_CLR, y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f"{OUT}/05_summary_dashboard.png")
plt.close(fig)
print("✓ Plot 5 saved")

print("\n✅ All 5 plots generated successfully!")
