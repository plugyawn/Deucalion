from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _apply_style():
    """Apply a readable, consistent Matplotlib style."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })


def plot_entropy_vs_alpha(per_j: Dict[int, Dict[float, float]], out_dir: Path):
    _apply_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    for j, adict in per_j.items():
        if not adict:
            continue
        alphas = sorted(adict.keys())
        ent = [adict[a] for a in alphas]
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        ax.plot(alphas, ent, marker="o")
        # Highlight best alpha (lowest entropy)
        best_idx = int(np.argmin(ent))
        best_a, best_e = alphas[best_idx], ent[best_idx]
        ax.scatter([best_a], [best_e], color="tab:red", zorder=3, label=f"best α={best_a:g}")
        ax.set_xlabel("alpha")
        ax.set_ylabel("entropy (intersection)")
        ax.set_title(f"Patchscope entropy vs alpha (j={j})")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(out_dir / f"patchscope_entropy_j{j}.png")
        plt.close(fig)


def plot_margins_per_prompt(un: List[float], st: List[float], out_dir: Path):
    _apply_style()
    x = np.arange(len(un))
    w = 0.4
    fig, ax = plt.subplots(figsize=(10.5, 4.25))
    ax.bar(x - w/2, un, width=w, label="unsteered margin")
    ax.bar(x + w/2, st, width=w, label="steered margin")
    # Summary statistics
    mu_un = float(np.mean(un)) if len(un) else 0.0
    mu_st = float(np.mean(st)) if len(st) else 0.0
    mu_delta = mu_st - mu_un
    ax.axhline(0.0, color="k", linewidth=0.8)
    ax.set_xlabel("prompt idx")
    ax.set_ylabel("DPO margin")
    ax.set_title("DPO implicit-reward margin per prompt")
    ax.legend(loc="upper right")
    # Annotate means in the top-left corner
    txt = f"mean(un)={mu_un:.3f}\nmean(st)={mu_st:.3f}\nΔmean={mu_delta:+.3f}"
    ax.text(0.01, 0.99, txt, transform=ax.transAxes, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, linewidth=0.5))
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "margins_per_prompt.png")
    plt.close(fig)


def plot_margin_deltas(deltas: List[float], out_dir: Path):
    _apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    if len(deltas) == 0:
        deltas = [0.0]
    ax.boxplot(deltas, vert=True, labels=["Δ = steered − unsteered"], showmeans=True)
    mu = float(np.mean(deltas))
    med = float(np.median(deltas))
    ax.axhline(mu, color="tab:orange", linestyle="--", linewidth=1.0, label=f"mean={mu:.3f}")
    ax.axhline(med, color="tab:green", linestyle=":", linewidth=1.0, label=f"median={med:.3f}")
    ax.set_ylabel("DPO margin delta")
    ax.set_title("Distribution of DPO margin deltas")
    ax.legend(loc="best")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "margin_delta_box.png")
    plt.close(fig)


def plot_embed_similarity(un: List[float], st: List[float], out_dir: Path):
    _apply_style()
    # Separate plots
    x = np.arange(len(un))
    mu_un = float(np.mean(un)) if len(un) else 0.0
    mu_st = float(np.mean(st)) if len(st) else 0.0

    fig1, ax1 = plt.subplots(figsize=(10.5, 4.25))
    ax1.bar(x, un)
    ax1.set_xlabel("prompt idx")
    ax1.set_ylabel("cosine sim")
    ax1.set_title(f"Embedding similarity (unsteered → ref) — mean={mu_un:.3f}")
    fig1.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig1.savefig(out_dir / "embed_sim_unsteered.png")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(10.5, 4.25))
    ax2.bar(x, st, color="tab:orange")
    ax2.set_xlabel("prompt idx")
    ax2.set_ylabel("cosine sim")
    ax2.set_title(f"Embedding similarity (steered → ref) — mean={mu_st:.3f}")
    fig2.tight_layout()
    fig2.savefig(out_dir / "embed_sim_steered.png")
    plt.close(fig2)

    # Side-by-side comparison plot
    fig3, ax3 = plt.subplots(figsize=(10.5, 4.25))
    w = 0.4
    x = np.arange(len(un))
    ax3.bar(x - w/2, un, width=w, label=f"unsteered→ref (μ={mu_un:.3f})")
    ax3.bar(x + w/2, st, width=w, label=f"steered→ref (μ={mu_st:.3f})")
    ax3.set_xlabel("prompt idx")
    ax3.set_ylabel("cosine sim")
    ax3.set_title("Embedding similarity vs ref (side-by-side)")
    ax3.legend(loc="best")
    fig3.tight_layout()
    fig3.savefig(out_dir / "embed_sim_side_by_side.png")
    plt.close(fig3)
