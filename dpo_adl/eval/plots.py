from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _apply_style(pretty: bool = False):
    """Apply plotting style.

    pretty=True: larger fonts, higher DPI, cleaner grid, nicer color cycle,
    suitable for blogs (PNG+SVG output handled by callers).
    """
    if pretty:
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update({
            "figure.dpi": 180,
            "savefig.dpi": 180,
            "axes.facecolor": "#FFFFFF",
            "grid.color": "#D9D9D9",
            "grid.alpha": 0.6,
            "axes.edgecolor": "#333333",
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "font.family": "DejaVu Sans",
            "axes.prop_cycle": plt.cycler(color=[
                "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
            ]),
        })
    else:
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


def plot_entropy_vs_alpha(per_j: Dict[int, Dict[float, float]], out_dir: Path, pretty: bool = False):
    _apply_style(pretty=pretty)
    out_dir.mkdir(parents=True, exist_ok=True)
    for j, adict in per_j.items():
        if not adict:
            continue
        alphas = sorted(adict.keys())
        ent = [adict[a] for a in alphas]
        fig, ax = plt.subplots(figsize=(8.0, 4.5) if pretty else (7.2, 4.0))
        ax.plot(alphas, ent, marker="o", linewidth=2.2 if pretty else 1.5)
        # Highlight best alpha (lowest entropy)
        best_idx = int(np.argmin(ent))
        best_a, best_e = alphas[best_idx], ent[best_idx]
        ax.scatter([best_a], [best_e], color="#d62728" if pretty else "tab:red", zorder=3, label=f"best α={best_a:g}")
        if pretty:
            ax.annotate(f"α={best_a:g}, H={best_e:.3g}", (best_a, best_e), xytext=(8, 8), textcoords="offset points")
        ax.set_xlabel("alpha (scale)")
        ax.set_ylabel("intersection entropy")
        ax.set_title(f"Patchscope entropy vs alpha — j={j}")
        ax.legend(loc="upper right")
        fig.tight_layout()
        png = out_dir / f"patchscope_entropy_j{j}.png"
        svg = out_dir / f"patchscope_entropy_j{j}.svg"
        fig.savefig(png)
        if pretty:
            fig.savefig(svg)
        plt.close(fig)


def plot_margins_per_prompt(un: List[float], st: List[float], out_dir: Path, pretty: bool = False):
    _apply_style(pretty=pretty)
    x = np.arange(len(un))
    w = 0.4
    fig, ax = plt.subplots(figsize=(11.5, 4.8) if pretty else (10.5, 4.25))
    bars1 = ax.bar(x - w/2, un, width=w, label="unsteered margin")
    bars2 = ax.bar(x + w/2, st, width=w, label="steered margin")
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
    # Value annotations (only if not too many bars)
    if pretty and len(un) <= 30:
        for b in list(bars1) + list(bars2):
            h = b.get_height()
            ax.annotate(f"{h:.2f}", (b.get_x() + b.get_width()/2, h), xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, rotation=0)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "margins_per_prompt.png"
    svg = out_dir / "margins_per_prompt.svg"
    fig.savefig(png)
    if pretty:
        fig.savefig(svg)
    plt.close(fig)


def plot_margin_deltas(deltas: List[float], out_dir: Path, pretty: bool = False):
    _apply_style(pretty=pretty)
    fig, ax = plt.subplots(figsize=(8.0, 4.5) if pretty else (7.2, 4.0))
    if len(deltas) == 0:
        deltas = [0.0]
    ax.boxplot(deltas, vert=True, labels=["Δ = steered − unsteered"], showmeans=True, medianprops=dict(linewidth=2))
    mu = float(np.mean(deltas))
    med = float(np.median(deltas))
    ax.axhline(mu, color="tab:orange", linestyle="--", linewidth=1.0, label=f"mean={mu:.3f}")
    ax.axhline(med, color="tab:green", linestyle=":", linewidth=1.0, label=f"median={med:.3f}")
    ax.set_ylabel("DPO margin delta")
    ax.set_title("Distribution of DPO margin deltas")
    ax.legend(loc="best")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "margin_delta_box.png"
    svg = out_dir / "margin_delta_box.svg"
    fig.savefig(png)
    if pretty:
        fig.savefig(svg)
    plt.close(fig)


def plot_embed_similarity(un: List[float], st: List[float], out_dir: Path, pretty: bool = False):
    _apply_style(pretty=pretty)
    # Separate plots
    x = np.arange(len(un))
    mu_un = float(np.mean(un)) if len(un) else 0.0
    mu_st = float(np.mean(st)) if len(st) else 0.0

    fig1, ax1 = plt.subplots(figsize=(11.5, 4.8) if pretty else (10.5, 4.25))
    ax1.bar(x, un, color="#1f77b4")
    ax1.set_xlabel("prompt idx")
    ax1.set_ylabel("cosine sim")
    ax1.set_title(f"Embedding similarity (unsteered → ref) — mean={mu_un:.3f}")
    fig1.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "embed_sim_unsteered.png"
    svg = out_dir / "embed_sim_unsteered.svg"
    fig1.savefig(png)
    if pretty:
        fig1.savefig(svg)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(11.5, 4.8) if pretty else (10.5, 4.25))
    ax2.bar(x, st, color="#ff7f0e")
    ax2.set_xlabel("prompt idx")
    ax2.set_ylabel("cosine sim")
    ax2.set_title(f"Embedding similarity (steered → ref) — mean={mu_st:.3f}")
    fig2.tight_layout()
    png = out_dir / "embed_sim_steered.png"
    svg = out_dir / "embed_sim_steered.svg"
    fig2.savefig(png)
    if pretty:
        fig2.savefig(svg)
    plt.close(fig2)

    # Side-by-side comparison plot
    fig3, ax3 = plt.subplots(figsize=(11.5, 4.8) if pretty else (10.5, 4.25))
    w = 0.4
    x = np.arange(len(un))
    ax3.bar(x - w/2, un, width=w, label=f"unsteered→ref (μ={mu_un:.3f})", color="#1f77b4")
    ax3.bar(x + w/2, st, width=w, label=f"steered→ref (μ={mu_st:.3f})", color="#ff7f0e")
    ax3.set_xlabel("prompt idx")
    ax3.set_ylabel("cosine sim")
    ax3.set_title("Embedding similarity vs ref (side-by-side)")
    ax3.legend(loc="best")
    fig3.tight_layout()
    png = out_dir / "embed_sim_side_by_side.png"
    svg = out_dir / "embed_sim_side_by_side.svg"
    fig3.savefig(png)
    if pretty:
        fig3.savefig(svg)
    plt.close(fig3)
