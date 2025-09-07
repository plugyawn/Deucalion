from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def plot_entropy_vs_alpha(per_j: Dict[int, Dict[float, float]], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for j, adict in per_j.items():
        alphas = sorted(adict.keys())
        ent = [adict[a] for a in alphas]
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(alphas, ent, marker="o")
        ax.set_xlabel("alpha")
        ax.set_ylabel("entropy (intersection)")
        ax.set_title(f"Patchscope entropy vs alpha (j={j})")
        fig.tight_layout()
        fig.savefig(out_dir / f"patchscope_entropy_j{j}.png")
        plt.close(fig)


def plot_margins_per_prompt(un: List[float], st: List[float], out_dir: Path):
    x = np.arange(len(un))
    w = 0.4
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - w/2, un, width=w, label="unsteered margin")
    ax.bar(x + w/2, st, width=w, label="steered margin")
    ax.set_xlabel("prompt idx")
    ax.set_ylabel("DPO margin")
    ax.set_title("DPO implicit-reward margin per prompt")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "margins_per_prompt.png")
    plt.close(fig)


def plot_margin_deltas(deltas: List[float], out_dir: Path):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.boxplot(deltas, vert=True, labels=["delta=steered-unsteered"]) 
    ax.set_ylabel("DPO margin delta")
    ax.set_title("Distribution of DPO margin deltas")
    fig.tight_layout()
    fig.savefig(out_dir / "margin_delta_box.png")
    plt.close(fig)


def plot_embed_similarity(un: List[float], st: List[float], out_dir: Path):
    # Separate plots
    x = np.arange(len(un))
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.bar(x, un)
    ax1.set_xlabel("prompt idx")
    ax1.set_ylabel("cosine sim")
    ax1.set_title("Embedding similarity (unsteered → ref)")
    fig1.tight_layout()
    fig1.savefig(out_dir / "embed_sim_unsteered.png")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.bar(x, st, color="tab:orange")
    ax2.set_xlabel("prompt idx")
    ax2.set_ylabel("cosine sim")
    ax2.set_title("Embedding similarity (steered → ref)")
    fig2.tight_layout()
    fig2.savefig(out_dir / "embed_sim_steered.png")
    plt.close(fig2)

    # Side-by-side comparison plot
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    w = 0.4
    x = np.arange(len(un))
    ax3.bar(x - w/2, un, width=w, label="unsteered→ref")
    ax3.bar(x + w/2, st, width=w, label="steered→ref")
    ax3.set_xlabel("prompt idx")
    ax3.set_ylabel("cosine sim")
    ax3.set_title("Embedding similarity vs ref (side-by-side)")
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(out_dir / "embed_sim_side_by_side.png")
    plt.close(fig3)

