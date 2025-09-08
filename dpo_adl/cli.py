from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
import tyro

from dpo_adl.backends.hf_hooks import load_model_and_tokenizer, num_layers
from dpo_adl.data.fineweb import iter_probe_texts
from dpo_adl.diff.delta_builder import build_delta
from dpo_adl.dpo.implicit_reward import dpo_margin
from dpo_adl.eval.steering import generate_text, steered_generation
from dpo_adl.eval.embeds import Embedder
from dpo_adl.eval.plots import plot_entropy_vs_alpha, plot_margins_per_prompt, plot_margin_deltas, plot_embed_similarity
from dpo_adl.patchscope.token_identity import DEFAULT_PROMPTS, patchscope_logits, top_tokens_from_probs
from dpo_adl.utils.logging import get_logger
from dpo_adl.utils.exp import create_exp_dir, snapshot_code
from dpo_adl.eval.report import bundle_plots_to_pdf
from dpo_adl.train.datasets import (
    load_synthetic_british,
    SyntheticBritishConfig,
    load_hf_preference_dataset,
    chosen_texts_from_spec,
)
from dpo_adl.train.dpo_trainer import DPOTrainConfig, train_dpo_on_dataset
from dpo_adl.diff.orth import project_out


log = get_logger()


@dataclass
class CmdBuildDelta:
    ref_model: str
    dpo_model: str
    n_probe: int = 10000
    k: int = 5
    layer_idx: Optional[int] = None
    batch_size: int = 64
    out: str = "artifacts/delta.pt"
    dataset: str = "HuggingFaceFW/fineweb-edu"
    split: str = "train"

    def __call__(self):
        Path(self.out).parent.mkdir(parents=True, exist_ok=True)
        # IMPORTANT: texts_iter used twice; collect into list for simplicity
        texts = list(iter_probe_texts(self.dataset, self.split, self.n_probe, seed=0))
        delta = build_delta(self.ref_model, self.dpo_model, texts, k=self.k, layer_idx=self.layer_idx, batch_size=self.batch_size)
        torch.save({"delta": delta, "k": self.k, "layer_idx": self.layer_idx}, self.out)
        log.info(f"Saved Δ to {self.out} shape={tuple(delta.shape)}")


@dataclass
class CmdPatchscope:
    reader_model: str
    delta: str
    alpha: float = 1.0
    alpha_sweep: Optional[str] = None  # comma-separated, e.g., "0.25,0.5,1.0,1.5"
    norm_match: bool = True
    norm_sample: int = 256
    j: Optional[int] = None  # which position delta_j to use; if None, scan j=0..k-1 and report best by entropy
    layer_idx: Optional[int] = None
    sentinel: str = "?"
    prompts_file: Optional[str] = None
    topk: int = 20

    def __call__(self):
        model, tok = load_model_and_tokenizer(self.reader_model)
        blob = torch.load(self.delta, map_location="cpu")
        delta = blob["delta"]  # [k, d]
        k = int(blob.get("k", delta.shape[0]))
        prompts = DEFAULT_PROMPTS
        if self.prompts_file:
            prompts = Path(self.prompts_file).read_text().splitlines()
        # Expected norm estimation for norm matching
        expected_norm = None
        if self.norm_match:
            # Use a small slice of probe texts from generic prompts as a heuristic
            sample_texts = prompts * ((self.norm_sample // max(1, len(prompts))) + 1)
            sample_texts = sample_texts[: self.norm_sample]
            L = num_layers(model)
            layer_idx = self.layer_idx if self.layer_idx is not None else L // 2
            expected_norm = __import__("dpo_adl.backends.hf_hooks", fromlist=["estimate_expected_norm"]).estimate_expected_norm(
                model, tok, sample_texts, layer_idx, k_first_tokens=8, skip_first=3, batch_size=32
            )
            log.info(f"Estimated expected norm at layer {layer_idx}: {expected_norm:.4f}")
        positions = [self.j] if self.j is not None else list(range(k))
        sweep = None
        if self.alpha_sweep:
            sweep = [float(x) for x in self.alpha_sweep.split(",") if x.strip()]
        results = []
        for j in positions:
            best = None
            alphas = sweep if sweep else [self.alpha]
            for a in alphas:
                probs_sets = []
                for p in prompts:
                    probs = patchscope_logits(
                        model,
                        tok,
                        self.layer_idx,
                        delta[j],
                        a,
                        p,
                        self.sentinel,
                        norm_match=self.norm_match,
                        expected_norm=expected_norm,
                    )
                    probs_sets.append(probs)
                top_idxs_sets = [torch.topk(p, k=100).indices for p in probs_sets]
                inter = set(top_idxs_sets[0].tolist())
                for s in top_idxs_sets[1:]:
                    inter &= set(s.tolist())
                if len(inter) == 0:
                    entropy = float("inf")
                    avg_p = torch.stack(probs_sets, dim=0).mean(dim=0)
                    top_tokens = top_tokens_from_probs(tok, avg_p, topk=self.topk)
                else:
                    avg_p = torch.stack(probs_sets, dim=0).mean(dim=0)
                    mask = torch.zeros_like(avg_p)
                    idx = torch.tensor(sorted(list(inter)), dtype=torch.long)
                    mask[idx] = 1.0
                    p_sel = (avg_p * mask)
                    p_sel = p_sel / p_sel.sum()
                    entropy = float((-p_sel[p_sel>0].log() * p_sel[p_sel>0]).sum().item())
                    top_tokens = top_tokens_from_probs(tok, avg_p, topk=self.topk)
                cand = {"j": j, "alpha": a, "entropy": entropy, "top": top_tokens}
                if best is None or cand["entropy"] < best["entropy"]:
                    best = cand
            results.append(best)
        # Pick best by lowest entropy if scanning
        if self.j is None:
            results.sort(key=lambda r: r["entropy"])  # lowest entropy first
        for r in results:
            print(json.dumps({"j": r["j"], "entropy": r["entropy"], "top": r["top"]}, ensure_ascii=False))


@dataclass
class CmdEvalSteer:
    ref_model: str
    dpo_model: str
    delta: str
    j: Optional[int] = None
    layer_idx: Optional[int] = None
    alpha: float = 1.0
    prompts: str = "prompts/generic20.txt"
    max_new_tokens: int = 128
    temperature: float = 0.0
    embed_model: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2"
    plot_out: Optional[str] = "artifacts/steer_margins.png"

    def __call__(self):
        dpo_m, dpo_tok = load_model_and_tokenizer(self.dpo_model)
        ref_m, ref_tok = load_model_and_tokenizer(self.ref_model)
        blob = torch.load(self.delta, map_location="cpu")
        delta = blob["delta"]  # [k, d]
        k = int(blob.get("k", delta.shape[0]))
        j = self.j if self.j is not None else max(0, min(4, k-1))
        prompts = [p for p in Path(self.prompts).read_text().splitlines() if p.strip()]
        # Assert delta dim matches hidden size if available
        if hasattr(dpo_m.config, "hidden_size"):
            assert delta.shape[1] == dpo_m.config.hidden_size, f"Δ dim {delta.shape[1]} != model hidden_size {dpo_m.config.hidden_size}"
        margins = []
        un_texts, st_texts = [], []
        for p in prompts:
            # Unsteered
            y0 = generate_text(dpo_m, dpo_tok, p, max_new_tokens=self.max_new_tokens, temperature=self.temperature)
            m0 = dpo_margin(dpo_m, ref_m, dpo_tok, ref_tok, p, y0[len(p):])
            # Steered
            y1 = steered_generation(
                dpo_m, dpo_tok, p, delta_vec=delta[j], layer_idx=self.layer_idx, alpha=self.alpha,
                max_new_tokens=self.max_new_tokens, temperature=self.temperature,
            )
            m1 = dpo_margin(dpo_m, ref_m, dpo_tok, ref_tok, p, y1[len(p):])
            margins.append({"prompt": p, "unsteered": m0, "steered": m1, "delta": m1 - m0})
            print(json.dumps({"prompt": p, "margin_unsteered": m0, "margin_steered": m1, "delta": m1-m0}))
            un_texts.append(y0)
            st_texts.append(y1)
        deltas = [m["delta"] for m in margins]
        log.info(f"Avg DPO margin delta over {len(deltas)} prompts: {sum(deltas)/max(1,len(deltas)):.4f}")
        # Embedding similarity: compare steered vs unsteered to ref_model outputs
        if self.embed_model:
            # Generate reference outputs from ref_model on same prompts
            ref_texts = [generate_text(ref_m, ref_tok, p, max_new_tokens=self.max_new_tokens, temperature=self.temperature) for p in prompts]
            emb = Embedder(self.embed_model)
            E_ref = emb.encode(ref_texts)
            E_un = emb.encode(un_texts)
            E_st = emb.encode(st_texts)
            import matplotlib.pyplot as plt
            import numpy as np
            sim_un = emb.cosine_sim_matrix(E_un, E_ref).diag().tolist()
            sim_st = emb.cosine_sim_matrix(E_st, E_ref).diag().tolist()
            Path(self.plot_out).parent.mkdir(parents=True, exist_ok=True)
            # Save separate and side-by-side plots
            from dpo_adl.eval.plots import plot_embed_similarity
            plot_embed_similarity(sim_un, sim_st, Path(self.plot_out).parent)
            # Also save a simple combined filename for backward compatibility
            x = np.arange(len(prompts))
            w = 0.35
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(x - w/2, sim_un, width=w, label="unsteered→ref sim")
            ax.bar(x + w/2, sim_st, width=w, label="steered→ref sim")
            ax.set_ylabel("cosine sim")
            ax.set_xlabel("prompt idx")
            ax.set_title("Embedding similarity vs ref_model outputs")
            ax.legend()
            plt.tight_layout()
            plt.savefig(self.plot_out)
            log.info(f"Saved comparison plot: {self.plot_out}")


@dataclass
class CmdRunExp:
    name: str = "exp02"
    ref_model: str = "Qwen/Qwen2.5-0.5B"
    dpo_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    n_probe: int = 1200
    k: int = 5
    batch_size: int = 16
    layer_idx: Optional[int] = None
    prompts: str = "prompts/generic20.txt"
    alpha_sweep: str = "0.5,1.0,1.5,2.0"
    norm_match: bool = True
    sentinel: str = "?"
    max_new_tokens: int = 64
    temperature: float = 0.0
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    make_pdf: bool = True
    # Embedding baseline target
    embed_to: str = "ref"  # "ref" or "dpo-chosen"
    embed_ds_name: Optional[str] = None
    embed_ds_split: str = "train_prefs"
    embed_ds_n: int = 1000
    # Orthogonalization
    orthogonalize: bool = False
    base_model: Optional[str] = None

    def __call__(self):
        exp_dir = create_exp_dir(self.name)
        snapshot_code(exp_dir)
        cfg = {
            "ref_model": self.ref_model,
            "dpo_model": self.dpo_model,
            "n_probe": self.n_probe,
            "k": self.k,
            "batch_size": self.batch_size,
            "layer_idx": self.layer_idx,
            "prompts": self.prompts,
            "alpha_sweep": self.alpha_sweep,
            "norm_match": self.norm_match,
            "sentinel": self.sentinel,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "embed_model": self.embed_model,
        }
        (exp_dir / "config.json").write_text(json.dumps(cfg, indent=2))

        # 1) Build Δ
        model, tok = load_model_and_tokenizer(self.dpo_model)
        from dpo_adl.data.fineweb import iter_probe_texts
        texts = list(iter_probe_texts("HuggingFaceFW/fineweb-edu", "train", self.n_probe, seed=0))
        from dpo_adl.diff.delta_builder import build_delta
        delta = build_delta(self.ref_model, self.dpo_model, texts, k=self.k, layer_idx=self.layer_idx, batch_size=self.batch_size)
        torch.save({"delta": delta, "k": self.k, "layer_idx": self.layer_idx}, exp_dir / "artifacts" / "delta.pt")
        # Optional Δ_SFT for orthogonalization
        delta_sft = None
        if self.orthogonalize:
            assert self.base_model is not None, "Provide --base_model when --orthogonalize is set."
            delta_sft = build_delta(self.base_model, self.ref_model, texts, k=self.k, layer_idx=self.layer_idx, batch_size=self.batch_size)
            torch.save({"delta": delta_sft, "k": self.k, "layer_idx": self.layer_idx}, exp_dir / "artifacts" / "delta_sft.pt")

        # 2) Patchscope alpha sweep & entropy plots per j
        alphas = [float(x) for x in self.alpha_sweep.split(",") if x.strip()]
        prompts = [p for p in Path(self.prompts).read_text().splitlines() if p.strip()]
        from dpo_adl.patchscope.token_identity import patchscope_logits, DEFAULT_PROMPTS
        token_id_prompts = DEFAULT_PROMPTS  # use token-identity prompts for Patchscope
        # Estimate expected norm (optional)
        expected_norm = None
        if self.norm_match:
            L = num_layers(model)
            layer_idx = self.layer_idx if self.layer_idx is not None else L // 2
            from dpo_adl.backends.hf_hooks import estimate_expected_norm
            expected_norm = estimate_expected_norm(model, tok, token_id_prompts * 32, layer_idx, k_first_tokens=8, skip_first=3, batch_size=32)

        entropies_per_j = {}
        best_by_j = {}
        entropies_per_j_orth = {} if self.orthogonalize else None
        best_by_j_orth = {} if self.orthogonalize else None
        for j in range(delta.shape[0]):
            ent_per_a = {}
            best = None
            for a in alphas:
                probs_sets = []
                for p in token_id_prompts:
                    probs = patchscope_logits(
                        model, tok, self.layer_idx, delta[j], a, p, self.sentinel, norm_match=self.norm_match, expected_norm=expected_norm
                    )
                    probs_sets.append(probs)
                top_sets = [torch.topk(p, k=100).indices for p in probs_sets]
                inter = set(top_sets[0].tolist())
                for s in top_sets[1:]:
                    inter &= set(s.tolist())
                avg_p = torch.stack(probs_sets, dim=0).mean(dim=0)
                if len(inter) == 0:
                    entropy = float("inf")
                else:
                    mask = torch.zeros_like(avg_p)
                    idx = torch.tensor(sorted(list(inter)), dtype=torch.long)
                    mask[idx] = 1.0
                    p_sel = (avg_p * mask)
                    p_sel = p_sel / p_sel.sum()
                    entropy = float((-p_sel[p_sel>0].log() * p_sel[p_sel>0]).sum().item())
                ent_per_a[a] = entropy
                top_tokens = top_tokens_from_probs(tok, avg_p, topk=20)
                cand = {"j": j, "alpha": a, "entropy": entropy, "top": top_tokens}
                if best is None or cand["entropy"] < best["entropy"]:
                    best = cand
            entropies_per_j[j] = ent_per_a
            best_by_j[j] = best
            if self.orthogonalize:
                ent_per_a_o = {}
                best_o = None
                d_orth = project_out(delta[j], delta_sft[j])
                for a in alphas:
                    probs_sets = []
                    for p in token_id_prompts:
                        probs = patchscope_logits(
                            model, tok, self.layer_idx, d_orth, a, p, self.sentinel, norm_match=self.norm_match, expected_norm=expected_norm
                        )
                        probs_sets.append(probs)
                    top_sets = [torch.topk(p, k=100).indices for p in probs_sets]
                    inter = set(top_sets[0].tolist())
                    for s in top_sets[1:]:
                        inter &= set(s.tolist())
                    avg_p = torch.stack(probs_sets, dim=0).mean(dim=0)
                    if len(inter) == 0:
                        entropy = float("inf")
                    else:
                        mask = torch.zeros_like(avg_p)
                        idx = torch.tensor(sorted(list(inter)), dtype=torch.long)
                        mask[idx] = 1.0
                        p_sel = (avg_p * mask)
                        p_sel = p_sel / p_sel.sum()
                        entropy = float((-p_sel[p_sel>0].log() * p_sel[p_sel>0]).sum().item())
                    ent_per_a_o[a] = entropy
                    top_tokens_o = top_tokens_from_probs(tok, avg_p, topk=20)
                    cand_o = {"j": j, "alpha": a, "entropy": entropy, "top": top_tokens_o}
                    if best_o is None or cand_o["entropy"] < best_o["entropy"]:
                        best_o = cand_o
                entropies_per_j_orth[j] = ent_per_a_o
                best_by_j_orth[j] = best_o
        plot_entropy_vs_alpha(entropies_per_j, exp_dir / "plots")
        if self.orthogonalize:
            # Save orth entropy plots and rename with suffix
            plot_entropy_vs_alpha(entropies_per_j_orth, exp_dir / "plots")
            for j in entropies_per_j_orth.keys():
                p = exp_dir / "plots" / f"patchscope_entropy_j{j}.png"
                if p.exists():
                    p.rename(exp_dir / "plots" / f"patchscope_entropy_j{j}_orth.png")
        # Choose best j overall
        best_overall = min(best_by_j.values(), key=lambda r: r["entropy"])
        (exp_dir / "artifacts" / "patchscope_best.json").write_text(json.dumps({"best": best_overall, "by_j": best_by_j}, ensure_ascii=False, indent=2))
        if self.orthogonalize:
            best_overall_orth = min(best_by_j_orth.values(), key=lambda r: r["entropy"])
            (exp_dir / "artifacts" / "patchscope_best_orth.json").write_text(json.dumps({"best": best_overall_orth, "by_j": best_by_j_orth}, ensure_ascii=False, indent=2))

        # 3) Steering + margins + embeddings and plots
        ref_m, ref_tok = load_model_and_tokenizer(self.ref_model)
        from dpo_adl.eval.steering import batched_eval_margins
        res = batched_eval_margins(
            prompts, model, ref_m, tok, ref_tok, delta_vec=delta[best_overall["j"]], layer_idx=self.layer_idx,
            alpha=best_overall["alpha"], max_new_tokens=self.max_new_tokens, temperature=self.temperature,
        )
        # Save results JSON
        out_rows = []
        un, st, deltas = [], [], []
        un_texts, st_texts = [], []
        for (p, y0, y1, m0, m1) in res:
            out_rows.append({"prompt": p, "unsteered": m0, "steered": m1, "delta": m1 - m0})
            un.append(m0); st.append(m1); deltas.append(m1 - m0)
            un_texts.append(y0); st_texts.append(y1)
        (exp_dir / "artifacts" / "steer_margins.json").write_text(json.dumps(out_rows, indent=2))
        # Embedding similarity vs target
        emb = Embedder(self.embed_model)
        if self.embed_to == "ref":
            ref_texts = [generate_text(ref_m, ref_tok, p, max_new_tokens=self.max_new_tokens, temperature=self.temperature) for p in prompts]
            E_ref = emb.encode(ref_texts)
            E_un = emb.encode(un_texts)
            E_st = emb.encode(st_texts)
            sim_un = (E_un @ E_ref.T).diagonal().tolist()
            sim_st = (E_st @ E_ref.T).diagonal().tolist()
        elif self.embed_to == "dpo-chosen":
            assert self.embed_ds_name is not None, "Provide --embed_ds_name for dpo-chosen baseline"
            chosen = chosen_texts_from_spec(self.embed_ds_name, self.embed_ds_split, self.embed_ds_n, seed=0)
            E_chosen = emb.encode(chosen)
            centroid = E_chosen.mean(dim=0, keepdim=True)
            centroid = centroid / centroid.norm(p=2)
            E_un = emb.encode(un_texts)
            E_st = emb.encode(st_texts)
            sim_un = (E_un @ centroid.T).squeeze(1).tolist()
            sim_st = (E_st @ centroid.T).squeeze(1).tolist()
        else:
            raise ValueError("embed_to must be 'ref' or 'dpo-chosen'")
        (exp_dir / "artifacts" / "embed_similarity.json").write_text(json.dumps({"un": sim_un, "st": sim_st}, indent=2))
        # Plots
        plot_margins_per_prompt(un, st, exp_dir / "plots")
        plot_margin_deltas(deltas, exp_dir / "plots")
        plot_embed_similarity(sim_un, sim_st, exp_dir / "plots")

        # Orthogonalized steering + embeddings
        if self.orthogonalize:
            d_orth_best = project_out(delta[best_overall_orth["j"]], delta_sft[best_overall_orth["j"]])
            res_o = batched_eval_margins(
                prompts, model, ref_m, tok, ref_tok, delta_vec=d_orth_best, layer_idx=self.layer_idx,
                alpha=best_overall_orth["alpha"], max_new_tokens=self.max_new_tokens, temperature=self.temperature,
            )
            out_rows_o = []
            un_o, st_o, deltas_o = [], [], []
            un_texts_o, st_texts_o = [], []
            for (p, y0, y1, m0, m1) in res_o:
                out_rows_o.append({"prompt": p, "unsteered": m0, "steered": m1, "delta": m1 - m0})
                un_o.append(m0); st_o.append(m1); deltas_o.append(m1 - m0)
                un_texts_o.append(y0); st_texts_o.append(y1)
            (exp_dir / "artifacts" / "steer_margins_orth.json").write_text(json.dumps(out_rows_o, indent=2))
            # Save plots with _orth suffix
            plot_margins_per_prompt(un_o, st_o, exp_dir / "plots")
            p1 = exp_dir / "plots" / "margins_per_prompt.png"
            if p1.exists(): p1.rename(exp_dir / "plots" / "margins_per_prompt_orth.png")
            plot_margin_deltas(deltas_o, exp_dir / "plots")
            p2 = exp_dir / "plots" / "margin_delta_box.png"
            if p2.exists(): p2.rename(exp_dir / "plots" / "margin_delta_box_orth.png")
            # Embedding sim for orth
            if self.embed_to == "ref":
                ref_texts = [generate_text(ref_m, ref_tok, p, max_new_tokens=self.max_new_tokens, temperature=self.temperature) for p in prompts]
                E_ref = emb.encode(ref_texts)
                E_un_o = emb.encode(un_texts_o)
                E_st_o = emb.encode(st_texts_o)
                sim_un_o = (E_un_o @ E_ref.T).diagonal().tolist()
                sim_st_o = (E_st_o @ E_ref.T).diagonal().tolist()
            else:
                chosen = chosen_texts_from_spec(self.embed_ds_name, self.embed_ds_split, self.embed_ds_n, seed=0)
                E_chosen = emb.encode(chosen)
                centroid = E_chosen.mean(dim=0, keepdim=True)
                centroid = centroid / centroid.norm(p=2)
                E_un_o = emb.encode(un_texts_o)
                E_st_o = emb.encode(st_texts_o)
                sim_un_o = (E_un_o @ centroid.T).squeeze(1).tolist()
                sim_st_o = (E_st_o @ centroid.T).squeeze(1).tolist()
            plot_embed_similarity(sim_un_o, sim_st_o, exp_dir / "plots")
            p3 = exp_dir / "plots" / "embed_sim_unsteered.png"
            p4 = exp_dir / "plots" / "embed_sim_steered.png"
            p5 = exp_dir / "plots" / "embed_sim_side_by_side.png"
            if p3.exists(): p3.rename(exp_dir / "plots" / "embed_sim_unsteered_orth.png")
            if p4.exists(): p4.rename(exp_dir / "plots" / "embed_sim_steered_orth.png")
            if p5.exists(): p5.rename(exp_dir / "plots" / "embed_sim_side_by_side_orth.png")

        # 4) Bundle report PDF
        if self.make_pdf:
            bundle_plots_to_pdf(
                exp_dir / "plots",
                exp_dir / "report.pdf",
                order=[
                    "patchscope_entropy_j",
                    "margins_per_prompt",
                    "margin_delta_box",
                    "embed_sim_unsteered",
                    "embed_sim_steered",
                    "embed_sim_side_by_side",
                ],
            )

        print(json.dumps({"exp_dir": str(exp_dir), "best": best_overall, "avg_margin_delta": sum(deltas)/max(1,len(deltas))}))


@dataclass
class CmdTrainDPO:
    ref_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    out_dir: str = "assets/trained/dpo_british"
    n_pairs: int = 200
    beta: float = 0.1
    learning_rate: float = 5e-6
    max_steps: int = 60
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_length: int = 256
    max_prompt_length: int = 128
    seed: int = 0
    use_lora: bool = True

    def __call__(self):
        ds = load_synthetic_british(SyntheticBritishConfig(n_pairs=self.n_pairs, seed=self.seed))
        cfg = DPOTrainConfig(
            ref_model=self.ref_model,
            out_dir=self.out_dir,
            beta=self.beta,
            learning_rate=self.learning_rate,
            max_steps=self.max_steps,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            max_length=self.max_length,
            max_prompt_length=self.max_prompt_length,
            seed=self.seed,
            use_lora=self.use_lora,
        )
        train_dpo_on_dataset(cfg, ds)
        print(json.dumps({"trained_model": self.out_dir, "pairs": self.n_pairs, "steps": self.max_steps}))


@dataclass
class CmdTrainDPOHF:
    ref_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dataset: str = "HuggingFaceH4/ultrafeedback_binarized"
    split: str = "train_prefs"
    n_pairs: int = 1000
    out_dir: str = "assets/trained/dpo_hf"
    beta: float = 0.1
    learning_rate: float = 5e-6
    max_steps: int = 60
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 2
    max_length: int = 256
    max_prompt_length: int = 128
    seed: int = 0
    use_lora: bool = True

    def __call__(self):
        ds = load_hf_preference_dataset(self.dataset, self.split, self.n_pairs, seed=self.seed)
        cfg = DPOTrainConfig(
            ref_model=self.ref_model,
            out_dir=self.out_dir,
            beta=self.beta,
            learning_rate=self.learning_rate,
            max_steps=self.max_steps,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            max_length=self.max_length,
            max_prompt_length=self.max_prompt_length,
            seed=self.seed,
            use_lora=self.use_lora,
        )
        train_dpo_on_dataset(cfg, ds)
        print(json.dumps({"trained_model": self.out_dir, "dataset": self.dataset, "split": self.split, "pairs": self.n_pairs, "steps": self.max_steps}))


def main():
    import sys
    if len(sys.argv) <= 1:
        print("Usage: dpo-adl [build-delta|patchscope|eval-steer] ...")
        return
    cmd = sys.argv[1]
    args = sys.argv[2:]
    if cmd == "build-delta":
        tyro.extras.set_accent_color("green")
        tyro.cli(CmdBuildDelta, args=args)()
    elif cmd == "patchscope":
        tyro.extras.set_accent_color("blue")
        tyro.cli(CmdPatchscope, args=args)()
    elif cmd == "eval-steer":
        tyro.extras.set_accent_color("magenta")
        tyro.cli(CmdEvalSteer, args=args)()
    elif cmd == "run-exp":
        tyro.extras.set_accent_color("cyan")
        tyro.cli(CmdRunExp, args=args)()
    elif cmd == "train-dpo":
        tyro.extras.set_accent_color("yellow")
        tyro.cli(CmdTrainDPO, args=args)()
    elif cmd == "train-dpo-hf":
        tyro.extras.set_accent_color("yellow")
        tyro.cli(CmdTrainDPOHF, args=args)()
    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
