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
from dpo_adl.train.dpo_trainer import (
    DPOTrainConfig, train_dpo_on_dataset,
    GRPOTrainConfig, train_grpo_on_dataset,
)
from dpo_adl.analysis.subnetwork import profile_param_deltas, save_profile
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
    ban_punct: bool = False
    words_only: bool = False

    def __call__(self):
        model, tok = load_model_and_tokenizer(self.reader_model)
        blob = torch.load(self.delta, map_location="cpu")
        delta = blob["delta"]  # [k, d]
        k = int(blob.get("k", delta.shape[0]))
        # Default to stored layer index if not provided
        if self.layer_idx is None and (blob.get("layer_idx") is not None):
            try:
                self.layer_idx = int(blob["layer_idx"])  # type: ignore
            except Exception:
                pass
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
        # Helper: token filters
        import unicodedata as _ud
        def _is_box_or_repeat(t: str) -> bool:
            if len(t) >= 3 and all(ch in "-_=~." for ch in t):
                return True
            for ch in t:
                code = ord(ch)
                if (0x2500 <= code <= 0x257F) or (0x2580 <= code <= 0x259F):
                    return True
            return False
        def _is_wordlike(s: str) -> bool:
            if s is None:
                return False
            t = s.replace("▁", "").replace("Ġ", "").strip()
            if t == "":
                return False
            # At least one unicode letter
            has_letter = any(_ud.category(ch).startswith('L') for ch in t)
            if not has_letter:
                return False
            # Allow letters, digits, apostrophes, hyphens and spaces
            for ch in t:
                cat = _ud.category(ch)
                if cat.startswith('L') or cat == 'Nd' or ch in "'’-– " or ch == "-":
                    continue
                # Disallow other punctuation-like tokens
                if cat.startswith('P') or cat.startswith('Z'):
                    return False
            return True
        def _is_banned_token(s: str) -> bool:
            if s is None:
                return True
            t = s
            if self.words_only:
                return not _is_wordlike(t)
            if not self.ban_punct:
                return False
            t = t.strip()
            if t == "":
                return True
            if _is_box_or_repeat(t):
                return True
            # Ban tokens that are only punctuation/separators
            only_punct = True
            for ch in t:
                cat = _ud.category(ch)
                if not (cat.startswith('P') or cat.startswith('Z')):
                    only_punct = False
                    break
            return only_punct
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
                # Optional: remove punctuation-ish tokens from selection set
                if (self.ban_punct or self.words_only) and len(inter) > 0:
                    inter = {i for i in inter if not _is_banned_token(tok.convert_ids_to_tokens(int(i)))}
                if len(inter) == 0:
                    entropy = float("inf")
                    avg_p = torch.stack(probs_sets, dim=0).mean(dim=0)
                    # Filter top tokens if requested
                    if (self.ban_punct or self.words_only):
                        toks = []
                        vals, idxs = torch.topk(avg_p, k= max(self.topk*25, 200))
                        for v,i in zip(vals.tolist(), idxs.tolist()):
                            ts = tok.convert_ids_to_tokens(int(i))
                            if not _is_banned_token(ts):
                                toks.append((ts, v))
                            if len(toks) >= self.topk:
                                break
                        top_tokens = toks
                    else:
                        top_tokens = top_tokens_from_probs(tok, avg_p, topk=self.topk)
                else:
                    avg_p = torch.stack(probs_sets, dim=0).mean(dim=0)
                    mask = torch.zeros_like(avg_p)
                    idx = torch.tensor(sorted(list(inter)), dtype=torch.long)
                    mask[idx] = 1.0
                    p_sel = (avg_p * mask)
                    p_sel = p_sel / p_sel.sum()
                    entropy = float((-p_sel[p_sel>0].log() * p_sel[p_sel>0]).sum().item())
                    if (self.ban_punct or self.words_only):
                        toks = []
                        vals, idxs = torch.topk(avg_p, k= max(self.topk*25, 200))
                        for v,i in zip(vals.tolist(), idxs.tolist()):
                            ts = tok.convert_ids_to_tokens(int(i))
                            if not _is_banned_token(ts):
                                toks.append((ts, v))
                            if len(toks) >= self.topk:
                                break
                        top_tokens = toks
                    else:
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
    # Steering schedule (default steer during generation only)
    positions: str = "first_n"  # "all" or "first_n"
    first_n: int = 32
    alpha_decay: float = 0.0
    # Embeddings
    embed_model: Optional[str] = "none"  # explicit: disabled by default; set HF encoder id to enable
    embed_provider: Optional[str] = None  # 'st' | 'hf'
    plot_out: Optional[str] = "artifacts/steer_margins.png"

    def __call__(self):
        dpo_m, dpo_tok = load_model_and_tokenizer(self.dpo_model)
        ref_m, ref_tok = load_model_and_tokenizer(self.ref_model)
        blob = torch.load(self.delta, map_location="cpu")
        delta = blob["delta"]  # [k, d]
        k = int(blob.get("k", delta.shape[0]))
        # Default to stored layer index if not provided
        if self.layer_idx is None and (blob.get("layer_idx") is not None):
            try:
                self.layer_idx = int(blob["layer_idx"])  # type: ignore
            except Exception:
                pass
        j = self.j if self.j is not None else max(0, min(4, k-1))
        prompts = [p for p in Path(self.prompts).read_text().splitlines() if p.strip()]
        # Assert delta dim matches hidden size if available
        if hasattr(dpo_m.config, "hidden_size"):
            assert delta.shape[1] == dpo_m.config.hidden_size, f"Δ dim {delta.shape[1]} != model hidden_size {dpo_m.config.hidden_size}"
        margins = []
        un_texts, st_texts = [], []
        for p in prompts:
            # Unsteered
            full0, comp0 = __import__("dpo_adl.eval.steering", fromlist=["generate_text_with_completion"]).generate_text_with_completion(
                dpo_m, dpo_tok, p, max_new_tokens=self.max_new_tokens, temperature=self.temperature
            )
            m0 = dpo_margin(dpo_m, ref_m, dpo_tok, ref_tok, p, comp0)
            # Steered
            full1, comp1 = __import__("dpo_adl.eval.steering", fromlist=["steered_generation_with_completion"]).steered_generation_with_completion(
                dpo_m, dpo_tok, p, delta_vec=delta[j], layer_idx=self.layer_idx, alpha=self.alpha,
                positions=self.positions, first_n=self.first_n, alpha_decay=self.alpha_decay,
                max_new_tokens=self.max_new_tokens, temperature=self.temperature,
            )
            m1 = dpo_margin(dpo_m, ref_m, dpo_tok, ref_tok, p, comp1)
            margins.append({"prompt": p, "unsteered": m0, "steered": m1, "delta": m1 - m0})
            print(json.dumps({"prompt": p, "margin_unsteered": m0, "margin_steered": m1, "delta": m1-m0}))
            un_texts.append(full0)
            st_texts.append(full1)
        deltas = [m["delta"] for m in margins]
        log.info(f"Avg DPO margin delta over {len(deltas)} prompts: {sum(deltas)/max(1,len(deltas)):.4f}")
        # Embedding similarity: compare steered vs unsteered to ref_model outputs
        if self.embed_model and str(self.embed_model).lower() not in {"none", "off", "skip"}:
            if not self.embed_provider or self.embed_provider.lower() not in {"hf", "huggingface"}:
                raise ValueError("Embeddings enabled but --embed_provider hf not set. No fallbacks.")
            # Generate reference outputs from ref_model on same prompts
            ref_texts = [generate_text(ref_m, ref_tok, p, max_new_tokens=self.max_new_tokens, temperature=self.temperature) for p in prompts]
            emb = Embedder(self.embed_model, provider=self.embed_provider)
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
    embed_model: str = "none"  # explicit: embeddings disabled by default; pass HF encoder id to enable
    embed_provider: Optional[str] = None  # set to 'hf' when enabling embeddings
    make_pdf: bool = True
    # Embedding baseline target
    embed_to: str = "ref"  # "ref" or "dpo-chosen"
    embed_ds_name: Optional[str] = None
    embed_ds_split: str = "train_prefs"
    embed_ds_n: int = 1000
    # Δ source and selection strategy
    delta_source: str = "fineweb"  # "fineweb" or "dpo-chosen"
    delta_ds_name: Optional[str] = None  # defaults to embed_ds_name if None
    delta_ds_split: str = "train_prefs"
    delta_ds_n: int = 1000
    select_by: str = "entropy"  # "entropy" or "margin"
    # Steering schedule
    positions: str = "first_n"  # "all" or "first_n"
    first_n: int = 16
    alpha_decay: float = 0.0
    # Holdout selection split
    select_frac: float = 0.5  # fraction of prompts used for selection; rest for holdout
    # Steering norm-match (scale Δ to expected norm at layer)
    steering_norm_match: bool = False
    steering_norm_sample: int = 256
    # Pretty plotting (blog-ready)
    pretty_plot: bool = False
    # Orthogonalization
    orthogonalize: bool = False
    base_model: Optional[str] = None
    # Speed option: skip Patchscope sweep (entropy, token lists)
    skip_patchscope: bool = False
    # Filter punctuation tokens when computing Patchscope entropy
    ban_punct: bool = False
    words_only: bool = False
    # Subnetwork integration: optional paths for Δθ profile and gradnorm jsonl; sweep top-N layers
    subnetwork_profile: Optional[str] = None
    gradnorm_jsonl: Optional[str] = None
    layer_sweep_topn: int = 8

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
            "embed_provider": self.embed_provider,
        }
        (exp_dir / "config.json").write_text(json.dumps(cfg, indent=2))

        # 1) Build Δ
        model, tok = load_model_and_tokenizer(self.dpo_model)
        # Select probe texts for Δ
        if self.delta_source == "fineweb":
            from dpo_adl.data.fineweb import iter_probe_texts
            texts = list(iter_probe_texts("HuggingFaceFW/fineweb-edu", "train", self.n_probe, seed=0))
        elif self.delta_source == "dpo-chosen":
            from dpo_adl.train.datasets import chosen_texts_from_spec
            name = self.delta_ds_name or self.embed_ds_name
            assert name is not None, "Provide --delta_ds_name or --embed_ds_name for dpo-chosen Δ source"
            texts = chosen_texts_from_spec(name, self.delta_ds_split, self.delta_ds_n, seed=0)
        else:
            raise ValueError("delta_source must be 'fineweb' or 'dpo-chosen'")
        from dpo_adl.diff.delta_builder import build_delta
        delta = build_delta(self.ref_model, self.dpo_model, texts, k=self.k, layer_idx=self.layer_idx, batch_size=self.batch_size)
        torch.save({"delta": delta, "k": self.k, "layer_idx": self.layer_idx}, exp_dir / "artifacts" / "delta.pt")
        # Optional Δ_SFT for orthogonalization
        delta_sft = None
        if self.orthogonalize:
            assert self.base_model is not None, "Provide --base_model when --orthogonalize is set."
            delta_sft = build_delta(self.base_model, self.ref_model, texts, k=self.k, layer_idx=self.layer_idx, batch_size=self.batch_size)
            torch.save({"delta": delta_sft, "k": self.k, "layer_idx": self.layer_idx}, exp_dir / "artifacts" / "delta_sft.pt")

        # 2) Patchscope alpha sweep & entropy plots per j (optional)
        alphas = [float(x) for x in self.alpha_sweep.split(",") if x.strip()]
        prompts = [p for p in Path(self.prompts).read_text().splitlines() if p.strip()]
        # Split into selection and holdout sets
        n_total = len(prompts)
        assert n_total >= 2, "Need at least 2 prompts to create selection+holdout."
        n_sel = max(1, int(round(self.select_frac * n_total)))
        n_sel = min(n_sel, n_total - 1)  # keep at least 1 for holdout
        sel_prompts = prompts[:n_sel]
        hold_prompts = prompts[n_sel:]
        if not self.skip_patchscope:
            from dpo_adl.patchscope.token_identity import patchscope_logits, DEFAULT_PROMPTS
            token_id_prompts = DEFAULT_PROMPTS  # use token-identity prompts for Patchscope
            # Estimate expected norm (optional) for Patchscope
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
            # Token filters for Patchscope
            import unicodedata as _ud
            def _is_box_or_repeat(t: str) -> bool:
                if len(t) >= 3 and all(ch in "-_=~." for ch in t):
                    return True
                for ch in t:
                    code = ord(ch)
                    if (0x2500 <= code <= 0x257F) or (0x2580 <= code <= 0x259F):
                        return True
                return False
            def _is_wordlike(s: str) -> bool:
                if s is None:
                    return False
                t = s.replace("▁", "").replace("Ġ", "").strip()
                if t == "":
                    return False
                has_letter = any(_ud.category(ch).startswith('L') for ch in t)
                if not has_letter:
                    return False
                for ch in t:
                    cat = _ud.category(ch)
                    if cat.startswith('L') or cat == 'Nd' or ch in "'’-– " or ch == "-":
                        continue
                    if cat.startswith('P') or cat.startswith('Z'):
                        return False
                return True
            def _is_banned_token(s: str) -> bool:
                if s is None:
                    return True
                if self.words_only:
                    return not _is_wordlike(s)
                if not self.ban_punct:
                    return False
                t = s.strip()
                if t == "":
                    return True
                if _is_box_or_repeat(t):
                    return True
                only_punct = True
                for ch in t:
                    cat = _ud.category(ch)
                    if not (cat.startswith('P') or cat.startswith('Z')):
                        only_punct = False
                        break
                return only_punct

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
                    # Optional: remove punctuation / non-word tokens from selection set
                    if (self.ban_punct or self.words_only) and len(inter) > 0:
                        inter = {i for i in inter if not _is_banned_token(tok.convert_ids_to_tokens(int(i)))}
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
                    if (self.ban_punct or self.words_only):
                        toks = []
                        vals, idxs = torch.topk(avg_p, k=max(200, 25*20))
                        for v, i in zip(vals.tolist(), idxs.tolist()):
                            ts = tok.convert_ids_to_tokens(int(i))
                            if not _is_banned_token(ts):
                                toks.append((ts, v))
                            if len(toks) >= 20:
                                break
                        top_tokens = toks
                    else:
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
                    # Log orthogonalization diagnostics
                    cos = float(torch.nn.functional.cosine_similarity(delta[j].unsqueeze(0), delta_sft[j].unsqueeze(0)).item())
                    diag = {"j": j, "cos_dpo_sft": cos, "norm_dpo": float(delta[j].norm().item()), "norm_sft": float(delta_sft[j].norm().item())}
                    # Accumulate per-j diagnostics to write later
                    __import__("json")
                    (exp_dir / "artifacts").mkdir(exist_ok=True, parents=True)
                    _orth_diag_path = exp_dir / "artifacts" / "orth_metrics.jsonl"
                    with open(_orth_diag_path, "a") as f:
                        f.write(__import__("json").dumps(diag) + "\n")
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
                        if (self.ban_punct or self.words_only):
                            toks = []
                            vals, idxs = torch.topk(avg_p, k=max(200, 25*20))
                            for v, i in zip(vals.tolist(), idxs.tolist()):
                                ts = tok.convert_ids_to_tokens(int(i))
                                if not _is_banned_token(ts):
                                    toks.append((ts, v))
                                if len(toks) >= 20:
                                    break
                            top_tokens_o = toks
                        else:
                            top_tokens_o = top_tokens_from_probs(tok, avg_p, topk=20)
                        cand_o = {"j": j, "alpha": a, "entropy": entropy, "top": top_tokens_o}
                        if best_o is None or cand_o["entropy"] < best_o["entropy"]:
                            best_o = cand_o
                    entropies_per_j_orth[j] = ent_per_a_o
                    best_by_j_orth[j] = best_o
            plot_entropy_vs_alpha(entropies_per_j, exp_dir / "plots", pretty=self.pretty_plot)
            if self.orthogonalize:
                # Save orth entropy plots and rename with suffix
                plot_entropy_vs_alpha(entropies_per_j_orth, exp_dir / "plots", pretty=self.pretty_plot)
                for j in entropies_per_j_orth.keys():
                    p_png = exp_dir / "plots" / f"patchscope_entropy_j{j}.png"
                    p_svg = exp_dir / "plots" / f"patchscope_entropy_j{j}.svg"
                    if p_png.exists():
                        p_png.rename(exp_dir / "plots" / f"patchscope_entropy_j{j}_orth.png")
                    if p_svg.exists():
                        p_svg.rename(exp_dir / "plots" / f"patchscope_entropy_j{j}_orth.svg")
            # Choose best j overall by entropy baseline
            best_entropy = min(best_by_j.values(), key=lambda r: r["entropy"])
            (exp_dir / "artifacts" / "patchscope_best.json").write_text(json.dumps({"best": best_entropy, "by_j": best_by_j}, ensure_ascii=False, indent=2))
            if self.orthogonalize:
                best_overall_orth = min(best_by_j_orth.values(), key=lambda r: r["entropy"])
                (exp_dir / "artifacts" / "patchscope_best_orth.json").write_text(json.dumps({"best": best_overall_orth, "by_j": best_by_j_orth}, ensure_ascii=False, indent=2))
        else:
            # Skipped patchscope to accelerate analysis
            (exp_dir / "artifacts").mkdir(parents=True, exist_ok=True)
            (exp_dir / "artifacts" / "patchscope_best.json").write_text(json.dumps({"skipped": True}))

        # 3) Margin-driven selection (optional)
        ref_m, ref_tok = load_model_and_tokenizer(self.ref_model)
        from dpo_adl.eval.steering import batched_eval_margins
        if self.select_by == "margin":
            margin_scores = {}
            best_margin = None
            # Optional steering norm-match scaling
            steering_expected_norm = None
            if self.steering_norm_match:
                L = num_layers(model)
                layer_idx = self.layer_idx if self.layer_idx is not None else L // 2
                from dpo_adl.backends.hf_hooks import estimate_expected_norm
                # Build a sample from selection prompts for norm estimation
                base = sel_prompts if len(sel_prompts) > 0 else prompts
                texts = (base * ((self.steering_norm_sample // max(1, len(base))) + 1))[: self.steering_norm_sample]
                steering_expected_norm = estimate_expected_norm(model, tok, texts, layer_idx, k_first_tokens=8, skip_first=3, batch_size=32)
            for j in range(delta.shape[0]):
                for a in alphas:
                    # Optionally scale Δ to expected norm
                    d_vec = delta[j]
                    if self.steering_norm_match:
                        import torch as _torch
                        denom = float(_torch.linalg.vector_norm(d_vec).item()) + 1e-6
                        d_vec = d_vec * (float(steering_expected_norm) / denom)
                    res_try = batched_eval_margins(
                        sel_prompts, model, ref_m, tok, ref_tok, delta_vec=d_vec, layer_idx=self.layer_idx,
                        alpha=a, positions=self.positions, first_n=self.first_n, alpha_decay=self.alpha_decay,
                        max_new_tokens=self.max_new_tokens, temperature=self.temperature,
                    )
                    avg_delta = float(sum((m1 - m0) for (_, _, _, m0, m1) in res_try) / max(1, len(res_try)))
                    margin_scores[(j, a)] = avg_delta
                    cand = {"j": j, "alpha": a, "avg_margin_delta": avg_delta}
                    if best_margin is None or avg_delta > best_margin["avg_margin_delta"]:
                        best_margin = cand
            (exp_dir / "artifacts" / "selection_margin.json").write_text(json.dumps({"best": best_margin, "scores": {f"{k[0]}:{k[1]}": v for k, v in margin_scores.items()}}, indent=2))
            if 'best_entropy' in locals():
                best_overall = {"j": best_margin["j"], "alpha": best_margin["alpha"], "entropy": best_entropy["entropy"], "top": best_entropy["top"]}
            else:
                best_overall = {"j": best_margin["j"], "alpha": best_margin["alpha"]}
        else:
            best_overall = best_entropy

        # Steering + margins + embeddings and plots (using selected best_overall) on HOLDOUT
        d_vec_best = delta[best_overall["j"]]
        if self.steering_norm_match:
            import torch as _torch
            denom = float(_torch.linalg.vector_norm(d_vec_best).item()) + 1e-6
            # Use selection-estimated norm if computed; otherwise estimate from holdout prompts
            if self.select_by == "margin":
                assert steering_expected_norm is not None, "steering_expected_norm should be computed during margin selection"
                norm_target = float(steering_expected_norm)
            else:
                L = num_layers(model)
                layer_idx = self.layer_idx if self.layer_idx is not None else L // 2
                from dpo_adl.backends.hf_hooks import estimate_expected_norm
                base = hold_prompts if len(hold_prompts) > 0 else prompts
                texts = (base * ((self.steering_norm_sample // max(1, len(base))) + 1))[: self.steering_norm_sample]
                norm_target = float(estimate_expected_norm(model, tok, texts, layer_idx, k_first_tokens=8, skip_first=3, batch_size=32))
            d_vec_best = d_vec_best * (norm_target / denom)

        res = batched_eval_margins(
            hold_prompts, model, ref_m, tok, ref_tok, delta_vec=d_vec_best, layer_idx=self.layer_idx,
            alpha=best_overall["alpha"], positions=self.positions, first_n=self.first_n, alpha_decay=self.alpha_decay,
            max_new_tokens=self.max_new_tokens, temperature=self.temperature,
        )
        # Save results JSON
        out_rows = []
        un, st, deltas = [], [], []
        un_texts, st_texts = [], []
        for (p, y0, y1, m0, m1) in res:
            out_rows.append({"prompt": p, "unsteered": m0, "steered": m1, "delta": m1 - m0})
            un.append(m0); st.append(m1); deltas.append(m1 - m0)
            un_texts.append(y0); st_texts.append(y1)
        (exp_dir / "artifacts" / "steer_margins_holdout.json").write_text(json.dumps(out_rows, indent=2))
        # Embedding similarity vs target (optional)
        do_embed = isinstance(self.embed_model, str) and self.embed_model.lower() not in {"none", "off", "skip"}
        if do_embed:
            if not self.embed_provider or self.embed_provider.lower() not in {"hf", "huggingface"}:
                raise ValueError("Embeddings enabled but --embed_provider hf not set. No fallbacks.")
            emb = Embedder(self.embed_model, provider=self.embed_provider)
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
        # Save holdout plots with suffix
        from pathlib import Path as _Path
        pdir = exp_dir / "plots"
        plot_margins_per_prompt(un, st, pdir, pretty=self.pretty_plot)
        p_png = _Path(pdir) / "margins_per_prompt.png"
        p_svg = _Path(pdir) / "margins_per_prompt.svg"
        if p_png.exists(): p_png.rename(pdir / "margins_per_prompt_holdout.png")
        if p_svg.exists(): p_svg.rename(pdir / "margins_per_prompt_holdout.svg")
        plot_margin_deltas(deltas, pdir, pretty=self.pretty_plot)
        p_png = _Path(pdir) / "margin_delta_box.png"
        p_svg = _Path(pdir) / "margin_delta_box.svg"
        if p_png.exists(): p_png.rename(pdir / "margin_delta_box_holdout.png")
        if p_svg.exists(): p_svg.rename(pdir / "margin_delta_box_holdout.svg")
        if do_embed:
            plot_embed_similarity(sim_un, sim_st, pdir, pretty=self.pretty_plot)
            p_png = _Path(pdir) / "embed_sim_unsteered.png"
            p_svg = _Path(pdir) / "embed_sim_unsteered.svg"
            if p_png.exists(): p_png.rename(pdir / "embed_sim_unsteered_holdout.png")
            if p_svg.exists(): p_svg.rename(pdir / "embed_sim_unsteered_holdout.svg")
            p_png = _Path(pdir) / "embed_sim_steered.png"
            p_svg = _Path(pdir) / "embed_sim_steered.svg"
            if p_png.exists(): p_png.rename(pdir / "embed_sim_steered_holdout.png")
            if p_svg.exists(): p_svg.rename(pdir / "embed_sim_steered_holdout.svg")
            p_png = _Path(pdir) / "embed_sim_side_by_side.png"
            p_svg = _Path(pdir) / "embed_sim_side_by_side.svg"
            if p_png.exists(): p_png.rename(pdir / "embed_sim_side_by_side_holdout.png")
            if p_svg.exists(): p_svg.rename(pdir / "embed_sim_side_by_side_holdout.svg")

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
            plot_margins_per_prompt(un_o, st_o, exp_dir / "plots", pretty=self.pretty_plot)
            p1_png = exp_dir / "plots" / "margins_per_prompt.png"
            p1_svg = exp_dir / "plots" / "margins_per_prompt.svg"
            if p1_png.exists(): p1_png.rename(exp_dir / "plots" / "margins_per_prompt_orth.png")
            if p1_svg.exists(): p1_svg.rename(exp_dir / "plots" / "margins_per_prompt_orth.svg")
            plot_margin_deltas(deltas_o, exp_dir / "plots", pretty=self.pretty_plot)
            p2_png = exp_dir / "plots" / "margin_delta_box.png"
            p2_svg = exp_dir / "plots" / "margin_delta_box.svg"
            if p2_png.exists(): p2_png.rename(exp_dir / "plots" / "margin_delta_box_orth.png")
            if p2_svg.exists(): p2_svg.rename(exp_dir / "plots" / "margin_delta_box_orth.svg")
            # Embedding sim for orth
            if do_embed:
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
                plot_embed_similarity(sim_un_o, sim_st_o, exp_dir / "plots", pretty=self.pretty_plot)
                p3_png = exp_dir / "plots" / "embed_sim_unsteered.png"
                p3_svg = exp_dir / "plots" / "embed_sim_unsteered.svg"
                p4_png = exp_dir / "plots" / "embed_sim_steered.png"
                p4_svg = exp_dir / "plots" / "embed_sim_steered.svg"
                p5_png = exp_dir / "plots" / "embed_sim_side_by_side.png"
                p5_svg = exp_dir / "plots" / "embed_sim_side_by_side.svg"
                if p3_png.exists(): p3_png.rename(exp_dir / "plots" / "embed_sim_unsteered_orth.png")
                if p3_svg.exists(): p3_svg.rename(exp_dir / "plots" / "embed_sim_unsteered_orth.svg")
                if p4_png.exists(): p4_png.rename(exp_dir / "plots" / "embed_sim_steered_orth.png")
                if p4_svg.exists(): p4_svg.rename(exp_dir / "plots" / "embed_sim_steered_orth.svg")
                if p5_png.exists(): p5_png.rename(exp_dir / "plots" / "embed_sim_side_by_side_orth.png")
                if p5_svg.exists(): p5_svg.rename(exp_dir / "plots" / "embed_sim_side_by_side_orth.svg")

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

        # 5) Optional: Subnetwork correlation/overlap (Δθ, gradnorm) vs Patchscope/Margins across layers
        if self.subnetwork_profile or self.gradnorm_jsonl:
            # Build candidate layer set from provided profiles
            top_layers_delta = []
            if self.subnetwork_profile:
                try:
                    prof = json.loads(Path(self.subnetwork_profile).read_text())
                    pl = prof.get("per_layer_l2", {})
                    top_layers_delta = sorted(((int(k), float(v)) for k, v in pl.items()), key=lambda x: x[1], reverse=True)[: self.layer_sweep_topn]
                except Exception as e:
                    log.warning(f"Failed to read subnetwork_profile: {e}")
            top_layers_grad = []
            if self.gradnorm_jsonl:
                try:
                    acc: dict[int, float] = {}
                    with open(self.gradnorm_jsonl, "r") as f:
                        for line in f:
                            rec = json.loads(line)
                            for lk, lv in rec.get("per_layer", {}).items():
                                li = int(lk)
                                acc[li] = acc.get(li, 0.0) + float(lv)
                    top_layers_grad = sorted(acc.items(), key=lambda x: x[1], reverse=True)[: self.layer_sweep_topn]
                except Exception as e:
                    log.warning(f"Failed to read gradnorm jsonl: {e}")
            cand_layers = {li for (li, _) in top_layers_delta} | {li for (li, _) in top_layers_grad}
            cand_layers = sorted(list(cand_layers))
            if not cand_layers:
                log.info("No candidate layers for subnetwork correlation; skipping")
                return
            # For each candidate layer, compute Patchscope best entropy and holdout margin delta
            from dpo_adl.patchscope.token_identity import patchscope_logits, DEFAULT_PROMPTS as _TOKID_PROMPTS
            # Expected norm for Patchscope per layer if norm_match
            def _expected_norm_for_layer(layer_idx: int) -> float | None:
                if not self.norm_match:
                    return None
                from dpo_adl.backends.hf_hooks import estimate_expected_norm
                texts = _TOKID_PROMPTS * 32
                try:
                    return float(estimate_expected_norm(model, tok, texts, layer_idx, k_first_tokens=8, skip_first=3, batch_size=32))
                except Exception:
                    return None
            # Evaluate margins per layer on the same HOLDOUT prompts using best (j,alpha) by entropy at that layer
            layer_records = {}
            alphas = [float(x) for x in self.alpha_sweep.split(",") if x.strip()]
            for li in cand_layers:
                # Patchscope at this layer
                expected_norm_li = _expected_norm_for_layer(li)
                best_li = None
                for j in range(delta.shape[0]):
                    for a in alphas:
                        probs_sets = []
                        for p in _TOKID_PROMPTS:
                            probs = patchscope_logits(
                                model, tok, li, delta[j], a, p, self.sentinel, norm_match=self.norm_match, expected_norm=expected_norm_li
                            )
                            probs_sets.append(probs)
                        top_sets = [torch.topk(p, k=100).indices for p in probs_sets]
                        inter = set(top_sets[0].tolist())
                        for s in top_sets[1:]:
                            inter &= set(s.tolist())
                        avg_p = torch.stack(probs_sets, dim=0).mean(dim=0)
                        if self.ban_punct and len(inter) > 0:
                            import unicodedata as _ud
                            def _is_box_or_repeat(t: str) -> bool:
                                if len(t) >= 3 and all(ch in "-_=~." for ch in t):
                                    return True
                                for ch in t:
                                    code = ord(ch)
                                    if (0x2500 <= code <= 0x257F) or (0x2580 <= code <= 0x259F):
                                        return True
                                return False
                            def _is_banned_token(s: str) -> bool:
                                if s is None:
                                    return False
                                t = s.strip()
                                if t == "":
                                    return True
                                if _is_box_or_repeat(t):
                                    return True
                                if any(ch.isalnum() for ch in t):
                                    return False
                                for ch in t:
                                    cat = _ud.category(ch)
                                    if not (cat.startswith('P') or cat.startswith('Z')):
                                        return False
                                return True
                            inter = {i for i in inter if not _is_banned_token(tok.convert_ids_to_tokens(int(i)))}
                        if len(inter) == 0:
                            p_sel = avg_p
                            entropy = float("inf")
                        else:
                            mask = torch.zeros_like(avg_p)
                            idx = torch.tensor(sorted(list(inter)), dtype=torch.long)
                            mask[idx] = 1.0
                            p_sel = (avg_p * mask)
                            p_sel = p_sel / p_sel.sum()
                            entropy = float((-p_sel[p_sel>0].log() * p_sel[p_sel>0]).sum().item())
                        cand = {"layer": li, "j": j, "alpha": a, "entropy": entropy}
                        if best_li is None or cand["entropy"] < best_li["entropy"]:
                            best_li = cand
                # Margins on HOLDOUT using best_li
                d_vec = delta[best_li["j"]]
                if self.steering_norm_match:
                    from dpo_adl.backends.hf_hooks import estimate_expected_norm
                    # Estimate per-layer steering norm on a small subset of holdout prompts
                    base = hold_prompts if len(hold_prompts) > 0 else prompts
                    texts = (base * ((self.steering_norm_sample // max(1, len(base))) + 1))[: self.steering_norm_sample]
                    norm_target = float(estimate_expected_norm(model, tok, texts, li, k_first_tokens=8, skip_first=3, batch_size=32))
                    denom = float(torch.linalg.vector_norm(d_vec).item()) + 1e-6
                    d_vec = d_vec * (norm_target / denom)
                res_li = batched_eval_margins(
                    hold_prompts, model, ref_m, tok, ref_tok, delta_vec=d_vec, layer_idx=li,
                    alpha=best_li["alpha"], positions=self.positions, first_n=self.first_n, alpha_decay=self.alpha_decay,
                    max_new_tokens=self.max_new_tokens, temperature=self.temperature,
                )
                avg_delta_li = float(sum((m1 - m0) for (_, _, _, m0, m1) in res_li) / max(1, len(res_li)))
                layer_records[li] = {"best": best_li, "avg_margin_delta": avg_delta_li}

            # Correlations (Spearman) between layer ranks
            def _spearman(xs: list[float], ys: list[float]) -> float:
                assert len(xs) == len(ys) and len(xs) > 1
                # Convert to ranks (1..n), largest -> rank 1
                import math
                n = len(xs)
                rx = {k: i+1 for i, (k, _) in enumerate(sorted(enumerate(xs), key=lambda kv: kv[1], reverse=True))}
                ry = {k: i+1 for i, (k, _) in enumerate(sorted(enumerate(ys), key=lambda kv: kv[1], reverse=True))}
                d2 = 0
                for i in range(n):
                    d = rx[i] - ry[i]
                    d2 += d*d
                return 1 - (6*d2) / (n*(n*n - 1))

            # Build aligned vectors over cand_layers
            delta_scores = []
            if top_layers_delta:
                pl_map = {int(k): float(v) for (k, v) in top_layers_delta}
                delta_scores = [pl_map.get(li, 0.0) for li in cand_layers]
            grad_scores = []
            if top_layers_grad:
                gn_map = {int(k): float(v) for (k, v) in top_layers_grad}
                grad_scores = [gn_map.get(li, 0.0) for li in cand_layers]
            ent_scores = [-float(layer_records[li]["best"]["entropy"]) for li in cand_layers]
            md_scores = [float(layer_records[li]["avg_margin_delta"]) for li in cand_layers]
            corrs = {}
            try:
                if delta_scores:
                    corrs["spearman_delta_vs_entropy"] = _spearman(delta_scores, ent_scores)
                    corrs["spearman_delta_vs_margin"] = _spearman(delta_scores, md_scores)
            except Exception:
                pass
            try:
                if grad_scores:
                    corrs["spearman_grad_vs_entropy"] = _spearman(grad_scores, ent_scores)
                    corrs["spearman_grad_vs_margin"] = _spearman(grad_scores, md_scores)
            except Exception:
                pass

            (exp_dir / "artifacts" / "subnetwork_correlation.json").write_text(json.dumps({
                "candidate_layers": cand_layers,
                "top_layers_by_delta": top_layers_delta,
                "top_layers_by_gradnorm": top_layers_grad,
                "per_layer_best": layer_records,
                "correlations": corrs,
            }, indent=2))


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
    subset: Optional[str] = None
    filter_keywords: Optional[List[str]] = None
    save_steps: int = 0

    def __call__(self):
        ds = load_hf_preference_dataset(self.dataset, self.split, self.n_pairs, seed=self.seed, subset=self.subset, filter_keywords=self.filter_keywords)
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
            save_steps=self.save_steps,
            eval_n=256,
        )
        train_dpo_on_dataset(cfg, ds)
        print(json.dumps({"trained_model": self.out_dir, "dataset": self.dataset, "split": self.split, "pairs": self.n_pairs, "steps": self.max_steps}))



@dataclass
class CmdTrainGRPOHF:
    ref_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    dataset: str = "Anthropic/hh-rlhf"
    split: str = "train"
    n_pairs: int = 4000
    out_dir: str = "assets/trained/grpo_hf"
    learning_rate: float = 1e-5
    max_steps: int = 2000
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    max_length: int = 256
    max_prompt_length: int = 128
    seed: int = 0
    subset: Optional[str] = None
    filter_keywords: Optional[List[str]] = None
    save_steps: int = 0
    reward_preset: Optional[str] = None

    def __call__(self):
        ds = load_hf_preference_dataset(self.dataset, self.split, self.n_pairs, seed=self.seed, subset=self.subset, filter_keywords=self.filter_keywords)
        # Infer reward preset if not provided
        preset = self.reward_preset
        if preset is None:
            dn = self.dataset.lower()
            if "hh-rlhf" in dn or (self.filter_keywords and len(self.filter_keywords) > 0):
                preset = "hh_refusal"
            elif "webgpt" in dn:
                preset = "webgpt"
            elif "summarize" in dn:
                preset = "summarize"
        cfg = GRPOTrainConfig(
            ref_model=self.ref_model,
            out_dir=self.out_dir,
            learning_rate=self.learning_rate,
            max_steps=self.max_steps,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            max_length=self.max_length,
            max_prompt_length=self.max_prompt_length,
            seed=self.seed,
            save_steps=self.save_steps,
            eval_n=256,
            reward_preset=preset,
        )
        train_grpo_on_dataset(cfg, ds)
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
    elif cmd == "train-grpo-hf":
        tyro.extras.set_accent_color("yellow")
        tyro.cli(CmdTrainGRPOHF, args=args)()
    elif cmd == "profile-subnetwork":
        @dataclass
        class CmdProfileSubnetwork:
            ref_model: str
            dpo_model: str
            out_dir: str

            def __call__(self):
                prof = profile_param_deltas(self.ref_model, self.dpo_model)
                save_profile(prof, Path(self.out_dir))
                print(json.dumps({"out": self.out_dir, "grand_l2": prof.get("grand_l2", 0.0)}))
        tyro.extras.set_accent_color("green")
        tyro.cli(CmdProfileSubnetwork, args=args)()
    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
