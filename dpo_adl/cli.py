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
from dpo_adl.patchscope.token_identity import DEFAULT_PROMPTS, patchscope_logits, top_tokens_from_probs
from dpo_adl.utils.logging import get_logger


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
            # Use a small slice of probe texts from generic prompts as a fallback
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
            import torch as _torch
            sim_un = emb.cosine_sim_matrix(E_un, E_ref).diag().tolist()
            sim_st = emb.cosine_sim_matrix(E_st, E_ref).diag().tolist()
            # Plot
            try:
                import matplotlib.pyplot as plt
                import numpy as np
                x = np.arange(len(prompts))
                w = 0.35
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.bar(x - w/2, sim_un, width=w, label="unsteered→ref sim")
                ax.bar(x + w/2, sim_st, width=w, label="steered→ref sim")
                ax.set_ylabel("cosine sim")
                ax.set_xlabel("prompt idx")
                ax.set_title("Embedding similarity vs ref_model outputs")
                ax.legend()
                Path(self.plot_out).parent.mkdir(parents=True, exist_ok=True)
                plt.tight_layout()
                plt.savefig(self.plot_out)
                log.info(f"Saved plot: {self.plot_out}")
            except Exception as e:
                log.warning(f"Plotting failed: {e}")


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
    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
