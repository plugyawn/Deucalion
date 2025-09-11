from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import json
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM


_LAYER_RE = re.compile(r"layers\.(\d+)\.")


def _bucket_param(name: str) -> Tuple[str, int | None]:
    """Return bucket ('layer', idx) or ('other', None) for a parameter name.

    Works for most HF decoder models where layers.N appears in the path.
    """
    m = _LAYER_RE.search(name)
    if m:
        return ("layer", int(m.group(1)))
    return ("other", None)


def _load_model_maybe_lora(model_id: str, ref_model_id: Optional[str] = None):
    """Load a model which may be a full HF model or a LoRA adapter directory.

    If `model_id` points to a directory containing a LoRA adapter (adapter_config.json),
    load the `ref_model_id` base and merge the adapter weights for comparison.
    """
    # Try direct load first
    try:
        m = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu")
        m.eval()
        return m
    except Exception:
        pass

    p = Path(model_id)
    if p.exists() and (p / "adapter_config.json").exists():
        assert ref_model_id is not None, "ref_model_id required to load LoRA adapter for profiling"
        base = AutoModelForCausalLM.from_pretrained(ref_model_id, device_map="cpu")
        base.eval()
        try:
            from peft import PeftModel
        except Exception as e:
            raise RuntimeError(
                "peft is required to load LoRA adapter; install with `pip install peft`"
            ) from e
        peft_m = PeftModel.from_pretrained(base, model_id)
        # Try to merge adapter into base weights for straightforward delta computation
        try:
            merged = peft_m.merge_and_unload()
            merged.eval()
            return merged
        except Exception:
            # Fall back to returning the PEFT-wrapped model (still usable for named_parameters)
            peft_m.eval()
            return peft_m

    # If it's a local directory without adapter file, re-raise the original error
    raise RuntimeError(f"Could not load model from {model_id}: not a HF model or LoRA adapter directory")


@torch.inference_mode()
def profile_param_deltas(ref_model_id: str, dpo_model_id: str) -> Dict:
    """Compute per-layer L2 norm of parameter deltas between dpo and ref models.

    Returns a dict with per-layer totals and a global summary.
    """
    ref = AutoModelForCausalLM.from_pretrained(ref_model_id, device_map="cpu"); ref.eval()
    dpo = _load_model_maybe_lora(dpo_model_id, ref_model_id=ref_model_id)

    # Build name->tensor maps on CPU
    ref_params = {n: p.detach().to(torch.float32) for n, p in ref.named_parameters()}
    del ref
    dpo_params = {n: p.detach().to(torch.float32) for n, p in dpo.named_parameters()}
    del dpo

    totals: Dict[int, float] = {}
    other_total: float = 0.0
    grand: float = 0.0
    count: Dict[int, int] = {}

    for n, dp in dpo_params.items():
        rp = ref_params.get(n)
        if rp is None or rp.shape != dp.shape:
            continue
        diff = (dp - rp).reshape(-1)
        val = float(torch.linalg.vector_norm(diff, ord=2).item())
        grand += val
        bucket, idx = _bucket_param(n)
        if bucket == "layer" and idx is not None:
            totals[idx] = totals.get(idx, 0.0) + val
            count[idx] = count.get(idx, 0) + 1
        else:
            other_total += val

    summary = {
        "per_layer_l2": totals,
        "other_l2": other_total,
        "grand_l2": grand,
        "layer_count": {k: count.get(k, 0) for k in totals.keys()},
    }
    return summary


def save_profile(profile: Dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "subnetwork_profile.json").write_text(json.dumps(profile, indent=2))
    # Plot bar of per-layer contributions (sorted by idx)
    items = sorted(profile["per_layer_l2"].items(), key=lambda x: x[0])
    if not items:
        return
    xs = [k for k, _ in items]
    ys = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(xs, ys)
    ax.set_xlabel("layer index")
    ax.set_ylabel("L2 norm of Δθ")
    ax.set_title("Per-layer parameter delta (DPO vs Ref)")
    plt.tight_layout()
    fig.savefig(out_dir / "subnetwork_delta_bar.png")
    plt.close(fig)


def plot_gradnorm_layers(jsonl_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    per_layer_acc: Dict[int, float] = {}
    total_acc: List[float] = []
    steps: List[int] = []
    with open(jsonl_path, "r") as f:
        for line in f:
            rec = json.loads(line)
            steps.append(int(rec.get("step", 0)))
            total_acc.append(float(rec.get("total", 0.0)))
            for k, v in rec.get("per_layer", {}).items():
                ki = int(k)
                per_layer_acc[ki] = per_layer_acc.get(ki, 0.0) + float(v)
    # Total over steps per layer bar
    if per_layer_acc:
        items = sorted(per_layer_acc.items(), key=lambda x: x[0])
        xs = [k for k, _ in items]
        ys = [v for _, v in items]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(xs, ys)
        ax.set_xlabel("layer index")
        ax.set_ylabel("Σ grad L2 over steps")
        ax.set_title("Per-layer cumulative grad-norm")
        plt.tight_layout()
        fig.savefig(out_dir / "gradnorm_layers_cumulative.png")
        plt.close(fig)
    # Total grad per step line
    if steps:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(steps, total_acc)
        ax.set_xlabel("step")
        ax.set_ylabel("total grad L2")
        ax.set_title("Total grad-norm per step")
        plt.tight_layout()
        fig.savefig(out_dir / "gradnorm_total_per_step.png")
        plt.close(fig)
