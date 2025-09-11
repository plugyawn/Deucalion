from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict

import torch
from transformers import TrainerCallback


_LAYER_RE = re.compile(r"layers\.(\d+)\.")


def _bucket_name(name: str) -> int | None:
    m = _LAYER_RE.search(name)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


class LayerGradNormCallback(TrainerCallback):
    """Logs per-layer gradient L2 norms each accumulation substep to a JSONL file.

    Each line: {"step": int, "total": float, "per_layer": {idx: value}, "other": float}

    We hook into on_substep_end so grads are populated (before optimizer zero_grad()).
    """

    def __init__(self, out_file: Path):
        self.out_file = Path(out_file)
        self.out_file.parent.mkdir(parents=True, exist_ok=True)

    def _log_gradnorms(self, model, state):
        per_layer: Dict[int, float] = {}
        other = 0.0
        total = 0.0
        saw_grad = False
        for name, p in model.named_parameters():
            g = p.grad
            if g is None:
                continue
            if g.numel() == 0:
                continue
            saw_grad = True
            val = float(torch.linalg.vector_norm(g.detach(), ord=2).item())
            total += val
            idx = _bucket_name(name)
            if idx is not None:
                per_layer[idx] = per_layer.get(idx, 0.0) + val
            else:
                other += val
        if not saw_grad:
            raise RuntimeError("LayerGradNormCallback: no gradients found at substep; check training loop")
        rec = {
            "step": int(getattr(state, "global_step", 0)),
            "total": total,
            "per_layer": per_layer,
            "other": other,
        }
        with self.out_file.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    # Called after each accumulation substep, before optimizer step
    def on_substep_end(self, args, state, control, **kwargs):
        trainer = kwargs.get("trainer", None)
        model = kwargs.get("model", None)
        if model is None and trainer is not None:
            model = getattr(trainer, "model", None)
        if model is None:
            raise RuntimeError("LayerGradNormCallback: no model found in callback kwargs")
        self._log_gradnorms(model, state)
