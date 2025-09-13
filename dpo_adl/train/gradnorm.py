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


class ParamGradSparsityCallback(TrainerCallback):
    """Accumulates per-parameter gradient norms across training for sparsity analysis.

    - sum_l2[name]: sum over steps of ||g_t||_F for that parameter tensor
    - sum_rms[name]: sum over steps of ||g_t||_F / sqrt(numel)

    Writes a single JSON with per-parameter stats and per-layer aggregates at train end.
    Only process_index==0 logs to avoid duplication under DDP.
    """

    def __init__(self, out_file: Path):
        self.out_file = Path(out_file)
        self.out_file.parent.mkdir(parents=True, exist_ok=True)
        self.sum_l2: Dict[str, float] = {}
        self.sum_rms: Dict[str, float] = {}
        self.numel: Dict[str, int] = {}

    def on_substep_end(self, args, state, control, **kwargs):
        # Only log on process 0
        if getattr(args, "process_index", 0) != 0:
            return
        trainer = kwargs.get("trainer", None)
        model = kwargs.get("model", None)
        if model is None and trainer is not None:
            model = getattr(trainer, "model", None)
        if model is None:
            return
        for name, p in model.named_parameters():
            g = p.grad
            if g is None or g.numel() == 0:
                continue
            nrm = float(torch.linalg.vector_norm(g.detach(), ord=2).item())
            n = int(g.numel())
            self.sum_l2[name] = self.sum_l2.get(name, 0.0) + nrm
            self.sum_rms[name] = self.sum_rms.get(name, 0.0) + (nrm / max(1, n) ** 0.5)
            if name not in self.numel:
                self.numel[name] = n

    def on_train_end(self, args, state, control, **kwargs):
        # Only write on process 0
        if getattr(args, "process_index", 0) != 0:
            return
        # Aggregate per-layer
        per_layer: Dict[int, Dict[str, float]] = {}
        for name, s in self.sum_l2.items():
            idx = _bucket_name(name)
            if idx is None:
                continue
            slot = per_layer.setdefault(idx, {"sum_l2": 0.0, "sum_rms": 0.0, "params": 0})
            slot["sum_l2"] += s
            slot["sum_rms"] += self.sum_rms.get(name, 0.0)
            slot["params"] += 1
        out = {
            "per_param": {
                name: {"sum_l2": self.sum_l2.get(name, 0.0), "sum_rms": self.sum_rms.get(name, 0.0), "numel": self.numel.get(name, 0)}
                for name in self.sum_l2.keys()
            },
            "per_layer": per_layer,
        }
        self.out_file.write_text(json.dumps(out, indent=2))
