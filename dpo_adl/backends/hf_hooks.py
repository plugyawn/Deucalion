from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from pathlib import Path
from tqdm import tqdm


def load_model_and_tokenizer(model_id: str, dtype: str = "auto") -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a CausalLM with device_map="auto" and matching tokenizer.

    Uses bf16 if available and dtype=="auto". Suitable for large-model inference
    sharded across multiple GPUs.
    """
    torch_dtype = None
    if dtype == "auto":
        if torch.cuda.is_available():
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            torch_dtype = torch.float32
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    # Detect PEFT adapter directory and load accordingly
    model_path = Path(model_id)
    peft_candidates = {"adapter_config.json", "adapter_model.safetensors"}
    is_peft_dir = model_path.is_dir() and any((model_path / f).exists() for f in peft_candidates)
    if is_peft_dir:
        from peft import AutoPeftModelForCausalLM  # require peft when adapter detected
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    model.eval()
    # Log unique devices used (helps validate multi-GPU sharding), fail fast on inconsistencies
    devs = {str(p.device) for p in model.parameters()}
    print(f"[dpo-adl] model devices: {sorted(list(devs))}")
    if torch.cuda.is_available():
        assert any("cuda" in d for d in devs), "CUDA available but model not on GPU devices."
    return model, tok


def resolve_decoder_layers(model: PreTrainedModel):
    """Return the list-like module of decoder layers for common HF CausalLMs.

    Handles PEFT-wrapped models by unwrapping to the base model, then checks
    typical attributes in order.
    """
    m = model
    # Unwrap PEFT models
    if hasattr(m, "get_base_model"):
        m = m.get_base_model()
    elif hasattr(m, "base_model"):
        m = getattr(m, "base_model")

    # Direct container on base model (e.g., Qwen2Model has .layers)
    if hasattr(m, "layers"):
        return getattr(m, "layers")

    # Try common decoder layer containers on m and m.model
    candidates = [(m, "model", "layers"), (m, "transformer", "h"), (m, "gpt_neox", "layers")]
    if hasattr(m, "model"):
        mm = getattr(m, "model")
        candidates.extend([(mm, "model", "layers"), (mm, "transformer", "h"), (mm, "gpt_neox", "layers")])

    for obj, root_name, layers_name in candidates:
        root = getattr(obj, root_name, None)
        if root is None:
            continue
        layers = getattr(root, layers_name, None)
        if layers is not None:
            return layers
    raise AttributeError("Could not resolve decoder layers for this model; adjust resolve_decoder_layers().")


def num_layers(model: PreTrainedModel) -> int:
    if hasattr(model.config, "num_hidden_layers"):
        return int(model.config.num_hidden_layers)
    layers = resolve_decoder_layers(model)
    return len(layers)


@dataclass
class ResidCapture:
    layer_idx: int
    k_first_tokens: int = 5
    mode: str = "pre"  # "pre" captures resid_pre via forward_pre_hook

    def __post_init__(self):
        self.sum: Optional[torch.Tensor] = None  # [k, d]
        self.count: int = 0

    def _pre(self, module, inputs):
        (hidden_states, *_) = inputs
        hs = hidden_states[:, : self.k_first_tokens, :]
        hs_cpu = hs.detach().to("cpu", copy=False)
        if self.sum is None:
            self.sum = torch.zeros(self.k_first_tokens, hs_cpu.shape[-1], dtype=torch.float64)
        self.sum += hs_cpu.sum(dim=0).to(torch.float64)
        self.count += hs_cpu.shape[0]
        return None

    def _post(self, module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        hs = hs[:, : self.k_first_tokens, :]
        hs_cpu = hs.detach().to("cpu", copy=False)
        if self.sum is None:
            self.sum = torch.zeros(self.k_first_tokens, hs_cpu.shape[-1], dtype=torch.float64)
        self.sum += hs_cpu.sum(dim=0).to(torch.float64)
        self.count += hs_cpu.shape[0]
        return None

    def mean(self) -> torch.Tensor:
        if self.sum is None or self.count == 0:
            raise RuntimeError("No activations captured; run a forward pass first.")
        return (self.sum / self.count).to(torch.float32)


class NormCapture:
    """Capture L2 norms across tokens to estimate expected activation norm at a layer.

    Stores norms on CPU as a running list to compute a median later.
    """

    def __init__(self, k_first_tokens: int = 8, skip_first: int = 3):
        self.k = k_first_tokens
        self.skip = skip_first
        self.norms: list[float] = []

    def _pre(self, module, inputs):
        (hidden_states, *_) = inputs
        hs = hidden_states[:, : self.k, :]
        if self.k > self.skip:
            hs = hs[:, self.skip :, :]
        n = torch.linalg.vector_norm(hs, ord=2, dim=-1).detach().to("cpu")
        self.norms.extend(n.flatten().tolist())
        return None


def estimate_expected_norm(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    texts: list[str],
    layer_idx: int,
    k_first_tokens: int = 8,
    skip_first: int = 3,
    batch_size: int = 32,
):
    layers = resolve_decoder_layers(model)
    layer = layers[layer_idx]
    cap = NormCapture(k_first_tokens=k_first_tokens, skip_first=skip_first)
    handle = layer.register_forward_pre_hook(cap._pre, with_kwargs=False)
    try:
        for i in tqdm(range(0, len(texts), batch_size), desc="estim-norm", leave=False):
            batch = texts[i : i + batch_size]
            if not batch:
                continue
            toks = tok(batch, return_tensors="pt", truncation=True, max_length=k_first_tokens, padding=True)
            toks = {k: v.to(model.device) for k, v in toks.items()}
            with torch.inference_mode():
                _ = model(**toks)
        assert len(cap.norms) > 0, "Expected norm capture collected no values."
        med = float(torch.tensor(cap.norms).median().item())
        assert med > 0, "Expected norm median is non-positive."
        return med
    finally:
        handle.remove()


@contextlib.contextmanager
def capture_residual_means(model: PreTrainedModel, layer_idx: int, k_first_tokens: int = 5, mode: str = "pre"):
    layers = resolve_decoder_layers(model)
    layer = layers[layer_idx]
    cap = ResidCapture(layer_idx=layer_idx, k_first_tokens=k_first_tokens, mode=mode)
    if mode == "pre":
        handle = layer.register_forward_pre_hook(cap._pre, with_kwargs=False)
    else:
        handle = layer.register_forward_hook(cap._post, with_kwargs=False)
    try:
        yield cap
    finally:
        handle.remove()


def identity_patch_at_position(model: PreTrainedModel, layer_idx: int, pos_idx: int, vec: torch.Tensor, alpha: float = 1.0):
    """Return a forward_pre_hook that overwrites hidden_states[:, pos_idx, :] with alpha*vec at the target layer."""
    vec = vec.detach()

    def _pre(module, inputs):
        (hidden_states, *rest) = inputs
        # Broadcast vec to dtype/device
        v = vec.to(dtype=hidden_states.dtype, device=hidden_states.device)
        hidden_states[:, pos_idx, :] = alpha * v
        return (hidden_states, *rest)

    layers = resolve_decoder_layers(model)
    layer = layers[layer_idx]
    handle = layer.register_forward_pre_hook(_pre, with_kwargs=False)
    return handle


def add_delta_all_positions(model: PreTrainedModel, layer_idx: int, vec: torch.Tensor, alpha: float = 1.0):
    """Return a forward_pre_hook that adds alpha*vec to all positions at the target layer."""
    vec = vec.detach()

    def _pre(module, inputs):
        (hidden_states, *rest) = inputs
        v = vec.to(dtype=hidden_states.dtype, device=hidden_states.device)
        hidden_states.add_(alpha * v)
        return (hidden_states, *rest)

    layers = resolve_decoder_layers(model)
    layer = layers[layer_idx]
    handle = layer.register_forward_pre_hook(_pre, with_kwargs=False)
    return handle

def add_delta_answer_schedule(
    model: PreTrainedModel,
    layer_idx: int,
    vec: torch.Tensor,
    alpha: float = 1.0,
    first_n: int = 32,
    decay_tau: float | None = None,
):
    """Add alpha-scaled vec only during the first N generated tokens.

    Implementation detail: we treat the first forward pass as the prefill (prompt)
    and do not inject there. On subsequent generation steps, we add the delta
    only to the last token position and increment a step counter t. If decay_tau
    is provided, scale alpha_t = alpha * exp(-t/decay_tau).
    """
    vec = vec.detach()
    state = {"seen_prefill": False, "t": 0}

    def _pre(module, inputs):
        (hidden_states, *rest) = inputs
        if not state["seen_prefill"]:
            state["seen_prefill"] = True
            return (hidden_states, *rest)
        t = state["t"]
        if t >= max(0, int(first_n)):
            return (hidden_states, *rest)
        a = alpha
        if decay_tau is not None and decay_tau > 0:
            import math
            a = float(alpha) * math.exp(-float(t) / float(decay_tau))
        v = vec.to(dtype=hidden_states.dtype, device=hidden_states.device)
        hidden_states[:, -1, :] = hidden_states[:, -1, :] + a * v
        state["t"] = t + 1
        return (hidden_states, *rest)

    layers = resolve_decoder_layers(model)
    layer = layers[layer_idx]
    handle = layer.register_forward_pre_hook(_pre, with_kwargs=False)
    return handle
