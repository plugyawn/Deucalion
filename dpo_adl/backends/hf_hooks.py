from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


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
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tok


def resolve_decoder_layers(model: PreTrainedModel):
    """Return the list-like module of decoder layers for common HF CausalLMs.

    Tries typical attributes in order.
    """
    cand_attrs = [
        ("model", "layers"),  # LLaMA/Qwen style: model.model.layers
        ("transformer", "h"),  # GPT2/Neo style: model.transformer.h
        ("gpt_neox", "layers"),  # GPT-NeoX style
    ]
    for root_name, layers_name in cand_attrs:
        root = getattr(model, root_name, None)
        if root is None:
            continue
        layers = getattr(root, layers_name, None)
        if layers is not None:
            return layers
    # Fallback: try model.layers directly
    if hasattr(model, "layers"):
        return getattr(model, "layers")
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

