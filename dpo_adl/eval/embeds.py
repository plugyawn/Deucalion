from __future__ import annotations

from typing import List, Optional

import torch
from transformers import AutoModel, AutoTokenizer


class Embedder:
    """Embedding provider (HF encoder-only). No fallbacks.

    Requirements:
      - provider must be 'hf'
      - model_name must be a valid HF encoder model id
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: Optional[str] = None, provider: Optional[str] = None):
        self.provider = (provider or '').lower()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        if self.provider not in {"hf", "huggingface"}:
            raise ValueError("Embedder requires provider='hf' and a valid HF encoder model id. No fallbacks.")
        if model_name is None or str(model_name).lower() in {"none", "off", "skip"}:
            raise ValueError("Embeddings disabled. Pass a valid HF encoder id to enable.")
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.enc = AutoModel.from_pretrained(model_name).to(self.device)
        self.enc.eval()

    @torch.inference_mode()
    def encode(self, texts: List[str]) -> torch.Tensor:
        # HF encoder path: mean-pool last hidden state with attention mask
        toks = self.tok(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        out = self.enc(**toks)
        hs = out.last_hidden_state  # [B, T, H]
        mask = toks.get("attention_mask", torch.ones(hs.shape[:2], device=hs.device, dtype=hs.dtype))
        mask = mask.unsqueeze(-1)
        summed = (hs * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1e-6)
        embs = summed / denom
        # L2 normalize
        embs = embs / embs.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-6)
        return embs

    @staticmethod
    def cosine_sim_matrix(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return (A @ B.T).clamp(-1, 1)
