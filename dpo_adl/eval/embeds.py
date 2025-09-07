from __future__ import annotations

from typing import List

import torch
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str | None = None):
        self.model = SentenceTransformer(model_name)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)

    @torch.inference_mode()
    def encode(self, texts: List[str]) -> torch.Tensor:
        embs = self.model.encode(texts, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
        return embs

    @staticmethod
    def cosine_sim_matrix(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return (A @ B.T).clamp(-1, 1)

