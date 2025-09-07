from __future__ import annotations

import torch


def project_out(v: torch.Tensor, u: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Project vector v onto the orthogonal complement of u: v - proj_u(v).

    v, u: [..., d]
    returns: [..., d]
    """
    num = (v * u).sum(dim=-1, keepdim=True)
    den = (u * u).sum(dim=-1, keepdim=True).clamp_min(eps)
    return v - (num / den) * u

