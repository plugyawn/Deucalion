#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dpo_adl.train.datasets import sample_texts_from_spec
from dpo_adl.diff.delta_builder import build_delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--dataset", default="openai/webgpt_comparisons")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default="assets/experiments/logs/webgpt_refref.json")
    ap.add_argument("--which", default="chosen", choices=["chosen", "rejected"], help="Which texts to sample from preference dataset")
    args = ap.parse_args()

    texts = sample_texts_from_spec(args.dataset, args.split, n_pairs=args.n, seed=0, which=args.which)
    delta = build_delta(args.model, args.model, texts, k=args.k, batch_size=args.batch)
    per_j = [float(delta[j].abs().max().item()) for j in range(delta.shape[0])]
    out = {
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "n": args.n,
        "k": args.k,
        "shape": list(delta.shape),
        "max_abs": float(delta.abs().max().item()),
        "per_j_max_abs": per_j,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
