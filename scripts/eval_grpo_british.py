#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch

from dpo_adl.backends.hf_hooks import load_model_and_tokenizer
from dpo_adl.eval.steering import generate_text_with_completion
from dpo_adl.train.grpo_british_prompts import generate_british_prompts_dataset
from dpo_adl.train.grpo_rewards import reward_british_spelling, reward_brevity


def main():
    ap = argparse.ArgumentParser(description="Evaluate a GRPO British model on synthetic British prompts; save rollouts + metrics.")
    ap.add_argument("model_dir", help="Path to trained model directory")
    ap.add_argument("out_dir", help="Where to write eval outputs (rollouts + summary)")
    ap.add_argument("--n", type=int, default=500, help="Number of evaluation prompts")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    roll_path = out_dir / "rollouts.jsonl"
    sum_path = out_dir / "summary.json"

    model, tok = load_model_and_tokenizer(args.model_dir)
    ds = generate_british_prompts_dataset(n=args.n, seed=args.seed)
    prompts: List[str] = [str(r["prompt"]) for r in ds]

    completions: List[str] = []
    with open(roll_path, "w", encoding="utf-8") as f:
        for p in prompts:
            full, comp = generate_text_with_completion(model, tok, p, max_new_tokens=args.max_new_tokens, temperature=0.0)
            completions.append(comp)
            f.write(json.dumps({"prompt": p, "completion": comp}, ensure_ascii=False) + "\n")

    # Rewards/metrics
    rb = reward_british_spelling(prompts, completions)
    rbv = [float(x) for x in rb.tolist()]
    rbrev = reward_brevity(prompts, completions)
    rbrev_v = [float(x) for x in rbrev.tolist()]
    summary = {
        "n": len(prompts),
        "british_reward_mean": sum(rbv) / max(1, len(rbv)),
        "british_reward_pos_frac": sum(1 for x in rbv if x > 0) / max(1, len(rbv)),
        "brevity_reward_mean": sum(rbrev_v) / max(1, len(rbrev_v)),
    }
    sum_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({"wrote": str(roll_path), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()

