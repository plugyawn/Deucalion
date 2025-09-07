from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_exp_dir(name: str) -> Path:
    base = Path("assets/experiments/outputs")
    base.mkdir(parents=True, exist_ok=True)
    t = timestamp()
    out = base / f"{t}_{name}"
    out.mkdir(parents=True, exist_ok=False)
    (out / "artifacts").mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    (out / "code").mkdir(parents=True, exist_ok=True)
    return out


def snapshot_code(dst_dir: Path):
    """Copy minimal codebase into dst_dir/code for provenance.

    Excludes common heavy/ephemeral paths.
    """
    dst = Path(dst_dir) / "code"
    dst.mkdir(parents=True, exist_ok=True)
    root = Path.cwd()
    include_files = [
        "pyproject.toml",
        "README.md",
        "dpo_adl",
        "prompts",
        "experiments",
    ]
    exclude = shutil.ignore_patterns(
        ".venv", "artifacts", "assets", "*.pt", "*.bin", "*.npz", "*.npy", "__pycache__", ".git", ".gitignore",
    )
    for rel in include_files:
        src = root / rel
        if not src.exists():
            continue
        dst_path = dst / rel
        if src.is_dir():
            shutil.copytree(src, dst_path, ignore=exclude)
        else:
            shutil.copy2(src, dst_path)

