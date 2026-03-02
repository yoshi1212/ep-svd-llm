"""Tutorial runner: compare SVD-LLM vs EP-SVD-LLM on TinyLlama.

This script is intended for public usage as a simple, reproducible example.
It runs compression + perplexity evaluation for two methods:
  - svd_llm
  - ep_svd_llm
across a small set of compression ratios.

Typical usage (from project root):
    python runners/run_tinyllama_tutorial.py

Notes:
- Results are saved via ``--save-results`` and can be compared afterwards.
- ``--no-save`` is enabled to avoid storing full compressed model weights.
- EP-SVD-LLM uses ``alpha=0.5`` by default.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMPRESS_SCRIPT = PROJECT_ROOT / "scripts" / "compress_model.py"


def build_command(method: str, ratio: float) -> list[str]:
    """Build one compression/evaluation command for a given method and ratio."""
    model = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    output_dir = f"models/tutorial_tinyllama_{method}_{ratio}"

    cmd = [
        sys.executable,
        str(COMPRESS_SCRIPT),
        "--model",
        model,
        "--method",
        method,
        "--compression-ratio",
        str(ratio),
        "--n-samples",
        "256",
        "--calib-seq-length",
        "2048",
        "--eval",
        "--eval-max-length",
        "2048",
        "--eval-stride",
        "2048",
        "--no-save",
        "--save-results",
        "--output",
        output_dir,
    ]

    if method == "ep_svd_llm":
        cmd.extend(["--alpha", "0.5"])

    return cmd


def run_one(method: str, ratio: float) -> None:
    """Execute one command and fail fast if the run exits with an error."""
    cmd = build_command(method=method, ratio=ratio)

    print("\n" + "=" * 72)
    print(f"Running TinyLlama tutorial: method={method}, ratio={ratio}")
    print("=" * 72)
    print("Command:")
    print(" ".join(cmd) + "\n")

    try:
        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
    except subprocess.CalledProcessError as exc:
        print(f"\nRun failed: method={method}, ratio={ratio}, exit={exc.returncode}")
        raise SystemExit(exc.returncode) from exc


def main() -> None:
    """Run a minimal comparison grid suitable for quick tutorial validation."""
    methods = ["svd_llm", "ep_svd_llm"]
    ratios = [0.2, 0.4, 0.6, 0.8]

    print("#" * 72)
    print("TinyLlama tutorial: SVD-LLM vs EP-SVD-LLM")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Methods: {methods}")
    print(f"Ratios: {ratios}")
    print("#" * 72)

    for method in methods:
        for ratio in ratios:
            run_one(method=method, ratio=ratio)
            time.sleep(1)

    print("\nAll tutorial runs completed successfully.")
    print("Check the latest JSON files under results/ for PPL comparison.")


if __name__ == "__main__":
    main()
