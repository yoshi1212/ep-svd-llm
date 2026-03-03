#!/usr/bin/env python3
"""
evaluate_model.py — CLI script to evaluate an LLM with perplexity on a dataset.

Usage examples:
    python scripts/evaluate_model.py \\
        --model-path models/tinyllama_svd_llm_0.2 \\
        --dataset wikitext2

    python scripts/evaluate_model.py \\
        --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
        --dataset wikitext2 \\
        --max-length 1024 \\
        --max-samples 5
"""

import argparse
import sys
import time
import json
import torch
from datetime import datetime
from pathlib import Path

# Allow running from the repository root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from ep_svd_llm.utils.metrics import compute_perplexity, print_gpu_memory
from ep_svd_llm.models.loader import (
    LowRankLinear,  # noqa: F401 - keep import for custom module registration
    load_model_and_tokenizer,
)


DATASET_CONFIGS = {
    "wikitext2": ("wikitext", "wikitext-2-raw-v1"),
    "ptb": ("ptb_text_only", "penn_treebank"),
    "c4": ("allenai/c4", "en"),
}


def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ------------------------------------------------------------------ #
    # Load model and tokenizer
    # ------------------------------------------------------------------ #
    model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        dtype=torch.float16,
        device=device,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded. Parameters: {n_params:,}")
    print_gpu_memory()

    # ------------------------------------------------------------------ #
    # Dataset configuration
    # ------------------------------------------------------------------ #
    dataset_key = args.dataset.lower()
    if dataset_key not in DATASET_CONFIGS:
        print(
            f"ERROR: Unknown dataset '{args.dataset}'. "
            f"Choose from: {', '.join(DATASET_CONFIGS)}"
        )
        sys.exit(1)

    dataset_name, dataset_config = DATASET_CONFIGS[dataset_key]

    # ------------------------------------------------------------------ #
    # Perplexity evaluation
    # ------------------------------------------------------------------ #
    print(f"\nEvaluating perplexity on {args.dataset} ({dataset_config}) ...")

    t0 = time.time()
    ppl = compute_perplexity(
        model=model,
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split="test",
        max_length=args.max_length,
        stride=args.max_length // 2,
        device=device,
        max_samples=args.max_samples,
    )
    t_eval = time.time() - t0

    print(f"\n{'='*40}")
    print(f"  Model      : {args.model_path}")
    print(f"  Dataset    : {args.dataset}")
    print(f"  Perplexity : {ppl:.4f}")
    print(f"{'='*40}")

    if args.save_results:
        peak_vram_eval_gb = (
            torch.cuda.max_memory_allocated() / 1024**3
            if torch.cuda.is_available() else 0.0
        )
        results = {
            "timestamp": datetime.now().isoformat(),
            "model": args.model_path,
            "method": "baseline",
            "eval_dataset": args.dataset,
            "eval_max_length": args.max_length,
            "ppl_compressed": round(ppl, 4),
            "time_eval_sec": round(t_eval, 2),
            "peak_vram_eval_gb": round(peak_vram_eval_gb, 2),
        }
        results_dir = Path(args.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = results_dir / f"ppl_baseline_{ts}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Baseline results saved to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a (possibly compressed) LLM by perplexity"
    )
    parser.add_argument(
        "--model-path", required=True,
        help="HuggingFace model name or local directory"
    )
    parser.add_argument(
        "--dataset", default="wikitext2",
        choices=list(DATASET_CONFIGS.keys()),
        help="Evaluation dataset (default: wikitext2)"
    )
    parser.add_argument(
        "--max-length", type=int, default=2048,
        help="Context window length (default: 2048)"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit evaluation tokens (useful for quick smoke tests)"
    )
    parser.add_argument("--save-results", action="store_true",
                        help="Save results to JSON")
    parser.add_argument("--results-dir", default="results/",
                        help="Directory for JSON results (default: results/)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
