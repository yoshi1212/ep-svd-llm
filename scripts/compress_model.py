#!/usr/bin/env python3
"""
compress_model.py — CLI script to compress an LLM with SVD-LLM, SC-SVD-LLM or EP-SVD-LLM.

Supports any HuggingFace causal LM architecture (LLaMA, Mistral, GPT-NeoX /
pythia, GPT-2, Falcon, Gemma, …) by auto-detecting Transformer decoder blocks.

The compression follows the correct sequential order:
  for each Decoder Block (from layer 0 to the last):
    1. Collect per-layer Hessians using the *current compressed model*'s output
       as the next block's input  (H_l uses X̂_l, not X_orig_l)
    2. Compress each linear layer in the block
    3. Forward the *compressed block* to obtain X̂_{l+1} for the next iteration

Usage examples:
    # pythia-70m (SC-SVD-LLM)
    python scripts/compress_model.py \\
        --model EleutherAI/pythia-70m \\
        --method sc_svd_llm \\
        --compression-ratio 0.2 \\
        --output models/pythia70m_sc_svd_llm_0.2

    # TinyLlama (EP-SVD-LLM)
    python scripts/compress_model.py \\
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
        --method ep_svd_llm \\
        --compression-ratio 0.2 \\
        --alpha 0.5 \\
        --output models/tinyllama_ep_svd_llm_0.2
"""

import argparse
import sys
import time
import json
import torch
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ep_svd_llm.models.loader import (
    load_model_and_tokenizer,
    merge_low_rank_layers,
)
from ep_svd_llm.core.svd_llm import SVDLLMCompressor
from ep_svd_llm.core.ep_svd_llm import EPSVDLLMCompressor
from ep_svd_llm.core.pipeline import SequentialCompressionPipeline
from ep_svd_llm.data.calibration import prepare_calibration_data





# ---------------------------------------------------------------------------
# Main compression function
# ---------------------------------------------------------------------------

def compress_model(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Keep SVD-related math in float32 for numerical stability.
    comp_dtype = torch.float32
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Load model (and original copy for SVD-LLM / EP-SVD-LLM)
    # ------------------------------------------------------------------
    model, tokenizer = load_model_and_tokenizer(
        args.model, torch_dtype="float16", device=device,
    )
    model.eval()
    model.config.use_cache = False

    orig_model = None
    if args.method in ("svd_llm", "ep_svd_llm"):
        print("Loading original reference model for original-model Hessians …")
        orig_model, _ = load_model_and_tokenizer(
            args.model, torch_dtype="float16", device="cpu",  # Load to CPU to save VRAM
        )
        orig_model.eval()
        orig_model.config.use_cache = False

    target_modules = None
    if args.target_modules:
        target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
        print(f"Using explicit target modules: {target_modules}")

    # ------------------------------------------------------------------
    # Build compressor
    # ------------------------------------------------------------------
    # - svd_llm / sc_svd_llm share the same compressor class.
    # - ep_svd_llm adds a delta-based correction term controlled by alpha.
    if args.method in ("svd_llm", "sc_svd_llm"):
        compressor = SVDLLMCompressor(
            regularization=args.regularization, device=device, dtype=comp_dtype,
        )
    elif args.method == "ep_svd_llm":
        compressor = EPSVDLLMCompressor(
            alpha=args.alpha, regularization=args.regularization,
            device=device, dtype=comp_dtype,
        )
    else:
        print(f"ERROR: Unknown method '{args.method}'.")
        sys.exit(1)

    print(f"\nCompressing with {args.method} | ratio={args.compression_ratio}"
          + (f" | alpha={args.alpha}" if args.method == "ep_svd_llm" else ""))

    # ------------------------------------------------------------------
    # Prepare calibration data from dataset
    # ------------------------------------------------------------------
    print("Preparing calibration data …")
    calib_samples = prepare_calibration_data(
        tokenizer,
        dataset_name=args.calib_dataset,
        dataset_config=args.calib_dataset_config,
        split=args.calib_split,
        num_samples=args.n_samples,
        seq_length=args.calib_seq_length,
    )

    # Orchestration is delegated to the pipeline layer so this script remains
    # a thin entrypoint (argument parsing + dependency wiring + I/O).
    pipeline = SequentialCompressionPipeline(device=device, dtype=comp_dtype)
    run_result = pipeline.run(
        model=model,
        calibration_samples=calib_samples,
        method=args.method,
        compressor=compressor,
        compression_ratio=args.compression_ratio,
        target_modules=target_modules,
        orig_model=orig_model,
    )

    overall_ratio = run_result.overall_ratio
    peak_vram_compress_gb = run_result.peak_vram_compress_gb
    print(f"\nCompression complete.")
    print(f"  Layers skipped : {run_result.skipped_layers}")
    print(f"  Overall compression ratio: {overall_ratio:.3f}")
    print(f"  Peak GPU memory (compression): {peak_vram_compress_gb:.2f} GB")

    # ------------------------------------------------------------------
    # Merge LowRankLinear → nn.Linear and save
    # ------------------------------------------------------------------
    print("\nMerging low-rank factors for HuggingFace-compatible saving …")
    merge_low_rank_layers(model)

    if not args.no_save:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        print(f"Model saved to: {output_dir}")
    else:
        print("\nSkipping model saving as --no-save was specified.")

    # ------------------------------------------------------------------
    # Optional: PPL evaluation
    # ------------------------------------------------------------------
    if args.eval:
        from ep_svd_llm.utils.metrics import compute_perplexity

        print("\nEvaluating perplexity …")
        model.to(device).eval()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        ppl = compute_perplexity(
            model, tokenizer,
            dataset_name=args.eval_dataset,
            dataset_config=args.eval_dataset_config,
            split=args.eval_split,
            max_length=args.eval_max_length,
            stride=args.eval_stride,
            device=device,
        )
        t_eval = time.time() - t0
        peak_vram_eval_gb = (
            torch.cuda.max_memory_allocated() / 1024**3
            if torch.cuda.is_available() else 0.0
        )
        print(f"Perplexity: {ppl:.4f}  ({t_eval:.1f}s)")
        print(f"Peak GPU memory (evaluation): {peak_vram_eval_gb:.2f} GB")

        if args.save_results:
            results = {
                "timestamp": datetime.now().isoformat(),
                "model": args.model,
                "method": args.method,
                "compression_ratio": args.compression_ratio,
                "alpha": args.alpha if args.method == "ep_svd_llm" else None,
                "n_samples": args.n_samples,
                "regularization": args.regularization,
                "overall_compression_ratio": round(overall_ratio, 4),
                "ppl_compressed": round(ppl, 4),
                "time_eval_sec": round(t_eval, 2),
                "eval_dataset": args.eval_dataset,
                "eval_dataset_config": args.eval_dataset_config,
                "eval_split": args.eval_split,
                "eval_max_length": args.eval_max_length,
                "eval_stride": args.eval_stride,
                "peak_vram_compress_gb": round(peak_vram_compress_gb, 2),
                "peak_vram_eval_gb": round(peak_vram_eval_gb, 2),
            }
            results_dir = Path(args.results_dir)
            results_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = results_dir / f"ppl_{args.method}_{ts}.json"
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {out_path}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compress an LLM with SVD-LLM, SC-SVD-LLM or EP-SVD-LLM (sequential block loop)"
    )
    parser.add_argument("--model", required=True,
                        help="HuggingFace model name or local path")
    parser.add_argument("--method", choices=["svd_llm", "sc_svd_llm", "ep_svd_llm"],
                        default="ep_svd_llm")
    parser.add_argument("--compression-ratio", type=float, default=0.2,
                        help="Target compression ratio per layer (default: 0.2)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="EP correction strength for EP-SVD-LLM (default: 0.5)")
    parser.add_argument("--regularization", type=float, default=1e-6,
                        help="Hessian regularization (default: 1e-6)")
    parser.add_argument("--n-samples", type=int, default=128,
                        help="Number of calibration samples (default: 128)")
    parser.add_argument("--calib-dataset", default="wikitext",
                        help="Calibration dataset name (default: wikitext)")
    parser.add_argument("--calib-dataset-config", default="wikitext-2-raw-v1",
                        help="Calibration dataset config (default: wikitext-2-raw-v1)")
    parser.add_argument("--calib-split", default="train",
                        help="Calibration split (default: train)")
    parser.add_argument("--calib-seq-length", type=int, default=2048,
                        help="Calibration sequence length (default: 2048)")
    parser.add_argument("--target-modules", default=None,
                        help="Comma-separated module name substrings to compress "
                             "(omit to auto-detect all nn.Linear layers)")
    parser.add_argument("--output", default="models/compressed",
                        help="Directory to save the compressed model")
    # --- Evaluation options ---
    parser.add_argument("--eval", action="store_true",
                        help="Run PPL evaluation after compression")
    parser.add_argument("--eval-dataset", default="wikitext",
                        help="Evaluation dataset name (default: wikitext)")
    parser.add_argument("--eval-dataset-config", default="wikitext-2-raw-v1",
                        help="Dataset config (default: wikitext-2-raw-v1)")
    parser.add_argument("--eval-split", default="test",
                        help="Evaluation split (default: test)")
    parser.add_argument("--eval-max-length", type=int, default=2048,
                        help="Context length for PPL evaluation (default: 2048)")
    parser.add_argument("--eval-stride", type=int, default=512,
                        help="Sliding window stride (default: 512)")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not save the compressed model to disk (useful for eval-only)")
    parser.add_argument("--save-results", action="store_true",
                        help="Save results to JSON")
    parser.add_argument("--results-dir", default="results/",
                        help="Directory for JSON results (default: results/)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compress_model(args)
