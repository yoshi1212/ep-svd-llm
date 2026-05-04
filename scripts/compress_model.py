#!/usr/bin/env python3
"""
CLI script to compress an LLM with SVD-LLM, SC-SVD-LLM or EP-SVD-LLM.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from ep_svd_llm.core.ep_svd_llm import EPSVDLLMCompressor
from ep_svd_llm.core.pipeline import SequentialCompressionPipeline
from ep_svd_llm.core.svd_llm import SVDLLMCompressor
from ep_svd_llm.data.calibration import prepare_calibration_data
from ep_svd_llm.models.loader import (
    load_model_and_tokenizer,
    merge_low_rank_layers,
    save_low_rank_state,
    save_svd_state,
)


def compress_model(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    comp_dtype = torch.float32
    print(f"Using device: {device}")

    model, tokenizer = load_model_and_tokenizer(args.model, dtype=torch.float16, device=device)
    model.eval()
    model.config.use_cache = False

    orig_model = None
    if args.method in ("svd_llm", "ep_svd_llm"):
        orig_model, _ = load_model_and_tokenizer(args.model, dtype=torch.float16, device="cpu")
        orig_model.eval()
        orig_model.config.use_cache = False

    target_modules = None
    if args.target_modules:
        target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]

    if args.method in ("svd_llm", "sc_svd_llm"):
        compressor = SVDLLMCompressor(regularization=args.regularization, device=device, dtype=comp_dtype)
    elif args.method == "ep_svd_llm":
        compressor = EPSVDLLMCompressor(alpha=args.alpha, regularization=args.regularization, device=device, dtype=comp_dtype)
    else:
        print(f"ERROR: Unknown method '{args.method}'.")
        sys.exit(1)

    calib_samples = prepare_calibration_data(
        tokenizer,
        dataset_name=args.calib_dataset,
        dataset_config=args.calib_dataset_config,
        split=args.calib_split,
        num_samples=args.n_samples,
        seq_length=args.calib_seq_length,
    )

    run_result = SequentialCompressionPipeline(device=device, dtype=comp_dtype).run(
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

    if not args.no_save:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.save_format == "low_rank":
            save_low_rank_state(model, str(output_dir / "low_rank_factors.pt"))
            tokenizer.save_pretrained(str(output_dir))
        elif args.save_format == "svd":
            save_svd_state(model, str(output_dir / "svd_factors.pt"))
            tokenizer.save_pretrained(str(output_dir))
        else:
            merge_low_rank_layers(model)
            model.save_pretrained(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))
        print(f"Model saved to: {output_dir}")
    else:
        print("\nSkipping model saving as --no-save was specified.")

    if args.eval:
        from ep_svd_llm.utils.metrics import compute_perplexity

        model.to(device).eval()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        ppl = compute_perplexity(
            model,
            tokenizer,
            dataset_name=args.eval_dataset,
            dataset_config=args.eval_dataset_config,
            split=args.eval_split,
            max_length=args.eval_max_length,
            stride=args.eval_stride,
            device=device,
        )
        t_eval = time.time() - t0
        peak_vram_eval_gb = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
        print(f"Perplexity: {ppl:.4f}  ({t_eval:.1f}s)")

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
            out_path = results_dir / f"ppl_{args.method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Compress an LLM with SVD-LLM, SC-SVD-LLM or EP-SVD-LLM.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", choices=["svd_llm", "sc_svd_llm", "ep_svd_llm"], default="ep_svd_llm")
    parser.add_argument("--compression-ratio", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--regularization", type=float, default=1e-6)
    parser.add_argument("--n-samples", type=int, default=128)
    parser.add_argument("--calib-dataset", default="wikitext")
    parser.add_argument("--calib-dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--calib-split", default="train")
    parser.add_argument("--calib-seq-length", type=int, default=2048)
    parser.add_argument("--target-modules", default=None)
    parser.add_argument("--output", default="models/compressed")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--eval-dataset", default="wikitext")
    parser.add_argument("--eval-dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--eval-max-length", type=int, default=2048)
    parser.add_argument("--eval-stride", type=int, default=512)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--save-format", choices=["hf", "low_rank", "svd"], default="hf")
    parser.add_argument("--save-results", action="store_true")
    parser.add_argument("--results-dir", default="results/")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compress_model(args)
