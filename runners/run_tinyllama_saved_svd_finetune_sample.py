"""Smoke runner: save TinyLlama SVD-LLM compression, then fine-tune it.

This public sample demonstrates the full post-compression workflow:

1. Compress TinyLlama 1.1B with ``svd_llm`` at ratio ``0.8``.
2. Save the compressed low-rank factors as ``svd_factors.pt``.
3. Fine-tune the saved compressed state with two adaptation strategies:
   ``svd_llm_sequential`` and ``pissa`` at rank ``8``.

The default settings are intentionally lightweight so the script can be used as
an end-to-end smoke test. Increase sample counts or training steps when you
want more meaningful quality measurements.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMPRESS_SCRIPT = PROJECT_ROOT / "scripts" / "compress_model.py"
FINETUNE_SCRIPT = PROJECT_ROOT / "scripts" / "post_compression_finetune.py"

DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "sample_runs" / "tinyllama_svd_llm_ratio_0p8_saved_finetune"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compress TinyLlama with SVD-LLM, save svd_factors.pt, and run "
            "post-compression fine-tuning with svd_llm_sequential and PiSSA."
        )
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable used for the child CLI runs.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--compression-method", choices=["svd_llm", "sc_svd_llm", "ep_svd_llm"], default="svd_llm")
    parser.add_argument("--compression-ratio", type=float, default=0.8)
    parser.add_argument("--alpha", type=float, default=0.5, help="Used only when --compression-method ep_svd_llm.")
    parser.add_argument("--regularization", type=float, default=1e-6)
    parser.add_argument("--train-rank", type=int, default=8)
    parser.add_argument("--strategies", default="svd_llm_sequential,pissa")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr-scheduler", choices=["none", "linear", "cosine"], default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--sequential-steps-v", type=int, default=4)
    parser.add_argument("--eval-interval", type=int, default=4)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calib-samples", type=int, default=16)
    parser.add_argument("--calib-seq-length", type=int, default=256)
    parser.add_argument("--train-samples", type=int, default=16)
    parser.add_argument("--train-seq-length", type=int, default=256)
    parser.add_argument("--val-samples", type=int, default=8)
    parser.add_argument("--val-seq-length", type=int, default=256)
    parser.add_argument("--ppl-max-length", type=int, default=256)
    parser.add_argument("--ppl-stride", type=int, default=256)
    parser.add_argument("--ppl-max-samples", type=int, default=8)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--force", action="store_true", help="Re-run compression even if svd_factors.pt already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser.parse_args()


def build_compress_command(args: argparse.Namespace, python_path: Path, compressed_dir: Path) -> list[str]:
    cmd = [
        str(python_path),
        str(COMPRESS_SCRIPT),
        "--model",
        args.model,
        "--method",
        args.compression_method,
        "--compression-ratio",
        str(args.compression_ratio),
        "--regularization",
        str(args.regularization),
        "--n-samples",
        str(args.calib_samples),
        "--calib-seq-length",
        str(args.calib_seq_length),
        "--save-format",
        "svd",
        "--output",
        str(compressed_dir),
    ]
    if args.compression_method == "ep_svd_llm":
        cmd.extend(["--alpha", str(args.alpha)])
    return cmd


def build_finetune_command(
    args: argparse.Namespace,
    python_path: Path,
    compressed_svd_model: Path,
    finetune_dir: Path,
) -> list[str]:
    cmd = [
        str(python_path),
        str(FINETUNE_SCRIPT),
        "--model",
        args.model,
        "--compression-method",
        args.compression_method,
        "--compression-ratio",
        str(args.compression_ratio),
        "--regularization",
        str(args.regularization),
        "--train-rank",
        str(args.train_rank),
        "--strategies",
        args.strategies,
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--lr-scheduler",
        args.lr_scheduler,
        "--warmup-ratio",
        str(args.warmup_ratio),
        "--max-grad-norm",
        str(args.max_grad_norm),
        "--steps",
        str(args.steps),
        "--sequential-steps-v",
        str(args.sequential_steps_v),
        "--eval-interval",
        str(args.eval_interval),
        "--early-stopping-patience",
        str(args.early_stopping_patience),
        "--early-stopping-min-delta",
        str(args.early_stopping_min_delta),
        "--seed",
        str(args.seed),
        "--calib-samples",
        str(args.calib_samples),
        "--calib-seq-length",
        str(args.calib_seq_length),
        "--train-samples",
        str(args.train_samples),
        "--train-seq-length",
        str(args.train_seq_length),
        "--val-samples",
        str(args.val_samples),
        "--val-seq-length",
        str(args.val_seq_length),
        "--ppl-max-length",
        str(args.ppl_max_length),
        "--ppl-stride",
        str(args.ppl_stride),
        "--ppl-max-samples",
        str(args.ppl_max_samples),
        "--compressed-svd-model",
        str(compressed_svd_model),
        "--results-dir",
        str(finetune_dir),
        "--recompute-compressed-ppl",
    ]
    if args.compression_method == "ep_svd_llm":
        cmd.extend(["--alpha", str(args.alpha)])
    return cmd


def run_command(cmd: list[str], title: str, dry_run: bool) -> None:
    print("=" * 100)
    print(title)
    print("=" * 100)
    print(" ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


def main() -> None:
    args = parse_args()
    python_path = Path(args.python).expanduser().resolve()
    if not python_path.exists():
        raise FileNotFoundError(f"Python executable not found: {python_path}")

    output_root = Path(args.output_root).expanduser().resolve()
    compressed_dir = output_root / "compressed_model"
    finetune_results_dir = output_root / "finetune"
    compressed_svd_model = compressed_dir / "svd_factors.pt"

    output_root.mkdir(parents=True, exist_ok=True)
    finetune_results_dir.mkdir(parents=True, exist_ok=True)

    print("TinyLlama saved-SVD fine-tune sample")
    print(f"python: {python_path}")
    print(f"model: {args.model}")
    print(f"compression_method: {args.compression_method}")
    print(f"compression_ratio: {args.compression_ratio}")
    print(f"strategies: {args.strategies}")
    print(f"output_root: {output_root}")

    if args.force or not compressed_svd_model.exists():
        compressed_dir.mkdir(parents=True, exist_ok=True)
        run_command(
            build_compress_command(args, python_path, compressed_dir),
            title="[1/2] compress model and save svd_factors.pt",
            dry_run=args.dry_run,
        )
    else:
        print(f"skip [1/2] existing compressed state: {compressed_svd_model}")

    if not args.dry_run and not compressed_svd_model.exists():
        raise FileNotFoundError(f"Compressed SVD state was not created: {compressed_svd_model}")

    run_command(
        build_finetune_command(args, python_path, compressed_svd_model, finetune_results_dir),
        title="[2/2] fine-tune from saved compressed state",
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
