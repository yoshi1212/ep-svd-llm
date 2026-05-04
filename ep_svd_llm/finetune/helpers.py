"""Lightweight helpers for post-compression fine-tuning.

This module intentionally avoids torch / transformers imports so CLI help and
unit tests can run without loading the full training stack.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


LEGACY_SHARED_WANDB_PROJECT = "ep-svd-llm-finetune"


def slugify_metric_value(value: float) -> str:
    text = f"{value:g}"
    if "." not in text:
        return text
    whole, frac = text.split(".", 1)
    frac = frac.rstrip("0")
    if not frac:
        return whole
    return f"{whole}p{frac}"


def normalize_model_label(raw: str) -> str:
    label = Path(raw).name if "/" in raw or "\\" in raw else raw.split("/")[-1]
    cleaned = []
    for char in label.lower():
        if char.isalnum():
            cleaned.append(char)
        else:
            cleaned.append("-")
    normalized = "".join(cleaned).strip("-")
    while "--" in normalized:
        normalized = normalized.replace("--", "-")
    return normalized or "model"


def resolve_wandb_project(args: argparse.Namespace) -> str:
    if args.wandb_project and args.wandb_project != LEGACY_SHARED_WANDB_PROJECT:
        return args.wandb_project
    model_label = normalize_model_label(args.model)
    ratio_label = slugify_metric_value(args.compression_ratio)
    return f"ep-svd-llm-{model_label}-r{ratio_label}"


def resolve_run_context_label(prefix: str | None) -> str | None:
    if not prefix:
        return None
    return prefix.strip()


def resolve_strategy_label(strategy: str) -> str:
    return "sequential" if strategy == "svd_llm_sequential" else strategy


def extract_output_timestamp(output_dir: Path) -> str:
    parts = output_dir.name.split("_")
    if len(parts) >= 3:
        return "_".join(parts[-2:])
    return output_dir.name


def build_wandb_run_name(args: argparse.Namespace, strategy: str, output_dir: Path) -> str:
    run_name_parts = [
        args.compression_method,
        resolve_strategy_label(strategy),
    ]
    context_label = resolve_run_context_label(args.wandb_run_name_prefix)
    if context_label:
        run_name_parts.append(context_label)
    run_name_parts.extend(
        [
            f"seed{args.seed}",
            output_dir.parent.name,
            extract_output_timestamp(output_dir),
        ]
    )
    return "-".join(run_name_parts)


def build_phase_plan(strategy: str, steps: int, sequential_steps_v: int | None) -> list[tuple[str, int]]:
    """Return the ordered train phases for one adaptation strategy."""
    if strategy == "full":
        return [("full", steps)]
    if strategy == "pissa":
        return [("pissa", steps)]
    if strategy == "svd_llm_sequential":
        if sequential_steps_v is None:
            u_steps = (steps + 1) // 2
            v_steps = steps - u_steps
            return [("u", u_steps), ("v", v_steps)]
        return [("u", max(steps - sequential_steps_v, 0)), ("v", sequential_steps_v)]
    raise ValueError(f"Unsupported strategy: {strategy}")


def parse_strategies(raw: str) -> list[str]:
    if raw is None:
        raise ValueError("At least one strategy must be specified.")
    strategies = [item.strip() for item in raw.split(",") if item.strip()]
    valid = {"pissa", "svd_llm_sequential"}
    unknown = [item for item in strategies if item not in valid]
    if unknown:
        raise ValueError(f"Unknown strategies: {unknown}")
    if not strategies:
        raise ValueError("At least one strategy must be specified.")
    return strategies


def infer_compression_metadata(args: argparse.Namespace) -> tuple[str, float]:
    """Prefer metadata encoded in saved compression paths over CLI defaults."""
    if not (args.compressed_model or args.compressed_svd_model):
        return args.compression_method, args.compression_ratio
    compressed_path = Path(args.compressed_svd_model or args.compressed_model)
    for candidate in [compressed_path.parent.name, compressed_path.name]:
        for method in ("ep_svd_llm", "sc_svd_llm", "svd_llm"):
            marker = f"{method}_"
            if marker in candidate:
                suffix = candidate.split(marker, 1)[1]
                parts = suffix.split("_")
                if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                    return method, float(f"{parts[0]}.{parts[1]}")
                return method, args.compression_ratio
    return args.compression_method, args.compression_ratio


def should_enable_early_stopping(strategy: str, phase_name: str) -> bool:
    """Delay early stopping for sequential training until the second factor."""
    return strategy in {"pissa", "full"} or (strategy == "svd_llm_sequential" and phase_name == "v")


def compute_phase_warmup_steps(phase_steps: int, warmup_ratio: float, warmup_steps: int | None) -> int:
    if phase_steps <= 0:
        return 0
    if warmup_steps is not None:
        return min(max(warmup_steps, 0), phase_steps)
    return min(max(int(phase_steps * warmup_ratio), 0), phase_steps)


def validate_args(args: argparse.Namespace) -> None:
    if args.compressed_model and args.compressed_svd_model:
        raise ValueError("--compressed-model and --compressed-svd-model are mutually exclusive.")
    if args.full_finetune and args.strategies is not None:
        raise ValueError("--full-finetune and --strategies cannot be used together.")
    if not args.full_finetune and args.strategies is None:
        raise ValueError("--strategies is required unless --full-finetune is set.")
    strategies = [] if args.full_finetune else parse_strategies(args.strategies)
    if args.steps < 0:
        raise ValueError("--steps must be >= 0.")
    if args.eval_interval <= 0:
        raise ValueError("--eval-interval must be > 0.")
    if args.early_stopping_patience < 0:
        raise ValueError("--early-stopping-patience must be >= 0.")
    if args.warmup_ratio < 0:
        raise ValueError("--warmup-ratio must be >= 0.")
    if args.warmup_steps is not None and args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be >= 0.")
    if "svd_llm_sequential" in strategies and args.sequential_steps_v is not None:
        if args.sequential_steps_v < 0:
            raise ValueError("--sequential-steps-v must be >= 0.")
        if args.sequential_steps_v > args.steps:
            raise ValueError("--sequential-steps-v must be <= --steps.")


def write_history_csv(history: list[dict], path: Path) -> None:
    if not history:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
