#!/usr/bin/env python3
"""
Run post-compression fine-tuning for compressed causal language models.

This module performs continued causal-LM training after compression. It is not
downstream task fine-tuning; the objective remains autoregressive next-token
prediction, and the purpose is to adapt or recover the compressed model.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, TYPE_CHECKING

from ep_svd_llm.finetune.helpers import (
    build_phase_plan,
    build_wandb_run_name,
    compute_phase_warmup_steps,
    infer_compression_metadata,
    parse_strategies,
    resolve_wandb_project,
    should_enable_early_stopping,
    validate_args,
    write_history_csv,
)

if TYPE_CHECKING:
    import torch


class WandbLogger:
    """Small optional W&B adapter used by CLI runs and config runners."""

    def __init__(self, args: argparse.Namespace, strategy: str, output_dir: Path) -> None:
        self.enabled = False
        self.run = None
        if not args.use_wandb:
            return
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "W&B logging was requested, but `wandb` is not installed. "
                "Install it and retry, or run without --use-wandb."
            ) from exc

        run_name = build_wandb_run_name(args, strategy, output_dir)

        config_payload = vars(args).copy()
        config_payload["strategy"] = strategy
        config_payload["output_dir"] = str(output_dir)

        self._wandb = wandb
        self.run = wandb.init(
            project=resolve_wandb_project(args),
            entity=args.wandb_entity,
            name=run_name,
            config=config_payload,
            tags=[args.compression_method, strategy, f"ratio_{args.compression_ratio}", f"rank_{args.train_rank}"],
        )
        run_url = getattr(self.run, "url", None)
        run_entity = getattr(self.run, "entity", None)
        run_project = getattr(self.run, "project", None)
        run_id = getattr(self.run, "id", None)
        print(
            "wandb_init:"
            f" entity={run_entity}"
            f" project={run_project}"
            f" run_name={run_name}"
            f" run_id={run_id}"
            f" run_url={run_url}"
        )
        self.enabled = True

    def log(self, payload: dict) -> None:
        if self.enabled:
            self._wandb.log(payload)

    def update_summary(self, payload: dict) -> None:
        if not self.enabled:
            return
        for key, value in payload.items():
            self.run.summary[key] = value

    def finish(self) -> None:
        if self.enabled:
            self.run.finish()


@dataclass
class RunSummary:
    """Metrics written to summary.json for one strategy run."""

    strategy: str
    final_train_loss: float
    final_eval_loss: float
    best_eval_loss: float
    best_iteration: int
    final_perplexity: float
    stopped_early: bool
    stop_reason: str | None
    training_time_sec: float


@dataclass
class StrategyTrainingSetup:
    """Prepared model, data, and checkpoint slots for one strategy."""

    model: torch.nn.Module
    train_samples: list[torch.Tensor]
    val_samples: list[torch.Tensor]
    phases: list[tuple[str, int]]
    total_steps: int
    best_state_dict: dict[str, torch.Tensor] | None
    best_u_state_dict: dict[str, torch.Tensor] | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="EleutherAI/pythia-70m")
    parser.add_argument("--compression-method", choices=["svd_llm", "sc_svd_llm", "ep_svd_llm"], default="svd_llm")
    parser.add_argument("--compression-ratio", type=float, default=0.6)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--regularization", type=float, default=1e-6)
    parser.add_argument("--train-rank", type=int, default=8)
    parser.add_argument("--strategies", default=None)
    parser.add_argument("--full-finetune", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr-scheduler", choices=["none", "linear", "cosine"], default="none")
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--sequential-steps-v", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calib-samples", type=int, default=64)
    parser.add_argument("--calib-seq-length", type=int, default=256)
    parser.add_argument("--train-samples", type=int, default=32)
    parser.add_argument("--train-seq-length", type=int, default=256)
    parser.add_argument("--val-samples", type=int, default=8)
    parser.add_argument("--val-seq-length", type=int, default=256)
    parser.add_argument("--calib-dataset", default="wikitext")
    parser.add_argument("--calib-dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--calib-split", default="train")
    parser.add_argument("--train-dataset", default="wikitext")
    parser.add_argument("--train-dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-dataset", default="wikitext")
    parser.add_argument("--val-dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--val-split", default="validation")
    parser.add_argument("--ppl-dataset", default="wikitext")
    parser.add_argument("--ppl-dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--ppl-split", default="test")
    parser.add_argument("--ppl-max-length", type=int, default=256)
    parser.add_argument("--ppl-stride", type=int, default=256)
    parser.add_argument("--ppl-max-samples", type=int, default=16)
    parser.add_argument("--target-modules", default=None)
    parser.add_argument("--compressed-model", default=None)
    parser.add_argument("--compressed-svd-model", default=None)
    parser.add_argument("--results-dir", default="results/post_compression_finetune")
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name-prefix", default=None)
    parser.add_argument(
        "--recompute-compressed-ppl",
        action="store_true",
        help="Recompute compressed-model PPL even when loading an existing compressed state.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_model_dtype(device: str) -> torch.dtype:
    import torch

    return torch.float16 if device == "cuda" else torch.float32


def prepare_lm_samples(tokenizer, dataset_name: str, dataset_config: str, split: str, num_samples: int, seq_length: int, seed: int) -> list[torch.Tensor]:
    from ep_svd_llm.data.calibration import prepare_calibration_data

    return prepare_calibration_data(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=split,
        num_samples=num_samples,
        seq_length=seq_length,
        seed=seed,
    )


def cycle_samples(samples: list[torch.Tensor]) -> Iterable[torch.Tensor]:
    """Repeat the finite sampled windows until the requested step count ends."""
    while True:
        for sample in samples:
            yield sample


def compute_mean_loss(model, samples: list[torch.Tensor], device: str) -> float:
    import torch

    was_training = model.training
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for sample in samples:
            batch = sample.to(device)
            outputs = model(batch, labels=batch)
            losses.append(float(outputs.loss.detach().cpu()))
    if was_training:
        model.train()
    return sum(losses) / max(len(losses), 1)


def grad_stats(trainable_params: list[torch.nn.Parameter]) -> tuple[float, bool]:
    import torch

    total_sq_norm = 0.0
    found_nonfinite = False
    for param in trainable_params:
        if param.grad is None:
            continue
        grad = param.grad.detach()
        if not torch.isfinite(grad).all():
            found_nonfinite = True
            continue
        grad_norm = float(grad.norm().detach().cpu())
        total_sq_norm += grad_norm * grad_norm
    return math.sqrt(total_sq_norm), found_nonfinite


def clone_state_dict_to_cpu(model) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def create_phase_scheduler(optimizer, scheduler_name: str, phase_steps: int, warmup_steps: int):
    if scheduler_name == "none" or phase_steps <= 0:
        return None
    from transformers import get_scheduler

    return get_scheduler(name=scheduler_name, optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=phase_steps)


def compress_model_once(args: argparse.Namespace, device: str):
    import torch

    from ep_svd_llm.core.ep_svd_llm import EPSVDLLMCompressor
    from ep_svd_llm.core.pipeline import SequentialCompressionPipeline
    from ep_svd_llm.core.svd_llm import SVDLLMCompressor
    from ep_svd_llm.data.calibration import prepare_calibration_data
    from ep_svd_llm.models.loader import load_model_and_tokenizer

    model_dtype = choose_model_dtype(device)
    comp_dtype = torch.float32
    model, tokenizer = load_model_and_tokenizer(args.model, dtype=model_dtype, device=device)
    model.eval()
    model.config.use_cache = False

    orig_model = None
    if args.compression_method in ("svd_llm", "ep_svd_llm"):
        # SVD-LLM and EP-SVD-LLM need full-precision reference activations.
        orig_model, _ = load_model_and_tokenizer(args.model, dtype=model_dtype, device="cpu")
        orig_model.eval()
        orig_model.config.use_cache = False

    compressor = (
        SVDLLMCompressor(regularization=args.regularization, device=device, dtype=comp_dtype)
        if args.compression_method in ("svd_llm", "sc_svd_llm")
        else EPSVDLLMCompressor(alpha=args.alpha, regularization=args.regularization, device=device, dtype=comp_dtype)
    )
    calib_samples = prepare_calibration_data(
        tokenizer=tokenizer,
        dataset_name=args.calib_dataset,
        dataset_config=args.calib_dataset_config,
        split=args.calib_split,
        num_samples=args.calib_samples,
        seq_length=args.calib_seq_length,
        seed=args.seed,
    )
    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()] if args.target_modules else None
    run_result = SequentialCompressionPipeline(device=device, dtype=comp_dtype).run(
        model=model,
        calibration_samples=calib_samples,
        method=args.compression_method,
        compressor=compressor,
        compression_ratio=args.compression_ratio,
        target_modules=target_modules,
        orig_model=orig_model,
    )
    model.cpu()
    if orig_model is not None:
        orig_model.cpu()
    return model, tokenizer, run_result


def prepare_strategy_training_setup(
    compressed_model,
    tokenizer,
    strategy: str,
    args: argparse.Namespace,
    device: str,
) -> StrategyTrainingSetup:
    from ep_svd_llm.models.loader import prepare_full_finetuning, prepare_low_rank_finetuning

    # Every strategy starts from the same compressed checkpoint so comparisons
    # are not affected by mutations from a previous run.
    model = copy.deepcopy(compressed_model)
    model.to(device)
    model.train()
    model.config.use_cache = False

    train_samples = prepare_lm_samples(
        tokenizer,
        args.train_dataset,
        args.train_dataset_config,
        args.train_split,
        args.train_samples,
        args.train_seq_length,
        args.seed + 1,
    )
    val_samples = prepare_lm_samples(
        tokenizer,
        args.val_dataset,
        args.val_dataset_config,
        args.val_split,
        args.val_samples,
        args.val_seq_length,
        args.seed + 2,
    )
    phases = build_phase_plan(strategy, args.steps, args.sequential_steps_v)
    total_steps = sum(step_count for _, step_count in phases)

    best_state_dict: dict[str, torch.Tensor] | None = None
    best_u_state_dict: dict[str, torch.Tensor] | None = None
    if strategy == "full":
        prepare_full_finetuning(model)
        best_state_dict = clone_state_dict_to_cpu(model)
    elif strategy == "pissa":
        # PiSSA-style adaptation trains the leading low-rank factors directly.
        prepare_low_rank_finetuning(model, args.train_rank, strategy="pissa")
        best_state_dict = clone_state_dict_to_cpu(model)
    else:
        # Sequential adaptation first updates U, then freezes that result and
        # updates V in a second phase.
        prepare_low_rank_finetuning(
            model,
            args.train_rank,
            strategy="svd_llm_sequential",
            sequential_stage="u",
        )
        best_u_state_dict = clone_state_dict_to_cpu(model)

    return StrategyTrainingSetup(
        model=model,
        train_samples=train_samples,
        val_samples=val_samples,
        phases=phases,
        total_steps=total_steps,
        best_state_dict=best_state_dict,
        best_u_state_dict=best_u_state_dict,
    )


def start_next_sequential_phase(model, best_u_state_dict: dict[str, torch.Tensor] | None) -> dict[str, torch.Tensor]:
    from ep_svd_llm.models.loader import advance_sequential_low_rank_finetuning

    if best_u_state_dict is not None:
        model.load_state_dict(best_u_state_dict)
    # Rebuild trainable parameters for the V phase from the best U checkpoint.
    advance_sequential_low_rank_finetuning(model, next_stage="v")
    return clone_state_dict_to_cpu(model)


def train_one_strategy(compressed_model, tokenizer, strategy: str, args: argparse.Namespace, output_dir: Path, device: str) -> tuple[RunSummary, list[dict]]:
    import torch
    from torch.optim import AdamW

    from ep_svd_llm.models.loader import merge_to_low_rank_layers, save_low_rank_state
    from ep_svd_llm.utils.metrics import compute_perplexity

    setup = prepare_strategy_training_setup(compressed_model, tokenizer, strategy, args, device)
    model = setup.model
    train_samples = setup.train_samples
    val_samples = setup.val_samples
    wandb_logger = WandbLogger(args=args, strategy=strategy, output_dir=output_dir)

    history: list[dict] = []
    global_step = 0
    start_time = time.time()
    phases = setup.phases
    total_steps = setup.total_steps
    sample_iter = cycle_samples(train_samples)
    best_eval_loss = math.inf
    best_iteration = 0
    best_state_dict = setup.best_state_dict
    no_improve_count = 0
    stopped_early = False
    stop_reason: str | None = None
    last_train_loss: float | None = None
    best_u_eval_loss = math.inf
    best_u_iteration = 0
    best_u_state_dict = setup.best_u_state_dict

    for phase_name, phase_steps in phases:
        if strategy == "svd_llm_sequential" and phase_name == "v":
            best_state_dict = start_next_sequential_phase(model, best_u_state_dict)
            no_improve_count = 0

        trainable_params = [param for param in model.parameters() if param.requires_grad]
        optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
        scheduler = create_phase_scheduler(optimizer, args.lr_scheduler, phase_steps, compute_phase_warmup_steps(phase_steps, args.warmup_ratio, args.warmup_steps))

        for _ in range(phase_steps):
            batch = next(sample_iter).to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch, labels=batch)
            loss = outputs.loss
            if not torch.isfinite(loss):
                stopped_early = True
                stop_reason = f"nonfinite_train_loss_at_step_{global_step + 1}"
                last_train_loss = math.nan
                print(
                    f"nonfinite_train_loss strategy={strategy} phase={phase_name} step={global_step + 1}",
                    flush=True,
                )
                break
            last_train_loss = float(loss.detach().cpu())
            loss.backward()
            total_grad_norm, grad_has_nonfinite = grad_stats(trainable_params)
            if grad_has_nonfinite or not math.isfinite(total_grad_norm):
                stopped_early = True
                stop_reason = f"nonfinite_gradients_at_step_{global_step + 1}"
                print(
                    f"nonfinite_gradients strategy={strategy} phase={phase_name} step={global_step + 1} grad_norm={total_grad_norm}",
                    flush=True,
                )
                break
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            global_step += 1
            if global_step == 1 or global_step % args.eval_interval == 0 or global_step == total_steps:
                eval_loss = compute_mean_loss(model, val_samples, device)
                if not math.isfinite(eval_loss):
                    history.append(
                        {
                            "iteration": global_step,
                            "strategy": strategy,
                            "phase": phase_name,
                            "train_loss": last_train_loss,
                            "eval_loss": eval_loss,
                            "best_eval_loss": best_u_eval_loss if strategy == "svd_llm_sequential" and phase_name == "u" else best_eval_loss,
                            "best_iteration": best_u_iteration if strategy == "svd_llm_sequential" and phase_name == "u" else best_iteration,
                            "improved": False,
                            "early_stopping_active": should_enable_early_stopping(strategy, phase_name),
                            "no_improve_count": no_improve_count,
                            "lr": float(optimizer.param_groups[0]["lr"]),
                        }
                    )
                    wandb_logger.log(history[-1])
                    stopped_early = True
                    stop_reason = f"nonfinite_eval_loss_at_step_{global_step}"
                    print(
                        f"nonfinite_eval_loss strategy={strategy} phase={phase_name} step={global_step}",
                        flush=True,
                    )
                    break
                phase_improved = False
                if strategy == "svd_llm_sequential" and phase_name == "u":
                    # U-phase quality only selects the starting point for V;
                    # final model selection happens after V has started.
                    phase_improved = (best_u_eval_loss - eval_loss) > args.early_stopping_min_delta
                    if phase_improved:
                        best_u_eval_loss = eval_loss
                        best_u_iteration = global_step
                        best_u_state_dict = clone_state_dict_to_cpu(model)
                else:
                    phase_improved = (best_eval_loss - eval_loss) > args.early_stopping_min_delta
                    if phase_improved:
                        best_eval_loss = eval_loss
                        best_iteration = global_step
                        best_state_dict = clone_state_dict_to_cpu(model)
                        no_improve_count = 0
                    elif should_enable_early_stopping(strategy, phase_name):
                        no_improve_count += 1

                history.append(
                    {
                        "iteration": global_step,
                        "strategy": strategy,
                        "phase": phase_name,
                        "train_loss": last_train_loss,
                        "eval_loss": eval_loss,
                        "best_eval_loss": best_u_eval_loss if strategy == "svd_llm_sequential" and phase_name == "u" else best_eval_loss,
                        "best_iteration": best_u_iteration if strategy == "svd_llm_sequential" and phase_name == "u" else best_iteration,
                        "improved": phase_improved,
                        "early_stopping_active": should_enable_early_stopping(strategy, phase_name),
                        "no_improve_count": no_improve_count,
                        "lr": float(optimizer.param_groups[0]["lr"]),
                    }
                )
                wandb_logger.log(history[-1])

                if args.early_stopping_patience > 0 and should_enable_early_stopping(strategy, phase_name) and no_improve_count >= args.early_stopping_patience:
                    stopped_early = True
                    stop_reason = f"early_stopping_after_{no_improve_count}_evals_without_{args.early_stopping_min_delta:.6f}_improvement"
                    break
        if stopped_early:
            break

    if strategy == "svd_llm_sequential" and best_state_dict is None:
        # If V had zero steps or stopped before evaluation, fall back to the
        # best U checkpoint instead of leaving the initial state selected.
        best_eval_loss = best_u_eval_loss
        best_iteration = best_u_iteration
        best_state_dict = best_u_state_dict

    if best_state_dict is not None:
        # Evaluate and optionally save the best validation checkpoint, not
        # necessarily the last optimizer step.
        model.load_state_dict(best_state_dict)

    selected_eval_loss = compute_mean_loss(model, val_samples, device)
    ppl = compute_perplexity(
        model=model.eval(),
        tokenizer=tokenizer,
        dataset_name=args.ppl_dataset,
        dataset_config=args.ppl_dataset_config,
        split=args.ppl_split,
        max_length=args.ppl_max_length,
        stride=args.ppl_stride,
        device=device,
        max_samples=args.ppl_max_samples,
    )

    if args.save_models:
        # Persist the adapted low-rank factors, keeping the compressed
        # parameterisation rather than expanding to full nn.Linear weights.
        merge_to_low_rank_layers(model)
        save_low_rank_state(model, str(output_dir / strategy / "low_rank_factors.pt"))

    elapsed = time.time() - start_time
    summary = RunSummary(strategy, last_train_loss if last_train_loss is not None else math.nan, selected_eval_loss, best_eval_loss, best_iteration, float(ppl), stopped_early, stop_reason, elapsed)
    wandb_logger.update_summary(asdict(summary))
    wandb_logger.finish()
    model.cpu()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary, history


def main() -> None:
    args = parse_args()
    validate_args(args)

    import torch

    from ep_svd_llm.models.loader import load_low_rank_state, load_model_and_tokenizer, load_svd_state
    from ep_svd_llm.utils.metrics import compute_perplexity

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    strategies = ["full"] if args.full_finetune else parse_strategies(args.strategies)
    output_dir = Path(args.results_dir) / f"{args.model.split('/')[-1]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.compressed_model or args.compressed_svd_model:
        # Common production path: restore a saved compressed state, then adapt.
        compressed_model, tokenizer = load_model_and_tokenizer(args.model, dtype=choose_model_dtype(device), device=device)
        compressed_model.eval()
        compressed_model.config.use_cache = False
        if args.compressed_model:
            load_low_rank_state(compressed_model, args.compressed_model, device=device)
        else:
            load_svd_state(compressed_model, args.compressed_svd_model, device=device)
        compression_result = None
    else:
        # Convenience path for small experiments: compress and adapt in one run.
        compressed_model, tokenizer, compression_result = compress_model_once(args, device)

    should_recompute_compressed_ppl = (
        compression_result is not None
        or args.recompute_compressed_ppl
    )
    compressed_ppl = None
    if should_recompute_compressed_ppl:
        compressed_ppl = compute_perplexity(
            model=compressed_model.to(device).eval(),
            tokenizer=tokenizer,
            dataset_name=args.ppl_dataset,
            dataset_config=args.ppl_dataset_config,
            split=args.ppl_split,
            max_length=args.ppl_max_length,
            stride=args.ppl_stride,
            device=device,
            max_samples=args.ppl_max_samples,
        )
    compressed_model.cpu()

    summaries: list[RunSummary] = []
    for strategy in strategies:
        summary, history = train_one_strategy(compressed_model, tokenizer, strategy, args, output_dir, device)
        summaries.append(summary)
        (output_dir / f"history_{strategy}.json").write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
        write_history_csv(history, output_dir / f"history_{strategy}.csv")

    resolved_method, resolved_ratio = infer_compression_metadata(args)
    args_payload = vars(args).copy()
    args_payload["compression_method"] = resolved_method
    args_payload["compression_ratio"] = resolved_ratio
    summary_payload = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "compression_method": resolved_method,
        "compression_ratio": resolved_ratio,
        "train_rank": args.train_rank,
        "compressed_ppl": compressed_ppl,
        "compression_run_result": asdict(compression_result) if compression_result is not None else None,
        "compressed_model_path": args.compressed_model,
        "compressed_svd_model_path": args.compressed_svd_model,
        "strategies": [asdict(item) for item in summaries],
        "args": args_payload,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
