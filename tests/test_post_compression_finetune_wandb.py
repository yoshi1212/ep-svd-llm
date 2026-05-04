import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ep_svd_llm.finetune import helpers as MODULE


def build_args(**overrides):
    payload = {
        "model": "Qwen/Qwen2.5-1.5B",
        "compression_method": "ep_svd_llm",
        "compression_ratio": 0.8,
        "seed": 43,
        "wandb_project": "ep-svd-llm-finetune",
        "wandb_run_name_prefix": "smoke",
        "train_rank": 16,
        "compressed_model": None,
        "compressed_svd_model": None,
        "full_finetune": False,
        "strategies": None,
        "steps": 10,
        "eval_interval": 5,
        "early_stopping_patience": 1,
        "warmup_ratio": 0.0,
        "warmup_steps": None,
        "sequential_steps_v": None,
    }
    payload.update(overrides)
    return argparse.Namespace(**payload)


def test_resolve_wandb_project_uses_model_and_ratio():
    args = build_args()
    assert MODULE.resolve_wandb_project(args) == "ep-svd-llm-qwen2-5-1-5b-r0p8"


def test_build_wandb_run_name_starts_with_method_and_strategy():
    args = build_args()
    output_dir = Path("results/finetune/ep_job/Qwen2.5-1.5B_20260411_120000")
    run_name = MODULE.build_wandb_run_name(args, "svd_llm_sequential", output_dir)
    assert run_name == "ep_svd_llm-sequential-smoke-seed43-ep_job-20260411_120000"


def test_validate_args_rejects_full_finetune_with_strategies():
    args = build_args(full_finetune=True, strategies="pissa")
    try:
        MODULE.validate_args(args)
    except ValueError as exc:
        assert "--full-finetune and --strategies" in str(exc)
    else:
        raise AssertionError("validate_args should reject full-finetune with strategies")
