"""
Lightweight tests for post-compression fine-tuning helpers.
"""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ep_svd_llm.finetune import helpers as MODULE


def make_args(**overrides):
    base = {
        "strategies": "pissa,svd_llm_sequential",
        "compressed_model": None,
        "compressed_svd_model": None,
        "full_finetune": False,
        "steps": 10,
        "eval_interval": 5,
        "early_stopping_patience": 4,
        "warmup_ratio": 0.0,
        "warmup_steps": None,
        "sequential_steps_v": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


class TestCompareFinetuneStrategiesHelpers:
    def test_parse_strategies(self):
        assert MODULE.parse_strategies("pissa,svd_llm_sequential") == ["pissa", "svd_llm_sequential"]

    def test_build_phase_plan(self):
        assert MODULE.build_phase_plan("pissa", 10, None) == [("pissa", 10)]
        assert MODULE.build_phase_plan("svd_llm_sequential", 10, None) == [("u", 5), ("v", 5)]
        assert MODULE.build_phase_plan("svd_llm_sequential", 10, 4) == [("u", 6), ("v", 4)]

    def test_write_history_csv(self, tmp_path):
        history = [{"iteration": 1, "strategy": "pissa", "phase": "pissa", "train_loss": 1.23, "eval_loss": 1.11, "best_eval_loss": 1.11, "best_iteration": 1, "improved": True, "early_stopping_active": True, "no_improve_count": 0, "lr": 1e-4}]
        output_path = tmp_path / "history.csv"
        MODULE.write_history_csv(history, output_path)
        text = output_path.read_text(encoding="utf-8")
        assert "iteration,strategy,phase,train_loss,eval_loss,best_eval_loss" in text
        assert "1,pissa,pissa,1.23,1.11,1.11,1,True,True,0,0.0001" in text

    def test_should_enable_early_stopping(self):
        assert MODULE.should_enable_early_stopping("pissa", "pissa") is True
        assert MODULE.should_enable_early_stopping("svd_llm_sequential", "u") is False
        assert MODULE.should_enable_early_stopping("svd_llm_sequential", "v") is True

    def test_compute_phase_warmup_steps(self):
        assert MODULE.compute_phase_warmup_steps(100, 0.1, None) == 10
        assert MODULE.compute_phase_warmup_steps(100, 0.1, 5) == 5
        assert MODULE.compute_phase_warmup_steps(10, 0.5, 20) == 10

    def test_validate_args_rejects_conflicting_compressed_inputs(self):
        args = make_args(compressed_model="a.pt", compressed_svd_model="b.pt")
        try:
            MODULE.validate_args(args)
        except ValueError as exc:
            assert "mutually exclusive" in str(exc)
        else:
            raise AssertionError("validate_args should reject conflicting compressed inputs")

    def test_validate_args_rejects_sequential_steps_larger_than_total_steps(self):
        args = make_args(strategies="svd_llm_sequential", steps=10, sequential_steps_v=11)
        try:
            MODULE.validate_args(args)
        except ValueError as exc:
            assert "<= --steps" in str(exc)
        else:
            raise AssertionError("validate_args should reject sequential_steps_v > steps")
