"""
Unit tests for low-rank fine-tuning after compression.
"""

import sys
import tempfile
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from ep_svd_llm.core.ep_svd_llm import EPSVDLLMCompressor
from ep_svd_llm.core.svd_llm import SVDLLMCompressor
from ep_svd_llm.models.loader import (
    FactoredTrainLowRankLinear,
    LowRankLinear,
    SequentialLowRankUpdateLinear,
    advance_sequential_low_rank_finetuning,
    convert_low_rank_linear,
    load_low_rank_state,
    load_svd_state,
    merge_to_low_rank_layers,
    prepare_low_rank_finetuning,
    save_low_rank_state,
    save_svd_state,
)


DTYPE = torch.float32


@pytest.fixture
def rng():
    return torch.Generator().manual_seed(123)


class TestFactoredTrainLowRankLinear:
    def test_initial_output_matches_low_rank_linear(self, rng):
        W_u = torch.randn(12, 5, generator=rng, dtype=DTYPE)
        W_v = torch.randn(5, 7, generator=rng, dtype=DTYPE)
        source = LowRankLinear(W_u, W_v)
        converted = convert_low_rank_linear(source, train_rank=3)
        x = torch.randn(4, 7, generator=rng, dtype=DTYPE)
        torch.testing.assert_close(converted(x), source(x), rtol=1e-5, atol=1e-5)

    def test_prepare_low_rank_finetuning_freezes_non_adapter_params(self, rng):
        class ToyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.prefix = nn.Linear(6, 6)
                self.adapter = LowRankLinear(
                    torch.randn(6, 4, generator=rng, dtype=DTYPE),
                    torch.randn(4, 6, generator=rng, dtype=DTYPE),
                )

        model = ToyModel()
        converted = prepare_low_rank_finetuning(model, train_rank=2)
        assert converted == ["adapter"]
        assert isinstance(model.adapter, FactoredTrainLowRankLinear)
        trainable = {name for name, param in model.named_parameters() if param.requires_grad}
        assert trainable == {"adapter.U_train", "adapter.V_train"}

    def test_merge_to_low_rank_layers(self, rng):
        class ToyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.adapter = LowRankLinear(
                    torch.randn(6, 4, generator=rng, dtype=DTYPE),
                    torch.randn(4, 6, generator=rng, dtype=DTYPE),
                )

        model = ToyModel()
        x = torch.randn(3, 6, generator=rng, dtype=DTYPE)
        orig_out = model.adapter(x)
        prepare_low_rank_finetuning(model, train_rank=2)
        merge_to_low_rank_layers(model)
        assert isinstance(model.adapter, LowRankLinear)
        torch.testing.assert_close(model.adapter(x), orig_out, rtol=1e-5, atol=1e-5)


class TestSequentialLowRankUpdateLinear:
    def test_prepare_sequential_finetuning_exposes_only_current_stage_params(self, rng):
        class ToyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.adapter = LowRankLinear(
                    torch.randn(6, 4, generator=rng, dtype=DTYPE),
                    torch.randn(4, 6, generator=rng, dtype=DTYPE),
                )

        model = ToyModel()
        converted = prepare_low_rank_finetuning(model, train_rank=2, strategy="svd_llm_sequential", sequential_stage="u")
        assert converted == ["adapter"]
        assert isinstance(model.adapter, SequentialLowRankUpdateLinear)
        trainable = {name for name, param in model.named_parameters() if param.requires_grad}
        assert trainable == {"adapter.A_u", "adapter.B_u"}

    def test_advance_stage_preserves_output(self, rng):
        class ToyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.adapter = LowRankLinear(
                    torch.randn(6, 4, generator=rng, dtype=DTYPE),
                    torch.randn(4, 6, generator=rng, dtype=DTYPE),
                )

        model = ToyModel()
        prepare_low_rank_finetuning(model, train_rank=2, strategy="svd_llm_sequential", sequential_stage="u")
        x = torch.randn(3, 6, generator=rng, dtype=DTYPE)
        before = model.adapter(x)
        advance_sequential_low_rank_finetuning(model, next_stage="v")
        torch.testing.assert_close(model.adapter(x), before, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize(("compressor_cls", "compress_kwargs", "strategy"), [(SVDLLMCompressor, {}, "pissa"), (SVDLLMCompressor, {}, "svd_llm_sequential"), (EPSVDLLMCompressor, {"alpha": 0.5}, "pissa"), (EPSVDLLMCompressor, {"alpha": 0.5}, "svd_llm_sequential")])
    def test_conversion_works_for_compressed_layers(self, rng, compressor_cls, compress_kwargs, strategy):
        weight = torch.randn(14, 8, generator=rng, dtype=DTYPE)
        activations = torch.randn(8, 32, generator=rng, dtype=DTYPE)
        compressor = compressor_cls(device="cpu", dtype=DTYPE, **compress_kwargs)
        if isinstance(compressor, EPSVDLLMCompressor):
            result = compressor.compress_layer(weight=weight, activations=activations, target_rank=4, layer_name="ep", activations_orig=activations + 0.1 * torch.randn(8, 32, generator=rng, dtype=DTYPE))
        else:
            result = compressor.compress_layer(weight=weight, activations=activations, target_rank=4, layer_name="svd")
        source = LowRankLinear(result.W_u, result.W_v)
        converted = convert_low_rank_linear(source, train_rank=2, strategy=strategy)
        x = torch.randn(5, 8, generator=rng, dtype=DTYPE)
        torch.testing.assert_close(converted(x), source(x), rtol=1e-5, atol=1e-5)


class TestSaveLoadState:
    def test_low_rank_roundtrip(self, rng):
        class ToyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = LowRankLinear(torch.randn(6, 3, generator=rng, dtype=DTYPE), torch.randn(3, 8, generator=rng, dtype=DTYPE), bias=torch.randn(6, generator=rng, dtype=DTYPE))

        model = ToyModel()
        x = torch.randn(2, 8, generator=rng, dtype=DTYPE)
        out = model.layer1(x)
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = str(Path(tmpdir) / "factors.pt")
            save_low_rank_state(model, save_path)

            class FreshModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.layer1 = nn.Linear(8, 6)

            fresh = FreshModel()
            load_low_rank_state(fresh, save_path)
            torch.testing.assert_close(fresh.layer1(x), out, rtol=1e-5, atol=1e-5)

    def test_svd_roundtrip(self, rng):
        class ToyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = LowRankLinear(torch.randn(6, 3, generator=rng, dtype=DTYPE), torch.randn(3, 8, generator=rng, dtype=DTYPE), bias=torch.randn(6, generator=rng, dtype=DTYPE))

        model = ToyModel()
        x = torch.randn(2, 8, generator=rng, dtype=DTYPE)
        out = model.layer1(x)
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = str(Path(tmpdir) / "svd_factors.pt")
            save_svd_state(model, save_path)

            class FreshModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.layer1 = nn.Linear(8, 6)

            fresh = FreshModel()
            load_svd_state(fresh, save_path)
            torch.testing.assert_close(fresh.layer1(x), out, rtol=1e-5, atol=1e-5)
