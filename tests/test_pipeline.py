"""Unit tests for SequentialCompressionPipeline.

These tests are lightweight and run on CPU without downloading real models.
"""

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from ep_svd_llm.core.base_compressor import CompressionResult
from ep_svd_llm.core.pipeline import SequentialCompressionPipeline
from ep_svd_llm.models.loader import LowRankLinear


DTYPE = torch.float32


class _FakeAccumulator:
    def __init__(self, hessian: torch.Tensor):
        self._hessian = hessian

    def get_hessian(self, normalize: bool = True) -> torch.Tensor:
        return self._hessian


class _FakeCompressor:
    def compute_target_rank(self, d_out: int, d_in: int, compression_ratio: float) -> int:
        return 1

    def compress_layer_from_hessian(
        self,
        weight: torch.Tensor,
        hessian: torch.Tensor,
        target_rank: int,
        layer_name: str = "",
    ) -> CompressionResult:
        d_out, d_in = weight.shape
        W_u = torch.ones(d_out, target_rank, dtype=weight.dtype)
        W_v = torch.ones(target_rank, d_in, dtype=weight.dtype)
        return CompressionResult(
            W_u=W_u,
            W_v=W_v,
            original_shape=(d_out, d_in),
            target_rank=target_rank,
            compression_ratio=1 - (W_u.numel() + W_v.numel()) / (d_out * d_in),
            layer_name=layer_name,
        )


class _DummyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, hidden_states, **kwargs):
        return self.proj(hidden_states)


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_DummyBlock()])

    def forward(self, x):
        return self.blocks[0](x)


class _SequentialDummyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4)
        self.down_proj = nn.Linear(4, 4)

    def forward(self, hidden_states, **kwargs):
        hidden_states = self.q_proj(hidden_states)
        return self.down_proj(hidden_states)


class _SequentialDummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_SequentialDummyBlock()])

    def forward(self, x):
        return self.blocks[0](x)


class TestSequentialCompressionPipeline:
    def test_find_full_name(self):
        model = _DummyModel()
        pipeline = SequentialCompressionPipeline(device="cpu", dtype=DTYPE)

        found = pipeline._find_full_name(model, model.blocks[0].proj)
        assert found == "blocks.0.proj"

        not_found = pipeline._find_full_name(model, nn.Linear(4, 4))
        assert not_found is None

    def test_compress_layers_from_hessian_replaces_layer(self):
        model = _DummyModel()
        pipeline = SequentialCompressionPipeline(device="cpu", dtype=DTYPE)
        compressor = _FakeCompressor()

        layer_map = {"proj": model.blocks[0].proj}
        h_accs = {"proj": _FakeAccumulator(torch.eye(4, dtype=DTYPE))}

        total_original, total_compressed, skipped = pipeline._compress_layers_from_hessian(
            model=model,
            layer_map=layer_map,
            h_accs=h_accs,
            compressor=compressor,
            compression_ratio=0.6,
            total_original=0,
            total_compressed=0,
            skipped_layers=0,
        )

        assert isinstance(model.blocks[0].proj, LowRankLinear)
        assert total_original == 16
        assert total_compressed == 8
        assert skipped == 0

    def test_run_sc_svd_llm_with_monkeypatch(self, monkeypatch):
        model = _DummyModel()
        pipeline = SequentialCompressionPipeline(device="cpu", dtype=DTYPE)
        compressor = _FakeCompressor()

        import ep_svd_llm.core.pipeline as pipeline_module

        def fake_get_decoder_blocks(_model):
            return _model.blocks, [], []

        def fake_find_layers_in_block(_block, target_modules=None):
            return {"proj": _block.proj}

        def fake_accumulate_block_hessians(*args, **kwargs):
            return {"proj": _FakeAccumulator(torch.eye(4, dtype=DTYPE))}

        monkeypatch.setattr(pipeline_module, "get_decoder_blocks", fake_get_decoder_blocks)
        monkeypatch.setattr(pipeline_module, "find_layers_in_block", fake_find_layers_in_block)
        monkeypatch.setattr(
            pipeline_module,
            "accumulate_block_hessians",
            fake_accumulate_block_hessians,
        )

        monkeypatch.setattr(
            pipeline,
            "_collect_block_inputs",
            lambda *args, **kwargs: [(torch.zeros(1, 4, dtype=DTYPE), {})],
        )
        monkeypatch.setattr(
            pipeline,
            "_forward_block",
            lambda _block, block_inputs: block_inputs,
        )

        result = pipeline.run(
            model=model,
            calibration_samples=[torch.zeros(1, 4, dtype=DTYPE)],
            method="sc_svd_llm",
            compressor=compressor,
            compression_ratio=0.6,
            target_modules=None,
            orig_model=None,
        )

        assert result.total_original == 16
        assert result.total_compressed == 8
        assert result.skipped_layers == 0
        assert result.overall_ratio == pytest.approx(0.5)
        assert isinstance(model.blocks[0].proj, LowRankLinear)

    def test_run_sc_svd_llm_uses_sequential_groups(self, monkeypatch):
        model = _SequentialDummyModel()
        pipeline = SequentialCompressionPipeline(device="cpu", dtype=DTYPE)
        compressor = _FakeCompressor()

        import ep_svd_llm.core.pipeline as pipeline_module

        accumulate_calls = []

        def fake_get_decoder_blocks(_model):
            return _model.blocks, [], []

        def fake_find_layers_in_block(_block, target_modules=None):
            layers = {
                "self_attn.q_proj": _block.q_proj,
                "mlp.down_proj": _block.down_proj,
            }
            if target_modules is None:
                return layers
            return {
                name: layer
                for name, layer in layers.items()
                if any(target in name for target in target_modules)
            }

        def fake_get_sequential_groups(_layer_names):
            return [["self_attn.q_proj"], ["mlp.down_proj"]]

        def fake_accumulate_block_hessians(*args, **kwargs):
            targets = tuple(kwargs["target_modules"])
            accumulate_calls.append(targets)
            return {
                target: _FakeAccumulator(torch.eye(4, dtype=DTYPE))
                for target in targets
            }

        monkeypatch.setattr(pipeline_module, "get_decoder_blocks", fake_get_decoder_blocks)
        monkeypatch.setattr(pipeline_module, "find_layers_in_block", fake_find_layers_in_block)
        monkeypatch.setattr(pipeline_module, "get_sequential_groups", fake_get_sequential_groups)
        monkeypatch.setattr(
            pipeline_module,
            "accumulate_block_hessians",
            fake_accumulate_block_hessians,
        )

        monkeypatch.setattr(
            pipeline,
            "_collect_block_inputs",
            lambda *args, **kwargs: [(torch.zeros(1, 4, dtype=DTYPE), {})],
        )
        monkeypatch.setattr(
            pipeline,
            "_forward_block",
            lambda _block, block_inputs: block_inputs,
        )

        result = pipeline.run(
            model=model,
            calibration_samples=[torch.zeros(1, 4, dtype=DTYPE)],
            method="sc_svd_llm",
            compressor=compressor,
            compression_ratio=0.6,
            target_modules=None,
            orig_model=None,
        )

        assert accumulate_calls == [
            ("self_attn.q_proj",),
            ("mlp.down_proj",),
        ]
        assert isinstance(model.blocks[0].q_proj, LowRankLinear)
        assert isinstance(model.blocks[0].down_proj, LowRankLinear)
        assert result.total_original == 32
        assert result.total_compressed == 16
