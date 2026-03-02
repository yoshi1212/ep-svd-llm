"""
Unit tests for loader utility functions and LowRankLinear.

All tests run on CPU — no GPU or model download required.
"""

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from ep_svd_llm.models.loader import (
    LowRankLinear,
    get_sequential_groups,
    get_layer_by_name,
    set_layer_by_name,
    find_layers_in_block,
)

DTYPE = torch.float32


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return torch.Generator().manual_seed(55)


# ---------------------------------------------------------------------------
# 1. LowRankLinear
# ---------------------------------------------------------------------------

class TestLowRankLinear:
    def test_forward_shape(self, rng):
        """Output shape should be (batch, d_out)."""
        W_u = torch.randn(16, 4, generator=rng, dtype=DTYPE)
        W_v = torch.randn(4, 8, generator=rng, dtype=DTYPE)
        layer = LowRankLinear(W_u, W_v)
        x = torch.randn(3, 8, dtype=DTYPE)
        out = layer(x)
        assert out.shape == (3, 16)

    def test_forward_equivalence(self, rng):
        """LowRankLinear(x) should match x @ (W_u @ W_v).T."""
        W_u = torch.randn(16, 4, generator=rng, dtype=DTYPE)
        W_v = torch.randn(4, 8, generator=rng, dtype=DTYPE)
        layer = LowRankLinear(W_u, W_v)
        x = torch.randn(5, 8, dtype=DTYPE)
        out = layer(x)
        expected = x @ (W_u @ W_v).T
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)

    def test_with_bias(self, rng):
        """Bias should be added to the output."""
        W_u = torch.randn(16, 4, generator=rng, dtype=DTYPE)
        W_v = torch.randn(4, 8, generator=rng, dtype=DTYPE)
        bias = torch.randn(16, dtype=DTYPE)
        layer = LowRankLinear(W_u, W_v, bias=bias)
        x = torch.randn(2, 8, dtype=DTYPE)
        out = layer(x)
        expected = x @ (W_u @ W_v).T + bias
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)

    def test_without_bias(self, rng):
        """When no bias is given, layer.bias should be None."""
        W_u = torch.randn(8, 2, generator=rng, dtype=DTYPE)
        W_v = torch.randn(2, 4, generator=rng, dtype=DTYPE)
        layer = LowRankLinear(W_u, W_v)
        assert layer.bias is None

    def test_repr(self, rng):
        """__repr__ should include in, out, and rank."""
        W_u = torch.randn(16, 4, generator=rng, dtype=DTYPE)
        W_v = torch.randn(4, 8, generator=rng, dtype=DTYPE)
        layer = LowRankLinear(W_u, W_v)
        repr_str = repr(layer)
        assert "in=8" in repr_str
        assert "out=16" in repr_str
        assert "rank=4" in repr_str

    def test_attributes(self, rng):
        """in_features, out_features, rank should be set correctly."""
        W_u = torch.randn(16, 4, generator=rng, dtype=DTYPE)
        W_v = torch.randn(4, 8, generator=rng, dtype=DTYPE)
        layer = LowRankLinear(W_u, W_v)
        assert layer.in_features == 8
        assert layer.out_features == 16
        assert layer.rank == 4


# ---------------------------------------------------------------------------
# 2. get_sequential_groups
# ---------------------------------------------------------------------------

class TestGetSequentialGroups:
    def test_llama_style_layers(self):
        """LLaMA-style layer names should be grouped into 4 groups."""
        names = [
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj", "mlp.up_proj",
            "mlp.down_proj",
        ]
        groups = get_sequential_groups(names)
        assert len(groups) == 4
        # Group 1: QKV
        assert set(groups[0]) == {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"}
        # Group 2: O (contains "attn" and "o_proj")
        assert groups[1] == ["self_attn.o_proj"]
        # Group 3: gate + up
        assert set(groups[2]) == {"mlp.gate_proj", "mlp.up_proj"}
        # Group 4: down
        assert groups[3] == ["mlp.down_proj"]

    def test_empty_input(self):
        """Empty list should produce empty groups."""
        groups = get_sequential_groups([])
        assert groups == []

    def test_unrecognized_names(self):
        """Unrecognized names should go to a leftovers group."""
        names = ["custom_layer_x", "custom_layer_y"]
        groups = get_sequential_groups(names)
        assert len(groups) == 1
        assert set(groups[0]) == {"custom_layer_x", "custom_layer_y"}

    def test_mixed_recognized_and_unrecognized(self):
        """Mix of recognized and unrecognized names."""
        names = ["self_attn.q_proj", "unknown_layer"]
        groups = get_sequential_groups(names)
        # Should have at least 2 groups
        all_names = [n for g in groups for n in g]
        assert "self_attn.q_proj" in all_names
        assert "unknown_layer" in all_names

    def test_opt_style_attention_output_precedes_mlp(self):
        """OPT-style out_proj must be grouped before fc1/fc2."""
        names = [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.out_proj",
            "fc1",
            "fc2",
        ]
        groups = get_sequential_groups(names)
        assert groups == [
            ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
            ["self_attn.out_proj"],
            ["fc1"],
            ["fc2"],
        ]


# ---------------------------------------------------------------------------
# 3. get_layer_by_name / set_layer_by_name
# ---------------------------------------------------------------------------

class TestGetSetLayerByName:
    def _make_dummy_model(self):
        """Create a simple nested model for testing."""
        model = nn.Sequential(
            nn.Linear(8, 16),
            nn.Sequential(
                nn.Linear(16, 32),
                nn.Linear(32, 4),
            ),
        )
        return model

    def test_get_layer(self):
        """get_layer_by_name should retrieve the correct submodule."""
        model = self._make_dummy_model()
        layer = get_layer_by_name(model, "1.0")
        assert isinstance(layer, nn.Linear)
        assert layer.in_features == 16
        assert layer.out_features == 32

    def test_set_layer(self):
        """set_layer_by_name should replace the target submodule."""
        model = self._make_dummy_model()
        new_layer = nn.Linear(16, 64)
        set_layer_by_name(model, "1.0", new_layer)
        retrieved = get_layer_by_name(model, "1.0")
        assert retrieved is new_layer
        assert retrieved.out_features == 64

    def test_round_trip(self):
        """Getting → setting → getting should work consistently."""
        model = self._make_dummy_model()
        original = get_layer_by_name(model, "0")
        replacement = nn.Linear(8, 16, bias=False)
        set_layer_by_name(model, "0", replacement)
        assert get_layer_by_name(model, "0") is replacement
        assert get_layer_by_name(model, "0") is not original


# ---------------------------------------------------------------------------
# 4. find_layers_in_block
# ---------------------------------------------------------------------------

class TestFindLayersInBlock:
    def test_finds_linear_layers(self):
        """Should find all nn.Linear layers that match target_modules."""
        block = nn.Module()
        block.q_proj = nn.Linear(8, 8)
        block.k_proj = nn.Linear(8, 8)
        block.relu = nn.ReLU()
        layers = find_layers_in_block(block, target_modules=["q_proj", "k_proj"])
        assert "q_proj" in layers
        assert "k_proj" in layers
        assert len(layers) == 2

    def test_excludes_non_linear(self):
        """Non-Linear modules should not be included."""
        block = nn.Module()
        block.norm = nn.LayerNorm(8)
        block.proj = nn.Linear(8, 16)
        layers = find_layers_in_block(block)
        assert "proj" in layers
        assert "norm" not in layers

    def test_auto_detect_with_exclude(self):
        """With target_modules=None, lm_head should be excluded by default."""
        block = nn.Module()
        block.proj = nn.Linear(8, 16)
        block.lm_head = nn.Linear(8, 100)
        layers = find_layers_in_block(block, target_modules=None)
        assert "proj" in layers
        assert "lm_head" not in layers
