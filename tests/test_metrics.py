"""
Unit tests for compute_layer_reconstruction_error.

All tests run on CPU with small dummy matrices — no GPU or model download required.
(compute_perplexity is skipped as it requires a real model & tokenizer.)
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from ep_svd_llm.utils.metrics import compute_layer_reconstruction_error

DTYPE = torch.float32


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return torch.Generator().manual_seed(77)


@pytest.fixture
def weight(rng):
    return torch.randn(16, 8, generator=rng, dtype=DTYPE)


@pytest.fixture
def activations(rng):
    return torch.randn(8, 32, generator=rng, dtype=DTYPE)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestReconstructionError:
    def test_identical_weight_zero_error(self, weight, activations):
        """When W_compressed == W_original, all errors should be zero."""
        result = compute_layer_reconstruction_error(weight, weight, activations)
        assert result["frobenius_error"] == pytest.approx(0.0, abs=1e-6)
        assert result["relative_error"] == pytest.approx(0.0, abs=1e-6)
        assert result["mean_absolute_error"] == pytest.approx(0.0, abs=1e-6)

    def test_tuple_input(self, weight, activations):
        """(W_u, W_v) tuple should work and give zero error when W_u @ W_v == W."""
        # Use SVD to factor the weight perfectly
        U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
        W_u = U * S.unsqueeze(0)  # (16, 8)
        W_v = Vh                   # (8, 8)
        result = compute_layer_reconstruction_error(weight, (W_u, W_v), activations)
        assert result["frobenius_error"] == pytest.approx(0.0, abs=1e-4)

    def test_all_keys_present(self, weight, activations, rng):
        """Result dict should contain all expected keys."""
        W_noisy = weight + torch.randn_like(weight) * 0.1
        result = compute_layer_reconstruction_error(weight, W_noisy, activations)
        assert "frobenius_error" in result
        assert "relative_error" in result
        assert "mean_absolute_error" in result

    def test_relative_error_bounded(self, weight, activations, rng):
        """Relative error for a small perturbation should be in a reasonable range."""
        W_noisy = weight + torch.randn_like(weight) * 0.01
        result = compute_layer_reconstruction_error(weight, W_noisy, activations)
        assert 0 <= result["relative_error"] <= 1.0

    def test_error_increases_with_noise(self, weight, activations, rng):
        """Larger perturbation should produce larger reconstruction error."""
        W_small_noise = weight + torch.randn(16, 8, generator=rng, dtype=DTYPE) * 0.01
        rng2 = torch.Generator().manual_seed(78)
        W_large_noise = weight + torch.randn(16, 8, generator=rng2, dtype=DTYPE) * 1.0
        err_small = compute_layer_reconstruction_error(weight, W_small_noise, activations)
        err_large = compute_layer_reconstruction_error(weight, W_large_noise, activations)
        assert err_large["frobenius_error"] > err_small["frobenius_error"]
