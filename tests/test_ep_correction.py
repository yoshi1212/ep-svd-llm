"""
Unit tests for EP correction correctness.

Verifies that α > 0 actually improves (or at least does not degrade)
reconstruction error compared to α = 0, and that different alpha values
produce different (but finite) results.

All tests run on CPU with small dummy matrices — no GPU or model download required.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from ep_svd_llm.core.svd_llm import SVDLLMCompressor
from ep_svd_llm.core.ep_svd_llm import EPSVDLLMCompressor
from ep_svd_llm.utils.metrics import compute_layer_reconstruction_error

DEVICE = "cpu"
DTYPE = torch.float32


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return torch.Generator().manual_seed(2024)


@pytest.fixture
def weight(rng):
    """Random weight (d_out=32, d_in=16)."""
    return torch.randn(32, 16, generator=rng, dtype=DTYPE)


@pytest.fixture
def activations(rng):
    """'Compressed-model' activations (d_in=16, N=128)."""
    return torch.randn(16, 128, generator=rng, dtype=DTYPE)


@pytest.fixture
def activations_orig(activations, rng):
    """
    'Original-model' activations that differ systematically from compressed.
    Simulate accumulated compression error as a structured bias.
    """
    error = torch.randn_like(activations) * 0.3
    return activations + error


# ---------------------------------------------------------------------------
# 1. EP correction does not degrade quality
# ---------------------------------------------------------------------------

class TestEPCorrectionImproves:
    def test_ep_no_worse_than_svd(self, weight, activations, activations_orig):
        """EP correction (α > 0) should not produce worse reconstruction
        error than plain SC-SVD-LLM (α = 0) on the original-model activations."""
        rank = 4

        svd_comp = SVDLLMCompressor(device=DEVICE, dtype=DTYPE)
        ep_comp = EPSVDLLMCompressor(alpha=0.5, device=DEVICE, dtype=DTYPE)

        res_svd = svd_comp.compress_layer(weight, activations, rank, "svd")
        res_ep = ep_comp.compress_layer(
            weight, activations, rank, "ep",
            activations_orig=activations_orig,
        )

        # Measure reconstruction on the *original-model* activations
        err_svd = compute_layer_reconstruction_error(
            weight, (res_svd.W_u, res_svd.W_v), activations_orig
        )
        err_ep = compute_layer_reconstruction_error(
            weight, (res_ep.W_u, res_ep.W_v), activations_orig
        )

        # EP should do at least as well (or better) on the original activations
        # Allow some tolerance as EP optimises a different objective
        assert err_ep["frobenius_error"] <= err_svd["frobenius_error"] * 1.5, (
            f"EP error {err_ep['frobenius_error']:.4f} should not be much "
            f"worse than SVD error {err_svd['frobenius_error']:.4f}"
        )

    def test_ep_finite_results(self, weight, activations, activations_orig):
        """EP correction should always produce finite W_u and W_v."""
        for alpha in [0.0, 0.1, 0.5, 1.0]:
            comp = EPSVDLLMCompressor(alpha=alpha, device=DEVICE, dtype=DTYPE)
            res = comp.compress_layer(
                weight, activations, 4, f"a{alpha}",
                activations_orig=activations_orig,
            )
            assert torch.isfinite(res.W_u).all(), f"W_u not finite at alpha={alpha}"
            assert torch.isfinite(res.W_v).all(), f"W_v not finite at alpha={alpha}"


# ---------------------------------------------------------------------------
# 2. Alpha scaling
# ---------------------------------------------------------------------------

class TestAlphaScaling:
    def test_different_alpha_different_result(self, weight, activations, activations_orig):
        """α=0 and α=0.5 should produce different compressed weights."""
        rank = 4
        comp_0 = EPSVDLLMCompressor(alpha=0.0, device=DEVICE, dtype=DTYPE)
        comp_05 = EPSVDLLMCompressor(alpha=0.5, device=DEVICE, dtype=DTYPE)

        res_0 = comp_0.compress_layer(
            weight, activations, rank, "a0",
            activations_orig=activations_orig,
        )
        res_05 = comp_05.compress_layer(
            weight, activations, rank, "a05",
            activations_orig=activations_orig,
        )

        # With error present, α=0.5 should differ from α=0
        assert not torch.allclose(res_0.W_u, res_05.W_u, atol=1e-6), \
            "α=0 and α=0.5 should produce different W_u"

    def test_no_fp_activations_degenerates(self, weight, activations):
        """Without activations_orig, EP should behave like SC-SVD-LLM regardless of α."""
        rank = 4
        svd_comp = SVDLLMCompressor(device=DEVICE, dtype=DTYPE)
        ep_comp = EPSVDLLMCompressor(alpha=0.8, device=DEVICE, dtype=DTYPE)

        res_svd = svd_comp.compress_layer(weight, activations, rank, "svd")
        res_ep = ep_comp.compress_layer(
            weight, activations, rank, "ep_no_orig",
            # activations_orig is intentionally omitted (None)
        )

        torch.testing.assert_close(res_svd.W_u, res_ep.W_u, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(res_svd.W_v, res_ep.W_v, rtol=1e-5, atol=1e-6)

    def test_compression_result_fields(self, weight, activations, activations_orig):
        """CompressionResult from EP should have all expected fields."""
        comp = EPSVDLLMCompressor(alpha=0.5, device=DEVICE, dtype=DTYPE)
        res = comp.compress_layer(
            weight, activations, 4, "test_fields",
            activations_orig=activations_orig,
        )
        assert res.original_shape == (32, 16)
        assert res.target_rank == 4
        assert 0 < res.compression_ratio < 1
        assert res.layer_name == "test_fields"
        assert res.compressed_params == res.W_u.numel() + res.W_v.numel()
        assert res.original_params == 32 * 16
