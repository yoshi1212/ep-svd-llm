"""
Unit tests for _compress_core, Tikhonov regularization, and compressor equivalence.

All tests run on CPU with small dummy matrices — no GPU or model download required.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from ep_svd_llm.core.base_compressor import BaseCompressor, CompressionResult
from ep_svd_llm.core.svd_llm import SVDLLMCompressor
from ep_svd_llm.core.ep_svd_llm import EPSVDLLMCompressor

DEVICE = "cpu"
DTYPE = torch.float32


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return torch.Generator().manual_seed(42)

@pytest.fixture
def weight(rng):
    """Random weight (d_out=32, d_in=16)."""
    return torch.randn(32, 16, generator=rng, dtype=DTYPE)

@pytest.fixture
def activations(rng):
    """Random activations (d_in=16, N=64)."""
    return torch.randn(16, 64, generator=rng, dtype=DTYPE)

@pytest.fixture
def activations_orig(rng):
    """Slightly different original-model activations to simulate accumulated error."""
    return torch.randn(16, 64, generator=rng, dtype=DTYPE) * 1.05


# ---------------------------------------------------------------------------
# 1. Tikhonov regularization: no NaN/Inf even with singular Hessian
# ---------------------------------------------------------------------------

class TestTikhonovStability:
    def test_zero_hessian(self):
        """A zero Hessian should not produce NaN/Inf."""
        comp = SVDLLMCompressor(device=DEVICE, dtype=DTYPE)
        H = torch.zeros(16, 16)
        P, P_inv = comp.compute_whitening_matrix(H)
        assert torch.isfinite(P).all()
        assert torch.isfinite(P_inv).all()

    def test_near_singular_hessian(self):
        """Hessian with some near-zero eigenvalues."""
        comp = SVDLLMCompressor(device=DEVICE, dtype=DTYPE)
        H = torch.eye(16) * 1e-12
        H[0, 0] = 1.0  # one large eigenvalue
        P, P_inv = comp.compute_whitening_matrix(H)
        assert torch.isfinite(P).all()
        assert torch.isfinite(P_inv).all()

    def test_compress_core_singular(self, weight):
        """_compress_core should not crash on a near-singular Hessian."""
        comp = SVDLLMCompressor(device=DEVICE, dtype=DTYPE)
        H = torch.eye(16) * 1e-14
        result = comp._compress_core(weight, H, target_rank=4, layer_name="test")
        assert torch.isfinite(result.W_u).all()
        assert torch.isfinite(result.W_v).all()


# ---------------------------------------------------------------------------
# 2. SVDLLMCompressor: compress_layer ≡ compress_layer_from_hessian
# ---------------------------------------------------------------------------

class TestSVDLLMEquivalence:
    def test_two_paths_match(self, weight, activations):
        comp = SVDLLMCompressor(device=DEVICE, dtype=DTYPE)
        rank = 4

        # Path 1: from raw activations
        res1 = comp.compress_layer(weight, activations, rank, "path1")

        # Path 2: manually compute Hessian, then from_hessian
        N = activations.shape[1]
        H = (activations @ activations.T) / N
        res2 = comp.compress_layer_from_hessian(weight, H, rank, "path2")

        # They should be identical (same code path now)
        torch.testing.assert_close(res1.W_u, res2.W_u, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(res1.W_v, res2.W_v, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# 3. EPSVDLLMCompressor: compress_layer ≡ compress_layer_from_stats
# ---------------------------------------------------------------------------

class TestEPSVDLLMEquivalence:
    def test_two_paths_match(self, weight, activations, activations_orig):
        comp = EPSVDLLMCompressor(alpha=0.5, device=DEVICE, dtype=DTYPE)
        rank = 4

        # Path 1: from raw activations
        res1 = comp.compress_layer(
            weight, activations, rank, "path1",
            activations_orig=activations_orig,
        )

        # Path 2: pre-compute stats, then from_stats
        X = activations.to(DTYPE)
        X_orig = activations_orig.to(DTYPE)
        N = X.shape[1]
        H_hat = (X @ X.T) / N
        delta = X_orig - X
        delta_X_hat_T = (delta @ X.T) / N

        res2 = comp.compress_layer_from_stats(
            weight, H_hat, delta_X_hat_T, rank, "path2",
        )

        torch.testing.assert_close(res1.W_u, res2.W_u, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(res1.W_v, res2.W_v, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# 3.5 EP whitening-input construction: old vs optimised form
# ---------------------------------------------------------------------------

class TestEPWhitenedInputEquivalence:
    def test_direct_wp_construction_matches_explicit_correction(self, rng):
        d_out, d_in = 12, 8
        alpha = 0.5

        W = torch.randn(d_out, d_in, generator=rng, dtype=DTYPE)
        X_hat = torch.randn(d_in, 32, generator=rng, dtype=DTYPE)
        X_orig = X_hat + 0.2 * torch.randn(d_in, 32, generator=rng, dtype=DTYPE)

        H_hat = (X_hat @ X_hat.T) / X_hat.shape[1]
        delta_X_hat_T = ((X_orig - X_hat) @ X_hat.T) / X_hat.shape[1]

        comp = EPSVDLLMCompressor(alpha=alpha, device=DEVICE, dtype=DTYPE)

        # Reference path: build the explicit EP correction first, then whiten.
        H_reg = H_hat + comp.regularization * torch.eye(d_in, dtype=DTYPE)
        U_h, S_h, _ = torch.linalg.svd(H_reg, full_matrices=False)
        eps = 1e-6
        sqrt_S_h = torch.sqrt(S_h.clamp(min=0.0) + 1e-10)
        inv_sqrt_S_h = 1.0 / (sqrt_S_h + eps)
        P = U_h * sqrt_S_h.unsqueeze(0)
        P_inv = (U_h * inv_sqrt_S_h.unsqueeze(0)).T

        H_hat_pinv = (U_h * (1.0 / (S_h + eps)).unsqueeze(0)) @ U_h.T
        correction = W @ delta_X_hat_T @ H_hat_pinv
        WP_explicit = (W + alpha * correction) @ P

        # Optimised path from the article:
        # W_star P = W P + alpha * W delta X_hat^T P^{-T}
        WP_direct = (W @ P) + alpha * (W @ delta_X_hat_T @ P_inv.T)

        torch.testing.assert_close(WP_direct, WP_explicit, rtol=1e-5, atol=1e-6)

    def test_compress_core_matches_explicit_reference_path(self, rng):
        d_out, d_in = 12, 8
        rank = 3
        alpha = 0.5

        W = torch.randn(d_out, d_in, generator=rng, dtype=DTYPE)
        X_hat = torch.randn(d_in, 40, generator=rng, dtype=DTYPE)
        X_orig = X_hat + 0.2 * torch.randn(d_in, 40, generator=rng, dtype=DTYPE)

        H_hat = (X_hat @ X_hat.T) / X_hat.shape[1]
        delta_X_hat_T = ((X_orig - X_hat) @ X_hat.T) / X_hat.shape[1]

        comp = EPSVDLLMCompressor(alpha=alpha, device=DEVICE, dtype=DTYPE)
        result = comp.compress_layer_from_stats(
            weight=W,
            hessian_compressed=H_hat,
            delta_hessian=delta_X_hat_T,
            target_rank=rank,
            layer_name="ep_ref",
        )

        # Reference path: explicitly build W_star, then whiten, then factorise.
        H_reg = H_hat + comp.regularization * torch.eye(d_in, dtype=DTYPE)
        U_h, S_h, _ = torch.linalg.svd(H_reg, full_matrices=False)
        eps = 1e-6
        sqrt_S_h = torch.sqrt(S_h.clamp(min=0.0) + 1e-10)
        inv_sqrt_S_h = 1.0 / (sqrt_S_h + eps)
        P = U_h * sqrt_S_h.unsqueeze(0)
        P_inv = (U_h * inv_sqrt_S_h.unsqueeze(0)).T

        H_hat_pinv = (U_h * (1.0 / (S_h + eps)).unsqueeze(0)) @ U_h.T
        W_star = W + alpha * (W @ delta_X_hat_T @ H_hat_pinv)
        WP_ref = W_star @ P

        U_k, Sigma_k, Vh_k = torch.linalg.svd(WP_ref, full_matrices=False)
        U_k = U_k[:, :rank]
        Sigma_k = Sigma_k[:rank]
        Vh_k = Vh_k[:rank, :]

        sqrt_sigma = torch.sqrt(Sigma_k)
        W_u_ref = U_k * sqrt_sigma.unsqueeze(0)
        W_v_ref = (sqrt_sigma.unsqueeze(1) * Vh_k) @ P_inv

        W_ref = W_u_ref @ W_v_ref
        W_result = result.W_u @ result.W_v

        torch.testing.assert_close(W_result, W_ref, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# 4. alpha=0 degenerates to SC-SVD-LLM
# ---------------------------------------------------------------------------

class TestAlphaZeroDegenerates:
    def test_ep_alpha0_equals_svd(self, weight, activations):
        svd_comp = SVDLLMCompressor(device=DEVICE, dtype=DTYPE)
        ep_comp = EPSVDLLMCompressor(alpha=0.0, device=DEVICE, dtype=DTYPE)
        rank = 4

        res_svd = svd_comp.compress_layer(weight, activations, rank, "svd")
        res_ep = ep_comp.compress_layer(
            weight, activations, rank, "ep0",
            activations_orig=torch.randn_like(activations),  # should be ignored
        )

        torch.testing.assert_close(res_svd.W_u, res_ep.W_u, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(res_svd.W_v, res_ep.W_v, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# 5. CompressionResult sanity
# ---------------------------------------------------------------------------

class TestCompressionResult:
    def test_shapes_and_ratio(self, weight, activations):
        comp = SVDLLMCompressor(device=DEVICE, dtype=DTYPE)
        rank = 4
        res = comp.compress_layer(weight, activations, rank, "test")

        assert res.W_u.shape == (32, 4)
        assert res.W_v.shape == (4, 16)
        assert res.target_rank == 4
        assert 0 < res.compression_ratio < 1
        assert res.compressed_params == 32 * 4 + 4 * 16
