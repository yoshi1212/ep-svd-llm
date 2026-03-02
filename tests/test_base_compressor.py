"""
Unit tests for BaseCompressor utility methods:
compute_hessian, compute_whitening_matrix, truncated_svd, compute_target_rank.

All tests run on CPU with small dummy matrices — no GPU or model download required.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from ep_svd_llm.core.svd_llm import SVDLLMCompressor

DEVICE = "cpu"
DTYPE = torch.float32


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def comp():
    return SVDLLMCompressor(device=DEVICE, dtype=DTYPE, regularization=1e-6)


@pytest.fixture
def rng():
    return torch.Generator().manual_seed(123)


# ---------------------------------------------------------------------------
# 1. compute_hessian
# ---------------------------------------------------------------------------

class TestComputeHessian:
    def test_normalization(self, comp, rng):
        """H should equal (X @ X.T) / N  +  reg * I."""
        X = torch.randn(8, 20, generator=rng, dtype=DTYPE)
        H = comp.compute_hessian(X)
        N = X.shape[1]
        expected = (X @ X.T) / N + comp.regularization * torch.eye(8)
        torch.testing.assert_close(H, expected, rtol=1e-5, atol=1e-6)

    def test_symmetry(self, comp, rng):
        """Hessian must be symmetric."""
        X = torch.randn(8, 30, generator=rng, dtype=DTYPE)
        H = comp.compute_hessian(X)
        torch.testing.assert_close(H, H.T, rtol=1e-6, atol=1e-7)

    def test_positive_semidefinite(self, comp, rng):
        """Hessian with regularization should be positive definite."""
        X = torch.randn(8, 30, generator=rng, dtype=DTYPE)
        H = comp.compute_hessian(X)
        eigenvalues = torch.linalg.eigvalsh(H)
        assert (eigenvalues > 0).all(), "All eigenvalues should be positive"


# ---------------------------------------------------------------------------
# 2. compute_whitening_matrix
# ---------------------------------------------------------------------------

class TestComputeWhiteningMatrix:
    def test_identity_hessian(self, comp):
        """When H ≈ I, P should be close to I (up to regularization)."""
        H = torch.eye(8)
        P, P_inv = comp.compute_whitening_matrix(H)
        # P @ P_inv should be approximately I
        product = P @ P_inv
        torch.testing.assert_close(
            product, torch.eye(8), rtol=1e-4, atol=1e-4
        )

    def test_inverse_round_trip(self, comp, rng):
        """P @ P_inv ≈ I for a random PSD Hessian."""
        X = torch.randn(8, 50, generator=rng, dtype=DTYPE)
        H = (X @ X.T) / X.shape[1] + 1e-4 * torch.eye(8)
        P, P_inv = comp.compute_whitening_matrix(H)
        product = P @ P_inv
        torch.testing.assert_close(
            product, torch.eye(8), rtol=1e-3, atol=1e-3
        )

    def test_output_finite(self, comp, rng):
        """P and P_inv should contain only finite values."""
        X = torch.randn(8, 30, generator=rng, dtype=DTYPE)
        H = (X @ X.T) / X.shape[1]
        P, P_inv = comp.compute_whitening_matrix(H)
        assert torch.isfinite(P).all()
        assert torch.isfinite(P_inv).all()


# ---------------------------------------------------------------------------
# 3. truncated_svd
# ---------------------------------------------------------------------------

class TestTruncatedSVD:
    def test_output_shapes(self, comp, rng):
        """U_k, Sigma_k, Vh_k should have correct shapes."""
        W = torch.randn(32, 16, generator=rng, dtype=DTYPE)
        k = 4
        U_k, Sigma_k, Vh_k = comp.truncated_svd(W, k)
        assert U_k.shape == (32, 4)
        assert Sigma_k.shape == (4,)
        assert Vh_k.shape == (4, 16)

    def test_singular_values_descending(self, comp, rng):
        """Singular values should be in descending order."""
        W = torch.randn(32, 16, generator=rng, dtype=DTYPE)
        _, Sigma_k, _ = comp.truncated_svd(W, 8)
        for i in range(len(Sigma_k) - 1):
            assert Sigma_k[i] >= Sigma_k[i + 1]

    def test_reconstruction_quality(self, comp, rng):
        """Truncated SVD should give a reasonable approximation."""
        W = torch.randn(16, 16, generator=rng, dtype=DTYPE)
        # Full rank reconstruction should be exact
        U_k, Sigma_k, Vh_k = comp.truncated_svd(W, 16)
        W_approx = U_k * Sigma_k.unsqueeze(0) @ Vh_k
        torch.testing.assert_close(W, W_approx, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# 4. compute_target_rank
# ---------------------------------------------------------------------------

class TestComputeTargetRank:
    def test_ratio_0_5(self, comp):
        """50% compression on (32, 16) → rank = (0.5 * 32*16) / (32+16) ≈ 5."""
        rank = comp.compute_target_rank(32, 16, 0.5)
        assert rank == 5

    def test_ratio_0_gives_max_rank(self, comp):
        """0% compression → rank = int(d_out * d_in / (d_out + d_in)).
        For (32, 16): int(1.0 * 512 / 48) = 10."""
        rank = comp.compute_target_rank(32, 16, 0.0)
        expected = int(1.0 * 32 * 16 / (32 + 16))  # = 10
        assert rank == expected

    def test_minimum_rank_is_1(self, comp):
        """Very high compression ratio should still yield rank >= 1."""
        rank = comp.compute_target_rank(32, 16, 0.999)
        assert rank >= 1

    def test_rank_does_not_exceed_min_dim(self, comp):
        """Rank should never exceed min(d_out, d_in)."""
        rank = comp.compute_target_rank(8, 4, 0.0)
        assert rank <= min(8, 4)
