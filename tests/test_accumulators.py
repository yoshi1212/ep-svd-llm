"""
Unit tests for HessianAccumulator and DeltaHessianAccumulator.

All tests run on CPU with small dummy matrices — no GPU or model download required.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from ep_svd_llm.utils.activation import HessianAccumulator, DeltaHessianAccumulator

DEVICE = "cpu"
DTYPE = torch.float32


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return torch.Generator().manual_seed(99)


# ---------------------------------------------------------------------------
# 1. HessianAccumulator
# ---------------------------------------------------------------------------

class TestHessianAccumulator:
    def test_single_batch(self, rng):
        """After one batch, get_hessian() should equal (X @ X.T) / N."""
        dim = 8
        X = torch.randn(dim, 20, generator=rng, dtype=DTYPE)
        acc = HessianAccumulator(dim=dim, device=DEVICE, dtype=DTYPE)
        acc.add(X)
        H = acc.get_hessian(normalize=True)
        expected = (X @ X.T) / X.shape[1]
        torch.testing.assert_close(H, expected, rtol=1e-5, atol=1e-6)

    def test_multiple_batches(self, rng):
        """Accumulating two batches should equal computing on concatenated data."""
        dim = 8
        X1 = torch.randn(dim, 15, generator=rng, dtype=DTYPE)
        X2 = torch.randn(dim, 25, generator=rng, dtype=DTYPE)

        acc = HessianAccumulator(dim=dim, device=DEVICE, dtype=DTYPE)
        acc.add(X1)
        acc.add(X2)
        H = acc.get_hessian(normalize=True)

        X_all = torch.cat([X1, X2], dim=1)
        expected = (X_all @ X_all.T) / X_all.shape[1]
        torch.testing.assert_close(H, expected, rtol=1e-5, atol=1e-6)

    def test_normalize_false(self, rng):
        """normalize=False should return the raw sum X @ X.T."""
        dim = 4
        X = torch.randn(dim, 10, generator=rng, dtype=DTYPE)
        acc = HessianAccumulator(dim=dim, device=DEVICE, dtype=DTYPE)
        acc.add(X)
        H_raw = acc.get_hessian(normalize=False)
        expected = X @ X.T
        torch.testing.assert_close(H_raw, expected, rtol=1e-5, atol=1e-6)

    def test_reset(self, rng):
        """After reset(), H should be all zeros and n_samples should be 0."""
        dim = 4
        X = torch.randn(dim, 10, generator=rng, dtype=DTYPE)
        acc = HessianAccumulator(dim=dim, device=DEVICE, dtype=DTYPE)
        acc.add(X)
        acc.reset()
        assert acc.n_samples == 0
        assert (acc.H == 0).all()

    def test_n_samples_tracking(self, rng):
        """n_samples should track total token count across batches."""
        dim = 4
        acc = HessianAccumulator(dim=dim, device=DEVICE, dtype=DTYPE)
        acc.add(torch.randn(dim, 10, generator=rng, dtype=DTYPE))
        acc.add(torch.randn(dim, 20, generator=rng, dtype=DTYPE))
        assert acc.n_samples == 30


# ---------------------------------------------------------------------------
# 2. DeltaHessianAccumulator
# ---------------------------------------------------------------------------

class TestDeltaHessianAccumulator:
    def test_single_batch(self, rng):
        """After one batch, result should equal (delta @ X_compressed.T) / N."""
        dim = 8
        X_orig = torch.randn(dim, 20, generator=rng, dtype=DTYPE)
        X_comp = torch.randn(dim, 20, generator=rng, dtype=DTYPE)

        acc = DeltaHessianAccumulator(dim=dim, device=DEVICE, dtype=DTYPE)
        acc.add(X_orig, X_comp)
        result = acc.get_delta_hessian(normalize=True)

        delta = X_orig - X_comp
        expected = (delta @ X_comp.T) / X_orig.shape[1]
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)

    def test_multiple_batches(self, rng):
        """Accumulating multiple batches should match concatenated computation."""
        dim = 8
        X_orig1 = torch.randn(dim, 10, generator=rng, dtype=DTYPE)
        X_comp1 = torch.randn(dim, 10, generator=rng, dtype=DTYPE)
        X_orig2 = torch.randn(dim, 15, generator=rng, dtype=DTYPE)
        X_comp2 = torch.randn(dim, 15, generator=rng, dtype=DTYPE)

        acc = DeltaHessianAccumulator(dim=dim, device=DEVICE, dtype=DTYPE)
        acc.add(X_orig1, X_comp1)
        acc.add(X_orig2, X_comp2)
        result = acc.get_delta_hessian(normalize=True)

        X_orig_all = torch.cat([X_orig1, X_orig2], dim=1)
        X_comp_all = torch.cat([X_comp1, X_comp2], dim=1)
        delta = X_orig_all - X_comp_all
        expected = (delta @ X_comp_all.T) / X_orig_all.shape[1]
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)

    def test_reset(self, rng):
        """After reset(), delta_X_hat should be all zeros."""
        dim = 4
        acc = DeltaHessianAccumulator(dim=dim, device=DEVICE, dtype=DTYPE)
        acc.add(
            torch.randn(dim, 10, generator=rng, dtype=DTYPE),
            torch.randn(dim, 10, generator=rng, dtype=DTYPE),
        )
        acc.reset()
        assert acc.n_samples == 0
        assert (acc.delta_X_hat == 0).all()

    def test_zero_error(self, rng):
        """When X_orig == X_compressed, delta Hessian should be zero."""
        dim = 4
        X = torch.randn(dim, 10, generator=rng, dtype=DTYPE)
        acc = DeltaHessianAccumulator(dim=dim, device=DEVICE, dtype=DTYPE)
        acc.add(X, X)
        result = acc.get_delta_hessian(normalize=True)
        torch.testing.assert_close(result, torch.zeros(dim, dim), atol=1e-7, rtol=0)
