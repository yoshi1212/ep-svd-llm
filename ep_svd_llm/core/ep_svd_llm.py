"""
EP-SVD-LLM: Error-Propagation SVD for LLM Compression.

Applies the Error Propagation (EP) idea from QEP to SVD-based compression.
Accumulated errors from previously compressed layers are explicitly propagated
and compensated, improving reconstruction accuracy.
"""

import torch
from typing import Optional
from .base_compressor import BaseCompressor, CompressionResult


class EPSVDLLMCompressor(BaseCompressor):
    """
    EP-SVD-LLM Compressor with error propagation compensation.

    Algorithm:
        1. Compute accumulated error δ = X_orig - X_compressed.
        2. Compute correction: correction = W @ δ @ X̂^T @ Ĥ^{-1}.
        3. Apply corrected weight: W* = W + α * correction.
        4. Apply SC-SVD-LLM whitening + SVD on W*.

    When alpha=0 or activations_orig is None, degenerates to SC-SVD-LLM.

    Both ``compress_layer`` and ``compress_layer_from_stats`` delegate to
    the shared ``_compress_core`` (defined in ``BaseCompressor``), which
    performs a **single** SVD on the Hessian.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        regularization: float = 1e-6,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
            alpha: Error propagation strength in [0, 1].
                   0 = no correction (equivalent to SC-SVD-LLM).
                   1 = full correction.
            regularization: Hessian regularization for numerical stability.
            device: Computation device.
            dtype: Computation dtype.
        """
        super().__init__(regularization, device, dtype)
        self.alpha = alpha

    def compress_layer(
        self,
        weight: torch.Tensor,
        activations: torch.Tensor,
        target_rank: int,
        layer_name: str = "",
        activations_orig: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CompressionResult:
        """
        Compress a single layer with EP-SVD-LLM.

        This method computes the Hessian and (optionally) the EP cross-term
        from raw activations, then delegates to ``_compress_core``.

        Args:
            weight: Weight matrix (d_out, d_in).
            activations: Compressed-model activations X̂_l (d_in, N).
            target_rank: Target rank.
            layer_name: Layer name for logging.
            activations_orig: Original-model activations X_l (d_in, N).
                              If None, behaves like SC-SVD-LLM.

        Returns:
            CompressionResult
        """
        X = activations.to(self.device).to(self.dtype)
        N = X.shape[1]

        # Per-token normalised Hessian (without regularisation —
        # _compress_core adds regularisation internally).
        H_hat = (X @ X.T) / max(N, 1)

        delta_X_hat_T = None
        if activations_orig is not None and self.alpha > 0:
            X_orig = activations_orig.to(self.device).to(self.dtype)
            delta = X_orig - X  # (d_in, N)
            delta_X_hat_T = (delta @ X.T) / max(N, 1)

        return self._compress_core(
            weight, H_hat, target_rank, layer_name,
            delta_X_hat_T=delta_X_hat_T, alpha=self.alpha,
        )

    def compress_layer_from_stats(
        self,
        weight: torch.Tensor,
        hessian_compressed: torch.Tensor,
        delta_hessian: torch.Tensor,
        target_rank: int,
        layer_name: str = "",
    ) -> CompressionResult:
        """
        Compress using pre-computed statistics (useful with online accumulation).

        Args:
            weight: Weight matrix (d_out, d_in).
            hessian_compressed: Compressed-activation Hessian Ĥ = X̂ @ X̂^T (d_in, d_in).
                                Must be normalised (/N) and **without** regularisation.
            delta_hessian: Error cross-term δ @ X̂^T (d_in, d_in), normalised (/N).
            target_rank: Target rank.
            layer_name: Layer name for logging.

        Returns:
            CompressionResult
        """
        return self._compress_core(
            weight, hessian_compressed, target_rank, layer_name,
            delta_X_hat_T=delta_hessian, alpha=self.alpha,
        )
