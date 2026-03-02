"""
SVD-LLM: Truncation-Aware Data Whitening for LLM Compression.

This module implements the core SVD-LLM algorithm (whitening + truncated SVD).
The same class is used for both **SVD-LLM** and **SC-SVD-LLM** — the
distinction depends solely on which activations are passed:

  - **SVD-LLM**:    pass original-model activations X   → Hessian reflects the original model
  - **SC-SVD-LLM**: pass compressed activations X̂ → Hessian reflects the
                     sequentially-compressed model (inference-time distribution)

Reference:
    SVD-LLM: Truncation-aware Singular Value Decomposition for Large Language Model Compression
    https://arxiv.org/abs/2403.07378
"""

import torch
from typing import Optional
from .base_compressor import BaseCompressor, CompressionResult


class SVDLLMCompressor(BaseCompressor):
    """
    SVD-LLM compressor via Truncation-Aware Data Whitening.

    This class implements the per-layer compression algorithm shared by
    SVD-LLM and SC-SVD-LLM.  The caller controls the behaviour by choosing
    which activations to provide:

      * Pass **original-model activations X**  → original SVD-LLM (Hessian from original model)
      * Pass **compressed activations X̂** → SC-SVD-LLM (Hessian from
        sequentially-compressed model)

    Algorithm (per layer):
        1. Compute Hessian H = X @ X^T  (or X̂ @ X̂^T) from activations.
        2. Obtain whitening matrix P and its inverse P_inv via SVD of H:
               H = U Λ U^T  →  P = U √Λ,  P_inv = (1/√Λ) U^T.
        3. Apply SVD to the whitened weight: W @ P = U Σ V^T.
        4. Truncate to top-k singular values.
        5. Undo whitening: W' = U_k Σ_k V_k^T P_inv.
        6. Store in low-rank factored form: W' = W_u @ W_v.

    Both ``compress_layer`` and ``compress_layer_from_hessian`` delegate to
    the shared ``_compress_core`` (defined in ``BaseCompressor``).
    """

    def compress_layer(
        self,
        weight: torch.Tensor,
        activations: torch.Tensor,
        target_rank: int,
        layer_name: str = "",
        **kwargs,
    ) -> CompressionResult:
        """
        Compress a single layer.

        Args:
            weight: Weight matrix (d_out, d_in).
            activations: Input activations (d_in, N).
                         Pass original-model activations for SVD-LLM, or
                         compressed activations for SC-SVD-LLM.
            target_rank: Target rank.
            layer_name: Layer name for logging.

        Returns:
            CompressionResult
        """
        X = activations.to(self.device).to(self.dtype)
        N = X.shape[1]

        # Per-token normalised Hessian (without regularisation —
        # _compress_core adds regularisation internally).
        H = (X @ X.T) / max(N, 1)

        return self._compress_core(weight, H, target_rank, layer_name)

    def compress_layer_from_hessian(
        self,
        weight: torch.Tensor,
        hessian: torch.Tensor,
        target_rank: int,
        layer_name: str = "",
    ) -> CompressionResult:
        """
        Compress using a pre-computed Hessian (useful with online Hessian accumulation).

        Args:
            weight: Weight matrix (d_out, d_in).
            hessian: Pre-computed Hessian (d_in, d_in).
                     Must be normalised (/N) and **without** regularisation.
            target_rank: Target rank.
            layer_name: Layer name for logging.

        Returns:
            CompressionResult
        """
        return self._compress_core(weight, hessian, target_rank, layer_name)
