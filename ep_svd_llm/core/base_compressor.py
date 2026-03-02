"""
Base compressor abstract class and CompressionResult dataclass.
"""

from abc import ABC, abstractmethod
import torch
from torch import nn
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class CompressionResult:
    """Stores the result of a layer-wise SVD compression."""

    W_u: torch.Tensor  # (d_out, rank)
    W_v: torch.Tensor  # (rank, d_in)
    original_shape: Tuple[int, int]
    target_rank: int
    compression_ratio: float
    layer_name: str

    @property
    def compressed_params(self) -> int:
        """Number of parameters after compression."""
        return self.W_u.numel() + self.W_v.numel()

    @property
    def original_params(self) -> int:
        """Number of parameters before compression."""
        return self.original_shape[0] * self.original_shape[1]

    @property
    def actual_compression_ratio(self) -> float:
        """Actual achieved compression ratio."""
        return 1 - (self.compressed_params / self.original_params)


class BaseCompressor(ABC):
    """
    Abstract base class for layer-wise SVD compression.

    Defines the common interface and shared utilities for
    SVDLLMCompressor and EPSVDLLMCompressor.
    """

    def __init__(
        self,
        regularization: float = 1e-6,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
            regularization: Hessian regularization for numerical stability.
            device: Computation device.
            dtype: Computation dtype (float32 recommended for SVD).
        """
        self.regularization = regularization
        self.device = device
        self.dtype = dtype

    @abstractmethod
    def compress_layer(
        self,
        weight: torch.Tensor,
        activations: torch.Tensor,
        target_rank: int,
        layer_name: str = "",
        **kwargs,
    ) -> CompressionResult:
        """
        Compress a single linear layer.

        Args:
            weight: Weight matrix (d_out, d_in).
            activations: Input activations (d_in, N).
            target_rank: Target rank for truncation.
            layer_name: Layer name for logging.

        Returns:
            CompressionResult
        """
        pass

    def compute_hessian(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute the Hessian H = (1/N) * X @ X^T with regularization.

        Normalising by the number of tokens N ensures that the scale of H
        (and therefore the whitening matrix P) is independent of the number
        of calibration samples.  This makes the regularisation term
        ``regularization * I`` have a consistent relative magnitude regardless
        of how many tokens were seen, and keeps the result compatible with
        :meth:`compress_layer_from_hessian` which also expects a per-token
        normalised Hessian (see :class:`HessianAccumulator`).

        Args:
            X: Activations (d_in, N).

        Returns:
            H: Regularised, normalised Hessian (d_in, d_in).
        """
        X = X.to(self.device).to(self.dtype)
        n_tokens = X.shape[1]
        H = (X @ X.T) / max(n_tokens, 1)
        H = H + self.regularization * torch.eye(
            H.shape[0], device=self.device, dtype=self.dtype
        )
        return H

    def compute_whitening_matrix(
        self, H: torch.Tensor
    ) -> tuple:
        """
        Compute the whitening matrix P and its inverse P_inv via SVD.

        For the symmetric positive semi-definite Hessian H = U Λ U^T (eigendecomposition
        via SVD), the whitening matrix and its inverse are defined as:

            P     = U @ diag(sqrt(λ_i))          (d_in, d_in)
            P_inv = diag(1/sqrt(λ_i)) @ U^T      (d_in, d_in)

        Small singular values whose reciprocal would be numerically unstable are
        treated as zero (pseudo-inverse), matching the methodology in Section 2.3.

        Args:
            H: Regularized Hessian (d_in, d_in), assumed symmetric PSD.

        Returns:
            P:     Whitening matrix     (d_in, d_in)  – use as  W @ P
            P_inv: Inverse of P         (d_in, d_in)  – use as  ... @ P_inv
        """
        # SVD of the symmetric Hessian: H = U S V^T, but since H is symmetric
        # positive semi-definite, S contains the eigenvalues and U == V.
        U, S, _ = torch.linalg.svd(H, full_matrices=False)

        sqrt_S = torch.sqrt(S.clamp(min=0.0) + 1e-10)

        # Tikhonov regularization: avoids division-by-zero while preserving
        # all directions (unlike threshold-based pseudo-inverse).
        eps = 1e-6
        inv_sqrt_S = 1.0 / (sqrt_S + eps)

        # P = U * sqrt(Λ),  shape (d_in, d_in)
        P = U * sqrt_S.unsqueeze(0)

        # P_inv = diag(1/sqrt(Λ)) @ U^T,  shape (d_in, d_in)
        P_inv = (U * inv_sqrt_S.unsqueeze(0)).T

        return P, P_inv

    def _compress_core(
        self,
        weight: torch.Tensor,
        H_hat: torch.Tensor,
        target_rank: int,
        layer_name: str = "",
        delta_X_hat_T: Optional[torch.Tensor] = None,
        alpha: float = 0.0,
    ) -> "CompressionResult":
        """
        Core compression logic shared by SC-SVD-LLM and EP-SVD-LLM.

        Performs a **single** SVD on the regularised Hessian and derives both
        the whitening transform and (optionally) the EP pseudo-inverse from it.

        Args:
            weight: Weight matrix (d_out, d_in).
            H_hat: Hessian of compressed-model activations (d_in, d_in).
                   Should be normalised (per-token) but **without** regularisation
                   (regularisation is added internally).
            target_rank: Target rank for truncated SVD.
            layer_name: Layer name for logging.
            delta_X_hat_T: EP cross-term  (1/N) δ @ X̂^T  (d_in, d_in).
                           Pass None (or leave alpha=0) for plain SC-SVD-LLM.
            alpha: EP correction strength.  0 = no correction.

        Returns:
            CompressionResult
        """
        d_out, d_in = weight.shape
        W = weight.to(self.device).to(self.dtype)
        H = H_hat.to(self.device).to(self.dtype)

        # Add regularisation to H
        H_reg = H + self.regularization * torch.eye(
            d_in, device=self.device, dtype=self.dtype
        )

        # --- Single SVD of the regularised Hessian -----------------------
        U_h, S_h, _ = torch.linalg.svd(H_reg, full_matrices=False)

        eps = 1e-6
        sqrt_S_h = torch.sqrt(S_h.clamp(min=0.0) + 1e-10)
        inv_sqrt_S_h = 1.0 / (sqrt_S_h + eps)

        P = U_h * sqrt_S_h.unsqueeze(0)            # (d_in, d_in)
        P_inv = (U_h * inv_sqrt_S_h.unsqueeze(0)).T  # (d_in, d_in)

        # --- Whitened input construction ---------------------------------
        # Build W_star @ P directly:
        #   W_star P = W P + alpha * W delta_X_hat_T P^{-T}
        #
        # This is algebraically equivalent to constructing
        #   W_star = W + alpha * W delta_X_hat_T H_hat^{-1}
        # first and then whitening, but avoids materialising the full
        # pseudo-inverse and the corrected dense weight matrix.
        WP = W @ P
        if alpha > 0 and delta_X_hat_T is not None:
            dxt = delta_X_hat_T.to(self.device).to(self.dtype)
            WP = WP + alpha * (W @ dxt @ P_inv.T)

        # --- Whitened truncated SVD --------------------------------------
        U_k, Sigma_k, Vh_k = self.truncated_svd(WP, target_rank)

        sqrt_sigma = torch.sqrt(Sigma_k)
        W_u = U_k * sqrt_sigma.unsqueeze(0)              # (d_out, k)
        W_v = (sqrt_sigma.unsqueeze(1) * Vh_k) @ P_inv   # (k, d_in)

        W_u = W_u.to(weight.dtype)
        W_v = W_v.to(weight.dtype)

        compression_ratio = 1 - (W_u.numel() + W_v.numel()) / (d_out * d_in)

        return CompressionResult(
            W_u=W_u,
            W_v=W_v,
            original_shape=(d_out, d_in),
            target_rank=target_rank,
            compression_ratio=compression_ratio,
            layer_name=layer_name,
        )

    def truncated_svd(
        self,
        W: torch.Tensor,
        target_rank: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform truncated SVD.

        Args:
            W: Input matrix (d_out, d_in).
            target_rank: Number of singular values/vectors to keep.

        Returns:
            U_k, Sigma_k, Vh_k
        """
        W = W.to(self.device).to(self.dtype)
        U, Sigma, Vh = torch.linalg.svd(W, full_matrices=False)
        U_k = U[:, :target_rank]
        Sigma_k = Sigma[:target_rank]
        Vh_k = Vh[:target_rank, :]
        return U_k, Sigma_k, Vh_k

    def compute_target_rank(
        self,
        d_out: int,
        d_in: int,
        compression_ratio: float,
    ) -> int:
        """
        Compute target rank from a desired compression ratio.

        After compression: (d_out + d_in) * rank parameters.
        Before compression: d_out * d_in parameters.

        rank = (1 - compression_ratio) * d_out * d_in / (d_out + d_in)

        Args:
            d_out: Output dimension.
            d_in: Input dimension.
            compression_ratio: Desired compression ratio in [0, 1).

        Returns:
            Target rank (int).
        """
        rank_ratio = 1 - compression_ratio
        rank = int(rank_ratio * d_out * d_in / (d_out + d_in))
        rank = max(1, rank)
        rank = min(rank, min(d_out, d_in))
        return rank
