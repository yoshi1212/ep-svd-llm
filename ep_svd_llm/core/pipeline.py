"""Sequential compression pipeline for SVD-LLM variants.

This module keeps the orchestration logic in one place:
1) collect inputs for the current block,
2) estimate per-layer statistics (Hessian / EP cross term),
3) compress target linear layers,
4) propagate compressed outputs to the next block.

The key design point is that calibration inputs are propagated *after* each
block is compressed, so later blocks see the same distribution as inference.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from ep_svd_llm.models.loader import (
    get_decoder_blocks,
    find_layers_in_block,
    set_layer_by_name,
    get_sequential_groups,
    LowRankLinear,
)
from ep_svd_llm.utils.activation import (
    accumulate_block_hessians,
    accumulate_block_ep_hessians,
)


@dataclass
class CompressionRunResult:
    """Summary metrics returned by one end-to-end compression run."""

    total_original: int
    total_compressed: int
    skipped_layers: int
    overall_ratio: float
    peak_vram_compress_gb: float


class SequentialCompressionPipeline:
    """Run block-by-block sequential compression for SVD-LLM variants."""

    def __init__(self, device: str, dtype: torch.dtype = torch.float32):
        self.device = device
        self.dtype = dtype

    def run(
        self,
        model: nn.Module,
        calibration_samples,
        method: str,
        compressor,
        compression_ratio: float,
        target_modules=None,
        orig_model: Optional[nn.Module] = None,
    ) -> CompressionRunResult:
        """
        Execute sequential compression for one model.

        Args:
            model: Model to be compressed in-place.
            calibration_samples: Token-id samples used for calibration.
            method: One of "svd_llm", "sc_svd_llm", or "ep_svd_llm".
            compressor: Compressor instance matching the method.
            compression_ratio: Per-layer target compression ratio.
            target_modules: Optional list of layer-name substrings to compress.
            orig_model: Optional original reference model (required by
                "svd_llm" and "ep_svd_llm").

        Returns:
            CompressionRunResult with parameter and memory statistics.
        """
        blocks, embed_modules, _ = get_decoder_blocks(model)
        orig_blocks = get_decoder_blocks(orig_model)[0] if orig_model is not None else None

        print(f"Detected {len(blocks)} decoder blocks.")

        print("Collecting initial block inputs from embedding layer …")
        block_inputs = self._collect_block_inputs(
            model,
            calibration_samples,
            embed_modules,
            blocks,
        )

        block_inputs_orig = None
        if orig_model is not None:
            orig_embed_modules = get_decoder_blocks(orig_model)[1]
            block_inputs_orig = self._collect_block_inputs(
                orig_model,
                calibration_samples,
                orig_embed_modules,
                orig_blocks,
            )

        total_original = 0
        total_compressed = 0
        skipped_layers = 0

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        for block_idx, block in enumerate(tqdm(blocks, desc="Blocks")):
            print(f"\n[Block {block_idx}] Collecting Hessians …")

            layer_map = find_layers_in_block(block, target_modules=target_modules)
            block.to(self.device)
            sequential_groups = get_sequential_groups(list(layer_map.keys()))

            if method == "ep_svd_llm":
                if orig_blocks is None or block_inputs_orig is None:
                    raise ValueError("EP-SVD-LLM requires orig_model and orig block inputs.")

                orig_block = orig_blocks[block_idx]
                orig_layer_map = find_layers_in_block(orig_block, target_modules=target_modules)

                # EP-SVD-LLM processes layers group-by-group in execution order,
                # so error propagation terms are estimated with the right context.
                for group_names in sequential_groups:
                    group_filtered = (
                        [n for n in group_names if any(t in n for t in target_modules)]
                        if target_modules
                        else group_names
                    )
                    subset = {n: layer_map[n] for n in group_filtered if n in layer_map}
                    subset_orig = {n: orig_layer_map[n] for n in group_filtered if n in orig_layer_map}

                    if not subset:
                        continue

                    h_accs, dh_accs = accumulate_block_ep_hessians(
                        block=block,
                        orig_block=orig_block,
                        block_inputs=block_inputs,
                        block_inputs_orig=block_inputs_orig,
                        subset=subset,
                        subset_orig=subset_orig,
                        device=self.device,
                        dtype=self.dtype,
                    )

                    for rel_name, layer in subset.items():
                        weight = layer.weight.data
                        d_out, d_in = weight.shape
                        target_rank = compressor.compute_target_rank(
                            d_out, d_in, compression_ratio
                        )
                        full_name = self._find_full_name(model, layer)
                        if full_name is None:
                            skipped_layers += 1
                            continue

                        H_hat = h_accs[rel_name].get_hessian(normalize=True)
                        delta_xhat = dh_accs[rel_name].get_delta_hessian(normalize=True)
                        try:
                            result = compressor.compress_layer_from_stats(
                                weight=weight,
                                hessian_compressed=H_hat,
                                delta_hessian=delta_xhat,
                                target_rank=target_rank,
                                layer_name=full_name,
                            )
                            total_original += result.original_params
                            total_compressed += result.compressed_params

                            bias = layer.bias.data if layer.bias is not None else None
                            low_rank = LowRankLinear(result.W_u, result.W_v, bias=bias)
                            set_layer_by_name(model, full_name, low_rank)
                            layer_map[rel_name] = low_rank
                        except Exception as exc:
                            print(f"\nWARNING: Failed to compress {full_name}: {exc}")
                            skipped_layers += 1

                orig_block.cpu()
                torch.cuda.empty_cache()

            elif method == "svd_llm":
                if orig_blocks is None or block_inputs_orig is None:
                    raise ValueError("SVD-LLM requires orig_model and orig block inputs.")

                orig_block = orig_blocks[block_idx]
                h_accs = accumulate_block_hessians(
                    orig_block,
                    block_inputs_orig,
                    device=self.device,
                    dtype=self.dtype,
                    target_modules=target_modules,
                )
                total_original, total_compressed, skipped_layers = self._compress_layers_from_hessian(
                    model=model,
                    layer_map=layer_map,
                    h_accs=h_accs,
                    compressor=compressor,
                    compression_ratio=compression_ratio,
                    total_original=total_original,
                    total_compressed=total_compressed,
                    skipped_layers=skipped_layers,
                )

            else:
                # SC-SVD-LLM also processes layers group-by-group in execution
                # order so its only difference from EP-SVD-LLM is the absence
                # of the EP correction term.
                for group_names in sequential_groups:
                    group_filtered = (
                        [n for n in group_names if any(t in n for t in target_modules)]
                        if target_modules
                        else group_names
                    )
                    subset = {n: layer_map[n] for n in group_filtered if n in layer_map}
                    if not subset:
                        continue

                    h_accs = accumulate_block_hessians(
                        block,
                        block_inputs,
                        device=self.device,
                        dtype=self.dtype,
                        target_modules=list(subset.keys()),
                    )
                    total_original, total_compressed, skipped_layers = self._compress_layers_from_hessian(
                        model=model,
                        layer_map=layer_map,
                        h_accs=h_accs,
                        compressor=compressor,
                        compression_ratio=compression_ratio,
                        total_original=total_original,
                        total_compressed=total_compressed,
                        skipped_layers=skipped_layers,
                        layer_names=list(subset.keys()),
                    )

            block.cpu()
            torch.cuda.empty_cache()

            # Crucial step: next block receives compressed activations X_hat.
            print(f"[Block {block_idx}] Propagating outputs …")
            block_inputs = self._forward_block(block, block_inputs)
            if orig_blocks is not None and block_inputs_orig is not None:
                block_inputs_orig = self._forward_block(orig_blocks[block_idx], block_inputs_orig)

        overall_ratio = 1 - total_compressed / max(total_original, 1)
        peak_vram_compress_gb = (
            torch.cuda.max_memory_allocated() / 1024**3
            if torch.cuda.is_available()
            else 0.0
        )

        return CompressionRunResult(
            total_original=total_original,
            total_compressed=total_compressed,
            skipped_layers=skipped_layers,
            overall_ratio=overall_ratio,
            peak_vram_compress_gb=peak_vram_compress_gb,
        )

    def _compress_layers_from_hessian(
        self,
        model: nn.Module,
        layer_map,
        h_accs,
        compressor,
        compression_ratio: float,
        total_original: int,
        total_compressed: int,
        skipped_layers: int,
        layer_names=None,
    ):
        """Compress all target layers in one block using pre-accumulated Hessians."""
        items = (
            ((name, layer_map[name]) for name in layer_names if name in layer_map)
            if layer_names is not None
            else layer_map.items()
        )
        for rel_name, layer in items:
            weight = layer.weight.data
            d_out, d_in = weight.shape
            target_rank = compressor.compute_target_rank(d_out, d_in, compression_ratio)

            full_name = self._find_full_name(model, layer)
            if full_name is None:
                skipped_layers += 1
                continue

            try:
                if rel_name in h_accs:
                    hessian = h_accs[rel_name].get_hessian(normalize=True)
                    result = compressor.compress_layer_from_hessian(
                        weight=weight,
                        hessian=hessian,
                        target_rank=target_rank,
                        layer_name=full_name,
                    )
                    total_original += result.original_params
                    total_compressed += result.compressed_params

                    bias = layer.bias.data if layer.bias is not None else None
                    low_rank = LowRankLinear(result.W_u, result.W_v, bias=bias)
                    set_layer_by_name(model, full_name, low_rank)
                    layer_map[rel_name] = low_rank
                else:
                    skipped_layers += 1
                    continue
            except Exception as exc:
                print(f"\nWARNING: Failed to compress {full_name}: {exc}")
                skipped_layers += 1
                continue

        return total_original, total_compressed, skipped_layers

    def _collect_block_inputs(self, model, calibration_samples, embed_modules, blocks):
        """
        Capture the input tensors for the first decoder block.

        We temporarily replace block 0 with a Catcher module that stores
        hidden states and keyword arguments, then raises to stop full forward.
        """
        for module in embed_modules:
            module.to(self.device)

        block_inputs = []

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def __getattr__(self, name):
                try:
                    return super().__getattr__(name)
                except AttributeError:
                    return getattr(self.module, name)

            def forward(self, hidden_states, **kwargs):
                # Store on CPU to reduce peak VRAM during calibration capture.
                block_inputs.append(
                    (
                        hidden_states.detach().cpu(),
                        {
                            k: (v.cpu() if isinstance(v, torch.Tensor) else v)
                            for k, v in kwargs.items()
                        },
                    )
                )
                raise ValueError("catcher")

        original_block0 = blocks[0]
        blocks[0] = Catcher(original_block0)

        with torch.no_grad():
            for ids in calibration_samples:
                ids = ids.to(self.device)
                try:
                    model(ids)
                except ValueError:
                    pass

        blocks[0] = original_block0

        for module in embed_modules:
            module.cpu()
        torch.cuda.empty_cache()

        return block_inputs

    def _forward_block(self, block, block_inputs):
        """Forward one block over all cached samples and return next-block inputs."""
        block.to(self.device)
        block.eval()
        next_inputs = []

        with torch.no_grad():
            for inp, kwargs in block_inputs:
                inp = inp.to(self.device)
                dev_kwargs = {}
                for key, value in kwargs.items():
                    if isinstance(value, torch.Tensor):
                        dev_kwargs[key] = value.to(self.device)
                    else:
                        dev_kwargs[key] = value
                out = block(inp, **dev_kwargs)
                hidden = out[0] if isinstance(out, (tuple, list)) else out
                next_inputs.append(
                    (
                        hidden.cpu(),
                        {
                            k: (v.cpu() if isinstance(v, torch.Tensor) else v)
                            for k, v in dev_kwargs.items()
                        },
                    )
                )

        block.cpu()
        torch.cuda.empty_cache()
        return next_inputs

    @staticmethod
    def _find_full_name(model: nn.Module, target: nn.Module) -> Optional[str]:
        """Return the dotted module path of target inside model, if found."""
        for name, module in model.named_modules():
            if module is target:
                return name
        return None
