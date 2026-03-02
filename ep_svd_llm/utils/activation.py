"""Activation and statistic accumulation utilities for compression.

The implementation intentionally avoids storing full activation histories.
Instead, it accumulates matrix statistics online so memory scales with O(d^2)
per layer, not O(num_tokens * d).
"""

import torch
from torch import nn
from typing import Dict, List, Optional, Callable, Tuple
from contextlib import contextmanager
from ep_svd_llm.models.loader import find_layers_in_block


class ActivationCollector:
    """
    Collect activations from specified layers via forward hooks.

    Usage::

        collector = ActivationCollector(model)
        collector.register_hooks(["model.layers.0.mlp.gate_proj"])

        with torch.no_grad():
            model(input_ids)

        X = collector.get_activations("model.layers.0.mlp.gate_proj")
        collector.clear()
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.activations: Dict[str, List[torch.Tensor]] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

    def _create_hook(self, name: str) -> Callable:
        """Create a forward hook that records the layer input."""
        def hook(module, input, output):
            inp = input[0] if isinstance(input, tuple) else input
            if name not in self.activations:
                self.activations[name] = []
            self.activations[name].append(inp.detach())
        return hook

    def register_hooks(self, layer_names: List[str]) -> None:
        """
        Register forward hooks on the specified layers.

        Args:
            layer_names: List of module names to hook.
        """
        self.clear_hooks()
        for name, module in self.model.named_modules():
            if name in layer_names:
                hook = module.register_forward_hook(self._create_hook(name))
                self.hooks.append(hook)

    def get_activations(self, name: str) -> Optional[torch.Tensor]:
        """
        Return collected activations concatenated as (d_in, total_tokens).

        .. note::
            This returns *all* stored activations as one big tensor.
            Prefer :meth:`streaming_hessians` when memory efficiency matters
            (i.e., always for production use).

        Args:
            name: Module name.

        Returns:
            Tensor of shape (d_in, total_tokens), or None if no data.
        """
        if name not in self.activations or len(self.activations[name]) == 0:
            return None

        all_acts = []
        for act in self.activations[name]:
            act_flat = act.reshape(-1, act.shape[-1])  # (B*seq, d)
            all_acts.append(act_flat)

        concatenated = torch.cat(all_acts, dim=0)  # (total_tokens, d)
        return concatenated.T  # (d, total_tokens)

    @contextmanager
    def streaming_hessians(
        self,
        layer_names: List[str],
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Context manager that accumulates per-sample Hessians *online*.

        This is the SC-SVD-LLM-recommended usage: instead of storing all
        activations and concatenating them (O(N * d) memory), each forward
        pass immediately contributes to H += X @ X^T and the activations are
        discarded.  Only one (d, d) matrix per layer is retained in memory.

        Usage::

            accumulators = {}  # filled by the context manager
            with collector.streaming_hessians(layer_names, device) as accumulators:
                for text in calibration_texts:
                    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
                    with torch.no_grad():
                        model(ids)        # hooks fire → accumulators updated

            H = accumulators["model.layers.0.mlp.gate_proj"].get_hessian()

        Args:
            layer_names: Modules to hook.
            device: Device for the Hessian matrices.
            dtype: Dtype for the Hessian matrices.

        Yields:
            Dict[str, HessianAccumulator] – one accumulator per layer.
        """
        # accumulators are initialised lazily on the first forward pass,
        # when the feature dimension is known.
        accumulators: Dict[str, "HessianAccumulator"] = {}

        def _make_stream_hook(name: str):
            def hook(module, input, output):
                inp = input[0] if isinstance(input, tuple) else input
                x = inp.detach().reshape(-1, inp.shape[-1]).T  # (d, tokens)
                x = x.to(device).to(dtype)
                if name not in accumulators:
                    accumulators[name] = HessianAccumulator(
                        dim=x.shape[0], device=device, dtype=dtype
                    )
                accumulators[name].add(x)
            return hook

        # Register temporary hooks.
        stream_hooks = []
        for name, module in self.model.named_modules():
            if name in layer_names:
                stream_hooks.append(
                    module.register_forward_hook(_make_stream_hook(name))
                )

        try:
            yield accumulators
        finally:
            for h in stream_hooks:
                h.remove()

    def clear(self) -> None:
        """Clear collected activations."""
        self.activations.clear()
        torch.cuda.empty_cache()

    def clear_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def __del__(self):
        self.clear_hooks()


class HessianAccumulator:
    """
    Incrementally accumulate H = X @ X^T without storing all activations.

    Usage::

        acc = HessianAccumulator(dim=4096, device="cuda")
        for X in batches:            # X: (d, n_tokens)
            acc.add(X)
        H = acc.get_hessian()
    """

    def __init__(self, dim: int, device: str = "cuda", dtype=torch.float32):
        self.dim = dim
        self.device = device
        self.dtype = dtype
        self.H = torch.zeros(dim, dim, device=device, dtype=dtype)
        self.n_samples = 0

    def add(self, X: torch.Tensor) -> None:
        """
        Accumulate H += X @ X^T.

        Args:
            X: Activations (d, n_tokens).
        """
        # Store unnormalized second moments and normalize only at read time.
        X = X.to(self.device).to(self.dtype)
        self.H.add_(X @ X.T)
        self.n_samples += X.shape[1]

    def get_hessian(self, normalize: bool = True) -> torch.Tensor:
        """
        Return the accumulated Hessian.

        Args:
            normalize: If True, divide by the total number of tokens.

        Returns:
            Hessian (d, d).
        """
        if normalize and self.n_samples > 0:
            return self.H / self.n_samples
        return self.H

    def reset(self) -> None:
        """Reset accumulator."""
        self.H.zero_()
        self.n_samples = 0


class DeltaHessianAccumulator:
    """
    Incrementally accumulate δ @ X̂^T for EP-SVD-LLM.

    where δ = X_orig - X_compressed (accumulated activation error).
    """

    def __init__(self, dim: int, device: str = "cuda", dtype=torch.float32):
        self.dim = dim
        self.device = device
        self.dtype = dtype
        self.delta_X_hat = torch.zeros(dim, dim, device=device, dtype=dtype)
        self.n_samples = 0

    def add(self, X_orig: torch.Tensor, X_compressed: torch.Tensor) -> None:
        """
        Accumulate delta_X_hat += (X_orig - X_compressed) @ X_compressed^T.

        Args:
            X_orig: Original-model activations (d, n_tokens).
            X_compressed: Compressed-model activations (d, n_tokens).
        """
        # EP correction uses delta = X_orig - X_hat, then accumulates delta @ X_hat^T.
        X_orig = X_orig.to(self.device).to(self.dtype)
        X_compressed = X_compressed.to(self.device).to(self.dtype)
        delta = X_orig - X_compressed
        self.delta_X_hat.add_(delta @ X_compressed.T)
        self.n_samples += X_orig.shape[1]

    def get_delta_hessian(self, normalize: bool = True) -> torch.Tensor:
        """
        Return the accumulated δ @ X̂^T.

        Args:
            normalize: If True, divide by the total number of tokens.

        Returns:
            Delta Hessian (d, d).
        """
        if normalize and self.n_samples > 0:
            return self.delta_X_hat / self.n_samples
        return self.delta_X_hat

    def reset(self) -> None:
        """Reset accumulator."""
        self.delta_X_hat.zero_()
        self.n_samples = 0

def accumulate_block_hessians(
    block: nn.Module,
    block_inputs,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    target_modules=None,
) -> Dict[str, HessianAccumulator]:
    """
    Accumulate per-layer Hessians for one decoder block.

    Args:
        block: Decoder block.
        block_inputs: List of (hidden_states, kwargs) tuples.
        device: Device for accumulation.
        dtype: Dtype for accumulation.
        target_modules: Optional list of substrings for layer filtering.

    Returns:
        Dict[layer_relative_name, HessianAccumulator]
    """
    # Names are relative to the block, making this helper architecture-agnostic
    # once decoder blocks are identified by the model loader.
    layers = find_layers_in_block(block, target_modules=target_modules)
    accumulators: Dict[str, HessianAccumulator] = {}

    def make_hook(name: str):
        def hook(module, inp, out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).T
            x = x.to(device).to(dtype)
            if name not in accumulators:
                accumulators[name] = HessianAccumulator(
                    dim=x.shape[0], device=device, dtype=dtype
                )
            accumulators[name].add(x)
        return hook

    handles = [
        layer.register_forward_hook(make_hook(name))
        for name, layer in layers.items()
    ]

    block.to(device).eval()
    with torch.no_grad():
        for inp, kwargs in block_inputs:
            # Inputs are cached on CPU by the pipeline and moved per sample.
            inp = inp.to(device)
            dev_kwargs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in kwargs.items()
            }
            block(inp, **dev_kwargs)

    block.cpu()
    for handle in handles:
        handle.remove()
    torch.cuda.empty_cache()
    return accumulators


def accumulate_block_ep_hessians(
    block: nn.Module,
    orig_block: nn.Module,
    block_inputs,
    block_inputs_orig,
    subset: Dict[str, nn.Module],
    subset_orig: Dict[str, nn.Module],
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> Tuple[Dict[str, HessianAccumulator], Dict[str, DeltaHessianAccumulator]]:
    """
    Accumulate EP-SVD-LLM statistics for one group inside one decoder block.

    Returns:
        (hessian_accumulators, delta_hessian_accumulators)
    """
    # Input dimension is known from layer weights, so accumulators can be
    # initialized eagerly for the selected subset.
    h_accs = {
        name: HessianAccumulator(
            dim=subset[name].weight.shape[1], device=device, dtype=dtype
        )
        for name in subset
    }
    dh_accs = {
        name: DeltaHessianAccumulator(
            dim=subset[name].weight.shape[1], device=device, dtype=dtype
        )
        for name in subset
    }

    hook_data = {}
    hook_data_orig = {}

    def make_hook(storage, name: str):
        def hook(module, inp, out):
            storage[name] = inp[0].detach().reshape(-1, inp[0].shape[-1]).T.to(device).to(dtype)
        return hook

    handles = []
    for name, module in subset.items():
        handles.append(module.register_forward_hook(make_hook(hook_data, name)))
    for name, module in subset_orig.items():
        handles.append(module.register_forward_hook(make_hook(hook_data_orig, name)))

    block.to(device).eval()
    orig_block.to(device).eval()

    with torch.no_grad():
        for (c_inp, kwargs), (f_inp, kwargs_orig) in zip(block_inputs, block_inputs_orig):
            # Forward compressed branch.
            inp = c_inp.to(device)
            dev_kwargs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in kwargs.items()
            }
            block(inp, **dev_kwargs)

            # Forward original-model branch.
            inp_orig = f_inp.to(device)
            dev_kwargs_orig = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in kwargs_orig.items()
            }
            orig_block(inp_orig, **dev_kwargs_orig)

            # If both hooks fired for the same layer, update H and EP cross term.
            for name in subset:
                if name in hook_data and name in hook_data_orig:
                    h_accs[name].add(hook_data[name])
                    dh_accs[name].add(hook_data_orig[name], hook_data[name])

            hook_data.clear()
            hook_data_orig.clear()

    for handle in handles:
        handle.remove()
    torch.cuda.empty_cache()
    return h_accs, dh_accs
