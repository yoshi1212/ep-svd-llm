"""
Model loading utilities and LowRankLinear layer.
"""

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass, field
from typing import Tuple, Optional, List


# ---------------------------------------------------------------------------
# Configuration dataclass (replaces the old src/config.py dependency)
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Configuration for a model to load and compress."""

    name: str
    """HuggingFace model name or local path."""

    torch_dtype: str = "float16"
    """Torch dtype string, e.g. 'float16', 'bfloat16', 'float32'."""

    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"]
    )
    """List of module name substrings identifying linear layers to compress."""


# ---------------------------------------------------------------------------
# Model / Tokenizer loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    model_name_or_path: str,
    torch_dtype: str = "float16",
    device: str = "cuda",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a causal LM and its tokenizer.

    Args:
        model_name_or_path: HuggingFace model name or local directory.
        torch_dtype: Dtype string ('float16', 'bfloat16', 'float32').
        device: Target device when not using device_map.
        load_in_8bit: Load in 8-bit quantisation (saves VRAM).
        load_in_4bit: Load in 4-bit quantisation (saves more VRAM).

    Returns:
        (model, tokenizer) tuple.
    """
    print(f"Loading model: {model_name_or_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict = {
        "trust_remote_code": True,
        "torch_dtype": getattr(torch, torch_dtype),
    }

    if load_in_8bit:
        load_kwargs["load_in_8bit"] = True
        load_kwargs["device_map"] = "auto"
    elif load_in_4bit:
        load_kwargs["load_in_4bit"] = True
        load_kwargs["device_map"] = "auto"
    # 8bit/4bit 以外は device_map を使わず .to(device) で移動。
    # device_map を指定すると accelerate が必要になるため。

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)
    if not load_in_8bit and not load_in_4bit:
        model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded. Parameters: {n_params:,}")

    return model, tokenizer


# ---------------------------------------------------------------------------
# Transformer Block helpers
# ---------------------------------------------------------------------------

# Keywords for layers that should NOT be compressed by default.
_DEFAULT_EXCLUDE_KEYWORDS: List[str] = ["lm_head", "embed_out"]

# Heuristic paths to the list of decoder blocks for common architectures.
# Each entry is a dot-separated attribute path from the model root.
_DECODER_BLOCK_PATHS = [
    "model.layers",            # LLaMA, Mistral, Gemma, Qwen2, …
    "model.decoder.layers",    # OPT
    "gpt_neox.layers",         # Pythia / GPT-NeoX
    "transformer.h",           # GPT-2 / Falcon
    "model.blocks",            # MPT
]

_EMBED_NORM_PATHS = {
    "model.layers": {
        "embed": ["model.embed_tokens"],
        "norm":  ["model.norm"],
    },
    "model.decoder.layers": {
        "embed": ["model.decoder.embed_tokens", "model.decoder.embed_positions",
                  "model.decoder.final_layer_norm"],
        "norm":  [],
    },
    "gpt_neox.layers": {
        "embed": ["gpt_neox.embed_in"],
        "norm":  ["gpt_neox.final_layer_norm"],
    },
    "transformer.h": {
        "embed": ["transformer.wte", "transformer.wpe"],
        "norm":  ["transformer.ln_f"],
    },
    "model.blocks": {
        "embed": ["model.wte"],
        "norm":  ["model.norm_f"],
    },
}


def _get_attr(model: nn.Module, path: str):
    """Traverse a dot-separated attribute path from model root."""
    obj = model
    for part in path.split("."):
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


def get_decoder_blocks(model: nn.Module):
    """
    Return ``(blocks, embed_modules, norm_modules)`` for a causal LM.

    - *blocks*: ``nn.ModuleList`` of Transformer decoder blocks (layers).
    - *embed_modules*: list of ``nn.Module`` objects that must be on-device
      to compute the first block's input (embeddings, positional embeddings…).
    - *norm_modules*: list of ``nn.Module`` objects applied after the last
      block (final LayerNorm, …).

    Raises ``ValueError`` if the architecture cannot be auto-detected.
    """
    for path in _DECODER_BLOCK_PATHS:
        try:
            blocks = _get_attr(model, path)
            if isinstance(blocks, nn.ModuleList) and len(blocks) > 0:
                meta = _EMBED_NORM_PATHS[path]
                embeds, norms = [], []
                for ep in meta["embed"]:
                    try:
                        embeds.append(_get_attr(model, ep))
                    except AttributeError:
                        pass
                for np_ in meta["norm"]:
                    try:
                        norms.append(_get_attr(model, np_))
                    except AttributeError:
                        pass
                return blocks, embeds, norms
        except AttributeError:
            continue

    raise ValueError(
        "Cannot auto-detect decoder blocks. "
        "Supported architectures: LLaMA/Mistral/Gemma, OPT, Pythia/GPT-NeoX, GPT-2/Falcon, MPT. "
        "Pass the block list manually if your architecture differs."
    )


def find_layers_in_block(block: nn.Module, target_modules=None, exclude_keywords=None):
    """
    Return {relative_name: nn.Linear} for linear layers inside a single
    decoder block.  Uses the same filter logic as :func:`get_linear_layers`.
    """
    _exclude = exclude_keywords if exclude_keywords is not None else _DEFAULT_EXCLUDE_KEYWORDS
    result = {}
    for name, module in block.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if target_modules is not None:
            if any(t in name for t in target_modules):
                result[name] = module
        else:
            if not any(ex in name for ex in _exclude):
                result[name] = module
    return result


def get_sequential_groups(layer_names: List[str]) -> List[List[str]]:
    """
    Groups layer names into a standard execution order for Transformer blocks.
    This is used for EP-SVD-LLM sequential error propagation.

    Expected order:
    1. Attention inputs (Q, K, V)
    2. Attention outputs (O)
    3. MLP inputs (Up, Gate, etc.)
    4. MLP outputs (Down, etc.)

    Args:
        layer_names: List of relative layer names found in a block.

    Returns:
        A list of groups (lists of strings), where each group contains
        the names of layers that can be processed in parallel.
    """
    groups = [[], [], [], []]
    leftovers = []

    for name in layer_names:
        name_lower = name.lower()
        
        # Group 1: Attention inputs (Q, K, V)
        if any(k in name_lower for k in ["q_proj", "k_proj", "v_proj", "query", "key", "value", "w_q", "w_k", "w_v", "c_attn"]):
            groups[0].append(name)
        # Group 2: Attention outputs (O, out_proj, dense in attn, c_proj in attn)
        elif any(k in name_lower for k in ["o_proj", "out_proj", "dense", "c_proj"]) and (
            "attn" in name_lower or "attention" in name_lower
        ):
            groups[1].append(name)
        # Group 3: MLP inputs (Up, Gate, W1, W3, FC1, dense_h_to_4h, c_fc)
        elif any(k in name_lower for k in ["gate_proj", "up_proj", "dense_h_to_4h", "c_fc", "w1", "w3", "fc1"]):
            groups[2].append(name)
        # Group 4: MLP outputs (Down, dense_4h_to_h, c_proj in mlp, W2, FC2)
        elif any(k in name_lower for k in ["down_proj", "dense_4h_to_h", "c_proj", "w2", "fc2"]) and (
            "mlp" in name_lower
            or "ffn" in name_lower
            or "fc2" in name_lower
            or "w2" in name_lower
        ):
            groups[3].append(name)
        else:
            leftovers.append(name)

    # Filter out empty groups, and append any leftovers as independent groups
    result = [g for g in groups if g]
    if leftovers:
        result.append(leftovers)
        
    return result


def get_linear_layers(
    model: nn.Module,
    target_modules: Optional[List[str]] = None,
    exclude_keywords: Optional[List[str]] = None,
) -> dict:
    """
    Return a dict of {layer_name: nn.Linear} for target linear layers.

    When *target_modules* is **None** (default), all ``nn.Linear`` layers are
    selected automatically, excluding those whose name contains a keyword from
    *exclude_keywords* (defaults to ``["lm_head", "embed_out"]`` to preserve
    the output projection and embedding layers).

    When *target_modules* is a list of strings, only layers whose name contains
    at least one of those strings are selected (legacy behaviour).

    Args:
        model: Loaded model.
        target_modules: Optional list of name substrings to include.
            ``None`` → auto-detect all linear layers (architecture-agnostic).
        exclude_keywords: Name substrings to always exclude from compression.
            Only used when *target_modules* is ``None``.

    Returns:
        Dict mapping layer name → nn.Linear module.
    """
    _exclude: List[str] = exclude_keywords if exclude_keywords is not None else _DEFAULT_EXCLUDE_KEYWORDS

    linear_layers = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if target_modules is not None:
            # Legacy: explicit name-based inclusion list
            if any(t in name for t in target_modules):
                linear_layers[name] = module
        else:
            # Auto-detect: all nn.Linear except excluded layers
            if not any(ex in name for ex in _exclude):
                linear_layers[name] = module

    mode = "auto-detect" if target_modules is None else "name-filter"
    print(f"Found {len(linear_layers)} target linear layers [{mode}]")
    return linear_layers


def get_layer_by_name(model: nn.Module, layer_name: str) -> nn.Module:
    """
    Retrieve a submodule by its dot-separated name.

    Args:
        model: Root module.
        layer_name: E.g. "model.layers.0.mlp.gate_proj".

    Returns:
        The corresponding nn.Module.
    """
    parts = layer_name.split(".")
    module = model
    for part in parts:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


def set_layer_by_name(model: nn.Module, layer_name: str, new_layer: nn.Module) -> None:
    """
    Replace a submodule identified by its dot-separated name.

    Args:
        model: Root module.
        layer_name: Dot-separated path to the target submodule.
        new_layer: Replacement module.
    """
    parts = layer_name.split(".")
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    setattr(parent, parts[-1], new_layer)


# ---------------------------------------------------------------------------
# LowRankLinear
# ---------------------------------------------------------------------------

class LowRankLinear(nn.Module):
    """
    A linear layer represented as a low-rank factorisation W' = W_u @ W_v.

    Replaces a standard nn.Linear after SVD compression.
    """

    def __init__(
        self,
        W_u: torch.Tensor,
        W_v: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            W_u: Left factor (d_out, rank).
            W_v: Right factor (rank, d_in).
            bias: Optional bias term from the original layer.
        """
        super().__init__()
        self.W_u = nn.Parameter(W_u)
        self.W_v = nn.Parameter(W_v)

        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.register_parameter("bias", None)

        self.in_features = W_v.shape[1]
        self.out_features = W_u.shape[0]
        self.rank = W_u.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_in)  →  out: (..., d_out)
        out = x @ self.W_v.T   # (..., rank)
        out = out @ self.W_u.T  # (..., d_out)
        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self) -> str:
        return (
            f"LowRankLinear(in={self.in_features}, "
            f"out={self.out_features}, rank={self.rank})"
        )


# ---------------------------------------------------------------------------
# Save/Load helpers
# ---------------------------------------------------------------------------

def merge_low_rank_layers(model: nn.Module) -> nn.Module:
    """
    Convert all :class:`LowRankLinear` layers in *model* back to
    ``nn.Linear`` by computing the full weight matrix ``W = W_u @ W_v``.

    This is required before calling ``model.save_pretrained()`` because
    HuggingFace's loader expects standard ``nn.Linear`` modules with a
    ``.weight`` key in the state dict.

    .. note::
        The resulting ``nn.Linear`` has the same shape as the original layer,
        but its weights represent the **low-rank approximation** produced by
        SC-SVD-LLM / EP-SVD-LLM.  The parameter count is identical to the
        original model, but the weight *quality* (approximation error) still
        reflects the compression — making this form suitable for perplexity
        benchmarking and for distributing the compressed checkpoint in a
        standard HuggingFace-compatible format.

    Args:
        model: Model whose ``LowRankLinear`` layers will be merged in-place.

    Returns:
        The same model object (modified in-place).
    """
    to_merge = [
        name
        for name, module in model.named_modules()
        if isinstance(module, LowRankLinear)
    ]

    for name in to_merge:
        layer: LowRankLinear = get_layer_by_name(model, name)  # type: ignore
        # W = (d_out, rank) @ (rank, d_in) → (d_out, d_in)
        W = (layer.W_u @ layer.W_v).detach().to(layer.W_u.dtype)
        new_linear = nn.Linear(
            in_features=layer.in_features,
            out_features=layer.out_features,
            bias=layer.bias is not None,
            device=W.device,
            dtype=W.dtype,
        )
        new_linear.weight.data = W
        if layer.bias is not None:
            new_linear.bias.data = layer.bias.data.clone()
        set_layer_by_name(model, name, new_linear)

    if to_merge:
        print(f"Merged {len(to_merge)} LowRankLinear → nn.Linear layers")
    return model
