"""
Model loading utilities and LowRankLinear layer.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ModelConfig:
    name: str
    dtype: Union[torch.dtype, str] = torch.float16
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


def load_model_and_tokenizer(
    model_name_or_path: str,
    dtype: Union[torch.dtype, str] = torch.float16,
    device: str = "cuda",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    print(f"Loading model: {model_name_or_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    requested_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    load_kwargs: dict = {
        "trust_remote_code": True,
        "dtype": requested_dtype,
    }

    if load_in_8bit:
        load_kwargs["load_in_8bit"] = True
        load_kwargs["device_map"] = "auto"
    elif load_in_4bit:
        load_kwargs["load_in_4bit"] = True
        load_kwargs["device_map"] = "auto"

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        fallback_kwargs = dict(load_kwargs)
        fallback_kwargs.pop("dtype", None)
        fallback_kwargs["torch_dtype"] = requested_dtype
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **fallback_kwargs)

    if not load_in_8bit and not load_in_4bit:
        model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded. Parameters: {n_params:,}")
    return model, tokenizer


_DEFAULT_EXCLUDE_KEYWORDS: List[str] = ["lm_head", "embed_out"]
_DECODER_BLOCK_PATHS = [
    "model.layers",
    "model.decoder.layers",
    "gpt_neox.layers",
    "transformer.h",
    "model.blocks",
]
_EMBED_NORM_PATHS = {
    "model.layers": {
        "embed": ["model.embed_tokens"],
        "norm": ["model.norm"],
    },
    "model.decoder.layers": {
        "embed": [
            "model.decoder.embed_tokens",
            "model.decoder.embed_positions",
            "model.decoder.final_layer_norm",
        ],
        "norm": [],
    },
    "gpt_neox.layers": {
        "embed": ["gpt_neox.embed_in"],
        "norm": ["gpt_neox.final_layer_norm"],
    },
    "transformer.h": {
        "embed": ["transformer.wte", "transformer.wpe"],
        "norm": ["transformer.ln_f"],
    },
    "model.blocks": {
        "embed": ["model.wte"],
        "norm": ["model.norm_f"],
    },
}


def _get_attr(model: nn.Module, path: str):
    obj = model
    for part in path.split("."):
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


def get_decoder_blocks(model: nn.Module):
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
        "Cannot auto-detect decoder blocks. Supported architectures: "
        "LLaMA/Mistral/Gemma, OPT, Pythia/GPT-NeoX, GPT-2/Falcon, MPT."
    )


def find_layers_in_block(block: nn.Module, target_modules=None, exclude_keywords=None):
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
    groups = [[], [], [], []]
    leftovers = []

    for name in layer_names:
        name_lower = name.lower()
        if any(
            k in name_lower
            for k in ["q_proj", "k_proj", "v_proj", "query", "key", "value", "w_q", "w_k", "w_v", "c_attn"]
        ):
            groups[0].append(name)
        elif any(k in name_lower for k in ["o_proj", "out_proj", "dense", "c_proj"]) and (
            "attn" in name_lower or "attention" in name_lower
        ):
            groups[1].append(name)
        elif any(k in name_lower for k in ["gate_proj", "up_proj", "dense_h_to_4h", "c_fc", "w1", "w3", "fc1"]):
            groups[2].append(name)
        elif any(k in name_lower for k in ["down_proj", "dense_4h_to_h", "c_proj", "w2", "fc2"]) and (
            "mlp" in name_lower or "ffn" in name_lower or "fc2" in name_lower or "w2" in name_lower
        ):
            groups[3].append(name)
        else:
            leftovers.append(name)

    result = [g for g in groups if g]
    if leftovers:
        result.append(leftovers)
    return result


def get_linear_layers(
    model: nn.Module,
    target_modules: Optional[List[str]] = None,
    exclude_keywords: Optional[List[str]] = None,
) -> dict:
    _exclude: List[str] = exclude_keywords if exclude_keywords is not None else _DEFAULT_EXCLUDE_KEYWORDS
    linear_layers = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if target_modules is not None:
            if any(t in name for t in target_modules):
                linear_layers[name] = module
        else:
            if not any(ex in name for ex in _exclude):
                linear_layers[name] = module

    mode = "auto-detect" if target_modules is None else "name-filter"
    print(f"Found {len(linear_layers)} target linear layers [{mode}]")
    return linear_layers


def get_layer_by_name(model: nn.Module, layer_name: str) -> nn.Module:
    parts = layer_name.split(".")
    module = model
    for part in parts:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


def set_layer_by_name(model: nn.Module, layer_name: str, new_layer: nn.Module) -> None:
    parts = layer_name.split(".")
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    setattr(parent, parts[-1], new_layer)


class LowRankLinear(nn.Module):
    def __init__(
        self,
        W_u: torch.Tensor,
        W_v: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
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
        out = x @ self.W_v.T
        out = out @ self.W_u.T
        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self) -> str:
        return f"LowRankLinear(in={self.in_features}, out={self.out_features}, rank={self.rank})"


class FactoredTrainLowRankLinear(nn.Module):
    def __init__(
        self,
        U_train: torch.Tensor,
        V_train: torch.Tensor,
        U_fixed: torch.Tensor,
        V_fixed: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        adapter_dtype = torch.float32

        if U_train.ndim != 2 or V_train.ndim != 2:
            raise ValueError("U_train and V_train must be rank-2 tensors.")
        if U_fixed.ndim != 2 or V_fixed.ndim != 2:
            raise ValueError("U_fixed and V_fixed must be rank-2 tensors.")
        if U_train.shape[1] != V_train.shape[0]:
            raise ValueError("U_train.shape[1] must match V_train.shape[0].")
        if U_fixed.shape[1] != V_fixed.shape[0]:
            raise ValueError("U_fixed.shape[1] must match V_fixed.shape[0].")
        if U_train.shape[0] != U_fixed.shape[0]:
            raise ValueError("U_train and U_fixed must have the same number of rows.")
        if V_train.shape[1] != V_fixed.shape[1]:
            raise ValueError("V_train and V_fixed must have the same number of columns.")

        self.U_train = nn.Parameter(U_train.detach().clone().to(adapter_dtype))
        self.V_train = nn.Parameter(V_train.detach().clone().to(adapter_dtype))
        self.register_buffer("U_fixed", U_fixed.detach().clone())
        self.register_buffer("V_fixed", V_fixed.detach().clone())

        if bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", bias.detach().clone())

        self.in_features = V_train.shape[1]
        self.out_features = U_train.shape[0]
        self.train_rank = U_train.shape[1]
        self.fixed_rank = U_fixed.shape[1]
        self.rank = self.train_rank + self.fixed_rank

    @property
    def weight(self) -> torch.Tensor:
        out_dtype = self.U_fixed.dtype
        train_part = (self.U_train @ self.V_train).to(out_dtype)
        fixed_part = self.U_fixed @ self.V_fixed
        return train_part + fixed_part

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fixed_out = F.linear(F.linear(x, self.V_fixed), self.U_fixed)
        if self.bias is not None:
            fixed_out = fixed_out + self.bias
        train_x = x.to(self.V_train.dtype)
        train_out = F.linear(F.linear(train_x, self.V_train), self.U_train)
        return fixed_out + train_out.to(fixed_out.dtype)

    def to_low_rank_linear(self) -> "LowRankLinear":
        out_dtype = self.U_fixed.dtype
        W_u = torch.cat([self.U_train.detach().to(out_dtype), self.U_fixed], dim=1)
        W_v = torch.cat([self.V_train.detach().to(out_dtype), self.V_fixed], dim=0)
        bias = None if self.bias is None else self.bias.detach().clone()
        return LowRankLinear(W_u, W_v, bias=bias)


class SequentialLowRankUpdateLinear(nn.Module):
    def __init__(
        self,
        W_u: torch.Tensor,
        W_v: torch.Tensor,
        train_rank: int,
        stage: str = "u",
        bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        adapter_dtype = torch.float32
        if W_u.ndim != 2 or W_v.ndim != 2:
            raise ValueError("W_u and W_v must both be rank-2 tensors.")
        if W_u.shape[1] != W_v.shape[0]:
            raise ValueError("W_u.shape[1] must match W_v.shape[0].")
        if train_rank < 1 or train_rank > W_u.shape[1]:
            raise ValueError(f"train_rank must be in [1, {W_u.shape[1]}], got {train_rank}.")
        if stage not in {"u", "v"}:
            raise ValueError(f"stage must be 'u' or 'v', got {stage}.")

        self.register_buffer("W_u_base", W_u.detach().clone())
        self.register_buffer("W_v_base", W_v.detach().clone())
        if bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", bias.detach().clone())

        self.in_features = W_v.shape[1]
        self.out_features = W_u.shape[0]
        self.rank = W_u.shape[1]
        self.train_rank = train_rank
        self.stage = stage

        if stage == "u":
            self.A_u = nn.Parameter(torch.randn(train_rank, self.rank, dtype=adapter_dtype, device=W_u.device) * 0.01)
            self.B_u = nn.Parameter(torch.zeros(self.out_features, train_rank, dtype=adapter_dtype, device=W_u.device))
            self.register_parameter("A_v", None)
            self.register_parameter("B_v", None)
        else:
            self.register_parameter("A_u", None)
            self.register_parameter("B_u", None)
            self.A_v = nn.Parameter(torch.randn(train_rank, self.in_features, dtype=adapter_dtype, device=W_v.device) * 0.01)
            self.B_v = nn.Parameter(torch.zeros(self.rank, train_rank, dtype=adapter_dtype, device=W_v.device))

    @property
    def effective_W_u(self) -> torch.Tensor:
        if self.stage == "u":
            return self.W_u_base.to(self.B_u.dtype) + (self.B_u @ self.A_u)
        return self.W_u_base

    @property
    def effective_W_v(self) -> torch.Tensor:
        if self.stage == "v":
            return self.W_v_base.to(self.B_v.dtype) + (self.B_v @ self.A_v)
        return self.W_v_base

    @property
    def weight(self) -> torch.Tensor:
        return self.effective_W_u @ self.effective_W_v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compute_dtype = torch.float32
        w_v = self.effective_W_v.to(compute_dtype)
        w_u = self.effective_W_u.to(compute_dtype)
        out = x.to(compute_dtype) @ w_v.T
        out = out @ w_u.T
        if self.bias is not None:
            out = out + self.bias.to(compute_dtype)
        return out.to(x.dtype)

    def advance_stage(self, next_stage: str = "v") -> "SequentialLowRankUpdateLinear":
        return SequentialLowRankUpdateLinear(
            W_u=self.effective_W_u.detach(),
            W_v=self.effective_W_v.detach(),
            train_rank=self.train_rank,
            stage=next_stage,
            bias=None if self.bias is None else self.bias.detach(),
        )


def convert_low_rank_linear(
    layer: LowRankLinear,
    train_rank: int,
    strategy: str = "pissa",
    sequential_stage: str = "u",
) -> Union[FactoredTrainLowRankLinear, SequentialLowRankUpdateLinear]:
    if not isinstance(layer, LowRankLinear):
        raise TypeError("layer must be an instance of LowRankLinear.")
    if train_rank < 1 or train_rank > layer.rank:
        raise ValueError(f"train_rank must be in [1, {layer.rank}], got {train_rank}.")

    W_u = layer.W_u.detach()
    W_v = layer.W_v.detach()
    bias = None if layer.bias is None else layer.bias.detach()

    if strategy == "pissa":
        return FactoredTrainLowRankLinear(
            U_train=W_u[:, :train_rank],
            V_train=W_v[:train_rank, :],
            U_fixed=W_u[:, train_rank:],
            V_fixed=W_v[train_rank:, :],
            bias=bias,
        )

    if strategy == "svd_llm_sequential":
        return SequentialLowRankUpdateLinear(
            W_u=W_u,
            W_v=W_v,
            train_rank=train_rank,
            stage=sequential_stage,
            bias=bias,
        )

    raise ValueError("strategy must be either 'pissa' or 'svd_llm_sequential'.")


def prepare_low_rank_finetuning(
    model: nn.Module,
    train_rank: int,
    strategy: str = "pissa",
    sequential_stage: str = "u",
) -> List[str]:
    for param in model.parameters():
        param.requires_grad = False

    converted_names: List[str] = []
    target_names = [name for name, module in model.named_modules() if isinstance(module, LowRankLinear)]
    for name in target_names:
        layer = get_layer_by_name(model, name)
        converted = convert_low_rank_linear(
            layer,
            train_rank,
            strategy=strategy,
            sequential_stage=sequential_stage,
        )
        set_layer_by_name(model, name, converted)
        converted_names.append(name)
    return converted_names


def prepare_full_finetuning(model: nn.Module) -> int:
    model.to(dtype=torch.float32)
    trainable_count = 0
    for param in model.parameters():
        param.requires_grad = True
        trainable_count += param.numel()
    return trainable_count


def advance_sequential_low_rank_finetuning(model: nn.Module, next_stage: str = "v") -> List[str]:
    advanced_names: List[str] = []
    target_names = [
        name for name, module in model.named_modules() if isinstance(module, SequentialLowRankUpdateLinear)
    ]
    for name in target_names:
        layer = get_layer_by_name(model, name)
        advanced = layer.advance_stage(next_stage)
        set_layer_by_name(model, name, advanced)
        advanced_names.append(name)
    return advanced_names


def merge_low_rank_layers(model: nn.Module) -> nn.Module:
    to_merge = [
        name
        for name, module in model.named_modules()
        if isinstance(module, (LowRankLinear, FactoredTrainLowRankLinear, SequentialLowRankUpdateLinear))
    ]

    for name in to_merge:
        layer = get_layer_by_name(model, name)
        if isinstance(layer, LowRankLinear):
            W = (layer.W_u @ layer.W_v).detach().to(layer.W_u.dtype)
            bias = None if layer.bias is None else layer.bias.detach().clone()
            in_features = layer.in_features
            out_features = layer.out_features
        elif isinstance(layer, FactoredTrainLowRankLinear):
            W = layer.weight.detach().to(layer.U_fixed.dtype)
            bias = None if layer.bias is None else layer.bias.detach().clone()
            in_features = layer.in_features
            out_features = layer.out_features
        else:
            W = layer.weight.detach().to(layer.W_u_base.dtype)
            bias = None if layer.bias is None else layer.bias.detach().clone()
            in_features = layer.in_features
            out_features = layer.out_features

        new_linear = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias is not None,
            device=W.device,
            dtype=W.dtype,
        )
        new_linear.weight.data = W
        if bias is not None:
            new_linear.bias.data = bias
        set_layer_by_name(model, name, new_linear)

    if to_merge:
        print(f"Merged {len(to_merge)} low-rank layers to nn.Linear layers")
    return model


def merge_to_low_rank_layers(model: nn.Module) -> nn.Module:
    to_merge = [
        name
        for name, module in model.named_modules()
        if isinstance(module, (FactoredTrainLowRankLinear, SequentialLowRankUpdateLinear))
    ]

    for name in to_merge:
        layer = get_layer_by_name(model, name)
        if isinstance(layer, FactoredTrainLowRankLinear):
            new_layer = layer.to_low_rank_linear()
        else:
            W_u = layer.effective_W_u.detach().to(layer.W_u_base.dtype)
            W_v = layer.effective_W_v.detach().to(layer.W_v_base.dtype)
            bias = None if layer.bias is None else layer.bias.detach().clone()
            new_layer = LowRankLinear(W_u, W_v, bias=bias)
        set_layer_by_name(model, name, new_layer)

    if to_merge:
        print(f"Merged {len(to_merge)} fine-tuning layers to LowRankLinear")
    return model


def _serialize_dtype(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _deserialize_dtype(dtype_name: str) -> torch.dtype:
    try:
        return getattr(torch, dtype_name)
    except AttributeError as exc:
        raise ValueError(f"Unsupported dtype name: {dtype_name}") from exc


def save_low_rank_state(model: nn.Module, path: str) -> None:
    import json
    from pathlib import Path as _Path

    save_path = _Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    state = {}
    metadata = {}
    for name, module in model.named_modules():
        if not isinstance(module, LowRankLinear):
            continue
        state[f"{name}.W_u"] = module.W_u.detach().cpu()
        state[f"{name}.W_v"] = module.W_v.detach().cpu()
        if module.bias is not None:
            state[f"{name}.bias"] = module.bias.detach().cpu()
        metadata[name] = {
            "in_features": module.in_features,
            "out_features": module.out_features,
            "rank": module.rank,
            "has_bias": module.bias is not None,
        }

    torch.save(state, str(save_path))
    save_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved low-rank state ({len(metadata)} layers) to {save_path}")


def save_svd_state(model: nn.Module, path: str) -> None:
    import json
    from pathlib import Path as _Path

    save_path = _Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    state = {}
    metadata = {}
    for name, module in model.named_modules():
        if not isinstance(module, LowRankLinear):
            continue

        weight = (module.W_u.detach().to(torch.float32) @ module.W_v.detach().to(torch.float32)).cpu()
        U, Sigma, Vh = torch.linalg.svd(weight, full_matrices=False)
        rank = module.rank
        U = U[:, :rank].clone()
        Sigma = Sigma[:rank].clone()
        Vh = Vh[:rank, :].clone()

        state[f"{name}.U"] = U
        state[f"{name}.Sigma"] = Sigma
        state[f"{name}.Vh"] = Vh
        if module.bias is not None:
            state[f"{name}.bias"] = module.bias.detach().cpu()
        metadata[name] = {
            "in_features": module.in_features,
            "out_features": module.out_features,
            "rank": rank,
            "has_bias": module.bias is not None,
            "factor_dtype": _serialize_dtype(module.W_u.dtype),
        }

    torch.save(state, str(save_path))
    save_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved SVD state ({len(metadata)} layers) to {save_path}")


def load_low_rank_state(model: nn.Module, path: str, device: str = "cpu") -> List[str]:
    import json
    from pathlib import Path as _Path

    save_path = _Path(path)
    state = torch.load(str(save_path), map_location=device, weights_only=True)
    metadata = json.loads(save_path.with_suffix(".json").read_text(encoding="utf-8"))

    replaced_names: List[str] = []
    for name in metadata:
        W_u = state[f"{name}.W_u"]
        W_v = state[f"{name}.W_v"]
        bias = state.get(f"{name}.bias", None)
        low_rank = LowRankLinear(W_u, W_v, bias=bias)
        set_layer_by_name(model, name, low_rank)
        replaced_names.append(name)

    print(f"Loaded low-rank state ({len(replaced_names)} layers) from {save_path}")
    return replaced_names


def load_svd_state(model: nn.Module, path: str, device: str = "cpu") -> List[str]:
    import json
    from pathlib import Path as _Path

    save_path = _Path(path)
    state = torch.load(str(save_path), map_location=device, weights_only=True)
    metadata = json.loads(save_path.with_suffix(".json").read_text(encoding="utf-8"))

    replaced_names: List[str] = []
    for name, info in metadata.items():
        U = state[f"{name}.U"]
        Sigma = state[f"{name}.Sigma"]
        Vh = state[f"{name}.Vh"]
        bias = state.get(f"{name}.bias", None)

        target_module = get_layer_by_name(model, name)
        target_dtype = (
            target_module.weight.dtype
            if hasattr(target_module, "weight")
            else _deserialize_dtype(info["factor_dtype"])
        )
        sqrt_sigma = torch.sqrt(Sigma.to(torch.float32))
        W_u = (U.to(torch.float32) * sqrt_sigma.unsqueeze(0)).to(target_dtype)
        W_v = (sqrt_sigma.unsqueeze(1) * Vh.to(torch.float32)).to(target_dtype)
        if bias is not None:
            bias = bias.to(target_dtype)

        low_rank = LowRankLinear(W_u, W_v, bias=bias)
        set_layer_by_name(model, name, low_rank)
        replaced_names.append(name)

    print(f"Loaded SVD state ({len(replaced_names)} layers) from {save_path}")
    return replaced_names
