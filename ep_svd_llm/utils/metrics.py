"""
Evaluation metrics for compressed LLMs.
"""

import torch
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizer
from datasets import load_dataset
from tqdm import tqdm
from typing import Optional, Dict, Any
import math


def compute_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "test",
    max_length: int = 2048,
    stride: int = 512,
    device: str = "cuda",
    max_samples: Optional[int] = None,
) -> float:
    """
    Compute perplexity using a sliding-window approach.

    Args:
        model: Model to evaluate.
        tokenizer: Corresponding tokenizer.
        dataset_name: HuggingFace dataset name.
        dataset_config: Dataset configuration string.
        split: Dataset split to use.
        max_length: Context window length.
        stride: Sliding window stride.
        device: Device for inference.
        max_samples: Cap the number of tokens evaluated (for quick testing).

    Returns:
        Perplexity value.
    """
    model.eval()

    dataset = load_dataset(dataset_name, dataset_config, split=split)

    if "text" in dataset.column_names:
        text = "\n\n".join(dataset["text"])
    else:
        text = "\n\n".join([str(x) for x in dataset[dataset.column_names[0]]])

    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)

    seq_len = input_ids.size(1)

    if max_samples is not None:
        seq_len = min(seq_len, max_samples * max_length)
        input_ids = input_ids[:, :seq_len]

    nlls = []
    prev_end_loc = 0

    pbar = tqdm(range(0, seq_len, stride), desc="Computing perplexity")

    for begin_loc in pbar:
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc

        input_chunk = input_ids[:, begin_loc:end_loc]
        target_ids = input_chunk.clone()
        target_ids[:, :-trg_len] = -100  # mask sliding portion

        with torch.no_grad():
            outputs = model(input_chunk, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood.item() * trg_len)
        prev_end_loc = end_loc

        if end_loc >= seq_len:
            break

        pbar.set_postfix({"current_ppl": math.exp(sum(nlls) / end_loc)})

    total_nll = sum(nlls)
    total_tokens = prev_end_loc
    ppl = math.exp(total_nll / total_tokens)
    return ppl


def compute_layer_reconstruction_error(
    W_original: torch.Tensor,
    W_compressed,
    X: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute layer-wise reconstruction error.

    Args:
        W_original: Original weight matrix (d_out, d_in).
        W_compressed: Compressed weight — either a full matrix (d_out, d_in)
                      or a (W_u, W_v) tuple representing W' = W_u @ W_v.
        X: Input activations (d_in, N).

    Returns:
        Dict with keys: "frobenius_error", "relative_error", "mean_absolute_error".
    """
    if isinstance(W_compressed, tuple):
        W_u, W_v = W_compressed
        W_compressed_full = W_u @ W_v
    else:
        W_compressed_full = W_compressed

    Y_original = W_original @ X       # (d_out, N)
    Y_compressed = W_compressed_full @ X  # (d_out, N)

    diff = Y_original - Y_compressed

    frob_error = torch.norm(diff, p='fro').item()
    relative_error = frob_error / (torch.norm(Y_original, p='fro').item() + 1e-8)
    mae = torch.mean(torch.abs(diff)).item()

    return {
        "frobenius_error": frob_error,
        "relative_error": relative_error,
        "mean_absolute_error": mae,
    }


def print_gpu_memory() -> None:
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
