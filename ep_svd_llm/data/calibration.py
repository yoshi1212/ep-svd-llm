"""Calibration dataset preparation utilities."""

import numpy as np


def prepare_calibration_data(
    tokenizer,
    dataset_name,
    dataset_config,
    split,
    num_samples,
    seq_length,
    seed=42,
):
    """
    Prepare calibration data from a HuggingFace dataset.

    Tokenises the full text of the dataset, then randomly samples
    *num_samples* windows of *seq_length* tokens.

    Why this design:
    - We build one long token stream to sample contiguous windows efficiently.
    - Random start indices approximate diverse contexts for Hessian estimation.
    - A fixed seed keeps calibration deterministic for regression checks.
    """
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, dataset_config, split=split)
    texts = [t for t in dataset["text"] if t and len(t.strip()) > 0]
    full_text = "\n\n".join(texts)
    encodings = tokenizer(full_text, return_tensors="pt")
    input_ids = encodings.input_ids[0]

    np.random.seed(seed)
    max_start = len(input_ids) - seq_length
    if max_start <= 0:
        raise ValueError(
            f"Dataset too short ({len(input_ids)} tokens) for seq_length={seq_length}"
        )
    start_indices = np.random.choice(max_start, size=num_samples, replace=False)

    samples = [input_ids[s : s + seq_length].unsqueeze(0) for s in sorted(start_indices)]
    print(
        f"Prepared {len(samples)} calibration samples "
        f"(seq_length={seq_length}, dataset={dataset_name}/{dataset_config}/{split})"
    )
    return samples
