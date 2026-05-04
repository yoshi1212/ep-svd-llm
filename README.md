# EP-SVD-LLM

**Error-Propagation SVD for Large Language Model Compression**

EP-SVD-LLM is a layer-wise SVD compression method for LLMs that improves upon [SVD-LLM](https://arxiv.org/abs/2403.07378) by explicitly propagating and compensating accumulated compression errors across layers.

> **Note on naming**: This package implements **SC-SVD-LLM** (Sequentially-Compressed SVD-LLM), which
> processes layers sequentially using compressed activations X̂, and **EP-SVD-LLM**, which additionally
> compensates accumulated errors. The original **SVD-LLM** paper uses FP activations X for all Hessians.
>
> **Note on "Hessian"**: Throughout this codebase, XX⊤ is referred to as the "Hessian". This terminology
> comes from the quantization literature (e.g. [GPTQ](https://arxiv.org/abs/2210.17323)), where the
> Hessian of the layer-wise compression error ‖WX − W′X‖² w.r.t. W′ equals 2XX⊤. The SVD-LLM papers
> themselves do not use this term.

## Method

Standard SC-SVD-LLM applies Truncation-Aware Data Whitening ([SVD-LLM](https://arxiv.org/abs/2403.07378); numerically stabilised via SVD-based decomposition from [V2](https://arxiv.org/abs/2503.12340)) to each layer sequentially, using compressed activations.  
EP-SVD-LLM adds an _error propagation_ (EP) step based on [QEP](https://arxiv.org/abs/2504.09629):

1. Track the accumulated activation error **δ = X_fp − X̂** between the full-precision model and the (partially) compressed model.
2. Compute a correction term: **correction = W δ X̂ᵀ Ĥ⁻¹**
3. Apply it to the weight before SVD: **W\* = W + α · correction**
4. Run SC-SVD-LLM whitening + truncated SVD on **W\***.

Setting `alpha=0` recovers plain SC-SVD-LLM.

## Installation

```bash
pip install -e .
```

**Requirements**: Python ≥ 3.10, PyTorch ≥ 2.0, Transformers ≥ 4.35

### GPU / CUDA setup

This package depends on `torch`, but it does not force a specific CUDA build.
If you want GPU acceleration, install a CUDA-enabled PyTorch build for your
environment before installing or syncing this project.

Examples:

```bash
# pip: install a CUDA build first, then install this project
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -e .
```

```bash
# uv: install a CUDA build first, then sync/install the project
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
uv sync
```

Use the PyTorch install selector to choose the correct command for your OS,
Python version, and CUDA runtime:

- [PyTorch Get Started](https://pytorch.org/get-started/locally/)

You can verify that GPU support is available with:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

For algorithm details and derivations, see the accompanying article / paper-style write-up outside this repository.

## Quick Start

### Fastest tutorial run (single script)

```bash
python runners/run_tinyllama_tutorial.py
ls -t results/ppl_*.json | head -n 6
```

This tutorial compares `svd_llm` and `ep_svd_llm` on TinyLlama at ratios `0.2, 0.4, 0.6, 0.8` and stores PPL results as JSON.

The tutorial uses `--no-save`, so the compressed layers remain in `LowRankLinear`
form during evaluation and no Hugging Face checkpoint is written.

### Compress a model

```bash
# SVD-LLM (original paper: FP activations for Hessians)
python scripts/compress_model.py \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --method svd_llm \
    --compression-ratio 0.2 \
    --output models/tinyllama_svd_llm

# SC-SVD-LLM (sequential compression with compressed activations)
python scripts/compress_model.py \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --method sc_svd_llm \
    --compression-ratio 0.2 \
    --output models/tinyllama_sc_svd_llm

# EP-SVD-LLM (error propagation compensation, default alpha=0.5)
python scripts/compress_model.py \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --method ep_svd_llm \
    --compression-ratio 0.2 \
    --alpha 0.5 \
    --save-format svd \
    --output models/tinyllama_ep_svd_llm
```

The default save format is `hf`, which converts `LowRankLinear` layers back into
standard `nn.Linear` weights before `save_pretrained()`. This produces a
Hugging Face-compatible checkpoint for evaluation and distribution, but it does
not preserve the low-rank parameterisation on disk.

For post-compression fine-tuning, use `--save-format svd` or `--save-format
low_rank` so the compressed low-rank structure can be restored later. For
eval-only runs, add `--no-save` to keep the low-rank modules in memory and skip
disk output.

### Evaluate (perplexity)

```bash
python scripts/evaluate_model.py \
    --model-path models/tinyllama_ep_svd_llm \
    --dataset wikitext2
```

### Post-compression fine-tuning

```bash
python scripts/post_compression_finetune.py \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --compressed-svd-model models/tinyllama_ep_svd_llm/svd_factors.pt \
    --compression-method ep_svd_llm \
    --compression-ratio 0.2 \
    --strategies pissa \
    --steps 40
```

This command runs continued causal-LM training after compression. It is not
downstream task fine-tuning; the objective remains next-token prediction, and
the goal is to adapt or recover the compressed language model.

W&B logging is optional and is enabled only when `--use-wandb` is passed.

### Run tests

```bash
pytest -q
```

## Python API

```python
from ep_svd_llm import EPSVDLLMCompressor, LowRankLinear, load_model_and_tokenizer

compressor = EPSVDLLMCompressor(alpha=1.0, device="cuda")
result = compressor.compress_layer(
    weight,
    activations_compressed,
    target_rank=64,
    activations_orig=activations_fp,
)

low_rank = LowRankLinear(result.W_u, result.W_v, bias=original_layer.bias)
```

## Package Structure

```
ep_svd_llm/
├── core/
│   ├── base_compressor.py   # BaseCompressor (abstract), CompressionResult
│   ├── svd_llm.py           # SVDLLMCompressor (X→SVD-LLM / X̂→SC-SVD-LLM)
│   ├── pipeline.py          # SequentialCompressionPipeline
│   └── ep_svd_llm.py        # EPSVDLLMCompressor
├── data/
│   └── calibration.py       # prepare_calibration_data
├── finetune/
│   ├── helpers.py           # lightweight CLI / logging helpers
│   └── post_compression.py  # post-compression causal-LM adaptation
├── utils/
│   ├── activation.py        # ActivationCollector, HessianAccumulator, DeltaHessianAccumulator
│   └── metrics.py           # compute_perplexity, compute_layer_reconstruction_error
└── models/
    └── loader.py            # LowRankLinear, load_model_and_tokenizer, ...
```

## Reference

- [SVD-LLM (Wang et al., 2024)](https://arxiv.org/abs/2403.07378) — Truncation-aware SVD for LLM compression
- [SVD-LLM V2 (Wang et al., 2024)](https://arxiv.org/abs/2503.12340) — Dynamic rank allocation & numerically stable whitening
- [QEP (Arai & Ichikawa, 2025)](https://arxiv.org/abs/2504.09629) — Error propagation compensation (α-weighted correction)
- [GPTQ (Frantar et al., 2022)](https://arxiv.org/abs/2210.17323) — Post-training quantization; origin of "Hessian" terminology for XX⊤
