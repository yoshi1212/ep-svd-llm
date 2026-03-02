"""
Unit tests for model-loading keyword arguments.

These tests mock HuggingFace loaders, so they run without downloading models.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from ep_svd_llm.models import loader


class _DummyTokenizer:
    pad_token = None
    eos_token = "</s>"


class _DummyModel:
    def __init__(self):
        self.moved_to = None

    def to(self, device):
        self.moved_to = device
        return self

    def eval(self):
        return self

    def parameters(self):
        return []


class TestLoadModelAndTokenizer:
    def test_uses_torch_dtype_keyword(self, monkeypatch):
        captured = {}

        def fake_tokenizer_from_pretrained(*args, **kwargs):
            return _DummyTokenizer()

        def fake_model_from_pretrained(*args, **kwargs):
            captured.update(kwargs)
            return _DummyModel()

        monkeypatch.setattr(loader.AutoTokenizer, "from_pretrained", fake_tokenizer_from_pretrained)
        monkeypatch.setattr(loader.AutoModelForCausalLM, "from_pretrained", fake_model_from_pretrained)

        loader.load_model_and_tokenizer(
            "dummy/model",
            torch_dtype="float16",
            device="cpu",
        )

        assert "torch_dtype" in captured
        assert captured["torch_dtype"] is torch.float16
        assert "dtype" not in captured
