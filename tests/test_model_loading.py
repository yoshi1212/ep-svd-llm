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
    def test_prefers_dtype_keyword(self, monkeypatch):
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
            dtype=torch.float16,
            device="cpu",
        )

        assert "dtype" in captured
        assert captured["dtype"] is torch.float16
        assert "torch_dtype" not in captured

    def test_falls_back_to_torch_dtype_for_older_api(self, monkeypatch):
        calls = []

        def fake_tokenizer_from_pretrained(*args, **kwargs):
            return _DummyTokenizer()

        def fake_model_from_pretrained(*args, **kwargs):
            calls.append(kwargs.copy())
            if "dtype" in kwargs:
                raise TypeError("from_pretrained() got an unexpected keyword argument 'dtype'")
            return _DummyModel()

        monkeypatch.setattr(loader.AutoTokenizer, "from_pretrained", fake_tokenizer_from_pretrained)
        monkeypatch.setattr(loader.AutoModelForCausalLM, "from_pretrained", fake_model_from_pretrained)

        loader.load_model_and_tokenizer(
            "dummy/model",
            dtype=torch.float16,
            device="cpu",
        )

        assert len(calls) == 2
        assert "dtype" in calls[0]
        assert "torch_dtype" not in calls[0]
        assert "torch_dtype" in calls[1]
        assert calls[1]["torch_dtype"] is torch.float16
