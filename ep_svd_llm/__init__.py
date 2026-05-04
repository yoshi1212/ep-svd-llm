"""EP-SVD-LLM: Error-Propagation SVD for LLM Compression."""

__all__ = [
    "BaseCompressor",
    "CompressionResult",
    "SVDLLMCompressor",
    "EPSVDLLMCompressor",
    "LowRankLinear",
    "load_model_and_tokenizer",
]
__version__ = "0.1.0"


def __getattr__(name: str):
    if name in {"BaseCompressor", "CompressionResult"}:
        from ep_svd_llm.core.base_compressor import BaseCompressor, CompressionResult

        return {"BaseCompressor": BaseCompressor, "CompressionResult": CompressionResult}[name]
    if name == "SVDLLMCompressor":
        from ep_svd_llm.core.svd_llm import SVDLLMCompressor

        return SVDLLMCompressor
    if name == "EPSVDLLMCompressor":
        from ep_svd_llm.core.ep_svd_llm import EPSVDLLMCompressor

        return EPSVDLLMCompressor
    if name in {"LowRankLinear", "load_model_and_tokenizer"}:
        from ep_svd_llm.models.loader import LowRankLinear, load_model_and_tokenizer

        return {"LowRankLinear": LowRankLinear, "load_model_and_tokenizer": load_model_and_tokenizer}[name]
    raise AttributeError(f"module 'ep_svd_llm' has no attribute {name!r}")
