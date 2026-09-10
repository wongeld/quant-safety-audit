"""
quant_safety_audit -- reusable infrastructure for testing whether a
quantization method changes model behavior on a safety-relevant benchmark.

Extension points, each independent of the others:
- quantizers.py: how to produce a quantized checkpoint from a base model.
- loaders.py:    how to load and generate from a given quantization method.
- benchmarks.py: how to turn a benchmark's data into a list of prompts to run.
- judges.py:     how to score a model's response.

pipeline.py ties loaders + benchmarks + judges together. Swapping any piece
-- a new quantization method, a new benchmark, a new judge -- never requires
touching the others.
"""

from .quantizers import (
    BaseQuantizer, build_quantizer,
    BnbQuantizer, HqqQuantizer, GptqQuantizer, AwqQuantizer, AutoRoundQuantizer, GgufDownloader,
)
from .loaders import BaseModelLoader, build_loader, TransformersLoader, AwqLoader, HqqLoader, GgufLlamaCppLoader
from .benchmarks import BaseBenchmark, BenchmarkItem, MaskBenchmark, parse_proposition
from .judges import BaseJudge, JudgeResult, MaskOpenLLMJudge
from .pipeline import ExperimentRunner

__all__ = [
    "BaseQuantizer", "build_quantizer",
    "BnbQuantizer", "HqqQuantizer", "GptqQuantizer", "AwqQuantizer", "AutoRoundQuantizer", "GgufDownloader",
    "BaseModelLoader", "build_loader",
    "TransformersLoader", "GptqLoader", "AwqLoader", "HqqLoader", "GgufLlamaCppLoader",
    "BaseBenchmark", "BenchmarkItem", "MaskBenchmark", "parse_proposition",
    "BaseJudge", "JudgeResult", "MaskOpenLLMJudge",
    "ExperimentRunner", "compare_judges",
]
