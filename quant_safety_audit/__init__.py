"""
quant_safety_audit -- reusable infrastructure for testing whether a
quantization method changes model behavior on a safety-relevant benchmark.

Three extension points, each independent of the other two:
- loaders.py:    how to load and generate from a given quantization method.
- benchmarks.py: how to turn a benchmark's data into a list of prompts to run.
- judges.py:     how to score a model's response.

pipeline.py ties them together. Swapping any one piece -- a new quantization
method, a new benchmark, a new judge -- never requires touching the other two.
"""

from .loaders import BaseModelLoader, build_loader
from .benchmarks import BaseBenchmark, BenchmarkItem, MaskBenchmark, parse_proposition
from .judges import BaseJudge, JudgeResult, MaskOpenLLMJudge
from .pipeline import ExperimentRunner

__all__ = [
    "BaseModelLoader", "build_loader",
    "BaseBenchmark", "BenchmarkItem", "MaskBenchmark", "parse_proposition",
    "BaseJudge", "JudgeResult", "MaskOpenLLMJudge",
    "ExperimentRunner",
]
