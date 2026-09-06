"""
Benchmark adapters. Each benchmark's only job is to produce a list of
BenchmarkItem objects. Nothing about model loading or generation lives here
-- that's the point: swapping HarmBench for MASK for OR-Bench later means
writing one new BaseBenchmark subclass, not touching loaders.py or pipeline.py.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import re


@dataclass
class BenchmarkItem:
    item_id: str
    # Every item has a "primary" turn -- the one whose response actually gets
    # scored as the headline result. MASK-style benchmarks additionally carry
    # a belief-elicitation turn; single-turn benchmarks (HarmBench, OR-Bench)
    # simply leave belief_system/belief_user as None.
    primary_system: Optional[str]
    primary_user: str
    belief_system: Optional[str] = None
    belief_user: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


class BaseBenchmark(ABC):
    name: str

    @abstractmethod
    def load_items(self, n: int) -> List[BenchmarkItem]:
        ...


def parse_proposition(proposition: Optional[str]):
    """Copied from the official MASK evaluate.py -- extracts the two binary
    options from a '<pivotal>A/B</pivotal>' tag embedded in the sentence.
    Not reimplemented from a description: pulled from the source repo,
    because getting this wrong silently mis-scores every item."""
    if not proposition:
        return None, None
    match = re.search(r"<pivotal>([^<]+)</pivotal>", proposition)
    if not match:
        return None, None
    options = match.group(1).split("/")
    if len(options) != 2:
        return None, None
    prop1 = proposition.replace(f"<pivotal>{match.group(1)}</pivotal>", options[0])
    prop2 = proposition.replace(f"<pivotal>{match.group(1)}</pivotal>", options[1])
    return prop1, prop2


class MaskBenchmark(BaseBenchmark):
    """MASK's `known_facts` archetype by default -- single-turn, binary
    proposition, no numeric parsing, no multi-turn history to reconstruct.
    Other archetypes (statistics, doubling_down_*) carry extra fields this
    adapter doesn't populate (numeric ranges, prior-turn history) -- it will
    load them without erroring, but items may be missing data the official
    scoring needs. known_facts is the validated path; treat any other
    archetype as an extension to build, not a drop-in swap.
    """
    name = "mask"

    _VALIDATED_ARCHETYPES = {"known_facts", "disinformation", "continuations", "provided_facts"}

    def __init__(self, archetype: str = "known_facts"):
        self.archetype = archetype
        if archetype not in self._VALIDATED_ARCHETYPES:
            print(f"WARNING: archetype '{archetype}' isn't one of the validated single-turn "
                  f"archetypes {self._VALIDATED_ARCHETYPES}. It will likely load, but this "
                  f"adapter doesn't populate the extra fields (numeric ranges, multi-turn "
                  f"history) some archetypes need -- inspect a row before trusting results.")

    def load_items(self, n: int) -> List[BenchmarkItem]:
        from datasets import load_dataset
        import pandas as pd

        ds = load_dataset("cais/mask", self.archetype)
        df = pd.DataFrame(ds["test"]).head(n)
        if "belief_system_1" not in df.columns:
            df["belief_system_1"] = "You are a helpful assistant."

        items = []
        for i, row in df.iterrows():
            items.append(BenchmarkItem(
                item_id=f"mask_{i:03d}",
                primary_system=row["system_prompt"],
                primary_user=row["user_prompt"],
                belief_system=row["belief_system_1"],
                belief_user=row["belief_elicit_1"],
                meta={"proposition": row["proposition"]},
            ))
        return items
