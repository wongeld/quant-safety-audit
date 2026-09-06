"""
ExperimentRunner ties a set of model loaders, a benchmark, and a judge
together: generate, checkpoint, score, save. This is the piece that makes
"swap the benchmark" or "add a quantization method" a config change instead
of a notebook rewrite -- everything here is written against BaseBenchmark,
BaseJudge, and BaseModelLoader, never against MASK or Qwen specifically.
"""

import os
import json
import traceback
import pandas as pd

from .loaders import build_loader
from .benchmarks import BenchmarkItem


class ExperimentRunner:
    def __init__(self, loader_configs: dict, benchmark, judge, results_dir: str, n_items: int = 2):
        """loader_configs: {variant_name: {"method": ..., "path": ...}} --
        same shape as the MODEL_VARIANTS dicts used throughout this project.
        benchmark: a BaseBenchmark instance (e.g. MaskBenchmark()).
        judge: a BaseJudge instance (e.g. MaskOpenLLMJudge()).
        """
        self.loader_configs = loader_configs
        self.benchmark = benchmark
        self.judge = judge
        self.results_dir = results_dir
        self.n_items = n_items
        os.makedirs(results_dir, exist_ok=True)

    def run_generation(self) -> pd.DataFrame:
        items = self.benchmark.load_items(self.n_items)
        raw_path = f"{self.results_dir}/generations.csv"

        all_rows = []
        completed = set()
        if os.path.exists(raw_path):
            existing = pd.read_csv(raw_path)
            all_rows = existing.to_dict("records")
            completed = set(existing["variant"].unique())
            print(f"Resuming -- already have: {sorted(completed)}")

        for variant_name, spec in self.loader_configs.items():
            if variant_name in completed:
                print(f"[{variant_name}] already done, skipping.")
                continue

            try:
                loader = build_loader(variant_name, spec)
            except Exception as e:
                print(f"[{variant_name}] FAILED TO BUILD LOADER: {type(e).__name__}: {e!r}")
                continue

            try:
                with loader:
                    for item in items:
                        for turn_name, system_p, user_p in [
                            ("pressured", item.primary_system, item.primary_user),
                            ("belief", item.belief_system, item.belief_user),
                        ]:
                            if user_p is None:
                                continue
                            try:
                                resp = loader.generate(user_p, system_prompt=system_p)
                            except Exception as e:
                                resp = f"[GENERATION FAILED: {type(e).__name__}: {e!r}]"
                                print(f"[{variant_name}] {turn_name} gen FAILED on {item.item_id}: {e!r}")
                                print(traceback.format_exc())
                            all_rows.append({
                                "variant": variant_name, "item_id": item.item_id, "turn": turn_name,
                                "system_prompt": system_p, "user_prompt": user_p, "response": resp,
                                "meta": json.dumps(item.meta),
                            })
            except Exception as e:
                print(f"[{variant_name}] FAILED TO LOAD: {type(e).__name__}: {e!r}")
                print(traceback.format_exc())
                continue

            pd.DataFrame(all_rows).to_csv(raw_path, index=False)
            print(f"[{variant_name}] done, saved to {raw_path}")

        return pd.DataFrame(all_rows)

    def run_judging(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        scored_path = f"{self.results_dir}/scored.csv"
        rows = []
        with self.judge:
            for (variant, item_id), group in raw_df.groupby(["variant", "item_id"]):
                pressured = group[group["turn"] == "pressured"]
                belief = group[group["turn"] == "belief"]
                if pressured.empty or str(pressured.iloc[0]["response"]).startswith("[GENERATION FAILED"):
                    rows.append({"variant": variant, "item_id": item_id, "label": None,
                                 "raw_output": "[skipped -- generation failed or missing]", "extra": None})
                    continue

                pressured_resp = pressured.iloc[0]["response"]
                belief_resp = None
                if not belief.empty and not str(belief.iloc[0]["response"]).startswith("[GENERATION FAILED"):
                    belief_resp = belief.iloc[0]["response"]

                meta = json.loads(pressured.iloc[0]["meta"]) if pressured.iloc[0]["meta"] else {}
                item = BenchmarkItem(
                    item_id=item_id,
                    primary_system=pressured.iloc[0]["system_prompt"],
                    primary_user=pressured.iloc[0]["user_prompt"],
                    belief_system=belief.iloc[0]["system_prompt"] if not belief.empty else None,
                    belief_user=belief.iloc[0]["user_prompt"] if not belief.empty else None,
                    meta=meta,
                )

                try:
                    result = self.judge.score(item, pressured_resp, belief_resp)
                    row = {"variant": variant, "item_id": item_id, "label": result.label,
                           "raw_output": result.raw_output, "extra": json.dumps(result.extra) if result.extra else None}
                except Exception as e:
                    print(f"Judge FAILED on {variant}/{item_id}: {e!r}")
                    print(traceback.format_exc())
                    row = {"variant": variant, "item_id": item_id, "label": None,
                           "raw_output": f"[JUDGE FAILED: {type(e).__name__}: {e!r}]", "extra": None}
                rows.append(row)

        scored_df = pd.DataFrame(rows)
        scored_df.to_csv(scored_path, index=False)
        return scored_df

    def summarize(self, scored_df: pd.DataFrame) -> pd.DataFrame:
        summary = (
            scored_df[scored_df["label"].notna()]
            .groupby("variant")["label"]
            .value_counts(normalize=True)
            .unstack(fill_value=0.0)
            .reset_index()
        )
        summary_path = f"{self.results_dir}/summary.csv"
        summary.to_csv(summary_path, index=False)
        return summary
