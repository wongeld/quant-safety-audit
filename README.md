# quant-safety-audit

**Does the quantization method you pick change how safe a model behaves — even when its capabilities look fine?**

An open, reproducible audit comparing publicly available quantization methods (GGUF, AWQ, GPTQ, bitsandbytes, HQQ, AutoRound) on the same model, across five safety-adjacent axes — under-refusal, over-refusal, toxicity, prompt-injection resistance, and sycophancy — while controlling for general capability, so we can tell "this method makes the model worse at everything" apart from "this method specifically breaks safety behavior while quality looks fine."

## Why this exists

Every public "which quantization should I use" guide compares speed, perplexity, and memory footprint. None of them put safety behavior on that decision matrix. People choosing between AWQ, GPTQ, GGUF, bitsandbytes, and HQQ for a real deployment currently have no comparative safety guidance at all — this project is trying to produce a first pass at that missing column.

## Status

Past feasibility checks now. Where things actually stand:

- **Quantization pipeline: validated.** All 6 methods produce checkpoints that load and generate correctly on the target model, including two that needed real debugging to get there (see [Loading corrections](#loading-corrections-and-why-they-mattered) below) — this isn't "the code runs," it's "confirmed against real generated output, method by method."
- **MASK (honesty under pressure) pilot: running clean.** Small-N pilot (5 items × 8 model variants = 80 generations) completed with zero failed generations across every method. Not yet run at full scale, and the pressured-prompt sample so far shows no answer variance — see the framework's own review tooling for why that limits what this pilot can show before the item count goes up.
- **Multi-judge cross-validation: built.** MASK's official judge is GPT-4o over a paid API; this project substitutes an open, locally-run model instead, and supports running several different judges against the same generations to check where they agree and disagree, rather than trusting one judge's labels by default.
- **Four of the five safety axes (over-refusal, toxicity, prompt-injection resistance, sycophancy) and HarmBench (under-refusal): scoped, not yet wired into the framework.** See `docs/PHASE_CHECKLIST.md` for the full plan and reasoning behind what's prioritized and what's explicitly excluded.

## Repo structure

```
quant-safety-audit/
├── quant_safety_audit/                                     # the reusable framework
│   ├── __init__.py
│   ├── loaders.py        # one class per quantization method, common load/generate/unload interface
│   ├── benchmarks.py     # one class per benchmark, produces BenchmarkItem lists
│   ├── judges.py         # one class per judge, scores (item, response) -> JudgeResult
│   └── pipeline.py       # ExperimentRunner (generate -> judge -> summarize) + compare_judges
├── notebooks/
│   ├── 1 Setup Verification Quantization Run - Final.ipynb    # model verification + all 6 quantization methods
│   └── 3 MASK Benchmark Pilot Pipeline Final.ipynb             # MASK pilot: generation, judging, review
├── docs/
│   └── PHASE_CHECKLIST.md                                   # full execution checklist, phase by phase
├── requirements.txt
├── .gitignore
└── LICENSE
```

## The framework

`quant_safety_audit/` is the reusable piece — the part meant to outlive this specific study. Three independent extension points, each unaware of the other two:

- **A new quantization method** → one new `BaseModelLoader` subclass in `loaders.py`, one new branch in `build_loader()`. Nothing else changes.
- **A new benchmark** (HarmBench, OR-Bench, BIPIA, sycophancy-eval, ...) → one new `BaseBenchmark` subclass in `benchmarks.py` returning `BenchmarkItem` objects. `ExperimentRunner` doesn't know or care which benchmark it's running.
- **A new judge** → one new `BaseJudge` subclass in `judges.py`. `ExperimentRunner.run_judging()` accepts any judge, and `compare_judges()` will diff its labels against any other judge's labels on the same generations.

`ExperimentRunner` ties these together: generate (with crash-safe checkpointing and per-model memory logging), judge, summarize. Swapping the benchmark or adding a quantization method is a config change, not a rewrite.

### Loading corrections, and why they mattered

Every quantization method needs a genuinely different loading path — this cost real debugging time to get right, and is documented in `loaders.py`'s docstrings rather than left implicit:

| Method | Loads via | Why not the obvious way |
|---|---|---|
| bitsandbytes | `transformers.AutoModelForCausalLM` | Works as expected — auto-detects from the saved checkpoint. |
| AutoRound | `transformers.AutoModelForCausalLM` + explicit `AutoRoundConfig(backend="torch")` | Needed explicitly at load time — confirmed necessary against a real checkpoint. |
| **GPTQ** | `gptqmodel.GPTQModel.load()` directly | Going through `transformers` routes through an `optimum` bridge with a real bug (`optimum/gptq/quantizer.py` references `QuantizeConfig` without importing it) — confirmed from a full traceback, not a guess. Bypassing `optimum` entirely, rather than patching around it, is the fix. |
| **AWQ** | `awq.AutoAWQForCausalLM.from_quantized()` | Generic `transformers` loading doesn't correctly dispatch AWQ's kernels. |
| **HQQ** | `hqq.models.hf.base.AutoHQQHFModel` | The current recommended entry point — the older `HQQModelForCausalLM` API is deprecated. |
| **GGUF** | `llama_cpp.Llama` via `create_chat_completion()` | Not `transformers(gguf_file=...)` — that path was never confirmed working. `create_chat_completion` (not raw completion mode) matters specifically because it applies the model's own chat template, which MASK's system+user structured prompts need. |

Also worth knowing if you're extending this: `bitsandbytes` logs a warning on every int8 matmul call (once per token generated, per quantized layer) — enough volume to hang a browser tab rendering it. `loaders.py` silences that logger at import time; it's not a sign generation is broken.


## Quickstart

1. Open `1 Setup Verification Quantization Run - Final.ipynb` in Colab (Runtime → Change runtime type → GPU). Run the setup/verification cells first — they confirm the model source is legitimate (checked against the HF API, not eyeballed from the repo name) and that every benchmark dataset downloads correctly. Then run the quantization cells to produce all 6 checkpoints, saved to Google Drive.
2. Open `3 MASK Benchmark Pilot Pipeline Final.ipynb`. This writes the `quant_safety_audit` package to disk, points it at the checkpoints from step 1, and runs the generate → judge → summarize pipeline on a small pilot sample.
3. To extend rather than just run: import `quant_safety_audit` directly, write a new `BaseBenchmark` or `BaseJudge` subclass, and pass it into `ExperimentRunner` — no notebook surgery required.

## Model choice, and why

Primary target: **`Qwen/Qwen2.5-3B-Instruct`**. Deliberately not the newest Qwen release — verified quantization-checkpoint availability mattered more than novelty for this project:

- `Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4` — official Qwen GPTQ checkpoint exists.
- `bartowski/Qwen2.5-3B-Instruct-GGUF` — extensive, well-maintained GGUF k-quant builds exist.
- bitsandbytes and HQQ need no pre-quantized checkpoint — produced directly in the notebook.

Two models were considered and deliberately excluded for this phase:
- `Qwen/Qwen3.5-4B` — confirmed multimodal (image-text-to-text), which adds VLM-specific complexity that several quantization tools handle inconsistently.
- The `Qwen3.6` family — the only open-weight checkpoints (`Qwen3.6-27B`, `Qwen3.6-35B-A3B`) are too large for this project's compute budget; `Qwen3.6-Plus`/`Qwen3.6-Max-Preview` are hosted API-only, not open weights. Also worth flagging: several HF repos with "Qwen3.6" in the name are not published by the official `Qwen` org — the notebook checks the actual publishing org via the HF API rather than trusting the repo name.

## Quantization methods covered

The usual four (GGUF via llama.cpp k-quants, AWQ, GPTQ, bitsandbytes), plus HQQ (data-free, fast to produce) and AutoRound (Intel's post-training quantization method, actively developed through 2026 and reported to outperform GPTQ/AWQ/GGUF on low-bit accuracy leaderboards, especially for small-to-medium models — included specifically because it's the most recent addition to this list). See [Loading corrections](#loading-corrections-and-why-they-mattered) above for how each is actually loaded.

## Judging

MASK's official evaluation uses GPT-4o over a paid API. This project substitutes an open, locally-run model instead (`MaskOpenLLMJudge`, default `Qwen/Qwen2.5-14B-Instruct`) — a real methodological deviation, not a minor detail. To manage that risk:

- The judge prompt templates are copied verbatim from MASK's own repo, not retyped from a paper.
- `ExperimentRunner.run_judging()` accepts any `BaseJudge`, so multiple judges can score the same generations independently (each saves to its own file, `scored_<judge_name>.csv` — no silent overwrites).
- `compare_judges()` reports pairwise agreement and flags exactly which rows different judges disagree on, so disagreement is something to go read, not something averaged away.
- Content-moderation/safety classifiers (Llama-Guard family, WildGuard, ShieldGemma) are deliberately excluded from the judge pool — they're fixed-taxonomy classifiers, not general reasoning judges, and misfire on this kind of open belief-comparison task.

## Scope — what this does *not* cover, on purpose

Biosecurity, chemical/physical safety, and cybersecurity uplift are excluded — those need review this project isn't resourced to do responsibly. Agentic/tool-use safety, fairness/allocational decision-making, deception, and broader adversarial robustness are also out of scope for now — each needs its own dedicated harness. See `docs/PHASE_CHECKLIST.md` for the full reasoning.

## License

Code in this repo: MIT (see `LICENSE`). Model weights, benchmark datasets, and any downloaded quantized checkpoints retain their own original licenses — this repo does not redistribute them, the notebooks only download from their original sources.