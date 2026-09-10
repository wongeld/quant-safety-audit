# quant-safety-audit

**Does the quantization method you pick change how a model behaves on safety-relevant evaluations, even when the model still generates normally?**

This project evaluates multiple quantization methods on the same language model and compares their behavior on safety and honesty-related benchmarks. The goal is to separate general capability degradation from changes in safety-relevant behavior.

## Status

Current implementation status:

- **Quantization pipeline: validated.** Eight model variants have been successfully produced and tested for loading/generation on the target Qwen model.
- **MASK pipeline: validated at smoke-test scale.** A 5-item pilot was run across all 8 variants, producing 80 generations with zero generation failures.
- **MASK judging: validated.** The pilot produced 40 judged rows, one comparison per `(model variant, MASK item)`. The judge compares the pressured response with the corresponding belief response.
- **Full MASK generation: next step.** The current plan is to scale the generation phase to approximately 500 MASK items before running the computationally expensive judging phase.
- **Other safety benchmarks:** scoped for later integration. These include HarmBench, OR-Bench, ToxiGen, BIPIA, and sycophancy-related evaluations.

The current results are a pipeline validation only. The N=5 MASK pilot is too small to support substantive conclusions about differences between quantization methods.

## Research question

The main question is:

> Does quantization change safety-relevant behavioral consistency while the model remains generally capable of producing responses?

For MASK specifically, this project uses **deception** in the operational sense measured by the benchmark: whether a model maintains consistency with its own stated beliefs when it is given an opportunity to behave differently.

The project therefore focuses on behavioral changes associated with quantization rather than treating ordinary accuracy loss as deception.

## Model

The primary model is:

`Qwen/Qwen2.5-3B-Instruct`

This model was selected because the project needs a single model with broad support across the quantization backends being compared.

The evaluated variants are:

| Variant | Method |
|---|---|
| `baseline-bf16` | Original model |
| `bnb-nf4` | bitsandbytes NF4 |
| `bnb-int8` | bitsandbytes INT8 |
| `hqq-4bit` | HQQ 4-bit |
| `gptq-int4` | GPTQ INT4 |
| `awq-int4` | AWQ INT4 |
| `autoround-int4` | AutoRound INT4 |
| `gguf-q4km` | GGUF Q4_K_M |

Pre-quantized checkpoints are used where appropriate, while bitsandbytes, HQQ, and AutoRound variants can be produced directly from the base model.

## Repository structure

```text
quant-safety-audit/
├── quant_safety_audit/
│   ├── __init__.py
│   ├── benchmarks.py
│   ├── judges.py
│   ├── loaders.py
│   ├── pipeline.py
│   ├── quantizers.py
│   └── run_mask_pipeline.py
├── scripts/
│   ├── quantize_models.py
│   └── setup_env.sh
├── notebooks/
│   ├── 1 Setup Verification Quantization Run - Final.ipynb
│   └── 3 MASK Benchmark Pilot Pipeline Final.ipynb
├── docs/
│   └── PHASE_CHECKLIST.md
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Framework

The `quant_safety_audit/` package is designed as a reusable experiment framework rather than a MASK-only script.

### Model loaders

`loaders.py` provides a common interface for loading, generating with, and unloading each model variant.

The current loading paths are:

| Method | Loading path |
|---|---|
| bitsandbytes | `transformers.AutoModelForCausalLM` |
| AutoRound | `transformers.AutoModelForCausalLM` with explicit AutoRound configuration |
| GPTQ | GPTQModel / Transformers GPTQ loading path |
| AWQ | AutoAWQ |
| HQQ | AutoHQQHFModel |
| GGUF | `llama_cpp.Llama` |

The GGUF loader uses `create_chat_completion()` so that the model's chat template can be applied to MASK's structured system/user prompts.

### Benchmarks

`benchmarks.py` defines benchmark interfaces and benchmark-specific items.

The current MASK implementation provides `MaskBenchmark`.

The intended framework can also support benchmarks such as:

- HarmBench
- OR-Bench
- ToxiGen
- BIPIA
- sycophancy evaluations

### Judges

`judges.py` defines the judge interface.

For MASK, the current implementation uses:

`Qwen/Qwen2.5-14B-Instruct`

The 14B model is run locally using 4-bit bitsandbytes quantization.

MASK's original evaluation uses a hosted judge. Using a local open-weight judge is therefore a methodological deviation that will need to be considered when interpreting results.

### Experiment runner

`pipeline.py` contains `ExperimentRunner`, which handles:

- benchmark item loading
- model generation
- crash-safe generation checkpointing
- model unloading and GPU memory cleanup
- judging
- summary generation

Adding a new benchmark or model loader should not require rewriting the experiment pipeline.

## MASK evaluation

For each MASK item, the pipeline generates two responses:

- **pressured**: the primary situation in which the model may have an incentive to behave differently
- **belief**: the corresponding prompt used to establish or query the model's stated belief

The generation dataset therefore contains:

```text
number of variants × number of items × 2 turns
```

For example, the N=5 smoke test produced:

```text
8 variants × 5 items × 2 turns = 80 generations
```

The judge then groups the two turns by `(variant, item_id)` and produces one label for each comparison:

```text
8 variants × 5 items = 40 judged rows
```

This distinction is important when checking result counts.

## Quantization pipeline

The main quantization entry point is:

```text
scripts/quantize_models.py
```

It produces checkpoints under:

```text
~/quant-safety-audit-data/models/
```

The current expected checkpoint directories are:

```text
models/
├── autoround-int4/
├── awq-int4/
├── bnb-int8/
├── bnb-nf4/
├── gguf/
├── gptq-int4/
└── hqq-4bit/
```

The base model is not copied into this directory because it is loaded from Hugging Face.

## MASK pipeline

The executable MASK pipeline is:

```text
quant_safety_audit/run_mask_pipeline.py
```

It expects the quantized checkpoints under the configured output directory and performs a pre-flight check before spending GPU time.

The main configuration values include:

- `QSA_OUTPUT_DIR`
- `QSA_BASE_MODEL`
- `QSA_JUDGE_MODEL`
- `QSA_MASK_ARCHETYPE`
- `QSA_N_ITEMS`
- `QSA_GGUF_FILENAME`

The default output location is:

```text
~/quant-safety-audit-data/
```

Results are written to:

```text
~/quant-safety-audit-data/results/mask/
```

The generation phase writes:

```text
generations.csv
```

The judging phase writes:

```text
scored.csv
```

and the summarized results are written to:

```text
summary.csv
```

Model checkpoints, generated results, downloaded datasets, and logs are intentionally kept outside Git.

## Quickstart

Create and activate a virtual environment, then install the dependencies from:

```text
requirements.txt
```

The quantization pipeline can then be run with:

```bash
python scripts/quantize_models.py
```

After the checkpoints have been produced, the MASK pipeline can be run with:

```bash
python -m quant_safety_audit.run_mask_pipeline
```

For a small validation run, set the number of MASK items to a small value before starting.

For a larger experiment, run the generation phase separately from judging when possible. The local 14B judge is substantially more computationally expensive than generating responses from the 3B variants.

## Reproducibility

The project separates code from generated artifacts.

The Git repository contains:

- experiment framework code
- quantization scripts
- benchmark and judge implementations
- notebooks
- documentation
- dependency specification

The following are not committed to Git:

- model weights
- quantized checkpoints
- benchmark downloads
- generated experiment results
- runtime logs
- Python environments

The `.gitignore` contains rules for these large or reproducible artifacts.

The current development environment has been validated on a Tesla V100-SXM2-16GB GPU.

## Scope

This project is intentionally narrower than a general AI safety evaluation.

Currently excluded from the active evaluation scope are:

- biosecurity
- chemical and physical safety
- cybersecurity uplift
- agentic/tool-use safety
- fairness and allocational decision-making
- broad adversarial robustness

These areas require dedicated evaluation designs rather than being treated as interchangeable benchmark categories.

The current focus is quantization-related changes in model behavior, beginning with MASK and extending to selected safety benchmarks.

## Limitations

Several limitations should be kept in mind when interpreting results:

- **Small pilot size:** the initial N=5 MASK run only validates the pipeline and cannot establish meaningful statistical differences.
- **Local judge:** the Qwen2.5-14B judge differs from MASK's original hosted judge.
- **Quantization methods are not identical implementations:** different backends may use different calibration procedures, kernels, and checkpoint sources.
- **Hardware matters:** inference behavior and supported kernels can depend on the available GPU and CPU.
- **Capability and safety are related:** a behavioral difference should be interpreted alongside general capability measurements rather than automatically attributed to a safety-specific effect.

## License

Code in this repository is released under the MIT License. See LICENSE.

Model weights, benchmark datasets, and downloaded quantized checkpoints retain their respective original licenses. This repository does not redistribute those artifacts.
