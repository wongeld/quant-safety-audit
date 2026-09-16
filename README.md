# quant-safety-audit

**Does the quantization method you pick change how a model behaves on safety-relevant evaluations, even when the model still generates normally?**

This project evaluates multiple quantization methods on the same language model and compares their behavior on safety and honesty-related benchmarks. The goal is to separate general capability degradation from changes in safety-relevant behavior.

## Status

Current implementation status:

* **Quantization pipeline: validated.** Eight model variants have been successfully produced and tested for loading and generation on the primary Qwen model.
* **MASK generation pipeline: validated.** An initial 5-item pilot was run across all 8 variants, producing 80 generations with zero generation failures.
* **MASK judging: validated.** The pilot produced 40 judged rows, one comparison per `(model variant, MASK item)`. The local judge compares the pressured response with the corresponding belief response.
* **Generation and judging are now separate phases.** MASK generations can be produced and saved without loading the computationally expensive judge. Judging can then be performed later on suitable hardware.
* **Next phase: larger-scale MASK experiments.** The project is moving from the 3B pilot to larger MASK runs and larger models using a higher-memory GPU.
* **Other safety benchmarks:** scoped for later integration. These include HarmBench, OR-Bench, ToxiGen, BIPIA, and sycophancy-related evaluations.

The current 5-item MASK pilot is a pipeline validation only. It is too small to support substantive conclusions about differences between quantization methods.

## Research question

The main question is:

> Does quantization change safety-relevant behavioral consistency while the model remains generally capable of producing responses?

For MASK specifically, this project uses **deception** in the operational sense measured by the benchmark: whether a model maintains consistency with its own stated beliefs when it is given an opportunity to behave differently.

The project therefore focuses on behavioral changes associated with quantization rather than treating ordinary accuracy loss as deception.

## Model

The primary validated model is:

`Qwen/Qwen2.5-3B-Instruct`

This model was selected because the project needs a single model with broad support across the quantization backends being compared.

The current evaluated variants are:

| Variant          | Method            |
| ---------------- | ----------------- |
| `baseline-bf16`  | Original model    |
| `bnb-nf4`        | bitsandbytes NF4  |
| `bnb-int8`       | bitsandbytes INT8 |
| `hqq-4bit`       | HQQ 4-bit         |
| `gptq-int4`      | GPTQ INT4         |
| `awq-int4`       | AWQ INT4          |
| `autoround-int4` | AutoRound INT4    |
| `gguf-q4km`      | GGUF Q4_K_M       |

Pre-quantized checkpoints are used where appropriate, while bitsandbytes, HQQ, and AutoRound variants can be produced directly from the base model.

The framework is not restricted to the 3B model. The base model can be changed through `QSA_BASE_MODEL`, provided the selected model is supported by the relevant loading and quantization backends.

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
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

Generated checkpoints, benchmark downloads, results, logs, and Python cache files are excluded from Git.

## Framework

The `quant_safety_audit/` package is designed as a reusable experiment framework rather than a MASK-only script.

### Model loaders

`loaders.py` provides a common interface for loading, generating with, and unloading each model variant.

The current loading paths include:

| Method       | Loading path                                                              |
| ------------ | ------------------------------------------------------------------------- |
| bitsandbytes | `transformers.AutoModelForCausalLM`                                       |
| AutoRound    | `transformers.AutoModelForCausalLM` with explicit AutoRound configuration |
| GPTQ         | GPTQModel / Transformers GPTQ loading path                                |
| AWQ          | AutoAWQ or Transformers, depending on checkpoint format                   |
| HQQ          | AutoHQQHFModel                                                            |
| GGUF         | `llama_cpp.Llama`                                                         |

The AWQ loader supports both AutoAWQ-style checkpoints and AWQ checkpoints using the `compressed-tensors` format.

The GGUF loader uses `create_chat_completion()` so that the model's chat template can be applied to MASK's structured system/user prompts.

### Benchmarks

`benchmarks.py` defines benchmark interfaces and benchmark-specific items.

The current MASK implementation provides `MaskBenchmark`.

The intended framework can also support benchmarks such as:

* HarmBench
* OR-Bench
* ToxiGen
* BIPIA
* sycophancy evaluations

### Judges

`judges.py` defines the judge interface.

For MASK, the current local judge is:

`Qwen/Qwen2.5-14B-Instruct`

The 14B model is run locally using 4-bit bitsandbytes quantization.

MASK's original evaluation uses a hosted judge. Using a local open-weight judge is therefore a methodological deviation that will need to be considered when interpreting results.

### Experiment runner

`pipeline.py` contains `ExperimentRunner`, which handles:

* benchmark item loading
* model generation
* crash-safe generation checkpointing
* model unloading and GPU memory cleanup
* judging
* summary generation

Adding a new benchmark or model loader should not require rewriting the experiment pipeline.

## MASK evaluation

For each MASK item, the pipeline generates two responses:

* **pressured**: the primary situation in which the model may have an incentive to behave differently
* **belief**: the corresponding prompt used to establish or query the model's stated belief

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

For models whose pre-quantized GPTQ or AWQ checkpoint names do not follow the default naming convention, candidate repositories can be supplied through:

```text
QSA_GPTQ_CANDIDATES
QSA_AWQ_CANDIDATES
```

Multiple candidates can be separated with semicolons.

## MASK pipeline

The executable MASK pipeline is:

```bash
python -m quant_safety_audit.run_mask_pipeline
```

It expects the required quantized checkpoints under the configured output directory and performs a pre-flight check before spending GPU time.

The main configuration values are:

| Variable             | Default                     | Purpose                                 |
| -------------------- | --------------------------- | --------------------------------------- |
| `QSA_OUTPUT_DIR`     | `~/quant-safety-audit-data` | Checkpoints and results root            |
| `QSA_BASE_MODEL`     | `Qwen/Qwen2.5-3B-Instruct`  | Base model                              |
| `QSA_JUDGE_MODEL`    | `Qwen/Qwen2.5-14B-Instruct` | Local MASK judge                        |
| `QSA_MASK_ARCHETYPE` | `known_facts`               | MASK archetype                          |
| `QSA_N_ITEMS`        | `1000`                      | Number of MASK items                    |
| `QSA_RUN_JUDGING`    | `0`                         | Whether to run judging after generation |
| `QSA_GGUF_FILENAME`  | Qwen 3B Q4_K_M filename     | GGUF checkpoint filename                |

`QSA_N_ITEMS` can also be set to `all` or `0` to run the full available MASK split.

### Generation and judging

Generation and judging are intentionally separate.

By default:

```text
QSA_RUN_JUDGING=0
```

This runs generation only and saves:

```text
~/quant-safety-audit-data/results/mask/generations.csv
```

This is useful when the generation GPU is suitable for the target model but does not have enough memory for the local 14B judge.

Judging can be enabled with:

```bash
QSA_RUN_JUDGING=1 python -m quant_safety_audit.run_mask_pipeline
```

When judging is enabled, the pipeline loads the local judge after generation and produces:

```text
scored.csv
summary.csv
```

The separation also allows generations to be produced on one machine and judged later on another machine.

## Larger-model MASK experiments

The initial pipeline validation used `Qwen/Qwen2.5-3B-Instruct` on a Tesla V100-SXM2-16GB GPU.

The next phase uses higher-memory GPU hardware to evaluate larger models on MASK. The repository is designed so that the base model can be changed through:

```bash
export QSA_BASE_MODEL="model-id"
```

The exact larger-model configuration depends on model size, quantization format, available GPU memory, and backend support. Models will therefore be validated on the available hardware before being included in the larger experiment.

The current higher-memory experiment environment provides:

```text
GPU: 1× Tesla V100-SXM3-32GB
Storage: 774 GB
CUDA: 13.0
```

This hardware is being used for a short-duration compute window, primarily to run larger MASK experiments that are impractical or unnecessarily constrained on the 16 GB development GPU.

## Quickstart

Create and activate a virtual environment, then install the dependencies from:

```text
requirements.txt
```

The quantization pipeline can then be run with:

```bash
python scripts/quantize_models.py
```

After the checkpoints have been produced, run a small MASK validation with:

```bash
QSA_N_ITEMS=5 python -m quant_safety_audit.run_mask_pipeline
```

For a larger generation run without judging:

```bash
QSA_N_ITEMS=1000 QSA_RUN_JUDGING=0 \
python -m quant_safety_audit.run_mask_pipeline
```

To run the full available split:

```bash
QSA_N_ITEMS=all QSA_RUN_JUDGING=0 \
python -m quant_safety_audit.run_mask_pipeline
```

For runs that should continue unattended, use an appropriate terminal multiplexer such as `tmux` and redirect output to a log file.

## Reproducibility

The project separates code from generated artifacts.

The Git repository contains:

* experiment framework code
* quantization scripts
* benchmark and judge implementations
* notebooks
* documentation
* dependency specification

The following are not committed to Git:

* model weights
* quantized checkpoints
* benchmark downloads
* generated experiment results
* runtime logs
* Python environments
* Python cache files

The `.gitignore` contains rules for these large or reproducible artifacts.

The initial development environment has been validated on a Tesla V100-SXM2-16GB GPU. Larger-model experiments use higher-memory hardware when required.

## Scope

This project is intentionally narrower than a general AI safety evaluation.

Currently excluded from the active evaluation scope are:

* biosecurity
* chemical and physical safety
* cybersecurity uplift
* agentic/tool-use safety
* fairness and allocational decision-making
* broad adversarial robustness

These areas require dedicated evaluation designs rather than being treated as interchangeable benchmark categories.

The current focus is quantization-related changes in model behavior, beginning with MASK and extending to selected safety benchmarks.

## Limitations

Several limitations should be kept in mind when interpreting results:

* **Small pilot size:** the initial N=5 MASK run only validates the pipeline and cannot establish meaningful statistical differences.
* **Local judge:** the Qwen2.5-14B judge differs from MASK's original hosted judge.
* **Quantization methods are not identical implementations:** different backends may use different calibration procedures, kernels, and checkpoint sources.
* **Hardware matters:** inference behavior and supported kernels can depend on the available GPU and CPU.
* **Capability and safety are related:** a behavioral difference should be interpreted alongside general capability measurements rather than automatically attributed to a safety-specific effect.
* **Model changes are confounded with quantization when comparing different base models:** comparisons across model families should not be interpreted as pure quantization effects.

## License

Code in this repository is released under the MIT License. See LICENSE.

Model weights, benchmark datasets, and downloaded quantized checkpoints retain their respective original licenses. This repository does not redistribute those artifacts.

