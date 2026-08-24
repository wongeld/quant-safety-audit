# quant-safety-audit

**Does the quantization method you pick change how safe a model behaves — even when its capabilities look fine?**

An open, reproducible audit comparing publicly available quantization methods (GGUF, AWQ, GPTQ, bitsandbytes, HQQ, AutoRound) on the same model, across five safety-adjacent axes — under-refusal, over-refusal, toxicity, prompt-injection resistance, and sycophancy — while controlling for general capability, so we can tell "this method makes the model worse at everything" apart from "this method specifically breaks safety behavior while quality looks fine."

## Why this exists

Every public "which quantization should I use" guide compares speed, perplexity, and memory footprint. None of them put safety behavior on that decision matrix. People choosing between AWQ, GPTQ, GGUF, bitsandbytes, and HQQ for a real deployment currently have no comparative safety guidance at all — this project is trying to produce a first pass at that missing column.

## Status

**Phase 0: feasibility checks and harness setup.** Nothing has been quantized or evaluated yet — right now the notebook verifies that the target model source is legitimate and that every benchmark dataset actually downloads and looks correct, before any model output gets generated. See `docs/PHASE_CHECKLIST.md` for the full phase-by-phase plan.

## Repo structure

```
quant-safety-audit/
├── notebooks/
│   └── 01_setup_verification_quantization.ipynb   # Colab: model verification, benchmark checks, quantization
├── docs/
│   └── PHASE_CHECKLIST.md                          # Full execution checklist, phase by phase
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Quickstart

1. Open `notebooks/01_setup_verification_quantization.ipynb` in Google Colab (Runtime → Change runtime type → GPU).
2. Run the setup and verification cells first. They confirm the model source is legitimate (checked against the HF API, not just eyeballed from the repo name) and that every benchmark dataset downloads and previews correctly.
3. Run the quantization cells — each method is a self-contained cell. Outputs are saved to a Google Drive folder so they survive when the Colab session ends.

## Model choice, and why

Primary target: **`Qwen/Qwen2.5-3B-Instruct`**. Deliberately not the newest Qwen release — verified quantization-checkpoint availability mattered more than novelty for this project:

- `Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4` — official Qwen GPTQ checkpoint exists.
- `bartowski/Qwen2.5-3B-Instruct-GGUF` — extensive, well-maintained GGUF k-quant builds exist.
- bitsandbytes and HQQ need no pre-quantized checkpoint — produced directly in the notebook.

Two models were considered and deliberately excluded for this phase:
- `Qwen/Qwen3.5-4B` — confirmed multimodal (image-text-to-text), which adds VLM-specific complexity that several quantization tools handle inconsistently.
- The `Qwen3.6` family — the only open-weight checkpoints (`Qwen3.6-27B`, `Qwen3.6-35B-A3B`) are too large for this project's compute budget; `Qwen3.6-Plus`/`Qwen3.6-Max-Preview` are hosted API-only, not open weights. Also worth flagging: several HF repos with "Qwen3.6" in the name are not published by the official `Qwen` org — the notebook checks the actual publishing org via the HF API rather than trusting the repo name.

## Quantization methods covered

The usual four (GGUF via llama.cpp k-quants, AWQ, GPTQ, bitsandbytes), plus HQQ (data-free, fast to produce) and AutoRound (Intel's post-training quantization method, actively developed through 2026 and reported to outperform GPTQ/AWQ/GGUF on low-bit accuracy leaderboards, especially for small-to-medium models — included specifically because it's the most recent addition to this list).

## Scope — what this does *not* cover, on purpose

Biosecurity, chemical/physical safety, and cybersecurity uplift are excluded — those need review this project isn't resourced to do responsibly. Agentic/tool-use safety, fairness/allocational decision-making, deception, and broader adversarial robustness are also out of scope for now — each needs its own dedicated harness. See `docs/PHASE_CHECKLIST.md` for the full reasoning.

## License

Code in this repo: MIT (see `LICENSE`). Model weights, benchmark datasets, and any downloaded quantized checkpoints retain their own original licenses — this repo does not redistribute them, the notebook only downloads from their original sources.
