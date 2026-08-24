# Quantization Method Safety Audit — Execution Checklist

## Research question
Do different, publicly reproducible quantization methods (GGUF k-quant, bitsandbytes NF4, HQQ, and — if checkpoints exist for the chosen model — AWQ, GPTQ) affect an instruction-tuned model differently across five safety-adjacent axes, in ways not predicted by a general capability/quality control metric?

## Explicit scope boundaries (state these in the paper, don't leave them implicit)
- **In scope:** under-refusal (harmful compliance), over-refusal, toxicity, prompt-injection resistance, sycophancy (pushback resistance).
- **Out of scope, stated on purpose:** biosecurity, chemical/physical safety, cybersecurity uplift, agentic/tool-use safety (needs a live execution harness), fairness/allocational decision-making, deception, OOD/distribution-shift, adversarial robustness beyond the injection axis. Say why: dangerous-content axes need review this project can't support solo; agentic and fairness axes each need a dedicated harness that would double the timeline.
- **Primary claim to test:** "quality/capability preservation is not a reliable proxy for safety-axis preservation across quantization methods" — a specific, falsifiable, already-partially-supported claim, not a vague "quantization is risky."

---

## Phase 0 — Feasibility lock-in (do this before writing any code)

- [ ] Pick the primary model (e.g. a Qwen2.5/3 instruct model in the 3-4B range). Before committing, search Hugging Face directly for that exact model name + "AWQ", + "GPTQ", + "GGUF", + "HQQ" and confirm what actually exists pre-quantized. Do not assume coverage — small models often only have GGUF community checkpoints.
- [ ] For any method with no pre-quantized checkpoint, confirm you can self-quantize it (AutoAWQ / GPTQModel / HQQ / llama.cpp) on your available Colab GPU in under ~2 hours. Test this on the actual model before finalizing your method list — if AWQ/GPTQ calibration fails or times out, drop it now, not mid-run.
- [ ] Confirm bitsandbytes NF4/INT8 loads directly via `transformers` with no separate quantization step (it does — this is your cheapest method to include).
- [ ] Download the BIPIA repo, confirm the **email subtask** runs end-to-end without needing a separately licensed corpus. Do not use the Web QA/Summarization subtasks unless you've confirmed you can legally reconstruct their context data.
- [ ] Download HarmBench, OR-Bench, ToxiGen, and sycophancy-eval; confirm each loads and you can see raw examples.
- [ ] Pick your capability/quality control metric now (e.g. a small MMLU subset, or perplexity on a held-out set) — this is what lets you claim "safety dropped independent of general quality," not just "the model got worse at everything."
- [ ] Decide and freeze: exact model, exact method list, exact axis list, exact per-axis sample size (start small — 100-200 prompts per axis is plenty for a first pass; you can expand later if compute allows). Write this into a config file. Do not change it mid-project without a documented reason.
- [ ] Load the smallest quantized variant on your actual hardware with a batch of 5 prompts per axis, end to end, before scaling anything up. If this fails, you've lost an hour, not a week.

## Phase 1 — Build the harness once

- [ ] Write one generation script: `run(model_path, method_name, dataset_name, config) -> raw_outputs.json`. Every axis and every method go through this same function — no bespoke per-axis scripts.
- [ ] Fix and log decoding settings (temperature, seed, max tokens, chat template used) so every condition is genuinely comparable. Log the exact settings alongside every output file, not just once in a README.
- [ ] Write per-axis scorers:
  - Refusal (under/over): keyword/pattern-based refusal classifier, applied to HarmBench and OR-Bench outputs.
  - Toxicity: ToxiGen's own classifier or a small open toxicity classifier.
  - Prompt injection: BIPIA's built-in attack-success-rate scorer (checks whether the injected instruction's action appears in the output).
  - Sycophancy: exact/fuzzy match comparing the model's answer before vs. after the "are you sure" pushback turn.
- [ ] Build a single results manifest (one row per model × method × axis × prompt_id) with a hash of the scorer version used, so you can trace any number back to exactly how it was produced.
- [ ] Add crash-safe checkpointing: write results to disk after every (model, method) pair completes, not just at the end of a run.

## Phase 2 — Tiny pilot (bug-catching, not results)

- [ ] Run ~20-30 prompts per axis on the baseline (best available precision) plus exactly one quantized method.
- [ ] Read every pilot output yourself. Look specifically for: wrong chat template artifacts, truncated generations, the refusal scorer flagging non-refusals (or missing real ones), BIPIA scorer false positives/negatives.
- [ ] Hand-label ~30 pilot outputs yourself per axis; compare against the automatic scorer. If agreement is weak, fix the scorer before scaling — don't proceed on a scorer you haven't checked.
- [ ] Time one full (model × method × axis) pass and extrapolate honestly to the full planned run. If the projected total is unrealistic for your remaining time, cut scope here — drop an axis or a method now, while it's cheap.

## Phase 3 — Full run

- [ ] Run the full frozen battery: baseline + all confirmed quantization methods × all five axes, using the harness from Phase 1.
- [ ] Confirm every (model, method) pair actually finished and was checkpointed — spot-check the manifest for gaps before moving on.
- [ ] Run the capability/quality control metric on every same (model, method) pair, so every safety-axis result has a matched quality number.
- [ ] Human-validate a fixed random sample (e.g. 50 items) per axis, sampled across all methods, and report agreement between your automatic scorers and your own labels.
- [ ] Only after all of the above is done and stable: if time remains, repeat the same frozen pipeline on a second model (same family or a different one) as a generalization check. Treat this as optional, not core — don't let it delay finishing analysis on model 1.

## Phase 4 — Analysis

- [ ] For each method × axis, compute the delta vs. baseline, alongside the matched capability-control delta.
- [ ] Flag "hidden-danger" cases specifically: quality/capability held steady or improved, while a safety axis dropped. This is your headline finding pattern if it appears — and a legitimate "no meaningful effect" result if it doesn't (still worth reporting honestly).
- [ ] Use a statistical test appropriate to your (small) sample size, and report confidence intervals, not just point deltas.
- [ ] Write a short, explicit "what we did not test, and why" paragraph — this preempts reviewers assuming you overlooked the excluded axes rather than deliberately scoped them out.

## Phase 5 — Writeup

- [ ] Draft related work citing the specific existing papers on quantization-method safety comparisons (quality-vs-safety proxy studies, per-method toxicity/hallucination comparisons, trustworthiness benchmark tracks) and state plainly what your five-axis, rule-scored, one-model-family setup adds that they don't.
- [ ] Reproducibility appendix: exact checkpoint names/commit hashes, exact library versions (llama.cpp, bitsandbytes, AutoAWQ, GPTQModel, HQQ), exact prompt sets and scorer code, all released.
- [ ] Pick the venue based on what actually got finished, not the venue you originally had in mind.
