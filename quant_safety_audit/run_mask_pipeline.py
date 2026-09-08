#!/usr/bin/env python3
"""
Run the MASK honesty pilot end to end, unattended -- safe to launch under
tmux and walk away from. No notebook, no interactive prompts.

Requires:
    HF_TOKEN environment variable set (MASK's dataset is gated on Hugging
    Face). This script does NOT fall back to an interactive login prompt --
    that would hang forever in a detached tmux session with nobody there to
    respond to it. Set it before running:
        export HF_TOKEN=hf_xxxxxxxx

Config (all optional, sensible defaults matching the notebooks):
    QSA_OUTPUT_DIR      where checkpoints/results live (default: ~/quant-safety-audit-data)
    QSA_BASE_MODEL      base model id (default: Qwen/Qwen2.5-3B-Instruct)
    QSA_JUDGE_MODEL     judge model id (default: Qwen/Qwen2.5-14B-Instruct)
    QSA_MASK_ARCHETYPE  MASK archetype (default: known_facts)
    QSA_N_ITEMS         number of MASK items to run (default: 5)

Run with `python -u` (unbuffered) so a log file updates in real time rather
than only flushing in large chunks:
    python -u scripts/run_mask_pipeline.py 2>&1 | tee -a logs/mask_run_$(date +%Y%m%d_%H%M%S).log

This script does NOT self-quantize -- it expects the 6 quantized checkpoints
to already exist under QSA_OUTPUT_DIR/models/ (produced by the setup/
quantization notebook, run once on this same machine). It checks for them
and fails loudly, before spending any GPU time, if any are missing.
"""
import os
import sys
import time


def ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{ts()}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Hard requirement: HF_TOKEN must already be set. No interactive fallback --
# this is the one thing that would otherwise hang a background run forever.
# ---------------------------------------------------------------------------
if not os.environ.get("HF_TOKEN"):
    log("ERROR: HF_TOKEN environment variable is not set.")
    log("MASK's dataset is gated on Hugging Face and this script cannot")
    log("prompt for login in the background. Set it before running:")
    log("  export HF_TOKEN=hf_xxxxxxxx")
    sys.exit(1)

from huggingface_hub import login  # noqa: E402
login(token=os.environ["HF_TOKEN"])
log("Logged in to Hugging Face.")

# ---------------------------------------------------------------------------
# Make the framework package importable regardless of the current working
# directory this script is launched from.
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import quant_safety_audit as qsa  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.environ.get("QSA_OUTPUT_DIR", os.path.expanduser("~/quant-safety-audit-data"))
RESULTS_DIR = f"{OUTPUT_DIR}/results/mask"
BASE_MODEL_ID = os.environ.get("QSA_BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
JUDGE_MODEL_ID = os.environ.get("QSA_JUDGE_MODEL", "Qwen/Qwen2.5-14B-Instruct")
MASK_ARCHETYPE = os.environ.get("QSA_MASK_ARCHETYPE", "known_facts")
N_ITEMS = int(os.environ.get("QSA_N_ITEMS", "5"))

# Matches scripts/quantize_models.py's GGUF_FILENAME -- was hardcoded to the
# Qwen filename here before, which would have silently looked for a Qwen
# GGUF file even after switching QSA_BASE_MODEL to Llama. Now driven by the
# same env var so both scripts stay in sync when the base model changes.
GGUF_FILENAME = os.environ.get("QSA_GGUF_FILENAME", "Qwen2.5-3B-Instruct-Q4_K_M.gguf")

os.makedirs(f"{OUTPUT_DIR}/models", exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_VARIANTS = {
    "baseline-bf16":  {"method": "baseline",  "path": BASE_MODEL_ID},
    "bnb-nf4":        {"method": "bnb",       "path": f"{OUTPUT_DIR}/models/bnb-nf4"},
    "bnb-int8":       {"method": "bnb",       "path": f"{OUTPUT_DIR}/models/bnb-int8"},
    "hqq-4bit":       {"method": "hqq",       "path": f"{OUTPUT_DIR}/models/hqq-4bit"},
    "gptq-int4":      {"method": "gptq",      "path": f"{OUTPUT_DIR}/models/gptq-int4"},
    "awq-int4":       {"method": "awq",       "path": f"{OUTPUT_DIR}/models/awq-int4"},
    "autoround-int4": {"method": "autoround", "path": f"{OUTPUT_DIR}/models/autoround-int4"},
    "gguf-q4km":      {"method": "gguf",      "path": f"{OUTPUT_DIR}/models/gguf/{GGUF_FILENAME}"},
}

log("=" * 70)
log("MASK pipeline run starting")
log(f"  OUTPUT_DIR:     {OUTPUT_DIR}")
log(f"  RESULTS_DIR:    {RESULTS_DIR}")
log(f"  BASE_MODEL_ID:  {BASE_MODEL_ID}")
log(f"  JUDGE_MODEL_ID: {JUDGE_MODEL_ID}")
log(f"  MASK_ARCHETYPE: {MASK_ARCHETYPE}")
log(f"  N_ITEMS:        {N_ITEMS}")
log(f"  variants:       {list(MODEL_VARIANTS.keys())}")
log("=" * 70)

# ---------------------------------------------------------------------------
# Pre-flight: confirm every checkpoint actually exists before spending any
# GPU time. Failing loudly now beats discovering a missing checkpoint three
# hours into an unattended run.
# ---------------------------------------------------------------------------
missing = []
for name, spec in MODEL_VARIANTS.items():
    if spec["method"] == "baseline":
        continue
    path = spec["path"]
    exists = os.path.isfile(path) if spec["method"] == "gguf" else os.path.isdir(path)
    if not exists:
        missing.append((name, path))

if missing:
    log("ERROR: the following checkpoints are missing. This script does not")
    log("self-quantize -- run scripts/quantize_models.py first (with the same")
    log("QSA_OUTPUT_DIR / QSA_BASE_MODEL env vars set), then re-run this script.")
    for name, path in missing:
        log(f"  {name}: {path}")
    sys.exit(1)

log("All checkpoint paths confirmed present.")

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
start = time.time()

benchmark = qsa.MaskBenchmark(archetype=MASK_ARCHETYPE)
judge = qsa.MaskOpenLLMJudge(judge_model_id=JUDGE_MODEL_ID)

runner = qsa.ExperimentRunner(
    loader_configs=MODEL_VARIANTS, benchmark=benchmark, judge=judge,
    results_dir=RESULTS_DIR, n_items=N_ITEMS,
)

log("Starting generation phase...")
raw_df = runner.run_generation()
log(f"Generation phase complete -- {len(raw_df)} rows, {time.time() - start:.0f}s elapsed.")

log("Starting judging phase...")
judge_start = time.time()
scored_df = runner.run_judging(raw_df)
log(f"Judging phase complete -- {len(scored_df)} rows, {time.time() - judge_start:.0f}s elapsed.")

summary = runner.summarize(scored_df)
log("Summary:")
for line in summary.to_string(index=False).splitlines():
    log(f"  {line}")

log(f"Done. Total elapsed: {time.time() - start:.0f}s.")
log(f"Results saved under: {RESULTS_DIR}")
