#!/usr/bin/env python3
"""
Produce all quantized checkpoints for the base model, unattended -- safe to
launch under tmux and walk away from. This is the step that has to run
before run_mask_pipeline.py (or any other benchmark script): that script
checks for these checkpoints and refuses to start if they're missing.

Config (all optional, sensible defaults matching the notebooks):
    QSA_OUTPUT_DIR   where checkpoints get saved (default: ~/quant-safety-audit-data)
    QSA_BASE_MODEL   base model id to quantize (default: Qwen/Qwen2.5-3B-Instruct)

To switch to a different model family (e.g. Llama), set QSA_BASE_MODEL and
also update GPTQ_CANDIDATES / AWQ_CANDIDATES / GGUF_REPO_ID / GGUF_FILENAME
below -- these can't be safely auto-derived from the base model id, since
publishers don't follow one consistent naming convention across families.
Getting one of these wrong just means that method falls back to
self-quantizing (GPTQ/AWQ) or fails loudly with a clear 404 (GGUF) --
nothing silently uses the wrong checkpoint.

Run with `python -u` (unbuffered) so a log file updates in real time:
    python -u scripts/quantize_models.py 2>&1 | tee -a logs/quantize_$(date +%Y%m%d_%H%M%S).log

Every step is idempotent -- already-complete checkpoints are skipped, so
this is safe to re-run after a crash or an interrupted step.
"""
import os
import sys
import time


def ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{ts()}] {msg}", flush=True)


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
BASE_MODEL_ID = os.environ.get("QSA_BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
MODELS_DIR = f"{OUTPUT_DIR}/models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Checked first for GPTQ/AWQ -- downloaded directly if any exists, since
# that's what a real user deploying this model would actually get, and it's
# faster than self-quantizing. Falls back to self-quantizing if none match.
# UPDATE THESE when switching base models.
GPTQ_CANDIDATES = [x for x in os.environ.get("QSA_GPTQ_CANDIDATES", "").split(";") if x] or ([f"{BASE_MODEL_ID}-GPTQ-Int4"] if "Qwen" in BASE_MODEL_ID else [])
AWQ_CANDIDATES = [x for x in os.environ.get("QSA_AWQ_CANDIDATES", "").split(";") if x] or ([f"{BASE_MODEL_ID}-AWQ"] if "Qwen" in BASE_MODEL_ID else [])

# GGUF has no safe auto-derivation at all -- naming varies too much by
# uploader. UPDATE THESE when switching base models (find bartowski's or
# another well-maintained uploader's repo for your chosen model).
GGUF_REPO_ID = os.environ.get("QSA_GGUF_REPO_ID", "bartowski/Qwen2.5-3B-Instruct-GGUF")
GGUF_FILENAME = os.environ.get("QSA_GGUF_FILENAME", "Qwen2.5-3B-Instruct-Q4_K_M.gguf")

log("=" * 70)
log("Quantization run starting")
log(f"  OUTPUT_DIR:     {OUTPUT_DIR}")
log(f"  BASE_MODEL_ID:  {BASE_MODEL_ID}")
log(f"  GPTQ candidates: {GPTQ_CANDIDATES or '(none -- will self-quantize)'}")
log(f"  AWQ candidates:  {AWQ_CANDIDATES or '(none -- will self-quantize)'}")
log(f"  GGUF source:     {GGUF_REPO_ID} / {GGUF_FILENAME}")
log("=" * 70)

# ---------------------------------------------------------------------------
# Each quantizer is independent -- if one fails, log it and move on to the
# rest rather than aborting the whole run over one method.
# ---------------------------------------------------------------------------
jobs = [
    ("bnb-nf4", qsa.build_quantizer("bnb-nf4", BASE_MODEL_ID, f"{MODELS_DIR}/bnb-nf4")),
    ("bnb-int8", qsa.build_quantizer("bnb-int8", BASE_MODEL_ID, f"{MODELS_DIR}/bnb-int8")),
    ("hqq-4bit", qsa.build_quantizer("hqq", BASE_MODEL_ID, f"{MODELS_DIR}/hqq-4bit")),
    ("gptq-int4", qsa.build_quantizer("gptq", BASE_MODEL_ID, f"{MODELS_DIR}/gptq-int4", candidate_repo_ids=GPTQ_CANDIDATES)),
    ("awq-int4", qsa.build_quantizer("awq", BASE_MODEL_ID, f"{MODELS_DIR}/awq-int4", candidate_repo_ids=AWQ_CANDIDATES)),
    ("autoround-int4", qsa.build_quantizer("autoround", BASE_MODEL_ID, f"{MODELS_DIR}/autoround-int4")),
    ("gguf-q4km", qsa.build_quantizer("gguf", BASE_MODEL_ID, f"{MODELS_DIR}/gguf/{GGUF_FILENAME}", repo_id=GGUF_REPO_ID, filename=GGUF_FILENAME)),
]

start = time.time()
results = {}

for name, quantizer in jobs:
    job_start = time.time()
    log(f"--- {name} ---")
    try:
        quantizer.run()
        results[name] = "OK"
    except Exception as e:
        import traceback
        log(f"[{name}] FAILED: {type(e).__name__}: {e!r}")
        log(traceback.format_exc())
        results[name] = f"FAILED: {e!r}"
    log(f"[{name}] {time.time() - job_start:.0f}s elapsed.")

log("=" * 70)
log("Summary:")
for name, status in results.items():
    log(f"  {name:16s} {status}")
log(f"Total elapsed: {time.time() - start:.0f}s")
log(f"Checkpoints saved under: {MODELS_DIR}")

failed = [name for name, status in results.items() if status != "OK"]
if failed:
    log(f"\n{len(failed)} method(s) failed: {failed}. Fix and re-run this script -- ")
    log("already-completed methods will be skipped, only the failed ones retry.")
    sys.exit(1)

log("\nAll methods completed. Ready for scripts/run_mask_pipeline.py.")
