#!/usr/bin/env bash
# One-time environment setup for running quant-safety-audit on a bare Linux
# GPU instance (not Colab -- no pre-populated Python environment to work
# around, so this just installs everything once, cleanly, before Python is
# ever launched).
#
# Run this once after cloning the repo. Re-run only if you change dependency
# versions later.
set -euo pipefail

echo "=== Confirming GPU is visible ==="
nvidia-smi

echo "=== Creating virtual environment ==="
python3 -m venv .venv
source .venv/bin/activate

echo "=== Installing core dependencies ==="
pip install -q -U pip
pip install -q -U transformers accelerate huggingface_hub datasets pandas

echo "=== Installing quantization backends ==="
pip install -q -U bitsandbytes hqq autoawq gptqmodel auto-round

echo "=== Installing llama-cpp-python with CUDA support ==="
# A plain `pip install llama-cpp-python` commonly gives a CPU-only wheel --
# explicit build flag needed for the gguf-q4km variant to actually use the GPU.
CMAKE_ARGS="-DGGML_CUDA=on" pip install -q --force-reinstall --no-cache-dir llama-cpp-python

echo "=== Reconciling numpy/scipy/scikit-learn ==="
# Several of the backends above have loose numpy constraints and can pull in
# a version whose ABI doesn't match scipy/scikit-learn. Reconciling this now,
# in its own step before Python is ever launched, avoids the whole
# restart-the-kernel problem the Colab notebooks needed to work around --
# there's no live process here yet for an installed package to go stale in.
pip install -q -U --force-reinstall numpy scipy scikit-learn

echo "=== Verifying core imports ==="
python3 -c "
import numpy, scipy, sklearn, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
print('numpy', numpy.__version__, '/ scipy', scipy.__version__, '/ sklearn', sklearn.__version__)
print('torch', torch.__version__, '-- CUDA available:', torch.cuda.is_available())
"

echo "=== Setup complete ==="
echo "Activate this environment in future shells (including new tmux panes) with:"
echo "  source $(pwd)/.venv/bin/activate"
