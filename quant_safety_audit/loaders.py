"""
Model loaders for quant-safety-audit.

Every quantization method needs a genuinely different loading API:
- bitsandbytes / GPTQ / AutoRound load through plain `transformers`, differing
  only in which `quantization_config` gets passed at load time.
- AWQ needs AutoAWQ's own loader class -- a plain `AutoModelForCausalLM`
  does not correctly dispatch its kernels.
- HQQ needs its own loader class (`AutoHQQHFModel`).
- GGUF loads through llama-cpp-python, not transformers at all.

BaseModelLoader is the common interface that hides all of that from the rest
of the pipeline: load(), generate(), unload(), used as a context manager so
a crash mid-generation still frees GPU memory.

Adding a new quantization method later means adding one subclass here and one
branch in build_loader() -- nothing else in this package needs to change.
"""

from abc import ABC, abstractmethod
from typing import Optional
import gc


def log_gpu_memory(tag: str):
    import torch
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"  [mem] {tag}: {allocated:.2f} GB allocated, {peak:.2f} GB peak")


class BaseModelLoader(ABC):
    """Common interface every quantization method's loader implements."""

    def __init__(self, name: str, path: str, max_new_tokens: int = 512):
        self.name = name
        self.path = path
        self.max_new_tokens = max_new_tokens
        self._loaded = False

    @abstractmethod
    def load(self) -> None:
        ...

    @abstractmethod
    def generate(self, user_prompt: str, system_prompt: Optional[str] = None,
                 max_new_tokens: Optional[int] = None) -> str:
        ...

    @abstractmethod
    def unload(self) -> None:
        ...

    def __enter__(self):
        print(f"[{self.name}] loading...")
        self.load()
        self._loaded = True
        log_gpu_memory(f"{self.name} loaded")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import torch
        if self._loaded:
            self.unload()
            self._loaded = False
            log_gpu_memory(f"{self.name} freed")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        return False  # never swallow exceptions


class TransformersLoader(BaseModelLoader):
    """Covers bitsandbytes, GPTQ, and AutoRound checkpoints. bitsandbytes
    checkpoints auto-detect their quantization from the saved config.json
    (quantization_config=None here); GPTQ and AutoRound need it passed
    explicitly at load time -- confirmed necessary against a real checkpoint,
    not a guess. See build_loader() for which config each method gets."""

    def __init__(self, name, path, quantization_config=None, torch_dtype=None, **kwargs):
        super().__init__(name, path, **kwargs)
        self.quantization_config = quantization_config
        self.torch_dtype = torch_dtype
        self.model = None
        self.tokenizer = None

    def load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.path)
        kwargs = {"device_map": "auto"}
        if self.quantization_config is not None:
            kwargs["quantization_config"] = self.quantization_config
        if self.torch_dtype is not None:
            kwargs["torch_dtype"] = self.torch_dtype
        self.model = AutoModelForCausalLM.from_pretrained(self.path, **kwargs)

    def generate(self, user_prompt, system_prompt=None, max_new_tokens=None):
        import torch
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True,
        ).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=False, pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def unload(self):
        del self.model, self.tokenizer
        self.model = None
        self.tokenizer = None


class AwqLoader(BaseModelLoader):
    """AWQ checkpoints need AutoAWQ's own loader class."""

    def __init__(self, name, path, **kwargs):
        super().__init__(name, path, **kwargs)
        self.model = None
        self.tokenizer = None

    def load(self):
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.path)
        self.model = AutoAWQForCausalLM.from_quantized(self.path, fuse_layers=False, safetensors=True)

    def generate(self, user_prompt, system_prompt=None, max_new_tokens=None):
        import torch
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=False, pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def unload(self):
        del self.model, self.tokenizer
        self.model = None
        self.tokenizer = None


class HqqLoader(BaseModelLoader):
    """HQQ checkpoints need AutoHQQHFModel -- the current recommended entry
    point (hqq.models.hf.base), not the older HQQModelForCausalLM."""

    def __init__(self, name, path, **kwargs):
        super().__init__(name, path, **kwargs)
        self.model = None
        self.tokenizer = None

    def load(self):
        import torch
        from hqq.models.hf.base import AutoHQQHFModel
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.path)
        self.model = AutoHQQHFModel.from_quantized(
            self.path, device="cuda" if torch.cuda.is_available() else "cpu",
        )

    def generate(self, user_prompt, system_prompt=None, max_new_tokens=None):
        import torch
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True,
        )
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=False, pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def unload(self):
        del self.model, self.tokenizer
        self.model = None
        self.tokenizer = None


class GgufLlamaCppLoader(BaseModelLoader):
    """GGUF checkpoints load through llama-cpp-python, not transformers --
    confirmed working (your test run loaded it successfully). Uses
    create_chat_completion so system/user roles and the GGUF's own embedded
    chat template are respected, matching how every other loader here builds
    its prompt -- the plain single-string completion mode used in the
    original sanity-check cell does NOT apply a chat template, which matters
    for MASK's system+user structured prompts."""

    def __init__(self, name, path, n_ctx=2048, n_gpu_layers=-1, **kwargs):
        super().__init__(name, path, **kwargs)
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.model = None

    def load(self):
        from llama_cpp import Llama
        self.model = Llama(
            model_path=self.path, n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers, verbose=False,
        )

    def generate(self, user_prompt, system_prompt=None, max_new_tokens=None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        out = self.model.create_chat_completion(
            messages=messages, max_tokens=max_new_tokens or self.max_new_tokens,
            temperature=0.0,  # matches do_sample=False elsewhere: greedy/deterministic
        )
        return out["choices"][0]["message"]["content"]

    def unload(self):
        del self.model
        self.model = None


def build_loader(name: str, spec: dict, max_new_tokens: int = 512) -> BaseModelLoader:
    """Factory: spec = {"method": ..., "path": ...}. `method` is one of
    'baseline' | 'bnb' | 'gptq' | 'autoround' | 'awq' | 'hqq' | 'gguf'.
    Adding a method later: one new elif branch here, one new class above --
    nothing in benchmarks.py, judges.py, or pipeline.py needs to change.
    """
    method = spec["method"]
    path = spec["path"]

    if method in ("baseline", "bnb"):
        import torch
        dtype = torch.bfloat16 if method == "baseline" else None
        return TransformersLoader(name, path, quantization_config=None, torch_dtype=dtype, max_new_tokens=max_new_tokens)
    if method == "gptq":
        from transformers import GPTQConfig
        return TransformersLoader(name, path, quantization_config=GPTQConfig(bits=4, backend="torch"), max_new_tokens=max_new_tokens)
    if method == "autoround":
        from transformers import AutoRoundConfig
        return TransformersLoader(name, path, quantization_config=AutoRoundConfig(backend="torch"), max_new_tokens=max_new_tokens)
    if method == "awq":
        return AwqLoader(name, path, max_new_tokens=max_new_tokens)
    if method == "hqq":
        return HqqLoader(name, path, max_new_tokens=max_new_tokens)
    if method == "gguf":
        return GgufLlamaCppLoader(name, path, max_new_tokens=max_new_tokens)
    raise ValueError(f"Unknown method: {method!r}. Add a branch here and a loader class above to support it.")
