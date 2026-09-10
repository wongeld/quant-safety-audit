"""
Quantizers for quant-safety-audit -- the producer side. loaders.py assumes a
quantized checkpoint already exists on disk; this module is what actually
produces one from a base model.

Same design as loaders.py, deliberately: one class per method, a common
interface (`run()`, which checks for an existing complete checkpoint before
doing any work), and a factory function so a new method is one new class +
one new branch, nothing else touched.

Where an official or well-maintained community checkpoint already exists,
prefer downloading it over self-quantizing -- that's what a real user
deploying this model would actually get, and it's faster. Where none exists
(or none is supplied), fall back to self-quantizing. GPTQ and AWQ take an
explicit `candidate_repo_ids` list for this rather than guessing a naming
pattern -- publishers don't follow one consistent convention across model
families, so a guessed pattern risks silently finding the wrong repo.
"""

from abc import ABC, abstractmethod
from typing import Optional, List
import os
import shutil


class BaseQuantizer(ABC):
    """Common interface every quantization method's producer implements."""

    def __init__(self, base_model_id: str, save_path: str):
        self.base_model_id = base_model_id
        self.save_path = save_path

    def already_done(self) -> bool:
        """Default completeness check: directory exists and contains at
        least one real weight file (not just config/tokenizer files) --
        this specific check exists because an earlier HQQ run silently
        produced a directory with no weight file at all, and a bare
        os.path.exists() check missed it."""
        if not os.path.isdir(self.save_path):
            return False
        files = os.listdir(self.save_path)
        weight_files = [f for f in files if not f.startswith("tokenizer") and not f.endswith(".json")]
        return len(weight_files) > 0

    @abstractmethod
    def quantize_and_save(self) -> None:
        ...

    def run(self) -> None:
        name = self.__class__.__name__
        if self.already_done():
            print(f"[{name}] already exists and looks complete at {self.save_path}, skipping.")
            return
        if os.path.exists(self.save_path):
            print(f"[{name}] found incomplete directory at {self.save_path}, cleaning up before retrying.")
            shutil.rmtree(self.save_path)
        print(f"[{name}] quantizing {self.base_model_id} -> {self.save_path} ...")
        self.quantize_and_save()
        if not self.already_done():
            print(f"[{name}] WARNING: finished without error, but no weight file found at {self.save_path}. "
                  f"Something is wrong -- inspect before trusting this checkpoint.")
        else:
            print(f"[{name}] done.")


class BnbQuantizer(BaseQuantizer):
    """Data-free: just loads with a BitsAndBytesConfig and saves. Produces
    two checkpoints in one pass to match this project's two bnb variants."""

    def __init__(self, base_model_id, save_path, load_in_4bit=True, load_in_8bit=False):
        super().__init__(base_model_id, save_path)
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit

    def quantize_and_save(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if self.load_in_4bit:
            config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        else:
            config = BitsAndBytesConfig(load_in_8bit=True)

        tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)
        model = AutoModelForCausalLM.from_pretrained(self.base_model_id, quantization_config=config, device_map="auto")
        model.save_pretrained(self.save_path)
        tokenizer.save_pretrained(self.save_path)
        del model
        torch.cuda.empty_cache()


class HqqQuantizer(BaseQuantizer):
    """Data-free, no calibration set needed -- fastest method to produce.
    Uses AutoHQQHFModel, the current recommended entry point."""

    def __init__(self, base_model_id, save_path, nbits=4, group_size=64):
        super().__init__(base_model_id, save_path)
        self.nbits = nbits
        self.group_size = group_size

    def quantize_and_save(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from hqq.models.hf.base import AutoHQQHFModel
        from hqq.core.quantize import BaseQuantizeConfig

        model = AutoModelForCausalLM.from_pretrained(self.base_model_id, torch_dtype=torch.float16, attn_implementation="eager")
        quant_config = BaseQuantizeConfig(nbits=self.nbits, group_size=self.group_size)
        AutoHQQHFModel.quantize_model(model, quant_config=quant_config)

        tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)
        os.makedirs(self.save_path, exist_ok=True)
        tokenizer.save_pretrained(self.save_path)
        AutoHQQHFModel.save_quantized(model, self.save_path)

        del model
        torch.cuda.empty_cache()


class GptqQuantizer(BaseQuantizer):
    """Tries each repo in candidate_repo_ids (official/community
    pre-quantized checkpoints) in order; downloads the first one found.
    Falls back to self-quantizing with gptqmodel, calibrated on a small
    wikitext slice, only if none of the candidates exist."""

    def __init__(self, base_model_id, save_path, candidate_repo_ids: Optional[List[str]] = None,
                 bits=4, group_size=128, calibration_size=128):
        super().__init__(base_model_id, save_path)
        self.candidate_repo_ids = candidate_repo_ids or []
        self.bits = bits
        self.group_size = group_size
        self.calibration_size = calibration_size

    def quantize_and_save(self):
        from huggingface_hub import HfApi, snapshot_download
        api = HfApi()

        for repo_id in self.candidate_repo_ids:
            try:
                api.model_info(repo_id)
                print(f"[GptqQuantizer] found existing checkpoint: {repo_id} -- downloading directly.")
                snapshot_download(repo_id=repo_id, local_dir=self.save_path)
                return
            except Exception:
                continue

        print("[GptqQuantizer] no candidate checkpoint found -- self-quantizing with gptqmodel. "
              "This needs a calibration set and real GPU time, expect it to be the slowest method here.")
        from gptqmodel import GPTQModel, QuantizeConfig
        from datasets import load_dataset

        calib_texts = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"]
        calib_texts = [t for t in calib_texts if t.strip()][: self.calibration_size]

        model = GPTQModel.load(self.base_model_id, QuantizeConfig(bits=self.bits, group_size=self.group_size))
        model.quantize(calib_texts)
        model.save(self.save_path)


class AwqQuantizer(BaseQuantizer):
    """Same pattern as GptqQuantizer: try known checkpoints first, self-
    quantize with AutoAWQ only as a fallback."""

    def __init__(self, base_model_id, save_path, candidate_repo_ids: Optional[List[str]] = None,
                 q_group_size=128, w_bit=4, version="GEMM"):
        super().__init__(base_model_id, save_path)
        self.candidate_repo_ids = candidate_repo_ids or []
        self.q_group_size = q_group_size
        self.w_bit = w_bit
        self.version = version

    def quantize_and_save(self):
        from huggingface_hub import HfApi, snapshot_download
        api = HfApi()

        for repo_id in self.candidate_repo_ids:
            try:
                api.model_info(repo_id)
                print(f"[AwqQuantizer] found existing checkpoint: {repo_id} -- downloading directly.")
                snapshot_download(repo_id=repo_id, local_dir=self.save_path)
                return
            except Exception:
                continue

        print("[AwqQuantizer] no candidate checkpoint found -- self-quantizing with AutoAWQ. "
              "This needs a calibration set and real GPU time.")
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer

        model = AutoAWQForCausalLM.from_pretrained(self.base_model_id)
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)
        model.quantize(tokenizer, quant_config={
            "zero_point": True, "q_group_size": self.q_group_size, "w_bit": self.w_bit, "version": self.version,
        })
        model.save_quantized(self.save_path)
        tokenizer.save_pretrained(self.save_path)


class AutoRoundQuantizer(BaseQuantizer):
    """Self-quantizes with Intel's AutoRound -- no widely available
    pre-quantized checkpoint ecosystem for this method yet, so this always
    self-quantizes rather than trying candidates first."""

    def __init__(self, base_model_id, save_path, bits=4, group_size=128,
                 dataset="NeelNanda/pile-10k", batch_size=2, seqlen=512, iters=200):
        super().__init__(base_model_id, save_path)
        self.bits = bits
        self.group_size = group_size
        self.dataset = dataset
        self.batch_size = batch_size
        self.seqlen = seqlen
        self.iters = iters

    def quantize_and_save(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from auto_round import AutoRound

        model = AutoModelForCausalLM.from_pretrained(self.base_model_id, torch_dtype=torch.float16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)

        autoround = AutoRound(
            model, tokenizer, bits=self.bits, group_size=self.group_size,
            dataset=self.dataset, batch_size=self.batch_size, seqlen=self.seqlen, iters=self.iters,
        )
        autoround.quantize()
        autoround.save_quantized(self.save_path, format="auto_round")

        del model, autoround
        torch.cuda.empty_cache()


class GgufDownloader(BaseQuantizer):
    """Not a quantizer in the training sense -- downloads a pre-built
    community GGUF k-quant file. Needs an explicit repo_id + filename since
    GGUF naming conventions vary a lot by uploader; there's no safe way to
    guess this from base_model_id alone."""

    def __init__(self, base_model_id, save_path, repo_id: str, filename: str):
        super().__init__(base_model_id, save_path)
        self.repo_id = repo_id
        self.filename = filename

    def already_done(self) -> bool:
        return os.path.isfile(self.save_path)

    def run(self) -> None:
        # GGUF is a single file, not a directory -- the base class's
        # cleanup-and-retry logic assumes a directory, so this overrides
        # run() entirely rather than reusing it.
        name = self.__class__.__name__
        if self.already_done():
            print(f"[{name}] already downloaded at {self.save_path}, skipping.")
            return
        print(f"[{name}] downloading {self.repo_id}/{self.filename} ...")
        self.quantize_and_save()
        print(f"[{name}] done." if self.already_done() else f"[{name}] WARNING: download finished but file not found.")

    def quantize_and_save(self):
        from huggingface_hub import hf_hub_download
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        hf_hub_download(repo_id=self.repo_id, filename=self.filename, local_dir=os.path.dirname(self.save_path))


def build_quantizer(method: str, base_model_id: str, save_path: str, **kwargs) -> BaseQuantizer:
    """Factory mirroring build_loader() in loaders.py. `method` is one of
    'bnb-nf4' | 'bnb-int8' | 'hqq' | 'gptq' | 'awq' | 'autoround' | 'gguf'.
    """
    if method == "bnb-nf4":
        return BnbQuantizer(base_model_id, save_path, load_in_4bit=True, **kwargs)
    if method == "bnb-int8":
        return BnbQuantizer(base_model_id, save_path, load_in_4bit=False, load_in_8bit=True, **kwargs)
    if method == "hqq":
        return HqqQuantizer(base_model_id, save_path, **kwargs)
    if method == "gptq":
        return GptqQuantizer(base_model_id, save_path, **kwargs)
    if method == "awq":
        return AwqQuantizer(base_model_id, save_path, **kwargs)
    if method == "autoround":
        return AutoRoundQuantizer(base_model_id, save_path, **kwargs)
    if method == "gguf":
        return GgufDownloader(base_model_id, save_path, repo_id=kwargs["repo_id"], filename=kwargs["filename"])
    raise ValueError(f"Unknown method: {method!r}. Add a branch here and a class above to support it.")
