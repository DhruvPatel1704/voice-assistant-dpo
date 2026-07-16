"""Model backends for the evaluation harness, behind one common interface.

ModelBackend.generate() is the single seam all generation passes through.

Hard rule (CLAUDE.md): real serving must route through the local
fine-tuned model served via vLLM, never the OpenAI API. VLLMClient talks
to vLLM's OpenAI-compatible server over HTTP (host is caller-supplied and
checked below) — it never talks to api.openai.com.

HFTransformersBackend loads a local HF `transformers` checkpoint directly,
in-process. It exists for CPU-only dev/smoke-testing the harness before a
GPU + vLLM server is available — it is not a substitute for vLLM serving.

The `openai` package is only imported lazily inside VLLMClient, so the HF
backend (and the harness generally) works without it installed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

try:
    from openai import OpenAI
except ImportError:  # optional dependency; not needed for the HF/CPU backend
    OpenAI = None

if TYPE_CHECKING:
    from eval.config import EndpointConfig

_FORBIDDEN_HOSTS = ("api.openai.com",)


@dataclass
class GenerationResult:
    text: str
    time_to_first_token: float
    total_latency: float


class ModelBackend(Protocol):
    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> GenerationResult: ...


class VLLMClient:
    """Generation via a local vLLM OpenAI-compatible server."""

    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY"):
        if any(host in base_url for host in _FORBIDDEN_HOSTS):
            raise ValueError(
                "VLLMClient must point at a local vLLM endpoint, not the OpenAI "
                "API. See CLAUDE.md hard rules."
            )
        if OpenAI is None:
            raise ImportError("The 'openai' package is required for VLLMClient (pip install openai).")
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> GenerationResult:
        start = time.perf_counter()
        first_token_time: float | None = None
        chunks: list[str] = []

        stream = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                chunks.append(delta)

        end = time.perf_counter()
        ttft = (first_token_time - start) if first_token_time is not None else (end - start)
        return GenerationResult(text="".join(chunks), time_to_first_token=ttft, total_latency=end - start)


class HFTransformersBackend:
    """Direct in-process generation via a local `transformers` checkpoint.

    CPU-only dev/smoke-test path — no server, no GPU, no vLLM. Loads the
    whole model into memory once at construction time.
    """

    def __init__(self, model: str, device: str = "cpu"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model
        self.device = device
        self._tokenizer = AutoTokenizer.from_pretrained(model)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(model).to(device)
        self._model.eval()

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> GenerationResult:
        import torch

        if seed is not None:
            torch.manual_seed(seed)

        start = time.perf_counter()
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        do_sample = temperature > 0.0
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)
        end = time.perf_counter()

        new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        # No token-level streaming here, so time-to-first-token == total latency.
        return GenerationResult(text=text, time_to_first_token=end - start, total_latency=end - start)


def make_backend(config: "EndpointConfig") -> ModelBackend:
    if config.backend == "vllm":
        return VLLMClient(base_url=config.base_url, model=config.model, api_key=config.api_key)
    if config.backend == "hf_transformers":
        return HFTransformersBackend(model=config.model, device=config.device)
    raise ValueError(f"Unknown backend: {config.backend!r}")
