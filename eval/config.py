"""Config-driven settings for evaluation runs. See CLAUDE.md: config-driven, seeded runs."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class EndpointConfig:
    name: str
    model: str
    backend: str = "vllm"  # "vllm" (real serving) or "hf_transformers" (CPU dev/smoke test)
    base_url: str = ""  # required for backend="vllm"
    api_key: str = "EMPTY"
    device: str = "cpu"  # used by backend="hf_transformers"


@dataclass
class GenerationConfig:
    max_tokens: int = 256
    temperature: float = 0.0


@dataclass
class JudgeConfig:
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    enabled: bool = True


@dataclass
class WandbConfig:
    project: str = "voice-assistant-eval"
    mode: str = "disabled"


@dataclass
class EvalConfig:
    seed: int
    prompts_path: str
    baseline: EndpointConfig
    candidate: EndpointConfig
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    output_dir: str = "eval/results"


def load_config(path: str | Path) -> EvalConfig:
    with Path(path).open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return EvalConfig(
        seed=raw["seed"],
        prompts_path=raw["prompts_path"],
        baseline=EndpointConfig(**raw["baseline"]),
        candidate=EndpointConfig(**raw["candidate"]),
        generation=GenerationConfig(**raw.get("generation", {})),
        judge=JudgeConfig(**raw.get("judge", {})),
        wandb=WandbConfig(**raw.get("wandb", {})),
        output_dir=raw.get("output_dir", "eval/results"),
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
