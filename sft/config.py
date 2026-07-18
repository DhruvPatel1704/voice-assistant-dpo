"""Config-driven settings for SFT training runs. See CLAUDE.md: config-driven, seeded runs."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class LoraConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )


@dataclass
class DatasetConfig:
    name: str = "tatsu-lab/alpaca"
    split: str = "train"
    num_samples: int = 200  # slice for fast iteration; -1 means use the full split
    instruction_field: str = "instruction"
    input_field: str = "input"
    output_field: str = "output"


@dataclass
class TrainingConfig:
    learning_rate: float = 2e-4
    epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 1024
    bf16: bool = True
    logging_steps: int = 10
    save_strategy: str = "epoch"


@dataclass
class WandbConfig:
    project: str = "voice-assistant-sft"
    mode: str = "disabled"


@dataclass
class SFTConfig:
    seed: int
    base_model: str
    output_dir: str
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    use_qlora: bool = False  # 4-bit base weights via bitsandbytes; GPU-only, see README


def load_config(path: str | Path) -> SFTConfig:
    with Path(path).open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return SFTConfig(
        seed=raw["seed"],
        base_model=raw["base_model"],
        output_dir=raw["output_dir"],
        dataset=DatasetConfig(**raw.get("dataset", {})),
        lora=LoraConfig(**raw.get("lora", {})),
        training=TrainingConfig(**raw.get("training", {})),
        wandb=WandbConfig(**raw.get("wandb", {})),
        use_qlora=raw.get("use_qlora", False),
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
