"""Import/construction check for sft.train -- no model, dataset, or GPU/bitsandbytes.

Stands in for "run it" per CLAUDE.md's build-before-you-train ordering: this
confirms the training module wires together correctly (imports cleanly on
CPU, builds a real peft.LoraConfig, builds a real formatting_func against a
fake row) without downloading Qwen2.5-3B, the full Alpaca split, or requiring
a GPU/bitsandbytes.
"""

from __future__ import annotations

from functools import partial

from sft.config import DatasetConfig, LoraConfig as LoraSettings, load_config
from sft.dataset import format_example
from sft.train import build_lora_config, build_trainer, load_model, load_tokenizer


def test_build_lora_config_produces_peft_config():
    from peft import LoraConfig as PeftLoraConfig

    settings = LoraSettings(r=8, alpha=16, dropout=0.1, target_modules=["q_proj", "v_proj"])
    peft_config = build_lora_config(settings)

    assert isinstance(peft_config, PeftLoraConfig)
    assert peft_config.r == 8
    assert peft_config.lora_alpha == 16
    assert peft_config.target_modules == {"q_proj", "v_proj"}
    assert peft_config.task_type == "CAUSAL_LM"


def test_prepare_dataset_formatting_func_needs_no_network():
    dataset_config = DatasetConfig(num_samples=0)
    # We only exercise the formatting_func here, not load_and_format_dataset
    # (which calls datasets.load_dataset and would hit the network/cache).
    formatting_func = partial(format_example, config=dataset_config)
    example = {"instruction": "Say hi.", "input": "", "output": "Hi!"}
    assert "Say hi." in formatting_func(example)


def test_build_trainer_constructs_against_cpu_smoke_config(tmp_path, monkeypatch):
    """Actually constructs TRL's SFTConfig/SFTTrainer via build_trainer, using
    the cpu_smoke config's real (tiny, already-cached) model. build_lora_config
    alone can't catch TRL API drift -- e.g. SFTConfig's max_seq_length ->
    max_length rename -- since it never touches SFTConfig/SFTTrainer.

    Uses in-memory fake rows instead of downloading Alpaca, and forces
    HF_HUB_OFFLINE so this fails loudly (not via network hang) if the model
    isn't cached.
    """
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    from datasets import Dataset

    config = load_config("sft/configs/sft.cpu_smoke.yaml")
    config.output_dir = str(tmp_path)

    tokenizer = load_tokenizer(config)
    model = load_model(config)
    peft_config = build_lora_config(config.lora)

    fake_rows = [{"instruction": "Say hi.", "input": "", "output": "Hi!"}] * 4
    dataset = Dataset.from_list(fake_rows)
    formatting_func = partial(format_example, config=config.dataset)

    trainer = build_trainer(config, model, tokenizer, dataset, formatting_func, peft_config)

    assert trainer.args.max_length == config.training.max_seq_length
