"""Parses both SFT configs into dataclasses -- no model/dataset downloads."""

from __future__ import annotations

from sft.config import SFTConfig, load_config


def test_load_real_config():
    config = load_config("sft/configs/sft.yaml")
    assert isinstance(config, SFTConfig)
    assert config.base_model == "Qwen/Qwen2.5-3B-Instruct"
    assert config.dataset.name == "tatsu-lab/alpaca"
    assert config.lora.r == 16
    assert config.training.epochs == 3
    assert config.use_qlora is False


def test_load_cpu_smoke_config():
    config = load_config("sft/configs/sft.cpu_smoke.yaml")
    assert config.base_model == "sshleifer/tiny-gpt2"
    assert config.dataset.num_samples == 8
    assert config.training.bf16 is False


def test_defaults_fill_in_when_section_omitted(tmp_path):
    minimal = tmp_path / "minimal.yaml"
    minimal.write_text("seed: 1\nbase_model: dummy\noutput_dir: out\n")

    config = load_config(minimal)
    assert config.dataset.name == "tatsu-lab/alpaca"
    assert config.lora.r == 16
    assert config.training.learning_rate == 2e-4
    assert config.wandb.mode == "disabled"
