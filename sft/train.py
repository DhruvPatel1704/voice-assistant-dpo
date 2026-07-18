"""SFT training entrypoint: LoRA/QLoRA fine-tuning via TRL's SFTTrainer.

Usage (once a GPU is available -- see sft/README.md):
    python -m sft.train --config sft/configs/sft.yaml

Nothing in this module downloads a model or dataset at import time -- all of
that happens inside the functions below, only when called from main().
"""

from __future__ import annotations

import argparse
from functools import partial

from sft.config import DatasetConfig, LoraConfig as LoraSettings, SFTConfig, load_config, set_seed
from sft.dataset import format_example, load_and_format_dataset


def load_tokenizer(config: SFTConfig):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(config: SFTConfig):
    import torch
    from transformers import AutoModelForCausalLM

    quantization_config = None
    if config.use_qlora:
        # GPU-only: bitsandbytes has no usable CPU build, so import it lazily
        # here rather than at module load time (see README "QLoRA" section).
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    return AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if config.training.bf16 else None,
    )


def build_lora_config(lora_settings: LoraSettings):
    from peft import LoraConfig

    return LoraConfig(
        r=lora_settings.r,
        lora_alpha=lora_settings.alpha,
        lora_dropout=lora_settings.dropout,
        target_modules=lora_settings.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )


def prepare_dataset(dataset_config: DatasetConfig):
    dataset = load_and_format_dataset(dataset_config)
    formatting_func = partial(format_example, config=dataset_config)
    return dataset, formatting_func


def build_trainer(config: SFTConfig, model, tokenizer, dataset, formatting_func, peft_config):
    from trl import SFTConfig as TRLSFTConfig
    from trl import SFTTrainer

    training_args = TRLSFTConfig(
        output_dir=config.output_dir,
        learning_rate=config.training.learning_rate,
        num_train_epochs=config.training.epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        max_length=config.training.max_seq_length,
        bf16=config.training.bf16,
        logging_steps=config.training.logging_steps,
        save_strategy=config.training.save_strategy,
        seed=config.seed,
        report_to="wandb" if config.wandb.mode != "disabled" else "none",
    )

    return SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        formatting_func=formatting_func,
        peft_config=peft_config,
        processing_class=tokenizer,
    )


def run(config: SFTConfig) -> None:
    set_seed(config.seed)

    if config.wandb.mode != "disabled":
        import wandb

        wandb.init(project=config.wandb.project, config=config.__dict__, mode=config.wandb.mode)

    tokenizer = load_tokenizer(config)
    model = load_model(config)
    dataset, formatting_func = prepare_dataset(config.dataset)
    peft_config = build_lora_config(config.lora)

    trainer = build_trainer(config, model, tokenizer, dataset, formatting_func, peft_config)
    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LoRA/QLoRA SFT on a base model")
    parser.add_argument("--config", default="sft/configs/sft.yaml")
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
