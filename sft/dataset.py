"""Loads and formats an Alpaca-style instruction dataset for SFTTrainer.

`format_example` is pure Python (no tokenizer, no network) so it can be
unit-tested on CPU against fake rows; it is passed to `SFTTrainer` as
`formatting_func`, which handles tokenization internally.
"""

from __future__ import annotations

from sft.config import DatasetConfig

PROMPT_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
)

PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n{output}"
)


def format_example(example: dict, config: DatasetConfig) -> str:
    instruction = example[config.instruction_field]
    output = example[config.output_field]
    input_text = example.get(config.input_field, "")

    if input_text:
        return PROMPT_WITH_INPUT.format(instruction=instruction, input=input_text, output=output)
    return PROMPT_NO_INPUT.format(instruction=instruction, output=output)


def load_and_format_dataset(config: DatasetConfig):
    """Loads `config.name`/`config.split` and slices to `config.num_samples`.

    Deferred import of `datasets` keeps this module importable without the
    package installed, and avoids any network call until this function is
    actually invoked.
    """
    from datasets import load_dataset

    dataset = load_dataset(config.name, split=config.split)
    if config.num_samples is not None and config.num_samples >= 0:
        dataset = dataset.select(range(min(config.num_samples, len(dataset))))
    return dataset
