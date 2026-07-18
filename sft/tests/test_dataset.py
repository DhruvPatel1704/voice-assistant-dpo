"""Unit-tests format_example() against fake rows -- no network, no tokenizer."""

from __future__ import annotations

from sft.config import DatasetConfig
from sft.dataset import format_example


def test_format_example_with_input():
    config = DatasetConfig()
    example = {
        "instruction": "Summarize the text.",
        "input": "The quick brown fox jumps over the lazy dog.",
        "output": "A fox jumps over a dog.",
    }

    text = format_example(example, config)

    assert "### Instruction:\nSummarize the text." in text
    assert "### Input:\nThe quick brown fox jumps over the lazy dog." in text
    assert "### Response:\nA fox jumps over a dog." in text


def test_format_example_without_input():
    config = DatasetConfig()
    example = {"instruction": "Name a primary color.", "input": "", "output": "Red."}

    text = format_example(example, config)

    assert "### Input:" not in text
    assert "### Instruction:\nName a primary color." in text
    assert "### Response:\nRed." in text


def test_format_example_respects_custom_field_names():
    config = DatasetConfig(instruction_field="prompt", input_field="context", output_field="completion")
    example = {"prompt": "Say hi.", "context": "", "completion": "Hi!"}

    text = format_example(example, config)

    assert "Say hi." in text
    assert "Hi!" in text
