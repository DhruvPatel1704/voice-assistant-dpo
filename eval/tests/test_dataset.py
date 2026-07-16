import json

from eval.dataset import load_prompts


def test_load_prompts(tmp_path):
    p = tmp_path / "prompts.jsonl"
    p.write_text(
        "\n".join(
            [
                json.dumps({"prompt": "hi"}),
                "",
                json.dumps({"prompt": "bye"}),
            ]
        )
    )
    assert load_prompts(p) == ["hi", "bye"]


def test_load_prompts_sample_file():
    prompts = load_prompts("data/eval_prompts.jsonl")
    assert len(prompts) == 10
    assert all(isinstance(p, str) and p for p in prompts)
