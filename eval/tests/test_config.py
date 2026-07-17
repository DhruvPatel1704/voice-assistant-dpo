from eval.config import load_config


def test_load_default_config():
    config = load_config("eval/configs/eval.yaml")
    assert config.seed == 42
    assert config.baseline.backend == "vllm"
    assert config.baseline.base_url.startswith("http://localhost")
    assert config.candidate.base_url.startswith("http://localhost")
    assert config.judge.enabled is True
    assert config.wandb.mode == "disabled"


def test_load_cpu_smoke_config():
    config = load_config("eval/configs/eval.cpu_smoke.yaml")
    assert config.baseline.backend == "hf_transformers"
    assert config.baseline.model == "sshleifer/tiny-gpt2"
    assert config.baseline.device == "cpu"
    assert config.judge.provider == "mock_conciseness"
    assert config.judge.enabled is True
