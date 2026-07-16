from unittest.mock import patch

import pytest

from eval.clients import make_backend
from eval.config import EndpointConfig


def test_make_backend_selects_vllm():
    config = EndpointConfig(name="baseline", model="m", backend="vllm", base_url="http://localhost:8000/v1")
    with patch("eval.clients.VLLMClient") as mock_cls:
        make_backend(config)
    mock_cls.assert_called_once_with(base_url="http://localhost:8000/v1", model="m", api_key="EMPTY")


def test_make_backend_selects_hf_transformers():
    config = EndpointConfig(name="baseline", model="sshleifer/tiny-gpt2", backend="hf_transformers", device="cpu")
    with patch("eval.clients.HFTransformersBackend") as mock_cls:
        make_backend(config)
    mock_cls.assert_called_once_with(model="sshleifer/tiny-gpt2", device="cpu")


def test_make_backend_rejects_unknown():
    config = EndpointConfig(name="baseline", model="m", backend="not-a-backend")
    with pytest.raises(ValueError):
        make_backend(config)
