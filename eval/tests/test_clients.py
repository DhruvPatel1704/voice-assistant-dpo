from unittest.mock import MagicMock, patch

import pytest

from eval.clients import VLLMClient


def _make_chunk(text):
    choice = MagicMock()
    choice.delta.content = text
    chunk = MagicMock()
    chunk.choices = [choice]
    return chunk


def test_generate_concatenates_stream_from_local_endpoint():
    with patch("eval.clients.OpenAI") as mock_openai_cls:
        mock_client = mock_openai_cls.return_value
        mock_client.chat.completions.create.return_value = iter(
            [_make_chunk("Hello"), _make_chunk(" world")]
        )
        client = VLLMClient(base_url="http://localhost:8000/v1", model="test-model")
        mock_openai_cls.assert_called_once_with(base_url="http://localhost:8000/v1", api_key="EMPTY")
        result = client.generate("hi", max_tokens=16, temperature=0.0, seed=42)

    assert result.text == "Hello world"
    assert result.time_to_first_token > 0
    assert result.total_latency >= result.time_to_first_token


def test_rejects_openai_api_base_url():
    with pytest.raises(ValueError):
        VLLMClient(base_url="https://api.openai.com/v1", model="gpt-4o")
