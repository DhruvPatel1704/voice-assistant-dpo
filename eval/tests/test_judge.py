import json
from unittest.mock import MagicMock, patch

from eval.judge import OpenAIJudge


def _fake_completion(winner: str):
    message = MagicMock()
    message.content = json.dumps({"winner": winner})
    choice = MagicMock(message=message)
    return MagicMock(choices=[choice])


def test_judge_parses_winner():
    with patch("eval.judge.OpenAI") as mock_openai_cls:
        mock_client = mock_openai_cls.return_value
        mock_client.chat.completions.create.return_value = _fake_completion("a")
        verdict = OpenAIJudge(model="gpt-4o-mini").judge("prompt", "resp a", "resp b")
    assert verdict.winner == "a"


def test_judge_defaults_to_tie_on_invalid_winner():
    with patch("eval.judge.OpenAI") as mock_openai_cls:
        mock_client = mock_openai_cls.return_value
        mock_client.chat.completions.create.return_value = _fake_completion("invalid")
        verdict = OpenAIJudge(model="gpt-4o-mini").judge("prompt", "resp a", "resp b")
    assert verdict.winner == "tie"
