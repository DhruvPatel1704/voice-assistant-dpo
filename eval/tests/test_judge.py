import json
import random
from unittest.mock import MagicMock, patch

import pytest

from eval.config import JudgeConfig
from eval.judge import JudgeVerdict, MockConcisenessJudge, OpenAIJudge, compare, make_judge


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


# --- MockConcisenessJudge ---------------------------------------------------


def test_mock_judge_prefers_shorter_response():
    judge = MockConcisenessJudge()
    assert judge.judge("p", "one two three", "one").winner == "b"
    assert judge.judge("p", "one", "one two three").winner == "a"


def test_mock_judge_ties_on_equal_length():
    judge = MockConcisenessJudge()
    assert judge.judge("p", "one two", "three four").winner == "tie"


# --- make_judge factory ------------------------------------------------------


def test_make_judge_selects_openai():
    config = JudgeConfig(provider="openai", model="gpt-4o-mini")
    with patch("eval.judge.OpenAI"):
        judge = make_judge(config)
    assert isinstance(judge, OpenAIJudge)


def test_make_judge_selects_mock_conciseness():
    config = JudgeConfig(provider="mock_conciseness")
    judge = make_judge(config)
    assert isinstance(judge, MockConcisenessJudge)


def test_make_judge_rejects_unknown():
    config = JudgeConfig(provider="not-a-provider")
    with pytest.raises(ValueError):
        make_judge(config)


# --- compare(): order randomization and unshuffling --------------------------


class _StubJudge:
    """Always declares whichever response is passed as `response_a` the winner."""

    def judge(self, prompt: str, response_a: str, response_b: str) -> JudgeVerdict:
        return JudgeVerdict(winner="a", raw="stub")


def test_compare_unshuffles_winner_regardless_of_position():
    judge = _StubJudge()
    # Force no swap: candidate is response_a, so the stub picks candidate.
    result = compare(judge, "p", "candidate text", "baseline text", rng=random.Random(0))
    # rng.random() with seed 0 is deterministic; assert against actual swap value
    # by checking internal consistency instead of a hardcoded bool.
    if result.swapped:
        assert result.winner == "baseline"
    else:
        assert result.winner == "candidate"


def test_compare_covers_both_swap_outcomes_and_unshuffles_correctly():
    judge = _StubJudge()
    seen_swapped = set()
    for seed in range(50):
        result = compare(judge, "p", "candidate text", "baseline text", rng=random.Random(seed))
        seen_swapped.add(result.swapped)
        expected = "baseline" if result.swapped else "candidate"
        assert result.winner == expected
    # Over 50 seeds we should see both swap outcomes at least once.
    assert seen_swapped == {True, False}


def test_compare_preserves_tie():
    class _TieJudge:
        def judge(self, prompt: str, response_a: str, response_b: str) -> JudgeVerdict:
            return JudgeVerdict(winner="tie", raw="stub")

    result = compare(_TieJudge(), "p", "candidate text", "baseline text", rng=random.Random(1))
    assert result.winner == "tie"


def test_compare_is_deterministic_given_same_seed():
    judge = _StubJudge()
    r1 = compare(judge, "p", "candidate text", "baseline text", rng=random.Random(7))
    r2 = compare(judge, "p", "candidate text", "baseline text", rng=random.Random(7))
    assert r1.swapped == r2.swapped
    assert r1.winner == r2.winner
