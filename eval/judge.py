"""Pairwise LLM-as-judge for win-rate scoring.

Per CLAUDE.md, OpenAI is used here only for offline judging/labeling —
never to generate the assistant responses being judged (those come from
an eval.clients.ModelBackend, e.g. the local vLLM-served model).

JudgeBackend is the swappable interface all judges implement, mirroring
eval.clients.ModelBackend. It is position-relative: judge() only knows
about "response_a" / "response_b", not which one is the candidate or the
baseline. Position bias (judges systematically favoring "a" or "b") is
controlled by compare(), which randomizes which side the candidate lands
on per-comparison and un-shuffles the verdict back into candidate/baseline
terms so downstream metrics never see position at all.

The `openai` package is imported lazily so the rest of the harness (e.g.
the CPU-only HF transformers backend) works without it installed.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

try:
    from openai import OpenAI
except ImportError:  # optional dependency; not needed for the HF/CPU backend
    OpenAI = None

if TYPE_CHECKING:
    from eval.config import JudgeConfig

# Raw, position-relative verdict as returned by a JudgeBackend.
RawVerdict = Literal["a", "b", "tie"]

# Un-shuffled verdict in terms of what was actually being compared.
Verdict = Literal["candidate", "baseline", "tie"]

_JUDGE_PROMPT = """You are judging two AI assistant responses to the same user \
prompt for a voice assistant. Prefer responses that are correct, natural to \
hear spoken aloud, and concise. Respond with strict JSON: {{"winner": "a" | "b" | "tie"}}.

User prompt:
{prompt}

Response A:
{response_a}

Response B:
{response_b}
"""


@dataclass
class JudgeVerdict:
    winner: RawVerdict
    raw: str


class JudgeBackend(Protocol):
    def judge(self, prompt: str, response_a: str, response_b: str) -> JudgeVerdict: ...


class OpenAIJudge:
    def __init__(self, model: str = "gpt-4o-mini"):
        if OpenAI is None:
            raise ImportError("The 'openai' package is required for OpenAIJudge (pip install openai).")
        self._client = OpenAI()
        self.model = model

    def judge(self, prompt: str, response_a: str, response_b: str) -> JudgeVerdict:
        content = _JUDGE_PROMPT.format(prompt=prompt, response_a=response_a, response_b=response_b)
        completion = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = completion.choices[0].message.content
        winner = json.loads(raw).get("winner", "tie")
        if winner not in ("a", "b", "tie"):
            winner = "tie"
        return JudgeVerdict(winner=winner, raw=raw)


class MockConcisenessJudge:
    """Deterministic judge for CPU smoke-testing: prefers the shorter response.

    No API, no GPU. Ties when word counts match. Exists purely to exercise
    the win-rate flow (position randomization, aggregation, results JSON)
    without any real judge dependency.
    """

    def judge(self, prompt: str, response_a: str, response_b: str) -> JudgeVerdict:
        words_a = len(response_a.split())
        words_b = len(response_b.split())
        if words_a < words_b:
            winner: RawVerdict = "a"
        elif words_b < words_a:
            winner = "b"
        else:
            winner = "tie"
        return JudgeVerdict(winner=winner, raw=f"words_a={words_a} words_b={words_b}")


def make_judge(config: "JudgeConfig") -> JudgeBackend:
    if config.provider == "openai":
        return OpenAIJudge(model=config.model)
    if config.provider == "mock_conciseness":
        return MockConcisenessJudge()
    raise ValueError(f"Unknown judge provider: {config.provider!r}")


@dataclass
class ComparisonResult:
    prompt: str
    winner: Verdict
    swapped: bool


def compare(
    judge: JudgeBackend,
    prompt: str,
    candidate_text: str,
    baseline_text: str,
    *,
    rng: random.Random,
) -> ComparisonResult:
    """Judge candidate vs. baseline with randomized position to control judge bias.

    `rng` decides, per call, whether the candidate is shown as response A or
    response B. The raw a/b/tie verdict from `judge` is then un-shuffled back
    into candidate/baseline/tie terms so callers never need to think about
    position.
    """
    swapped = rng.random() < 0.5
    if swapped:
        response_a, response_b = baseline_text, candidate_text
        position_of = {"a": "baseline", "b": "candidate"}
    else:
        response_a, response_b = candidate_text, baseline_text
        position_of = {"a": "candidate", "b": "baseline"}

    raw_winner = judge.judge(prompt, response_a, response_b).winner
    winner: Verdict = "tie" if raw_winner == "tie" else position_of[raw_winner]

    return ComparisonResult(prompt=prompt, winner=winner, swapped=swapped)
