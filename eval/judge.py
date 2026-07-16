"""Pairwise LLM-as-judge for win-rate scoring.

Per CLAUDE.md, OpenAI is used here only for offline judging/labeling —
never to generate the assistant responses being judged (those come from
an eval.clients.ModelBackend, e.g. the local vLLM-served model).

The `openai` package is imported lazily so the rest of the harness (e.g.
the CPU-only HF transformers backend) works without it installed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

try:
    from openai import OpenAI
except ImportError:  # optional dependency; not needed for the HF/CPU backend
    OpenAI = None

Verdict = Literal["a", "b", "tie"]

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
    winner: Verdict
    raw: str


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
