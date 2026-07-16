"""Metrics for the evaluation harness: latency, conciseness, win-rate."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from statistics import mean

from eval.clients import GenerationResult
from eval.judge import Verdict


def word_count(text: str) -> int:
    return len(text.split())


@dataclass
class LatencyStats:
    mean_ttft: float
    mean_total: float


def latency_stats(results: Sequence[GenerationResult]) -> LatencyStats:
    return LatencyStats(
        mean_ttft=mean(r.time_to_first_token for r in results),
        mean_total=mean(r.total_latency for r in results),
    )


@dataclass
class ConcisenessStats:
    mean_words: float
    mean_words_baseline: float

    @property
    def ratio(self) -> float:
        if self.mean_words_baseline == 0:
            return float("inf")
        return self.mean_words / self.mean_words_baseline


def conciseness_stats(candidate_texts: Iterable[str], baseline_texts: Iterable[str]) -> ConcisenessStats:
    return ConcisenessStats(
        mean_words=mean(word_count(t) for t in candidate_texts),
        mean_words_baseline=mean(word_count(t) for t in baseline_texts),
    )


@dataclass
class WinRateStats:
    wins: int
    losses: int
    ties: int

    @property
    def total(self) -> int:
        return self.wins + self.losses + self.ties

    @property
    def win_rate(self) -> float:
        return self.wins / self.total if self.total else 0.0


def win_rate_stats(verdicts: Sequence[Verdict]) -> WinRateStats:
    return WinRateStats(
        wins=sum(1 for v in verdicts if v == "a"),
        losses=sum(1 for v in verdicts if v == "b"),
        ties=sum(1 for v in verdicts if v == "tie"),
    )
