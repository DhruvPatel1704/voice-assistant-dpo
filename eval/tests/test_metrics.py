from eval.clients import GenerationResult
from eval.metrics import conciseness_stats, latency_stats, win_rate_stats, word_count


def test_word_count():
    assert word_count("hello world") == 2
    assert word_count("") == 0


def test_latency_stats():
    results = [GenerationResult("a", 0.1, 0.5), GenerationResult("b", 0.3, 0.7)]
    stats = latency_stats(results)
    assert stats.mean_ttft == 0.2
    assert stats.mean_total == 0.6


def test_conciseness_stats():
    stats = conciseness_stats(["one two"], ["one two three four"])
    assert stats.mean_words == 2
    assert stats.mean_words_baseline == 4
    assert stats.ratio == 0.5


def test_conciseness_ratio_guards_zero_baseline():
    stats = conciseness_stats(["one"], [""])
    assert stats.ratio == float("inf")


def test_win_rate_stats():
    stats = win_rate_stats(["candidate", "candidate", "baseline", "tie"])
    assert stats.wins == 2
    assert stats.losses == 1
    assert stats.ties == 1
    assert stats.win_rate == 0.5


def test_win_rate_stats_empty():
    stats = win_rate_stats([])
    assert stats.win_rate == 0.0
