# Evaluation harness

Compares a candidate model against a baseline on the same prompt set: latency,
conciseness, and LLM-as-judge win-rate. Run before and after every model
change (see CLAUDE.md: "Build the evaluation harness before training
anything.").

```
python -m eval.run_eval --config eval/configs/eval.yaml
```

## Win-rate

For each prompt, both models generate a response, then a judge is asked which
response is better (or a tie). The win-rate is `candidate wins / total
comparisons`.

### Judge backends

`JudgeBackend` is the swappable interface (same pattern as `ModelBackend` in
`eval/clients.py`): `judge(prompt, response_a, response_b) -> JudgeVerdict`,
where the verdict is position-relative (`"a" | "b" | "tie"`) — the judge only
ever sees "response A" / "response B", never "candidate" / "baseline".

Two backends, selected via `judge.provider` in the config:

- `openai` (`OpenAIJudge`) — the real judge. Per CLAUDE.md's hard rules, this
  is the *only* place OpenAI is allowed to appear in this harness — it's used
  offline to judge/label responses, never to generate the responses being
  evaluated. Requires an OpenAI API key.
- `mock_conciseness` (`MockConcisenessJudge`) — deterministic stand-in that
  prefers whichever response has fewer words (ties on equal word count). No
  API, no GPU. Exists so the entire win-rate flow (position randomization,
  aggregation, results JSON) can be smoke-tested on CPU. A future real judge
  backend just needs to implement `JudgeBackend` and get registered in
  `make_judge()`.

### Controlling position bias

LLM judges are known to be biased toward whichever response is shown first.
`eval.judge.compare()` controls for this: for each comparison it flips a coin
(from a `random.Random(seed)` seeded off the run's config seed, so it's
reproducible) to decide whether the candidate is shown as response A or
response B, calls the judge, then un-shuffles the raw `a/b/tie` verdict back
into `candidate/baseline/tie` terms. Downstream code (metrics, reports) only
ever sees the un-shuffled verdict — it never has to reason about position.

`run_eval.py` records, per prompt, whether that comparison was swapped, so the
mapping is auditable after the fact.

## Running it

**CPU smoke test** (no GPU, no vLLM server, no OpenAI key):

```
python -m eval.run_eval --config eval/configs/eval.cpu_smoke.yaml
```

Uses `hf_transformers` backends with a tiny model for generation and the
`mock_conciseness` judge — proves the whole pipeline (config -> generation ->
judge -> metrics -> report) end to end.

**Real run** (vLLM-served candidate + baseline, OpenAI judge):

```
python -m eval.run_eval --config eval/configs/eval.yaml
```

Requires both `baseline.base_url` and `candidate.base_url` pointing at local
vLLM servers, and an OpenAI API key available for `OpenAIJudge`.

## Output

Each run writes `eval/results/eval_<timestamp>.json` with:

- `win_rate`: aggregate `{wins, losses, ties, total, win_rate}` (candidate
  wins / losses / ties against the baseline).
- `judge_verdicts`: per-prompt list of `{prompt, winner, swapped}`, where
  `winner` is `"candidate" | "baseline" | "tie"` and `swapped` records whether
  the candidate was shown as response A or B for that comparison.
- `candidate_latency` / `baseline_latency`: mean time-to-first-token and mean
  total latency.
- `conciseness`: mean word counts and candidate/baseline ratio.

If `judge.enabled` is `false` in the config, `win_rate` and `judge_verdicts`
are both `null`.
