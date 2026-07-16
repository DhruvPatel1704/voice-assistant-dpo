"""Entrypoint: run candidate vs baseline evaluation (win-rate, conciseness, latency).

Usage:
    python -m eval.run_eval --config eval/configs/eval.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from eval.clients import make_backend
from eval.config import EvalConfig, load_config, set_seed
from eval.dataset import load_prompts
from eval.judge import OpenAIJudge
from eval.metrics import conciseness_stats, latency_stats, win_rate_stats


def run(config: EvalConfig) -> dict:
    set_seed(config.seed)
    prompts = load_prompts(config.prompts_path)

    baseline_backend = make_backend(config.baseline)
    candidate_backend = make_backend(config.candidate)

    gen_kwargs = {
        "max_tokens": config.generation.max_tokens,
        "temperature": config.generation.temperature,
        "seed": config.seed,
    }
    baseline_results = [baseline_backend.generate(p, **gen_kwargs) for p in prompts]
    candidate_results = [candidate_backend.generate(p, **gen_kwargs) for p in prompts]

    verdicts = []
    if config.judge.enabled:
        judge = OpenAIJudge(model=config.judge.model)
        for prompt, candidate, baseline in zip(prompts, candidate_results, baseline_results):
            verdicts.append(judge.judge(prompt, candidate.text, baseline.text).winner)

    report = {
        "num_prompts": len(prompts),
        "candidate_latency": asdict(latency_stats(candidate_results)),
        "baseline_latency": asdict(latency_stats(baseline_results)),
        "conciseness": asdict(
            conciseness_stats([r.text for r in candidate_results], [r.text for r in baseline_results])
        ),
        "win_rate": asdict(win_rate_stats(verdicts)) if verdicts else None,
    }

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"eval_{int(time.time())}.json").write_text(json.dumps(report, indent=2))

    if config.wandb.mode != "disabled":
        import wandb

        wandb_run = wandb.init(project=config.wandb.project, config=asdict(config.generation), mode=config.wandb.mode)
        wandb.log(report)
        wandb_run.finish()

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline vs candidate evaluation")
    parser.add_argument("--config", default="eval/configs/eval.yaml")
    args = parser.parse_args()
    report = run(load_config(args.config))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
