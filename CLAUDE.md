# CLAUDE.md

## Project
Real-time speech-to-speech voice assistant with a DPO-aligned language model at its core.

## Stack
Python 3.x, PyTorch, Hugging Face TRL + PEFT, faster-whisper, vLLM, FastAPI.

## Hard rules
- Generation MUST route through the local fine-tuned model (served via vLLM), never the OpenAI API. OpenAI is allowed only offline for preference-pair generation or labeling, or as an ASR/TTS provider.
- Streaming everywhere in the audio path. No blocking or synchronous calls between ASR, generation, and TTS. Overlap stages to keep time-to-first-audio under one second.
- Build the evaluation harness before training anything. Every model change is measured against baseline (win-rate, conciseness/length, latency).

## Conventions
- LoRA/QLoRA for all fine-tuning so it fits on a single GPU.
- Config-driven, seeded runs. Log every run to W&B or MLflow.
- Keep the base model small during iteration (1.5B to 3B) for fast turnaround.

## Decisions
- Base model: Qwen2.5-3B-Instruct (small, fits single-GPU LoRA/QLoRA iteration)
- GPU: single RTX 4090 (24GB VRAM)
- Experiment tracking: W&B

## Test / run
- Tests: `pytest`
- Serve: `vllm serve <checkpoint-path> --port 8000` then `uvicorn app.main:app --reload --port 8080` (FastAPI front door, proxies to vLLM)
