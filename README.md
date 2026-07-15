# AI Voice Assistant with DPO Fine-Tuning

Real-time speech-to-speech conversational assistant built around a language model aligned with Direct Preference Optimization (DPO) to improve response quality and conciseness.

## Stack
Python, PyTorch, Hugging Face (TRL, PEFT), Whisper (faster-whisper), vLLM, FastAPI

## Approach
The core of the project is the alignment pipeline: supervised fine-tuning (SFT) followed by DPO on a custom preference dataset, with generation served through the fine-tuned model. The voice layer (streaming ASR, aligned LLM, streaming TTS) wraps that model into a real-time demo.

## Roadmap
- Phase 1: Baseline serving (vLLM + FastAPI) and evaluation harness (win-rate, conciseness, latency)
- Phase 2: SFT with LoRA/QLoRA (TRL SFTTrainer)
- Phase 3: Custom preference dataset construction
- Phase 4: DPO training and iteration (TRL DPOTrainer)
- Phase 5: Serve the aligned model, benchmark vs baseline
- Phase 6: Streaming voice pipeline (faster-whisper + VAD, streaming TTS)
- Phase 7: Hardening, deployment, experimental writeup

## Status
Scaffolding. Setup and build in progress.
