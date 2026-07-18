# SFT (supervised fine-tuning)

This is Phase 2: before the DPO alignment step, the base model (Qwen2.5-3B-Instruct)
gets a LoRA adapter trained on instruction/response pairs so it reliably follows
the assistant format we'll later apply preference optimization on top of. Per
CLAUDE.md: the eval harness (`eval/`) already exists, so every adapter produced
here gets compared against the un-tuned base model before anything else happens
to it.

This directory is **scaffolding only** as of now -- no training has been run,
no model or full dataset has been downloaded. It's structured so that once a
cloud GPU is available, training is a single command.

## Running it (once a GPU is available)

```
python -m sft.train --config sft/configs/sft.yaml
```

This will, on first run, download `Qwen/Qwen2.5-3B-Instruct` from the Hugging
Face Hub and the `tatsu-lab/alpaca` dataset (sliced to `dataset.num_samples`
rows per the config). Start with the default `num_samples: 2000` to confirm
the loop works end-to-end before scaling up toward the full ~52k-row split.

Everything is config-driven (`sft/configs/sft.yaml`): base model, dataset
name/slice size, LoRA rank/alpha/dropout/target_modules, learning rate,
epochs, batch size, gradient accumulation, max sequence length, output
directory, and seed. Nothing is hardcoded in `sft/train.py`.

## VRAM ballpark

Qwen2.5-3B-Instruct in bf16 with LoRA (rank 16, default config) on a single
RTX 4090 (24GB):

- Base model weights (bf16): ~6GB
- LoRA adapter + optimizer states: a few hundred MB (LoRA only optimizes the
  adapter, not the base weights)
- Activations/KV cache at `max_seq_length: 1024`, batch size 4: a few more GB,
  scales with batch size and sequence length

Expect roughly **12-16GB** total for the default config -- comfortable
headroom on 24GB. If it doesn't fit (e.g. after raising batch size or
sequence length), set `use_qlora: true` in the config to load the base model
in 4-bit via `bitsandbytes`, which cuts base weight memory roughly in half at
some throughput cost. `use_qlora` is GPU-only -- `bitsandbytes` isn't imported
unless it's set to `true`.

## Dataset

Default is `tatsu-lab/alpaca` (Alpaca-style `instruction`/`input`/`output`
fields), formatted via `sft/dataset.py:format_example` into the standard
Alpaca prompt template before being handed to TRL's `SFTTrainer` as a
`formatting_func`. Swap `dataset.name` for any other dataset with equivalent
fields (or point `instruction_field`/`input_field`/`output_field` at different
column names) without touching `sft/train.py`.

## Evaluating the resulting adapter

Once training finishes, `config.output_dir` holds the LoRA adapter + tokenizer
files. To compare it against the un-tuned baseline with the existing eval
harness:

1. Serve the baseline (un-tuned) model with vLLM, e.g. on port 8000:
   ```
   vllm serve Qwen/Qwen2.5-3B-Instruct --port 8000
   ```
2. Serve the SFT adapter with vLLM's LoRA support (`--enable-lora`), pointing
   at the base model plus the adapter directory, e.g. on port 8001:
   ```
   vllm serve Qwen/Qwen2.5-3B-Instruct --port 8001 --enable-lora \
     --lora-modules sft-adapter=sft/adapters/qwen2.5-3b-alpaca-lora
   ```
3. Point `eval/configs/eval.yaml`'s `candidate.base_url`/`candidate.model` at
   the adapter's endpoint/name and `baseline` at the un-tuned server, then run:
   ```
   python -m eval.run_eval --config eval/configs/eval.yaml
   ```

This produces the same win-rate / conciseness / latency report as any other
eval run, so the SFT adapter is measured against baseline exactly like every
subsequent model change (see CLAUDE.md: "Every model change is measured
against baseline").

## Verifying the scaffolding without a GPU

No training, model download, or full dataset download happens in CI or on a
laptop. Instead:

```
pytest sft/
```

covers:
- `sft/tests/test_config.py` -- both `sft/configs/sft.yaml` and
  `sft/configs/sft.cpu_smoke.yaml` parse into `SFTConfig` correctly, and
  defaults fill in when a config omits a section.
- `sft/tests/test_dataset.py` -- `format_example` builds the correct Alpaca
  prompt text from fake rows (with and without an `input` field, and with
  custom field names), with no network call.
- `sft/tests/test_train_imports.py` -- `sft.train` imports cleanly on CPU
  (confirming `bitsandbytes` is never imported unless `use_qlora: true`),
  `build_lora_config` produces a real `peft.LoraConfig` from the settings, and
  `build_trainer` actually constructs a TRL `SFTConfig`/`SFTTrainer` against
  `sft/configs/sft.cpu_smoke.yaml`'s tiny model (using in-memory fake rows
  instead of downloading Alpaca) -- this is what catches TRL API drift, e.g. a
  `SFTConfig` kwarg rename, before a real GPU run does.

`sft/configs/sft.cpu_smoke.yaml` (tiny model, 8-row slice, no QLoRA) exists
for when you *do* want to smoke-test the full training loop cheaply on a
machine with a small GPU or a lot of patience on CPU. Its model
(`sshleifer/tiny-gpt2`) also backs the `build_trainer` test above via
`HF_HUB_OFFLINE`, so that test only works once it's been downloaded/cached at
least once; the full Alpaca dataset is never downloaded by the test suite.
