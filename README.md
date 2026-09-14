# Local MoE router — no LM Studio

This replaces your Gradio-over-LM-Studio setup with a stack that talks to
`llama.cpp` directly, controlled entirely from Python. It's built for a 30B
MoE model on an RTX 5070 (12GB VRAM) + 32GB system RAM.

## What's real vs what was in your diagrams

Your diagrams describe: per-token PCIe streaming of individual experts,
double-buffered ping-pong VRAM cache, CUDA-graph-traced execution, and a
self-updating router. Some of that is genuine research territory (see
"Fiddler" and "MoE-Infinity" papers on expert offloading) but requires
patching the inference engine's C++ internals — not something reachable
from a Python wrapper, and not something to hand-roll safely.

What actually ships today and gets you most of the real benefit:

- **Static GPU/CPU tensor split** (`-ot` flag in llama.cpp): attention and
  shared layers stay resident on GPU, MoE expert (`ffn_*_exps`) tensors stay
  in system RAM. llama.cpp's own top-k gating picks which expert to use per
  token from whatever is already resident — there's no dynamic re-streaming
  step for you to build.
- **Speculative decoding**: a small draft model proposes tokens, the big
  model verifies them in one batched pass. Implemented by hand in
  `local_engine.py` so every accept/reject is visible.
- **Online draft-model adaptation**: every rejected draft token becomes a
  training example for a LoRA adapter on the draft model, trained in the
  background (`draft_trainer.py`). Real cost, not zero, but small and
  off the critical path — see the docstring in that file for specifics.

## Setup

1. Build `llama.cpp` yourself (or grab a release binary) so `llama-server`
   and `llama-quantize` are on your `PATH`. This is the actual inference
   engine; nothing here reimplements it.
2. `pip install -r requirements.txt`
3. Edit `config.py`:
   - `big_model_gguf`: your 30B model, quantized (Q4_K_M is a reasonable
     starting point for a 30B on this hardware).
   - `draft_model_gguf`: a small GGUF model for fast speculative proposals
     (e.g. Qwen2.5-0.5B-Instruct).
   - `draft_model_hf`: the *same* draft model in full-precision
     `transformers` format — this is the trainable copy, separate from the
     quantized inference copy. You can't backprop through a quantized GGUF
     file directly.
   - `moe_cpu_tensor_regex`: check your model's actual tensor names (run
     `llama-server` once and read its startup log, or use `gguf-dump`) and
     adjust the regex if it doesn't match `ffn_(gate|down|up)_exps`.
4. `python app.py`

## Self-improvement: automated, not manual

The online trainer (`draft_trainer.py`) fine-tunes a LoRA adapter on the
draft model from every rejected speculative-decoding token. Every
`Runtime.refresh_every_n_steps` training steps, it automatically merges the
adapter into the base weights, reconverts to GGUF (`convert_hf_to_gguf.py`,
vendored in `tools/`), requantizes (`llama-quantize`), and hot-swaps the
running draft model to the refreshed copy — no manual steps, no restart.
This whole cycle is minutes of work per refresh, not per-token; that's the
actual shape of "self-learning" here.

**Caveat, found by benchmarking (see below):** the draft model correcting
itself doesn't translate into faster generation on this hardware/model
combo — speculative decoding itself doesn't win here. The refresh loop now
feeds a different, genuinely useful path instead: routing "quick" queries
to the draft model directly, with the big model verifying asynchronously in
the background and any mismatch becoming a training example. See
`router.py` / `app.py`'s fast-path wiring and `EXECUTIVE_SUMMARY.md`.

## Status / what's been measured

- [`BENCHMARK_RESULTS.md`](BENCHMARK_RESULTS.md): rigorous 3-way benchmark
  (baseline vs hand-rolled speculative vs native C++ speculative) on real
  prompts. Bottom line: speculative decoding does **not** win wall-clock on
  this hardware/model/llama-server-build combo — hand-rolled speculative
  was 0.25x baseline, native speculative ~1.02x (a wash).
- [`EXECUTIVE_SUMMARY.md`](EXECUTIVE_SUMMARY.md): running project log —
  bugs found and fixed, the pivot away from speculative decoding toward the
  fast-path/async-verify architecture, and why.
- [`experiments/`](experiments/): a separate research track asking whether
  MoE expert selection is predictable enough, token to token, to justify a
  weight-prefetch pipeline (CPU RAM → GPU staging buffer) underneath
  llama.cpp itself:
  - [`EXPERT_TRACE_FINDINGS.md`](experiments/EXPERT_TRACE_FINDINGS.md) —
    yes, predictable (44.2% avg overlap, ~7x random baseline, at scale
    across 8 prompts/5 topics), but three different trained predictors
    (Markov, MLP, residual MLP) all underperformed the trivial "same as
    last token" heuristic.
  - [`PREFETCH_FEASIBILITY.md`](experiments/PREFETCH_FEASIBILITY.md) — real
    PCIe bandwidth measured on this machine (53.7 GB/s) comfortably covers
    the transfer volume a prefetch cache would need; the compute-side
    upside is still unmeasured (two proxy methods tried, both diagnosed and
    discarded for good reasons, documented in the file).

## Known version-sensitivity

- `local_engine.py`'s `logits_for_tokens` assumes llama-server's
  `/completion` response has a `completion_probabilities[i]["probs"][0]["tok_str"]`
  field. This schema has shifted across llama.cpp releases — `curl` your
  build's `/completion` endpoint with `"n_probs": 1` and adjust field names
  if generation errors out.
- `-ot` flag syntax and available regex targets are model/llama.cpp-version
  specific. Check `llama-server --help` on your build.
