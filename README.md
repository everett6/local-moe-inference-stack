# Local MoE router — no LM Studio

A fully local chat stack for a 30B mixture-of-experts model on consumer
hardware, driving `llama.cpp`'s `llama-server` directly from Python. No LM
Studio app, no cloud.

**Built for:** RTX 5070 (12 GB VRAM), Ryzen 9 7950X, 32 GB RAM.
**Model:** Qwen3-30B-A3B-Instruct-2507, Q4_K_M.

**Speed:** ~71 tok/s in the app as it starts, up from 47 with the original
config. About 82 tok/s is the ceiling with nothing else on the card.

> **Start here:** [`PLAN.md`](PLAN.md): current state, what's settled, and the
> prioritized next steps.

## How it works

### Experts split between GPU and RAM, fitted at launch

A 30B MoE at Q4_K_M is ~17 GB and doesn't fit in 12 GB. llama.cpp's
`--n-cpu-moe N` keeps the expert weights of the first N of 48 layers in system
RAM and puts everything else on the GPU. Every layer of experts moved onto the
GPU is worth a few tok/s, so the goal is the smallest N that fits.

How small depends on what else is using VRAM when the server starts: the app's
own draft model, a browser, another model server. So `BigModelServer` doesn't
hardcode N. It starts at `Runtime.n_cpu_moe` and, if the server fails to load,
leaves less than `vram_headroom_mb` free, or crashes on a ~700-token warm-up
prompt, it restarts with `n_cpu_moe_step` more layers in RAM. That warm-up
matters: CUDA allocates part of its memory only on the first real prompt, so a
server can load fine and still crash on your first message.

(The original config pinned **every** expert to RAM with `-ot`, leaving 10 GB of
the card empty: 47 tok/s. See `SPEC_DECODING.md` §5.)

### Request routing

`router.prompt_bucket` sorts each prompt:

- **`quick`**: a small draft model (Qwen2.5-Coder-0.5B) answers on its own,
  shown immediately. The 30B then checks the answer in the background, corrects
  it if they disagree, and turns each disagreement into a training example.
- **everything else**: the 30B alone, using its chat template and the full
  conversation, **streamed** as it generates.

### Online self-improvement

`draft_trainer.py` LoRA-tunes the draft model from those disagreements, in the
background. Every `refresh_every_n_steps` steps it merges the adapter,
reconverts to GGUF, requantizes, and hot-swaps the running draft with no
restart.

It runs **on the CPU with 4 threads** by default (`trainer_device`,
`trainer_cpu_threads`). On the GPU it takes VRAM from the 30B's experts on every
reply (~62 vs ~71 tok/s). With all CPU cores, it slows replies to 27-43 tok/s
while a training step runs.

## Setup

1. **`llama-server`:** `config.py` points at the CUDA 12 build bundled with LM
   Studio (`Paths.llama_server_bin`) and its vendored CUDA libraries
   (`llama_server_ld_library_path`). Point them at your own llama.cpp build if
   you have one. `llama-quantize` is needed only for the self-improvement
   refresh.
2. `pip install -r requirements.txt`
3. In `config.py`, set:
   - `big_model_gguf`: the 30B model.
   - `draft_model_gguf`: the small GGUF draft used by the quick path.
   - `draft_model_hf`: the same draft in `transformers` format. This is the
     trainable copy, because you can't backprop through a quantized GGUF.
4. `python app.py`, then open http://127.0.0.1:7860

### Settings that matter (`config.Runtime`)

| setting | default | what it does |
|---|---|---|
| `n_cpu_moe` | 20 | fastest split to *try*; launch backs off from here |
| `n_cpu_moe_step` | 1 | layers moved to RAM per failed attempt |
| `vram_headroom_mb` | 768 | free VRAM to keep after load (draft hot-swap, a browser) |
| `server_slots` | 1 | llama-server parallel slots; 1 is +2.8 tok/s for a single user |
| `trainer_device` | `cpu` | where the trainer's model lives |
| `trainer_cpu_threads` | 4 | caps training's CPU use so replies don't stall |
| `threads` | 16 | llama-server CPU threads |
| `n_ctx` | 4096 | context length; raising it costs VRAM, so experts |

## What's been measured

- [`PLAN.md`](PLAN.md): current state and next steps.
- [`SPEC_DECODING.md`](SPEC_DECODING.md): the MoE split (1.74x); why
  speculative decoding and draft-model training don't pay off on this hardware
  (EAGLE3, Qwen3-0.6B, ngram, code-only drafting, all measured); trainer
  placement; runtime settings.
- [`BENCHMARK_RESULTS.md`](BENCHMARK_RESULTS.md): the original 3-way benchmark.
  **Outdated:** it predates the MoE fix, and its "native speculative = 1.02x"
  was plain generation (corrected in the file).
- [`EAGLE3_DRAFT_MODEL.md`](EAGLE3_DRAFT_MODEL.md): the EAGLE3 attempt.
- [`EXECUTIVE_SUMMARY.md`](EXECUTIVE_SUMMARY.md): earlier sessions' log, the
  bugs fixed and the pivot to the quick-path design.
- [`experiments/`](experiments/): every measurement script and its raw JSON
  result, plus the expert-prediction / prefetch research track
  ([`EXPERT_TRACE_FINDINGS.md`](experiments/EXPERT_TRACE_FINDINGS.md),
  [`PREFETCH_FEASIBILITY.md`](experiments/PREFETCH_FEASIBILITY.md)).

## Gotchas

- **Measure speed with nothing else on the GPU.** Another model server (Ollama,
  LM Studio) changes how many experts fit and skews every number.
- **llama-server keeps running when a draft model is rejected.** It logs
  `vocabs are not compatible` and quietly serves without speculation.
  Experiment scripts here treat that as an error.
- **`/completion` applies no chat template.** Send chat to
  `/v1/chat/completions`, or the Instruct model continues your text instead of
  answering it.
- `local_engine.py` reads `completion_probabilities[i]["id"]` from
  `/completion`. That schema has shifted across llama.cpp releases, so check it
  with `"n_probs": 1` if the quick-path verification breaks.
