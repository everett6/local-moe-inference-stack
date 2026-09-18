# Local MoE router — no LM Studio

A fully local chat stack for a 30B mixture-of-experts model on consumer
hardware, driving `llama.cpp`'s `llama-server` directly from Python. No LM
Studio app, no cloud.

**Built for:** RTX 5070 (12 GB VRAM), Ryzen 9 7950X, 32 GB RAM.
**Model:** Qwen3-30B-A3B-Instruct-2507, **Q2_K** by default (choose with
`AI2_BIG_MODEL`, see below).

**Speed:** **~185-197 tok/s** on short prompts in the shipped configuration, up
from 47 with the original config and 82 with the same model at Q4_K_M -- **4.1x**
and 2.3x. On the original benchmark's five prompts, re-run unchanged:
46.19 -> 188.5 tok/s ([`BENCHMARK_RESULTS.md`](BENCHMARK_RESULTS.md)). A
~4,000-token document answers at ~145. Accuracy cost, measured: GSM8K 96.0% vs
95.6%, HumanEval 89.6% vs 91.5% -- neither difference is significant (p=1.00 and
p=0.51, McNemar). `AI2_BIG_MODEL=ud-q3_k_xl` is the quality option -- 115 tok/s
with no measurable loss against Q4_K_M on either benchmark.

(An earlier figure of 191-212 was measured at `n_ctx` 4096, where the model fits
at `--n-cpu-moe 0`. The 8192 context shipped since costs two expert layers and
about 6% of decode -- see `Runtime.n_ctx` for why that trade is worth taking.)

Since the remote-desktop daemons were removed there is enough VRAM for all 48
layers' experts (`--n-cpu-moe 0`) -- at `n_ctx` 4096. The 8192-token context
shipped instead takes two of those layers back, which is what pays for a
~5,000-token document being answerable at all: at 4096 the server refused it
outright rather than truncating. A conversation that outgrows the window drops
its oldest exchanges and says so, rather than refusing every later message.

> **Before any GPU work:** this card crashes at its stock 250 W limit (four
> `Xid 79` bus drops, none at 175 W) and **the cap resets on every reboot**. Run
> `sudo nvidia-smi -pl 175`, or install it once with
> [`tools/ai2-gpu-power-cap.service`](tools/ai2-gpu-power-cap.service) (see
> PLAN.md, Phase A4). The dashboard warns whenever the limit is above the cap.

> **Start here:** [`PLAN.md`](PLAN.md): current state, what's settled, and the
> prioritized next steps.

## Choosing the model file

Same model, different quantization. Measured on this box through the app's own
launch path; details and sources in `config.BIG_MODELS` and `PLAN.md`.

| `AI2_BIG_MODEL` | file | decode tok/s | vs Q4_K_M: KL divergence / HumanEval / GSM8K |
|---|---|---|---|
| `q4_k_m` | 17.3 GiB | 82 | reference: 150/164, 239/250 |
| `ud-q3_k_xl` | 12.9 GiB | 115 | 0.044 / 151/164 / 241/250 |
| `iq3_xxs` | 11.4 GiB | ~113 | 0.076 / - / - |
| `q2_k` (default) | 10.2 GiB | 185-197 | 0.098 / 147/164 / 240/250 |

The three smaller files come from `python3 tools/download_quants.py` (Hugging Face,
sha256-verified, 44 GB total with UD-IQ2_XXS). If the chosen file is missing the app
falls back to Q4_K_M.

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

If CUDA doesn't come up at all (e.g. after the GPU drops off the PCIe bus),
llama-server would quietly run on the CPU; `BigModelServer` refuses that and
raises `ServerUnavailable` with what to do. If the server dies mid-reply, the chat
marks the reply incomplete and shows the error. Covered by
`python3 tests/test_server_failures.py`.

(The original config pinned **every** expert to RAM with `-ot`, leaving 10 GB of
the card empty: 47 tok/s. See `SPEC_DECODING.md` §5.)

### Request routing

`router.prompt_bucket` sorts each prompt, matching whole words. Code and
analysis requests (write, explain, debug, Python…) always go to the 30B. Short
prompts and who/what/why questions otherwise count as `quick`.

- **`quick`**: a small draft model (Qwen2.5-Coder-0.5B, on the CPU) answers on
  its own, shown immediately. The 30B then checks the answer in the background,
  corrects it if they disagree, and turns each disagreement into a training
  example. Both models get the conversation formatted with the 30B's chat
  template, as the same token ids. Conversations longer than 512 tokens skip the
  draft and go straight to the 30B.
- **everything else**: the 30B alone, using its chat template and the full
  conversation, **streamed** as it generates.

Routing is covered by `python3 tests/test_router.py`.

### Online self-improvement

`draft_trainer.py` LoRA-tunes the draft model from those disagreements, in the
background. Every `refresh_every_n_steps` steps it merges the adapter,
reconverts to GGUF, requantizes, and hot-swaps the running draft with no
restart.

It runs **on the CPU with 4 threads** by default (`trainer_device`,
`trainer_cpu_threads`). On the GPU it takes VRAM from the 30B's experts on every
reply (~62 vs ~71 tok/s, measured with Q4_K_M). With all CPU cores, it slows replies to 27-43 tok/s
while a training step runs.

## Setup

1. **`llama-server`:** `config.py` points at the CUDA 12 build bundled with LM
   Studio (`Paths.llama_server_bin`) and its vendored CUDA libraries
   (`llama_server_ld_library_path`). Point them at your own llama.cpp build if
   you have one. `llama-quantize` is needed only for the self-improvement
   refresh.
2. `pip install -r requirements.txt`
3. In `config.py`, set:
   - `BIG_MODELS`: paths to the 30B model files (or just set `AI2_BIG_MODEL`).
   - `draft_model_gguf`: the small GGUF draft used by the quick path.
   - `draft_model_hf`: the same draft in `transformers` format. This is the
     trainable copy, because you can't backprop through a quantized GGUF.
4. `python app.py`, then open http://127.0.0.1:7860

### Settings that matter (`config.Runtime`)

| setting | default | what it does |
|---|---|---|
| `n_cpu_moe` | per model (0 for Q2_K, 12 for UD-Q3_K_XL) | fastest split to *try*; launch backs off from here |
| `n_cpu_moe_step` | 1 | layers moved to RAM per failed attempt |
| `vram_headroom_mb` | 256 | free VRAM to leave for other programs; the server itself allocates nothing after warm-up. 256 is the largest margin that still fits every expert layer -- below it nothing changes |
| `repeat_penalty` | 1.0 | off: 1.1 cost 6% at ~190 tok/s and skewed the quick path's check (the draft samples at 1.0) |
| `quick_max_prompt_tokens` | 512 | longer conversations skip the CPU draft and go to the 30B |
| `server_slots` | 1 | llama-server parallel slots; 1 is +2.8 tok/s for a single user |
| `trainer_device` | `cpu` | where the trainer's model lives |
| `trainer_cpu_threads` | 4 | caps training's CPU use so replies don't stall |
| `threads` | 16 | llama-server CPU threads (only used for experts left in RAM; at `--n-cpu-moe 0` there are none) |
| `n_ctx` | 8192 | context length. Past it the server rejects the request outright (it does not truncate), so this is a hard edge. KV cache competes with the experts for VRAM: 8192 costs 2 layers and 6% of decode, 16384 costs 5 and 14%, 32768 costs 13 and 33% |

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
