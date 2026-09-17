# AI2: where things stand, and what to do next

*Updated 2026-09-16. For the history of how we got here, see
[`EXECUTIVE_SUMMARY.md`](EXECUTIVE_SUMMARY.md) (earlier sessions) and
[`SPEC_DECODING.md`](SPEC_DECODING.md) (this one).*

## Where things stand

**Hardware:** RTX 5070 (12 GB), Ryzen 9 7950X (16 cores), 32 GB RAM.
**Model:** Qwen3-30B-A3B-Instruct-2507, Q4_K_M, served by LM Studio's bundled
`llama-server`.

### Speed

| setup | decode tok/s |
|---|---|
| Original config (every MoE expert in RAM) | 47 |
| **The app as it starts today** (trainer on CPU, draft on GPU) | **~71** |
| Ceiling: 30B alone on an otherwise empty card | ~82 |

The gap between 71 and 82 is the in-process Qwen2.5-Coder draft model (~600 MB of
VRAM) that the quick path and the online trainer use.

### How a request is served

1. `router.prompt_bucket` sorts the prompt.
2. **`quick` bucket:** the 0.5B draft model answers on its own and is shown
   immediately. The 30B then replays the answer in the background. Any
   disagreement corrects the displayed answer and becomes a training example.
3. **Everything else:** the 30B alone, with its chat template and the whole
   conversation, streamed as it generates (`BigModelServer.stream_chat`).
4. The trainer (CPU, 4 threads) LoRA-tunes the draft from those mismatches and
   periodically hot-swaps a refreshed GGUF into the engine.

### Settled, don't reopen without new hardware

- **Speculative decoding of any kind loses on this box.** EAGLE3, Qwen3-0.6B,
  ngram lookup, even a code-only draft at 91% acceptance: all 0.6-0.99x. The
  draft's VRAM costs more experts than speculation wins back. Self-training a
  draft can't change that. Full evidence: `SPEC_DECODING.md`.
- **Qwen2.5-Coder-0.5B can't be a llama-server draft for Qwen3.** It fails the
  vocab check on 4 Qwen3-only special tokens. Plain text tokenizes identically,
  so the app's own hand-rolled use of it is fine.
- **Trainer belongs on the CPU with capped threads.** GPU trainer costs ~9 tok/s
  on every reply. Uncapped CPU trainer halves decode while it trains.

### Fixed this session

- 1.74x from `--n-cpu-moe` instead of pinning every expert to CPU.
- `BigModelServer` fits the split at launch, 1 layer at a time, with a
  ~700-token warm-up, so the app can't fail to start or crash on its first
  message when VRAM is tight. The warm-up exists because a server that loaded
  fine crashed on its first real prompt (CUDA allocates the cuBLAS workspace
  lazily).
- Trainer moved to the CPU, capped at 4 threads: ~71 vs ~62 tok/s.
- One llama-server slot instead of four: +2.8 tok/s. KV cache precision,
  threads and ubatch re-tested; the existing values were already best.
- Chat template + conversation history + streaming for 30B replies, with
  correct UTF-8 (the first version garbled "—", emoji, non-Latin text).
- `BENCHMARK_RESULTS.md`'s "native speculative = 1.02x" was plain generation
  (the draft never loaded). Corrected.

## Next, in priority order

### 1. Router sends many real questions to the 0.5B model (bug, small)

`prompt_bucket` checks for `"what"`, `"why"`, `"no"`, `"fix"`… as **substrings**.
`"no"` matches *know, now, another, note, cannot*; `"what"` matches *whatever*.
Any prompt that's 12 words or fewer, or contains one of those anywhere, gets
answered by the 0.5B draft instead of the 30B. Match whole words, and reconsider
whether "12 words or fewer" should mean "quick" at all. A short question isn't
an easy question.

### 2. Quick path has the same missing-chat-template bug (medium)

`generate_fast` sends raw text to Qwen2.5-Coder-0.5B-*Instruct*, and
`verify_fast_answer` compares it to the 30B's raw continuation. So both sides of
the check are un-templated. The user sees a continuation, not an answer, and the
trainer learns to predict un-templated text. Fixing it means templating both
sides consistently, which changes the training signal, so treat it as its own
change.

### 3. Is the quick path worth its VRAM? (measure, then decide)

The draft costs ~11 tok/s on every 30B reply (71 vs ~82). Options:
- keep it on the GPU (today)
- load the GGUF draft on the CPU: frees the VRAM, but quick answers get slower
  (0.5B on 16 cores is likely still fast; unmeasured)
- drop the quick path entirely and send everything to the 30B

Measure quick-path latency on CPU vs GPU and how often quick answers get
corrected, then choose.

### 4. Long documents: consider ubatch 1024 (optional)

The settings sweep kept ubatch at 512 because chat replies matter more, but 1024
reads prompts 54% faster (2,439 vs 1,579 tok/s) at ~3 tok/s slower replies. If
you start pasting long files or logs, it's worth making that a per-use setting.
See `SPEC_DECODING.md` §6c.

### 5. Refresh the stale benchmarks

Every number in `BENCHMARK_RESULTS.md` predates this session. Re-run
`benchmark_all.py` once 1-3 are settled, so the new baseline reflects the real
app.

### Not worth doing

- Any draft-model / speculative-decoding work, including EAGLE3 retraining.
- `--spec-type ngram-*`.
- The EAGLE3 quantization diagnostic (answer doesn't change any decision).
