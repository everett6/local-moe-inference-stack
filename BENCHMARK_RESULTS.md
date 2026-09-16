# AI2 3-way benchmark results

> **CORRECTION (superseded in part by `SPEC_DECODING.md`).** Two findings below
> are wrong:
>
> 1. **"Native speculative decoding is only ~1.02x"** — speculative decoding was
>    never running. Qwen2.5-Coder-0.5B is a Qwen2.5 model and the target is
>    Qwen3; llama-server logs `the target and draft vocabs are not compatible`
>    and then keeps serving *without* speculation. The 1.02x was plain
>    generation. Conclusion #2 below, and the "speculative decoding isn't a
>    productive place to optimize" reasoning built on it, do not follow from
>    this data.
> 2. **Every tok/s number here is ~1.74x pessimistic.** They were measured with
>    all MoE experts pinned to CPU, which left 10 of the card's 12 GB unused.
>    `--n-cpu-moe 20` raises the baseline from 47.1 to 81.7 tok/s. Applied in
>    `config.py`; this file has not been re-measured.
>
> The thread-count and flash-attention tuning below still stands.

Hardware: RTX 5070 (12GB VRAM) + 32GB RAM, 32 CPU threads. Model: Qwen3-30B-A3B-2507
(Q4_K_M), MoE experts pinned to CPU RAM via `-ot`, draft model Qwen2.5-Coder-0.5B (Q8_0).
Prompts: 5 real, varied prompts (code, math, summarization, bug-fix). max_tokens ceiling
128 per prompt; actual length varies since all three modes now correctly stop on the
model's own EOS token (this was a real bug in the hand-rolled path before this session
fixed it -- it previously always ran to the full ceiling regardless of natural stopping
points).

## Before this benchmark: bugs fixed and tuning applied

- Fixed a bug where the hand-rolled speculative loop's draft/verify comparison always
  returned a 0% accept rate (relied on an `n_predict: 0` scoring mode this llama-server
  build silently doesn't support) -- accept rate is now genuinely 33-80% depending on
  prompt, as shown below.
- Fixed a bug where that same loop never checked for the model's stop token, so it
  always ran to the full token ceiling instead of stopping naturally.
- Thread count tuned empirically: tested 8/12/16/20/24/32 threads on this hardware.
  16 is the real optimum (~48 tok/s); the original code's default of 8 left ~12% on
  the table, and naively maxing out at 32 threads (all cores) made throughput
  **28% worse** (~34 tok/s) from memory-bandwidth contention, not better.
- Flash attention forced `on` (small, consistent ~2% gain over `auto`).
- KV cache quantization (`q8_0`) tested and rejected -- no throughput benefit here
  since VRAM isn't the bottleneck for this workload.

## Key findings

1. **The hand-rolled Python speculative decoding path is 4x *slower* than doing
   nothing (0.25x)**, even with both correctness bugs fixed and a 33-80% draft accept
   rate. Root cause (verified empirically, not guessed): this llama-server build's
   HTTP API has no way to teacher-force-score a hypothetical continuation without
   paying full generation cost, so "verifying" k draft tokens costs as much big-model
   compute as generating k tokens directly, on top of draft-model overhead and
   extra HTTP round trips.
2. ~~**Native C++ speculative decoding (llama-server's own `--spec-type draft-simple`)
   is only ~1.02x -- essentially parity with plain generation, not a real win either**~~
   **RETRACTED -- speculation never engaged (vocab mismatch, see the correction at
   the top). The measurement below is plain generation.** The original text:
   on this specific 30B-A3B MoE + 0.5B draft pairing on this hardware. This is the
   more important finding: it means speculative decoding itself isn't currently a
   productive place to keep optimizing for this model/hardware combination, even
   using llama.cpp's real, properly-batched C++ implementation -- the bottleneck is
   elsewhere (most likely CPU-side MoE expert compute, which speculative decoding
   doesn't reduce: verifying more tokens per big-model call doesn't shrink the total
   amount of expert computation needed).
3. Thread count and flash-attention tuning were the only levers that produced a real,
   reproducible improvement in this session -- both now applied in `config.py` /
   `local_engine.py`.

## Summary (aggregate tok/s across all prompts)

| Mode | Avg tok/s | Speedup vs baseline |
|---|---|---|
| Baseline (no speculative decoding) | 46.29 | 1.00x |
| Hand-rolled Python speculative (this project) | 11.76 | 0.25x |
| ~~Native llama-server speculative (--spec-type draft-simple)~~ *(speculation never engaged -- this is the baseline again)* | 47.37 | 1.02x |

## Baseline -- per prompt

| Prompt | Tokens out | Time (s) | Tok/s |
|---|---|---|---|
| `Write a Python function that reverses a singly ...` | 128 | 2.80 | 45.79 |
| `Explain the CAP theorem in two sentences.` | 87 | 1.92 | 45.35 |
| `What is 17 times 24?` | 128 | 2.71 | 47.15 |
| `Summarize the plot of Romeo and Juliet in one s...` | 128 | 2.70 | 47.36 |
| `Fix this bug: `def add(a, b): return a - b`` | 95 | 2.10 | 45.30 |

## Hand-rolled speculative -- per prompt

| Prompt | Tokens out | Time (s) | Tok/s | Draft accept rate |
|---|---|---|---|---|
| `Write a Python function that reverses a singly ...` | 128 | 7.24 | 17.67 | 79.9% |
| `Explain the CAP theorem in two sentences.` | 87 | 9.84 | 8.84 | 42.2% |
| `What is 17 times 24?` | 128 | 7.76 | 16.49 | 75.4% |
| `Summarize the plot of Romeo and Juliet in one s...` | 128 | 16.18 | 7.91 | 33.5% |
| `Fix this bug: `def add(a, b): return a - b`` | 59 | 4.05 | 14.55 | 68.2% |

## Native speculative -- per prompt

| Prompt | Tokens out | Time (s) | Tok/s |
|---|---|---|---|
| `Write a Python function that reverses a singly ...` | 128 | 2.69 | 47.50 |
| `Explain the CAP theorem in two sentences.` | 88 | 1.87 | 47.07 |
| `What is 17 times 24?` | 128 | 2.66 | 48.16 |
| `Summarize the plot of Romeo and Juliet in one s...` | 128 | 2.72 | 47.13 |
| `Fix this bug: `def add(a, b): return a - b`` | 96 | 2.05 | 46.77 |
