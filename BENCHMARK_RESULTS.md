# AI2 3-way benchmark results (historical: 2026-09-16)

> **Read this as a record of where the project started, not as current numbers.**
> Everything below was measured with every MoE expert pinned to CPU via `-ot`,
> which left 10 of the card's 12 GB unused, so the whole table is ~4.5x behind
> the current default configuration. For what the stack does now see
> [`README.md`](README.md) (summary), [`PLAN.md`](PLAN.md) (state and evidence)
> and [`SPEC_DECODING.md`](SPEC_DECODING.md) (speculation in detail).
>
> The same five prompts at the same 128-token ceiling, re-run on the current
> configuration (`python3 experiments/rerun_original_benchmark.py`), are in
> [Re-run on the current stack](#re-run-on-the-current-stack-2026-09-18) at the
> bottom: **4.1x**, 46.19 -> 188.5 tok/s.
>
> **CORRECTION (superseded in part by `SPEC_DECODING.md`).** Three findings below
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
> 3. **"Speculative decoding isn't a productive place to keep optimizing"** was
>    the conclusion drawn from finding 1, and it is wrong twice over. The run it
>    rests on had no speculation in it at all, and n-gram drafting -- which needs
>    no draft model, so no vocabulary to mismatch -- is now on by default and
>    worth +8% on code edits (196.5 vs 181.2 tok/s at the current split) and +22%
>    in `experiments/ngram_q2k.py`. What *was* right is narrower and still holds:
>    a **separate 0.5B draft model** does not pay on this hardware, re-tested at
>    Q2_K in `experiments/draft_model_q2k.py` (0.96x at best).
>
> One later caveat on speculation, from `experiments/spec_determinism.py`: it is
> not bit-identical. The same prompt answered three times gives three different
> replies with it on, one reply with it off. It costs no measured accuracy
> (HumanEval 147/164 with, 145/164 without, across 414 graded problems), but it
> is not the "same tokens" guarantee it was once described as.
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

## Re-run on the current stack (2026-09-18)

The same five prompts, the same 128-token ceiling, on the current default
(Q2_K, `--n-cpu-moe 2`, ubatch 1024, n-gram speculation, `n_ctx` 8192, 175 W
cap). Medians of 2 interleaved rounds, `experiments/rerun_original_benchmark.py`:

| Prompt | 2026-09-16 tok/s | now | no spec | speedup |
|---|---|---|---|---|
| `Write a Python function that reverses a sing...` | 45.79 | **190.5** | 189.7 | **4.2x** |
| `Explain the CAP theorem in two sentences.` | 45.35 | **187.0** | 189.6 | **4.1x** |
| `What is 17 times 24?` | 47.15 | **188.6** | 188.9 | **4.0x** |
| `Summarize the plot of Romeo and Juliet in on...` | 47.36 | **188.4** | 190.1 | **4.0x** |
| `Fix this bug: `def add(a, b): return a - b`` | 45.30 | **187.9** | 189.3 | **4.1x** |
| **average** | **46.19** | **188.5** | 189.5 | **4.1x** |

Almost none of that 4.1x is speculation. Nearly all of it is the MoE split
(`--n-cpu-moe`, fitted at launch to the VRAM actually free) and the move from
Q4_K_M to Q2_K; `-ub 1024` and the 175 W cap account for the rest.

**Speculation measures 0.99x here, and that is the expected result.** N-gram
drafting proposes tokens that already appeared in the context, so it can only
pay when the reply repeats the prompt -- editing, refactoring, adding types to
pasted code. All five of these prompts are write-from-scratch, which is the one
workload where it has nothing to draft. On edit prompts the same setting is
worth +8% at this split and +22% in `experiments/ngram_q2k.py`.

That is worth saying plainly, because it is the trap this file fell into the
first time. The original conclusion -- "speculative decoding isn't a productive
place to keep optimizing" -- was reached from a run in which speculation never
engaged at all. Had it engaged, this prompt set would *still* have shown ~1.0x,
and the conclusion would have looked confirmed while remaining wrong. The
prompt set, not just the measurement, was the problem.
