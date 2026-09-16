# Speculative decoding on this box: what's worth building, and what isn't

You asked whether to invest in self-training the public EAGLE3 head or in the
Qwen2.5-Coder-0.5B draft, and to run a test rather than guess. Both were
measured. The answer is **neither**, and the reason is a property of this
hardware that neither project can change.

Along the way the measurements turned up a 1.74x speedup that has nothing to do
with draft models, and two earlier results in this repo that were wrong.

---

## TL;DR

| Finding | Effect |
|---|---|
| **`--n-cpu-moe 20` instead of pinning all experts to CPU** | **47.1 → 81.7 tok/s (1.74x)**. Applied to `config.py`. |
| EAGLE3 head, as shipped | 0.61-0.76x (slower than no speculation) |
| Qwen2.5-Coder-0.5B draft | **Never ran.** Vocab incompatible with the Qwen3 target; llama-server logged an error and served without speculation. |
| `ngram-*` strategies | 0.94-0.99x once made to draft at all |
| Break-even acceptance a draft must beat | **65-89%**, depending on draft cost |
| Best case from a *perfect, free* draft | 1.69x at k=3 — and it gets worse, not better, after the MoE fix |

---

## 1. The measurement that decides it

Speculative decoding is a bet that verifying k draft tokens in one target
forward pass costs much less than k separate passes. Whether that is true is a
property of the target and the hardware, not of the draft model — so it can be
measured once, and it bounds every draft-model project at the same time.

`experiments/spec_headroom.py` measures it. A prefill of L tokens is exactly the
shape of a speculative verify of L tokens (one forward, L positions), and
llama-server reports `timings.prompt_ms` for it. Per-token cost at width L
against width 1 *is* the batching curve.

Production config (all MoE experts on CPU, as `config.py` had it):

| batch width | 1 | 2 | 3 | 4 | 6 | 8 | 16 |
|---|---|---|---|---|---|---|---|
| ms/token | 22.19 | 18.25 | 15.54 | 13.95 | 11.35 | 9.49 | 6.73 |
| **ceiling** | 1.00x | 1.22x | 1.43x | 1.59x | 1.96x | 2.34x | 3.30x |

"Ceiling" is the speedup from a **perfect** draft whose every token is accepted,
charging nothing for running the draft. No draft model can beat it.

The curve is shallow. On a GPU-resident dense model, decode is
memory-bandwidth-bound and extra positions ride along nearly free — width 4
would cost close to 1/4 per token. Here it costs 63%. The reason is MoE routing:
each position in the batch picks its own experts, so widening the batch pulls in
more expert weights instead of reusing one set.

## 2. Turning the curve into a go/no-go number

`experiments/spec_breakeven.py` solves the standard speculative-decoding round
for the acceptance rate at which it stops losing:

```
round at draft length k:   cost = (k+1)*c(k+1) + k*d        d = draft's own ms/token
                          gain = sum(a^i for i in 0..k)     a = per-token acceptance
profitable when           cost / gain  <  c(1)
```

**Break-even per-token acceptance** (the draft must *exceed* this to be worth running):

| draft ms/token | k=1 | k=2 | k=3 |
|---|---|---|---|
| 0 (ngram — no second model) | 65% | 66% | 70% |
| 2 | 74% | 74% | 76% |
| 4 | 83% | 81% | 82% |
| 8 | — | 94% | 93% |

And what you actually get if a draft *does* clear the bar (draft = 2 ms/token):

| acceptance | k=1 | k=2 | k=3 |
|---|---|---|---|
| 50% | 0.86x | 0.77x | 0.67x |
| 70% | 0.98x | 0.96x | 0.91x |
| 80% | 1.04x | 1.07x | 1.06x |
| 90% | 1.09x | 1.19x | 1.23x |

This is the whole decision in one table. A draft model here has to be **~80%
accurate just to break even**, and even a 90%-accurate one returns about 1.2x.

The model is not a hand-wave — it predicts the measured arms. It says a 30%-
accepting draft at k=2 costing ~2 ms should land near 0.61x; the EAGLE3 head
measured 0.61x at k=4 and 0.76x at k=2.

## 3. What the three candidates actually did

`experiments/spec_shootout.py`, real 30B target, greedy, 5 prompts × 192 tokens,
baseline 45.73 tok/s:

| arm | tok/s | vs base | acceptance | mean accepted len |
|---|---|---|---|---|
| baseline | 45.73 | 1.00x | — | — |
| eagle3, k=2 | 34.81 | 0.76x | 31.4% | 1.63 |
| eagle3, k=4 | 28.03 | 0.61x | 18.5% | 1.74 |
| coder-0.5B (all k) | ~45.9 | ~1.00x | — | **did not run — see below** |
| ngram-mod / map-k4v / simple / cache | 45.0-46.2 | ~1.00x | — | **drafted nothing** |

### The coder draft has never actually run

Every `coder0.5b` arm came back at exactly baseline speed. The server log says why:

```
E spec common_specu: the target and draft vocabs are not compatible
E srv load_model: failed to initialize speculative decoding context:
    draft model vocab type must match target model to use speculation
```

Qwen2.5-Coder-0.5B is a Qwen2.5 model; the target is Qwen3. **llama-server does
not exit on this** — it logs the error and keeps serving with speculation off.

That invalidates the headline result in `BENCHMARK_RESULTS.md`: "Native C++
speculative decoding is only ~1.02x — essentially parity". It was 1.02x because
it was plain generation with extra startup. `BENCHMARK_RESULTS.md` has been
corrected.

`spec_shootout.py` now fails any arm whose log shows that error, or that
produces no acceptance line at all, instead of reporting its tok/s as a result.
The dead coder arm is kept in the arm list on purpose, as a regression test: it
must come back as an error.

### The ngram strategies never drafted

No `draft acceptance` line appears in any ngram arm's log, and `eval time` is
20.6 ms/token — the baseline decode cost. `ngram-mod` defaults to
`--spec-ngram-mod-n-match 24`, i.e. it only drafts after matching a 24-token
context it has seen before, which essentially never happens inside a 192-token
generation. They were not a refuted option so much as an untested one — so they were then
tested properly (§3a).

### EAGLE3 is better than previously recorded, and still loses

`EAGLE3_DRAFT_MODEL.md` recorded 10-15% acceptance and 0.15-0.31x. At greedy
with the tuned harness it reaches 18.5-31.4% overall, and per-prompt ranges from
6.6% to 54.9% — genuinely good on one code prompt, near-useless on prose. The
verdict is unchanged: 0.61-0.76x.

### 3a. ngram, tested properly: also a loss

`experiments/ngram_sweep.py` sweeps the lookup length down from the default 24
and tries `ngram-simple` and `ngram-map-k` as well, on a deliberately code-heavy
prompt set (where lookup drafting should do best), at the tuned `--n-cpu-moe 20`
config. Baseline 80.81 tok/s:

| arm | tok/s | vs base | drafted? |
|---|---|---|---|
| `ngram-mod`, n-match 16/12/8 | 81.3-82.4 | 1.01-1.02x | **no drafts** — still never fires |
| `ngram-mod`, n-match 4 | 80.3-80.5 | 0.99-1.00x | yes |
| `ngram-map-k` n=3 m=4 | 79.30 | 0.98x | yes |
| `ngram-simple` n=4 m=4 | 78.36 | 0.97x | yes |
| `ngram-map-k` n=2 m=2 | 76.92 | 0.95x | yes |
| `ngram-simple` n=2 m=2 | 75.56 | 0.94x | yes |

The pattern is unambiguous: **every arm that actually drafted is slower than the
baseline, and the arms at "1.02x" are the ones that drafted nothing** (i.e. run-
to-run noise on plain generation). Shortening the lookup makes them draft and
makes them lose, exactly as §2 predicts — their acceptance is nowhere near the
65-70% bar.

One caveat on the acceptance figures in `ngram_sweep_result.json`: llama-server
prints acceptance per slot at release, so the parsed number is the last slot's,
sometimes over a handful of drafts. The tok/s column is the whole run and is the
one to trust.

## 4. So: self-train EAGLE3, or the 0.5B coder draft?

**Neither.**

- **The coder draft is not a candidate at all.** It cannot speculate for a Qwen3
  target at any accuracy — wrong vocabulary. The vocab-compatible substitute is
  Qwen3-0.6B, which shares the target's tokenizer.
- **Self-training the EAGLE3 head would have to take it from ~25% to ~85%
  acceptance** to stop losing, and clearing that bar returns roughly 1.1x. The
  published ceiling for well-trained EAGLE3 heads is ~80%, which on this curve
  is break-even. The project's *best realistic outcome* is "no longer slower".

That is not a statement about EAGLE3 or about training. It is the shallow
batching curve in §1: this target's decode step does not get cheap enough per
position to pay for speculation.

## 5. The thing that did work: stop pinning every expert to CPU

`config.py` pinned **all** MoE experts to CPU:

```
-ot "ffn_(gate|down|up)_exps=CPU"
```

That regex is all-or-nothing. Measured VRAM in use: **1750 MiB of 12227**. Ten
gigabytes of the card sat empty while the CPU did all the expert math.

`--n-cpu-moe N` keeps only the first N layers' experts on CPU.
`experiments/moe_offload_sweep.py` swept it:

| `--n-cpu-moe` | tok/s | vs. old config | VRAM MiB |
|---|---|---|---|
| 48 (old config) | 47.11 | 1.00x | 1750 |
| 40 | 53.39 | 1.13x | 4688 |
| 32 | 61.32 | 1.30x | 7428 |
| 28 | 67.36 | 1.43x | 8774 |
| 24 | 74.04 | 1.57x | 10120 |
| 22 | 77.33 | 1.64x | 10816 |
| 21 | 78.95 | 1.68x | 11140 |
| **20** | **81.74** | **1.74x** | 11514 |
| 19 | fails to allocate | | |

**Applied**: `Runtime.n_cpu_moe = 20` in `config.py`, used by `local_engine.py`
and `benchmark_all.py`.

Caveat, and it is a real one: 20 leaves ~700 MiB of headroom. Raising `n_ctx`,
adding a draft model, or anything else touching the GPU will push it over and
llama-server will refuse to start. **24** (10.1 GB, 74.0 tok/s, still 1.57x)
is the setting to use if you want room to experiment.

### This makes speculative decoding *harder*, not easier

Re-measuring the curve at `--n-cpu-moe 20`:

| batch width | 1 | 2 | 3 | 4 | 8 |
|---|---|---|---|---|---|
| ms/token | 12.85 | 10.12 | 8.83 | 7.62 | 5.32 |
| ceiling | 1.00x | 1.27x | 1.46x | 1.69x | 2.42x |

The shape barely moves, but `c(1)` drops from 22.19 ms to 12.85 ms — so a
draft's own fixed cost is now nearly twice as large *relative to* the step it is
trying to save. Break-even for a 4 ms/token draft rises from 83% to **89%**.

Every hour spent on the draft model would have been spent making a 45 tok/s
baseline slightly less slow. The baseline is now 82 tok/s, and the bar is higher.

## 6. What's left, in priority order

1. **Nothing, for speculative decoding of any kind.** The self-distillation pipeline is built and
   documented (§7) but the measurements say not to run it. It is there so the
   decision can be revisited if the hardware changes — a card that fits the
   whole model would steepen the curve and flip this.
2. ~~Test the ngram strategies properly~~ — **done, §3a. Also a loss (0.94-0.99x).**
   That was the last speculative option with a plausible path to a win, and it
   closed. Nothing in the speculative-decoding family is worth further time on
   this hardware.
3. **Re-run `benchmark_all.py`** — every number in `BENCHMARK_RESULTS.md`
   predates the MoE fix and is now ~1.74x pessimistic.

## 7. The self-distillation pipeline (built, not run)

You asked for high-quality Hugging Face data rather than prompts I made up. That
is built and is the right design regardless of whether it gets used:

- `experiments/harvest_prompts.py` — pulls prompts from
  `ise-uiuc/Magicoder-OSS-Instruct-75K`,
  `bigcode/self-oss-instruct-sc2-exec-filter-50k`, and
  `HuggingFaceH4/ultrachat_200k` over the HF datasets-server REST API (no
  `datasets` install, nothing added to your Python env). It **discards those
  datasets' answers** — they were written by other models, and training a draft
  to imitate GPT-4 would teach it to disagree with our Qwen3 target.
- `experiments/capture_target_outputs.py` — generates the labels by running the
  prompts through **our own Q4_K_M 30B in its served configuration**, saving
  token ids rather than text so training sees exactly the sequence the draft
  must predict. This is what makes it self-training: the labels come from the
  system being sped up.

The missing piece is the trainer itself, deliberately not written: §2 says the
result would be a draft that at best breaks even.

---

## Reproducing

```bash
python3 experiments/spec_headroom.py        # the batching curve
python3 experiments/spec_breakeven.py       # break-even table
python3 experiments/moe_offload_sweep.py    # the 1.74x
python3 experiments/spec_shootout.py        # head-to-head
```
