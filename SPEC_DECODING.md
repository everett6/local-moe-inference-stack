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
| **`--n-cpu-moe` instead of pinning all experts to CPU** | **47.1 → 81.7 tok/s (1.74x)** on an empty card. Applied, with launch-time fitting so it can't fail to start (§5). |
| Code-specialised drafting, with confidence gating | 85-91% acceptance on code, **still 0.89-0.93x** once the draft's VRAM cost is counted (§4c) |
| EAGLE3 head, as shipped | 0.61-0.76x (slower than no speculation) |
| Qwen2.5-Coder-0.5B draft | **Never ran.** Vocab incompatible with the Qwen3 target; llama-server logged an error and served without speculation. |
| Qwen3-0.6B draft (the vocab-compatible substitute) | 0.62-0.91x at 18.5-61.3% acceptance |
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
  target at any accuracy — wrong vocabulary. Its vocab-compatible substitute,
  Qwen3-0.6B, was measured (§4b) and also loses: 0.91x at its best setting.
- **Self-training the EAGLE3 head would have to take it from ~25% to ~85%
  acceptance** to stop losing, and clearing that bar returns roughly 1.1x. The
  published ceiling for well-trained EAGLE3 heads is ~80%, which on this curve
  is break-even. The project's *best realistic outcome* is "no longer slower".

That is not a statement about EAGLE3 or about training. It is the shallow
batching curve in §1: this target's decode step does not get cheap enough per
position to pay for speculation.

### 4b. Qwen3-0.6B, the real version of the "small draft model" question

`experiments/qwen3_draft_test.py`. Draft served alone first to measure its own
decode cost — **498.8 tok/s standalone, so d = 2.00 ms/token**, which is the one
term §2 has to be given rather than derive. Then end-to-end at `--n-cpu-moe 24`
(not 20: the draft needs VRAM too, and its baseline is measured in the same run
so the ratio stays honest). Baseline 73.01 tok/s:

| draft length k | tok/s | vs base | acceptance | mean accepted len |
|---|---|---|---|---|
| 1 | 66.17 | 0.91x | 61.3% | 1.61 |
| 2 | 60.29 | 0.83x | 42.4% | 1.85 |
| 3 | 56.40 | 0.77x | 40.3% | 2.21 |
| 4 | 51.55 | 0.71x | 31.6% | 2.26 |
| 6 | 44.98 | 0.62x | 18.5% | 2.11 |

61.3% acceptance at k=1 is a perfectly healthy draft model — and break-even at
this config is **75%**. It falls short and it loses. That is the cleanest
statement of the whole result: *the draft model is not the problem.*

**This also validates the model in §2**, which was built from the batching curve
alone, before any of these arms ran:

| k | acceptance | §2 predicts | measured |
|---|---|---|---|
| 1 | 61.3% | 0.92x | **0.91x** |
| 2 | 42.4% | 0.74x | 0.83x |
| 3 | 40.3% | 0.63x | 0.77x |

Exact at k=1, and conservative by 0.09-0.14x at k=2-3. The gap is the geometric
acceptance assumption: real acceptance is bursty rather than i.i.d., so measured
`mean len` runs above `sum(a^i)` (1.85 measured vs 1.60 predicted at k=2). So §2
slightly *understates* what a draft delivers — and every arm still loses.

### 4c. What about a *code* draft? Acceptance clears the bar; the VRAM bill doesn't

The natural follow-up to "would the coder 0.5B be better": acceptance is not
uniform. EAGLE3 ranged 6.6% to 54.9% by prompt, best on code. Maybe a draft
clears the bar on code specifically and the mixed prompt set hid it. There was
also one lever no arm had used: `--spec-draft-p-min`, which drafts only when the
draft is confident. Every run above used the default of 0.0, i.e. always draft.

`experiments/workload_pmin_sweep.py`, same Qwen3-0.6B draft, run separately on
pure-code and pure-prose prompts at `--n-cpu-moe 24` (acceptance now summed over
every slot's log line, fixing the last-slot caveat from §3a):

| arm | code tok/s | code accept | prose tok/s | prose accept |
|---|---|---|---|---|
| baseline | 66.47 | — | 69.79 | — |
| k=1, p-min 0.0 | 0.97x | 75.6% | 0.84x | 57.9% |
| k=2, p-min 0.8 | 0.99x | 91.0% | 0.83x | 85.2% |
| k=4, p-min 0.8 | **1.06x** | 87.7% | 0.85x | 83.5% |

So there is something here. On code, ungated acceptance is 75.6%, right on the
75% break-even, and gating pushes it to 88-91%. That's the first arm in this
investigation to beat its baseline. Prose loses however it's tuned.

Two reasons not to take the 1.06x at face value:

1. **Noise.** That code baseline was 66.47 tok/s. The same config measured 73.01
   an hour earlier. A ~10% swing is bigger than the 6% win.
2. **Wrong baseline.** The draft takes VRAM that would otherwise hold experts,
   and §5 shows experts in VRAM are worth more per GB than anything else here.
   Comparing draft vs no-draft at the *same* split hides that cost.

`experiments/code_draft_confirm.py` fixes both: three interleaved rounds (so
drift hits every arm equally), and every arm fitted to the tightest `--n-cpu-moe`
it can actually load at, so the draft pays for its VRAM in experts:

| arm (code prompts) | fitted split | median tok/s | min-max | vs no draft | acceptance |
|---|---|---|---|---|---|
| **no draft** | 21 | **79.57** | 78.85-80.05 | 1.00x | — |
| k=6, p-min 0.8 | 24 | 74.15 | 71.46-75.22 | 0.93x | 84.7% |
| k=4, p-min 0.8 | 24 | 72.89 | 72.59-73.92 | 0.92x | 87.7% |
| k=8, p-min 0.9 | 24 | 70.76 | 67.82-71.15 | 0.89x | 90.5% |
| k=6, p-min 0.9 | 24 | 68.42 | 67.89-70.46 | 0.86x | 90.8% |

The ranges don't overlap. The draft costs three layers of experts (21 → 24),
worth ~6-7 tok/s. At the same split, the best draft arm is roughly at parity
with no draft (74.15 vs the 73-74 that split measures without one). So on code,
a well-gated draft about pays for its own compute and nothing more. It can't
cover the experts it pushed off the card.

**That answers the coder-draft question for good.** A code-specialised draft at
85-91% acceptance is about the best case any draft-model project here could
produce, self-trained or not, and it's still 7% slower than giving the same
VRAM to experts.

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
and `benchmark_all.py`. Verified end-to-end through the app's own launch path
(`BigModelServer`, not the experiment harness) on the exact prompt
`BENCHMARK_RESULTS.md` measured at 45.79 tok/s: **79.72 tok/s**.

### The first version of this broke startup, now fixed

Hardcoding 20 was a mistake, and it shipped in the first commit on this branch.
20 leaves ~700 MiB free on an **empty** card, and a real desktop doesn't have an
empty card:

- **Firefox held 456 MiB** during a later run, and 20 failed to load. It missed
  by 221 MiB (`cudaMalloc failed: out of memory` in the compute-buffer reserve).
- **`LocalMoEEngine` loads its draft model onto the GPU (~600 MiB) *before*
  starting the 30B.** So `app.py` would have failed to start at 20 even with no
  browser open. The earlier 79.72 tok/s check missed this because it launched
  `BigModelServer` directly and skipped the draft.

The fix is to pick the split at launch instead of hardcoding it.
`BigModelServer` now starts at `Runtime.n_cpu_moe` (still 20, the fastest), and
if llama-server dies during load, or loads but leaves less than
`Runtime.vram_headroom_mb` (768) free, it stops the process, waits for VRAM to
actually be released, and retries with `n_cpu_moe_step` (2) more layers on CPU,
up to all 48. The chosen value is on `BigModelServer.n_cpu_moe`. A failed
attempt costs a second or two, since llama-server exits quickly on a failed
allocation.

Verified on exactly the case that broke: full `LocalMoEEngine`, draft on the
GPU, Firefox open. It stepped 20 → 22 → 24, was up in 7 s with 1144 MiB free,
and ran **73.76 tok/s**. That's the realistic number for `app.py` as built.
81.7 is the ceiling on an otherwise empty card.

`benchmark_all.py`'s native-spec launch has no such loop and also puts a draft on
the card, so it now uses `max(n_cpu_moe, 24)`.

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

1. **Nothing, for speculative decoding of any kind.** All four families are now
   measured — EAGLE3, a standalone small draft, ngram lookup, and the
   vocab-mismatched coder draft that never ran. The self-distillation pipeline is built and
   documented (§7) but the measurements say not to run it. It is there so the
   decision can be revisited if the hardware changes — a card that fits the
   whole model would steepen the curve and flip this.
2. ~~Test the ngram strategies properly~~ — **done, §3a. Also a loss (0.94-0.99x).**
   That was the last speculative option with a plausible path to a win, and it
   closed. Nothing in the speculative-decoding family is worth further time on
   this hardware.
3. **Re-run `benchmark_all.py`** — every number in `BENCHMARK_RESULTS.md`
   predates the MoE fix and is now ~1.74x pessimistic.
4. **The EAGLE3 quantization diagnostic is built but not run**, and is now
   optional rather than blocking. `experiments/eagle3_quant_diagnostic.py` tests
   whether quantizing the target is what wrecks EAGLE3's acceptance, using
   Qwen3-1.7B (dense, fits in VRAM at BF16) so quantization is the only variable.
   It needs a 3.4 GB BF16 GGUF that was still downloading when this was written.
   Worth knowing, but it no longer changes the decision: §2 caps the payoff from
   *any* draft-head improvement at roughly 1.1x, so "yes, quantization is the
   cause" and "no, it isn't" lead to the same recommendation.
5. **Your call: the app's own draft model costs ~6 tok/s of VRAM.**
   `LocalMoEEngine` keeps Qwen2.5-Coder-0.5B on the GPU for the hand-rolled
   Python speculative loop (0.25x in `BENCHMARK_RESULTS.md`) and for online
   LoRA training. It forces the big model from split 20 or 21 to 24: 73.8 tok/s
   instead of ~80. Loading that draft with `n_gpu_layers=0` (CPU) would give
   those experts back, at the price of a slower draft in a loop that's already a
   loss. Not changed here, since the online-training feature depends on it.

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
