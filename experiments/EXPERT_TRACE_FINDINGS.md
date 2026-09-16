# Expert-selection trace: first experiment results

This is Track 2's "first experiment" from the research roadmap: before building
any prefetcher, is MoE expert selection predictable enough token-to-token to
justify one?

## Method

- Hooked llama.cpp's existing `ggml_backend_sched_eval_callback` (`cb_eval`)
  via `llama-cpp-python`'s low-level ctypes bindings against the real
  Qwen3-30B-A3B-Instruct model -- **no C++ patch, no recompile**.
- The tensor already exists and is already named by llama.cpp itself:
  `ffn_moe_topk-<layer>` at [`llama-graph.cpp:2109`](../llama.cpp-src/src/llama-graph.cpp) --
  an `[n_expert_used, n_tokens]` int32 tensor of selected expert IDs, emitted
  fresh at every layer of every forward pass.
- Ran CPU-only (`n_gpu_layers=0`): `llama-cpp-python` doesn't bind
  `ggml_backend_tensor_get`, so there's no ready way to copy a GPU-resident
  tensor back to host from Python. CPU-only means every tensor's `->data` is
  already host memory, directly readable via ctypes -- no copy step needed.
  Expert *selection* is a property of the gating network's computation, not
  of where that computation physically runs, so this doesn't compromise the
  result, only the collection speed.
- `struct ggml_tensor`'s layout isn't exposed by `llama-cpp-python` either
  (only used as opaque `c_void_p` in its bindings) -- defined by hand from
  `ggml/include/ggml.h` to cast and read `.name` / `.data` / `.ne`.
- Prompt: "The history of the Roman Empire begins with", greedy decode
  (temperature 0), 24 generated tokens + 8 prompt tokens = 32 traced
  positions x 48 layers = 1536 trace rows, written to `expert_trace.jsonl`.
- Confirmed from GGUF metadata: 128 total experts, 8 selected per token
  (`qwen3moe.expert_count=128`, `qwen3moe.expert_used_count=8`).

## Result

**Average overlap between consecutive tokens' selected experts (same layer):
49.6%** across all 48 layers -- i.e., roughly 4 of the 8 experts a token
selects were *also* selected by the immediately preceding token, at the same
layer.

The random-chance baseline for drawing 8 of 128 experts twice independently
is `8*8/128 = 0.5` experts of overlap on average, i.e. **6.25%**. The observed
49.6% is **~8x the random baseline** -- a clear, non-trivial signal that
expert selection has real temporal structure, not noise.

Per-layer entropy tells a consistent story: average 4.96 bits of usage
entropy per layer against a ~6.3-bit ceiling (observed distinct experts per
layer ranged 34-78 of 128) -- usage is concentrated, not uniform, which is a
second independent signal pointing the same direction as the overlap number.

Per-layer detail is not uniform: early/late layers (0, 47) show the lowest
overlap (~17-22%), while a broad middle band (6-9, 18-19, 30-31) shows the
highest (58-73%). If this holds up at scale, it suggests a layer-aware
prefetch strategy would beat a uniform one.

## Honest caveats

- **N=24 generated tokens, one prompt, greedy decoding.** This is a
  proof-of-mechanism pilot, not a statistically powered study. It answers
  "does the collection method work and is there anything here worth
  measuring more" -- yes to both -- not "here is the final predictability
  number."
- Greedy/temperature-0 decoding may have different expert-selection dynamics
  than sampled generation; both should eventually be traced.
- Only tested on a narrative-prose prompt. Code, math, and multi-turn
  conversation prompts could show different overlap characteristics --
  exactly the kind of variation a bigger trace run would surface.

## Update: scaled-up run + trained predictors (same session, follow-up)

Scaled from 1 prompt/24 tokens to **8 prompts spanning 5 topic types
(quick/code/analysis/long/creative) x 200 tokens each = 1710 tokens, 82,080
trace rows**, with a genuine held-out test split (6 prompts train, 2 test --
`quick` and `analysis` topics held out, never touched during training) so
none of the numbers below are measured on their own training data.

### Raw predictability holds up at scale

- Avg t->t+1 overlap across all layers, all 82K pairs: **44.2%** (pilot: 49.6% on
  24 tokens) -- consistent, still ~7x the random baseline (6.2%).
- Decays gracefully with distance, staying well above random out to k=5:
  k=1: 44.2%, k=2: 37.2%, k=3: 35.1%, k=4: 33.8%, k=5: 32.8%.
- Holds across every topic tested: creative/long generation (~48-49%) shows
  the most redundancy, code the least (~39%) -- plausible, since code has
  more syntactic/structural token-to-token variety.

**This part of the premise is confirmed, not just suggested by a small pilot.**

### But: three different predictors, and none beats the trivial baseline

This is the part worth taking seriously before building anything bigger.
Evaluated on the 2 held-out test prompts (20,160 examples), recall@8 (of the
8 experts truly selected at t+1, how many did the predictor's top-8 contain):

| Predictor | Recall@8 (held-out) |
|---|---|
| **naive-repeat** (predict E\_{t+1} = E\_t, zero training) | **47.8%** |
| Markov, per-layer pairwise co-occurrence, properly normalized | 26.8% |
| MLP (128-dim multi-hot + layer embedding -> 2-layer net) | 35.5% |
| MLP with an explicit learnable residual/skip connection to the repeat prior | 41.3% |

Every trained model underperformed the untrained heuristic. The residual MLP
(which was given the repeat signal directly and only had to learn a
correction on top -- `skip_scale` converged to 3.63, confirming it leaned on
that signal) closed most of the gap but still landed 6.5 points below simply
predicting "same as last time."

**Read honestly, this says: the exploitable structure in this data is almost
entirely "experts recur as themselves across adjacent tokens" -- not richer
cross-expert transition patterns that a Markov table or a small MLP can
extract beyond that. Both attempts to model something more sophisticated
than the trivial heuristic made predictions *worse*, not better, at this
data scale.**

### What this means for the prefetch prototype

Good news for the "build something" question: **you don't need a trained
predictor at all.** The best-performing predictor found across three
approaches is the free one: a prefetch cache that keeps whichever experts
were resident for the *previous* token warm is already within a few points
of the best result any model here achieved, and beats all of them. That's
the shape of prototype the roadmap called for anyway (Track 2's "smallest
possible prototype") -- this result is a concrete argument for building
exactly that, not a fancier learned version, at least as a first cut.

### Honest caveats (at the time)

- 8 prompts, greedy decoding, one model. More/longer prompts, sampled
  (non-greedy) decoding, and other MoE models would all be worth checking
  before treating "trained predictors don't help" as settled.
- The MLP here is deliberately small and trained briefly (30 epochs, ~62K
  examples pooled across 48 non-iid layers sharing one set of weights). A
  per-layer model, more data, or more training could still close the gap --
  but the effort-to-benefit case for that is weaker now that the free
  baseline is this close to the ceiling.

## Update: v2 -- per-layer models beat naive-repeat (later session)

Acted on the exact caveat above instead of leaving it as a maybe. Two
changes, both aimed at named weaknesses in v1, not a bigger model:

1. **Per-layer models, not pooled.** v1 shared one set of weights across
   all 48 layers with just a layer-embedding hint. Per-layer overlap
   characteristics genuinely differ (layers 0/47: 17-22% overlap; the
   30s-band: 58-73%), so a shared model has to compromise across regimes
   that don't behave alike. v2 trains 48 small, independent models, one
   per layer (`train_predictor_v2.py`).
2. **More data.** Scaled `expert_trace.py`'s `PROMPTS` list from 8 to 25
   (5 per topic instead of 1-2), 255K trace rows total, so each of the 48
   per-layer models still gets several thousand training examples of its
   own. Held out one full prompt per topic (5 total, never touched in
   training) -- a larger, more representative test set than v1's 2.

**Result, recall@8 on held-out prompts, naive-repeat recomputed fresh on
this same data for a fair comparison:**

| Predictor | Recall@8 (held-out) |
|---|---|
| naive-repeat | 45.9% |
| Markov (normalized, same as v1's method) | 41.0% |
| **v2 per-layer MLP (1-step history)** | **50.6%** |
| **v2 per-layer MLP (2-step history)** | **51.0%** |

**First trained predictor in this line of work to beat naive-repeat.**
+5.0 points (51.0% vs 45.9%), and it beats the baseline on 43 of 48
individual layers, not just on average. 2-step history (also seeing
E_{t-2}) helps marginally over 1-step (51.0% vs 50.6%) -- most of the
signal is still in the immediately preceding step, consistent with the
overlap-decay numbers (45.8% at k=1 dropping to 34.9% at k=5).

The win is concentrated exactly where you'd predict from the pilot's
per-layer breakdown: layer 47 goes from 25.0% (naive-repeat) to 42.4%
(model), layer 0 from 10.6% to 26.4% -- the two layers where "just repeat
the last token's experts" was weakest to begin with, because those layers
don't recur as much. A per-layer model can learn each layer's actual
regime instead of a compromise; a small number of middle layers (e.g. 18,
30) see a slight decrease (~1-1.5 points) where naive-repeat was already
very strong (>60%) and there wasn't much room to add.

**What this changes for the prefetch prototype:** the honest v1
conclusion -- "you don't need a trained predictor, the free heuristic is
competitive" -- needs updating. A per-layer model provides a real,
measured lift in prediction accuracy, particularly at the layers where
the free heuristic was worst. `PREFETCH_FEASIBILITY.md`'s compute-upside
simulation used naive-repeat's 44.2%/47.8% hit rate as its predictor;
rerunning that simulation with v2's ~51% hit rate (higher, and more evenly
distributed across layers instead of concentrated in the already-easy
middle band) is a natural next step before finalizing a prefetch design.

All code artifacts are in this directory: `expert_trace.py` (collection,
`PROMPTS` list now 25 entries), `analyze_trace.py` (overlap, decay, Markov
baseline, per-topic held-out split), `train_predictor.py` /
`expert_predictor.pt` (v1, pooled model, kept for reference),
`train_predictor_v2.py` / `expert_predictor_v2.pt` (v2, per-layer models,
the current best). `expert_trace.jsonl` (255K rows) and `split.json` are
the underlying data.
