# AI2 Local MoE Stack — Executive Summary

## What this session did

Took the AI2 codebase from "won't run" to a working, fully-local (no LM Studio)
speculative-decoding stack with real benchmarking and an actual auto-refreshing
self-improvement loop, then diagnosed why the self-improvement/speculative path
underperforms and researched what would fix that properly.

## Bugs found and fixed

1. **`llama-server` deadlocked on the first real request.** `BigModelServer`
   launched it with `stdout=subprocess.PIPE` and never drained the pipe. Once
   request logging filled the 64KB OS pipe buffer, the server's `write()` call
   blocked and the entire process hung — even a fresh 1-token request would
   time out with no error. Fixed by redirecting to a real log file
   (`state/llama_server.log`). This one was silent and would have made the
   whole stack look "randomly frozen" under any real use.

2. **Verification logic was fundamentally broken — 0% draft-accept rate,
   zero training signal, ever.** `logits_for_tokens()` called `/completion`
   with `n_predict: 0` expecting per-position teacher-forced probabilities
   for the whole prompt. Empirically verified this llama-server build
   returns **no `completion_probabilities` field at all** when `n_predict: 0`
   (undocumented behavior, not just a naming drift). Every draft token was
   therefore silently treated as a mismatch, and — because the fallback path
   also required a non-null `top_tok_str` — mismatches never even reached
   `on_mismatch()`, so `draft_trainer.py` never got a single training example
   despite the app actively generating text.

   **Fix:** replaced with `greedy_tokens_with_ids()`, which has the big model
   greedily generate the same number of tokens the draft proposed, using the
   generated token **IDs** (not re-tokenized text) as ground truth. For
   greedy/temperature-0 decoding this is exactly correct — the big model's own
   free n-token greedy run from the same context *is* the reference sequence.
   Verified after the fix: 66–75% draft accept rate on real prompts, coherent
   text, real mismatches now flowing to the trainer.

3. **Gradio 6 incompatibility.** `gr.Chatbot(type="messages")` — the
   installed version (6.27.0) removed `type` entirely; messages format is now
   the only format. One-line fix.

4. **Shutdown race in `OnlineDraftTrainer.stop()`.** It set the stop flag and
   immediately called `save_adapter()` without joining the background thread.
   If shutdown landed mid-refresh-cycle, two threads could call
   `model.save_pretrained()` into the same adapter directory concurrently,
   risking a corrupted checkpoint. Fixed with `self._thread.join(timeout=30)`
   before the final save.

## What was missing and got built

- Wired `config.py` from placeholder paths to your actual files (30B model
  under LM Studio's directory, the 0.5B draft GGUF under `Model Training/`).
- `llama-server`/`llama-quantize` binaries: reused LM Studio's bundled CUDA12
  backend (needed `LD_LIBRARY_PATH` pointed at its vendored
  `libcudart.so.12`/`libcublas.so.12` — not on the system loader path).
- Downloaded the fp16 HF copy of Qwen2.5-Coder-0.5B-Instruct (draft_trainer.py
  needs a trainable copy separate from the quantized inference copy).
- Vendored `convert_hf_to_gguf.py` + its `conversion`/`gguf-py` support
  packages from the llama.cpp repo (not shipped by LM Studio, which only
  ships compiled binaries) — needed to turn a trained draft model back into
  GGUF.
- **Automated the refresh cycle end to end.** The original README described
  this as a fully manual, one-off procedure. `draft_trainer.py` now runs it
  automatically every `refresh_every_n_steps` training steps: merge LoRA →
  requantize → hot-swap the live engine's draft model (`LocalMoEEngine.reload_draft`),
  with old/new instance swapped atomically so in-flight generation never sees
  a gap. Trainer panel now shows refresh count / last refresh time / currently
  serving path / last refresh error.
- **Real benchmark UI**: speculative decoding ON vs OFF, same model/server/prompts,
  tok/s + per-prompt draft accept rate + speedup ratio, with an actual varied
  prompt set (code, math, summarization, bug-fix) instead of a placeholder.

## Does it match your CUDA-graph / PCIe-prefetch diagram? No.

Confirmed directly: static `-ot` tensor split (experts pinned to CPU RAM once
at load), llama.cpp's own internal top-k gating, and a hand-rolled Python
speculative loop are what's running. None of it is CUDA-graph-traced, there is
no double-buffered ping-pong VRAM cache, and there is no per-token predictive
PCIe prefetch of individual experts. This matches the project's own README,
which already flags that architecture as requiring patches to llama.cpp's C++
internals — not reachable from a Python wrapper. I did not fake
instrumentation to make it look otherwise (unlike `/home/everett/AI/ShadowRouter.py`,
found earlier this session, whose UI claims "predictive MoE warmup" while the
code just writes a hardcoded confidence number to a JSON file).

## The real finding: speculative decoding is currently ~3x *slower*, not faster

Benchmarked on real prompts after the correctness fix:

| Mode | Throughput |
|---|---|
| Speculative decoding ON (fixed) | ~14.8 tok/s |
| Speculative decoding OFF (big model alone) | ~44.5 tok/s |
| **Speedup** | **0.33x** |

Root cause, verified empirically: this llama-server build's HTTP `/completion`
API has **no teacher-forced, zero-generation scoring mode**. Setting
`n_predict: 0` doesn't score a hypothetical continuation — it just silently
generates anyway. That means "verifying" k draft tokens costs exactly as much
big-model compute as generating k tokens directly would, and speculative mode
pays draft-model compute and extra HTTP round-trips *on top of* that with no
compute saved. This isn't a tuning problem — it's a capability gap in the
server's exposed API. (This matches a known, longstanding issue in the
llama.cpp/llama-cpp-python ecosystem: [`echo=True, max_tokens=0` logprob
scoring is unreliable/broken across versions](https://github.com/abetlen/llama-cpp-python/issues/349).)

**A real fix exists and was verified working**: this exact llama-server binary
has **native speculative decoding built into the C++ server** —
`--spec-draft-model`, `--spec-type draft-simple`, `--spec-draft-n-max`. Tested
directly: healthy, coherent output, no HTTP round-trip per draft token (real
batched verification in C++). Tradeoff: its HTTP responses don't expose
per-token accept/reject, so it can't directly feed the online training loop
the way the hand-rolled path does. **Recommendation:** use native mode for
serving speed, and keep the current hand-rolled path running periodically in
the background purely to harvest training signal (see below).

## Literature review — concrete improvement paths

- **[EAGLE / EAGLE-2](https://arxiv.org/abs/2401.15077)** ([EAGLE-2](https://arxiv.org/html/2406.16858v1)):
  drafts at the second-to-top-layer *feature* level (not token level) plus
  dynamic draft trees; 3–4.3x lossless speedup, current state of the art for
  draft quality. Requires access to the big model's hidden states directly —
  not reachable through a black-box HTTP completion API, only via in-process
  weights (llama-cpp-python low-level API or a patched server).
- **[Medusa](https://arxiv.org/abs/2401.10774)**: adds extra decoding heads
  *on the big model itself* instead of maintaining a separate draft model —
  eliminates this project's entire "two models, two vocabs, adapter drift,
  manual requantize cycle" problem by construction. 2.2–3.6x speedup. Would
  mean training heads on the Qwen3-30B-A3B model, not the 0.5B model.
- **[Online Speculative Decoding](https://arxiv.org/abs/2310.07177)**:
  directly validates this project's core idea (continuously adapt the draft
  model to the observed query distribution instead of a static draft model).
  Its method trains on distributional signal (distillation loss against the
  big model's logits), not a single hard-label per mismatch —
  suggests upgrading `draft_trainer.py`'s loss once/if per-position logits
  become available (e.g., from native speculative mode's internals, or by
  running the big model in-process).
- **[OmniDraft](https://arxiv.org/abs/2507.02659)**: cross-vocabulary online
  adaptive drafting — relevant if you ever want a draft model whose tokenizer
  doesn't match the big model's (currently they happen to match exactly,
  verified this session — that's why the ID-based comparison works at all).
- **[Fiddler](https://arxiv.org/abs/2402.07033)** (ICLR'25) /
  **[MoE-Infinity](https://arxiv.org/abs/2401.14361)**: the actual prior art
  behind your architecture diagram. Fiddler computes experts *on* the CPU
  rather than shuttling weights over PCIe (avoids the transfer entirely for
  single-batch inference); MoE-Infinity uses a sparsity-aware expert *cache*
  with access-pattern-aware prefetching — much closer to your diagram's
  intent than what exists today, but both are multi-month C++ systems
  projects, not configuration changes.

## Recommended next steps, in order of effort

1. **Cheap, do now:** switch live serving to native `--spec-type draft-simple`
   (already verified healthy on this hardware) for actual speed, keep the
   current hand-rolled path only as an offline/background "training data
   harvester" that periodically samples recent prompts.
2. **Medium:** change `draft_trainer.py`'s loss from single-token
   cross-entropy to a KL-divergence-to-big-model-logits objective once a
   logit source is available — directly informed by the Online Speculative
   Decoding paper above.
3. **Large, real research project:** move toward Medusa-style extra heads on
   the big model, which sidesteps the two-model architecture's structural
   problems entirely. Not a weekend change.

## Update: architecture pivot after the benchmark (this is the current state)

The 3-way benchmark (full numbers in `BENCHMARK_RESULTS.md`) confirmed
speculative decoding isn't a productive lever on this hardware at all --
hand-rolled: 0.25x (4x slower), native C++ speculative: 1.02x (parity, not a
win). Chasing it further wasn't worth it, so the app was rearchitected around
what the benchmark data actually supports:

- **`router.py`'s bucket classifier is now actually wired up** (it existed
  before but was only ever displayed, never used to route). "Quick" bucket
  prompts (short factual questions) are answered **entirely by the draft
  model alone** -- fully GPU-resident, not subject to the big model's
  CPU-bound MoE bottleneck, so its own speed is finally worth something.
  Everything else uses `generate_baseline()` (plain big model, no
  speculative overhead -- the fastest *correct* option per the benchmark).
- **Always-verify-async**: the draft's fast answer is shown to the user
  immediately, then checked against the big model in the background within
  the same request (Gradio generator with two yields). If they disagree, the
  displayed answer is corrected in place and the mismatch is queued for
  training -- this is where the self-improvement loop now earns its keep: on
  the fast path's *accuracy*, not on speeding up the big model.
- Two more real bugs found and fixed during this work:
  1. The hand-rolled speculative loop never checked for the model's stop
     token (`<|im_end|>` / `<|endoftext|>`), so it always ran to the full
     token ceiling instead of stopping naturally -- fixed, and this also
     removed an unfair handicap from the original ON/OFF benchmark.
  2. `complete_greedy()` / `greedy_tokens_with_ids()` used temperature=0 with
     **no repetition penalty**, which degenerated into runaway repetition
     loops on some prompts (caught live: a wrong "456" answer for 17x24 got
     "corrected" into a repeating garbage loop before this fix). Fixed with
     `repeat_penalty: 1.1` on both call sites.
- **Verified live, end to end**: asked "What is 17 times 24?" -- draft
  answered instantly (456, wrong), background verification caught the
  disagreement at the very first token, and corrected the displayed answer to
  a coherent, correct derivation (408) while queuing the mismatch for
  training. This is the whole redesigned pipeline working as intended.
- Thread count and flash-attention tuning (see `BENCHMARK_RESULTS.md`) are
  still in effect: 16 threads, flash-attn forced on.

### What's still open

- The "quick" bucket's classifier (`router.py`) is a keyword/length
  heuristic, not itself learned -- it decides what counts as "answerable by
  the draft model alone" by word count and a fixed keyword list, not by
  measured draft accuracy per topic. Worth revisiting once there's enough
  fast-path history to see which topics the draft model is actually reliable
  on vs not.
- Async verification currently costs one background big-model call per quick
  query regardless of outcome -- fine at low volume, but if quick-path
  traffic gets heavy this is where a confidence-gating or sampling strategy
  (discussed and explicitly not chosen this round) would trade off training
  signal for lower background load.
- No new full benchmark run was done after the repeat-penalty fix (this
  session prioritized not overflowing context); worth a quick baseline
  re-check if output quality on longer generations becomes a concern.

## Current running state

`app.py` is running locally at `http://127.0.0.1:7860` with the big model
server on port 8090, both healthy, running the fast-path + async-verify
architecture described above.
