# Prefetch feasibility check: is there PCIe latency worth hiding?

This is the second gate from the roadmap, run immediately after
[EXPERT_TRACE_FINDINGS.md](EXPERT_TRACE_FINDINGS.md) confirmed expert
selection is predictable enough to be worth exploiting. Before writing any
staging-buffer/CUDA code: if transferring the predicted experts from CPU RAM
to GPU VRAM takes longer than the compute it's supposed to hide behind, a
prefetcher can't win regardless of how good the predictor is. This check
answers that with real numbers measured on this machine, not spec-sheet
estimates.

## Method

Script: [`prefetch_bandwidth_check.py`](prefetch_bandwidth_check.py).

- **Expert size**: read directly from the GGUF tensor table (not computed
  from hyperparameters), so real quantization is reflected exactly:
  `blk.0.ffn_{gate,up,down}_exps.weight` total 391.6 MB for 128 experts →
  **2.92 MiB per expert** (gate+up+down combined).
- **Fetch volume**: using the measured 44.2% t→t+1 overlap from
  `analyze_trace.py`'s scaled run, a "keep last token's experts resident"
  prefetch scheme needs to fetch the other 55.8% fresh → **4.46 of 8 experts
  per layer, 214.3 across all 48 layers, 655.6 MB per token** on average.
- **Bandwidth**: measured directly on this RTX 5070 via pinned-memory async
  `cudaMemcpy`-equivalent (`torch.Tensor.copy_(..., non_blocking=True)` on a
  dedicated stream) — both the realistic case (500 back-to-back ~2.92MB
  copies, matching one expert's actual size) and a best-case reference (one
  512MB contiguous copy).
- **Compute budget**: the real measured hybrid GPU+CPU baseline from
  `BENCHMARK_RESULTS.md` — 46.3 tok/s = 21.6 ms/token, spec OFF — divided
  evenly across 48 layers as a first-order per-layer budget (0.45 ms/layer).

## Result

```
measured H2D BW: 53.66 GB/s (many 2.92MB async copies), 57.52 GB/s (one 512MB contiguous copy)

new experts/layer needing fetch (avg): 4.46 of 8
new experts/token across 48 layers: 214.3
bytes to transfer/token: 655.6 MB
projected H2D time/token: 12.22 ms (measured BW) / 11.40 ms (peak BW)
projected H2D time/layer: 0.255 ms

current measured pipeline: 21.60 ms/token total, ~0.450 ms/layer (naive even split)
per-layer transfer (0.255 ms) vs per-layer budget (0.450 ms): fits within a single layer's slice of existing compute time
```

**The transfer is not the bottleneck.** At this machine's real measured PCIe
throughput (53.7 GB/s sustained even for many small ~3MB transfers, not just
one big one — this GPU/slot has no small-transfer penalty worth worrying
about), fetching a full token's worth of "new" experts costs ~12.2ms in
bulk, or ~0.255ms if pipelined one layer ahead — comfortably inside the
~0.45ms average per-layer slice of the existing 21.6ms/token budget. A
prefetch pipeline that fetches layer L's *predicted* experts (using the
previous token's layer-L selection) while the current token is still
computing layers before L has roughly 1.8x the transfer time it needs, as
slack.

This is a green light on the specific question asked: **there is enough
compute time per token to hide the required transfer, given the predictor
this data supports (naive-repeat / keep-warm).**

## What this does NOT tell you (two honest gaps, both chased down)

The remaining open question is the *upside*: if expert compute moved from
CPU to GPU, how much faster would it actually be? Two attempts to measure
this, both inconclusive, both instructive about why:

**Attempt 1 — isolated PyTorch matmul proxy.** Same shape as one expert's
gate/up/down (2048→768→2048), batch=1, fp32. Came back at ~0.257ms/expert →
~98.8ms/token extrapolated across all 384 expert evaluations (8×48) — nearly
5x the entire real measured pipeline (21.6ms, attention + experts +
everything). Not believable: llama.cpp's CPU kernels use quantized SIMD dot
products and batch the 8 active experts together per layer; a lone batch-1
PyTorch fp32 matmul is dominated by per-call dispatch overhead, not real
FLOPs. Discarded.

**Attempt 2 — real `cb_eval` timestamps around llama.cpp's own MoE ops**
(`moe_compute_timing.py`, same hook `expert_trace.py` already had working,
bracketing `ffn_norm-{il}` → `ffn_moe_out-{il}` per layer, 100 real decode
steps, CPU-only, 16 threads to match production). Result: **52.6ms/token**
average MoE-block time, 1.10ms/layer — again *higher* than the entire
hybrid pipeline's 21.6ms/token budget, which is the same red flag as
attempt 1. Before discarding it, checked *why*: added a call counter, and
**50.7% of all `cb_eval` invocations happen inside a layer's MoE window,
which matches almost exactly the 51.5% share of wall-clock time it
accounts for.** That proportionality is the tell — real compute time
depends on FLOPs and matrix shape, not on how many graph nodes the
scheduler happens to walk through. What's actually being measured is
Python/ctypes callback dispatch overhead (the MoE branch has far more graph
nodes per layer than attention — topk, per-expert gather, per-expert
matmul, weighted sum — so it racks up proportionally more `cb_eval` calls),
not ggml's real compute time. Discarded too, for a cleanly diagnosed reason
rather than a hunch.

**Conclusion: this repo's Python-hook instrumentation approach (great for
*what* experts get selected, which is discrete data attached to few
easily-identified tensors) does not have the resolution to cleanly time
*how long* compute takes, because the instrumentation's own per-call
overhead scales with the same thing it's trying to measure (graph node
count). Getting a clean compute-upside number needs either a native
`std::chrono` timer patched directly into llama.cpp's C++ (a small, targeted
change, but real compilation — a different category of effort than
everything done so far this session) or just building the prototype and
measuring real end-to-end tok/s, which sidesteps the decomposition problem
entirely.**

## Attempt 3 -- native C++ timer, patched directly into llama.cpp

Added a small instrumented example, `llama.cpp-src/examples/moe-timing/`
(not part of this repo -- lives in the separate, gitignored llama.cpp
checkout used for reference reading). Same bracketing idea as attempt 2
(`ffn_norm-{il}` -> `ffn_moe_out-{il}`), but the timer is C++ code compiled
directly into the binary, called from the real scheduler loop -- no
ctypes, no Python, no per-call marshalling.

Built CPU-only (`-DGGML_CUDA=OFF`), 16 threads to match production. First
result: **47.6ms/token** average MoE-block time -- almost identical to the
(supposedly contaminated) Python measurement from attempt 2. That was the
first sign something else was going on: removing the suspected
contamination barely moved the number.

**Checked against the real thing.** Launched the actual production
binary (LM Studio's bundled `llama-server`, the one `local_engine.py`
actually runs) with its real flags and hit `/completion` directly --
no proxy, no instrumentation, just the numbers the app itself reports:

| Config (same binary, same model, same prompt) | measured tok/s | ms/token |
|---|---|---|
| Hybrid: `-ngl 999 -ot "ffn_(gate\|down\|up)_exps=CPU"` (production) | 45.4 | 22.0 |
| Full CPU: `-ngl 0` (same binary, everything on CPU incl. experts) | 29.1 | 34.4 |

Both numbers reproduce `BENCHMARK_RESULTS.md`'s original 46.3 tok/s
baseline closely -- that number was real and still holds up today. And
**34.4ms is the *entire* per-token cost of the real production binary
computing everything (attention + all 48 layers of experts) on CPU** --
which is already less than my custom build's isolated *MoE-only* claim of
47.6ms. My build was measuring something slower than the real thing, not
a lower bound on it.

**Ruled out version skew.** LM Studio's binary reports `commit 8172e65`
(via `--version`) -- I fetched that exact commit from upstream, checked it
out, and rebuilt. Still 44.6ms/token for MoE alone. Ruled out ISA target
too: LM Studio's build targets baseline `avx2`, mine defaulted to
`-march=native` -- rebuilt a third time with `-DGGML_NATIVE=OFF
-DGGML_AVX2=ON` to match exactly. Still ~45ms. Exact same source commit,
matching declared instruction set, and my build is still slower than the
production binary's *entire token* by a wide margin.

**What's left unmatched, and why I stopped there:** LM Studio's binary
reports `built with GNU 12.3.0`; this machine's default compiler is GCC
15.2.0. A ~3-major-version compiler gap is a real, plausible source of
this gap (auto-vectorization of ggml's hand-tuned quantized-matmul
intrinsics is exactly the kind of code where GCC versions diverge in
practice). Installing a matching GCC 12 toolchain to test that would need
`sudo` (blocked the same way `gh`/`apt` were earlier in this session).
Separately, real profiling of the live production binary via `perf` --
which would have settled this cleanly regardless of build differences --
is blocked by this system's kernel policy (`perf_event_paranoid=4`), which
also needs `sudo` to lower. Both remaining paths are real, identified, and
both gated on the same permission this session doesn't have unprompted.

## Bottom line

- Predictability: confirmed at scale (44.2% overlap, ~7x random) —
  [EXPERT_TRACE_FINDINGS.md](EXPERT_TRACE_FINDINGS.md).
- Transfer feasibility: confirmed on real hardware — this document, first
  section. Real PCIe measurement, no instrumentation confound (the
  bandwidth test does no per-node Python callback at all). **Not a
  blocker.**
- Compute upside: not cleanly isolated, but usefully *bounded* by a real,
  reproducible, same-binary measurement: moving attention (and everything
  except CPU-pinned experts) from CPU to GPU already buys the production
  pipeline a confirmed **36% speedup** (34.4ms full-CPU -> 22.0ms hybrid,
  live, same model, same prompt, same binary, measured directly against
  `local_engine.py`'s actual production flags). That gap is real evidence
  the core premise -- moving compute off the CPU helps on this hardware --
  holds. What it doesn't tell us: how much of the *remaining* 22.0ms hybrid
  budget is CPU-expert-compute versus GPU-attention, which is the number
  that would tell you the prefetch prototype's true ceiling. Three
  independent attempts to isolate that fraction (Python callback timing,
  native C++ timing at HEAD, native C++ timing at the exact matched commit
  + ISA target) all produced numbers inconsistent with the real binary's
  own total time -- see "Attempt 3" above. The remaining candidate causes
  (compiler version mismatch, `perf` profiling) both need a `sudo`-gated
  install this session doesn't have standing permission for.

**Practical read:** the roadmap's stated gate has real data on two of three
legs -- predictable (yes) and transfer-affordable (yes, with margin) -- and
a directionally strong but not exactly-quantified signal on the third
(CPU-to-GPU moves already measurably help, by 36%, on this hardware).
Getting an exact "prefetch would save you N ms/token" number from here
needs either `sudo` (to match GCC 12, or lower `perf_event_paranoid` for
real profiling) or just building the prototype and measuring real
end-to-end tok/s directly, which sidesteps needing the decomposed number
at all. That's a genuine scope escalation from everything done so far this
session (Python/ctypes hooks and now a local instrumented rebuild, both
against a stock binary or reference source, no changes to the actual
production pipeline) to modifying and rebuilding llama.cpp's CUDA backend
itself -- the right next checkpoint to confirm before starting it.

## Phase 2 -- data-driven simulation (no sudo needed)

Rather than chase the exact number through a `sudo`-gated compiler match or
`perf` session, built a simulation instead: `prefetch_simulation.py`, using
real measurements throughout except for one missing piece, derived rather
than proxied.

**Real, directly measured:**
- GPU per-expert matmul time (gate/up/down, real shape): **0.0217 ms**,
  via PyTorch CUDA -- a trustworthy proxy here, unlike the earlier CPU
  attempts, because PyTorch's CUDA matmul already uses near-optimal cuBLAS
  kernels for this shape.
- GPU per-layer attention-projection time (Q/K/V/O, real shapes: 32 heads,
  4 KV heads, head dim 128, GQA): **0.0851 ms**. RoPE/softmax/KV-read are
  not modeled -- negligible at batch=1 next to these four projections.
- 48-layer GPU attention total: **4.09 ms**, i.e. an estimated **18.6%**
  of the measured 22.0ms hybrid budget.

**Derived, not measured:** subtracting that attention estimate from the
real measured 22.0ms hybrid total leaves 17.91ms attributed to CPU expert
compute, i.e. **0.0467 ms/expert**. This is inference from one trusted real
number (22.0ms, live-measured against the actual production binary), not
a proxy benchmark -- it sidesteps the exact failure mode that sank the
three earlier attempts (all of which tried to measure this directly and
got numbers inconsistent with reality). It depends on the attention
estimate being roughly right; flagged here as the one number in this
simulation that isn't a hard measurement.

Interesting on its own: 0.0467ms CPU vs 0.0217ms GPU is only a ~2.2x gap,
not the 10-50x folklore figure for GPU-vs-CPU matmul. That's expected at
batch=1 -- a single-token matmul this small is dominated by per-call
overhead on *both* sides, not raw throughput, so the GPU's parallelism
advantage is largely unrealized here. It's also *why* llama.cpp's CPU
K-quant kernels are competitive enough to make CPU-offloaded MoE viable on
consumer hardware in the first place.

**Simulation**, walking all 82K real trace rows (8 prompts): for every
real token-to-token step, naive-repeat predicts next-token's experts as
this token's experts (47.8%-recall@8 heuristic, the best one found). Hits
get GPU compute; misses get whichever is cheaper, per real numbers: fall
back to CPU (today's behavior), or pay a fresh synchronous transfer + GPU
compute. Given the real numbers above, **CPU-fallback always wins**
(0.0467ms < 0.057ms transfer alone, before even adding GPU compute on
top) -- a fresh transfer is never worth it for a miss on this hardware.
That's a genuine, useful architectural finding: the real prototype doesn't
need an on-demand transfer path for cache misses at all. It only needs to
route *correctly-predicted* experts to a speculatively-prefetched GPU
cache (populated during the prior token's slack, already shown affordable
in Phase 1) and leave everything else on CPU exactly as today -- simpler
than the staging-buffer design originally sketched in the roadmap.

**Result:**

| | ms/token | tok/s |
|---|---|---|
| Measured hybrid (real, today) | 22.0 | 45.5 |
| Simulated hybrid, no-prefetch control | 22.0 (by construction) | 45.5 |
| **Simulated, prefetch-enabled** | **14.56** | **68.7** |

**+33.8% estimated speedup.** The no-prefetch control reproducing 22.0ms
exactly is expected, not a validation -- the CPU-per-expert number was
solved to make it so. The real signal is the *prefetch* case: it's built
from the same trusted inputs (real trace, real GPU measurement, real hit
rate) with no free parameters tuned to hit a target.

## Bottom line (updated)

All three roadmap gates now have real, data-driven answers:
- **Predictability**: confirmed at scale (44.2% overlap, ~7x random).
- **Transfer feasibility**: confirmed (~1.8x PCIe headroom).
- **Compute upside**: estimated at **+33.8%** (45.5 -> 68.7 tok/s) via a
  simulation built from real measurements plus one clearly-flagged derived
  number, after three direct-measurement attempts were individually
  diagnosed and ruled non-representative.

**This is a go signal for the real prototype**, with two honest caveats
carried forward: the CPU-per-expert figure is derived, not measured
(getting a measured one still requires the `sudo`-gated compiler match or
`perf` access from Phase 0/Attempt 3), and the simulation assumes ideal
one-layer-ahead pipelining with no cache-management or misprediction
overhead. The real prototype -- per the original roadmap's Phase 3 --
means implementing dynamic per-token, per-expert GPU/CPU dispatch inside
llama.cpp's MoE forward pass, which needs `nvcc` (`sudo`) and is a
multi-day C++/CUDA effort in a codebase this project doesn't own, best
done as a private, non-upstream fork. Worth doing next given this result,
but a distinct, larger commitment to confirm explicitly before starting.
