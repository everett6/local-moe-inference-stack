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

## Bottom line

- Predictability: confirmed at scale (44.2% overlap, ~7x random) —
  [EXPERT_TRACE_FINDINGS.md](EXPERT_TRACE_FINDINGS.md).
- Transfer feasibility: confirmed on real hardware — this document, first
  section. Real PCIe measurement, no instrumentation confound (the
  bandwidth test does no per-node Python callback at all). **Not a
  blocker.**
- Compute upside: **still unmeasured**, and now for a well-diagnosed reason
  rather than an untried one. Two independent proxy methods both failed,
  for two different, understood reasons (dispatch-dominated batch-1 matmul;
  callback-count-proportional wall time). Neither failure casts doubt on
  the transfer-feasibility result above — that number came from direct
  hardware measurement, not from either contaminated method.

The roadmap's stated gate ("if that holds up, then the small prefetch
prototype... is justified") has been checked with real data on the
predictability side (yes) and the transfer-feasibility side (yes, with
headroom to spare — ~1.8x). The compute-upside question — how much wall
time a working prototype would actually save, if any — cannot be answered
without either patching llama.cpp's C++ or building the prototype and
measuring it directly. That's a genuine scope escalation from everything
done so far this session (Python + ctypes hooks against a stock binary, no
recompilation) to modifying and rebuilding a large C++/CUDA codebase — the
right next checkpoint to confirm before starting it.
