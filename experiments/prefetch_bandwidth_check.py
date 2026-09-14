"""
Track 2, step 2 of the roadmap: before writing any staging-buffer/CUDA code,
check whether there's actually enough PCIe bandwidth on this machine to hide
expert-weight prefetch transfers behind per-token compute. If the transfer
itself is already the bottleneck, a prefetcher can't help no matter how
clever the predictor is.

Real numbers used here, not estimates:
- Expert tensor sizes: read directly from the GGUF file's tensor table
  (blk.0.ffn_{gate,up,down}_exps.weight), not computed from hparams, so
  quantization (Q4_K_M/Q6_K mix) is reflected exactly as stored on disk.
- Overlap rate (44.2%): the measured t->t+1 expert-overlap result from
  analyze_trace.py's scaled run (82K trace rows, 8 prompts) -- i.e. on
  average 55.8% of a layer's 8 experts wculdn't have been in the previous
  token's set and would need fetching under a "keep last token's experts
  resident" prefetch scheme.
- Per-token compute budget (21.6ms @ 46.3 tok/s): the measured hybrid
  GPU+CPU baseline from BENCHMARK_RESULTS.md, not a theoretical figure.
- H2D bandwidth: measured directly on this machine below, not a spec-sheet
  number -- both a realistic case (many ~3MB async copies, matching one
  expert's weight size) and a best-case reference (one large contiguous
  512MB copy).
"""
import torch
import time

torch.set_num_threads(16)

BYTES_PER_EXPERT = 3_059_712  # blk.0 ffn_{gate,up,down}_exps.weight / 128, from GGUF tensor table
N_LAYERS = 48
OVERLAP = 0.442  # analyze_trace.py, scaled run, all-layers avg t->t+1 overlap
BASELINE_TOK_S = 46.3  # BENCHMARK_RESULTS.md, hybrid GPU+CPU, spec OFF


def measure_bandwidth():
    assert torch.cuda.is_available()
    print("GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))

    buf = torch.empty(BYTES_PER_EXPERT, dtype=torch.uint8, pin_memory=True)
    gpu_buf = torch.empty(BYTES_PER_EXPERT, dtype=torch.uint8, device="cuda")
    stream = torch.cuda.Stream()

    N = 500
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.cuda.stream(stream):
        for _ in range(N):
            gpu_buf.copy_(buf, non_blocking=True)
    torch.cuda.synchronize()
    per_copy = (time.time() - t0) / N
    bw_small = BYTES_PER_EXPERT / per_copy

    big = 512 * 1024 * 1024
    bigbuf = torch.empty(big, dtype=torch.uint8, pin_memory=True)
    biggpu = torch.empty(big, dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()
    t0 = time.time()
    biggpu.copy_(bigbuf, non_blocking=True)
    torch.cuda.synchronize()
    bw_peak = big / (time.time() - t0)

    print(f"measured H2D BW: {bw_small/1e9:.2f} GB/s (many {BYTES_PER_EXPERT/1e6:.2f}MB async copies), "
          f"{bw_peak/1e9:.2f} GB/s (one 512MB contiguous copy, reference ceiling)")
    return bw_small, bw_peak


def project(bw_small, bw_peak):
    new_experts_per_layer = 8 * (1 - OVERLAP)
    new_experts_per_token = new_experts_per_layer * N_LAYERS
    total_bytes_per_token = new_experts_per_token * BYTES_PER_EXPERT

    t_token_small = total_bytes_per_token / bw_small
    t_token_peak = total_bytes_per_token / bw_peak
    t_layer_small = (new_experts_per_layer * BYTES_PER_EXPERT) / bw_small

    budget_ms = 1000 / BASELINE_TOK_S
    per_layer_budget_ms = budget_ms / N_LAYERS

    print()
    print(f"new experts/layer needing fetch (avg): {new_experts_per_layer:.2f} of 8")
    print(f"new experts/token across {N_LAYERS} layers: {new_experts_per_token:.1f}")
    print(f"bytes to transfer/token: {total_bytes_per_token/1e6:.1f} MB")
    print(f"projected H2D time/token: {t_token_small*1000:.2f} ms (measured BW) / {t_token_peak*1000:.2f} ms (peak BW)")
    print(f"projected H2D time/layer: {t_layer_small*1000:.3f} ms")
    print()
    print(f"current measured pipeline: {budget_ms:.2f} ms/token total, ~{per_layer_budget_ms:.3f} ms/layer (naive even split)")
    fits = t_layer_small * 1000 < per_layer_budget_ms
    print(f"per-layer transfer ({t_layer_small*1000:.3f} ms) vs per-layer budget ({per_layer_budget_ms:.3f} ms): "
          f"{'fits within' if fits else 'EXCEEDS'} a single layer's slice of existing compute time")
    return fits


if __name__ == "__main__":
    bw_small, bw_peak = measure_bandwidth()
    project(bw_small, bw_peak)
