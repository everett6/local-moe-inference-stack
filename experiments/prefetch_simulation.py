"""
Data-driven simulation of a GPU expert-prefetch cache, built entirely from
real measurements already collected this session -- no llama.cpp rebuild,
no sudo, no new permissions. Answers: is the invasive C++/CUDA engineering
effort (a real dynamic per-token GPU/CPU expert dispatch inside llama.cpp's
MoE forward pass) likely to pay off, before spending days building it?

Real inputs used as-is (nothing here is a guess):
  - expert_trace.jsonl: 82K real per-token, per-layer expert selections
    from 8 diverse prompts (expert_trace.py)
  - naive-repeat predictor: 47.8% recall@8 on held-out prompts, the best
    of everything tried (Markov, MLP, residual MLP) per
    EXPERT_TRACE_FINDINGS.md -- used here as the prefetch heuristic
  - PCIe H2D bandwidth: 53.7 GB/s, measured on this GPU
    (prefetch_bandwidth_check.py)
  - expert weight size: 2.92 MiB/expert, from the GGUF tensor table
  - production reference points, measured live against the actual
    llama-server binary with production flags: 34.4ms/token full-CPU,
    22.0ms/token hybrid (PREFETCH_FEASIBILITY.md, "Attempt 3")

One new measurement, done here: GPU matmul time for one expert's
gate/up/down and one layer's attention projections, via PyTorch CUDA. This
is a trustworthy proxy for GPU compute (unlike the earlier CPU proxy
attempts in PREFETCH_FEASIBILITY.md, which failed specifically because
hand-tuned CPU SIMD quantized kernels beat naive fp32 loops by a wide
margin) -- PyTorch's CUDA matmul already uses near-optimal cuBLAS kernels,
so a batch-1 GPU matmul measured this way is representative.

What we don't have: a direct, trustworthy measurement of real per-expert
CPU compute time (three attempts, all inconsistent with the real binary --
see PREFETCH_FEASIBILITY.md). Worked around here by inference instead of
proxy: measure GPU attention compute directly (trustworthy), then back out
the implied CPU-expert-compute budget from the one real number we do
trust -- the measured 22.0ms/token hybrid total. This makes the CPU-side
number derived, not measured -- flagged clearly in the output, and it makes
the "hybrid, no prefetch" control case reproduce 22.0ms by construction,
which is a check on the arithmetic, not independent validation.
"""
import json
import time
from collections import defaultdict

import torch

TRACE_PATH = "/home/everett/AI2/experiments/expert_trace.jsonl"

N_LAYERS = 48
N_EXPERT_USED = 8
BYTES_PER_EXPERT = 3_059_712  # gate+up+down, from GGUF tensor table (PREFETCH_FEASIBILITY.md)
PCIE_BW_BYTES_S = 53.66e9     # measured, many ~3MB async H2D copies (prefetch_bandwidth_check.py)

N_EMBD = 2048
N_FF_EXP = 768
N_HEAD = 32
N_HEAD_KV = 4
N_EMBD_HEAD = 128  # key_length == value_length

HYBRID_MS_PER_TOKEN = 22.0   # measured live, production binary, -ngl 999 -ot ffn_*_exps=CPU
FULL_CPU_MS_PER_TOKEN = 34.4  # measured live, production binary, -ngl 0


def bench_cuda(fn, n_warmup=20, n_iter=500):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter


def measure_gpu_expert_time():
    x = torch.randn(1, N_EMBD, device="cuda")
    gate_w = torch.randn(N_FF_EXP, N_EMBD, device="cuda")
    up_w = torch.randn(N_FF_EXP, N_EMBD, device="cuda")
    down_w = torch.randn(N_EMBD, N_FF_EXP, device="cuda")

    def one_expert():
        g = x @ gate_w.T
        u = x @ up_w.T
        h = torch.nn.functional.silu(g) * u
        h @ down_w.T

    return bench_cuda(one_expert)


def measure_gpu_attention_time():
    # RoPE/softmax/KV-read are not modeled -- negligible at batch=1, short context,
    # next to these four projections. o_in stands in for the attn-output (softmax(QK^T)V,
    # concatenated across heads); its own cost isn't part of what's being measured here.
    x = torch.randn(1, N_EMBD, device="cuda")
    q_w = torch.randn(N_HEAD * N_EMBD_HEAD, N_EMBD, device="cuda")
    k_w = torch.randn(N_HEAD_KV * N_EMBD_HEAD, N_EMBD, device="cuda")
    v_w = torch.randn(N_HEAD_KV * N_EMBD_HEAD, N_EMBD, device="cuda")
    o_w = torch.randn(N_EMBD, N_HEAD * N_EMBD_HEAD, device="cuda")
    o_in = torch.randn(1, N_HEAD * N_EMBD_HEAD, device="cuda")

    def one_layer_attn_proj():
        x @ q_w.T
        x @ k_w.T
        x @ v_w.T
        o_in @ o_w.T

    return bench_cuda(one_layer_attn_proj)


def load_trace():
    by_pl = defaultdict(dict)
    with open(TRACE_PATH) as f:
        for line in f:
            r = json.loads(line)
            by_pl[(r["prompt_id"], r["layer"])][r["token_pos"]] = set(r["experts"])
    return by_pl


def main():
    assert torch.cuda.is_available()
    print("GPU:", torch.cuda.get_device_name(0))

    gpu_expert_s = measure_gpu_expert_time()
    gpu_attn_s = measure_gpu_attention_time()
    print(f"\nmeasured GPU per-expert matmul time: {gpu_expert_s*1000:.4f} ms")
    print(f"measured GPU per-layer attention-projection time: {gpu_attn_s*1000:.4f} ms")

    gpu_attn_total_ms = gpu_attn_s * 1000 * N_LAYERS
    print(f"implied GPU attention total (48 layers): {gpu_attn_total_ms:.2f} ms")
    print(f"(measured hybrid total is {HYBRID_MS_PER_TOKEN} ms -- attention is "
          f"{gpu_attn_total_ms/HYBRID_MS_PER_TOKEN:.1%} of it under this estimate)")

    cpu_experts_total_ms = HYBRID_MS_PER_TOKEN - gpu_attn_total_ms
    if cpu_experts_total_ms <= 0:
        print("\nWARNING: GPU attention estimate exceeds the entire measured hybrid budget.")
        print("The attention proxy (Q/K/V/O projections only, no RoPE/softmax/KV-read) is")
        print("too expensive relative to reality, or the hybrid number needs rechecking.")
        print("Stopping -- the derived CPU-per-expert number would be meaningless (negative).")
        return
    cpu_per_expert_ms = cpu_experts_total_ms / (N_LAYERS * N_EXPERT_USED)
    print(f"implied CPU expert-compute budget (48 layers x 8 experts): {cpu_experts_total_ms:.2f} ms")
    print(f"implied CPU per-expert time (derived, not measured): {cpu_per_expert_ms:.4f} ms")
    print("(derived by subtracting the GPU attention estimate from the real measured")
    print(" 22.0ms hybrid total -- not an independent measurement; see module docstring)")

    transfer_per_expert_ms = (BYTES_PER_EXPERT / PCIE_BW_BYTES_S) * 1000
    print(f"\nmeasured transfer time per expert: {transfer_per_expert_ms:.4f} ms")

    by_pl = load_trace()
    layers = sorted(set(l for (_, l) in by_pl.keys()))
    prompt_ids = sorted(set(p for (p, _) in by_pl.keys()))

    total_hit_frac = []
    per_token_prefetch_ms = []
    per_token_no_prefetch_ms = []

    for pid in prompt_ids:
        for layer in layers:
            d = by_pl.get((pid, layer))
            if not d:
                continue
            positions = sorted(d.keys())
            for a, b in zip(positions, positions[1:]):
                predicted = d[a]   # naive-repeat: predict E_{t+1} = E_t
                actual = d[b]
                hits = len(predicted & actual)
                misses = N_EXPERT_USED - hits
                total_hit_frac.append(hits / N_EXPERT_USED)

                gpu_hit_ms = hits * gpu_expert_s * 1000
                cpu_fallback_ms = max(gpu_hit_ms, misses * cpu_per_expert_ms)
                transfer_then_gpu_ms = max(gpu_hit_ms, misses * transfer_per_expert_ms) + misses * gpu_expert_s * 1000
                layer_prefetch_ms = min(cpu_fallback_ms, transfer_then_gpu_ms)

                layer_no_prefetch_ms = N_EXPERT_USED * cpu_per_expert_ms

                per_token_prefetch_ms.append(layer_prefetch_ms)
                per_token_no_prefetch_ms.append(layer_no_prefetch_ms)

    avg_hit = sum(total_hit_frac) / len(total_hit_frac)
    avg_layer_prefetch_ms = sum(per_token_prefetch_ms) / len(per_token_prefetch_ms)
    avg_layer_no_prefetch_ms = sum(per_token_no_prefetch_ms) / len(per_token_no_prefetch_ms)

    sim_token_prefetch_ms = gpu_attn_total_ms + avg_layer_prefetch_ms * N_LAYERS
    sim_token_no_prefetch_ms = gpu_attn_total_ms + avg_layer_no_prefetch_ms * N_LAYERS

    print(f"\nreal avg cache-hit rate (naive-repeat, all 8 prompts, all layers): {avg_hit:.1%}")
    print(f"\nsimulated 'hybrid, no prefetch' control: {sim_token_no_prefetch_ms:.2f} ms/token")
    print(f"  (should equal {HYBRID_MS_PER_TOKEN} ms by construction -- arithmetic check, not independent)")
    print(f"\nsimulated 'prefetch-enabled' estimate: {sim_token_prefetch_ms:.2f} ms/token "
          f"({1000/sim_token_prefetch_ms:.1f} tok/s)")
    speedup = (HYBRID_MS_PER_TOKEN - sim_token_prefetch_ms) / HYBRID_MS_PER_TOKEN
    print(f"vs measured real hybrid baseline ({HYBRID_MS_PER_TOKEN} ms/token, "
          f"{1000/HYBRID_MS_PER_TOKEN:.1f} tok/s): {speedup:+.1%}")

    with open("/home/everett/AI2/experiments/prefetch_simulation_result.json", "w") as f:
        json.dump({
            "gpu_per_expert_ms": gpu_expert_s * 1000,
            "gpu_attn_total_ms": gpu_attn_total_ms,
            "cpu_per_expert_ms_derived": cpu_per_expert_ms,
            "transfer_per_expert_ms": transfer_per_expert_ms,
            "avg_hit_rate": avg_hit,
            "sim_no_prefetch_ms_per_token": sim_token_no_prefetch_ms,
            "sim_prefetch_ms_per_token": sim_token_prefetch_ms,
            "measured_hybrid_ms_per_token": HYBRID_MS_PER_TOKEN,
            "measured_full_cpu_ms_per_token": FULL_CPU_MS_PER_TOKEN,
            "speedup_vs_measured_hybrid": speedup,
        }, f, indent=2)
    print("\nsaved to experiments/prefetch_simulation_result.json")


if __name__ == "__main__":
    main()
