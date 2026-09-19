"""
Does speculative decoding have ANY headroom on this box, for this model?

This is the test that decides whether to invest in a better draft model at all
-- either by self-training the EAGLE3 head, or by continuing with the
Qwen2.5-Coder-0.5B draft. Both projects are bets on the same assumption:

    "if the draft model were more accurate, generation would get faster."

That assumption is only true if verifying k tokens in one target forward pass
costs meaningfully less than k separate forward passes. On a GPU-resident dense
model it does (the weights are read once and applied to k vectors -- decode is
memory-bandwidth bound, so k tokens ride along nearly free). On *this* setup the
target is an MoE with its experts pinned to CPU RAM, and each token in the batch
routes to its own set of experts, so widening the batch may pull in k disjoint
sets of expert weights and buy nothing.

BENCHMARK_RESULTS.md already measured the consequence: the 0.5B draft reaches a
33-80% accept rate and still yields only 1.02x. That is the signature of zero
batching headroom, but it was inferred, not measured. This measures it directly.

Method: llama-server reports `timings.prompt_ms` / `timings.prompt_n` for a
prefill. A prefill of L tokens is exactly the shape of a speculative verify of L
tokens -- one forward pass, L positions. So per-token cost at width L, relative
to width 1, IS the batching curve that bounds speculative decoding.

    speedup_ceiling(k) = k * cost_per_token(1) / cost_per_token(k)

That is a hard ceiling: it assumes a *perfect* draft model whose every token is
accepted, and charges nothing for running the draft. No draft model, however
well trained, can beat it.

Run with the real production config (experts on CPU) and, for contrast, with the
model fully on GPU if it fits -- the difference isolates whether CPU expert
offload is what kills it.
"""
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request

AI2 = "/home/everett/AI2"
BACKEND = "/home/everett/.lmstudio/extensions/backends/llama.cpp-linux-x86_64-nvidia-cuda12-avx2-2.37.0"
VENDOR = "/home/everett/.lmstudio/extensions/backends/vendor/linux-llama-cuda12-vendor-v1"
SERVER = os.path.join(BACKEND, "llama-server")

BIG = "/home/everett/.lmstudio/models/lmstudio-community/Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"
PORT = 8098

# Batch widths to probe. 1 is the decode-step baseline; 2-8 is the range a real
# speculative decoder actually verifies at (--spec-draft-n-max is 3-5 typically).
WIDTHS = [1, 2, 3, 4, 6, 8, 16]
REPEATS = 5


def wait_healthy(timeout=600):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(2)
    return False


def tokenize(text):
    body = json.dumps({"content": text}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/tokenize", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["tokens"]


def prefill_ms(tokens):
    """One forward pass over exactly len(tokens) positions; return ms and n."""
    body = json.dumps({
        "prompt": tokens,          # token ids, so width is exact, not tokenizer-dependent
        "n_predict": 1,
        "temperature": 0,
        "cache_prompt": False,
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())
    t = out["timings"]
    return t["prompt_ms"], t["prompt_n"]


def run_config(name, extra_args):
    log_path = os.path.join(AI2, "state", f"headroom_{name}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    cmd = [SERVER, "-m", BIG, "--port", str(PORT), "-c", "4096",
           "-fa", "on", "-t", "16", "--no-warmup"] + extra_args
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{VENDOR}:{BACKEND}"

    with open(log_path, "w") as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
        try:
            if not wait_healthy():
                proc.kill()
                return {"config": name, "error": "server never became healthy", "log": log_path}

            # A long filler prompt gives a pool of real token ids to slice widths from.
            filler = tokenize("The quick brown fox jumps over the lazy dog. " * 60)
            # Warm the graph so the first timed call isn't paying one-off setup.
            prefill_ms(filler[:8])

            per_width = {}
            for w in WIDTHS:
                samples = []
                for i in range(REPEATS):
                    # Offset the slice each repeat so we aren't measuring one lucky
                    # cache state, and cache_prompt is off so nothing is reused.
                    chunk = filler[i * 3: i * 3 + w]
                    ms, n = prefill_ms(chunk)
                    if n != w:
                        print(f"  warn: asked width {w}, server reported {n}", file=sys.stderr)
                    samples.append(ms / max(n, 1))
                per_width[w] = round(statistics.median(samples), 3)
                print(f"  [{name}] width {w:2d}: {per_width[w]:7.3f} ms/token", file=sys.stderr)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)

    base = per_width[1]
    ceiling = {w: round(base / per_width[w], 3) for w in WIDTHS}
    return {"config": name, "ms_per_token": per_width, "speedup_ceiling": ceiling,
            "log": log_path}


def main():
    if not os.path.exists(BIG):
        print(f"missing target: {BIG}", file=sys.stderr)
        sys.exit(1)

    results = []
    # The production config from config.py: everything on GPU except the MoE
    # expert FFNs, which live in CPU RAM.
    print("=== production: -ngl 999 with experts on CPU ===", file=sys.stderr)
    results.append(run_config("experts_cpu",
                              ["-ngl", "999", "-ot", "ffn_(gate|down|up)_exps=CPU"]))
    # Contrast: no GPU at all. If the curve looks the same, the GPU layers were
    # never the bottleneck and the experts are the whole story.
    print("=== contrast: CPU only ===", file=sys.stderr)
    results.append(run_config("cpu_only", ["-ngl", "0"]))
    # The tuned split found by moe_offload_sweep.py. Speculative decoding's
    # break-even depends entirely on the shape of this curve, and moving experts
    # into VRAM changes the shape -- so the verdict has to be re-checked against
    # the config we would actually ship, not the old one.
    tuned = os.environ.get("TUNED_N_CPU_MOE")
    if tuned:
        print(f"=== tuned: --n-cpu-moe {tuned} ===", file=sys.stderr)
        results.append(run_config(f"ncmoe{tuned}", ["-ngl", "999", "--n-cpu-moe", tuned]))

    out_path = os.path.join(AI2, "experiments", "spec_headroom_result.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== SPECULATIVE DECODING HEADROOM ===")
    print("speedup ceiling = best possible speedup from a PERFECT draft model at that width\n")
    hdr = "config".ljust(14) + "".join(f"{('w=%d' % w):>9s}" for w in WIDTHS)
    print(hdr)
    for r in results:
        if "error" in r:
            print(f"{r['config']:14s} ERROR: {r['error']}")
            continue
        row = r["config"].ljust(14) + "".join(f"{r['speedup_ceiling'][w]:9.2f}" for w in WIDTHS)
        print(row)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
