"""
How many MoE layers should stay in VRAM? (The CUDA-side lever.)

spec_breakeven.py showed why this matters more than any draft model. With the
experts on CPU, a target forward at width 4 costs 13.95 ms/position against
22.19 at width 1 -- only a 37% saving for 4x the work. That shallow curve is
what makes speculative decoding unprofitable here: it forces the break-even
acceptance rate up to 65-83%, above what a small draft reaches. The curve is
shallow because the MoE expert FFNs live in CPU RAM, and each extra position in
the batch routes to its own experts, so widening the batch buys far less than it
would for a GPU-resident dense model.

That points the optimization at the target, not the draft. `config.py` currently
pins EVERY expert tensor to CPU via

    -ot "ffn_(gate|down|up)_exps=CPU"

which is all-or-nothing -- it never asks whether some experts could fit in the
12GB card alongside attention and KV. llama.cpp has `--n-cpu-moe N`, which keeps
the MoE weights of only the FIRST N layers on CPU and leaves the rest on GPU. N
= n_layers is today's config; lower N moves more experts into VRAM.

This sweeps N to find the lowest value that still loads, and measures decode
throughput at each. Every ms shaved off c(1) here is a straight win on every
token generated, speculative decoding or not.

Arms that OOM are reported as such rather than silently skipped -- the boundary
is the interesting part of the result.
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

AI2 = "/home/everett/AI2"
BACKEND = "/home/everett/.lmstudio/extensions/backends/llama.cpp-linux-x86_64-nvidia-cuda12-avx2-2.37.0"
VENDOR = "/home/everett/.lmstudio/extensions/backends/vendor/linux-llama-cuda12-vendor-v1"
SERVER = os.path.join(BACKEND, "llama-server")
BIG = "/home/everett/.lmstudio/models/lmstudio-community/Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"
PORT = 8096

PROMPTS = [
    "Write a Python function that reverses a singly linked list.",
    "Explain the CAP theorem in two sentences.",
    "Write a Python class implementing a least-recently-used cache with get and put.",
    "Summarize the plot of Romeo and Juliet in one short paragraph.",
]
MAX_TOKENS = 192

# 48 = every layer's experts on CPU, i.e. what config.py does today.
N_CPU_MOE = [48, 44, 40, 36, 32, 28, 24, 22, 21, 20, 19, 18, 17, 16]


def wait_healthy(proc, timeout=420):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False            # died during load -- almost always VRAM
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(2)
    return False


def gpu_used_mb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def run(n_cpu_moe):
    name = f"ncmoe{n_cpu_moe}"
    log_path = os.path.join(AI2, "state", f"moe_{name}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    cmd = [SERVER, "-m", BIG, "--port", str(PORT), "-ngl", "999",
           "--n-cpu-moe", str(n_cpu_moe),
           "-c", "4096", "-fa", "on", "-t", "16", "--no-warmup"]
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{VENDOR}:{BACKEND}"

    with open(log_path, "w") as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
        try:
            if not wait_healthy(proc):
                proc.kill()
                tail = ""
                with open(log_path) as f:
                    tail = "".join(f.readlines()[-4:]).strip().replace("\n", " | ")
                return {"n_cpu_moe": n_cpu_moe, "error": "failed to load",
                        "detail": tail[:400], "log": log_path}

            vram = gpu_used_mb()
            total_tokens = 0
            total_elapsed = 0.0
            for p in PROMPTS:
                body = json.dumps({"prompt": p, "n_predict": MAX_TOKENS,
                                   "temperature": 0, "cache_prompt": False}).encode()
                req = urllib.request.Request(
                    f"http://127.0.0.1:{PORT}/completion", data=body,
                    headers={"Content-Type": "application/json"})
                t0 = time.perf_counter()
                with urllib.request.urlopen(req, timeout=900) as r:
                    out = json.loads(r.read())
                total_elapsed += time.perf_counter() - t0
                total_tokens += out.get("tokens_predicted", 0)
            tok_s = total_tokens / total_elapsed if total_elapsed else 0.0
            print(f"  [ncmoe={n_cpu_moe}] {tok_s:6.2f} tok/s  vram={vram} MiB",
                  file=sys.stderr)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)

    return {"n_cpu_moe": n_cpu_moe, "tok_s": round(tok_s, 2),
            "tokens": total_tokens, "elapsed_s": round(total_elapsed, 2),
            "vram_used_mb": vram, "log": log_path}


def main():
    results = []
    fails = 0
    out_path = os.path.join(AI2, "experiments", "moe_offload_sweep_result.json")
    for n in N_CPU_MOE:
        print(f"=== --n-cpu-moe {n} ===", file=sys.stderr)
        r = run(n)
        results.append(r)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        if "error" in r:
            print(f"  failed: {r['detail'][:200]}", file=sys.stderr)
            fails += 1
            # Two consecutive failures means we are genuinely past the VRAM edge,
            # not hitting a one-off allocation hiccup.
            if fails >= 2:
                print("  stopping sweep: past the VRAM limit", file=sys.stderr)
                break
        else:
            fails = 0

    base = next((r["tok_s"] for r in results if r.get("n_cpu_moe") == 48 and "tok_s" in r), None)
    print("\n=== MoE OFFLOAD SWEEP ===")
    print("%-12s %9s %9s %10s" % ("n_cpu_moe", "tok/s", "vs 48", "vram MiB"))
    for r in results:
        if "error" in r:
            print("%-12s %s" % (r["n_cpu_moe"], r["error"]))
            continue
        ratio = ("%.2fx" % (r["tok_s"] / base)) if base else "--"
        print("%-12s %9.2f %9s %10s" % (r["n_cpu_moe"], r["tok_s"], ratio,
                                        r.get("vram_used_mb")))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
