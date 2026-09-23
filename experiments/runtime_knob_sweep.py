"""
Re-tune the llama-server runtime knobs now that VRAM is the binding constraint.

Every knob in config.py was tuned under the OLD configuration, where every MoE
expert sat in CPU RAM and 10 GB of the card was empty. moe_offload_sweep.py
changed that: the GPU is now packed to the edge, and each extra layer of experts
it can hold is worth ~2-3 tok/s. That invalidates two earlier conclusions and
makes a third worth re-checking:

  KV CACHE PRECISION. BENCHMARK_RESULTS.md rejected `-ctk/-ctv q8_0` with "no
  throughput benefit here since VRAM isn't the bottleneck". It is now. At
  n_ctx 4096 an f16 KV cache for this model is ~750 MiB; q8_0 halves it, which is
  roughly one more layer of experts on the GPU. Whether that beats the small
  per-token cost of dequantising KV is exactly the kind of thing to measure.

  SERVER SLOTS. llama-server defaults to `-np auto` = 4 parallel slots. The app is
  a single-user Gradio UI and sends one request at a time. Slots cost compute
  buffer and bookkeeping; `-np 1` may free VRAM for experts.

  THREADS. 16 was the optimum when all 48 layers' experts ran on the CPU. Now
  only ~21-24 do, so the CPU is doing half the work and the best thread count
  may have moved. This box is a Ryzen 9 7950X: 16 physical cores across two CCDs
  (cores 0-7 and 8-15 each have their own L3), with SMT siblings at 16-31. So
  pinning to physical cores, and to a single CCD, are both tested -- expert
  matmul is memory-bandwidth-bound, and cross-CCD hops or SMT siblings fighting
  over one core's cache can cost more than the extra threads buy.

  UBATCH. Affects prompt processing (prefill), not decode. Larger ubatch = faster
  prefill of long chat prompts, but a bigger CUDA compute buffer = fewer experts
  in VRAM = slower decode. A real tradeoff, so both are measured.

Every arm is VRAM-FITTED -- started at --n-cpu-moe 18 and stepped up one layer
until it loads -- because several of these knobs only pay off through the extra
layer they free, and a fixed split would hide that. Rounds are interleaved so
drift lands on every arm equally.

Decode and prefill are read from llama-server's own `timings` (predicted_per_second,
prompt_per_second), which excludes HTTP overhead and is steadier than wall clock.
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
PORT = 8088
ROUNDS = 2
START_N = 18

DECODE_PROMPTS = [
    "Write a Python class implementing a least-recently-used cache with get and put.",
    "Explain the CAP theorem in two sentences.",
    "Summarize the plot of Romeo and Juliet in one short paragraph.",
]
MAX_TOKENS = 192
# ~2k-token prompt for prefill: long enough that ubatch size actually matters.
PREFILL_PROMPT = ("Here is a log of events from a distributed system. Summarize the failures.\n" +
                  "\n".join(f"[node-{i % 7}] t={1000 + i * 13}ms heartbeat ok, replica lag {i % 11}ms, "
                            f"queue depth {(i * 37) % 250}" for i in range(110)))

F16, Q8 = [], ["-ctk", "q8_0", "-ctv", "q8_0"]

# name -> extra args (threads and fitted split are added per arm)
PHASES = {
    "kv_and_slots": [
        ("f16_np4", F16 + ["-t", "16"]),
        ("q8_np4", Q8 + ["-t", "16"]),
        ("f16_np1", F16 + ["-t", "16", "-np", "1"]),
        ("q8_np1", Q8 + ["-t", "16", "-np", "1"]),
    ],
}


def thread_arms(base):
    return [
        ("t8_ccd0", base + ["-t", "8", "--cpu-range", "0-7", "--cpu-strict", "1"]),
        ("t12", base + ["-t", "12"]),
        ("t16", base + ["-t", "16"]),
        ("t16_phys", base + ["-t", "16", "--cpu-range", "0-15", "--cpu-strict", "1"]),
        ("t24", base + ["-t", "24"]),
    ]


def ubatch_arms(base):
    return [
        ("ub256", base + ["-ub", "256", "-b", "256"]),
        ("ub512", base + ["-ub", "512"]),
        ("ub1024", base + ["-ub", "1024", "-b", "2048"]),
    ]


def wait_healthy(proc, timeout=420):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(1)
    return False


def post(payload):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def warm_ok(proc):
    """A split that loads can still abort on its first request: CUDA allocates the
    cuBLAS workspace lazily, and the first run of this sweep crashed exactly that
    way ("CUDA error: out of memory" in cublas_handle) after /health was already
    OK. So a split only counts as fitting if a real generation also succeeds."""
    # Must be longer than one ubatch: a 2-token warm-up passed and the next real
    # prompt still crashed, because tiny batches never touch the cuBLAS path.
    try:
        post({"prompt": PREFILL_PROMPT, "n_predict": 4, "temperature": 0, "cache_prompt": False})
    except Exception:
        return False
    return proc.poll() is None


def gpu_free_mb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def launch(n, extra, log_path):
    cmd = [SERVER, "-m", BIG, "--port", str(PORT), "-ngl", "999", "--n-cpu-moe", str(n),
           "-c", "4096", "-fa", "on", "--no-warmup"] + extra
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{VENDOR}:{BACKEND}"
    logf = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
    return proc, logf


def stop(proc, logf):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
    logf.close()


def wait_for_foreign_gpu_users(max_wait_s=3600):
    """Pause while another program (e.g. Ollama) has a model on the GPU.

    The first attempt at this sweep found Ollama holding 6.3 GB. Measuring then
    would fit every arm to a card with half its memory gone -- numbers that look
    valid and mean nothing -- and would fight the user's own session for VRAM.
    """
    waited = 0
    while waited < max_wait_s:
        busy = subprocess.run(["pgrep", "-f", "^/usr/local/lib/ollama/"],
                              capture_output=True).returncode == 0
        if not busy:
            return waited
        if waited == 0:
            print("  (paused: Ollama has a model loaded; waiting for it to unload)", file=sys.stderr)
        time.sleep(15)
        waited += 15
    raise RuntimeError("another process held the GPU for over an hour; aborting sweep")


def foreign_gpu_user():
    return subprocess.run(["pgrep", "-f", "^/usr/local/lib/ollama/"],
                          capture_output=True).returncode == 0


def run_arm(name, extra, fitted, rnd, attempts=6):
    """Fit and measure one arm, discarding any attempt that overlapped Ollama.

    The first version of this checked for Ollama only before starting, and cached
    each arm's fitted split for the next round. Ollama then loaded a model
    mid-fit, so f16_np4 "fitted" at split 28 with 2.3 GB free (should be ~21),
    and that wrong split would have been reused. Now: the fit restarts from
    START_N every round, and Ollama is checked throughout; if it appeared at any
    point, the whole attempt is thrown away and retried once it unloads.
    """
    for _ in range(attempts):
        wait_for_foreign_gpu_users()
        r = _measure_arm(name, extra, rnd)
        if r.get("contaminated"):
            print(f"  [{name}] r{rnd} discarded: Ollama loaded a model mid-run; retrying",
                  file=sys.stderr)
            continue
        if "n_cpu_moe" in r:
            fitted[name] = r["n_cpu_moe"]
        return r
    return {"error": "kept getting interrupted by another GPU user"}


def _measure_arm(name, extra, rnd):
    n = START_N
    while n <= 48:
        r = _measure_at(name, extra, rnd, n)
        if r.get("does_not_fit"):
            n += 1
            continue
        return r
    return {"error": "never loaded"}


def _measure_at(name, extra, rnd, n):
    log_path = os.path.join(AI2, "state", f"knob_{name}_r{rnd}.log")
    if foreign_gpu_user():
        return {"contaminated": True}
    proc, logf = launch(n, extra, log_path)
    if not (wait_healthy(proc) and warm_ok(proc)):
        stop(proc, logf)
        return {"does_not_fit": True}
    try:
        return _measure_running(proc, n)
    except Exception:
        # The server died mid-measurement (a lazily-allocated buffer the warm-up
        # didn't reach). That split doesn't really fit; try one more layer on CPU.
        if proc.poll() is not None:
            return {"does_not_fit": True}
        raise
    finally:
        stop(proc, logf)


def _measure_running(proc, n):
    if foreign_gpu_user():
        return {"contaminated": True}
    free = gpu_free_mb()
    dec_tok = dec_ms = 0.0
    for p in DECODE_PROMPTS:
        t = post({"prompt": p, "n_predict": MAX_TOKENS, "temperature": 0,
                  "cache_prompt": False})["timings"]
        dec_tok += t["predicted_n"]
        dec_ms += t["predicted_ms"]
    t = post({"prompt": PREFILL_PROMPT, "n_predict": 1, "temperature": 0,
              "cache_prompt": False})["timings"]
    if foreign_gpu_user():
        # Ollama started while we were measuring: it may have been competing
        # for PCIe / CPU the whole time.
        return {"contaminated": True}
    return {"n_cpu_moe": n, "free_mb": free,
            "decode_tps": 1000 * dec_tok / dec_ms,
            "prefill_tps": t["prompt_per_second"], "prefill_n": t["prompt_n"]}


def sweep(label, arms, results):
    fitted = {}
    samples = {name: [] for name, _ in arms}
    for rnd in range(ROUNDS):
        for name, extra in arms:
            r = run_arm(name, extra, fitted, rnd)
            if "error" in r:
                print(f"  [{label}/{name}] r{rnd} ERROR {r['error']}", file=sys.stderr)
                continue
            samples[name].append(r)
            print(f"  [{label}/{name}] r{rnd} ncmoe={r['n_cpu_moe']} free={r['free_mb']}MiB "
                  f"decode={r['decode_tps']:.2f} prefill={r['prefill_tps']:.0f} tok/s", file=sys.stderr)
    table = []
    for name, extra in arms:
        s = samples[name]
        if not s:
            table.append({"arm": name, "error": "never loaded"})
            continue
        table.append({
            "arm": name, "args": extra,
            "n_cpu_moe": s[-1]["n_cpu_moe"], "free_mb": s[-1]["free_mb"],
            "decode_tps": round(statistics.median(x["decode_tps"] for x in s), 2),
            "decode_range": [round(min(x["decode_tps"] for x in s), 2),
                             round(max(x["decode_tps"] for x in s), 2)],
            "prefill_tps": round(statistics.median(x["prefill_tps"] for x in s), 1),
            "prefill_n": s[-1]["prefill_n"],
        })
    results[label] = table
    with open(os.path.join(AI2, "experiments", "runtime_knob_sweep_result.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n=== {label} ===")
    print("%-12s %6s %8s %9s %13s %9s" % ("arm", "ncmoe", "freeMiB", "decode", "range", "prefill"))
    for r in table:
        if "error" in r:
            print("%-12s ERROR" % r["arm"])
            continue
        print("%-12s %6d %8s %9.2f %13s %9.1f" % (r["arm"], r["n_cpu_moe"], r["free_mb"], r["decode_tps"],
                                                  "%.1f-%.1f" % tuple(r["decode_range"]), r["prefill_tps"]))
    return table


def best(table):
    ok = [r for r in table if "error" not in r]
    return max(ok, key=lambda r: r["decode_tps"])


def strip_threads(args):
    out, skip = [], False
    for a in args:
        if skip:
            skip = False
            continue
        if a in ("-t", "--cpu-range", "--cpu-strict"):
            skip = True
            continue
        out.append(a)
    return out


def main():
    results = {}
    t1 = sweep("kv_and_slots", PHASES["kv_and_slots"], results)
    base = strip_threads(best(t1)["args"])
    print(f"\ncarrying forward: {base}", file=sys.stderr)
    t2 = sweep("threads", thread_arms(base), results)
    base = best(t2)["args"]
    print(f"\ncarrying forward: {base}", file=sys.stderr)
    sweep("ubatch", ubatch_arms(base), results)
    print("\nsaved experiments/runtime_knob_sweep_result.json")


if __name__ == "__main__":
    main()
