"""
CPU and GPU runtime knobs, re-tested for Q2_K, where only 3 of 48 layers' experts
run on the CPU.

runtime_knob_sweep.py tuned these for Q4_K_M, where 23 layers ran on the CPU and
the CPU was half the work. Now the GPU does ~95% of it, so the answers may move.

CPU (Ryzen 9 7950X: two CCDs, cores 0-7 and 8-15, each with its own 32 MB L3
cache; SMT siblings 16-31; governor already `performance`):
  threads      16 (current), 8, 4. Fewer threads means less synchronization per
               token, and 3 layers of experts may not need 16.
  pinning      --cpu-range + --cpu-strict 1 onto one CCD, so a token's expert
               work stays inside one L3 cache instead of bouncing between them.
               CCD1 (8-15) too, since the OS and the GPU driver's interrupts
               tend to land on the low cores.
  polling      --poll 0 (threads sleep between tokens) and 100 (spin). Default 50.
GPU (RTX 5070 on PCIe 5.0 x16):
  CUDA graphs  GGML_CUDA_DISABLE_GRAPHS=1. Graphs replay the GPU kernels with one
               launch; if disabling them is slower, they are active and helping.
               They can be silently disabled by CPU splits, which is the question.
  ubatch       256 / 1024 vs 512. Prompt reading (prefill) uses big matrix
               multiplies (cuBLAS, the GPU's tensor math); a bigger ubatch reads
               long prompts faster but its compute buffer takes VRAM.

Every arm: Q2_K, --n-cpu-moe 3 (the app's fitted split), -c 4096, -fa on, -np 1,
warm-up, decode over 3 prompts x 192 tokens (repeat_penalty 1.0, as the app now
sends), prefill of a ~2,950-token prompt, free VRAM. 3 interleaved rounds.
"""
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request

AI2 = "/home/everett/AI2"
sys.path.insert(0, AI2)
from config import BIG_MODELS, Paths  # noqa: E402
from local_engine import _WARMUP_PROMPT, _gpu_free_mb  # noqa: E402

P = Paths()
# MODEL/SPLIT: the model whose split these knobs are being tuned for. Thread count
# and CCD pinning matter in proportion to how many layers run on the CPU, so the
# default model's own split is the one to tune (UD-Q3_K_XL: 13 layers on the CPU;
# Q2_K: 3).
MODEL = BIG_MODELS[os.environ.get("MODEL", "q2_k")][0]
SPLIT = os.environ.get("SPLIT", "3")
PORT = 8101
ROUNDS = int(os.environ.get("ROUNDS", "3"))

# (name, extra args, extra env)
ARMS = [
    ("t16 (current)", ["-t", "16"], {}),
    ("t8", ["-t", "8"], {}),
    ("t8 CCD0 strict", ["-t", "8", "--cpu-range", "0-7", "--cpu-strict", "1"], {}),
    ("t8 CCD1 strict", ["-t", "8", "--cpu-range", "8-15", "--cpu-strict", "1"], {}),
    ("t4 CCD1 strict", ["-t", "4", "--cpu-range", "8-11", "--cpu-strict", "1"], {}),
    ("t16 physical strict", ["-t", "16", "--cpu-range", "0-15", "--cpu-strict", "1"], {}),
    ("t16 poll 0", ["-t", "16", "--poll", "0"], {}),
    ("t16 poll 100", ["-t", "16", "--poll", "100"], {}),
    ("t16 no CUDA graphs", ["-t", "16"], {"GGML_CUDA_DISABLE_GRAPHS": "1"}),
    ("t16 ubatch 256", ["-t", "16", "-ub", "256"], {}),
    ("t16 ubatch 1024", ["-t", "16", "-ub", "1024", "-b", "2048"], {}),
]
DECODE_PROMPTS = [
    "Explain how a hash map handles collisions.",
    "Write a Python class implementing a least-recently-used cache with get and put.",
    "Summarize the causes of the French Revolution in a few paragraphs.",
]
PREFILL = ("Here is a log of events from a distributed system. Summarize the failures.\n" +
           "\n".join(f"[node-{i % 7}] t={1000 + i * 13}ms heartbeat ok, replica lag {i % 11}ms, "
                     f"queue depth {(i * 37) % 250}" for i in range(110)))


def post(path, payload, timeout=600):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def run(name, extra, env_over, rnd):
    env = {k: v for k, v in os.environ.items() if not k.startswith("GGML_CUDA_")}
    env.update(env_over, LD_LIBRARY_PATH=P.llama_server_ld_library_path)
    cmd = [P.llama_server_bin, "-m", MODEL, "-c", "4096", "--port", str(PORT), "-ngl", "999", "-fa", "on",
           "-np", "1", "--n-cpu-moe", SPLIT] + extra
    log = open(os.path.join(AI2, "state", f"knobs_{name.replace(' ', '_')}_r{rnd}.log"), "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    try:
        for _ in range(300):
            if proc.poll() is not None:
                return {"error": "load_failed"}
            try:
                if urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3).status == 200:
                    break
            except Exception:
                time.sleep(1)
        try:
            post("/completion", {"prompt": _WARMUP_PROMPT, "n_predict": 8, "temperature": 0})
            tok = ms = 0.0
            texts = []
            for p in DECODE_PROMPTS:
                r = post("/v1/chat/completions", {"messages": [{"role": "user", "content": p}], "max_tokens": 192,
                                                  "temperature": 0, "repeat_penalty": 1.0, "cache_prompt": False})
                tok += r["timings"]["predicted_n"]
                ms += r["timings"]["predicted_ms"]
                texts.append(r["choices"][0]["message"]["content"])
            pf = post("/completion", {"prompt": PREFILL, "n_predict": 1, "temperature": 0,
                                      "cache_prompt": False})["timings"]
            return {"decode_tps": 1000 * tok / ms, "prefill_tps": pf["prompt_per_second"],
                    "free_mb": _gpu_free_mb(), "texts": texts}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log.close()
        time.sleep(2)


def main():
    out_path = os.path.join(AI2, "experiments", "cpu_gpu_knobs_result.json")
    res = {n: [] for n, *_ in ARMS}
    for rnd in range(ROUNDS):
        # Rotate the order each round so no arm always runs first (coldest).
        order = ARMS[rnd % len(ARMS):] + ARMS[:rnd % len(ARMS)]
        for name, extra, env_over in order:
            r = run(name, extra, env_over, rnd)
            res[name].append(r)
            msg = r.get("error") or (f"decode {r['decode_tps']:.1f} prefill {r['prefill_tps']:.0f} "
                                     f"free {r['free_mb']} MiB")
            print(f"  r{rnd} [{name}] {msg}", file=sys.stderr, flush=True)
            json.dump(res, open(out_path, "w"), indent=1)

    base = [r for r in res["t16 (current)"] if "error" not in r]
    bd = statistics.median(r["decode_tps"] for r in base)
    bp = statistics.median(r["prefill_tps"] for r in base)
    print(f"\n=== CPU/GPU KNOBS (Q2_K, split {SPLIT}, medians of {ROUNDS} rounds) ===")
    print("%-22s %8s %7s %14s %9s %7s %6s %s" % ("arm", "decode", "vs cur", "range", "prefill", "vs cur", "free",
                                                "same text"))
    for name, *_ in ARMS:
        ok = [r for r in res[name] if "error" not in r]
        if not ok:
            print("%-22s ERROR %s" % (name, res[name][0]["error"]))
            continue
        d = statistics.median(r["decode_tps"] for r in ok)
        p = statistics.median(r["prefill_tps"] for r in ok)
        rng = "%.1f-%.1f" % (min(r["decode_tps"] for r in ok), max(r["decode_tps"] for r in ok))
        same = sum(a == b for a, b in zip(ok[-1]["texts"], base[-1]["texts"]))
        print("%-22s %8.1f %6.2fx %14s %9.0f %6.2fx %6s %d/3" % (name, d, d / bd, rng, p, p / bp, ok[-1]["free_mb"],
                                                                same))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
