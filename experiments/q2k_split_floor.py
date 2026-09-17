"""
Q2_K is 2.09x Q4_K_M in the app (159.6 vs 76.3 tok/s, quant_speed_quality.py) but
still leaves 5-6 layers of experts in RAM. How fast is it with fewer, and what
does it take to get there?

VRAM at --n-cpu-moe 6 (llama-server's own log): model 9,049 MiB, KV cache 384 MiB
(f16, 4096 ctx), compute 147 MiB; the desktop holds ~690 MiB (gnome-shell,
Firefox, remote desktop). Each Q2_K layer's experts are 189-209 MiB, so split 0
needs ~1.2 GB more than the app's fitting leaves. Two free levers:
  - Runtime.vram_headroom_mb (768 MiB): the app refuses any split that leaves less
    free after warm-up. Worth ~3-4 layers if it isn't needed (PLAN.md item 1).
  - KV cache at q8_0 instead of f16: -192 MiB at 4096 ctx, ~1 layer.

Each arm: bare llama-server (no headroom rule), ~700-token warm-up (the same one
BigModelServer uses, which triggers the lazy cuBLAS workspace allocation), free
VRAM, decode tok/s over 3 prompts x 192 tokens, then a stress request (~2,950-
token prompt + 512 generated tokens, the largest compute graph the app makes) to
see whether a server with no margin survives real use, and free VRAM after it.
A series stops at its first split that fails to load or crashes; lower splits
need strictly more VRAM.

Output is also compared token-for-token across splits of the same model: which
layer's experts run where must not change what the model says.
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
from config import Paths  # noqa: E402
from local_engine import _WARMUP_PROMPT, _gpu_free_mb  # noqa: E402

P = Paths()
Q = os.path.join(AI2, "models", "quants")
Q2K = os.path.join(Q, "Qwen_Qwen3-30B-A3B-Instruct-2507-Q2_K.gguf")
IQ2 = os.path.join(Q, "Qwen3-30B-A3B-Instruct-2507-UD-IQ2_XXS.gguf")
PORT = 8096
KV_Q8 = ["-ctk", "q8_0", "-ctv", "q8_0"]

# (series, model, extra args, splits high -> low)
SERIES = [
    ("Q2_K kv f16", Q2K, [], [6, 5, 4, 3, 2, 1, 0]),
    ("Q2_K kv q8_0", Q2K, KV_Q8, [4, 3, 2, 1, 0]),
    ("UD-IQ2_XXS kv f16", IQ2, [], [2, 1, 0]),
]

DECODE_PROMPTS = [
    "Explain how a hash map handles collisions.",
    "Write a Python class implementing a least-recently-used cache with get and put.",
    "Summarize the causes of the French Revolution in a few paragraphs.",
]
STRESS_PROMPT = ("Here is a log of events from a distributed system. Summarize the failures, then propose fixes.\n" +
                 "\n".join(f"[node-{i % 7}] t={1000 + i * 13}ms heartbeat ok, replica lag {i % 11}ms, "
                           f"queue depth {(i * 37) % 250}" for i in range(110)))


def post(path, payload, timeout=600):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def chat(prompt, n):
    return post("/v1/chat/completions", {"messages": [{"role": "user", "content": prompt}], "max_tokens": n,
                                         "temperature": 0, "cache_prompt": False})


def run(label, model, extra, split):
    log_path = os.path.join(AI2, "state", f"q2kfloor_{label.replace(' ', '_')}_{split}.log")
    env = dict(os.environ, LD_LIBRARY_PATH=P.llama_server_ld_library_path)
    cmd = [P.llama_server_bin, "-m", model, "-c", "4096", "-t", "16", "--port", str(PORT), "-ngl", "999",
           "-fa", "on", "-np", "1", "--n-cpu-moe", str(split)] + extra
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
        try:
            deadline = time.time() + 300
            while True:
                if proc.poll() is not None:
                    return {"error": "load_failed"}
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3) as r:
                        if r.status == 200:
                            break
                except Exception:
                    pass
                if time.time() > deadline:
                    return {"error": "load_timeout"}
                time.sleep(1)
            out = {}
            try:
                post("/completion", {"prompt": _WARMUP_PROMPT, "n_predict": 8, "temperature": 0})
                out["free_after_warmup_mb"] = _gpu_free_mb()
                tok = ms = 0.0
                texts = []
                for p in DECODE_PROMPTS:
                    r = chat(p, 192)
                    tok += r["timings"]["predicted_n"]
                    ms += r["timings"]["predicted_ms"]
                    texts.append(r["choices"][0]["message"]["content"])
                out["decode_tps"] = 1000 * tok / ms
                out["texts"] = texts
                s = chat(STRESS_PROMPT, 512)
                out["stress_prompt_tokens"] = s["timings"]["prompt_n"]
                out["stress_generated"] = s["timings"]["predicted_n"]
                out["free_after_stress_mb"] = _gpu_free_mb()
            except Exception as e:
                out["error"] = f"crashed: {type(e).__name__}: {e}" if proc.poll() is not None else \
                    f"{type(e).__name__}: {e}"
            return out
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
            time.sleep(2)


def main():
    results = {}
    out_path = os.path.join(AI2, "experiments", "q2k_split_floor_result.json")
    for label, model, extra, splits in SERIES:
        results[label] = {}
        for split in splits:
            r = run(label, model, extra, split)
            results[label][split] = r
            json.dump(results, open(out_path, "w"), indent=2, ensure_ascii=False)
            if "error" in r:
                print(f"  [{label}] split {split}: {r['error']}", file=sys.stderr, flush=True)
                break
            print(f"  [{label}] split {split}: decode {r['decode_tps']:.1f} tok/s, free after warm-up "
                  f"{r['free_after_warmup_mb']} MiB, after stress {r['free_after_stress_mb']} MiB",
                  file=sys.stderr, flush=True)

    print("\n=== Q2_K SPLIT FLOOR (no headroom rule) ===")
    for label, per in results.items():
        ok = {s: r for s, r in per.items() if "error" not in r}
        ref = next(iter(ok.values()), None)
        for s, r in per.items():
            if "error" in r:
                print(f"{label:18s} split {s}: {r['error']}")
                continue
            same = sum(a == b for a, b in zip(r["texts"], ref["texts"]))
            print(f"{label:18s} split {s}: {r['decode_tps']:6.1f} tok/s  free {r['free_after_warmup_mb']:5} -> "
                  f"{r['free_after_stress_mb']:5} MiB  same output as split {next(iter(ok))}: {same}/3")
        if len(ok) >= 3:
            xs = list(ok)
            ys = [1000 / ok[s]["decode_tps"] for s in xs]
            mx, my = statistics.mean(xs), statistics.mean(ys)
            slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
            print(f"{label:18s} fit: {slope:.3f} ms per CPU layer, floor {my - slope * mx:.2f} ms "
                  f"({1000 / (my - slope * mx):.0f} tok/s at split 0)")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
