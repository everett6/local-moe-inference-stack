"""
Re-run the very first benchmark, unchanged, so BENCHMARK_RESULTS.md has a real
before/after instead of a retraction.

BENCHMARK_RESULTS.md is this project's oldest measurement: five prompts, 128
tokens each, Q4_K_M with every MoE expert pinned to CPU via `-ot`. It recorded
46.29 tok/s and concluded that speculative decoding "isn't a productive place to
keep optimizing". Both halves have since been overtaken:

  - 46.29 tok/s was measured with 10 of the card's 12 GB unused. The fitted
    `--n-cpu-moe` split alone took Q4_K_M to ~82.
  - the speculation conclusion came from a run where speculation never engaged
    (Qwen2.5 draft, Qwen3 target, incompatible vocabularies -- llama-server
    logged it and served without speculation). N-gram drafting, which needs no
    draft model at all, is now on by default and worth +8% on code edits.

Rather than patch prose onto a stale table, this re-runs the identical five
prompts at the same 128-token ceiling on the current default configuration, and
prints the comparison table ready to paste. Nothing about the prompts or the
ceiling changes; only the machine underneath them.

Two arms, because the original file's headline claim was about speculation:

  current            the shipped configuration (Q2_K, fitted split, ubatch 1024,
                     n-gram speculation, n_ctx 8192)
  current, no spec   the same, with speculation off -- the honest test of the
                     retracted claim, this time with speculation that actually
                     engages

Run at a 175 W cap with nothing else on the GPU. ~4 minutes.
"""
import json
import os
import statistics
import sys
from dataclasses import replace

AI2 = "/home/everett/AI2"
sys.path.insert(0, AI2)
from config import BIG_MODELS, BIG_MODEL, Paths, Runtime  # noqa: E402
from local_engine import BigModelServer  # noqa: E402

ROUNDS = int(os.environ.get("ROUNDS", "2"))
MAX_TOKENS = 128                     # the original ceiling, unchanged

# Verbatim from BENCHMARK_RESULTS.md (and app.DEFAULT_BENCH_PROMPTS).
PROMPTS = [
    "Write a Python function that reverses a singly linked list.",
    "Explain the CAP theorem in two sentences.",
    "What is 17 times 24?",
    "Summarize the plot of Romeo and Juliet in one short paragraph.",
    "Fix this bug: `def add(a, b): return a - b`",
]
# The 2026-09-16 baseline, for the comparison column.
ORIGINAL = [45.79, 45.35, 47.15, 47.36, 45.30]

ARMS = [("current", tuple(Runtime().spec_args)), ("current, no spec", ())]


def run_arm(spec):
    path, start = BIG_MODELS[BIG_MODEL]
    rt = replace(Runtime(), n_cpu_moe=start, spec_args=tuple(spec))
    srv = BigModelServer(replace(Paths(), big_model_gguf=path), rt, port=8109)
    try:
        out = {"n_cpu_moe": srv.n_cpu_moe, "per_prompt": []}
        for p in PROMPTS:
            t = {}
            for ev in srv.stream_chat([{"role": "user", "content": p}], MAX_TOKENS):
                t = ev.get("timings", t)
            out["per_prompt"].append({
                "prompt": p,
                "tokens": t.get("predicted_n", 0),
                "seconds": round(t.get("predicted_ms", 0.0) / 1000, 2),
                "tps": 1000 * t.get("predicted_n", 0) / max(t.get("predicted_ms", 1), 1),
            })
        return out
    finally:
        srv.stop()


def main():
    out_path = os.path.join(AI2, "experiments", "rerun_original_benchmark_result.json")
    res = {n: [] for n, _ in ARMS}
    for rnd in range(ROUNDS):
        for name, spec in (ARMS if rnd % 2 == 0 else ARMS[::-1]):
            try:
                r = run_arm(spec)
            except Exception as e:
                r = {"error": f"{type(e).__name__}: {e}"[:200]}
            res[name].append(r)
            msg = r.get("error") or ("split %d, %s" % (r["n_cpu_moe"], ", ".join(
                f"{d['tps']:.0f}" for d in r["per_prompt"])))
            print(f"  r{rnd} [{name}] {msg}", file=sys.stderr, flush=True)
            json.dump(res, open(out_path, "w"), indent=1)

    def med(name, i):
        ok = [r for r in res[name] if "error" not in r]
        return statistics.median(r["per_prompt"][i]["tps"] for r in ok) if ok else None

    ok_cur = [r for r in res["current"] if "error" not in r]
    split = ok_cur[-1]["n_cpu_moe"] if ok_cur else "?"
    print(f"\n=== THE ORIGINAL FIVE PROMPTS, RE-RUN ({BIG_MODEL}, --n-cpu-moe {split}, "
          f"{MAX_TOKENS} tokens, medians of {ROUNDS} rounds) ===\n")
    print("| Prompt | 2026-09-16 tok/s | now | no spec | speedup |")
    print("|---|---|---|---|---|")
    for i, p in enumerate(PROMPTS):
        cur, nos = med("current", i), med("current, no spec", i)
        short = (p[:44] + "...") if len(p) > 47 else p
        print("| `%s` | %.2f | **%.1f** | %.1f | **%.1fx** |"
              % (short, ORIGINAL[i], cur or 0, nos or 0, (cur or 0) / ORIGINAL[i]))
    cur_all = [med("current", i) for i in range(len(PROMPTS))]
    nos_all = [med("current, no spec", i) for i in range(len(PROMPTS))]
    if all(cur_all):
        a, b = statistics.mean(cur_all), statistics.mean(nos_all or [0])
        print("| **average** | **%.2f** | **%.1f** | %.1f | **%.1fx** |"
              % (statistics.mean(ORIGINAL), a, b, a / statistics.mean(ORIGINAL)))
        print(f"\nSpeculation on these five prompts: {a / b:.2f}x "
              "(they are write-from-scratch prompts, where n-gram drafting has "
              "little to repeat -- see experiments/ngram_q2k.py for edit prompts)")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
