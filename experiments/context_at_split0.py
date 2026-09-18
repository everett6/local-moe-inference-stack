"""
How much context can this fit now, and what does it cost?

`Runtime.n_ctx` is 4096 and always has been -- it is llama-server's default, not
a measured choice, and it is the one setting here that nobody ever justified. It
matters more than it looks: router.prompt_bucket has a whole "long" bucket for
prompts of 80+ words, a three-turn coding conversation already spends a good part
of 4096, and the app silently truncates beyond it.

The reason to re-ask now is that the fit changed. With the remote-desktop daemons
gone and the margin at 256 MiB, Q2_K loads at --n-cpu-moe 0: all 48 layers of
experts in VRAM, 437 MiB free. KV cache comes out of that same 437 MiB, so more
context is paid for in expert layers pushed back to the CPU -- about 200 MiB and
~0.2 ms/token each. That is a real exchange rate, and it is now cheap: one layer
is ~2% of decode, where at the old split of 4 the same VRAM was worth much more.

So: fit the model the way the app does at each context size and measure what it
costs. Two decode numbers per size, because they can diverge:

  short   3 prompts x 192 tokens from a cold cache -- what most chat turns are.
  long    a ~3,000-token document then a question about it, where attention has
          a real amount of KV to read. A larger -c does not slow this down by
          itself (attention reads what is *used*, not what is reserved), but a
          layer pushed to the CPU does, so this is where the cost should show.

Prefill (prompt tokens/s) is reported too: it is what a long document actually
waits on.

Run with nothing else on the GPU. ~3 minutes per context size.
"""
import json
import os
import statistics
import sys
from dataclasses import replace

AI2 = "/home/everett/AI2"
sys.path.insert(0, AI2)
from config import BIG_MODELS, Paths, Runtime  # noqa: E402
from local_engine import BigModelServer, _gpu_free_mb  # noqa: E402

CONTEXTS = [int(c) for c in os.environ.get("CONTEXTS", "4096,8192,16384,32768").split(",")]
ROUNDS = int(os.environ.get("ROUNDS", "2"))

SHORT = [
    "Explain how a hash map handles collisions.",
    "Write a Python class implementing a least-recently-used cache with get and put.",
    "Summarize the causes of the French Revolution in a few paragraphs.",
]

# ~2,800 tokens of plain prose with a question at the end: it has to fit the
# smallest context tested (4096) alongside 192 generated tokens. Built rather
# than pasted so the file stays readable and the token count is stable.
# Each section is numbered and its numbers differ, so the passage is not the
# same string twenty-six times -- repeated text is exactly what the n-gram drafter
# predicts for free, which would flatter decode here and nowhere else.
_PARA = (
    "Section {i}. The scheduler assigns request {i} to slot {j}, and a slot holds the key-value "
    "cache for one conversation. When slot {j} is reused the cache is discarded unless the new "
    "prompt shares a prefix with the old one, in which case the shared prefix is kept and only "
    "the remaining {k} tokens are processed. This is why a long system prompt costs little after "
    "the first turn, and why changing a single early token costs the whole prompt again. "
)
LONG_DOC = "".join(_PARA.format(i=i, j=i % 7, k=100 + 13 * i) for i in range(26))


def long_prompt():
    return (LONG_DOC + "\n\nBased only on the passage above, explain in your own words when a "
            "cached prefix is discarded and why an early edit is expensive.")


def measure(n_ctx):
    path, start = BIG_MODELS["q2_k"]
    srv = BigModelServer(replace(Paths(), big_model_gguf=path),
                         replace(Runtime(), n_ctx=n_ctx, n_cpu_moe=start), port=8105)
    try:
        out = {"n_ctx": n_ctx, "n_cpu_moe": srv.n_cpu_moe}
        tok = ms = 0.0
        for p in SHORT:
            t = {}
            for ev in srv.stream_chat([{"role": "user", "content": p}], 192):
                t = ev.get("timings", t)
            tok += t.get("predicted_n", 0)
            ms += t.get("predicted_ms", 0.0)
        out["short_tps"] = 1000 * tok / ms

        t = {}
        for ev in srv.stream_chat([{"role": "user", "content": long_prompt()}], 192):
            t = ev.get("timings", t)
        out["long_tps"] = 1000 * t.get("predicted_n", 0) / max(t.get("predicted_ms", 1), 1)
        out["prompt_n"] = t.get("prompt_n", 0)
        out["prefill_tps"] = 1000 * t.get("prompt_n", 0) / max(t.get("prompt_ms", 1), 1)
        out["free_mb"] = _gpu_free_mb()
        return out
    finally:
        srv.stop()


def main():
    res = {c: [] for c in CONTEXTS}
    out_path = os.path.join(AI2, "experiments", "context_at_split0_result.json")
    for rnd in range(ROUNDS):
        for c in (CONTEXTS if rnd % 2 == 0 else CONTEXTS[::-1]):
            try:
                r = measure(c)
            except Exception as e:
                r = {"n_ctx": c, "error": f"{type(e).__name__}: {e}"[:200]}
            res[c].append(r)
            print(f"  r{rnd} -c {c:6d}: " + (r.get("error") or
                  f"split {r['n_cpu_moe']}, short {r['short_tps']:.1f}, long {r['long_tps']:.1f} tok/s, "
                  f"prefill {r['prefill_tps']:.0f} tok/s on {r['prompt_n']} tokens, {r['free_mb']} MiB left"),
                  file=sys.stderr, flush=True)
            json.dump({str(k): v for k, v in res.items()}, open(out_path, "w"), indent=1)

    base = [r for r in res[CONTEXTS[0]] if "error" not in r]
    bs = statistics.median(r["short_tps"] for r in base) if base else None
    bl = statistics.median(r["long_tps"] for r in base) if base else None
    print(f"\n=== CONTEXT SIZE AT THE NEW FIT (Q2_K, medians of {ROUNDS} rounds) ===")
    print("%8s %6s %9s %8s %9s %8s %9s %9s" % ("-c", "split", "short", "vs 4096", "long",
                                               "vs 4096", "prefill", "free MiB"))
    for c in CONTEXTS:
        ok = [r for r in res[c] if "error" not in r]
        if not ok:
            print("%8d ERROR %s" % (c, res[c][0]["error"]))
            continue
        s = statistics.median(r["short_tps"] for r in ok)
        l = statistics.median(r["long_tps"] for r in ok)
        p = statistics.median(r["prefill_tps"] for r in ok)
        print("%8d %6d %9.1f %7.2fx %8.1f %7.2fx %8.0f %9d"
              % (c, ok[-1]["n_cpu_moe"], s, s / bs if bs else 0, l, l / bl if bl else 0, p, ok[-1]["free_mb"]))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
