"""
Phase B5: with the desktop mostly off the card, how much VRAM margin is worth keeping?

Runtime.vram_headroom_mb is free VRAM left for other programs after the model
loads. vram_contention.py settled what it is NOT for: with another process holding
the card until it OOMed, 25 MiB free, the server kept answering at full speed --
the other program gets the error, not llama-server.

What changed since: Google Remote Desktop is purged and GNOME's is disabled
(-164 MiB), so the fit reached --n-cpu-moe 1 where it was 4 earlier the same day.
At Q2_K each remaining layer is ~200 MiB and ~0.2 ms/token, so the last layer is
in reach if the margin drops. The counter-argument is that at 0 margin no other
CUDA program can even initialize (measured: PyTorch failed to allocate anything).

So: fit the model the way the app does at each margin, and measure what it costs
and what it leaves. Decode is 3 prompts x 192 tokens through stream_chat, i.e.
the app's own request path, with whatever is in Runtime (speculation, ubatch).
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

HEADROOMS = [int(h) for h in os.environ.get("HEADROOMS", "512,256,128,0").split(",")]
ROUNDS = int(os.environ.get("ROUNDS", "2"))
PROMPTS = [
    "Explain how a hash map handles collisions.",
    "Write a Python class implementing a least-recently-used cache with get and put.",
    "Summarize the causes of the French Revolution in a few paragraphs.",
]


def measure(headroom):
    path, start = BIG_MODELS["q2_k"]
    srv = BigModelServer(replace(Paths(), big_model_gguf=path),
                         replace(Runtime(), n_cpu_moe=start, vram_headroom_mb=headroom), port=8104)
    try:
        tok = ms = 0.0
        for p in PROMPTS:
            t = {}
            for ev in srv.stream_chat([{"role": "user", "content": p}], 192):
                t = ev.get("timings", t)
            tok += t.get("predicted_n", 0)
            ms += t.get("predicted_ms", 0.0)
        return {"headroom": headroom, "n_cpu_moe": srv.n_cpu_moe, "decode_tps": 1000 * tok / ms,
                "free_mb": _gpu_free_mb()}
    finally:
        srv.stop()


def main():
    res = {h: [] for h in HEADROOMS}
    out_path = os.path.join(AI2, "experiments", "headroom_recheck_result.json")
    for rnd in range(ROUNDS):
        for h in (HEADROOMS if rnd % 2 == 0 else HEADROOMS[::-1]):
            try:
                r = measure(h)
            except Exception as e:
                r = {"headroom": h, "error": f"{type(e).__name__}: {e}"[:200]}
            res[h].append(r)
            print(f"  r{rnd} headroom {h:4d}: " + (r.get("error") or
                  f"split {r['n_cpu_moe']}, {r['decode_tps']:.1f} tok/s, {r['free_mb']} MiB left"),
                  file=sys.stderr, flush=True)
            json.dump({str(k): v for k, v in res.items()}, open(out_path, "w"), indent=1)

    base = [r for r in res[HEADROOMS[0]] if "error" not in r]
    bd = statistics.median(r["decode_tps"] for r in base) if base else None
    print(f"\n=== VRAM MARGIN RE-CHECK (Q2_K, medians of {ROUNDS} rounds) ===")
    print("%8s %7s %10s %9s %9s" % ("margin", "split", "decode", "vs 512", "free MiB"))
    for h in HEADROOMS:
        ok = [r for r in res[h] if "error" not in r]
        if not ok:
            print("%8d ERROR %s" % (h, res[h][0]["error"]))
            continue
        d = statistics.median(r["decode_tps"] for r in ok)
        print("%8d %7d %10.1f %8.2fx %9d" % (h, ok[-1]["n_cpu_moe"], d, d / bd if bd else 0, ok[-1]["free_mb"]))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
