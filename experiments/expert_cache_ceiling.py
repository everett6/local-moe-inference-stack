"""
Phase E go/no-go: what is the most an expert cache could possibly win here?

llama.cpp PR #27861 (`--moe-expert-cache`) keeps a GPU-resident LRU cache of the
experts for host-offloaded layers. The cost of trying it is real -- NVIDIA's CUDA
12.8+/13.x toolkit (root, several GB, since Ubuntu ships 12.4 and this card is
sm_120), building llama.cpp from source, applying a draft PR, and maintaining a
custom binary in place of the vendored one. So the question is worth asking
numerically before anyone installs anything.

An expert cache pays *only* in proportion to the layers that are host-resident:
a layer whose experts already live in VRAM has nothing to cache. So its ceiling
is exactly the decode time currently spent on CPU-side expert layers, and that
is measurable without building anything.

Method, per model: fit the model the way the app does, then re-measure at
`--n-cpu-moe` + STEP. The difference gives ms/token per host layer at that
quantization. The ceiling for a perfect cache is then:

    ceiling_tps = 1000 / (ms_per_token_now - fitted_split * ms_per_layer)

i.e. what decode would be if every host-resident layer became free. A real cache
reaches some fraction of that (the PR reports +11.7% to +40%; FATE claims 1.91x
against a naive-offload baseline, not against a fitted split like ours).

This deliberately measures the *upper bound*. If the upper bound is small, no
implementation detail can rescue it.
"""
import json
import os
import statistics
import sys
from dataclasses import replace

AI2 = "/home/everett/AI2"
sys.path.insert(0, AI2)
from config import BIG_MODELS, Paths, Runtime  # noqa: E402
from local_engine import BigModelServer  # noqa: E402

MODELS = [m for m in os.environ.get("MODELS", "q2_k,ud-q3_k_xl,q4_k_m").split(",") if m]
STEP = int(os.environ.get("STEP", "4"))
ROUNDS = int(os.environ.get("ROUNDS", "1"))
PROMPTS = [
    "Explain how a hash map handles collisions.",
    "Write a Python class implementing a least-recently-used cache with get and put.",
    "Summarize the causes of the French Revolution in a few paragraphs.",
]


def decode_at(model, split=None):
    """Decode tok/s at a given --n-cpu-moe (None = let the app fit it)."""
    path, start = BIG_MODELS[model]
    rt = replace(Runtime(), n_cpu_moe=start if split is None else split)
    srv = BigModelServer(replace(Paths(), big_model_gguf=path), rt, port=8110)
    try:
        tok = ms = 0.0
        for p in PROMPTS:
            t = {}
            for ev in srv.stream_chat([{"role": "user", "content": p}], 192):
                t = ev.get("timings", t)
            tok += t.get("predicted_n", 0)
            ms += t.get("predicted_ms", 0.0)
        return srv.n_cpu_moe, 1000 * tok / ms
    finally:
        srv.stop()


def main():
    out_path = os.path.join(AI2, "experiments", "expert_cache_ceiling_result.json")
    res = {}
    for model in MODELS:
        runs = []
        for rnd in range(ROUNDS):
            try:
                fitted, tps_fitted = decode_at(model)
                # Same model, more layers on the CPU: isolates the per-layer cost.
                deeper, tps_deeper = decode_at(model, split=min(fitted + STEP, 48))
                runs.append({"fitted": fitted, "tps_fitted": tps_fitted,
                             "deeper": deeper, "tps_deeper": tps_deeper})
                print(f"  [{model}] r{rnd} split {fitted}: {tps_fitted:.1f} tok/s | "
                      f"split {deeper}: {tps_deeper:.1f} tok/s", file=sys.stderr, flush=True)
            except Exception as e:
                runs.append({"error": f"{type(e).__name__}: {e}"[:200]})
                print(f"  [{model}] r{rnd} ERROR {runs[-1]['error']}", file=sys.stderr, flush=True)
            res[model] = runs
            json.dump(res, open(out_path, "w"), indent=1)

    print("\n=== CEILING FOR AN EXPERT CACHE (per model, at the current config) ===")
    print("%-12s %7s %10s %11s %12s %11s" % ("model", "split", "decode", "ms/layer",
                                             "ceiling", "max gain"))
    for model in MODELS:
        ok = [r for r in res.get(model, []) if "error" not in r]
        if not ok:
            print("%-12s ERROR" % model)
            continue
        fitted = ok[-1]["fitted"]
        tps = statistics.median(r["tps_fitted"] for r in ok)
        tps_d = statistics.median(r["tps_deeper"] for r in ok)
        deeper = ok[-1]["deeper"]
        ms_now, ms_deep = 1000 / tps, 1000 / tps_d
        n_extra = deeper - fitted
        ms_layer = (ms_deep - ms_now) / n_extra if n_extra else 0.0
        # Every host-resident layer made free -- the unreachable best case.
        ms_ceiling = ms_now - fitted * ms_layer
        ceiling = 1000 / ms_ceiling if ms_ceiling > 0 else float("inf")
        print("%-12s %7d %10.1f %11.3f %12.1f %10.0f%%"
              % (model, fitted, tps, ms_layer, ceiling, 100 * (ceiling / tps - 1)))
    print("\nCeiling = decode if EVERY host-resident expert layer cost nothing.")
    print("A real cache reaches a fraction of it; PR #27861 reports +11.7% to +40%.")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
