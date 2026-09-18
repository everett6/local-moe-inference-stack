"""
Do the two knobs that survived the sweep still pay when combined?

cpu_gpu_knobs.py (Q2_K, split 4, medians of 2 rounds vs 176.0 tok/s baseline):
  --cpu-range 0-15 --cpu-strict 1   181.1 decode (+3%), ranges don't overlap the
                                    baseline's -- the 7950X's 16 physical cores,
                                    no SMT siblings
  -ub 1024                          prefill 4012 vs 3004 (+34%), decode unchanged
Both are free, but they touch the same thing from different sides (a bigger
ubatch does more work per CPU dispatch), so the combination is measured rather
than assumed. 3 rounds, rotated order, run() and the workload come from
cpu_gpu_knobs.py unchanged.
"""
import importlib.util
import json
import os
import statistics
import sys

AI2 = "/home/everett/AI2"
sys.path.insert(0, AI2)
_spec = importlib.util.spec_from_file_location("knobs", os.path.join(AI2, "experiments", "cpu_gpu_knobs.py"))
knobs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(knobs)

PIN = ["--cpu-range", "0-15", "--cpu-strict", "1"]
ARMS = [
    ("baseline t16", ["-t", "16"], {}),
    ("pinned", ["-t", "16"] + PIN, {}),
    ("ubatch 1024", ["-t", "16", "-ub", "1024", "-b", "2048"], {}),
    ("pinned + ubatch 1024", ["-t", "16", "-ub", "1024", "-b", "2048"] + PIN, {}),
]
ROUNDS = int(os.environ.get("ROUNDS", "3"))


def main():
    out_path = os.path.join(AI2, "experiments", "knob_combo_ab_result.json")
    res = {n: [] for n, *_ in ARMS}
    for rnd in range(ROUNDS):
        order = ARMS[rnd % len(ARMS):] + ARMS[:rnd % len(ARMS)]
        for name, extra, env in order:
            r = knobs.run(name, extra, env, rnd)
            res[name].append(r)
            msg = r.get("error") or (f"decode {r['decode_tps']:.1f} prefill {r['prefill_tps']:.0f} "
                                     f"free {r['free_mb']} MiB")
            print(f"  r{rnd} [{name}] {msg}", file=sys.stderr, flush=True)
            json.dump(res, open(out_path, "w"), indent=1)

    base = [r for r in res["baseline t16"] if "error" not in r]
    bd = statistics.median(r["decode_tps"] for r in base)
    bp = statistics.median(r["prefill_tps"] for r in base)
    print(f"\n=== KNOB COMBINATION (Q2_K, split {knobs.SPLIT}, medians of {ROUNDS} rounds) ===")
    print("%-22s %8s %8s %15s %9s %8s %s" % ("arm", "decode", "vs base", "range", "prefill", "vs base", "same text"))
    for name, *_ in ARMS:
        ok = [r for r in res[name] if "error" not in r]
        if not ok:
            print("%-22s ERROR %s" % (name, res[name][0]["error"]))
            continue
        d = statistics.median(r["decode_tps"] for r in ok)
        p = statistics.median(r["prefill_tps"] for r in ok)
        rng = "%.1f-%.1f" % (min(r["decode_tps"] for r in ok), max(r["decode_tps"] for r in ok))
        same = sum(a == b for a, b in zip(ok[-1]["texts"], base[-1]["texts"]))
        print("%-22s %8.1f %7.2fx %15s %9.0f %7.2fx %d/3" % (name, d, d / bd, rng, p, p / bp, same))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
