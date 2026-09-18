"""
A real draft model, re-tested for the same reason n-gram drafting started winning.

SPEC_DECODING.md ruled out Qwen3-0.6B as a draft at Q4_K_M: 0.91x at best, 61%
acceptance against a 75% break-even. The arithmetic behind that bar was
`c(k)/c(1)` -- what it costs the target to check k tokens at once -- measured when
20+ layers of experts were in RAM, so every drafted token dragged a wider batch
through the CPU. Q2_K leaves 2-4 layers there, and ngram_q2k.py duly found
n-gram drafting is now worth 1.22x on code edits. So the draft-model question is
open again, and this measures it instead of assuming either way.

What is different this time, and what isn't:
  + checking k drafted tokens is now a GPU batch, not a CPU one
  + the draft can be checked together with n-gram drafting (--spec-type takes a
    comma-separated list), so both are tried, separately and together
  - the draft model still costs VRAM: Qwen3-0.6B Q4_K_M plus its own KV cache is
    ~0.7-1.0 GB, which is 3-5 layers of Q2_K experts pushed back to the CPU.
    That is why every arm here is measured at the SAME --n-cpu-moe as its
    baseline, and the arms that load a draft are also reported at the split they
    would really get (--n-cpu-moe + 4), so the cost is not hidden.

Prompt sets from ngram_q2k.py: `edit` (paste code, ask for a modified copy --
where drafting pays) and `fresh` (write/explain from scratch -- where it must not
cost anything). Greedy, repeat_penalty 1.0, 512 tokens, 2 interleaved rounds.
Acceptance and draft timings come from llama-server's own log.
"""
import importlib.util
import json
import os
import statistics
import sys

AI2 = "/home/everett/AI2"
sys.path.insert(0, AI2)
_spec = importlib.util.spec_from_file_location("ngram", os.path.join(AI2, "experiments", "ngram_q2k.py"))
ng = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ng)

DRAFT = os.path.join(AI2, "Model Training", "Qwen3-0.6B-Q4_K_M.gguf")
NGRAM = ["--spec-type", "ngram-mod", "--spec-ngram-mod-n-match", "12",
         "--spec-ngram-mod-n-min", "1", "--spec-ngram-mod-n-max", "16"]
SPLIT = int(os.environ.get("SPLIT", "4"))
DRAFT_SPLIT = SPLIT + int(os.environ.get("DRAFT_EXTRA_SPLIT", "4"))   # the VRAM the draft takes back


def draft_args(n_max, p_min, with_ngram=False, ctx=2048):
    t = "draft-simple,ngram-mod" if with_ngram else "draft-simple"
    args = ["--spec-type", t, "-md", DRAFT, "-ngld", "99", "--spec-draft-n-max", str(n_max),
            "--spec-draft-p-min", str(p_min), "-cd", str(ctx)]
    return args + (NGRAM[2:] if with_ngram else [])


# (name, extra args, --n-cpu-moe for this arm)
ARMS = [
    ("baseline (no spec)", [], SPLIT),
    ("ngram only", NGRAM, SPLIT),
    ("draft n3 p0.0", draft_args(3, 0.0), DRAFT_SPLIT),
    ("draft n5 p0.0", draft_args(5, 0.0), DRAFT_SPLIT),
    ("draft n5 p0.6", draft_args(5, 0.6), DRAFT_SPLIT),
    ("draft n8 p0.8", draft_args(8, 0.8), DRAFT_SPLIT),
    ("draft n5 + ngram", draft_args(5, 0.6, with_ngram=True), DRAFT_SPLIT),
    ("ngram only, draft's split", NGRAM, DRAFT_SPLIT),   # what the draft's VRAM costs, alone
]
ROUNDS = int(os.environ.get("ROUNDS", "2"))


def main():
    out_path = os.path.join(AI2, "experiments", "draft_model_q2k_result.json")
    res = {n: [] for n, *_ in ARMS}
    for rnd in range(ROUNDS):
        order = ARMS if rnd % 2 == 0 else ARMS[::-1]
        for name, extra, split in order:
            ng.SPLIT = str(split)
            r = ng.run(name, extra, rnd)
            r["split"] = split
            res[name].append(r)
            msg = r.get("error") or (f"split {split} edit {r['edit_tps']:.1f} fresh {r['fresh_tps']:.1f} tok/s, "
                                     f"accepted {r['accepted']}/{r['drafted']}")
            print(f"  r{rnd} [{name}] {msg}", file=sys.stderr, flush=True)
            json.dump(res, open(out_path, "w"), indent=1)

    base = [r for r in res["baseline (no spec)"] if "error" not in r]
    be = statistics.median(r["edit_tps"] for r in base)
    bf = statistics.median(r["fresh_tps"] for r in base)
    print(f"\n=== DRAFT MODEL vs N-GRAM ON Q2_K (split {SPLIT}; draft arms at {DRAFT_SPLIT}; "
          f"medians of {ROUNDS} rounds) ===")
    print("%-26s %6s %8s %8s %8s %8s %11s" % ("arm", "split", "edit", "vs base", "fresh", "vs base", "acceptance"))
    for name, _, split in ARMS:
        ok = [r for r in res[name] if "error" not in r]
        if not ok:
            print("%-26s ERROR %s" % (name, res[name][0]["error"]))
            continue
        e = statistics.median(r["edit_tps"] for r in ok)
        f = statistics.median(r["fresh_tps"] for r in ok)
        acc = sum(r["accepted"] for r in ok) / max(sum(r["drafted"] for r in ok), 1)
        print("%-26s %6d %8.1f %7.2fx %8.1f %7.2fx %10.0f%%" % (name, split, e, e / be, f, f / bf, 100 * acc))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
