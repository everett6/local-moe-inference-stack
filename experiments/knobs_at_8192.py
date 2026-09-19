"""
Re-tune the two knobs that were set at a different operating point.

`-ub 1024` and the n-gram speculation window were both chosen when 3-4 layers of
experts sat in CPU RAM and the context was 4096. Since then the fit reached
--n-cpu-moe 0 and the context doubled to 8192 (which costs 2 layers back). Both
changes move exactly what these knobs trade against:

  ubatch    is the prefill batch. It was worth +32% over the 512 default when
            prompts were short and VRAM was tight. At 8192 a long prompt is now
            *possible* at all, so prefill matters more -- but a larger ubatch
            also grows the compute buffer, which comes out of the same VRAM as
            the experts, so it can cost a layer. That trade is the point here,
            and it is why each arm reports the split it actually fitted at.
  n-gram    proposes tokens that already appeared in the context and has the
            target check them in one batch. Checking k tokens is much cheaper
            once the layers verifying them are on the GPU, so the window that
            paid at split 4 is not obviously the one that pays at split 2.

Three prompt kinds, because these knobs pull in different directions:
  edit     paste a function, ask for a modified copy -- where n-gram drafting
           pays, since the reply repeats the prompt.
  fresh    write something new -- where drafting must not cost anything.
  long     a ~5,000-token document and a question about it. This one only exists
           because of the 8192 context; at 4096 the server refused it. Prefill
           dominates its wall clock, so it is the ubatch case.

Greedy, repeat_penalty 1.0, through the app's own BigModelServer so the fit,
warm-up and request path are the ones the app uses.
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

ROUNDS = int(os.environ.get("ROUNDS", "1"))

NGRAM = ("--spec-type", "ngram-mod", "--spec-ngram-mod-n-match", "12",
         "--spec-ngram-mod-n-min", "1", "--spec-ngram-mod-n-max", "16")
WIDER = ("--spec-type", "ngram-mod", "--spec-ngram-mod-n-match", "8",
         "--spec-ngram-mod-n-min", "1", "--spec-ngram-mod-n-max", "24")

# (name, ubatch, spec args)
ARMS = [
    ("ub 512  + ngram 12/16", 512, NGRAM),
    ("ub 1024 + ngram 12/16", 1024, NGRAM),       # today's default
    ("ub 2048 + ngram 12/16", 2048, NGRAM),
    ("ub 1024 + no spec", 1024, ()),
    ("ub 1024 + ngram 8/24", 1024, WIDER),
]

_FN = '''def summarize_orders(orders):
    total = 0
    count = 0
    for order in orders:
        if order["status"] == "complete":
            total += order["amount"]
            count += 1
    average = total / count
    return {"total": total, "count": count, "average": average}
'''
EDIT = [
    "Add type hints and a docstring to this function, and return the whole file:\n\n" + _FN,
    "Rewrite this function to handle an empty list without dividing by zero. "
    "Return the complete function:\n\n" + _FN,
]
FRESH = [
    "Explain why merge sort is O(n log n) and quicksort is not, in a few paragraphs.",
    "Write a short guide to choosing between a process pool and a thread pool in Python.",
]

_PARA = ("Section {i}. Request {i} is assigned to slot {j}; a slot holds the key-value cache for "
         "one conversation. When slot {j} is reused the cache is discarded unless the new prompt "
         "shares a prefix with the old one, in which case only the remaining {k} tokens are "
         "processed. This is why a long system prompt costs little after the first turn. ")
# 64 sections, not more: at ~78 tokens each this is ~5,000 tokens, which fits 8192
# with room for the reply. 150 sections came to 11,734 and the server refused it
# -- which is what PromptTooLong is for, and it said so on the first attempt.
LONG = ["".join(_PARA.format(i=i, j=i % 7, k=100 + 13 * i) for i in range(64)) +
        "\n\nBased only on the passage above, explain when a cached prefix is discarded."]


def measure(srv, prompts, max_tokens):
    tok = ms = ptok = pms = 0.0
    texts = []
    for p in prompts:
        t, text = {}, ""
        for ev in srv.stream_chat([{"role": "user", "content": p}], max_tokens):
            text += ev.get("delta", "") or ""
            t = ev.get("timings", t)
        texts.append(text)
        tok += t.get("predicted_n", 0)
        ms += t.get("predicted_ms", 0.0)
        ptok += t.get("prompt_n", 0)
        pms += t.get("prompt_ms", 0.0)
    return (1000 * tok / ms if ms else 0.0), (1000 * ptok / pms if pms else 0.0), texts


def run_arm(ubatch, spec):
    path, start = BIG_MODELS["q2_k"]
    rt = replace(Runtime(), n_cpu_moe=start, ubatch=ubatch, batch=max(2048, ubatch), spec_args=tuple(spec))
    srv = BigModelServer(replace(Paths(), big_model_gguf=path), rt, port=8107)
    try:
        out = {"n_cpu_moe": srv.n_cpu_moe}
        out["edit_tps"], _, out["edit_texts"] = measure(srv, EDIT, 384)
        out["fresh_tps"], _, _ = measure(srv, FRESH, 384)
        out["long_tps"], out["long_prefill_tps"], _ = measure(srv, LONG, 192)
        return out
    finally:
        srv.stop()


def main():
    out_path = os.path.join(AI2, "experiments", "knobs_at_8192_result.json")
    res = {n: [] for n, *_ in ARMS}
    for rnd in range(ROUNDS):
        for name, ub, spec in (ARMS if rnd % 2 == 0 else ARMS[::-1]):
            try:
                r = run_arm(ub, spec)
            except Exception as e:
                r = {"error": f"{type(e).__name__}: {e}"[:200]}
            res[name].append(r)
            print(f"  r{rnd} [{name}] " + (r.get("error") or
                  f"split {r['n_cpu_moe']}, edit {r['edit_tps']:.1f}, fresh {r['fresh_tps']:.1f}, "
                  f"long {r['long_tps']:.1f} tok/s, prefill {r['long_prefill_tps']:.0f} tok/s"),
                  file=sys.stderr, flush=True)
            json.dump(res, open(out_path, "w"), indent=1)

    base = [r for r in res["ub 1024 + ngram 12/16"] if "error" not in r]
    b = {k: statistics.median(r[k] for r in base) for k in ("edit_tps", "fresh_tps", "long_tps",
                                                            "long_prefill_tps")} if base else {}
    print(f"\n=== KNOBS AT n_ctx 8192 (Q2_K, medians of {ROUNDS} round(s); vs today's default) ===")
    print("%-24s %6s %8s %8s %8s %8s %9s %8s" % ("arm", "split", "edit", "vs", "fresh", "long",
                                                 "prefill", "vs"))
    for name, *_ in ARMS:
        ok = [r for r in res[name] if "error" not in r]
        if not ok:
            print("%-24s ERROR %s" % (name, res[name][0]["error"]))
            continue
        m = {k: statistics.median(r[k] for r in ok) for k in b}
        print("%-24s %6d %8.1f %7.2fx %8.1f %8.1f %9.0f %7.2fx"
              % (name, ok[-1]["n_cpu_moe"], m["edit_tps"], m["edit_tps"] / b["edit_tps"],
                 m["fresh_tps"], m["long_tps"], m["long_prefill_tps"],
                 m["long_prefill_tps"] / b["long_prefill_tps"]))

    # Speculation was assumed lossless under greedy decoding -- drafted tokens are
    # only accepted when they match what the model would have chosen anyway -- so
    # every arm should give the same edit replies. They do not, and chasing that
    # is how spec_determinism.py found the assumption is wrong on this build:
    # checking a k-token batch changes floating-point reduction order, so the
    # model's own choice shifts on near ties. Reported, not asserted.
    ref = next((r["edit_texts"] for r in res["ub 1024 + no spec"] if "error" not in r), None)
    if ref:
        for name, *_ in ARMS:
            for r in res[name]:
                if "error" in r:
                    continue
                same = [a.strip() == b_.strip() for a, b_ in zip(r["edit_texts"], ref)]
                if not all(same):
                    print(f"  NOTE {name}: {same.count(False)}/{len(same)} edit replies differ from "
                          "the no-speculation run -- expected, see spec_determinism.py")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
