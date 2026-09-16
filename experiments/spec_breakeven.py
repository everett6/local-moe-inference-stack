"""
Turn the measured batching curve into a go/no-go number for each draft-model plan.

spec_headroom.py measured what a target forward pass costs per position at each
batch width. That is the only hardware fact needed to decide whether ANY draft
model is worth building, because the speculative decoding loop is fully
determined by it:

    one round at draft length k
      target cost   = (k+1) * c(k+1)        one forward over k drafts + 1 bonus
      draft cost    = k * d                 d = draft model's own per-token time
      tokens gained = E[accepted] + 1

With per-token acceptance probability a (the standard geometric model, which is
what llama.cpp's own `mean len` measures):

      E[accepted] + 1 = sum_{i=0..k} a^i

So the round is profitable exactly when

      ((k+1)*c(k+1) + k*d) / sum_{i=0..k} a^i  <  c(1)

Solving that for `a` gives the break-even acceptance rate: the accuracy a draft
model must reach before it stops making generation slower. Everything below that
line is a project that cannot pay off no matter how well it is executed, which is
the question actually on the table -- self-train the EAGLE3 head, or invest in a
separate small draft model.

d is measured, not assumed: pass --draft-ms for the candidate draft's own
per-token decode time (0 models the ngram strategies, which run no second model).
"""
import argparse
import json
import os

AI2 = "/home/everett/AI2"
HEADROOM = [os.path.join(AI2, "experiments", "spec_headroom_result.json"),
            os.path.join(AI2, "experiments", "spec_headroom_tuned_result.json")]


def expected_len(a, k):
    """E[accepted]+1 for draft length k at per-token acceptance a."""
    return sum(a ** i for i in range(k + 1))


def breakeven_a(c1, ck1, k, d):
    """Smallest per-token acceptance that makes draft length k not a loss."""
    round_cost = (k + 1) * ck1 + k * d
    # E must exceed round_cost / c1. E is bounded by k+1, so if even a perfect
    # draft can't cover the round, no acceptance rate works.
    need = round_cost / c1
    if need > k + 1:
        return None
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if expected_len(mid, k) < need:
            lo = mid
        else:
            hi = mid
    return hi


def speedup(c1, ck1, k, d, a):
    return c1 / (((k + 1) * ck1 + k * d) / expected_len(a, k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experts_cpu")
    ap.add_argument("--draft-ms", type=float, nargs="+", default=[0.0, 2.0, 4.0, 8.0],
                    help="candidate drafts' per-token decode cost, ms")
    args = ap.parse_args()

    data = []
    for path in HEADROOM:
        if os.path.exists(path):
            with open(path) as f:
                data.extend(json.load(f))
    cfg = next((r for r in data if r["config"] == args.config), None)
    if cfg is None:
        print("no headroom data for config %r; have: %s"
              % (args.config, ", ".join(r["config"] for r in data)))
        return
    cost = {int(k): v for k, v in cfg["ms_per_token"].items()}
    c1 = cost[1]

    print(f"target config: {args.config}   baseline decode = {c1:.2f} ms/token "
          f"({1000/c1:.1f} tok/s)\n")

    print("BREAK-EVEN per-token acceptance (draft must EXCEED this to be worth running)")
    print("  rows = draft model's own per-token cost; '--' = impossible at any accuracy\n")
    ks = [k for k in sorted(cost) if k + 1 in cost and k >= 1]
    print("%-12s" % "draft ms" + "".join("%10s" % f"k={k}" for k in ks))
    for d in args.draft_ms:
        row = "%-12.1f" % d
        for k in ks:
            a = breakeven_a(c1, cost[k + 1], k, d)
            row += "%10s" % ("--" if a is None else f"{a*100:.0f}%")
        print(row)

    print("\nSPEEDUP actually delivered, if a draft hits a given acceptance rate")
    for d in args.draft_ms:
        print(f"\n  draft cost {d:.1f} ms/token")
        print("  %-10s" % "accept" + "".join("%10s" % f"k={k}" for k in ks))
        for a in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95):
            row = "  %-10s" % f"{a*100:.0f}%"
            for k in ks:
                s = speedup(c1, cost[k + 1], k, d, a)
                row += "%10s" % f"{s:.2f}x"
            print(row)


if __name__ == "__main__":
    main()
