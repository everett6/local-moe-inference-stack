"""
Is n-gram speculation actually lossless? (Measured: no, and the repo said it was.)

ngram_q2k.py states "N-gram drafting is lossless under greedy decoding: every
drafted token is checked against the model's own choice, so output text must be
identical to the baseline's", and PLAN.md's speed table repeats it as "same
tokens, checked". The argument is right in principle and wrong in practice, and
knobs_at_8192.py caught it: every speculated arm produced different edit replies
from the unspeculated one.

The confound to rule out first is that the model is simply non-deterministic. So
this asks the narrower question the claim depends on: does ONE server, with ONE
setting, answer the SAME prompt the same way three times?

Result (Q2_K, greedy, repeat_penalty 1.0, 384 tokens):

    no speculation      3 requests -> 1 distinct reply   deterministic
    n-gram speculation  3 requests -> 3 distinct replies  NOT deterministic

So the non-determinism arrives with speculation, not with the model. Verifying k
drafted tokens is a k-token batch, batch shape decides the order of floating-
point reductions in the matmuls, and a near-tied argmax can land either way. The
draft is checked against the model's choice, but the model's choice is itself
computed slightly differently when it is checking a batch.

The difference is not always cosmetic. In the run below the unspeculated reply
guarded the division and the speculated one did not:

    no spec:  average = total / count if count > 0 else 0.0
    spec:     average = total / count

That is the case for measuring quality rather than asserting it -- and it has
been measured, on 414 graded problems, in model_quality_eval_result.json (with
speculation, the shipped setting) against ..._nospec.json:

    Q2_K   HumanEval 147/164 with, 145/164 without.  GSM8K 240/250 both.

i.e. no systematic quality cost, and the difference that does exist favours
speculation. So the setting stays; the claim that it is bit-identical does not.
"""
import sys
sys.path.insert(0, "/home/everett/AI2")
from dataclasses import replace
from config import BIG_MODELS, Paths, Runtime
from local_engine import BigModelServer

FN = ('def summarize_orders(orders):\n    total = 0\n    count = 0\n    for order in orders:\n'
      '        if order["status"] == "complete":\n            total += order["amount"]\n'
      '            count += 1\n    average = total / count\n'
      '    return {"total": total, "count": count, "average": average}\n')
P = "Add type hints and a docstring to this function, and return the whole file:\n\n" + FN

def texts(spec, n=3):
    path, start = BIG_MODELS["q2_k"]
    rt = replace(Runtime(), n_cpu_moe=start, spec_args=tuple(spec))
    srv = BigModelServer(replace(Paths(), big_model_gguf=path), rt, port=8108)
    out = []
    try:
        for _ in range(n):
            t = ""
            for ev in srv.stream_chat([{"role": "user", "content": P}], 384):
                t += ev.get("delta", "") or ""
            out.append(t)
    finally:
        srv.stop()
    return out

NONE = ()
NGRAM = ("--spec-type", "ngram-mod", "--spec-ngram-mod-n-match", "12",
         "--spec-ngram-mod-n-min", "1", "--spec-ngram-mod-n-max", "16")
a = texts(NONE)
b = texts(NGRAM)
print("\n=== same server, no speculation, 3 identical requests ===")
print("  all three identical:", len(set(a)) == 1, f"({len(set(a))} distinct)")
print("=== same server, n-gram speculation, 3 identical requests ===")
print("  all three identical:", len(set(b)) == 1, f"({len(set(b))} distinct)")
print("=== no-spec vs spec ===")
print("  identical:", a[0].strip() == b[0].strip())
if a[0].strip() != b[0].strip():
    for i, (x, y) in enumerate(zip(a[0], b[0])):
        if x != y:
            print(f"  first difference at char {i}: {a[0][max(0,i-60):i+25]!r}")
            print(f"                        vs      {b[0][max(0,i-60):i+25]!r}")
            break
