"""
Which quantization should the app serve? Code and math accuracy, at the app's
exact settings, with enough questions to tell the models apart.

quant_speed_quality.py scored 50 GSM8K questions: Q4_K_M 50, Q2_K 47-49. That
gap is inside the noise, and GSM8K says nothing about code, the main use of this
box. So:

  HumanEval   all 164 problems (openai/openai_humaneval). The model gets the
              function signature + docstring and returns the function in a
              ```python block; the block runs against the problem's own unit
              tests. pass@1, greedy. Each program runs under bubblewrap: read-only
              filesystem, no network, private /tmp, 10 s timeout.
  GSM8K       first 250 test questions, 1024 max tokens, graded by the reply's
              boxed/"answer" number (quant_speed_quality.extract_answer).

Models: Q4_K_M (reference), UD-Q3_K_XL (half Q2_K's KL divergence, 1.35x) and
Q2_K (the one that reaches 2x). IQ3_XXS and UD-IQ2_XXS are left out: each is
beaten on both speed and KL divergence by one of these.

Settings are the app's: Runtime defaults (repeat_penalty, vram_headroom_mb),
BigModelServer's launch fitting from config.BIG_MODELS, stream_chat. Decode
tok/s is summed over every reply, so it is the speed of a real mixed workload.

Decision rule, fixed before the run: Q2_K is the default if it is within 5
percentage points of Q4_K_M on BOTH HumanEval and GSM8K. Otherwise the fastest
model that is within 5 points on both. Paired comparison against Q4_K_M: the
questions only one of the two got right, with an exact two-sided McNemar p-value.
"""
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import replace

AI2 = "/home/everett/AI2"
sys.path.insert(0, AI2)
from config import BIG_MODELS, Paths, Runtime  # noqa: E402
from local_engine import BigModelServer  # noqa: E402

_spec = importlib.util.spec_from_file_location("qsq", os.path.join(AI2, "experiments", "quant_speed_quality.py"))
qsq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qsq)

MODELS = [m for m in os.environ.get("MODELS", "q4_k_m,ud-q3_k_xl,q2_k").split(",") if m]
QDIR = os.path.join(AI2, "data", "quality")
HUMANEVAL = [json.loads(l) for l in open(os.path.join(QDIR, "humaneval.jsonl"))]
GSM8K = [json.loads(l) for l in open(os.path.join(QDIR, "gsm8k_test_first250.jsonl"))]
OUT = os.path.join(AI2, "experiments", "model_quality_eval_result.json")
FENCE = re.compile(r"```(?:python|py|Python)?[ \t]*\n(.*?)```", re.DOTALL)
SANDBOX = ["bwrap", "--ro-bind", "/", "/", "--tmpfs", "/tmp", "--dev", "/dev", "--proc", "/proc",
           "--unshare-all", "--die-with-parent", "--new-session", "--chdir", "/tmp", "python3", "-"]


def humaneval_prompt(item):
    return ("Complete this Python function. Reply with the complete function, including its "
            "signature and any imports it needs, in a single ```python code block.\n\n"
            f"```python\n{item['prompt']}```")


def humaneval_program(item, reply):
    blocks = FENCE.findall(reply)
    code = next((b for b in blocks if f"def {item['entry_point']}" in b), blocks[0] if blocks else reply)
    if f"def {item['entry_point']}" not in code:
        code = item["prompt"] + code          # a body without its signature
    # The prompt's own imports (typing etc.) may not be repeated in the reply.
    header = "\n".join(l for l in item["prompt"].splitlines() if l.startswith(("import ", "from ")))
    return f"{header}\n{code}\n\n{item['test']}\n\ncheck({item['entry_point']})\n"


def run_sandboxed(program, timeout=10):
    try:
        r = subprocess.run(SANDBOX, input=program, capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0, r.stderr[-300:]
    except subprocess.TimeoutExpired:
        return False, "timeout"


def ask(srv, prompt, max_tokens):
    parts, t = [], {}
    for ev in srv.stream_chat([{"role": "user", "content": prompt}], max_tokens):
        parts.append(ev.get("delta", ""))
        t = ev.get("timings", t)
    return "".join(parts), t


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def evaluate(model):
    path, start = BIG_MODELS[model]
    srv = BigModelServer(replace(Paths(), big_model_gguf=path), replace(Runtime(), n_cpu_moe=start), port=8100)
    res = {"n_cpu_moe": srv.n_cpu_moe, "humaneval": [], "gsm8k": [], "tok": 0, "ms": 0.0}
    try:
        for item in HUMANEVAL:
            reply, t = ask(srv, humaneval_prompt(item), 768)
            ok, err = run_sandboxed(humaneval_program(item, reply))
            res["humaneval"].append({"task_id": item["task_id"], "ok": ok, "err": err, "reply": reply,
                                     "hit_limit": t.get("predicted_n", 0) >= 768})
            res["tok"] += t.get("predicted_n", 0)
            res["ms"] += t.get("predicted_ms", 0.0)
        print(f"  [{model}] HumanEval {sum(r['ok'] for r in res['humaneval'])}/164", file=sys.stderr, flush=True)
        for item in GSM8K:
            reply, t = ask(srv, item["question"], 1024)
            got = qsq.extract_answer(reply)
            res["gsm8k"].append({"answer": item["answer"], "got": got, "ok": qsq.is_correct(got, item["answer"]),
                                 "reply": reply, "hit_limit": t.get("predicted_n", 0) >= 1024})
            res["tok"] += t.get("predicted_n", 0)
            res["ms"] += t.get("predicted_ms", 0.0)
        print(f"  [{model}] GSM8K {sum(r['ok'] for r in res['gsm8k'])}/250", file=sys.stderr, flush=True)
    finally:
        srv.stop()
    return res


def main():
    results = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for model in MODELS:
        if model in results and os.environ.get("RERUN") != "1":
            continue
        results[model] = evaluate(model)
        json.dump(results, open(OUT, "w"), indent=1, ensure_ascii=False)

    print("\n=== MODEL QUALITY (app settings; greedy; 95% Wilson intervals) ===")
    print("%-11s %5s %8s %22s %22s %10s %10s" % ("model", "split", "tok/s", "HumanEval pass@1", "GSM8K/250",
                                                   "HE limit", "GSM limit"))
    ref = results.get("q4_k_m")
    for model in [m for m in ("q4_k_m", "ud-q3_k_xl", "iq3_xxs", "q2_k") if m in results]:
        r = results[model]
        he, gs = sum(x["ok"] for x in r["humaneval"]), sum(x["ok"] for x in r["gsm8k"])
        lo1, hi1 = wilson(he, 164)
        lo2, hi2 = wilson(gs, 250)
        print("%-11s %5s %8.1f %6.1f%% (%4.1f-%4.1f) %6.1f%% (%4.1f-%4.1f) %10d %10d" % (
            model, r["n_cpu_moe"], 1000 * r["tok"] / r["ms"], 100 * he / 164, lo1, hi1, 100 * gs / 250, lo2, hi2,
            sum(x["hit_limit"] for x in r["humaneval"]), sum(x["hit_limit"] for x in r["gsm8k"])))
        if ref is not None and model != "q4_k_m":
            for task in ("humaneval", "gsm8k"):
                b = sum(a["ok"] and not x["ok"] for a, x in zip(ref[task], r[task]))
                c = sum(x["ok"] and not a["ok"] for a, x in zip(ref[task], r[task]))
                print(f"{'':11s}   vs q4_k_m on {task}: {b} lost, {c} gained, McNemar p = {mcnemar_p(b, c):.3f}")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
