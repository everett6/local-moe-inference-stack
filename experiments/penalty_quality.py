"""
Is repeat_penalty 1.1 doing anything for reply quality, or just costing speed?

request_overhead_ab.py: at Q2_K's ~190 tok/s the penalty costs 6% of decode.
It also breaks the quick path's check (the draft samples at 1.0, the 30B at
1.1). The usual reason to keep a penalty with greedy decoding is to stop the
model repeating itself in a loop, and a 2.9-bit model could be more prone to
that. So this measures the loop rate as well as correctness, for both models:

  GSM8K     first 50 test questions, 768 max tokens, graded by the model's
            boxed/"answer" number (quant_speed_quality.extract_answer). Also
            counts replies that ran into the token limit.
  code      8 "write a complete Python ..." requests, 1024 max tokens. Each
            fenced Python block must parse (ast.parse); replies that hit the
            limit are counted.
  prose     4 long-form requests, 1024 max tokens.
  loops     any reply whose last 150 characters already appeared earlier in it,
            i.e. it ended by repeating itself.

Same server per model, both penalties interleaved by task. Greedy, through
BigModelServer.stream_chat, i.e. exactly what the app sends.
"""
import ast
import importlib.util
import json
import os
import re
import sys
from dataclasses import replace

AI2 = "/home/everett/AI2"
sys.path.insert(0, AI2)
from config import BIG_MODELS, Paths, Runtime  # noqa: E402
from local_engine import BigModelServer  # noqa: E402

_spec = importlib.util.spec_from_file_location("qsq", os.path.join(AI2, "experiments", "quant_speed_quality.py"))
qsq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qsq)

MODELS = ["q2_k", "q4_k_m"]
PENALTIES = [1.1, 1.0]
CODE = [
    "Write a complete Python module implementing a thread-safe LRU cache with TTL expiry, with docstrings.",
    "Write a complete Python script that parses an Apache access log from stdin and prints the top 10 IPs, "
    "status code counts, and requests per hour.",
    "Write a Python implementation of Dijkstra's algorithm and A* search on a grid, with a small demo.",
    "Write a Python class for a bank account system with deposits, withdrawals, transfers and transaction "
    "history, plus unittest tests for it.",
    "Write a Python asyncio web crawler that respects a max depth and concurrency limit, using aiohttp.",
    "Write a Python function to convert a Markdown table into a list of dicts, and another to convert it back.",
    "Write a complete Python tokenizer and recursive-descent parser for arithmetic expressions with +, -, *, /, "
    "parentheses and unary minus, returning an AST, plus an evaluator.",
    "Write a Python dataclass-based inventory system with JSON save/load and a command-line interface using argparse.",
]
PROSE = [
    "Write a detailed essay on the history of the printing press and its effects on European society.",
    "Explain in depth how TCP congestion control works, covering slow start, AIMD, fast retransmit and modern "
    "algorithms like CUBIC and BBR.",
    "Write a long, detailed guide to planning a two-week trip to Japan on a budget.",
    "Describe the full lifecycle of a star from nebula to its possible end states, in detail.",
]
FENCE = re.compile(r"```(?:python|py)?\n(.*?)```", re.DOTALL)


def looped(text):
    tail = text[-150:]
    return len(text) > 600 and tail in text[:-150]


def reply(srv, prompt, max_tokens):
    parts, timings = [], {}
    for ev in srv.stream_chat([{"role": "user", "content": prompt}], max_tokens):
        if "delta" in ev:
            parts.append(ev["delta"])
        timings = ev.get("timings", timings)
    text = "".join(parts)
    return text, timings.get("predicted_n", 0) >= max_tokens, timings


def main():
    results = {}
    out_path = os.path.join(AI2, "experiments", "penalty_quality_result.json")
    for model in MODELS:
        path, start = BIG_MODELS[model]
        rt = Runtime()
        srv = BigModelServer(replace(Paths(), big_model_gguf=path), replace(rt, n_cpu_moe=start), port=8098)
        per = {p: {"gsm8k": [], "code": [], "prose": [], "tok": 0, "ms": 0.0} for p in PENALTIES}
        try:
            tasks = [("gsm8k", item["question"], 768, item) for item in qsq.GSM8K] + \
                    [("code", p, 1024, None) for p in CODE] + [("prose", p, 1024, None) for p in PROSE]
            for kind, prompt, limit, item in tasks:
                for pen in PENALTIES:
                    srv.rt = replace(srv.rt, repeat_penalty=pen)
                    text, hit_limit, t = reply(srv, prompt, limit)
                    per[pen]["tok"] += t.get("predicted_n", 0)
                    per[pen]["ms"] += t.get("predicted_ms", 0.0)
                    rec = {"hit_limit": hit_limit, "looped": looped(text), "reply": text}
                    if kind == "gsm8k":
                        got = qsq.extract_answer(text)
                        rec.update(answer=item["answer"], got=got, ok=qsq.is_correct(got, item["answer"]))
                    elif kind == "code":
                        blocks = FENCE.findall(text)
                        ok = bool(blocks)
                        for b in blocks:
                            try:
                                ast.parse(b)
                            except SyntaxError:
                                ok = False
                        rec.update(blocks=len(blocks), parses=ok)
                    per[pen][kind].append(rec)
            results[model] = {"n_cpu_moe": srv.n_cpu_moe, **{str(p): v for p, v in per.items()}}
        finally:
            srv.stop()
        json.dump(results, open(out_path, "w"), indent=2, ensure_ascii=False)

    print("\n=== REPEAT PENALTY: QUALITY + SPEED (greedy, app's stream_chat) ===")
    print("%-7s %-7s %6s %9s %10s %11s %9s %7s %8s" % ("model", "penalty", "split", "GSM8K/50", "GSM8K lim",
                                                        "code parses", "code lim", "loops", "tok/s"))
    for model, r in results.items():
        for p in PENALTIES:
            v = r[str(p)]
            allrec = v["gsm8k"] + v["code"] + v["prose"]
            print("%-7s %-7s %6s %9d %10d %9d/8 %9d %7d %8.1f" % (
                model, p, r["n_cpu_moe"], sum(x["ok"] for x in v["gsm8k"]), sum(x["hit_limit"] for x in v["gsm8k"]),
                sum(x["parses"] for x in v["code"]), sum(x["hit_limit"] for x in v["code"]),
                sum(x["looped"] for x in allrec), 1000 * v["tok"] / max(v["ms"], 1)))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
