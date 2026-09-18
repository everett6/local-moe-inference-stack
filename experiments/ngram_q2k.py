"""
N-gram speculative decoding, re-tested now that checking drafts runs on the GPU.

ngram_sweep.py (Q4_K_M, 20-23 layers of experts in RAM) found nothing: every
arm within ~2% of baseline, several slower. Two things were against it:
  - Verifying k drafted tokens is a k-token batch through the model, and with
    half the experts in RAM that batch ran through the CPU. With Q2_K at
    split 3, 45 of 48 layers verify on the GPU, where a 4-token batch costs
    little more than 1.
  - Its prompts asked for new text. N-gram drafting proposes tokens that already
    appeared in the context, so it can only help when the reply repeats the
    prompt: editing, refactoring, adding types or docstrings to pasted code.
    That is a common request on this box, and the earlier test had none.

So this uses two prompt sets and reports them separately:
  edit    paste ~40-80 lines of code and ask for a modified version of all of it
  fresh   write/explain from scratch (must not get slower)

N-gram drafting was assumed lossless under greedy decoding: every drafted token
is checked against the model's own choice, so the output text should be
identical to the baseline's. Measured later, that is FALSE on this build --
see experiments/spec_determinism.py. Checking k drafted tokens is a k-token
batch, batch shape changes the order of floating-point reductions, and a
near-tied argmax can land either way; with speculation on, the same prompt
answered three times gives three different replies. What is true is the thing
that matters: across 414 graded problems it costs no accuracy (HumanEval
147/164 with speculation, 145/164 without; GSM8K 240/250 either way).

Q2_K, --n-cpu-moe 3, repeat_penalty 1.0 (as the app sends), 512 max tokens,
2 interleaved rounds. Draft acceptance is read from the server log.
"""
import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.request

AI2 = "/home/everett/AI2"
sys.path.insert(0, AI2)
from config import BIG_MODELS, Paths  # noqa: E402
from local_engine import _WARMUP_PROMPT  # noqa: E402

P = Paths()
MODEL = BIG_MODELS[os.environ.get("MODEL", "q2_k")][0]
SPLIT = os.environ.get("SPLIT", "3")
PORT = 8102
ROUNDS = int(os.environ.get("ROUNDS", "2"))
MAX_TOKENS = 512

ARMS = [("baseline", [])]
for m, n in ((4, 4), (8, 8), (12, 16)):
    ARMS.append((f"mod m{m} n{n}", ["--spec-type", "ngram-mod", "--spec-ngram-mod-n-match", str(m),
                                    "--spec-ngram-mod-n-min", "1", "--spec-ngram-mod-n-max", str(n)]))
for n, m in ((3, 8), (4, 16)):
    ARMS.append((f"simple n{n} m{m}", ["--spec-type", "ngram-simple", "--spec-ngram-simple-size-n", str(n),
                                       "--spec-ngram-simple-size-m", str(m)]))
    ARMS.append((f"map-k n{n} m{m}", ["--spec-type", "ngram-map-k", "--spec-ngram-map-k-size-n", str(n),
                                      "--spec-ngram-map-k-size-m", str(m)]))

_CODE_A = '''import json, os

def load(path):
    if not os.path.exists(path):
        return {}
    f = open(path)
    data = json.load(f)
    f.close()
    return data

def save(path, data):
    f = open(path, "w")
    json.dump(data, f)
    f.close()

def add_item(path, name, qty, price):
    data = load(path)
    if name in data:
        data[name]["qty"] = data[name]["qty"] + qty
    else:
        data[name] = {"qty": qty, "price": price}
    save(path, data)

def remove_item(path, name, qty):
    data = load(path)
    if name not in data:
        print("no such item")
        return
    data[name]["qty"] = data[name]["qty"] - qty
    if data[name]["qty"] <= 0:
        del data[name]
    save(path, data)

def total_value(path):
    data = load(path)
    total = 0
    for name in data:
        total = total + data[name]["qty"] * data[name]["price"]
    return total

def report(path):
    data = load(path)
    for name in sorted(data):
        print(name, data[name]["qty"], data[name]["price"])
    print("total", total_value(path))
'''
_CODE_B = '''class Matrix:
    def __init__(self, rows):
        self.rows = rows
        self.n = len(rows)
        self.m = len(rows[0])

    def add(self, other):
        out = []
        for i in range(self.n):
            row = []
            for j in range(self.m):
                row.append(self.rows[i][j] + other.rows[i][j])
            out.append(row)
        return Matrix(out)

    def mul(self, other):
        out = []
        for i in range(self.n):
            row = []
            for j in range(other.m):
                s = 0
                for k in range(self.m):
                    s += self.rows[i][k] * other.rows[k][j]
                row.append(s)
            out.append(row)
        return Matrix(out)

    def transpose(self):
        out = []
        for j in range(self.m):
            row = []
            for i in range(self.n):
                row.append(self.rows[i][j])
            out.append(row)
        return Matrix(out)
'''
EDIT = [
    f"Add type hints and a one-line docstring to every function in this code. Return the complete code.\n\n```python\n{_CODE_A}```",
    f"Rewrite this code to use `with open(...)` for every file operation. Change nothing else. Return the complete code.\n\n```python\n{_CODE_A}```",
    f"Rename the attributes `n` and `m` to `num_rows` and `num_cols` everywhere in this class. Return the complete class.\n\n```python\n{_CODE_B}```",
    f"Add input validation to `add` and `mul` that raises ValueError on mismatched shapes. Return the complete class.\n\n```python\n{_CODE_B}```",
]
FRESH = [
    "Explain how a hash map handles collisions.",
    "Write a Python class implementing a least-recently-used cache with get and put.",
    "Summarize the causes of the French Revolution in a few paragraphs.",
]
ACC_RE = re.compile(r"draft acceptance = ([0-9.]+) \(\s*(\d+) accepted /\s*(\d+) generated\)")


def post(path, payload, timeout=600):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def measure(prompts):
    tok = ms = 0.0
    texts = []
    for p in prompts:
        r = post("/v1/chat/completions", {"messages": [{"role": "user", "content": p}], "max_tokens": MAX_TOKENS,
                                          "temperature": 0, "repeat_penalty": 1.0, "cache_prompt": False})
        tok += r["timings"]["predicted_n"]
        ms += r["timings"]["predicted_ms"]
        texts.append(r["choices"][0]["message"]["content"])
    return 1000 * tok / ms, texts


def run(name, extra, rnd):
    log_path = os.path.join(AI2, "state", f"ngramq2k_{name.replace(' ', '_')}_r{rnd}.log")
    env = dict(os.environ, LD_LIBRARY_PATH=P.llama_server_ld_library_path)
    cmd = [P.llama_server_bin, "-m", MODEL, "-c", "4096", "-t", "16", "--port", str(PORT), "-ngl", "999",
           "-fa", "on", "-np", "1", "--n-cpu-moe", SPLIT] + extra
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            for _ in range(300):
                if proc.poll() is not None:
                    return {"error": "load_failed"}
                try:
                    if urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3).status == 200:
                        break
                except Exception:
                    time.sleep(1)
            post("/completion", {"prompt": _WARMUP_PROMPT, "n_predict": 8, "temperature": 0})
            edit_tps, edit_texts = measure(EDIT)
            fresh_tps, fresh_texts = measure(FRESH)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            time.sleep(2)
    acc = [(int(a), int(g)) for _, a, g in ACC_RE.findall(open(log_path).read())]
    return {"edit_tps": edit_tps, "fresh_tps": fresh_tps, "edit_texts": edit_texts, "fresh_texts": fresh_texts,
            "accepted": sum(a for a, _ in acc), "drafted": sum(g for _, g in acc)}


def main():
    out_path = os.path.join(AI2, "experiments", "ngram_q2k_result.json")
    res = {n: [] for n, _ in ARMS}
    for rnd in range(ROUNDS):
        order = ARMS if rnd % 2 == 0 else ARMS[::-1]
        for name, extra in order:
            r = run(name, extra, rnd)
            res[name].append(r)
            msg = r.get("error") or (f"edit {r['edit_tps']:.1f} fresh {r['fresh_tps']:.1f} tok/s, "
                                     f"accepted {r['accepted']}/{r['drafted']}")
            print(f"  r{rnd} [{name}] {msg}", file=sys.stderr, flush=True)
            json.dump(res, open(out_path, "w"), indent=1)

    base = [r for r in res["baseline"] if "error" not in r]
    be = statistics.median(r["edit_tps"] for r in base)
    bf = statistics.median(r["fresh_tps"] for r in base)
    print(f"\n=== N-GRAM SPECULATION ON Q2_K (split {SPLIT}, medians of {ROUNDS} rounds) ===")
    print("%-18s %8s %7s %8s %7s %10s %s" % ("arm", "edit", "vs base", "fresh", "vs base", "acceptance",
                                             "identical text (edit/fresh)"))
    for name, _ in ARMS:
        ok = [r for r in res[name] if "error" not in r]
        if not ok:
            print("%-18s ERROR %s" % (name, res[name][0]["error"]))
            continue
        e = statistics.median(r["edit_tps"] for r in ok)
        f = statistics.median(r["fresh_tps"] for r in ok)
        acc = sum(r["accepted"] for r in ok) / max(sum(r["drafted"] for r in ok), 1)
        se = sum(a == b for a, b in zip(ok[0]["edit_texts"], base[0]["edit_texts"]))
        sf = sum(a == b for a, b in zip(ok[0]["fresh_texts"], base[0]["fresh_texts"]))
        print("%-18s %8.1f %6.2fx %8.1f %6.2fx %9.0f%% %d/%d %d/%d" % (name, e, e / be, f, f / bf, 100 * acc, se,
                                                                      len(EDIT), sf, len(FRESH)))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
