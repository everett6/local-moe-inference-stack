"""
Head-to-head: which speculative decoding strategy actually wins on the real 30B?

experiments/spec_headroom.py established that batching is NOT free but is far
from useless on this box: verifying 4 positions in one target forward costs
13.95 ms/token vs 22.19 ms/token at width 1, so a *perfect* draft would give
1.59x at width 4 and 2.34x at width 8. That is the budget. This experiment finds
out who spends it best.

Three strategies, all against the production target (Qwen3-30B-A3B-Instruct-2507
Q4_K_M, experts pinned to CPU):

  draft-simple + Qwen2.5-Coder-0.5B-Q8_0
      A real 0.5B model drafts. BENCHMARK_RESULTS.md measured 33-80% acceptance
      but only 1.02x overall -- the draft's own forward passes eat the winnings.
      Retested here across n-max, which that benchmark never swept.

  draft-eagle3 + qwen3-30b-a3b-eagle3.gguf
      The EAGLE3 head. EAGLE3_DRAFT_MODEL.md measured 10-15% acceptance and a
      3-6x SLOWDOWN. The open question was whether self-training the head would
      rescue it.

  ngram-*  (no draft model at all)
      Draft by looking up what this same context produced earlier in the
      generation. Costs essentially zero -- there is no second model to run --
      so every accepted token is pure profit. Nobody proposed this; it showed up
      in `llama-server --help` for this build. For code and structured text,
      where the model repeats identifiers and boilerplate, hit rates can be
      high.

The comparison that matters is not acceptance rate, it is tok/s. A strategy with
low acceptance but zero draft cost can beat one with high acceptance that pays
for a second model on every step.

Acceptance is read from llama-server's own log (server-context.cpp:677), not
recomputed.
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

AI2 = "/home/everett/AI2"
BACKEND = "/home/everett/.lmstudio/extensions/backends/llama.cpp-linux-x86_64-nvidia-cuda12-avx2-2.37.0"
VENDOR = "/home/everett/.lmstudio/extensions/backends/vendor/linux-llama-cuda12-vendor-v1"
SERVER = os.path.join(BACKEND, "llama-server")

BIG = "/home/everett/.lmstudio/models/lmstudio-community/Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"
CODER_DRAFT = os.path.join(AI2, "Model Training", "Qwen2.5-Coder-0.5B-Instruct-Q8_0.gguf")
EAGLE_DRAFT = os.path.join(AI2, "Model Training", "qwen3-30b-a3b-eagle3.gguf")
# Qwen3-0.6B, not Qwen2.5-Coder-0.5B: see SPEC_DECODING.md. The coder draft is a
# Qwen2.5 model and llama-server refuses it against a Qwen3 target ("the target
# and draft vocabs are not compatible"), then keeps serving with speculation
# silently off. Qwen3-0.6B shares the target's tokenizer, so it actually runs.
Q3_DRAFT_Q8 = os.path.join(AI2, "Model Training", "Qwen3-0.6B-Q8_0.gguf")
Q3_DRAFT_Q4 = os.path.join(AI2, "Model Training", "Qwen3-0.6B-Q4_K_M.gguf")
PORT = 8097

# Same prompt mix as BENCHMARK_RESULTS.md so numbers are comparable to it, with
# the code-heavy ones kept: ngram methods should do best exactly there, and a
# prompt set that hid that would be a rigged comparison.
PROMPTS = [
    "Write a Python function that reverses a singly linked list.",
    "Explain the CAP theorem in two sentences.",
    "Write a Python class implementing a least-recently-used cache with get and put.",
    "Summarize the plot of Romeo and Juliet in one short paragraph.",
    "Refactor this into idiomatic Python:\n"
    "def f(l):\n  r = []\n  for i in range(len(l)):\n    if l[i] % 2 == 0:\n      r.append(l[i]*2)\n  return r",
]
MAX_TOKENS = 192

ACC_RE = re.compile(
    r"draft acceptance = ([0-9.]+) \(\s*(\d+) accepted /\s*(\d+) generated\), mean len =\s*([0-9.]+)"
)

# name -> extra llama-server args. Production target args are added to all of them.
ARMS = [
    ("baseline", []),
    # Kept as a regression test: this arm MUST come back as an error, not a number.
    ("coder0.5b_n4", ["--spec-type", "draft-simple", "-md", CODER_DRAFT, "-ngld", "999", "--spec-draft-n-max", "4"]),
    ("qwen3_0.6b_q8_n2", ["--spec-type", "draft-simple", "-md", Q3_DRAFT_Q8, "-ngld", "999", "--spec-draft-n-max", "2"]),
    ("qwen3_0.6b_q8_n4", ["--spec-type", "draft-simple", "-md", Q3_DRAFT_Q8, "-ngld", "999", "--spec-draft-n-max", "4"]),
    ("qwen3_0.6b_q8_n6", ["--spec-type", "draft-simple", "-md", Q3_DRAFT_Q8, "-ngld", "999", "--spec-draft-n-max", "6"]),
    ("qwen3_0.6b_q8_n8", ["--spec-type", "draft-simple", "-md", Q3_DRAFT_Q8, "-ngld", "999", "--spec-draft-n-max", "8"]),
    ("qwen3_0.6b_q4_n4", ["--spec-type", "draft-simple", "-md", Q3_DRAFT_Q4, "-ngld", "999", "--spec-draft-n-max", "4"]),
    ("eagle3_n2", ["--spec-type", "draft-eagle3", "-md", EAGLE_DRAFT, "-ngld", "999", "--spec-draft-n-max", "2"]),
    ("eagle3_n4", ["--spec-type", "draft-eagle3", "-md", EAGLE_DRAFT, "-ngld", "999", "--spec-draft-n-max", "4"]),
    ("ngram_mod_n4", ["--spec-type", "ngram-mod", "--spec-ngram-mod-n-max", "4"]),
    ("ngram_mod_n8", ["--spec-type", "ngram-mod", "--spec-ngram-mod-n-max", "8"]),
    ("ngram_map_k4v", ["--spec-type", "ngram-map-k4v"]),
    ("ngram_simple", ["--spec-type", "ngram-simple"]),
    ("ngram_cache", ["--spec-type", "ngram-cache"]),
]


def wait_healthy(timeout=600):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(2)
    return False


def run_arm(name, extra):
    log_path = os.path.join(AI2, "state", f"shootout_{name}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    cmd = [SERVER, "-m", BIG, "--port", str(PORT),
           "-ngl", "999", "-ot", "ffn_(gate|down|up)_exps=CPU",
           "-c", "4096", "-fa", "on", "-t", "16", "--no-warmup"] + extra

    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{VENDOR}:{BACKEND}"

    with open(log_path, "w") as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
        try:
            if not wait_healthy():
                proc.kill()
                return {"arm": name, "error": "server never became healthy", "log": log_path}

            total_tokens = 0
            total_elapsed = 0.0
            per_prompt = []
            for p in PROMPTS:
                body = json.dumps({
                    "prompt": p,
                    "n_predict": MAX_TOKENS,
                    "temperature": 0,     # greedy everywhere, so arms differ only in strategy
                    "cache_prompt": False,
                }).encode()
                req = urllib.request.Request(
                    f"http://127.0.0.1:{PORT}/completion", data=body,
                    headers={"Content-Type": "application/json"})
                t0 = time.perf_counter()
                with urllib.request.urlopen(req, timeout=900) as r:
                    out = json.loads(r.read())
                elapsed = time.perf_counter() - t0
                n = out.get("tokens_predicted", 0)
                total_tokens += n
                total_elapsed += elapsed
                per_prompt.append({"tokens": n, "s": round(elapsed, 2)})
            print(f"  [{name}] {total_tokens} tok in {total_elapsed:.2f}s = "
                  f"{total_tokens/total_elapsed:.2f} tok/s", file=sys.stderr)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)

    # llama-server does NOT exit when a draft model is rejected -- it logs an
    # error and serves without speculation. An arm that fell back like that looks
    # exactly like the baseline and would otherwise be reported as a real
    # measurement. This is how BENCHMARK_RESULTS.md came to record a "native
    # speculative decoding = 1.02x" number that was actually plain generation.
    spec_failed = None
    acc = None
    with open(log_path) as f:
        for line in f:
            if ("failed to initialize speculative decoding" in line
                    or "vocabs are not compatible" in line):
                spec_failed = line.strip()
            m = ACC_RE.search(line)
            if m:
                acc = {"rate": float(m.group(1)), "accepted": int(m.group(2)),
                       "generated": int(m.group(3)), "mean_len": float(m.group(4))}

    if extra and spec_failed:
        return {"arm": name, "error": "speculation did not engage: " + spec_failed,
                "log": log_path}
    if extra and acc is None and not name.startswith("baseline"):
        return {"arm": name, "error": "no acceptance line logged -- speculation likely inactive",
                "log": log_path}

    return {"arm": name, "tokens": total_tokens, "elapsed_s": round(total_elapsed, 2),
            "tok_s": round(total_tokens / total_elapsed, 2) if total_elapsed else 0.0,
            "acceptance": acc, "per_prompt": per_prompt, "log": log_path}


def main():
    only = sys.argv[1:] or None
    needed = {BIG}
    for name, extra in ARMS:
        if only and name not in only:
            continue
        for i, a in enumerate(extra):
            if a == "-md":
                needed.add(extra[i + 1])
    for p in sorted(needed):
        if not os.path.exists(p):
            print(f"missing: {p}", file=sys.stderr)
            sys.exit(1)

    out_path = os.path.join(AI2, "experiments", "spec_shootout_result.json")
    results = []
    if os.path.exists(out_path) and only:
        # Allow re-running a subset without losing the arms already measured.
        with open(out_path) as f:
            results = [r for r in json.load(f) if r["arm"] not in only]

    for name, extra in ARMS:
        if only and name not in only:
            continue
        print(f"=== {name} ===", file=sys.stderr)
        results.append(run_arm(name, extra))
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    base = next((r["tok_s"] for r in results if r["arm"] == "baseline" and "tok_s" in r), None)
    print("\n=== SHOOTOUT ===")
    print("%-16s %8s %8s %8s %9s" % ("arm", "tok/s", "vs base", "accept", "mean len"))
    for r in sorted(results, key=lambda x: -x.get("tok_s", 0)):
        if "error" in r:
            print("%-16s ERROR: %s" % (r["arm"], r["error"]))
            continue
        acc = r.get("acceptance")
        acc_s = ("%.1f%%" % (acc["rate"] * 100)) if acc else "--"
        len_s = ("%.2f" % acc["mean_len"]) if acc else "--"
        ratio = ("%.2fx" % (r["tok_s"] / base)) if base else "--"
        print("%-16s %8.2f %8s %8s %9s" % (r["arm"], r["tok_s"], ratio, acc_s, len_s))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
