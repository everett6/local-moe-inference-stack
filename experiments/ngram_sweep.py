"""
Make the ngram speculative strategies actually draft, then see if they win.

In spec_shootout.py every `ngram-*` arm came back at exactly baseline speed with
no `draft acceptance` line in the log at all -- they never proposed a single
token. The reason is the default lookup length: `--spec-ngram-mod-n-match 24`
means the strategy only drafts after it has matched a 24-token context it has
already seen. Inside a 192-token generation that essentially never fires.

Worth fixing rather than writing off, because ngram drafting is the only
candidate with no second model to run. spec_breakeven.py puts its break-even at
65-70% acceptance versus 83-89% for a real draft model -- the lowest bar of any
option here -- and code generation, which is what this box is mostly used for, is
exactly the repetitive text these methods are designed for.

Shorter n-match means more drafts and worse ones, so there is a real optimum to
find rather than "lower is better": every rejected draft still costs a wider
target forward pass.

Run against the tuned `--n-cpu-moe` split, since that is what ships now and the
break-even depends on it.
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
PORT = 8094
N_CPU_MOE = "20"

# Code-heavy on purpose: this is where ngram lookup should pay, and it is the
# workload the box is actually used for. Two prose prompts kept so a win that
# only exists for code is visible as such.
PROMPTS = [
    "Write a Python class implementing a least-recently-used cache with get and put.",
    "Write a Python function that reverses a singly linked list, with a docstring and type hints.",
    "Write a Python dataclass for a 2D vector with add, subtract, dot product and magnitude methods.",
    "Explain the CAP theorem in two sentences.",
    "Summarize the plot of Romeo and Juliet in one short paragraph.",
]
MAX_TOKENS = 192

ACC_RE = re.compile(
    r"draft acceptance = ([0-9.]+) \(\s*(\d+) accepted /\s*(\d+) generated\), mean len =\s*([0-9.]+)"
)

ARMS = [("baseline", [])]
for match in (4, 8, 12, 16):
    for nmax in (2, 4):
        ARMS.append((f"mod_m{match}_n{nmax}",
                     ["--spec-type", "ngram-mod",
                      "--spec-ngram-mod-n-match", str(match),
                      "--spec-ngram-mod-n-max", str(nmax)]))
for n, m in ((2, 2), (3, 4), (4, 4)):
    ARMS.append((f"simple_n{n}_m{m}",
                 ["--spec-type", "ngram-simple",
                  "--spec-ngram-simple-size-n", str(n),
                  "--spec-ngram-simple-size-m", str(m)]))
for n, m in ((2, 2), (3, 4)):
    ARMS.append((f"mapk_n{n}_m{m}",
                 ["--spec-type", "ngram-map-k",
                  "--spec-ngram-map-k-size-n", str(n),
                  "--spec-ngram-map-k-size-m", str(m)]))


def wait_healthy(proc, timeout=420):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(2)
    return False


def run_arm(name, extra):
    log_path = os.path.join(AI2, "state", f"ngram_{name}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    cmd = [SERVER, "-m", BIG, "--port", str(PORT), "-ngl", "999",
           "--n-cpu-moe", N_CPU_MOE, "-c", "4096", "-fa", "on", "-t", "16",
           "--no-warmup"] + extra
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{VENDOR}:{BACKEND}"

    with open(log_path, "w") as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
        try:
            if not wait_healthy(proc):
                proc.kill()
                return {"arm": name, "error": "server never became healthy", "log": log_path}
            total_tokens = 0
            total_elapsed = 0.0
            for p in PROMPTS:
                body = json.dumps({"prompt": p, "n_predict": MAX_TOKENS,
                                   "temperature": 0, "cache_prompt": False}).encode()
                req = urllib.request.Request(
                    f"http://127.0.0.1:{PORT}/completion", data=body,
                    headers={"Content-Type": "application/json"})
                t0 = time.perf_counter()
                with urllib.request.urlopen(req, timeout=900) as r:
                    out = json.loads(r.read())
                total_elapsed += time.perf_counter() - t0
                total_tokens += out.get("tokens_predicted", 0)
            tok_s = total_tokens / total_elapsed if total_elapsed else 0.0
            print(f"  [{name}] {tok_s:6.2f} tok/s", file=sys.stderr)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)

    acc = None
    with open(log_path) as f:
        for line in f:
            m = ACC_RE.search(line)
            if m:
                acc = {"rate": float(m.group(1)), "accepted": int(m.group(2)),
                       "generated": int(m.group(3)), "mean_len": float(m.group(4))}

    # An arm that logged no acceptance line drafted nothing -- that is the bug
    # this experiment exists to fix, so report it as such rather than as a tok/s.
    return {"arm": name, "tok_s": round(tok_s, 2), "tokens": total_tokens,
            "elapsed_s": round(total_elapsed, 2), "acceptance": acc,
            "drafted": acc is not None or not extra, "log": log_path}


def main():
    results = []
    out_path = os.path.join(AI2, "experiments", "ngram_sweep_result.json")
    for name, extra in ARMS:
        print(f"=== {name} ===", file=sys.stderr)
        results.append(run_arm(name, extra))
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    base = next((r["tok_s"] for r in results if r["arm"] == "baseline" and "tok_s" in r), None)
    print("\n=== NGRAM SWEEP (--n-cpu-moe %s) ===" % N_CPU_MOE)
    print("%-16s %9s %9s %9s %10s" % ("arm", "tok/s", "vs base", "accept", "mean len"))
    for r in sorted(results, key=lambda x: -x.get("tok_s", 0)):
        if "error" in r:
            print("%-16s ERROR %s" % (r["arm"], r["error"]))
            continue
        acc = r.get("acceptance")
        acc_s = ("%.1f%%" % (acc["rate"] * 100)) if acc else ("--" if r["drafted"] else "NO DRAFTS")
        len_s = ("%.2f" % acc["mean_len"]) if acc else "--"
        ratio = ("%.2fx" % (r["tok_s"] / base)) if base else "--"
        print("%-16s %9.2f %9s %9s %10s" % (r["arm"], r["tok_s"], ratio, acc_s, len_s))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
