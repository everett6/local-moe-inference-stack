"""
The one draft-model candidate that can actually run: Qwen3-0.6B.

spec_shootout.py showed Qwen2.5-Coder-0.5B is not a candidate at all -- it is a
Qwen2.5 model and llama-server refuses it against a Qwen3 target, then serves
with speculation silently disabled. Qwen3-0.6B shares the target's tokenizer, so
it is the real version of the question "would a small standalone draft model beat
the EAGLE3 head?".

spec_breakeven.py predicts it loses: a draft costing ~2-4 ms/token needs 74-89%
per-token acceptance to break even here, and a 0.6B drafting for a 30B typically
lands around 60-75%. This measures rather than assumes, because it is the last
unmeasured option and the prediction is close enough to the bar to be worth
checking.

Two things are measured:

  1. The draft's own decode cost `d`, by serving the 0.6B alone. `d` is the term
     spec_breakeven.py has to be given rather than derive, so measuring it turns
     the break-even table from an estimate into a number for this exact pairing.

  2. End-to-end tok/s with the 0.6B drafting for the 30B, at several draft
     lengths, at the shipped `--n-cpu-moe` split.

Note the split used here is 24, not the shipped 20: the draft model needs VRAM
too, and 20 leaves only ~700 MiB. Its baseline is measured in the same run, so
the ratio is honest -- it is just measured against a 74 tok/s baseline instead of
an 82 tok/s one.
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
DRAFT = os.path.join(AI2, "Model Training", "Qwen3-0.6B-Q4_K_M.gguf")
PORT = 8093
N_CPU_MOE = "24"   # leave VRAM for the draft model; see module docstring

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


def run(name, cmd_extra, model=BIG):
    log_path = os.path.join(AI2, "state", f"q3draft_{name}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    cmd = [SERVER, "-m", model, "--port", str(PORT), "-ngl", "999",
           "-c", "4096", "-fa", "on", "-t", "16", "--no-warmup"] + cmd_extra
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{VENDOR}:{BACKEND}"

    with open(log_path, "w") as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
        try:
            if not wait_healthy(proc):
                proc.kill()
                with open(log_path) as f:
                    tail = "".join(f.readlines()[-4:]).strip().replace("\n", " | ")
                return {"arm": name, "error": "server never became healthy",
                        "detail": tail[:400], "log": log_path}
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
            print(f"  [{name}] {tok_s:7.2f} tok/s", file=sys.stderr)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)

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

    if spec_failed:
        return {"arm": name, "error": "speculation did not engage: " + spec_failed,
                "log": log_path}
    return {"arm": name, "tok_s": round(tok_s, 2), "tokens": total_tokens,
            "elapsed_s": round(total_elapsed, 2), "acceptance": acc, "log": log_path}


def main():
    if not os.path.exists(DRAFT):
        print(f"missing draft: {DRAFT}", file=sys.stderr)
        sys.exit(1)

    results = []
    # The draft alone, to get `d` for spec_breakeven.py.
    print("=== draft alone (measures d) ===", file=sys.stderr)
    results.append(run("draft_alone", [], model=DRAFT))

    print(f"=== baseline, --n-cpu-moe {N_CPU_MOE} ===", file=sys.stderr)
    results.append(run("baseline", ["--n-cpu-moe", N_CPU_MOE]))

    for k in (1, 2, 3, 4, 6):
        print(f"=== qwen3-0.6B draft, n-max {k} ===", file=sys.stderr)
        results.append(run(f"q3draft_n{k}",
                           ["--n-cpu-moe", N_CPU_MOE, "--spec-type", "draft-simple",
                            "-md", DRAFT, "-ngld", "999", "--spec-draft-n-max", str(k)]))

    out_path = os.path.join(AI2, "experiments", "qwen3_draft_test_result.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    d_row = next((r for r in results if r["arm"] == "draft_alone" and "tok_s" in r), None)
    if d_row:
        print(f"\ndraft decode cost d = {1000/d_row['tok_s']:.2f} ms/token "
              f"({d_row['tok_s']:.1f} tok/s standalone)")
    base = next((r["tok_s"] for r in results if r["arm"] == "baseline" and "tok_s" in r), None)

    print("\n=== QWEN3-0.6B DRAFT (--n-cpu-moe %s) ===" % N_CPU_MOE)
    print("%-16s %9s %9s %9s %10s" % ("arm", "tok/s", "vs base", "accept", "mean len"))
    for r in results:
        if r["arm"] == "draft_alone":
            continue
        if "error" in r:
            print("%-16s ERROR %s" % (r["arm"], r["error"][:90]))
            continue
        acc = r.get("acceptance")
        acc_s = ("%.1f%%" % (acc["rate"] * 100)) if acc else "--"
        len_s = ("%.2f" % acc["mean_len"]) if acc else "--"
        ratio = ("%.2fx" % (r["tok_s"] / base)) if base else "--"
        print("%-16s %9.2f %9s %9s %10s" % (r["arm"], r["tok_s"], ratio, acc_s, len_s))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
