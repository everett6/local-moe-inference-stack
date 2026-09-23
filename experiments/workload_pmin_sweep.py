"""
Is a CODE-specialised draft worth building? And does confidence-gating rescue any
draft at all?

Two questions left open by qwen3_draft_test.py, both of which have to be answered
before "use a coder draft model" can be dismissed rather than just deferred.

1. WORKLOAD. That test measured 61.3% acceptance on a mixed prompt set against a
   75% break-even, and lost. But acceptance is not uniform across prompts -- the
   EAGLE3 arms in spec_shootout.py ranged 6.6% to 54.9% depending on the prompt,
   best on code. If code-only acceptance clears the bar while prose drags the
   average under it, then a code-specialised draft is worth building *for coding
   workloads specifically*, and the mixed-set result was the wrong measurement.
   So: identical arms, run separately on a pure-code set and a pure-prose set.

2. CONFIDENCE GATING. Every arm so far ran with `--spec-draft-p-min 0.00`, the
   default: draft unconditionally, every step. That is the worst case for a
   mediocre draft, because a rejected draft still costs a wider target forward
   pass. `--spec-draft-p-min P` drafts only when the draft model's own
   probability exceeds P, which trades away the easy wins to avoid the losses.
   spec_breakeven.py's model has no term for this -- it assumes you always draft
   -- so gating is the one lever that could beat the model's prediction rather
   than confirm it.

Each (arm, workload) pair gets its own llama-server launch so the acceptance
numbers are attributable to one workload and not mixed across slots.

Acceptance here SUMS every `draft acceptance` line in the log rather than taking
the last one. llama-server prints that line per slot at release, so the last line
is one slot's tally -- which is why ngram_sweep_result.json's acceptance column
carried a caveat. Summing gives the whole run.
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
PORT = 8091
N_CPU_MOE = "24"   # same as qwen3_draft_test.py, so results are comparable to it

WORKLOADS = {
    # Boilerplate-heavy, repetitive, highly predictable token-to-token. If a
    # draft model is ever going to clear the bar here, it is on this.
    "code": [
        "Write a Python class implementing a least-recently-used cache with get and put.",
        "Write a Python dataclass for a 2D vector with add, subtract, dot product and magnitude methods.",
        "Write a Python function that merges two sorted lists into one sorted list, with type hints and a docstring.",
        "Write a Python context manager that times the block it wraps and logs the elapsed milliseconds.",
    ],
    # Free-form prose: high branching factor, the case a draft should do worst on.
    "prose": [
        "Explain the CAP theorem in two sentences.",
        "Summarize the plot of Romeo and Juliet in one short paragraph.",
        "Explain why distributed consensus is hard, for a reader who knows no theory.",
        "Describe the difference between correlation and causation with an everyday example.",
    ],
}
MAX_TOKENS = 192

ACC_RE = re.compile(
    r"draft acceptance = ([0-9.]+) \(\s*(\d+) accepted /\s*(\d+) generated\), mean len =\s*([0-9.]+)"
)

# (name, draft length k, p-min).  k=None means no speculation at all.
ARMS = [
    ("baseline", None, None),
    ("k1_p0.0", 1, "0.0"),
    ("k1_p0.6", 1, "0.6"),
    ("k2_p0.6", 2, "0.6"),
    ("k2_p0.8", 2, "0.8"),
    ("k4_p0.8", 4, "0.8"),
]


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


def run(arm, k, pmin, workload):
    name = f"{workload}_{arm}"
    log_path = os.path.join(AI2, "state", f"wpmin_{name}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    cmd = [SERVER, "-m", BIG, "--port", str(PORT), "-ngl", "999",
           "--n-cpu-moe", N_CPU_MOE, "-c", "4096", "-fa", "on", "-t", "16",
           "--no-warmup"]
    if k is not None:
        cmd += ["--spec-type", "draft-simple", "-md", DRAFT, "-ngld", "999",
                "--spec-draft-n-max", str(k), "--spec-draft-p-min", pmin]

    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{VENDOR}:{BACKEND}"

    with open(log_path, "w") as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
        try:
            if not wait_healthy(proc):
                proc.kill()
                return {"arm": arm, "workload": workload,
                        "error": "server never became healthy", "log": log_path}
            total_tokens = 0
            total_elapsed = 0.0
            for p in WORKLOADS[workload]:
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
    accepted = generated = 0
    n_lines = 0
    with open(log_path) as f:
        for line in f:
            if ("failed to initialize speculative decoding" in line
                    or "vocabs are not compatible" in line):
                spec_failed = line.strip()
            m = ACC_RE.search(line)
            if m:
                accepted += int(m.group(2))
                generated += int(m.group(3))
                n_lines += 1

    if k is not None and spec_failed:
        return {"arm": arm, "workload": workload,
                "error": "speculation did not engage: " + spec_failed, "log": log_path}

    acc = None
    if generated:
        acc = {"rate": accepted / generated, "accepted": accepted,
               "generated": generated, "slots_logged": n_lines}
    return {"arm": arm, "workload": workload, "k": k, "p_min": pmin,
            "tok_s": round(tok_s, 2), "tokens": total_tokens,
            "elapsed_s": round(total_elapsed, 2), "acceptance": acc, "log": log_path}


def main():
    if not os.path.exists(DRAFT):
        print(f"missing draft: {DRAFT}", file=sys.stderr)
        sys.exit(1)

    results = []
    out_path = os.path.join(AI2, "experiments", "workload_pmin_result.json")
    for workload in WORKLOADS:
        for arm, k, pmin in ARMS:
            print(f"=== {workload} / {arm} ===", file=sys.stderr)
            results.append(run(arm, k, pmin, workload))
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)

    print("\n=== WORKLOAD x P-MIN (--n-cpu-moe %s, draft Qwen3-0.6B Q4_K_M) ===" % N_CPU_MOE)
    for workload in WORKLOADS:
        base = next((r["tok_s"] for r in results
                     if r["workload"] == workload and r["arm"] == "baseline" and "tok_s" in r), None)
        print(f"\n  {workload}   (baseline {base if base else '--'} tok/s)")
        print("  %-12s %9s %9s %9s %12s" % ("arm", "tok/s", "vs base", "accept", "acc/gen"))
        for r in results:
            if r["workload"] != workload:
                continue
            if "error" in r:
                print("  %-12s ERROR %s" % (r["arm"], r["error"][:80]))
                continue
            acc = r.get("acceptance")
            acc_s = ("%.1f%%" % (acc["rate"] * 100)) if acc else "--"
            raw = ("%d/%d" % (acc["accepted"], acc["generated"])) if acc else "--"
            ratio = ("%.2fx" % (r["tok_s"] / base)) if base else "--"
            print("  %-12s %9.2f %9s %9s %12s" % (r["arm"], r["tok_s"], ratio, acc_s, raw))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
