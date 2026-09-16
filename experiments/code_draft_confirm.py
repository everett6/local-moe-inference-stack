"""
Does the one winning draft arm survive noise and a fair comparison?

workload_pmin_sweep.py produced the first speculative arm in this whole
investigation to beat its baseline: Qwen3-0.6B drafting on a pure-code workload
with confidence gating (k=4, --spec-draft-p-min 0.8) measured 1.06x. Two reasons
not to believe it yet:

1. NOISE. That run's code baseline was 66.47 tok/s. The identical config measured
   73.01 tok/s in qwen3_draft_test.py an hour earlier. A ~10% baseline swing is
   larger than the 6% win. So this interleaves repeats -- every arm runs once per
   round, round after round -- so slow drift (thermal, page cache, background
   load) lands on all arms equally instead of on whichever ran last.

2. THE WRONG BASELINE. The draft model occupies VRAM that would otherwise hold MoE
   experts, and moving experts into VRAM is worth more per GB than anything else
   measured here (moe_offload_sweep.py: 47 -> 82 tok/s). "Draft at n-cpu-moe 24
   vs no draft at n-cpu-moe 24" hides that cost. The question a user actually
   faces is: given 12 GB, spend the last ~1.5 GB on experts, or on a draft model?
   So the reference arm is no draft at all, and every arm -- reference included --
   is fitted to the tightest --n-cpu-moe it can actually load at, right now.

Also pushes past k=4, since at p-min 0.8 the sweep improved monotonically with k
(0.99x at k=2 -> 1.06x at k=4) and never found the top.
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
BACKEND = "/home/everett/.lmstudio/extensions/backends/llama.cpp-linux-x86_64-nvidia-cuda12-avx2-2.37.0"
VENDOR = "/home/everett/.lmstudio/extensions/backends/vendor/linux-llama-cuda12-vendor-v1"
SERVER = os.path.join(BACKEND, "llama-server")
BIG = "/home/everett/.lmstudio/models/lmstudio-community/Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"
DRAFT = os.path.join(AI2, "Model Training", "Qwen3-0.6B-Q4_K_M.gguf")
PORT = 8089
ROUNDS = 3

# Same code set as workload_pmin_sweep.py, so the two runs are comparable.
PROMPTS = [
    "Write a Python class implementing a least-recently-used cache with get and put.",
    "Write a Python dataclass for a 2D vector with add, subtract, dot product and magnitude methods.",
    "Write a Python function that merges two sorted lists into one sorted list, with type hints and a docstring.",
    "Write a Python context manager that times the block it wraps and logs the elapsed milliseconds.",
]
MAX_TOKENS = 192

ACC_RE = re.compile(
    r"draft acceptance = ([0-9.]+) \(\s*(\d+) accepted /\s*(\d+) generated\), mean len =\s*([0-9.]+)"
)


def draft_args(k, pmin):
    return ["--spec-type", "draft-simple", "-md", DRAFT, "-ngld", "999",
            "--spec-draft-n-max", str(k), "--spec-draft-p-min", pmin]


# (name, extra args). No fixed --n-cpu-moe: every arm is fitted -- started at 20
# and stepped up one layer at a time until it loads -- so each gets the most
# experts in VRAM it can actually have alongside whatever else is on the card.
# A first version of this script hardcoded 20/22 and every arm but one failed to
# load, because a browser was holding 456 MiB. Fitting per arm is also what makes
# the comparison fair: the draft model's VRAM cost shows up as fewer experts on
# the GPU, which is exactly the cost a user would pay.
ARMS = [
    ("nodraft", []),                        # the bar to beat
    ("draft_k4_p0.8", draft_args(4, "0.8")),
    ("draft_k6_p0.8", draft_args(6, "0.8")),
    ("draft_k6_p0.9", draft_args(6, "0.9")),
    ("draft_k8_p0.9", draft_args(8, "0.9")),
]
START_N_CPU_MOE = 20


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


def run_once(name, ncmoe, extra, rnd):
    log_path = os.path.join(AI2, "state", f"confirm_{name}_r{rnd}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    cmd = [SERVER, "-m", BIG, "--port", str(PORT), "-ngl", "999",
           "--n-cpu-moe", ncmoe, "-c", "4096", "-fa", "on", "-t", "16",
           "--no-warmup"] + extra
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{VENDOR}:{BACKEND}"

    with open(log_path, "w") as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
        try:
            if not wait_healthy(proc):
                proc.kill()
                return {"error": "failed to load (likely VRAM)"}
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
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)

    accepted = generated = 0
    spec_failed = False
    with open(log_path) as f:
        for line in f:
            if ("failed to initialize speculative decoding" in line
                    or "vocabs are not compatible" in line):
                spec_failed = True
            m = ACC_RE.search(line)
            if m:
                accepted += int(m.group(2))
                generated += int(m.group(3))
    if extra and spec_failed:
        return {"error": "speculation did not engage"}
    return {"tok_s": total_tokens / total_elapsed if total_elapsed else 0.0,
            "accepted": accepted, "generated": generated}


def main():
    if not os.path.exists(DRAFT):
        print(f"missing draft: {DRAFT}", file=sys.stderr)
        sys.exit(1)

    samples = {name: [] for name, _ in ARMS}
    errors = {}
    acc = {name: [0, 0] for name, _ in ARMS}
    fitted = {}
    out_path = os.path.join(AI2, "experiments", "code_draft_confirm_result.json")

    for rnd in range(ROUNDS):
        for name, extra in ARMS:
            if name in errors:
                continue          # an arm that cannot load will not load next round either
            r = None
            n = fitted.get(name, START_N_CPU_MOE)
            while n <= 48:
                r = run_once(name, str(n), extra, rnd)
                if "error" not in r or "failed to load" not in r["error"]:
                    break
                n += 1
            fitted[name] = n
            if "error" in r:
                errors[name] = r["error"]
                print(f"  r{rnd} [{name}] ERROR {r['error']}", file=sys.stderr)
                continue
            samples[name].append(r["tok_s"])
            acc[name][0] += r["accepted"]
            acc[name][1] += r["generated"]
            print(f"  r{rnd} [{name}] n-cpu-moe {fitted[name]}: {r['tok_s']:7.2f} tok/s", file=sys.stderr)
            with open(out_path, "w") as f:
                json.dump({"samples": samples, "acceptance": acc, "errors": errors,
                           "fitted_n_cpu_moe": fitted}, f, indent=2)

    ref = statistics.median(samples["nodraft"]) if samples["nodraft"] else None
    print("\n=== CODE DRAFT CONFIRMATION (%d interleaved rounds, each arm VRAM-fitted) ===" % ROUNDS)
    print("reference = no draft, with the draft's VRAM given to experts instead\n")
    print("%-16s %6s %8s %8s %8s %9s %9s" % ("arm", "ncmoe", "median", "min", "max", "vs nodraft", "accept"))
    for name, _ in ARMS:
        if name in errors and not samples[name]:
            print("%-16s ERROR %s" % (name, errors[name]))
            continue
        s = samples[name]
        med = statistics.median(s)
        a, g = acc[name]
        acc_s = ("%.1f%%" % (100 * a / g)) if g else "--"
        ratio = ("%.2fx" % (med / ref)) if ref else "--"
        print("%-16s %6s %8.2f %8.2f %8.2f %9s %9s" % (name, fitted.get(name, "--"), med, min(s), max(s), ratio, acc_s))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
