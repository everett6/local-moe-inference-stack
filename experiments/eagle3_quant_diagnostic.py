"""
EAGLE3 diagnostic: is the 10-15% draft acceptance in EAGLE3_DRAFT_MODEL.md caused
by *quantization of the target model*, or by something else?

Why this experiment exists
--------------------------
The EAGLE3 write-up landed on "most likely cause: quantization/CPU-offload
mismatch (not confirmed)". Two hypotheses were ruled out there (context length,
greedy-vs-sampled). Confirming the quantization hypothesis on the real 30B target
is impossible on this box: isolating quantization from CPU-offload would require
a non-quantized 30B fully resident in 12GB of VRAM.

So: reproduce the same question on a target small enough that *every* confound
can be held fixed. Qwen3-1.7B is dense (no MoE), 28 layers, and fits entirely in
VRAM at BF16 -- so both arms below run fully GPU-resident with no `-ot` offload,
and the ONLY variable is the target's numeric precision.

  arm A: target = Qwen3-1.7B BF16     + EAGLE3 draft, -ngl 999, no offload
  arm B: target = Qwen3-1.7B Q4_K_M   + EAGLE3 draft, -ngl 999, no offload

Reading the result:
  - A high (50-80%), B much lower   -> quantization IS the killer. Retraining the
                                       draft against quantized hidden states is a
                                       justified project.
  - A high, B also high             -> quantization is fine. Our 30B problem is
                                       specific to MoE and/or CPU expert offload.
  - A low AND B low                 -> llama.cpp's EAGLE3 path is broken/limited
                                       generally; no retraining would fix it.

Note the layer-index question was already settled separately and is NOT the
cause: llama.cpp's converter emits target_layers = [2, n//2, n-3], which is
exactly SGLang's *internal* capture space (sglang qwen3_moe.py
set_eagle3_layers_to_capture applies `+1` to the config-space ids that SpecForge
defaults to `[1, n//2-1, n-4]`). Both projects capture the input to layer i, so
[2, 14, 25] here is correct, not off by one.

Acceptance is read from llama-server's own log line, not recomputed:
  "draft acceptance = %f (%d accepted / %d generated), mean len = %f"
(tools/server/server-context.cpp:677). `mean len` is the acceptance length that
published EAGLE3 benchmarks report, so it is directly comparable.
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

DRAFT_GGUF = os.path.join(AI2, "Model Training", "qwen3-1.7b-eagle3.gguf")
GGUF_DIR = os.path.join(AI2, "models", "diag-gguf")
PORT = 8099

# Same shape of prompts as the original EAGLE3 test: a spread of task types, so a
# single easy/repetitive prompt can't flatter the acceptance number.
PROMPTS = [
    "Write a Python function that implements binary search on a sorted list.",
    "Explain in detail how photosynthesis works.",
    "Compare and contrast supervised and unsupervised machine learning.",
    "Write a short story about a robot discovering music for the first time.",
]
MAX_TOKENS = 128

# Matches tools/server/server-context.cpp:677
ACC_RE = re.compile(
    r"draft acceptance = ([0-9.]+) \(\s*(\d+) accepted /\s*(\d+) generated\), mean len =\s*([0-9.]+)"
)


def wait_healthy(timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(1)
    return False


def run_arm(name, target_gguf, use_draft=True, temp=0.7):
    """Launch a server for one arm, run all prompts, return timing + acceptance."""
    log_path = os.path.join(AI2, "state", f"diag_{name}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    cmd = [
        SERVER,
        "-m", target_gguf,
        "--port", str(PORT),
        "-ngl", "999",          # fully GPU-resident -- no -ot offload in either arm
        "-c", "4096",
        "-fa", "on",
        "--no-warmup",
    ]
    if use_draft:
        cmd += [
            "--spec-draft-model", DRAFT_GGUF,
            "--spec-type", "draft-eagle3",
            "--spec-draft-n-max", "3",   # matches the checkpoint README's num-steps 3
            "--spec-draft-ngl", "999",
        ]

    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{VENDOR}:{BACKEND}"

    with open(log_path, "w") as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
        try:
            if not wait_healthy():
                proc.kill()
                return {"arm": name, "error": "server did not become healthy", "log": log_path}

            total_tokens = 0
            total_elapsed = 0.0
            for p in PROMPTS:
                body = json.dumps({
                    "prompt": p,
                    "n_predict": MAX_TOKENS,
                    "temperature": temp,
                    "top_p": 0.9,
                    "cache_prompt": False,
                }).encode()
                req = urllib.request.Request(
                    f"http://127.0.0.1:{PORT}/completion", data=body,
                    headers={"Content-Type": "application/json"},
                )
                t0 = time.perf_counter()
                with urllib.request.urlopen(req, timeout=600) as r:
                    out = json.loads(r.read())
                elapsed = time.perf_counter() - t0
                n = out.get("tokens_predicted", 0)
                total_tokens += n
                total_elapsed += elapsed
                print(f"  [{name}] {n:4d} tok in {elapsed:6.2f}s "
                      f"({n/elapsed if elapsed else 0:5.2f} tok/s)", file=sys.stderr)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)

    # Acceptance is printed by the server at slot-release time; take the last
    # (cumulative) line so it reflects the whole run, not one prompt.
    acc = None
    with open(log_path) as f:
        for line in f:
            m = ACC_RE.search(line)
            if m:
                acc = {
                    "rate": float(m.group(1)),
                    "accepted": int(m.group(2)),
                    "generated": int(m.group(3)),
                    "mean_len": float(m.group(4)),
                }

    return {
        "arm": name,
        "target": os.path.basename(target_gguf),
        "draft": use_draft,
        "tokens": total_tokens,
        "elapsed_s": round(total_elapsed, 2),
        "tok_s": round(total_tokens / total_elapsed, 2) if total_elapsed else 0.0,
        "acceptance": acc,
        "log": log_path,
    }


def main():
    bf16 = os.path.join(GGUF_DIR, "Qwen3-1.7B-BF16.gguf")
    q4 = os.path.join(GGUF_DIR, "Qwen3-1.7B-Q4_K_M.gguf")
    for p in (bf16, q4, DRAFT_GGUF):
        if not os.path.exists(p):
            print(f"missing: {p}", file=sys.stderr)
            sys.exit(1)

    results = []
    # No-draft baselines first, so the speedup ratio for each arm is measured
    # against that same target's own speed -- BF16 and Q4 differ in raw tok/s,
    # and comparing a BF16-with-draft number to a Q4 baseline would be bogus.
    print("=== baseline: BF16, no draft ===", file=sys.stderr)
    results.append(run_arm("bf16_nodraft", bf16, use_draft=False))
    print("=== baseline: Q4_K_M, no draft ===", file=sys.stderr)
    results.append(run_arm("q4_nodraft", q4, use_draft=False))
    print("=== arm A: BF16 + EAGLE3 ===", file=sys.stderr)
    results.append(run_arm("bf16_eagle3", bf16, use_draft=True))
    print("=== arm B: Q4_K_M + EAGLE3 ===", file=sys.stderr)
    results.append(run_arm("q4_eagle3", q4, use_draft=True))

    out_path = os.path.join(AI2, "experiments", "eagle3_quant_diagnostic_result.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== SUMMARY ===")
    print(f"{'arm':16s} {'tok/s':>8s} {'acceptance':>12s} {'mean len':>9s}")
    for r in results:
        acc = r.get("acceptance")
        acc_s = f"{acc['rate']:.1%}" if acc else "--"
        len_s = f"{acc['mean_len']:.2f}" if acc else "--"
        print(f"{r['arm']:16s} {r.get('tok_s', 0):8.2f} {acc_s:>12s} {len_s:>9s}")
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
