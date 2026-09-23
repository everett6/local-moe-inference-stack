"""
Standalone, reproducible 3-way benchmark:

  1. BASELINE       -- big model alone, no draft model, no speculative decoding.
  2. HAND-ROLLED ON -- this project's own Python draft-propose/big-verify loop
                       (local_engine.LocalMoEEngine.generate).
  3. NATIVE SPEC    -- llama-server's own built-in C++ speculative decoding
                       (--spec-draft-model / --spec-type draft-simple), which
                       does real batched verification with no per-round HTTP
                       round trip.

Run modes SEQUENTIALLY (one llama-server process at a time) rather than
concurrently: two live copies of the 30B model would roughly double VRAM/RAM
use for no benchmarking benefit, and this hardware doesn't have headroom to
spare. Same prompt set, same max_tokens ceiling, real wall-clock timing for
all three -- the only thing that differs between runs is which server flags
are active and which code path drives generation.

Usage: python3 benchmark_all.py
Writes BENCHMARK_RESULTS.md in this directory.
"""
import json
import os
import subprocess
import sys
import time

import requests

from config import Paths, Runtime
from local_engine import BigModelServer, LocalMoEEngine

PROMPTS = [
    "Write a Python function that reverses a singly linked list.",
    "Explain the CAP theorem in two sentences.",
    "What is 17 times 24?",
    "Summarize the plot of Romeo and Juliet in one short paragraph.",
    "Fix this bug: `def add(a, b): return a - b`",
]
MAX_TOKENS = 128
PORT = 8090


def wait_healthy(base_url: str, timeout: float = 120.0) -> None:
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{base_url}/health", timeout=2)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(1.0)
    raise RuntimeError(f"{base_url} did not become healthy in time")


def run_baseline_and_handrolled(paths: Paths, rt: Runtime):
    """One engine instance serves both modes 1 and 2, since generate() and
    generate_baseline() share the same underlying plain (non-native-spec)
    server -- no need to restart anything between them."""
    engine = LocalMoEEngine(paths, rt)
    results = {"baseline": [], "handrolled": []}
    try:
        for p in PROMPTS:
            prompt_tokens = engine.draft.tokenize(p.encode("utf-8"))

            start = time.perf_counter()
            base_out = engine.generate_baseline(list(prompt_tokens), max_tokens=MAX_TOKENS)
            base_elapsed = time.perf_counter() - start
            results["baseline"].append({
                "prompt": p, "tokens": len(base_out), "elapsed_s": base_elapsed,
                "tok_s": (len(base_out) / base_elapsed) if base_elapsed > 0 else 0.0,
            })

            accepts_before, rejects_before = engine.accepts, engine.rejects
            start = time.perf_counter()
            spec_out = engine.generate(list(prompt_tokens), max_tokens=MAX_TOKENS)
            spec_elapsed = time.perf_counter() - start
            d_a = engine.accepts - accepts_before
            d_r = engine.rejects - rejects_before
            accept_rate = (d_a / (d_a + d_r)) if (d_a + d_r) > 0 else 0.0
            results["handrolled"].append({
                "prompt": p, "tokens": len(spec_out), "elapsed_s": spec_elapsed,
                "tok_s": (len(spec_out) / spec_elapsed) if spec_elapsed > 0 else 0.0,
                "accept_rate": accept_rate,
            })
            print(f"[baseline+handrolled] done: {p[:40]!r}", file=sys.stderr)
    finally:
        engine.big.stop()
    return results


def run_native_spec(paths: Paths, rt: Runtime):
    """Launch llama-server with its own built-in speculative decoding and hit
    plain /completion -- the server does propose+verify internally, in C++,
    with no per-round HTTP call from us."""
    env = dict(os.environ)
    if paths.llama_server_ld_library_path:
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (
            f"{paths.llama_server_ld_library_path}:{existing}" if existing else paths.llama_server_ld_library_path
        )
    log_path = os.path.join(os.path.dirname(paths.mismatch_log_path) or ".", "llama_server_native_spec.log")
    log_file = open(log_path, "w")
    cmd = [
        paths.llama_server_bin,
        "-m", paths.big_model_gguf,
        "-c", str(rt.n_ctx),
        "-t", str(rt.threads),
        "--port", str(PORT),
        "-ngl", "999",
        # The draft model shares the card here, and this launch has no fallback
        # like BigModelServer's, so leave room for it: at 20 there is ~700 MiB
        # free on an empty card, less than the draft needs.
        "--n-cpu-moe", str(max(rt.n_cpu_moe, 24)),
        "-fa", "on",
        "--spec-draft-model", paths.draft_model_gguf,
        "--spec-type", "draft-simple",
        "--spec-draft-n-max", str(rt.speculative_k),
        "--spec-draft-ngl", "999",
    ]
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True, env=env)
    base_url = f"http://127.0.0.1:{PORT}"
    results = {"native_spec": []}
    try:
        wait_healthy(base_url)
        # Reuse the draft model's tokenizer (same vocab, verified this session) just
        # to count prompt/output tokens consistently with the other two modes.
        from llama_cpp import Llama
        tok = Llama(model_path=paths.draft_model_gguf, n_ctx=rt.n_ctx, n_gpu_layers=0, vocab_only=True, verbose=False)
        for p in PROMPTS:
            prompt_tokens = tok.tokenize(p.encode("utf-8"))
            start = time.perf_counter()
            r = requests.post(f"{base_url}/completion",
                               json={"prompt": list(prompt_tokens), "n_predict": MAX_TOKENS, "temperature": 0.0},
                               timeout=180)
            elapsed = time.perf_counter() - start
            r.raise_for_status()
            d = r.json()
            n_tokens = d.get("tokens_predicted", 0)
            results["native_spec"].append({
                "prompt": p, "tokens": n_tokens, "elapsed_s": elapsed,
                "tok_s": (n_tokens / elapsed) if elapsed > 0 else 0.0,
                "server_reported_tok_s": d.get("timings", {}).get("predicted_per_second", 0.0),
            })
            print(f"[native_spec] done: {p[:40]!r}", file=sys.stderr)
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=15)
        log_file.close()
    return results


def summarize(rows):
    total_tokens = sum(r["tokens"] for r in rows)
    total_time = sum(r["elapsed_s"] for r in rows)
    return (total_tokens / total_time) if total_time > 0 else 0.0


def main():
    paths = Paths()
    rt = Runtime()

    print("=== Phase 1/2: baseline + hand-rolled speculative (shared plain server) ===", file=sys.stderr)
    r1 = run_baseline_and_handrolled(paths, rt)
    time.sleep(2)

    print("=== Phase 3: native C++ speculative decoding ===", file=sys.stderr)
    r2 = run_native_spec(paths, rt)

    all_results = {**r1, **r2}
    avg = {mode: summarize(rows) for mode, rows in all_results.items()}

    lines = []
    lines.append("# AI2 3-way benchmark results\n")
    lines.append(f"Prompts: {len(PROMPTS)}, max_tokens ceiling per prompt: {MAX_TOKENS} "
                 f"(actual length varies -- all three modes stop on the model's own EOS token)\n")
    lines.append("## Summary (aggregate tok/s across all prompts)\n")
    lines.append("| Mode | Avg tok/s | Speedup vs baseline |")
    lines.append("|---|---|---|")
    base_avg = avg.get("baseline", 0.0)
    for mode, label in [("baseline", "Baseline (no speculative decoding)"),
                         ("handrolled", "Hand-rolled Python speculative (this project)"),
                         ("native_spec", "Native llama-server speculative (--spec-type draft-simple)")]:
        a = avg.get(mode, 0.0)
        speedup = (a / base_avg) if base_avg > 0 else 0.0
        lines.append(f"| {label} | {a:.2f} | {speedup:.2f}x |")
    lines.append("")

    for mode, label in [("baseline", "Baseline"), ("handrolled", "Hand-rolled speculative"), ("native_spec", "Native speculative")]:
        lines.append(f"## {label} -- per prompt\n")
        cols = "| Prompt | Tokens out | Time (s) | Tok/s |"
        if mode == "handrolled":
            cols += " Draft accept rate |"
        lines.append(cols)
        sep = "|---|---|---|---|"
        if mode == "handrolled":
            sep += "---|"
        lines.append(sep)
        for row in all_results.get(mode, []):
            label_p = row["prompt"] if len(row["prompt"]) <= 50 else row["prompt"][:47] + "..."
            line = f"| `{label_p}` | {row['tokens']} | {row['elapsed_s']:.2f} | {row['tok_s']:.2f} |"
            if mode == "handrolled":
                line += f" {row.get('accept_rate', 0.0):.1%} |"
            lines.append(line)
        lines.append("")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "BENCHMARK_RESULTS.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nWrote {out_path}", file=sys.stderr)
    print(json.dumps(avg, indent=2))


if __name__ == "__main__":
    main()
