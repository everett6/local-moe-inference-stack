"""
What does the app ACTUALLY get, started the way app.py starts it?

Every speed number so far launched llama-server on its own. app.py doesn't. It
builds, in this order:

  1. OnlineDraftTrainer  -- an fp16 transformers copy of the draft model, placed
                            on Runtime.trainer_device, plus PyTorch's CUDA context
  2. LocalMoEEngine      -- the GGUF draft via llama-cpp-python, fully on the GPU
  3. BigModelServer      -- the 30B, fitted to whatever VRAM is left

so (1) and (2) are paid out of the big model's expert budget on every request,
whether or not anything is training. An earlier check of step 2+3 only got
73.76 tok/s at split 24; step 1 was never included.

This runs one child process per configuration (a child is the only reliable
way to hand PyTorch's CUDA context back), and in each:
  - builds trainer + engine exactly as app.py does
  - records the split BigModelServer settled on and VRAM left over
  - measures decode speed through BigModelServer.stream_chat (the path app.py
    now uses for big-model replies), on the prompts BENCHMARK_RESULTS.md used
  - on the first run only: compares a raw /completion answer to a chat-templated
    one, which is the evidence for the stream_chat change
  - times one training step on the configured device with realistic short
    contexts, since moving the trainer to CPU is only acceptable if training
    still finishes in reasonable time

Auto-refresh is disabled in the child (refresh_every_n_steps=0) so a test run
never merges, requantizes, or hot-swaps anything.
"""
import json
import os
import subprocess
import sys
import time

AI2 = "/home/everett/AI2"
sys.path.insert(0, AI2)

PROMPTS = [
    "Write a Python function that reverses a singly linked list.",
    "Explain the CAP theorem in two sentences.",
    "What is 17 times 24?",
    "Summarize the plot of Romeo and Juliet in one sentence.",
]
MAX_TOKENS = 192


def child(trainer_device, compare_template):
    import requests
    from dataclasses import replace
    from config import Paths, Runtime
    from draft_trainer import OnlineDraftTrainer
    from local_engine import LocalMoEEngine, _gpu_free_mb

    rt = replace(Runtime(), trainer_device=trainer_device, refresh_every_n_steps=0)
    paths = Paths()
    out = {"trainer_device": trainer_device, "free_before_mb": _gpu_free_mb()}

    t0 = time.time()
    trainer = OnlineDraftTrainer(paths, rt)
    out["free_after_trainer_mb"] = _gpu_free_mb()
    engine = LocalMoEEngine(paths, rt, on_mismatch=trainer.report_mismatch)
    out["startup_s"] = round(time.time() - t0, 1)
    out["n_cpu_moe"] = engine.big.n_cpu_moe
    out["free_after_engine_mb"] = _gpu_free_mb()
    try:
        tok = ms = 0.0
        for p in PROMPTS:
            timings = {}
            for ev in engine.big.stream_chat([{"role": "user", "content": p}], MAX_TOKENS):
                timings = ev.get("timings", timings)
            tok += timings.get("predicted_n", 0)
            ms += timings.get("predicted_ms", 0)
        out["decode_tps"] = round(1000 * tok / ms, 2) if ms else None

        if compare_template:
            q = "What is 17 times 24?"
            raw = requests.post(f"{engine.big.base_url}/completion", json={
                "prompt": engine.draft.tokenize(q.encode()), "n_predict": 96,
                "temperature": 0.0, "repeat_penalty": 1.1}, timeout=300).json()
            chat = "".join(ev.get("delta", "") for ev in
                           engine.big.stream_chat([{"role": "user", "content": q}], 96))
            out["template_compare"] = {"prompt": q, "raw_completion": raw.get("content", ""),
                                       "raw_tokens": raw.get("tokens_predicted"),
                                       "chat_completion": chat}

        # One training step, realistic short contexts (quick-path prompts + answers).
        base = engine.draft.tokenize(("def add(a, b):\n    return a + b\n" * 12).encode())
        t1 = time.time()
        for i in range(rt.train_batch_size):
            trainer.report_mismatch(base[: 120 + 10 * i], 198)
        deadline = time.time() + 900
        while trainer.stats.steps < 1 and time.time() < deadline:
            time.sleep(0.5)
        out["train_step_s"] = round(time.time() - t1, 1) if trainer.stats.steps >= 1 else None
    finally:
        trainer.stop()
        engine.big.stop()
    print("RESULT " + json.dumps(out), flush=True)


def contention_child(threads):
    """Decode speed while a CPU training step is actually running.

    The startup runs above time training AFTER decoding, so they never overlap.
    In the app they can: a quick-path answer is verified, its mismatches queue up,
    and a 16-example CPU training step starts while the next reply is generating.
    PyTorch's default is every physical core -- the same cores llama-server's
    CPU-side experts run on. This measures what that costs, and whether capping
    the trainer's threads (Runtime.trainer_cpu_threads) avoids it.
    """
    from dataclasses import replace
    from config import Paths, Runtime
    from draft_trainer import OnlineDraftTrainer
    from local_engine import LocalMoEEngine

    rt = replace(Runtime(), trainer_device="cpu", trainer_cpu_threads=threads,
                 refresh_every_n_steps=0)
    paths = Paths()
    trainer = OnlineDraftTrainer(paths, rt)
    engine = LocalMoEEngine(paths, rt, on_mismatch=trainer.report_mismatch)

    def decode():
        tok = ms = 0.0
        for p in PROMPTS:
            timings = {}
            for ev in engine.big.stream_chat([{"role": "user", "content": p}], MAX_TOKENS):
                timings = ev.get("timings", timings)
            tok += timings.get("predicted_n", 0)
            ms += timings.get("predicted_ms", 0)
        return round(1000 * tok / ms, 2) if ms else None

    out = {"trainer_cpu_threads": threads, "n_cpu_moe": engine.big.n_cpu_moe}
    try:
        out["decode_idle_tps"] = decode()
        # Queue several batches so training is still running for the whole decode.
        base = engine.draft.tokenize(("def add(a, b):\n    return a + b\n" * 12).encode())
        for i in range(rt.train_batch_size * 6):
            trainer.report_mismatch(base[: 120 + (i % 16) * 10], 198)
        while trainer.stats.steps < 1:          # wait until training is genuinely underway
            time.sleep(0.2)
        t0 = time.time()
        out["decode_during_training_tps"] = decode()
        out["decode_window_s"] = round(time.time() - t0, 1)
        out["steps_finished_by_end"] = trainer.stats.steps
        steps_before = trainer.stats.steps
        t1 = time.time()
        while trainer.stats.steps < steps_before + 1 and time.time() - t1 < 600:
            time.sleep(0.2)
        out["train_step_s"] = round(time.time() - t1, 1)
    finally:
        trainer.stop()
        engine.big.stop()
    print("RESULT " + json.dumps(out), flush=True)


def contention_main():
    results = []
    for threads in (0, 4, 0, 4):
        cmd = [sys.executable, __file__, "--contention", str(threads)]
        print(f"=== contention, trainer threads={threads or 'default'} ===", file=sys.stderr)
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=AI2, timeout=3600)
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
        if not line:
            print(p.stdout[-2000:], p.stderr[-3000:], file=sys.stderr)
            results.append({"trainer_cpu_threads": threads, "error": "child failed"})
            continue
        r = json.loads(line[len("RESULT "):])
        results.append(r)
        print(f"  split={r['n_cpu_moe']} idle={r['decode_idle_tps']} "
              f"during_training={r['decode_during_training_tps']} "
              f"step={r['train_step_s']}s", file=sys.stderr)
        with open(os.path.join(AI2, "experiments", "trainer_contention_result.json"), "w") as f:
            json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


def main():
    results = []
    order = ["cuda", "cpu", "cuda", "cpu"]      # interleaved
    for i, dev in enumerate(order):
        cmd = [sys.executable, __file__, "--child", dev, "1" if i == 0 else "0"]
        print(f"=== trainer on {dev} (run {i}) ===", file=sys.stderr)
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=AI2, timeout=3600)
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
        if not line:
            print(p.stdout[-2000:], p.stderr[-3000:], file=sys.stderr)
            results.append({"trainer_device": dev, "error": "child failed"})
            continue
        r = json.loads(line[len("RESULT "):])
        results.append(r)
        print(f"  split={r['n_cpu_moe']} decode={r['decode_tps']} tok/s "
              f"free_after_trainer={r['free_after_trainer_mb']} train_step={r['train_step_s']}s",
              file=sys.stderr)
        with open(os.path.join(AI2, "experiments", "app_startup_vram_result.json"), "w") as f:
            json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(sys.argv[2], sys.argv[3] == "1")
    elif len(sys.argv) > 1 and sys.argv[1] == "--contention":
        contention_child(int(sys.argv[2]))
    elif len(sys.argv) > 1 and sys.argv[1] == "--contention-sweep":
        contention_main()
    else:
        main()
