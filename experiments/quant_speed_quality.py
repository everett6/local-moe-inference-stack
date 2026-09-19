"""
Smaller quantizations of the same model: how much faster, and how much worse?

Every runtime lever that keeps Qwen3-30B-A3B at Q4_K_M has been measured (split
sweep, slots/threads/ubatch, AVX-512, CUDA env vars, speculative decoding), and
none gets near 2x. The physics: decode costs 0.321 ms per layer of experts left
in RAM, and a 17.3 GiB Q4_K_M model leaves ~23 of 48 layers there on a 12 GB
card. A 2x gain needs the model to (nearly) fit in VRAM, which means fewer bits
per weight, which costs quality. This measures both sides of that trade.

Models (same weights, different quantization; all from Hugging Face, sha256-
verified by tools/download_quants.py):
  Q4_K_M       17.28 GiB  lmstudio-community, what the app runs today (reference)
  UD-Q3_K_XL   12.88 GiB  unsloth dynamic
  IQ3_XXS      11.38 GiB  bartowski, imatrix
  Q2_K         10.16 GiB  bartowski, imatrix
  UD-IQ2_XXS    9.63 GiB  unsloth dynamic

Speed, the way the app gets it: BigModelServer itself fits the split (starts at
the lowest plausible --n-cpu-moe, steps 1 layer, ~700-token warm-up, keeps
Runtime.vram_headroom_mb = 768 MiB free), then decode tok/s over 3 prompts x
192 tokens via stream_chat, and prefill tok/s on a ~2,950-token prompt, both from
llama-server's own timings. Two interleaved rounds.

Quality, part 1 (this script): GSM8K, first 50 test questions, through
stream_chat (chat template), greedy, 768 max tokens; a reply is right if its
last number equals the reference answer. One pass per model (greedy is
deterministic).
Quality, part 2 (separate, CPU): llama-perplexity KL-divergence against Q4_K_M's
logits on a local ~10k-token prose+code corpus -- see quant_kld.sh.
"""
import json
import os
import re
import statistics
import sys
import time
from dataclasses import replace

AI2 = "/home/everett/AI2"
sys.path.insert(0, AI2)

from config import Paths, Runtime  # noqa: E402
from local_engine import BigModelServer, _gpu_free_mb  # noqa: E402

Q = os.path.join(AI2, "models", "quants")
MODELS = [
    # (label, path, --n-cpu-moe to start fitting from)
    ("Q4_K_M", Paths().big_model_gguf, 20),
    ("UD-Q3_K_XL", os.path.join(Q, "Qwen3-30B-A3B-Instruct-2507-UD-Q3_K_XL.gguf"), 4),
    ("IQ3_XXS", os.path.join(Q, "Qwen_Qwen3-30B-A3B-Instruct-2507-IQ3_XXS.gguf"), 0),
    ("Q2_K", os.path.join(Q, "Qwen_Qwen3-30B-A3B-Instruct-2507-Q2_K.gguf"), 0),
    ("UD-IQ2_XXS", os.path.join(Q, "Qwen3-30B-A3B-Instruct-2507-UD-IQ2_XXS.gguf"), 0),
]
ROUNDS = int(os.environ.get("ROUNDS", "2"))

DECODE_PROMPTS = [
    "Explain how a hash map handles collisions.",
    "Write a Python class implementing a least-recently-used cache with get and put.",
    "Summarize the causes of the French Revolution in a few paragraphs.",
]
PREFILL_PROMPT = ("Here is a log of events from a distributed system. Summarize the failures.\n" +
                  "\n".join(f"[node-{i % 7}] t={1000 + i * 13}ms heartbeat ok, replica lag {i % 11}ms, "
                            f"queue depth {(i * 37) % 250}" for i in range(110)))
GSM8K = [json.loads(l) for l in open(os.path.join(AI2, "data", "quality", "gsm8k_test_first50.jsonl"))]
NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
BOXED = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
ANSWER = re.compile(r"answer", re.IGNORECASE)


def extract_answer(text):
    """The model's final number: inside its last \\boxed{}, else the first number
    after its last "answer", else the last number in the reply. (The first version
    took the last number only, which failed correct replies: "$26.00" vs "26", and
    "45 miles away at the end of those 4 hours" read as 4.)"""
    boxed = BOXED.findall(text)
    if boxed and NUM.findall(boxed[-1]):
        return NUM.findall(boxed[-1])[-1]
    hits = list(ANSWER.finditer(text))
    if hits:
        nums = NUM.findall(text[hits[-1].end():])
        if nums:
            return nums[0]
    nums = NUM.findall(text)
    return nums[-1] if nums else None


def is_correct(got, answer):
    try:
        return got is not None and abs(float(got.replace(",", "")) - float(answer.replace(",", ""))) < 1e-6
    except ValueError:
        return False


def measure(label, path, start, rnd, do_gsm8k):
    if not os.path.exists(path):
        return {"error": "model file missing"}
    rt = replace(Runtime(), n_cpu_moe=start, n_cpu_moe_step=1)
    paths = replace(Paths(), big_model_gguf=path)
    t0 = time.time()
    srv = BigModelServer(paths, rt, port=8083)
    out = {"n_cpu_moe": srv.n_cpu_moe, "startup_s": round(time.time() - t0, 1), "free_mb": _gpu_free_mb()}
    try:
        tok = ms = 0.0
        for p in DECODE_PROMPTS:
            timings = {}
            for ev in srv.stream_chat([{"role": "user", "content": p}], 192):
                timings = ev.get("timings", timings)
            tok += timings.get("predicted_n", 0)
            ms += timings.get("predicted_ms", 0)
        out["decode_tps"] = 1000 * tok / ms
        # complete_greedy's prompt may be a string; the server tokenizes it.
        pf = srv.complete_greedy(PREFILL_PROMPT, 1).get("timings", {})
        out["prefill_tps"] = pf.get("prompt_per_second")
        if do_gsm8k:
            right, details = 0, []
            for item in GSM8K:
                reply = "".join(ev.get("delta", "") for ev in srv.stream_chat(
                    [{"role": "user", "content": item["question"]}], 768))
                got = extract_answer(reply)
                ok = is_correct(got, item["answer"])
                right += ok
                details.append({"answer": item["answer"], "got": got, "ok": ok, "reply": reply})
            out["gsm8k_correct"] = right
            out["gsm8k_details"] = details
    finally:
        srv.stop()
    return out


def main():
    only = set(sys.argv[1:])
    models = [m for m in MODELS if not only or m[0] in only]
    results = {m[0]: [] for m in models}
    out_path = os.path.join(AI2, "experiments", "quant_speed_quality_result.json")
    for rnd in range(ROUNDS):
        for label, path, start in models:
            r = measure(label, path, start, rnd, do_gsm8k=(rnd == 0))
            results[label].append(r)
            if "error" in r:
                print(f"  r{rnd} [{label}] ERROR {r['error']}", file=sys.stderr, flush=True)
            else:
                g = f" gsm8k {r['gsm8k_correct']}/50" if "gsm8k_correct" in r else ""
                print(f"  r{rnd} [{label}] split {r['n_cpu_moe']} decode {r['decode_tps']:.1f} tok/s "
                      f"prefill {r['prefill_tps']:.0f} tok/s free {r['free_mb']} MiB "
                      f"startup {r['startup_s']}s{g}", file=sys.stderr, flush=True)
            json.dump(results, open(out_path, "w"), indent=2, ensure_ascii=False)

    ref = statistics.median(r["decode_tps"] for r in results.get("Q4_K_M", []) if "error" not in r) \
        if results.get("Q4_K_M") else None
    print("\n=== QUANT SPEED + QUALITY (app-realistic fitting, 768 MiB headroom) ===")
    print("%-11s %6s %9s %8s %8s %9s" % ("model", "split", "decode", "vs Q4KM", "prefill", "GSM8K/50"))
    for label, *_ in models:
        ok = [r for r in results[label] if "error" not in r]
        if not ok:
            print("%-11s ERROR %s" % (label, results[label][0]["error"]))
            continue
        dec = statistics.median(r["decode_tps"] for r in ok)
        pf = statistics.median(r["prefill_tps"] for r in ok)
        g = next((r["gsm8k_correct"] for r in ok if "gsm8k_correct" in r), None)
        print("%-11s %6s %9.1f %7.2fx %8.0f %9s" % (label, ok[-1]["n_cpu_moe"], dec,
                                                     dec / ref if ref else 0, pf, g))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
