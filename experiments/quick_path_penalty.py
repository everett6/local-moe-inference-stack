"""
Quick path: does matching the 30B's repeat_penalty to the draft's cut the
corrections that weren't real disagreements?

The quick path shows the 0.5B draft's greedy answer, then has the 30B generate
greedily from the same token ids and compares token by token
(LocalMoEEngine.verify_fast_answer). Any difference replaces the shown answer
and becomes a training example. The draft (llama-cpp-python) samples with
repeat_penalty 1.0; the 30B was sent 1.1. Now Runtime.repeat_penalty = 1.0.

For 24 quick-bucket questions the draft answers once (its output doesn't depend
on the 30B), then the 30B verifies it at 1.1 and at 1.0. Reported per model:
token agreement and how many answers got corrected. Q2_K (the new default) and
Q4_K_M. CPU-heavy (the draft runs on the CPU), so don't run it alongside a GPU
speed benchmark.
"""
import json
import os
import sys
from dataclasses import replace

AI2 = "/home/everett/AI2"
sys.path.insert(0, AI2)
from config import BIG_MODELS, Paths, Runtime  # noqa: E402
from local_engine import LocalMoEEngine  # noqa: E402
from router import prompt_bucket  # noqa: E402

PROMPTS = [
    "What is 17 times 24?", "Who wrote Pride and Prejudice?", "What is the capital of Australia?",
    "When did World War II end?", "Is a tomato a fruit? Yes or no.", "What does HTTP stand for?",
    "Where is Mount Kilimanjaro?", "Why is the sky blue?", "What is the boiling point of water?",
    "Who painted the Mona Lisa?", "How many continents are there?", "What is the largest planet?",
    "What does CPU stand for?", "Who discovered penicillin?", "What is the square root of 144?",
    "How many days are in a leap year?", "What is the chemical symbol for gold?",
    "Which ocean is the largest?", "What language is spoken in Brazil?", "Who was the first person on the moon?",
    "What is the freezing point of water in Fahrenheit?", "How many legs does a spider have?",
    "What year did the Titanic sink?", "What is the speed of light?",
]
MAX_TOKENS = 96
PENALTIES = [1.1, 1.0]


def main():
    for p in PROMPTS:
        assert prompt_bucket(p) == "quick", f"not a quick-bucket prompt: {p!r}"
    models = [m for m in os.environ.get("MODELS", "q2_k,q4_k_m").split(",") if m]
    results = {}
    for model in models:
        path, start = BIG_MODELS[model]
        engine = LocalMoEEngine(replace(Paths(), big_model_gguf=path), replace(Runtime(), n_cpu_moe=start))
        rows = []
        try:
            for p in PROMPTS:
                ids = engine.chat_prompt_tokens([{"role": "user", "content": p}])
                fast = engine.generate_fast(list(ids), max_tokens=MAX_TOKENS)
                row = {"prompt": p, "draft": engine.draft.detokenize(fast).decode("utf-8", "ignore")}
                for pen in PENALTIES:
                    engine.big.rt = replace(engine.big.rt, repeat_penalty=pen)
                    agree, checked, corrected = engine.verify_fast_answer(list(ids), fast, MAX_TOKENS)
                    row[str(pen)] = {"agree": agree, "checked": checked, "corrected": corrected is not None}
                rows.append(row)
                print(f"  [{model}] {p!r}: " + "  ".join(
                    f"{pen}: {row[str(pen)]['agree']}/{row[str(pen)]['checked']}"
                    f"{' corrected' if row[str(pen)]['corrected'] else ''}" for pen in PENALTIES),
                    file=sys.stderr, flush=True)
        finally:
            engine.big.stop()
        results[model] = {"n_cpu_moe": engine.big.n_cpu_moe, "rows": rows}
        json.dump(results, open(os.path.join(AI2, "experiments", "quick_path_penalty_result.json"), "w"),
                  indent=1, ensure_ascii=False)

    print("\n=== QUICK PATH: 30B verification at repeat_penalty 1.1 vs 1.0 (draft samples at 1.0) ===")
    for model, r in results.items():
        for pen in PENALTIES:
            v = [row[str(pen)] for row in r["rows"]]
            agree, checked = sum(x["agree"] for x in v), sum(x["checked"] for x in v)
            print(f"{model:7s} penalty {pen}: token agreement {agree}/{checked} = {agree / max(checked, 1):.1%}, "
                  f"corrected {sum(x['corrected'] for x in v)}/{len(v)}")


if __name__ == "__main__":
    main()
