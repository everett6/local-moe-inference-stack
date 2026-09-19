"""
Quick path: raw prompt vs chat-templated prompt, measured.

The quick path answers with the 0.5B draft, shows that answer, then replays it
against the 30B token by token (verify_fast_answer). Both sides were fed the
user's message as bare text, with no chat template. Both models are Instruct
models, so given bare text they tend to continue it instead of answering, and
the 30B side of the check was scoring the draft against a continuation, not
against a reply.

The fix builds the prompt with the 30B's own chat template
(LocalMoEEngine.chat_prompt_tokens) and uses those ids for both models. This
script runs the same quick-bucket prompts both ways, through the real engine
methods, and records for each:
  - the draft's answer (is it actually an answer?)
  - token agreement with the 30B and whether the answer got corrected
  - the draft's answer latency (what the user waits for)

It also checks the rendered template against the chat endpoint: the 30B's
greedy /completion reply from chat_prompt_tokens must equal its
/v1/chat/completions reply for the same message. If they differ, the quick path
would be verifying against a different prompt than the one the 30B path uses.
"""
import json
import os
import sys
import time

AI2 = "/home/everett/AI2"
sys.path.insert(0, AI2)

import requests  # noqa: E402

from config import Paths, Runtime  # noqa: E402
from local_engine import LocalMoEEngine  # noqa: E402
from router import prompt_bucket  # noqa: E402

PROMPTS = [
    "What is 17 times 24?",
    "Who wrote Pride and Prejudice?",
    "What is the capital of Australia?",
    "When did World War II end?",
    "Is a tomato a fruit? Yes or no.",
    "What does HTTP stand for?",
    "Where is Mount Kilimanjaro?",
    "Why is the sky blue?",
]
MAX_TOKENS = 128


def run_mode(engine, mode, prompt):
    if mode == "raw":
        ids = engine.draft.tokenize(prompt.encode("utf-8"))            # what app.py used to do
    else:
        ids = engine.chat_prompt_tokens([{"role": "user", "content": prompt}])
    t0 = time.perf_counter()
    fast = engine.generate_fast(list(ids), max_tokens=MAX_TOKENS)
    draft_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    agree, checked, corrected = engine.verify_fast_answer(list(ids), fast, MAX_TOKENS)
    verify_s = time.perf_counter() - t1
    detok = lambda t: engine.draft.detokenize(t).decode("utf-8", "ignore")  # noqa: E731
    return {
        "draft_answer": detok(fast),
        "final_answer": detok(corrected) if corrected is not None else detok(fast),
        "agree": agree, "checked": checked, "corrected": corrected is not None,
        "draft_s": round(draft_s, 3), "verify_s": round(verify_s, 2),
        "draft_tokens": len(fast),
    }


def main():
    for p in PROMPTS:
        assert prompt_bucket(p) == "quick", f"not a quick-bucket prompt: {p!r}"

    engine = LocalMoEEngine(Paths(), Runtime())        # no trainer: nothing to train here
    results = {"per_prompt": [], "template_check": []}
    try:
        # 1. Does the rendered template match what the chat endpoint uses?
        for p in PROMPTS[:3]:
            ids = engine.chat_prompt_tokens([{"role": "user", "content": p}])
            via_ids = engine.big.complete_greedy(ids, 64).get("content", "")
            via_chat = requests.post(f"{engine.big.base_url}/v1/chat/completions", json={
                "messages": [{"role": "user", "content": p}], "max_tokens": 64,
                "temperature": 0.0, "repeat_penalty": 1.1}, timeout=300).json()["choices"][0]["message"]["content"]
            results["template_check"].append({"prompt": p, "match": via_ids == via_chat,
                                              "via_ids": via_ids[:120], "via_chat": via_chat[:120]})
            print(f"template check {'MATCH' if via_ids == via_chat else 'DIFF '} {p!r}", file=sys.stderr)

        # 2. Raw vs templated, interleaved per prompt.
        for p in PROMPTS:
            row = {"prompt": p}
            for mode in ("raw", "templated"):
                row[mode] = run_mode(engine, mode, p)
                r = row[mode]
                print(f"  [{mode:9s}] agree {r['agree']:3d}/{r['checked']:<3d} "
                      f"corrected={str(r['corrected']):5s} draft {r['draft_s']:.2f}s  "
                      f"{p!r} -> {r['draft_answer'][:60]!r}", file=sys.stderr)
            results["per_prompt"].append(row)
            with open(os.path.join(AI2, "experiments", "quick_path_template_result.json"), "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
    finally:
        engine.big.stop()

    print("\n=== QUICK PATH: raw vs chat-templated prompt ===")
    for mode in ("raw", "templated"):
        rows = [r[mode] for r in results["per_prompt"]]
        agree = sum(r["agree"] for r in rows)
        checked = sum(r["checked"] for r in rows)
        corrected = sum(r["corrected"] for r in rows)
        draft_s = sum(r["draft_s"] for r in rows) / len(rows)
        toks = sum(r["draft_tokens"] for r in rows) / len(rows)
        print(f"{mode:9s}: token agreement {agree}/{checked} = {agree / max(checked, 1):.1%}, "
              f"corrected {corrected}/{len(rows)}, draft answer {draft_s:.2f}s avg, "
              f"{toks:.0f} tokens avg")
    ok = sum(c["match"] for c in results["template_check"])
    print(f"template matches chat endpoint: {ok}/{len(results['template_check'])}")
    print("saved experiments/quick_path_template_result.json")


if __name__ == "__main__":
    main()
