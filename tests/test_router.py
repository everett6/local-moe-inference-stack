"""
Routing tests for router.prompt_bucket.

Two bugs these pin down:

1. Keywords matched as substrings. `"no" in text` is true for "know", "now",
   "another", "note", "cannot"; `"what" in text` for "whatever"; `"cli" in text`
   for "client"; `"plan" in text` for "explanation". Any of those sent the prompt
   to the 0.5B draft model instead of the 30B.

2. The quick check ran first. `word_count <= 12 or <question word>` returned
   "quick" before the code and analysis keyword lists were ever consulted, so a
   short, explicit request like "Write a Python LRU cache class" went to the
   0.5B model despite matching both lists.

The routing *policy* is unchanged: short prompts and who/what/why-style
questions still take the quick path, as long as they aren't code or
analysis requests.

Run:  python3 tests/test_router.py      (also works under pytest)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from router import prompt_bucket  # noqa: E402

# (prompt, expected bucket)
CASES = [
    # --- substring bug: these must NOT be quick just because of a word fragment
    ("I would like to know how the TCP three way handshake establishes a reliable connection between hosts", "analysis"),
    ("Now give a longer description of the causes of the French Revolution and its effects on Europe", "analysis"),
    ("Tell me another interesting fact about octopus intelligence and how researchers have measured it over time", "analysis"),
    ("Please take a note of this recipe and convert every measurement from imperial units to metric units", "analysis"),
    ("Whatever you think is best, recommend a good structure for a weekly study schedule for learning Spanish", "analysis"),
    ("I cannot get the client connection to stay open for more than thirty seconds at a time with websockets", "analysis"),
    ("Give an explanation of photosynthesis suitable for a twelve year old student who likes biology a lot", "analysis"),

    # --- ordering bug: short but explicit code / analysis requests go to the 30B
    ("Write a Python LRU cache class", "code"),
    ("Fix this bug: def add(a, b): return a - b", "code"),
    ("Explain the CAP theorem in two sentences.", "analysis"),
    ("Summarize the plot of Romeo and Juliet.", "analysis"),
    ("Compare Rust and Go for CLI tools", "code"),
    ("Why does my CUDA kernel run out of memory?", "code"),
    ("Optimize this SQL query", "code"),
    ("Debugging tips for segfaults in C++", "code"),

    # --- genuine quick questions stay quick
    ("What is 17 times 24?", "quick"),
    ("Who wrote Pride and Prejudice?", "quick"),
    ("Is Paris the capital of France? Yes or no.", "quick"),
    ("hi", "quick"),
    ("", "quick"),
    ("   ", "quick"),
    ("When did the Berlin Wall fall, and what were the main political reasons that led to it happening that year?", "quick"),

    # --- whole-word keywords still match their real forms
    ("Profiling shows the hot loop is in the tokenizer; I want to understand the results before changing anything", "code"),
    ("We are planning a migration of the reporting service to a new database over the next quarter carefully", "analysis"),
    ("Can you design an onboarding flow for new employees joining a small remote company with ten people total", "analysis"),

    # --- code follow-ups. Found by driving the app in a browser: after a request
    # for a merge function, "Now change that function so it removes duplicates
    # from the merged result" routed to quick -- 12 words, and none of the
    # original keywords are in it. A follow-up is where a coding session spends
    # most of its turns, so these are the cases that matter most.
    ("Now change that function so it removes duplicates from the merged result.", "code"),
    ("now refactor it to use a heap", "code"),
    ("implement binary search", "code"),
    ("Why does this regex not match the last group?", "code"),
    ("commit that to git and push", "code"),
    ("the JSON it returns has the fields in the wrong order", "code"),

    # --- and the words deliberately NOT in the code list, because they are
    # ordinary English at least as often as they are technical
    ("what time is my yoga class tonight", "quick"),
    ("test my knowledge of Spanish vocabulary please", "quick"),
    # Accepted false positive: "function" is worth matching for the case above,
    # and the cost is a mislabelled bucket -- both buckets are served by the 30B.
    ("what is the function of the mitochondria", "code"),
]


def test_cases():
    failures = []
    for prompt, expected in CASES:
        got = prompt_bucket(prompt)
        if got != expected:
            failures.append((prompt, expected, got))
    assert not failures, "\n" + "\n".join(
        f"  {p!r}\n    expected {e}, got {g}" for p, e, g in failures)


if __name__ == "__main__":
    bad = 0
    for prompt, expected in CASES:
        got = prompt_bucket(prompt)
        ok = got == expected
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {expected:>8} <- {got:<8} {prompt[:70]!r}")
    print(f"\n{len(CASES) - bad}/{len(CASES)} passed")
    sys.exit(1 if bad else 0)
