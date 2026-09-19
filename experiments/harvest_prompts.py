"""
Build a training prompt set for draft-model self-distillation, from real
Hugging Face datasets rather than prompts we made up.

Why the source matters. A draft model is not being taught to be correct -- it is
being taught to *predict what our specific target model will say next*. Two
consequences:

  1. The prompts must look like the traffic the draft will actually face. Prompts
     invented ad hoc cluster around whatever phrasing the author happened to
     think of, and a draft tuned on them overfits to that narrow style. These
     come from curated, human/model-vetted instruction corpora.

  2. The *completions* in these datasets are irrelevant and are deliberately
     discarded. They were written by other models (GPT-3.5/4 era). Training on
     them would teach the draft to imitate OpenAI, which is exactly wrong -- the
     draft would then disagree with our Qwen3-30B target and get rejected. Only
     the prompts are taken; the targets come from our own quantized 30B in
     capture_target_outputs.py.

Mix, chosen to match this box's actual workload (see BENCHMARK_RESULTS.md's
prompt mix -- code, explanation, summarization):

  ise-uiuc/Magicoder-OSS-Instruct-75K    code instructions, seeded from real
                                         open-source code
  bigcode/self-oss-instruct-sc2-exec-filter-50k
                                         code instructions filtered by actually
                                         executing the solution
  HuggingFaceH4/ultrachat_200k           general assistant chat, the SFT split

Fetched over the HF datasets-server REST API, so this needs no `datasets`
install and touches nothing in the user's Python environment.
"""
import argparse
import json
import os
import random
import sys
import time
import urllib.parse
import urllib.request

AI2 = "/home/everett/AI2"
OUT = os.path.join(AI2, "data", "draft_train_prompts.jsonl")
API = "https://datasets-server.huggingface.co/rows"

# (dataset, config, split, field holding the prompt, how many to take)
SOURCES = [
    ("ise-uiuc/Magicoder-OSS-Instruct-75K", "default", "train", "problem", 1200),
    ("bigcode/self-oss-instruct-sc2-exec-filter-50k", "default", "train", "instruction", 800),
    ("HuggingFaceH4/ultrachat_200k", "default", "train_sft", "prompt", 1000),
]

PAGE = 100          # datasets-server caps rows per request
MIN_CHARS = 40      # drop stubs -- too short to produce a useful continuation
MAX_CHARS = 2000    # drop essays -- prefill would dominate the capture cost


def fetch_page(dataset, config, split, offset, length):
    url = (f"{API}?dataset={urllib.parse.quote(dataset)}"
           f"&config={urllib.parse.quote(config)}"
           f"&split={urllib.parse.quote(split)}"
           f"&offset={offset}&length={length}")
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.loads(r.read())["rows"]
        except Exception as e:
            if attempt == 4:
                raise
            # datasets-server rate-limits and occasionally 502s; back off.
            time.sleep(2 ** attempt)
    return []


def harvest(dataset, config, split, field, want, rng):
    rows = []
    # Start at a random offset so repeated runs don't always train on the same
    # head of the dataset.
    base = rng.randrange(0, 20000)
    offset = base
    seen = set()
    while len(rows) < want:
        page = fetch_page(dataset, config, split, offset, PAGE)
        if not page:
            if offset == base:
                break
            offset = 0          # wrap once, then give up if still empty
            base = 0
            continue
        for item in page:
            text = item["row"].get(field)
            if not isinstance(text, str):
                continue
            text = text.strip()
            if not (MIN_CHARS <= len(text) <= MAX_CHARS):
                continue
            key = text[:200]
            if key in seen:
                continue
            seen.add(key)
            rows.append({"prompt": text, "source": dataset})
            if len(rows) >= want:
                break
        offset += PAGE
        print(f"  {dataset}: {len(rows)}/{want}", file=sys.stderr, end="\r")
    print(f"  {dataset}: {len(rows)}/{want}    ", file=sys.stderr)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply every source's target count (0.1 for a smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    all_rows = []
    for dataset, config, split, field, want in SOURCES:
        n = max(1, int(want * args.scale))
        print(f"harvesting {dataset} ({n})", file=sys.stderr)
        all_rows.extend(harvest(dataset, config, split, field, n, rng))

    # Interleave sources so any later train/holdout split by index stays balanced.
    rng.shuffle(all_rows)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")

    by_source = {}
    for r in all_rows:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    print(f"\nwrote {len(all_rows)} prompts -> {args.out}")
    for k, v in sorted(by_source.items()):
        print(f"  {v:5d}  {k}")


if __name__ == "__main__":
    main()
