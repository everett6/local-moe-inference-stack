"""
Fetch the two public benchmarks the model-quality eval scores against, from
Hugging Face's datasets-server REST API (no `datasets` dependency):

  data/quality/gsm8k_test_first250.jsonl   openai/gsm8k, main, test rows 0-249
                                           {"question", "answer"}  (answer = the
                                           number after "####")
  data/quality/humaneval.jsonl             openai/openai_humaneval, all 164
                                           {"task_id", "prompt", "entry_point", "test"}

50 GSM8K questions (gsm8k_test_first50.jsonl, the first 50 of these) couldn't
separate the models: Q4_K_M 50, Q2_K 47-49 is inside the noise. 250 cuts the
standard error by more than half. HumanEval covers code, which GSM8K doesn't.
"""
import json
import os
import time
import urllib.parse
import urllib.request

AI2 = "/home/everett/AI2"
OUT = os.path.join(AI2, "data", "quality")
API = "https://datasets-server.huggingface.co/rows"
PAGE = 100


def rows(dataset, config, split, total):
    out = []
    while len(out) < total:
        url = (f"{API}?dataset={urllib.parse.quote(dataset)}&config={urllib.parse.quote(config)}"
               f"&split={split}&offset={len(out)}&length={min(PAGE, total - len(out))}")
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    page = [x["row"] for x in json.loads(r.read())["rows"]]
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
        if not page:
            break
        out.extend(page)
    return out


def main():
    gsm = rows("openai/gsm8k", "main", "test", 250)
    with open(os.path.join(OUT, "gsm8k_test_first250.jsonl"), "w") as f:
        for r in gsm:
            f.write(json.dumps({"question": r["question"],
                                "answer": r["answer"].split("####")[-1].strip().replace(",", "")}) + "\n")
    he = rows("openai/openai_humaneval", "openai_humaneval", "test", 164)
    with open(os.path.join(OUT, "humaneval.jsonl"), "w") as f:
        for r in he:
            f.write(json.dumps({k: r[k] for k in ("task_id", "prompt", "entry_point", "test")}) + "\n")
    print(f"gsm8k {len(gsm)} rows, humaneval {len(he)} rows")


if __name__ == "__main__":
    main()
