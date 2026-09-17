"""
Where do the app's missing tok/s go once the model is (almost) all on the GPU?

Q2_K at --n-cpu-moe 6 decoded 172.8 tok/s from a bare non-streaming chat request
(q2k_split_floor.py) but 156.9 through BigModelServer.stream_chat
(quant_speed_quality.py), same prompts, same split. At ~6 ms/token, a fixed
~0.6 ms per token of CPU-side work is 10%, where at Q4_K_M's ~13 ms it was
noise. stream_chat differs from the bare request in two ways, both tested here:

  stream        SSE: one JSON chunk per token, serialized by the server, parsed
                by the client.
  penalty       repeat_penalty 1.1, which the app sends so its replies match
                complete_greedy's. llama-server's default is 1.0 (off).

Plus one server flag:

  --backend-sampling   (experimental in this build) runs sampling on the GPU, so
                       the 151,936-entry logit vector isn't copied to the host
                       and walked by the CPU sampler chain every token.

Q2_K at split 3 (fits with ~540 MiB free). For each server (default, backend
sampling), the four request styles run interleaved, 3 rounds x 3 prompts x 192
tokens. Reported: llama-server's own decode timing and the client's wall-clock
rate (first to last token), and whether greedy output matches the plain request.
"""
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request

import requests

AI2 = "/home/everett/AI2"
sys.path.insert(0, AI2)
from config import Paths  # noqa: E402
from local_engine import _WARMUP_PROMPT  # noqa: E402

P = Paths()
MODEL = os.path.join(AI2, "models", "quants", "Qwen_Qwen3-30B-A3B-Instruct-2507-Q2_K.gguf")
SPLIT = os.environ.get("SPLIT", "3")
PORT = 8097
ROUNDS = 3
SERVERS = [("default", []), ("backend_sampling", ["--backend-sampling"])]
STYLES = [("plain", False, 1.0), ("penalty", False, 1.1), ("stream", True, 1.0), ("stream+penalty (app)", True, 1.1)]
PROMPTS = [
    "Explain how a hash map handles collisions.",
    "Write a Python class implementing a least-recently-used cache with get and put.",
    "Summarize the causes of the French Revolution in a few paragraphs.",
]


def request(prompt, stream, penalty):
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 192, "temperature": 0.0,
            "repeat_penalty": penalty, "cache_prompt": False, "stream": stream}
    url = f"http://127.0.0.1:{PORT}/v1/chat/completions"
    if not stream:
        t0 = time.perf_counter()
        r = requests.post(url, json=body, timeout=300).json()
        wall = time.perf_counter() - t0
        t = r["timings"]
        # Non-streaming wall time includes prefill; subtract the server's prefill.
        return r["choices"][0]["message"]["content"], t, t["predicted_n"] / max(wall - t["prompt_ms"] / 1000, 1e-9)
    r = requests.post(url, json=body, stream=True, timeout=300)
    text, timings, first, last, n = [], None, None, None, 0
    for raw in r.iter_lines():
        raw = raw.decode("utf-8")
        if not raw.startswith("data: ") or raw == "data: [DONE]":
            continue
        chunk = json.loads(raw[6:])
        timings = chunk.get("timings") or timings
        for c in chunk.get("choices", []):
            d = c.get("delta", {}).get("content")
            if d:
                now = time.perf_counter()
                first = first or now
                last = now
                n += 1
                text.append(d)
    return "".join(text), timings, (n - 1) / max(last - first, 1e-9)


def main():
    results = {}
    for sname, flags in SERVERS:
        env = dict(os.environ, LD_LIBRARY_PATH=P.llama_server_ld_library_path)
        cmd = [P.llama_server_bin, "-m", MODEL, "-c", "4096", "-t", "16", "--port", str(PORT), "-ngl", "999",
               "-fa", "on", "-np", "1", "--n-cpu-moe", SPLIT] + flags
        log = open(os.path.join(AI2, "state", f"reqoverhead_{sname}.log"), "w")
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            for _ in range(300):
                if proc.poll() is not None:
                    raise RuntimeError(f"{sname}: server exited during load")
                try:
                    if urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3).status == 200:
                        break
                except Exception:
                    time.sleep(1)
            requests.post(f"http://127.0.0.1:{PORT}/completion",
                          json={"prompt": _WARMUP_PROMPT, "n_predict": 8, "temperature": 0}, timeout=300)
            per = {s[0]: {"server": [], "client": [], "texts": []} for s in STYLES}
            for rnd in range(ROUNDS):
                for style, stream, penalty in STYLES:
                    tok = ms = 0.0
                    client = []
                    texts = []
                    for p in PROMPTS:
                        text, t, crate = request(p, stream, penalty)
                        tok += t["predicted_n"]
                        ms += t["predicted_ms"]
                        client.append(crate)
                        texts.append(text)
                    per[style]["server"].append(1000 * tok / ms)
                    per[style]["client"].append(statistics.mean(client))
                    per[style]["texts"] = texts
                    print(f"  [{sname}] r{rnd} {style:22s} server {1000 * tok / ms:6.1f}  client "
                          f"{statistics.mean(client):6.1f} tok/s", file=sys.stderr, flush=True)
            results[sname] = per
        finally:
            proc.terminate()
            proc.wait(timeout=30)
            log.close()
            time.sleep(2)
        json.dump(results, open(os.path.join(AI2, "experiments", "request_overhead_ab_result.json"), "w"),
                  indent=2, ensure_ascii=False)

    ref = results["default"]["plain"]["texts"]
    print(f"\n=== REQUEST OVERHEAD (Q2_K, split {SPLIT}; medians of {ROUNDS} rounds) ===")
    print("%-17s %-22s %8s %8s  %s" % ("server", "request", "server", "client", "same text as default/plain"))
    for sname, per in results.items():
        for style, *_ in STYLES:
            v = per[style]
            same = sum(a == b for a, b in zip(v["texts"], ref))
            print("%-17s %-22s %8.1f %8.1f  %d/3" % (sname, style, statistics.median(v["server"]),
                                                     statistics.median(v["client"]), same))


if __name__ == "__main__":
    main()
