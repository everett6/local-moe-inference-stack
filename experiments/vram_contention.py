"""
Does a running llama-server survive another program taking all the VRAM it left?

Runtime.vram_headroom_mb keeps VRAM free after the big model loads. Whether the
server needs that for itself was the open question in PLAN.md. q2k_split_floor.py
already showed it allocates nothing after warm-up, even through a 2,950-token
prompt. This checks the other half: squeeze it.

  1. BigModelServer on Q2_K, fitted the way the app fits it (HEADROOM env var,
     default Runtime.vram_headroom_mb; HEADROOM=0 packs the card as tightly as
     it will run, split 1 on this desktop with ~126 MiB free).
  2. Reference replies: a short chat, and the long 2,950-token prompt + 512 tokens.
  3. A second process (PyTorch) allocates VRAM in 16 MiB blocks until CUDA refuses,
     and holds it.
  4. With the card full, the same requests again, twice, plus a 3-turn
     conversation. The server must answer every one, with identical text.
  5. Release the memory, one more round.
"""
import json
import os
import subprocess
import sys
import time
from dataclasses import replace

AI2 = "/home/everett/AI2"
sys.path.insert(0, AI2)
from config import BIG_MODELS, Paths, Runtime  # noqa: E402
from local_engine import BigModelServer, _gpu_free_mb  # noqa: E402

HOG = r"""
import sys, torch
blocks = []
try:
    torch.cuda.init()
    while True:
        blocks.append(torch.empty(16 * 2**20, dtype=torch.uint8, device="cuda"))
except Exception as e:
    print(f"HOG {len(blocks) * 16} MiB then {type(e).__name__}", flush=True)
sys.stdin.read()   # hold until the parent closes stdin
"""
LONG = ("Here is a log of events from a distributed system. Summarize the failures, then propose fixes.\n" +
        "\n".join(f"[node-{i % 7}] t={1000 + i * 13}ms heartbeat ok, replica lag {i % 11}ms, "
                  f"queue depth {(i * 37) % 250}" for i in range(110)))
REQUESTS = [
    ("short", [{"role": "user", "content": "Explain how a hash map handles collisions."}], 192),
    ("long", [{"role": "user", "content": LONG}], 512),
]
CONVO = ["Write a Python function that checks whether a string is a palindrome.",
         "Now make it ignore punctuation and case.",
         "Add doctests to it."]


def ask(srv, messages, n):
    parts, t = [], {}
    for ev in srv.stream_chat(messages, n):
        parts.append(ev.get("delta", ""))
        t = ev.get("timings", t)
    return "".join(parts), t


def conversation(srv):
    msgs = []
    for turn in CONVO:
        msgs.append({"role": "user", "content": turn})
        text, _ = ask(srv, msgs, 384)
        msgs.append({"role": "assistant", "content": text})
    return [m["content"] for m in msgs if m["role"] == "assistant"]


def round_(srv, label, ref):
    out = {"label": label, "free_mb": _gpu_free_mb(), "requests": []}
    for name, msgs, n in REQUESTS:
        try:
            text, t = ask(srv, msgs, n)
            out["requests"].append({"name": name, "ok": True, "same_as_reference": ref is None or text == ref[name],
                                    "tokens": t.get("predicted_n"), "tok_s": t.get("predicted_per_second"),
                                    "text": text})
        except Exception as e:
            out["requests"].append({"name": name, "ok": False, "error": f"{type(e).__name__}: {e}"})
    try:
        convo = conversation(srv)
        out["requests"].append({"name": "3-turn conversation", "ok": True,
                                "same_as_reference": ref is None or convo == ref["convo"], "text": convo})
    except Exception as e:
        out["requests"].append({"name": "3-turn conversation", "ok": False, "error": f"{type(e).__name__}: {e}"})
    out["server_alive"] = srv.proc.poll() is None
    out["free_after_mb"] = _gpu_free_mb()
    return out


def main():
    path, start = BIG_MODELS["q2_k"]
    headroom = int(os.environ.get("HEADROOM", Runtime().vram_headroom_mb))
    srv = BigModelServer(replace(Paths(), big_model_gguf=path),
                         replace(Runtime(), n_cpu_moe=start, vram_headroom_mb=headroom), port=8099)
    rounds, hog = [], None
    try:
        base = round_(srv, "reference (no contention)", None)
        rounds.append(base)
        ref = {r["name"]: r["text"] for r in base["requests"][:2]}
        ref["convo"] = base["requests"][2]["text"]
        hog = subprocess.Popen([sys.executable, "-c", HOG], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        hog_line = hog.stdout.readline().strip()
        for i in range(2):
            rounds.append(round_(srv, f"card full, round {i + 1} ({hog_line})", ref))
        hog.stdin.close()
        hog.wait(timeout=30)
        hog = None
        time.sleep(2)
        rounds.append(round_(srv, "after release", ref))
    finally:
        if hog is not None:
            hog.kill()
        srv.stop()
    result = {"headroom_mb": headroom, "n_cpu_moe": srv.n_cpu_moe, "rounds": rounds}
    json.dump(result, open(os.path.join(AI2, "experiments", f"vram_contention_headroom{headroom}_result.json"), "w"),
              indent=2, ensure_ascii=False)

    print(f"\n=== VRAM CONTENTION (Q2_K, split {srv.n_cpu_moe}, headroom {headroom}) ===")
    for r in rounds:
        print(f"{r['label']}: free {r['free_mb']} -> {r['free_after_mb']} MiB, server alive {r['server_alive']}")
        for q in r["requests"]:
            if q["ok"]:
                extra = f" {q['tokens']} tok at {q['tok_s']:.1f} tok/s" if q.get("tok_s") else ""
                print(f"   {q['name']:20s} ok, same text as reference: {q['same_as_reference']}{extra}")
            else:
                print(f"   {q['name']:20s} FAILED {q['error']}")


if __name__ == "__main__":
    main()
