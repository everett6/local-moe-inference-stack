"""
Download the quantizations listed in models/quants/manifest.json, resumably,
and verify each against Hugging Face's own size and sha256.

Sequential on purpose: earlier parallel downloads on this connection competed
for bandwidth, and one silently stopped at 57% while curl still exited 0. So a
file only counts as done when its size AND sha256 match the manifest; anything
short is resumed (curl -C -), anything the right size with a wrong hash is
deleted and re-fetched.

Prints one line per event (progress every ~5%, done, failed) for a monitor.
"""
import hashlib
import json
import os
import subprocess
import sys
import time

AI2 = "/home/everett/AI2"
DIR = os.path.join(AI2, "models", "quants")
MAX_ATTEMPTS = 40


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(64 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fetch(item):
    path = os.path.join(DIR, item["file"])
    want = item["size"]
    for attempt in range(1, MAX_ATTEMPTS + 1):
        have = os.path.getsize(path) if os.path.exists(path) else 0
        if have > want:
            os.remove(path)
            have = 0
        if have == want:
            print(f"VERIFYING {item['file']}", flush=True)
            if sha256_of(path) == item["sha256"]:
                print(f"DONE {item['file']} ({want / 2**30:.2f} GiB, sha256 ok)", flush=True)
                return True
            print(f"BAD_HASH {item['file']}: deleting and re-downloading", flush=True)
            os.remove(path)
            continue
        # Run curl in the background of this loop so progress can be reported.
        proc = subprocess.Popen(
            ["curl", "-L", "-C", "-", "--retry", "5", "--retry-all-errors", "--speed-limit", "100000",
             "--speed-time", "60", "-s", "-o", path, item["url"]])
        last_pct = -5
        t0, b0 = time.time(), have
        while proc.poll() is None:
            time.sleep(10)
            cur = os.path.getsize(path) if os.path.exists(path) else 0
            pct = 100 * cur / want
            if pct >= last_pct + 5:
                rate = (cur - b0) / max(time.time() - t0, 1) / 2**20
                eta = (want - cur) / max(rate * 2**20, 1) / 60
                print(f"PROGRESS {item['file']} {pct:5.1f}% {rate:6.1f} MiB/s eta {eta:5.0f} min", flush=True)
                last_pct = pct - (pct % 5)
        if proc.returncode not in (0, None):
            print(f"RETRY {item['file']} attempt {attempt}: curl exit {proc.returncode}", flush=True)
            time.sleep(5)
    print(f"FAILED {item['file']} after {MAX_ATTEMPTS} attempts", flush=True)
    return False


def main():
    manifest = json.load(open(os.path.join(DIR, "manifest.json")))
    only = set(sys.argv[1:])
    ok = True
    for item in manifest:
        if only and item["file"] not in only:
            continue
        ok &= fetch(item)
    print("ALL_DONE" if ok else "ALL_DONE_WITH_FAILURES", flush=True)


if __name__ == "__main__":
    main()
