"""
Does an AVX-512 CPU backend make the 30B faster?

LM Studio only ships `avx2` llama.cpp builds; this is a Ryzen 9 7950X (Zen 4,
AVX-512). The CPU-side expert layers are ~55% of each token's time
(moe_offload_sweep: 0.321 ms per CPU-side layer), so a faster CPU backend is the
cheapest possible speedup: no engine redesign, just a different libggml-cpu.so.

How the builds were made (see build/README or PLAN.md):
  * llama.cpp source at 8172e65, the exact commit LM Studio's server reports.
  * Only the ggml-cpu target, built twice with the same compiler (GCC 15):
      avx2    -msse4.2 -mf16c -mfma -mbmi2 -mavx -mavx2   (control)
      znver4  -march=native                               (AVX-512)
  * Dropped into full copies of LM Studio's backend folder under build/.
    Symbol-compatible with LM Studio's other libraries (only libstdc++ template
    names differ). LM Studio's own install is never touched.

What a first probe found, and why this script has --no-host arms:
  Swapping in the AVX-512 library alone changed nothing (stock 71.4-72.1 tok/s,
  AVX-512 69.8-71.9). Both loaded the RAM-side experts identically: 8,422 MiB of
  plain memory-mapped Q4_K, no repacked buffer. llama.cpp can repack Q4_K into an
  interleaved q4_K_8x8 layout with dedicated SIMD kernels, but llama-model.cpp's
  make_cpu_buft_list puts the GPU's host buffer AHEAD of the CPU "extra" (repack)
  buffers, so with a GPU present those tensors never reach the repack path.
  `--no-host` removes the host buffer from that list.

Arms, all at the app's split (--n-cpu-moe 23) so only the CPU side differs:
  stock           LM Studio's server as the app runs it today
  stock_nohost    LM Studio's AVX2 library, repacked
  avx2_nohost     our GCC 15 AVX2 build, repacked       (compiler control)
  znver4_nohost   our GCC 15 AVX-512 build, repacked

Measured per launch (llama-server's own timings, after one untimed warm-up):
  decode tok/s over 3 prompts x 160 tokens; prefill tok/s on a ~2k-token prompt;
  server RSS (repacked weights live in anonymous memory, not the page cache);
  greedy output of a fixed prompt, compared to the stock arm.
Interleaved rounds so drift lands on every arm equally.
"""
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request

AI2 = "/home/everett/AI2"
VENDOR = "/home/everett/.lmstudio/extensions/backends/vendor/linux-llama-cuda12-vendor-v1"
STOCK = "/home/everett/.lmstudio/extensions/backends/llama.cpp-linux-x86_64-nvidia-cuda12-avx2-2.37.0"
BIG = "/home/everett/.lmstudio/models/lmstudio-community/Qwen3-30B-A3B-Instruct-2507-GGUF/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"
PORT = 8085
ROUNDS = int(os.environ.get("ROUNDS", "2"))
N_CPU_MOE = os.environ.get("N_CPU_MOE", "23")

ARMS = [
    ("stock", STOCK, []),
    ("stock_nohost", STOCK, ["--no-host"]),
    ("avx2_nohost", os.path.join(AI2, "build", "llama-server-avx2"), ["--no-host"]),
    ("znver4_nohost", os.path.join(AI2, "build", "llama-server-znver4"), ["--no-host"]),
]

DECODE_PROMPTS = [
    "Write a detailed explanation of how hash maps handle collisions.",
    "Write a Python class implementing a least-recently-used cache with get and put.",
    "Summarize the causes of the French Revolution in a few paragraphs.",
]
PREFILL_PROMPT = ("Here is a log of events from a distributed system. Summarize the failures.\n" +
                  "\n".join(f"[node-{i % 7}] t={1000 + i * 13}ms heartbeat ok, replica lag {i % 11}ms, "
                            f"queue depth {(i * 37) % 250}" for i in range(110)))
CHECK_PROMPT = "List the first ten prime numbers, then explain in two sentences why 1 is not prime."


def post(payload, timeout=900):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def rss_gb(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1048576, 2)
    except OSError:
        return None


def swap_used_gb():
    with open("/proc/meminfo") as f:
        m = {l.split(":")[0]: int(l.split()[1]) for l in f}
    return round((m["SwapTotal"] - m["SwapFree"]) / 1048576, 2)


def run(name, server_dir, extra, rnd):
    log_path = os.path.join(AI2, "state", f"cpuab_{name}_r{rnd}.log")
    env = dict(os.environ)
    # The server's own folder must come right after the vendored CUDA libs:
    # LD_LIBRARY_PATH outranks the binary's $ORIGIN runpath, so pointing it at
    # LM Studio's folder would silently load the stock CPU library in every arm.
    env["LD_LIBRARY_PATH"] = f"{VENDOR}:{server_dir}"
    cmd = [os.path.join(server_dir, "llama-server"), "-m", BIG, "--port", str(PORT), "-ngl", "999",
           "--n-cpu-moe", N_CPU_MOE, "-c", "4096", "-fa", "on", "-t", "16", "-np", "1", "--no-warmup"] + extra
    t_launch = time.time()
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
        try:
            deadline = time.time() + 600
            while time.time() < deadline:
                if proc.poll() is not None:
                    return {"error": "server exited during load", "log": log_path}
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3) as r:
                        if r.status == 200:
                            break
                except Exception:
                    time.sleep(1)
            else:
                return {"error": "load timeout", "log": log_path}
            load_s = time.time() - t_launch
            with open(f"/proc/{proc.pid}/maps") as f:
                cpu_libs = sorted({l.split()[-1] for l in f if "libggml-cpu" in l})
            post({"prompt": PREFILL_PROMPT, "n_predict": 8, "temperature": 0, "cache_prompt": False})  # warm-up
            tok = ms = 0.0
            for p in DECODE_PROMPTS:
                t = post({"prompt": p, "n_predict": 160, "temperature": 0, "cache_prompt": False})["timings"]
                tok += t["predicted_n"]
                ms += t["predicted_ms"]
            pf = post({"prompt": PREFILL_PROMPT, "n_predict": 1, "temperature": 0, "cache_prompt": False})["timings"]
            check = post({"prompt": CHECK_PROMPT, "n_predict": 96, "temperature": 0, "cache_prompt": False})["content"]
            return {"decode_tps": 1000 * tok / ms, "prefill_tps": pf["prompt_per_second"],
                    "prefill_n": pf["prompt_n"], "rss_gb": rss_gb(proc.pid), "load_s": round(load_s, 1),
                    "cpu_lib": cpu_libs, "check_output": check, "log": log_path}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}", "log": log_path}
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)


def main():
    print(f"split --n-cpu-moe {N_CPU_MOE}, {ROUNDS} rounds; swap in use at start: {swap_used_gb()} GB",
          file=sys.stderr)
    samples = {n: [] for n, _, _ in ARMS}
    for rnd in range(ROUNDS):
        for name, d, extra in ARMS:
            r = run(name, d, extra, rnd)
            samples[name].append(r)
            if "error" in r:
                print(f"  r{rnd} [{name}] ERROR {r['error']}", file=sys.stderr)
                continue
            print(f"  r{rnd} [{name}] decode {r['decode_tps']:6.2f}  prefill {r['prefill_tps']:7.1f} "
                  f"({r['prefill_n']} tok)  rss {r['rss_gb']} GB  load {r['load_s']}s  "
                  f"lib {os.path.dirname(r['cpu_lib'][0]).split('/')[-1] if r['cpu_lib'] else '?'}",
                  file=sys.stderr)
    out = os.path.join(AI2, "experiments", f"cpu_backend_ab_ncmoe{N_CPU_MOE}_result.json")
    json.dump({"n_cpu_moe": N_CPU_MOE, "swap_end_gb": swap_used_gb(), "samples": samples},
              open(out, "w"), indent=2, ensure_ascii=False)

    ref_ok = [s for s in samples["stock"] if "error" not in s]
    ref_out = ref_ok[0]["check_output"] if ref_ok else None
    base = statistics.median(s["decode_tps"] for s in ref_ok) if ref_ok else None
    base_pf = statistics.median(s["prefill_tps"] for s in ref_ok) if ref_ok else None
    print(f"\n=== CPU BACKEND A/B (--n-cpu-moe {N_CPU_MOE}) ===")
    print("%-14s %8s %8s %9s %8s %7s %s" % ("arm", "decode", "vs stock", "prefill", "vs stock", "RSS GB", "same output as stock"))
    for name, _, _ in ARMS:
        ok = [s for s in samples[name] if "error" not in s]
        if not ok:
            print("%-14s ERROR %s" % (name, samples[name][0].get("error")))
            continue
        dec = statistics.median(s["decode_tps"] for s in ok)
        pf = statistics.median(s["prefill_tps"] for s in ok)
        same = all(s["check_output"] == ref_out for s in ok) if ref_out is not None else "?"
        print("%-14s %8.2f %7.2fx %9.1f %7.2fx %7s %s" % (name, dec, dec / base, pf, pf / base_pf,
                                                          ok[-1]["rss_gb"], same))
    print(f"swap in use at end: {swap_used_gb()} GB\nsaved {out}")


if __name__ == "__main__":
    main()
