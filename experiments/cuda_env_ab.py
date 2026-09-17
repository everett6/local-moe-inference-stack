"""
Do llama.cpp's runtime CUDA environment variables make the 30B faster?

LM Studio's CUDA build is already right for this GPU: native Blackwell kernels
(sm_120 in libggml-cuda.so) and CUDA runtime 12.8. So rebuilding CUDA buys
nothing, but the build reads several environment variables at startup that can
be tested for free. What each does, from the source at 8172e65 (LM Studio's base
commit):

  GGML_CUDA_GRAPH_OPT=1       off by default. Extra CUDA-graph optimization:
                              finds fan-out/join points (e.g. Q/K/V) and runs
                              branches on separate streams to allow more fusion.
  GGML_CUDA_REGISTER_HOST=1   off by default. Pins host buffers via
                              cudaHostRegister. In upstream 8172e65 nothing in
                              llama-server calls the function it gates (it is
                              only exported), so this is expected to be a no-op
                              unless LM Studio's fork calls it. VmPin in
                              /proc/<pid>/status shows whether anything got
                              pinned.
  GGML_CUDA_PDL=0             PDL kernel launches are ON by default when
                              compiled in (the variable's name is in LM Studio's
                              library, so it is). This checks the default is
                              right.
  GGML_CUDA_ENABLE_UNIFIED_MEMORY=1
                              every GPU allocation becomes cudaMallocManaged,
                              so VRAM can overflow into system RAM. Tested with
                              --n-cpu-moe 0 (all experts "on the GPU", ~18 GB on
                              a 12 GB card): the one setting that could change
                              the split question entirely. The docs say it hurts
                              performance, so it gets a hard per-request timeout.

Not tested: GGML_CUDA_DISABLE_FUSION, GGML_CUDA_DISABLE_GRAPHS and
GGML_CUDA_NO_PINNED only switch existing optimizations OFF.
GGML_CUDA_CUBLAS_COMPUTE_TYPE only changes prefill matmul precision (f32 is
slower, bf16 ~ f16).

All arms use the stock LM Studio server at the app's split (--n-cpu-moe 23),
except unified memory. Two interleaved rounds. Measured: decode tok/s (3 prompts
x 160 tokens), prefill tok/s (~2,950-token prompt), server RSS, pinned memory,
the process's VRAM, and whether greedy output matches the baseline.
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
PORT = 8084
ROUNDS = int(os.environ.get("ROUNDS", "2"))

# (name, env overrides, --n-cpu-moe, per-request timeout s)
ARMS = [
    ("baseline", {}, "23", 900),
    ("graph_opt", {"GGML_CUDA_GRAPH_OPT": "1"}, "23", 900),
    ("register_host", {"GGML_CUDA_REGISTER_HOST": "1"}, "23", 900),
    ("pdl_off", {"GGML_CUDA_PDL": "0"}, "23", 900),
    ("graph_opt+reg", {"GGML_CUDA_GRAPH_OPT": "1", "GGML_CUDA_REGISTER_HOST": "1"}, "23", 900),
    ("unified_mem_all_gpu", {"GGML_CUDA_ENABLE_UNIFIED_MEMORY": "1"}, "0", 240),
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


def post(payload, timeout):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def proc_status(pid):
    out = {}
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                k = line.split(":")[0]
                if k in ("VmRSS", "VmPin", "VmLck"):
                    out[k] = round(int(line.split()[1]) / 1048576, 2)
    except OSError:
        pass
    return out


def proc_vram_mb(pid):
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        for line in out.splitlines():
            p, mem = [x.strip() for x in line.split(",")]
            if int(p) == pid:
                return int(mem)
    except Exception:
        pass
    return None


def run(name, env_over, ncmoe, timeout, rnd):
    log_path = os.path.join(AI2, "state", f"cudaenv_{name}_r{rnd}.log")
    env = dict(os.environ)
    for k in list(env):
        if k.startswith("GGML_CUDA_"):
            del env[k]                       # nothing leaks in from the calling shell
    env.update(env_over)
    env["LD_LIBRARY_PATH"] = f"{VENDOR}:{STOCK}"
    cmd = [os.path.join(STOCK, "llama-server"), "-m", BIG, "--port", str(PORT), "-ngl", "999",
           "--n-cpu-moe", ncmoe, "-c", "4096", "-fa", "on", "-t", "16", "-np", "1", "--no-warmup"]
    t0 = time.time()
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
        try:
            deadline = time.time() + 600
            while time.time() < deadline:
                if proc.poll() is not None:
                    return {"error": "server exited during load"}
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3) as r:
                        if r.status == 200:
                            break
                except Exception:
                    time.sleep(1)
            else:
                return {"error": "load timeout"}
            load_s = round(time.time() - t0, 1)
            post({"prompt": PREFILL_PROMPT, "n_predict": 8, "temperature": 0, "cache_prompt": False}, timeout)
            tok = ms = 0.0
            for p in DECODE_PROMPTS:
                t = post({"prompt": p, "n_predict": 160, "temperature": 0, "cache_prompt": False}, timeout)["timings"]
                tok += t["predicted_n"]
                ms += t["predicted_ms"]
            pf = post({"prompt": PREFILL_PROMPT, "n_predict": 1, "temperature": 0, "cache_prompt": False},
                      timeout)["timings"]
            check = post({"prompt": CHECK_PROMPT, "n_predict": 96, "temperature": 0, "cache_prompt": False},
                         timeout)["content"]
            st = proc_status(proc.pid)
            return {"decode_tps": 1000 * tok / ms, "prefill_tps": pf["prompt_per_second"],
                    "rss_gb": st.get("VmRSS"), "pinned_gb": st.get("VmPin"), "locked_gb": st.get("VmLck"),
                    "vram_mb": proc_vram_mb(proc.pid), "load_s": load_s, "check_output": check}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)


def main():
    samples = {n: [] for n, *_ in ARMS}
    out_path = os.path.join(AI2, "experiments", "cuda_env_ab_result.json")
    for rnd in range(ROUNDS):
        for name, env_over, ncmoe, timeout in ARMS:
            r = run(name, env_over, ncmoe, timeout, rnd)
            samples[name].append(r)
            if "error" in r:
                print(f"  r{rnd} [{name}] ERROR {r['error']}", file=sys.stderr)
            else:
                print(f"  r{rnd} [{name}] decode {r['decode_tps']:6.2f}  prefill {r['prefill_tps']:7.1f}  "
                      f"rss {r['rss_gb']} GB  pinned {r['pinned_gb']} GB  vram {r['vram_mb']} MiB  "
                      f"load {r['load_s']}s", file=sys.stderr)
            json.dump(samples, open(out_path, "w"), indent=2, ensure_ascii=False)

    ref = next((s for s in samples["baseline"] if "error" not in s), None)
    base = statistics.median(s["decode_tps"] for s in samples["baseline"] if "error" not in s)
    base_pf = statistics.median(s["prefill_tps"] for s in samples["baseline"] if "error" not in s)
    print("\n=== CUDA ENV A/B ===")
    print("%-20s %8s %8s %9s %8s %7s %7s %s" % ("arm", "decode", "vs base", "prefill", "vs base",
                                                "RSS GB", "pin GB", "same output"))
    for name, *_ in ARMS:
        ok = [s for s in samples[name] if "error" not in s]
        if not ok:
            print("%-20s ERROR %s" % (name, samples[name][0]["error"]))
            continue
        dec = statistics.median(s["decode_tps"] for s in ok)
        pf = statistics.median(s["prefill_tps"] for s in ok)
        same = all(s["check_output"] == ref["check_output"] for s in ok)
        rng = "%.1f-%.1f" % (min(s["decode_tps"] for s in ok), max(s["decode_tps"] for s in ok))
        print("%-20s %8.2f %7.2fx %9.1f %7.2fx %7s %7s %s   (decode range %s)" % (
            name, dec, dec / base, pf, pf / base_pf, ok[-1]["rss_gb"], ok[-1]["pinned_gb"], same, rng))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
