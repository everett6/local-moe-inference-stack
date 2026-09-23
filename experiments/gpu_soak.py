"""
GPU soak test: can this box hold a near-all-GPU model under sustained load?

Why: at 00:41 on 2026-09-17, 2-3 minutes into Q2_K's accuracy run at split 2
(46 of 48 layers' experts on the GPU), the kernel logged "NVRM: Xid 79, GPU has
fallen off the bus" and then "Xid 154, Node Reboot Required". Q2_K had run about
an hour of earlier tests without a fault, so it doesn't reproduce on demand. Xid
79 is usually power delivery, PCIe signal integrity, or heat. Q2_K (2x speed)
can't be the app's default until this passes. See PLAN.md, Plan 2.

What it does, for DURATION_MIN minutes (default 45):
  - BigModelServer on MODEL (default q2_k) at the app's fitting and settings.
  - Continuous generation: alternating long code and prose replies, back to back,
    the load that crashed.
  - Every 2 s: nvidia-smi power draw, temperature, utilization, P-state, clock
    throttle reasons, PCIe link gen/width, and the PCIe replay counter (link-level
    retransmits, which climb before a link drops).
  - Every 10 s: kernel log since the start, for any new NVRM Xid.
Stops at the first Xid, dead server, or failed request, and says which.

Output: experiments/gpu_soak_<model>_result.json (summary) and
state/gpu_soak_<model>_telemetry.csv (every sample).

Passes if the whole duration runs with no Xid, no server death, and a flat PCIe
replay counter. If Q2_K fails, re-run with MODEL=ud-q3_k_xl: clean there means the
fault is load-dependent.
"""
import csv
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import replace

AI2 = "/home/everett/AI2"
sys.path.insert(0, AI2)
from config import BIG_MODELS, Paths, Runtime  # noqa: E402
from local_engine import BigModelServer, ServerUnavailable  # noqa: E402

MODEL = os.environ.get("MODEL", "q2_k")
DURATION_MIN = float(os.environ.get("DURATION_MIN", "45"))
FIELDS = ["power.draw", "temperature.gpu", "utilization.gpu", "pstate", "clocks_throttle_reasons.active",
          "pcie.link.gen.current", "pcie.link.width.current", "memory.used"]
PROMPTS = [
    "Write a complete Python module implementing a B-tree with insert, delete, search and in-order "
    "traversal, with docstrings and unittest tests.",
    "Write a long, detailed essay on the history of computing from Babbage to modern GPUs.",
    "Write a complete asyncio-based chat server and client in Python with rooms, nicknames and history.",
    "Explain in depth how modern CPUs execute code: pipelining, branch prediction, out-of-order execution, "
    "caches, and speculative execution vulnerabilities.",
]


def sample_gpu():
    """One nvidia-smi sample as a dict; {'error': ...} if the GPU can't be queried."""
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={','.join(FIELDS)}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return {"error": (out.stdout + out.stderr).strip()[:200]}
        vals = [v.strip() for v in out.stdout.strip().splitlines()[0].split(",")]
        return dict(zip(FIELDS, vals))
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def pcie_replay_count():
    """The PCIe replay counter from `nvidia-smi -q`, or None."""
    try:
        out = subprocess.run(["nvidia-smi", "-q", "-d", "PCIE"], capture_output=True, text=True, timeout=10).stdout
        for line in out.splitlines():
            if "Replay Number" in line or "Replays" in line:
                return int(line.split(":")[-1].strip())
    except Exception:
        pass
    return None


def new_xids(since_epoch):
    """Kernel NVRM Xid lines logged since `since_epoch` (seconds)."""
    try:
        out = subprocess.run(["journalctl", "-k", "--no-pager", "--since", f"@{int(since_epoch)}"],
                             capture_output=True, text=True, timeout=20).stdout
        return [l for l in out.splitlines() if "NVRM: Xid" in l]
    except Exception:
        return []


def main():
    start = time.time()
    tag = MODEL.replace("-", "_")
    csv_path = os.path.join(AI2, "state", f"gpu_soak_{tag}_telemetry.csv")
    out_path = os.path.join(AI2, "experiments", f"gpu_soak_{tag}_result.json")
    summary = {"model": MODEL, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "duration_target_min": DURATION_MIN,
               "replay_start": pcie_replay_count()}
    stop = threading.Event()
    samples = []

    def telemetry():
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s"] + FIELDS + ["error"])
            last_xid_check = 0.0
            while not stop.is_set():
                s = sample_gpu()
                samples.append(s)
                w.writerow([round(time.time() - start, 1)] + [s.get(k, "") for k in FIELDS] + [s.get("error", "")])
                f.flush()
                if time.time() - last_xid_check >= 10:
                    last_xid_check = time.time()
                    x = new_xids(start)
                    if x:
                        summary["xid"] = x[:5]
                        stop.set()
                stop.wait(2.0)

    path, split = BIG_MODELS[MODEL]
    srv = None
    th = threading.Thread(target=telemetry, daemon=True)
    th.start()
    replies = tokens = 0
    try:
        srv = BigModelServer(replace(Paths(), big_model_gguf=path), replace(Runtime(), n_cpu_moe=split), port=8103)
        summary["n_cpu_moe"] = srv.n_cpu_moe
        deadline = start + DURATION_MIN * 60
        while time.time() < deadline and not stop.is_set():
            t = {}
            for ev in srv.stream_chat([{"role": "user", "content": PROMPTS[replies % len(PROMPTS)]}], 1536):
                t = ev.get("timings", t)
                if stop.is_set():
                    break
            replies += 1
            tokens += t.get("predicted_n", 0)
            print(f"  {(time.time() - start) / 60:5.1f} min  reply {replies}: {t.get('predicted_n', 0)} tok at "
                  f"{t.get('predicted_per_second', 0):.1f} tok/s  last sample {samples[-1] if samples else {}}",
                  file=sys.stderr, flush=True)
        summary["outcome"] = "xid" if "xid" in summary else "completed"
    except ServerUnavailable as e:
        summary["outcome"] = "server_unavailable"
        summary["error"] = str(e)[:2000]
    except Exception as e:
        summary["outcome"] = "request_failed"
        summary["error"] = f"{type(e).__name__}: {e}"[:2000]
    finally:
        stop.set()
        th.join(timeout=15)
        if srv is not None:
            srv.stop()
    time.sleep(3)
    summary["xid"] = summary.get("xid") or new_xids(start)[:5]
    if summary["xid"] and summary["outcome"] == "completed":
        summary["outcome"] = "xid"
    good = [s for s in samples if "error" not in s]

    def col(k):
        vals = []
        for s in good:
            try:
                vals.append(float(s[k]))
            except (KeyError, ValueError):
                pass
        return vals

    summary.update(
        minutes=round((time.time() - start) / 60, 1), replies=replies, tokens=tokens,
        replay_end=pcie_replay_count(), samples=len(samples), failed_samples=len(samples) - len(good),
        power_w_max=max(col("power.draw"), default=None), temp_c_max=max(col("temperature.gpu"), default=None),
        throttle_reasons_seen=sorted({s.get("clocks_throttle_reasons.active") for s in good} - {None}),
        pcie_links_seen=sorted({f"gen{s.get('pcie.link.gen.current')} x{s.get('pcie.link.width.current')}"
                                for s in good}),
    )
    passed = (summary["outcome"] == "completed" and not summary["xid"] and summary["minutes"] >= DURATION_MIN * 0.98
              and (summary["replay_start"] is None or summary["replay_end"] == summary["replay_start"]))
    summary["passed"] = passed
    json.dump(summary, open(out_path, "w"), indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "error"}, indent=2))
    print(("PASS" if passed else "FAIL") + f": {summary['outcome']} after {summary['minutes']} min; saved {out_path}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
