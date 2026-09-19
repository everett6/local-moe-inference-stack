"""
Which part of the GPU is failing: its memory, or its core/power delivery?

The card has dropped off the PCIe bus four times (NVRM Xid 79), always at the
stock 250 W limit, never at 175 W, and the owner sees the same freezes in Windows
under Minecraft -- so it is hardware. The card sits under a custom water loop
(core 47 C even at full load) and is mounted directly in the slot, so a riser is
ruled out and the two candidates left look very different under load:

  memory   a waterblock cools VRAM only through thermal pads. Wrong thickness or
           uneven mounting leaves memory hot while the core stays cold, and it
           fails only under sustained memory traffic. Both this model (streaming
           MoE weights) and shader-heavy Minecraft are memory-bandwidth workloads.
  core     power delivery or an unstable core/memory clock, which fails on
           current draw rather than on memory traffic.

So this runs two phases that load opposite halves of the chip, each for
PHASE_MIN minutes, and reports which one the card dies in:

  bandwidth  huge tensors copied and summed: ~all VRAM traffic, low compute,
             modest power. Also writes a known pattern and reads it back every
             pass, so silent VRAM corruption shows up as a mismatch count.
  compute    small matrices that fit in cache, multiplied over and over: high
             core utilization and high power, almost no VRAM traffic.
  transient  20 ms of that compute load, then 10 ms idle, over and over: the same
             peak current but swung on and off ~30 times a second. Inference does
             this once per token (~190/s) and a game does it once per frame, while
             a steady stress test does not -- and fast current swings (di/dt) are
             what marginal power delivery actually fails on.
  pcie       large host<->device copies over pinned memory: saturates the PCIe
             link with little GPU work. Xid 79 is a *bus* failure, the CPU-side
             expert layers cross the bus every token, and games stream over it
             constantly; the steady phases above barely touch it.

Telemetry every 2 s to state/gpu_stress_<phase>.csv, kernel Xids checked
continuously; it stops at the first fault and names the phase. Run it at the
power limit you are investigating (`nvidia-smi -q -d POWER` to check), with
tools/gpu_watchdog.py running for a second record.
"""
import csv
import json
import os
import subprocess
import sys
import time

AI2 = "/home/everett/AI2"
sys.path.insert(0, AI2)
PHASE_MIN = float(os.environ.get("PHASE_MIN", "10"))
PHASES = [p for p in os.environ.get("PHASES", "bandwidth,compute,transient,pcie").split(",") if p]
FIELDS = ["power.draw", "power.limit", "temperature.gpu", "utilization.gpu", "clocks.sm", "clocks.mem",
          "clocks_throttle_reasons.active", "pcie.link.gen.current", "memory.used"]


def sample():
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={','.join(FIELDS)}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15)
        if out.returncode != 0:
            return None
        return [v.strip() for v in out.stdout.strip().splitlines()[0].split(",")]
    except Exception:
        return None


def new_xids(since):
    try:
        out = subprocess.run(["journalctl", "-k", "--no-pager", "--since", f"@{int(since)}"],
                             capture_output=True, text=True, timeout=30).stdout
        return [l for l in out.splitlines() if "NVRM: Xid" in l]
    except Exception:
        return []


def run_phase(torch, phase, start_epoch):
    """Returns a result dict; raises nothing -- GPU faults come back as 'error'."""
    csv_path = os.path.join(AI2, "state", f"gpu_stress_{phase}.csv")
    f = open(csv_path, "w", newline="")
    w = csv.writer(f)
    w.writerow(["t_s"] + FIELDS)
    out = {"phase": phase, "mismatches": 0, "passes": 0}
    t0 = time.time()
    deadline = t0 + PHASE_MIN * 60
    last_sample = last_xid = 0.0
    try:
        if phase == "bandwidth":
            # ~6 GB of tensors, copied and reduced: memory-bound, plus a pattern
            # written and read back so VRAM corruption is visible.
            n = 384 * 1024 * 1024 // 4                      # 384 MiB of float32
            src = torch.full((n,), 1.2345, device="cuda", dtype=torch.float32)
            bufs = [torch.empty_like(src) for _ in range(12)]
            expect = src[0].item()
        elif phase == "pcie":
            # Pinned host buffers copied both ways: PCIe-bound, little GPU work.
            host = torch.empty((256 * 1024 * 1024 // 4,), dtype=torch.float32, pin_memory=True)
            host.fill_(2.5)
            dev = torch.empty_like(host, device="cuda")
            back = torch.empty_like(host)
        else:
            # Small enough to live in cache: compute- and power-bound, little VRAM.
            a = torch.randn((4096, 4096), device="cuda", dtype=torch.float16)
            b = torch.randn((4096, 4096), device="cuda", dtype=torch.float16)

        while time.time() < deadline:
            if phase == "bandwidth":
                for buf in bufs:
                    buf.copy_(src)
                total = sum(float(buf.sum().item()) for buf in bufs[:2])   # forces a read back
                bad = sum(int((buf[::4096] != expect).sum().item()) for buf in bufs)
                out["mismatches"] += bad
                if bad:
                    print(f"  VRAM MISMATCH: {bad} elements differ from the written pattern",
                          file=sys.stderr, flush=True)
                del total
            elif phase == "pcie":
                dev.copy_(host, non_blocking=True)
                back.copy_(dev, non_blocking=True)
                torch.cuda.synchronize()
                bad = int((back[::4096] != 2.5).sum().item())
                out["mismatches"] += bad
                if bad:
                    print(f"  PCIe TRANSFER MISMATCH: {bad} elements", file=sys.stderr, flush=True)
            elif phase == "transient":
                burst = time.time() + 0.02
                while time.time() < burst:
                    c = a @ b
                    a = (c * 0.0001).half()
                torch.cuda.synchronize()
                time.sleep(0.01)                      # let the rails snap back
            else:
                for _ in range(40):
                    c = a @ b
                    a = (c * 0.0001).half()
                torch.cuda.synchronize()
            out["passes"] += 1

            now = time.time()
            if now - last_sample >= 2:
                last_sample = now
                s = sample()
                if s is None:
                    out["error"] = "GPU became unreadable (nvidia-smi failed)"
                    break
                w.writerow([round(now - t0, 1)] + s)
                f.flush()
            if now - last_xid >= 10:
                last_xid = now
                x = new_xids(start_epoch)
                if x:
                    out["error"] = "Xid: " + x[-1][-120:]
                    break
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"[:300]
    finally:
        f.close()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    out["minutes"] = round((time.time() - t0) / 60, 1)
    rows = [r for r in csv.DictReader(open(csv_path))]
    for k, label in (("power.draw", "power_w"), ("temperature.gpu", "temp_c"), ("clocks.sm", "sm_mhz")):
        vals = [float(r[k]) for r in rows if r.get(k) not in (None, "", "N/A")]
        out[f"{label}_max"] = max(vals) if vals else None
        out[f"{label}_avg"] = round(sum(vals) / len(vals), 1) if vals else None
    return out


def main():
    import torch
    if not torch.cuda.is_available():
        print("CUDA is not available -- if the GPU fell off the bus, reboot first.", file=sys.stderr)
        sys.exit(2)
    start = time.time()
    results = []
    for phase in PHASES:
        print(f"=== {phase} phase, {PHASE_MIN} min ===", file=sys.stderr, flush=True)
        r = run_phase(torch, phase, start)
        results.append(r)
        print(f"  {phase}: {r.get('error', 'completed')} after {r['minutes']} min, "
              f"{r['passes']} passes, {r['mismatches']} VRAM mismatches, "
              f"max {r['power_w_max']} W / {r['temp_c_max']} C", file=sys.stderr, flush=True)
        json.dump(results, open(os.path.join(AI2, "experiments", "gpu_stress_discriminate_result.json"), "w"),
                  indent=2)
        if "error" in r:
            break

    print("\n=== GPU STRESS: WHICH HALF FAILS ===")
    for r in results:
        print(f"{r['phase']:10s} {r['minutes']:5.1f} min  max {r['power_w_max']} W  {r['temp_c_max']} C  "
              f"VRAM mismatches {r['mismatches']}  {r.get('error', 'completed clean')}")
    failed = [r for r in results if "error" in r]
    if not failed:
        print("\nBoth phases survived: this power limit and this load did not reproduce it.")
    else:
        cause = {
            "bandwidth": "MEMORY traffic while the core was light -- points at VRAM or its cooling "
                         "(thermal pad contact under the waterblock), or an unstable memory clock",
            "compute": "steady CORE/POWER load with little memory traffic -- points at power delivery "
                       "or core clock stability rather than the memory",
            "transient": "SWINGING load at the same peak power that steady load survived -- points at "
                         "power delivery under fast current changes: the PSU's response, the 12V-2x6 "
                         "cable and its contacts, or the card's VRM. This is what inference and games "
                         "do and what steady stress tests do not",
            "pcie": "PCIe TRAFFIC with little GPU work -- points at the bus itself: slot contacts, the "
                    "card's PCIe interface, or board-level signal integrity at Gen 5",
        }[failed[0]["phase"]]
        print(f"\nFailed under {cause}.")


if __name__ == "__main__":
    main()
