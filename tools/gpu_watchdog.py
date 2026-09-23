"""
Always-on GPU telemetry, so the next fault leaves evidence instead of a guess.

This card fell off the PCIe bus four times on 2026-09-17 ("NVRM: Xid 79, GPU has
fallen off the bus"), always at the stock 250 W limit. Every crash so far has had
to be reconstructed afterwards from whatever file happened to be open. This keeps
a rolling log running all the time:

  state/gpu_watchdog.csv   one line every SAMPLE_S seconds: power, temperature,
                           utilization, clocks, throttle reasons, PCIe link
                           gen/width, VRAM used, and the enforced power limit
  state/gpu_watchdog.log   one line per event: start, any new NVRM Xid in the
                           kernel log, the GPU becoming unreadable, the power
                           limit changing (it resets to stock on every reboot)

Rotates the CSV at ~20 MB, keeping one previous file, so it can run forever.
Needs no root. Install as a user service (starts at login, no sudo):

  systemctl --user enable --now ai2-gpu-watchdog

after writing ~/.config/systemd/user/ai2-gpu-watchdog.service:

  [Unit]
  Description=AI2 GPU telemetry watchdog
  [Service]
  ExecStart=/usr/bin/python3 /home/everett/AI2/tools/gpu_watchdog.py
  Restart=always
  [Install]
  WantedBy=default.target
"""
import csv
import os
import subprocess
import sys
import time

AI2 = "/home/everett/AI2"
STATE = os.path.join(AI2, "state")
CSV_PATH = os.path.join(STATE, "gpu_watchdog.csv")
LOG_PATH = os.path.join(STATE, "gpu_watchdog.log")
SAMPLE_S = float(os.environ.get("SAMPLE_S", "5"))
XID_EVERY_S = 30
MAX_BYTES = 20 * 1024 * 1024
FIELDS = ["power.draw", "power.limit", "temperature.gpu", "utilization.gpu", "clocks.sm", "pstate",
          "clocks_throttle_reasons.active", "pcie.link.gen.current", "pcie.link.width.current", "memory.used"]


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def sample():
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={','.join(FIELDS)}",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
        if out.returncode != 0:
            return None, (out.stdout + out.stderr).strip().splitlines()[0][:150]
        return [v.strip() for v in out.stdout.strip().splitlines()[0].split(",")], ""
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"[:150]


def xids_since(epoch):
    try:
        out = subprocess.run(["journalctl", "-k", "--no-pager", "--since", f"@{int(epoch)}"],
                             capture_output=True, text=True, timeout=30).stdout
        return [l.split("kernel:")[-1].strip() for l in out.splitlines() if "NVRM: Xid" in l]
    except Exception:
        return []


def main():
    os.makedirs(STATE, exist_ok=True)
    log(f"watchdog start (pid {os.getpid()}, sampling every {SAMPLE_S}s)")
    last_xid_check = time.time()
    last_limit = None
    gpu_was_readable = True
    while True:
        row, err = sample()
        if row is None:
            if gpu_was_readable:
                log(f"GPU UNREADABLE: {err}")
                gpu_was_readable = False
        else:
            if not gpu_was_readable:
                log("GPU readable again")
                gpu_was_readable = True
            limit = row[FIELDS.index("power.limit")]
            if last_limit is not None and limit != last_limit:
                log(f"power limit changed: {last_limit} -> {limit} W")
            last_limit = limit
            try:
                if os.path.getsize(CSV_PATH) > MAX_BYTES:
                    os.replace(CSV_PATH, CSV_PATH + ".1")
            except OSError:
                pass
            new_file = not os.path.exists(CSV_PATH)
            with open(CSV_PATH, "a", newline="") as f:
                w = csv.writer(f)
                if new_file:
                    w.writerow(["time"] + FIELDS)
                w.writerow([time.strftime("%Y-%m-%d %H:%M:%S")] + row)

        if time.time() - last_xid_check >= XID_EVERY_S:
            for line in xids_since(last_xid_check):
                log(f"KERNEL {line}")
            last_xid_check = time.time()
        time.sleep(SAMPLE_S)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("watchdog stopped")
        sys.exit(0)
