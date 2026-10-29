"""Clock offset car - workstation, measured with the workstation's Windows clock.

Run with the Windows Python of the workstation (the clock run.py stamps
`arrival` with). For each host, ONE ssh session is opened and kept open; the
remote side answers each input line with `date +%s.%N`, so the ssh handshake
is not inside the round trip. For each query:

    t0 = local wall clock, send a line, read the car's time r, t1 = local wall clock
    offset (car - workstation) = r - (t0 + t1) / 2,  error bound +-(t1 - t0) / 2

The sample with the smallest round trip is the estimate (NTP's rule); the
median offset over all samples is printed as a cross-check.

Local wall time comes from GetSystemTimePreciseAsFileTime (same system clock
as Python's time.time(), without its coarse tick). time.time() is recorded
next to it, and its tick is reported, because run.py's `arrival` uses
time.time().

Read-only on the cars: the remote command only prints the date (and, once,
`timedatectl` NTP lines). Usage:

    python clock_offset.py --out offset.json --n 20 nvidia@EGO_HOST nvidia@PEER_HOST
"""
import argparse
import ctypes
import json
import os
import statistics
import subprocess
import sys
import time

EPOCH_DIFF_100NS = 116444736000000000  # 1601-01-01 to 1970-01-01

REMOTE = ('while read l; do if [ "$l" = NTP ]; then '
          'timedatectl 2>&1 | grep -iE "synchroni|NTP|time zone"; echo END; '
          'else date +%s.%N; fi; done')


def precise_now():
    ft = ctypes.c_ulonglong(0)
    ctypes.windll.kernel32.GetSystemTimePreciseAsFileTime(ctypes.byref(ft))
    return (ft.value - EPOCH_DIFF_100NS) / 1e7


def time_tick(seconds=1.0):
    """Smallest non-zero step of time.time() seen over `seconds`."""
    steps, last, end = [], time.time(), time.perf_counter() + seconds
    while time.perf_counter() < end:
        now = time.time()
        if now != last:
            steps.append(now - last)
            last = now
    return min(steps) if steps else float("nan"), statistics.median(steps) if steps else float("nan")


SSH = r"C:\Windows\System32\OpenSSH\ssh.exe"
# optional: ssh key file (default: ssh's own); needed when the process runs as a user whose profile has no key
KEY = os.environ.get("QCAR_SSH_KEY")


def query_host(target, n, warmup):
    cmd = [SSH] + (["-i", KEY] if KEY else []) + ["-o", "BatchMode=yes", "-o", "ConnectTimeout=5", target, REMOTE]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, bufsize=0)
    samples = []
    try:
        for i in range(warmup + n):
            t0 = precise_now()
            tt0 = time.time()
            proc.stdin.write(b"x\n")
            proc.stdin.flush()
            line = proc.stdout.readline()
            t1 = precise_now()
            tt1 = time.time()
            if not line:
                raise RuntimeError("ssh to %s closed: %s" % (target, proc.stderr.read().decode(errors="replace")))
            if i < warmup:
                continue
            r = float(line.decode().strip())
            samples.append({"t0": t0, "t1": t1, "time_time_t0": tt0, "time_time_t1": tt1,
                            "car": r, "rtt": t1 - t0, "offset": r - (t0 + t1) / 2})
            time.sleep(0.05)
        proc.stdin.write(b"NTP\n")
        proc.stdin.flush()
        ntp = []
        while True:
            line = proc.stdout.readline().decode(errors="replace").rstrip()
            if not line or line == "END":
                break
            ntp.append(line.strip())
    finally:
        try:
            proc.stdin.close()
        except OSError:
            pass
        proc.wait(timeout=10)
    best = min(samples, key=lambda s: s["rtt"])
    return {"target": target, "n": len(samples),
            "offset_car_minus_ws_s": best["offset"], "error_bound_s": best["rtt"] / 2,
            "best_rtt_s": best["rtt"],
            "median_offset_s": statistics.median(s["offset"] for s in samples),
            "median_rtt_s": statistics.median(s["rtt"] for s in samples),
            "offset_range_s": [min(s["offset"] for s in samples), max(s["offset"] for s in samples)],
            "timedatectl": ntp, "samples": samples}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="+", help="user@host of each car")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", required=True)
    opt = ap.parse_args()
    if sys.platform != "win32":
        raise SystemExit("run with the workstation's Windows Python")
    tick_min, tick_median = time_tick()
    res = {"started_precise": precise_now(), "python": sys.version,
           "time_time_clock_info": str(time.get_clock_info("time")),
           "time_time_tick_min_s": tick_min, "time_time_tick_median_s": tick_median,
           "hosts": []}
    for t in opt.targets:
        h = query_host(t, opt.n, opt.warmup)
        res["hosts"].append(h)
        print("%s: offset car-ws %+.4f s (+-%.4f, best rtt %.4f; median offset %+.4f, median rtt %.4f) %s"
              % (t, h["offset_car_minus_ws_s"], h["error_bound_s"], h["best_rtt_s"],
                 h["median_offset_s"], h["median_rtt_s"], h["timedatectl"]))
    res["finished_precise"] = precise_now()
    with open(opt.out, "w") as f:
        json.dump(res, f, indent=1)
    print("time.time() tick: min %.4f s, median %.4f s" % (tick_min, tick_median))
    print("written", opt.out)


if __name__ == "__main__":
    main()
