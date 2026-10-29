"""Link-impairment injector at the pairing step of the live loop (run_takes.py). Pure logic, no I/O except the optional trigger file.

A decision is taken for every (ego frame, peer frame) pair the loop is about to process. Specs (JSON, one per take):
  {"type": "none"}
  {"type": "window", "x": X, "y": Y}              peer withheld while  t0 + X <= now < t0 + X + Y   (t0 = READY, now = ego frame arrival, ws clock)
  {"type": "window_trigger", "y": Y, "trigger_file": P}   peer withheld for Y s from the moment the file P first exists (operator: `echo > P`)
  {"type": "bernoulli", "p": p, "seed": s}        peer withheld with probability p, one random.Random(seed) draw per processed pair
  {"type": "delay", "d": d [, "tol": 0.11]}       the peer frame used is the one nearest to (ego stamp - d) within `tol` (default 0.11 s, one frame period plus margin: the stream has gaps of one frame; the delay actually applied is logged per frame); none -> withheld
decide() returns {"status": "used"|"withheld"|"delayed", "peer": peer frame to use or None, "delay_s": ego stamp - used peer stamp (None if
withheld), "reason": ..., "spec": ...}; the loop logs it in every record.

Relation to the replay's loss schedules (collision/replay.py: burst windows in source time, Bernoulli per frame with a per-episode seed): `window` is a
burst anchored on READY (to be placed by timing the approach), `window_trigger` is anchored by the operator on an event they see, `bernoulli` is
the replay's pattern with a CLI seed (not the SHA-256 per-episode seed).
"""
import os
import random


class Injector:
    def __init__(self, spec=None, tol=0.11):
        self.spec = dict(spec or {"type": "none"})
        self.kind = self.spec["type"]
        if self.kind not in ("none", "window", "window_trigger", "bernoulli", "delay"):
            raise ValueError("unknown impairment type %r" % self.kind)
        if self.kind == "bernoulli":
            if not 0.0 <= self.spec["p"] <= 1.0:
                raise ValueError("p must be in [0, 1]")
            self.rng = random.Random(self.spec["seed"])
        if self.kind in ("window",) and (self.spec["x"] < 0 or self.spec["y"] <= 0):
            raise ValueError("window needs x >= 0 and y > 0")
        if self.kind == "window_trigger" and self.spec["y"] <= 0:
            raise ValueError("window_trigger needs y > 0")
        if self.kind == "delay" and self.spec["d"] < 0:
            raise ValueError("delay d must be >= 0")
        self.tol = float(self.spec.get("tol", tol))     # delay: one frame period (0.1 s at 10 Hz) plus margin; the sync drops about one frame in ten (nearest gap 0.095-0.099 s seen), so half a period is not enough
        self.t0 = None
        self.t_trigger = None

    def set_ready(self, t0):
        self.t0 = t0

    def needs_history(self):
        return self.kind == "delay"

    def decide(self, now, ego, peer, history=()):
        """ego, peer: frames with "stamp"; history: older peer frames (oldest first), only used by the delay type."""
        out = {"status": "used", "peer": peer, "delay_s": ego["stamp"] - peer["stamp"], "reason": None, "spec": self.kind}
        if self.kind == "none":
            return out
        if self.t0 is None and self.kind == "window":
            raise RuntimeError("set_ready() must be called before the first decision")
        withheld = None
        if self.kind == "window":
            if self.t0 + self.spec["x"] <= now < self.t0 + self.spec["x"] + self.spec["y"]:
                withheld = "window"
        elif self.kind == "window_trigger":
            if self.t_trigger is None and os.path.exists(self.spec["trigger_file"]):
                self.t_trigger = now
            if self.t_trigger is not None and now < self.t_trigger + self.spec["y"]:
                withheld = "trigger_window"
        elif self.kind == "bernoulli":
            draw = self.rng.random()
            out["draw"] = draw
            if draw < self.spec["p"]:
                withheld = "bernoulli"
        elif self.kind == "delay":
            target = ego["stamp"] - self.spec["d"]
            allf = list(history) + [peer]
            out["history_frames"] = len(allf)
            out["nearest_gap_s"] = min((abs(f["stamp"] - target) for f in allf), default=None)
            cands = [f for f in allf if abs(f["stamp"] - target) <= self.tol]
            if not cands:
                withheld = "no_delayed_frame"
            else:
                best = min(cands, key=lambda f: abs(f["stamp"] - target))
                out.update(status="delayed" if self.spec["d"] > 0 else "used", peer=best, delay_s=ego["stamp"] - best["stamp"])
        if withheld:
            out.update(status="withheld", peer=None, delay_s=None, reason=withheld)
        return out
