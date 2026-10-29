"""Online collision module on the live detections: the replay's model, called frame by frame in arrival order.

It reads the live loop's output jsonl (tailing it while it is written, or all at once with --simulate) and writes one row per frame.
The frame construction, the tracker, the risk model and the evaluate() call are those of collision/replay.py; replaying the same frames with
collision.replay.replay gives the same rows (online == batch).

What a frame is (the same as collision.frames.build_frames, stream "cooperative", method "cooperative_ideal", config "coop" values):
  time_s   = ego capture time on the workstation clock = ego_stamp - offset(car - ws)
  ego      = footprint centre of the ego (motion-capture marker + OFFSETS["ego"] rotated by the RAW yaw), yaw = raw yaw + yaw offset,
             vx, vy = backward difference of the ego marker position over the step of the pose sample times (envelope wall_time_unix)
  cooperative = world boxes of the live detections (perception/heal_export/world_boxes.py box_world: centroid of the 4 bottom corners, long edge),
             minus the boxes whose BEV IoU >= 0.10 with the ego or the cooperating car's footprint (raw yaw, OFFSETS)
  min confidence and the warning horizon come from the config.
  Modes: the record says which stream produced its boxes ("stream": coop | ego_only; absent = coop). coop -> method
  cooperative_ideal; ego_only -> ego_only; fallback -> uncertainty (primary) + no_fallback, hold, cv, with delivered = (stream == coop), as the
  batch replay's delivery mask. The boxes go into the frame's matching stream list, the other list is empty.
No future frame is read.

    python3 live/online_collision.py --live <live.jsonl> --offsets <offset_before.json> --yaw 3.0 --out <dir> [--mode coop|ego_only|fallback]
                                     [--tracker reference|kalman --sigma-a 0.1] [--simulate] [--seconds N]
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from collision.detection import Detection  # noqa: E402
from collision.frames import EGO_FOOTPRINT, OFFSETS, box_fields, centre  # noqa: E402
from collision.geometry import Footprint, State  # noqa: E402
from collision.own_vehicles import remove_known_vehicles  # noqa: E402
from collision.replay import DEFAULT_CONFIG, METHODS, per_stream, stream_risk_configs, validate_settings  # noqa: E402
from collision.risk import FallbackCollisionModel  # noqa: E402
from perception.heal_export.world_boxes import box_world  # noqa: E402
from tracking.reference import ReferenceTracker  # noqa: E402


def sha256_file(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


MODE_METHODS = {"coop": ["cooperative_ideal"], "ego_only": ["ego_only"], "fallback": ["uncertainty", "no_fallback", "hold", "cv"]}
# fallback: `uncertainty` is the primary method (stale cooperative tracks are kept and propagated with a growing uncertainty envelope, the
# frame's own stream is ego_only when the peer is withheld); no_fallback / hold / cv are the other treatments of the same frames, logged too.


class MethodState:
    """One method of collision.replay.METHODS, advanced one frame at a time: the body of collision.replay.replay for one frame."""

    def __init__(self, oc, method):
        if method not in METHODS:
            raise ValueError("unknown replay method %r" % method)
        self.oc, self.method = oc, method
        self.tracker = oc.tracker_factory(dict(oc.cfg["tracker"], expire_source_age_s=oc.risk_config.expire_source_age_s))
        self.models, self.warning_on = {}, False

    def step(self, frame, delivered):
        oc, method = self.oc, self.method
        now = frame["time_s"]
        use_cooperative = (method == "cooperative_ideal" or (method != "ego_only" and delivered))
        stream = "cooperative" if use_cooperative else "ego_only"
        detections = [Detection(**item) for item in frame[stream]]
        detections = [item for item in detections if item.confidence >= oc.thresholds[stream]]
        updates = self.tracker.step(detections, now, stream)
        updated_ids = set()
        for estimate, target_shape in updates:
            if estimate.track_id not in self.models:
                self.models[estimate.track_id] = [FallbackCollisionModel(oc.risk_config), target_shape]
            model = self.models[estimate.track_id][0]
            model.update(estimate, now)
            self.models[estimate.track_id][1] = target_shape
            updated_ids.add(estimate.track_id)
        ego = State(**frame["ego"])
        if not all(math.isfinite(value) for value in vars(ego).values()):
            raise ValueError("Ego state must be finite even when targets are unavailable")
        results = []
        if not frame.get("ego_motion_valid", True):
            return {"frame_id": frame["frame_id"], "time_s": now, "partner_delivered": delivered, "observation_stream": stream,
                    "method": method, "warning": False, "evidence_status": "unknown", "reason": "ego_velocity_warmup", "tracks": []}
        for key, (model, target_shape) in sorted(self.models.items()):
            if method in ("no_fallback", "uncertainty_nofb") and key not in updated_ids:
                continue
            strategy = ("uncertainty" if method in ("uncertainty", "uncertainty_nofb") else ("hold" if method == "hold" else "cv"))
            model.config = oc.stream_configs[stream]
            result = model.evaluate(ego, now, oc.ego_shape, target_shape, strategy)
            results.append(asdict(result))
        active = [item for item in results if item["mode"] not in ("unavailable", "expired")]
        return {"frame_id": frame["frame_id"], "time_s": now, "partner_delivered": delivered, "observation_stream": stream,
                "method": method, "warning": any(item["warning"] for item in active),
                "evidence_status": "available" if active else "unknown", "tracks": results}


class OnlineCollision:
    def __init__(self, config_path, yaw_offset_deg, offset_car_minus_ws, known_iou=0.10, method=None, methods=None, mode=None, tracker_factory=None):
        self.cfg = json.load(open(config_path))
        validate_settings(self.cfg)
        self.stream_configs = stream_risk_configs(self.cfg)
        self.thresholds = per_stream(self.cfg["min_detection_confidence"], "min_detection_confidence")
        self.risk_config = self.stream_configs["cooperative"]
        self.ego_shape = Footprint(**EGO_FOOTPRINT)
        self.tracker_factory = tracker_factory or ReferenceTracker
        self.yaw_offset_deg, self.off, self.iou = yaw_offset_deg, offset_car_minus_ws, known_iou
        self.methods = list(methods or ([method] if method else MODE_METHODS[mode or "coop"]))
        self.primary = self.methods[0]
        self.state = {m: MethodState(self, m) for m in self.methods}
        self.prev = None
        self.skipped_nonincreasing = 0

    @property
    def tracker(self):
        return self.state[self.primary].tracker

    # ---- record -> frame (what collision.frames.build_frames does for one row)
    def frame_from_record(self, rec):
        e, p = rec["ego_pose"], rec["peer_pose"]
        t = rec["ego_stamp"] - self.off
        t_pose = e["wall_time_unix"]
        if self.prev is not None and t <= self.prev[0]:
            self.skipped_nonincreasing += 1
            return None, None
        ego_c = centre(e["x"], e["y"], e["yaw"], OFFSETS["ego"])
        ego_yaw = e["yaw"] + math.radians(self.yaw_offset_deg)
        vehicles = {"ego": (ego_c[0], ego_c[1], e["yaw"]),
                    "coop": centre(p["x"], p["y"], p["yaw"], OFFSETS["peer"]) + (p["yaw"],)}
        pose = [e["x"] * 10, e["y"] * 10, 0.0, 0.0, math.degrees(e["yaw"] + math.radians(self.yaw_offset_deg)), 0.0]
        boxes = []
        for d in rec["detections"]:
            b = box_world(d["corners_ego_model"], pose)
            b["confidence"] = d["score"]
            boxes.append(b)
        boxes = [box_fields(b) for b in boxes]
        kept, gone, who = remove_known_vehicles(boxes, vehicles, self.iou)
        vx, vy = 0.0, 0.0
        if self.prev:
            dtp = t_pose - self.prev[3]
            if dtp > 0:
                vx, vy = (e["x"] - self.prev[1]) / dtp, (e["y"] - self.prev[2]) / dtp
            elif not (dtp == 0 and e["x"] == self.prev[1] and e["y"] == self.prev[2]):
                raise ValueError("ego pose time does not advance (time %r after %r) although the pose changes or goes back" % (t_pose, self.prev[3]))
        stream = rec.get("stream", "coop")           # which stream produced the boxes of this record
        key = "cooperative" if stream == "coop" else "ego_only"
        frame = {"frame_id": str(rec["pair"]), "time_s": t, "ego_motion_valid": self.prev is not None,
                 "ego": {"x": ego_c[0], "y": ego_c[1], "yaw": ego_yaw, "vx": vx, "vy": vy},
                 "cooperative": kept if key == "cooperative" else [], "ego_only": kept if key == "ego_only" else []}
        self.prev = (t, e["x"], e["y"], t_pose)
        return frame, {"world_boxes": len(boxes), "removed_known_agent": len(gone), "stream": stream}

    def step(self, frame, delivered=True):
        """Row of the primary method (the batch replay's row for that frame)."""
        return self.state[self.primary].step(frame, delivered)

    def step_all(self, frame, delivered):
        return {m: st.step(frame, delivered) for m, st in self.state.items()}

    def process(self, rec, clock):
        """One live record -> (primary row, log entry, warning-start line or None). `clock(proc_s)` gives the current workstation time.
        Every method's row is in self.last_rows."""
        t0 = time.perf_counter()
        out = {"pair": rec["pair"], "ego_stamp": rec["ego_stamp"]}
        frame, info = self.frame_from_record(rec)
        if frame is None:
            out.update(skipped="non_increasing_capture_time")
            self.last_rows = {}
            return None, out, None
        delivered = info["stream"] == "coop"
        rows = self.step_all(frame, delivered)
        self.last_rows = rows
        proc = time.perf_counter() - t0
        t_now = clock(proc)
        per = {}
        for m, row in rows.items():
            active = [t for t in row["tracks"] if t["mode"] not in ("unavailable", "expired")]
            st = self.state[m]
            started = row["warning"] and not st.warning_on
            ended = st.warning_on and not row["warning"]
            st.warning_on = row["warning"]
            per[m] = {"warning": row["warning"], "warning_started": started, "warning_ended": ended, "tracks_active": len(active),
                      "tracks_total": len(row["tracks"]), "observation_stream": row["observation_stream"],
                      "evidence_status": row["evidence_status"],
                      "min_nominal_contact_s": min([t["nominal_contact_s"] for t in active if t["nominal_contact_s"] is not None], default=None),
                      "tracks": [{"id": t["track_id"], "mode": t["mode"], "warning": t["warning"], "nominal_contact_s": t["nominal_contact_s"],
                                  "conservative_contact_s": t["conservative_contact_s"],
                                  "x": t["target_state"]["x"] if t["target_state"] else None, "y": t["target_state"]["y"] if t["target_state"] else None,
                                  "vx": t["target_state"]["vx"] if t["target_state"] else None, "vy": t["target_state"]["vy"] if t["target_state"] else None}
                                 for t in active]}
        P = per[self.primary]
        out.update(t_capture=frame["time_s"], t_now=t_now, warning_latency_s=t_now - frame["time_s"], proc_ms=proc * 1000.0,
                   world_boxes=info["world_boxes"], removed_known_agent=info["removed_known_agent"], stream=info["stream"],
                   peer_status=rec.get("peer_status"), primary=self.primary, boxes_kept=len(frame["cooperative"]) + len(frame["ego_only"]),
                   tracks_active=P["tracks_active"], tracks_total=P["tracks_total"], warning=P["warning"], evidence_status=P["evidence_status"],
                   warning_started=P["warning_started"], warning_ended=P["warning_ended"], ego_motion_valid=frame["ego_motion_valid"],
                   ego_v=[frame["ego"]["vx"], frame["ego"]["vy"]], min_nominal_contact_s=P["min_nominal_contact_s"], tracks=P["tracks"],
                   methods={m: {k: v for k, v in d.items() if k != "tracks"} for m, d in per.items()})
        line = None
        if P["warning_started"]:
            w = [t for t in P["tracks"] if t["warning"]]
            line = "WARNING method=%s stream=%s t_capture=%.3f t_now=%.3f latency=%.3f s tracks=%s" % (
                self.primary, info["stream"], frame["time_s"], t_now, t_now - frame["time_s"],
                ";".join("%s contact_in=%.2fs at(%.2f,%.2f) v(%.2f,%.2f)" % (
                    t["id"], t["nominal_contact_s"], t["x"], t["y"], t["vx"], t["vy"]) for t in w))
        return rows[self.primary], out, line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", required=True, help="the live loop's output jsonl (tailed)")
    ap.add_argument("--offsets", required=True, help="offset_before.json of clock_offset.py (ego host car - ws offset)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--yaw", type=float, required=True, help="Vicon yaw offset in degrees (the live loop's)")
    ap.add_argument("--ego_host", default=os.environ.get("QCAR_EGO_HOST", "nvidia@EGO_HOST"))
    ap.add_argument("--mode", choices=sorted(MODE_METHODS), default="coop", help="coop: cooperative_ideal; ego_only: ego_only; fallback: uncertainty (+ no_fallback, hold, cv logged)")
    ap.add_argument("--methods", default=None, help="comma list overriding the mode's methods (first = primary, the one that prints WARNING)")
    ap.add_argument("--tracker", choices=("reference", "kalman"), default="reference")
    ap.add_argument("--sigma-a", type=float, default=0.1, help="process noise of the Kalman tracker, m/s^2")
    ap.add_argument("--simulate", action="store_true",
                    help="step 1: read the whole file at once; t_now = recorded output time (ego_stamp + latency) + measured processing time")
    ap.add_argument("--seconds", type=float, default=None, help="safety stop (tail mode)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    off = [h for h in json.load(open(a.offsets))["hosts"] if h["target"] == a.ego_host][0]["offset_car_minus_ws_s"]
    factory = None
    if a.tracker == "kalman":
        from tracking.kalman import kalman_factory
        factory = kalman_factory(a.sigma_a)
    oc = OnlineCollision(a.config, a.yaw, off, mode=a.mode, methods=a.methods.split(",") if a.methods else None, tracker_factory=factory)
    json.dump({"config_sha256": sha256_file(a.config), "tracker": a.tracker,
               "offset_car_minus_ws_s": off, "yaw_offset_deg": a.yaw, "mode": a.mode, "methods": oc.methods, "args": vars(a)}, open(os.path.join(a.out, "online_start.json"), "w"), indent=1)
    log = open(os.path.join(a.out, "online_log.jsonl"), "w", buffering=1)
    rows = open(os.path.join(a.out, "online_rows.jsonl"), "w", buffering=1)
    extra = {m: open(os.path.join(a.out, "online_rows_%s.jsonl" % m), "w", buffering=1) for m in oc.methods[1:]}

    def handle(rec, clock):
        try:
            row, entry, line = oc.process(rec, clock)
        except Exception as exc:  # a safety process must not die on one frame; the error is logged and the frame skipped
            entry, row, line = {"pair": rec.get("pair"), "error": "%s: %s" % (type(exc).__name__, exc)}, None, None
            print("ERROR pair %s: %s" % (rec.get("pair"), entry["error"]), flush=True)
        log.write(json.dumps(entry) + "\n")
        if row is not None:
            rows.write(json.dumps(row) + "\n")
            for m, f in extra.items():
                f.write(json.dumps(oc.last_rows[m]) + "\n")
        if line:
            print(line, flush=True)

    if a.simulate:
        for l in open(a.live):
            rec = json.loads(l)
            done = rec["ego_stamp"] + rec["latency_from_ego_capture_sec"]
            handle(rec, lambda proc, done=done: done + proc)
    else:
        end = None if a.seconds is None else time.time() + a.seconds
        while not os.path.exists(a.live):          # the live loop creates its jsonl when it starts
            if end is not None and time.time() > end:
                raise SystemExit("live file never appeared: %s" % a.live)
            time.sleep(0.1)
        print("online collision: tailing %s" % a.live, flush=True)
        with open(a.live) as f:
            buf = ""
            while end is None or time.time() < end:
                chunk = f.readline()
                if not chunk:
                    time.sleep(0.005)
                    continue
                buf += chunk
                if buf.endswith("\n"):
                    rec, buf = json.loads(buf), ""
                    handle(rec, lambda proc: time.time())
    log.close(), rows.close()
    for f in extra.values():
        f.close()


if __name__ == "__main__":
    main()
