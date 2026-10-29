"""Online collision wrapper: a synthetic approach that must warn, and online == batch on synthetic live files (all modes, both trackers)."""
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "live"))
import online_collision as OC  # noqa: E402
from collision import replay as RP  # noqa: E402
from collision.frames import build_episode  # noqa: E402
from perception.heal_export.world_boxes import box_world  # noqa: E402
from tracking.kalman import kalman_factory  # noqa: E402

CFG = RP.DEFAULT_CONFIG
SCRIPT = os.path.join(REPO, "live", "online_collision.py")


def corners(cx, cy, l=4.25, w=1.92):
    b = [[cx + dx, cy + dy, 0.0] for dx, dy in ((l / 2, w / 2), (l / 2, -w / 2), (-l / 2, -w / 2), (-l / 2, w / 2))]
    return b + [[p[0], p[1], 1.0] for p in b]


def synthetic(v=0.5, x0=2.0, dt=0.27, n=14, score=0.97, off_car=-0.01, withheld=(), empty_when_withheld=False):
    """Parked ego (corrected yaw 0); a vehicle straight ahead comes closer at v m/s; boxes in the ego frame, model units x10."""
    ego_raw_yaw = -math.radians(3.0)
    recs = []
    for k in range(n):
        t = 1000.0 + k * dt
        x = x0 - v * k * dt
        recs.append({"pair": k, "ego_stamp": t + off_car, "latency_from_ego_capture_sec": 0.4,
                     "ego_pose": {"x": -2.0, "y": -0.1, "z": 0.1, "yaw": ego_raw_yaw, "wall_time_unix": t + 0.01 + k * 1e-6},
                     "peer_pose": {"x": 1.5, "y": 2.5, "z": 0.1, "yaw": 0.0, "wall_time_unix": t + 0.01},
                     "stream": "ego_only" if k in withheld else "coop", "peer_status": "withheld" if k in withheld else "used",
                     "detections": [] if (empty_when_withheld and k in withheld) else [{"corners_ego_model": corners(x * 10, 0.0), "score": score}]})
    return recs


def offsets_file(d, off=-0.01):
    p = os.path.join(d, "off.json")
    json.dump({"hosts": [{"target": "nvidia@EGO_HOST", "offset_car_minus_ws_s": off}]}, open(p, "w"))
    return p


def batch_rows(recs, off, yaw, methods, factory=None):
    """The batch path from the same live records, independent of the wrapper's frame code: world rows -> collision.frames -> collision.replay."""
    world_rows, peer_poses, delivered = [], [], []
    for r in recs:
        e, p = r["ego_pose"], r["peer_pose"]
        pose = [e["x"] * 10, e["y"] * 10, 0.0, 0.0, math.degrees(e["yaw"] + math.radians(yaw)), 0.0]
        boxes = []
        for d in r["detections"]:
            b = box_world(d["corners_ego_model"], pose)
            b["confidence"] = d["score"]
            boxes.append(b)
        is_coop = r.get("stream", "coop") == "coop"
        world_rows.append({"frame": str(r["pair"]), "time": e["wall_time_unix"], "image_time_vicon_s": r["ego_stamp"] - off, "session": "live",
                           "ego": {"x": e["x"], "y": e["y"], "yaw": e["yaw"]},
                           "coop": boxes if is_coop else [], "ego_only": [] if is_coop else boxes})
        peer_poses.append((e["wall_time_unix"], p["x"], p["y"], p["yaw"]))
        delivered.append(is_coop)
    episode = build_episode("live", world_rows, peer_poses, 0.10, {"live": yaw})
    cfg = json.load(open(CFG))
    return episode, {m: RP.replay(episode, cfg, list(delivered), m, factory) for m in methods}


norm = lambda rows: json.loads(json.dumps(rows))


class T(unittest.TestCase):
    def run_online(self, recs, **kw):
        oc = OC.OnlineCollision(CFG, 3.0, -0.01, **kw)
        out = [oc.process(r, lambda proc: r["ego_stamp"] + 0.4 + proc) for r in recs]
        return oc, out

    def test_approach_warns_and_static_does_not(self):
        _, out = self.run_online(synthetic(v=0.5))
        starts = [(o[1]["t_capture"], o[2]) for o in out if o[2]]
        self.assertEqual(len(starts), 1)
        self.assertTrue(starts[0][1].startswith("WARNING method=cooperative_ideal stream=coop t_capture="))
        # contact when the gap closes: x - 0.425 <= 0.8 s * 0.5 m/s  ->  x <= 0.825 m  ->  t >= (2.0 - 0.825) / 0.5 = 2.35 s after the first frame
        t_first = out[0][1]["t_capture"]
        self.assertTrue(2.2 <= starts[0][0] - t_first <= 2.9, starts[0][0] - t_first)
        _, still = self.run_online(synthetic(v=0.0))
        self.assertFalse(any(o[2] for o in still))
        self.assertFalse(any(o[1].get("warning") for o in still))

    def test_low_confidence_and_known_vehicle_boxes_are_dropped(self):
        _, out = self.run_online(synthetic(v=0.5, score=0.9))        # below the 0.95 threshold of the default config
        self.assertFalse(any(o[1].get("warning") for o in out))
        self.assertEqual(max(o[1].get("tracks_active", 0) for o in out), 0)
        # a box on top of the ego footprint is removed by the known-vehicle rule
        _, out = self.run_online(synthetic(v=0.0, x0=0.0))
        self.assertTrue(all(o[1].get("removed_known_agent") == 1 or o[1].get("skipped") for o in out))

    def test_warmup_frame_and_nonincreasing_time(self):
        recs = synthetic()
        _, out = self.run_online(recs)
        self.assertFalse(out[0][1]["ego_motion_valid"])
        dup = recs[:3] + [dict(recs[2], pair=99)] + recs[3:5]
        _, out = self.run_online(dup)
        self.assertIn("skipped", out[3][1])

    def check_equivalence(self, recs, mode, factory=None, tracker_args=()):
        with tempfile.TemporaryDirectory() as d:
            live = os.path.join(d, "live.jsonl")
            with open(live, "w") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
            out = os.path.join(d, "o")
            res = subprocess.run([sys.executable, "-B", SCRIPT, "--live", live, "--offsets", offsets_file(d), "--out", out, "--yaw", "3.0",
                                  "--simulate", "--mode", mode, *tracker_args], capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, res.stderr)
            oc = OC.OnlineCollision(CFG, 3.0, -0.01, mode=mode, tracker_factory=factory)
            episode, batch = batch_rows(recs, -0.01, 3.0, oc.methods, factory)
            for k, m in enumerate(oc.methods):
                path = os.path.join(out, "online_rows.jsonl" if k == 0 else "online_rows_%s.jsonl" % m)
                got = [json.loads(l) for l in open(path)]
                self.assertEqual(len(got), len(episode["frames"]))
                self.assertEqual(got, norm(batch[m]), m)
            return res.stdout, batch

    def test_cli_online_equals_batch_with_a_warning(self):
        stdout, batch = self.check_equivalence(synthetic(v=0.5), "coop")
        self.assertEqual(sum(l.startswith("WARNING") for l in stdout.splitlines()), 1)
        self.assertGreater(sum(r["warning"] for r in batch["cooperative_ideal"]), 0)      # the warning path is inside the compared rows

    def test_cli_online_equals_batch_fallback_and_ego_only(self):
        _, batch = self.check_equivalence(synthetic(v=0.5, n=16, withheld=(5, 6, 9, 10, 11)), "fallback")
        self.assertEqual(sorted(batch), ["cv", "hold", "no_fallback", "uncertainty"])
        self.check_equivalence([dict(r, stream="ego_only") for r in synthetic(v=0.5)], "ego_only")

    def test_cli_online_equals_batch_kalman_tracker(self):
        self.check_equivalence(synthetic(v=0.5, n=16, withheld=(5, 6)), "fallback", kalman_factory(0.1), ("--tracker", "kalman", "--sigma-a", "0.1"))

    def test_fallback_mode_keeps_stale_track_unless_no_fallback(self):
        recs = synthetic(v=0.5, n=14, withheld=(8, 9), empty_when_withheld=True)
        oc = OC.OnlineCollision(CFG, 3.0, -0.01, mode="fallback")
        self.assertEqual(oc.methods, ["uncertainty", "no_fallback", "hold", "cv"])
        res = []
        for r in recs:
            oc.process(r, lambda proc: r["ego_stamp"] + 0.4 + proc)
            res.append({m: row["warning"] for m, row in oc.last_rows.items()})
        # frame 9 (t = 2.43 s, source age 0.54 s < 0.6 s expiry): the stale cooperative track is propagated by cv/uncertainty, not evaluated by no_fallback
        self.assertTrue(res[9]["cv"] and res[9]["uncertainty"])
        self.assertFalse(res[9]["no_fallback"])
        self.assertEqual(oc.last_rows["cv"]["observation_stream"], "cooperative")      # frame 13 delivered again
        # the stream of a withheld frame is ego_only
        oc2 = OC.OnlineCollision(CFG, 3.0, -0.01, mode="fallback")
        for r in recs[:9]:
            oc2.process(r, lambda proc: 0.0)
        self.assertEqual(oc2.last_rows["cv"]["observation_stream"], "ego_only")
        self.assertFalse(oc2.last_rows["cv"]["partner_delivered"])

    def test_ego_only_mode_uses_only_the_ego_only_stream(self):
        recs = [dict(r, stream="ego_only") for r in synthetic(v=0.5)]
        oc = OC.OnlineCollision(CFG, 3.0, -0.01, mode="ego_only")
        self.assertEqual(oc.methods, ["ego_only"])
        starts = 0
        for r in recs:
            _, out, line = oc.process(r, lambda proc: 0.0)
            starts += bool(line)
            self.assertEqual(oc.last_rows["ego_only"]["observation_stream"], "ego_only")
        self.assertEqual(starts, 1)


if __name__ == "__main__":
    unittest.main()
