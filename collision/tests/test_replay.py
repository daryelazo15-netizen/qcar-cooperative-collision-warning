"""Tests of the frame-by-frame replay, the loss masks, the frame builder and the known-vehicle filter."""
import collections
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO)
from collision import replay as RP  # noqa: E402
from collision.frames import build_episode, build_frames, centre, OFFSETS  # noqa: E402
from collision.own_vehicles import bev_iou, remove_known_vehicles  # noqa: E402
from tracking.kalman import kalman_factory  # noqa: E402
from tracking.reference import ReferenceTracker  # noqa: E402

CFG = json.load(open(RP.DEFAULT_CONFIG))


def box(x, y=0.0, conf=0.97):
    return {"x": x, "y": y, "yaw": 0.0, "length": 0.42, "width": 0.19, "confidence": conf}


def episode(n=14, dt=0.27, v=0.5, x0=2.0, lost=(), valid_first=False, lost_empty=True):
    """Parked ego at the origin, a car straight ahead closing at v m/s. Cooperative stream sees it on every frame; the ego-only stream does too."""
    frames = []
    for k in range(n):
        x = x0 - v * k * dt
        frames.append({"frame_id": str(k), "time_s": 10.0 + k * dt, "ego_motion_valid": k > 0 or valid_first,
                       "ego": {"x": 0.0, "y": 0.0, "yaw": 0.0, "vx": 0.0, "vy": 0.0},
                       "cooperative": [box(x)], "ego_only": [box(x)]})
    return {"episode_id": "e1", "coordinate_frame": "world_si", "time_basis": "source_time",
            "ego_footprint": {"length": 0.425, "width": 0.192}, "frames": frames}


class TestMasks(unittest.TestCase):
    frames = [{"time_s": 0.1 * i} for i in range(20)]

    def test_ideal_and_burst_half_open(self):
        self.assertTrue(all(RP.delivery_mask(self.frames, "ideal")))
        m = RP.delivery_mask(self.frames, "burst", bursts=[(0.5, 1.0)])
        self.assertEqual([i for i, d in enumerate(m) if not d], [5, 6, 7, 8, 9])

    def test_bernoulli_is_deterministic_and_has_the_right_rate(self):
        a = RP.delivery_mask([{"time_s": i} for i in range(4000)], "bernoulli", seed=3, drop_probability=0.25)
        b = RP.delivery_mask([{"time_s": i} for i in range(4000)], "bernoulli", seed=3, drop_probability=0.25)
        self.assertEqual(a, b)
        self.assertAlmostEqual(a.count(False) / 4000, 0.25, delta=0.03)

    def test_invalid_masks(self):
        with self.assertRaises(ValueError):
            RP.delivery_mask(self.frames, "burst", bursts=[(1.0, 0.5)])
        with self.assertRaises(ValueError):
            RP.delivery_mask(self.frames, "bernoulli", drop_probability=1.5)
        with self.assertRaises(ValueError):
            RP.delivery_mask(self.frames, "nope")

    def test_episode_seed_is_a_pure_function(self):
        self.assertEqual(RP.episode_seed("3", "e1"), RP.episode_seed("3", "e1"))
        self.assertNotEqual(RP.episode_seed("3", "e1"), RP.episode_seed("3", "e2"))
        self.assertNotEqual(RP.episode_seed("4", "e1"), RP.episode_seed("3", "e1"))


class TestSettings(unittest.TestCase):
    def test_per_stream(self):
        self.assertEqual(RP.per_stream(0.9, "x"), {"cooperative": 0.9, "ego_only": 0.9})
        self.assertEqual(RP.per_stream({"coop": 0.9, "ego_only": 0.8}, "x"), {"cooperative": 0.9, "ego_only": 0.8})
        with self.assertRaises(ValueError):
            RP.per_stream({"coop": 0.9}, "x")

    def test_validate_settings(self):
        RP.validate_settings(CFG)
        bad = json.loads(json.dumps(CFG))
        bad["tracker"]["max_speed_mps"] = -1
        with self.assertRaises(ValueError):
            RP.validate_settings(bad)
        bad = json.loads(json.dumps(CFG))
        bad["min_detection_confidence"] = 1.2
        with self.assertRaises(ValueError):
            RP.validate_settings(bad)


class TestReplay(unittest.TestCase):
    def rows(self, ep, method, mask=None, factory=None):
        return RP.replay(ep, CFG, mask or [True] * len(ep["frames"]), method, factory)

    def test_warmup_row_and_row_fields(self):
        rows = self.rows(episode(), "cooperative_ideal")
        self.assertEqual(rows[0]["reason"], "ego_velocity_warmup")
        self.assertFalse(rows[0]["warning"])
        self.assertEqual(rows[1]["evidence_status"], "available")      # the track seen in frame 0 is confirmed by its second observation
        self.assertEqual(set(rows[3]), {"frame_id", "time_s", "partner_delivered", "observation_stream", "method", "warning", "evidence_status", "tracks"})
        self.assertEqual(rows[3]["observation_stream"], "cooperative")

    def test_approach_warns_once_inside_the_horizon(self):
        rows = self.rows(episode(), "cooperative_ideal")
        first = next(r for r in rows if r["warning"])
        # contact when x - 0.425 <= 0.8 * 0.5  ->  x <= 0.825  ->  t >= 2.35 s after the first frame
        self.assertTrue(2.2 <= first["time_s"] - 10.0 <= 2.9, first["time_s"])
        self.assertFalse(any(r["warning"] for r in self.rows(episode(v=0.0), "cooperative_ideal")))

    def test_confidence_threshold_per_stream(self):
        ep = episode()
        for f in ep["frames"]:
            f["cooperative"] = [box(f["cooperative"][0]["x"], conf=0.9)]
        self.assertFalse(any(r["warning"] for r in self.rows(ep, "cooperative_ideal")))
        self.assertTrue(any(r["warning"] for r in self.rows(ep, "ego_only")))      # the ego-only stream still has 0.97

    def test_ignoring_methods_do_not_depend_on_the_mask(self):
        ep = episode()
        mask = [k % 3 != 0 for k in range(len(ep["frames"]))]
        strip = lambda rows: [{k: v for k, v in r.items() if k != "partner_delivered"} for r in rows]       # the flag echoes the mask
        for m in ("ego_only", "cooperative_ideal"):
            self.assertEqual(strip(self.rows(ep, m, mask)), strip(self.rows(ep, m)))
        self.assertEqual([r["observation_stream"] for r in self.rows(ep, "cv", mask)], ["ego_only" if k % 3 == 0 else "cooperative" for k in range(14)])

    def test_stale_track_policies_on_a_lost_stream(self):
        ep = episode(n=14)
        for k in (8, 9):
            ep["frames"][k]["ego_only"] = []
        mask = [k not in (8, 9) for k in range(14)]
        mask[8] = mask[9] = False
        res = {m: self.rows(ep, m, mask) for m in ("no_fallback", "hold", "cv", "uncertainty", "uncertainty_nofb")}
        # frame 9: the cooperative track is 0.54 s old (< 0.6 s expiry); delivery is lost, the ego-only stream is empty
        self.assertEqual(res["no_fallback"][9]["tracks"], [])
        self.assertEqual(res["uncertainty_nofb"][9]["tracks"], [])
        for m in ("hold", "cv", "uncertainty"):
            self.assertEqual(res[m][9]["tracks"][0]["mode"], "fallback")
        self.assertTrue(res["cv"][9]["warning"] and res["uncertainty"][9]["warning"])
        self.assertFalse(res["no_fallback"][9]["warning"])
        # hold keeps the last state, cv extrapolates it
        self.assertLess(res["cv"][9]["tracks"][0]["target_state"]["x"], res["hold"][9]["tracks"][0]["target_state"]["x"])

    def test_tracker_factory_reaches_every_method(self):
        ep = episode()
        for m in RP.METHODS:
            ref = self.rows(ep, m)
            self.assertEqual(self.rows(ep, m, factory=ReferenceTracker), ref)
            kal = self.rows(ep, m, factory=kalman_factory(0.1))
            self.assertNotEqual(kal, ref, m)

    def test_input_validation(self):
        ep = episode()
        with self.assertRaises(ValueError):
            RP.replay(ep, CFG, [True] * 3, "cv")
        with self.assertRaises(ValueError):
            RP.replay(ep, CFG, [1] * 14, "cv")
        with self.assertRaises(ValueError):
            RP.replay(ep, CFG, [True] * 14, "nope")
        with self.assertRaises(ValueError):
            RP.replay(dict(ep, coordinate_frame="image"), CFG, [True] * 14, "cv")
        dup = dict(ep, frames=ep["frames"][:2] + [dict(ep["frames"][1])])
        with self.assertRaises(ValueError):
            RP.replay(dup, CFG, [True] * 3, "cv")
        bad = episode()
        bad["frames"][3]["ego"]["x"] = float("nan")
        with self.assertRaises(ValueError):
            self.rows(bad, "cv")

    def test_no_future_information(self):
        ep = episode()
        full = self.rows(ep, "uncertainty")
        for k in (4, 8, 12):
            part = dict(ep, frames=ep["frames"][:k + 1])
            self.assertEqual(RP.replay(part, CFG, [True] * (k + 1), "uncertainty")[-1], full[k])

    def test_cli(self):
        with tempfile.TemporaryDirectory() as d:
            src, out = os.path.join(d, "e.jsonl"), os.path.join(d, "rows.jsonl")
            with open(src, "w") as f:
                f.write(json.dumps(dict(episode(), evaluation={"ignored": True})) + "\n")
            run = lambda *a: subprocess.run([sys.executable, "-B", "-m", "collision.replay", "--episodes", src, "--out", out, *a], cwd=REPO,
                                            capture_output=True, text=True)
            r = run("--methods", "cv,uncertainty", "--loss", "bernoulli", "--drop-probability", "0.5", "--seed", "7", "--tracker", "kalman", "--sigma-a", "0.1")
            self.assertEqual(r.returncode, 0, r.stderr)
            with open(out) as f:
                got = [json.loads(l) for l in f]
            self.assertEqual(len(got), 28)
            ep = episode()
            mask = RP.delivery_mask(ep["frames"], "bernoulli", seed=RP.episode_seed("7", "e1"), drop_probability=0.5)
            want = RP.replay(ep, CFG, mask, "cv", kalman_factory(0.1))
            self.assertEqual([{k: v for k, v in g.items() if k != "episode_id"} for g in got if g["method"] == "cv"], json.loads(json.dumps(want)))
            r = run("--loss", "burst", "--burst", "11.0", "12.0")
            self.assertEqual(r.returncode, 0, r.stderr)
            with open(out) as f:
                rows = [r for r in map(json.loads, f) if r["method"] == "uncertainty"]
            self.assertEqual([x["partner_delivered"] for x in rows].count(False), sum(1 for f in ep["frames"] if 11.0 <= f["time_s"] < 12.0))
            self.assertNotEqual(run("--methods", "nope").returncode, 0)


class TestOwnVehicles(unittest.TestCase):
    def test_iou(self):
        a = (0, 0, 0, 0.4, 0.2)
        self.assertAlmostEqual(bev_iou(a, a), 1.0)
        self.assertEqual(bev_iou(a, (2, 0, 0, 0.4, 0.2)), 0.0)
        self.assertAlmostEqual(bev_iou(a, (0.2, 0, 0, 0.4, 0.2)), 1 / 3)

    def test_remove_known_vehicles(self):
        boxes = [dict(box(0.0), length=0.425, width=0.192), box(1.0)]
        kept, gone, who = remove_known_vehicles(boxes, {"ego": (0.0, 0.0, 0.0)})
        self.assertEqual((len(kept), len(gone), who), (1, 1, [{"ego"}]))
        kept, gone, _ = remove_known_vehicles(boxes, {"ego": (0.0, 0.0, 0.0)}, iou_min=1.1)
        self.assertEqual((len(kept), len(gone)), (2, 0))


class TestFrames(unittest.TestCase):
    def rows(self):
        out = []
        for k in range(5):
            out.append({"frame": str(k), "time": 100.0 + 0.1 * k, "image_time_vicon_s": 100.05 + 0.1 * k, "session": "s",
                        "ego": {"x": 0.1 * k, "y": 0.0, "yaw": 0.2}, "coop": [box(2.0)], "ego_only": []})
        return out

    def test_velocity_and_warmup_and_centre(self):
        fr = build_frames(self.rows())
        self.assertFalse(fr[0]["ego_motion_valid"])
        self.assertAlmostEqual(fr[2]["ego"]["vx"], 1.0)
        cx, cy = centre(0.0, 0.0, 0.2, OFFSETS["ego"])
        self.assertAlmostEqual(fr[0]["ego"]["x"], cx)
        self.assertAlmostEqual(fr[0]["time_s"], 100.05)

    def test_sorted_by_image_time_and_nonincreasing_skipped(self):
        rows = self.rows()
        shuffled = [rows[3], rows[0], rows[2], dict(rows[2], frame="dup"), rows[1], rows[4]]
        fr = build_frames(shuffled)
        self.assertEqual([f["frame_id"] for f in fr], ["0", "1", "2", "3", "4"])      # sorted by image time; the repeated time is skipped
        times = [f["time_s"] for f in fr]
        self.assertEqual(times, sorted(set(times)))

    def test_missing_image_time_is_refused(self):
        rows = self.rows()
        del rows[2]["image_time_vicon_s"]
        with self.assertRaises(ValueError):
            build_frames(rows)

    def test_pose_that_moves_without_time_is_refused(self):
        rows = self.rows()
        rows[3]["time"] = rows[2]["time"]
        with self.assertRaises(ValueError):
            build_frames(rows)

    def test_yaw_offset_only_on_matching_session_and_own_vehicle_removal(self):
        rows = self.rows()
        fr = build_frames(rows, yaw_offsets={"s": 3.0})
        self.assertAlmostEqual(fr[0]["ego"]["yaw"], 0.2 + math.radians(3.0))
        self.assertAlmostEqual(build_frames(rows, yaw_offsets={"other": 3.0})[0]["ego"]["yaw"], 0.2)
        # a detection on the ego footprint is removed, one on the cooperator's pose is removed, one elsewhere stays
        ego_c = centre(0.0, 0.0, 0.2, OFFSETS["ego"])
        peer_c = centre(3.0, 0.0, 0.0, OFFSETS["peer"])
        rows[0]["coop"] = [dict(box(ego_c[0], ego_c[1]), yaw=0.2, length=0.425, width=0.192),
                           dict(box(peer_c[0], peer_c[1]), length=0.425, width=0.192), box(1.0, 1.0)]
        counter = collections.Counter()
        fr = build_frames(rows, peer_poses=[(100.0, 3.0, 0.0, 0.0)], removed=counter)
        self.assertEqual([(b["x"], b["y"]) for b in fr[0]["cooperative"]], [(1.0, 1.0)])
        self.assertEqual(dict(counter), {("cooperative", "ego"): 1, ("cooperative", "coop"): 1})

    def test_episode_replays(self):
        ep = build_episode("e", self.rows())
        self.assertEqual(ep["coordinate_frame"], "world_si")
        rows = RP.replay(ep, CFG, [True] * 5, "cooperative_ideal")
        self.assertEqual(len(rows), 5)


if __name__ == "__main__":
    unittest.main()
