"""Alarm logic of the viewer on a recorded online log (fixtures/online_log_approach.jsonl: 41 lines, warning on pairs 26, 28, 29; two warning starts) and the drawn banner.

  python3 -B -m unittest discover -s live/tests

The drawing test stubs the two HEAL imports of LiveViewer (qcar camera extrinsics, draw_boxes): it checks the banner, the border, the tracks and
the snapshot files of the viewer's own code, not the box projection.
"""
import json
import os
import shutil
import sys
import tempfile
import types
import unittest

try:
    import cv2  # noqa: F401
    import numpy as np
except ImportError:
    raise unittest.SkipTest("opencv-python and numpy are needed for the viewer tests (pip install opencv-python numpy Pillow)")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import viewer as V  # noqa: E402

FIXTURE = os.path.join(HERE, "..", "fixtures", "online_log_approach.jsonl")
ROWS = [json.loads(l) for l in open(FIXTURE)]


def banner_pixels(png_or_img, h=100):
    """Fraction of the banner strip that is exactly the alarm red (BGR 0,0,220)."""
    import cv2
    img = cv2.imread(png_or_img) if isinstance(png_or_img, str) else png_or_img
    strip = img[:h]
    return float(np.mean(np.all(strip == np.array(V.ALARM_COLOR, np.uint8), axis=2)))


class TestAlarmFrameByFrame(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.path = os.path.join(self.d, "online_log.jsonl")
        shutil.copy(FIXTURE, self.path)

    def tearDown(self):
        shutil.rmtree(self.d)

    def test_banner_matches_warning_frame_by_frame_with_hold(self):
        al = V.Alarm(V.OnlineLogTail(self.path), hold_s=1.0)
        last_warn_t = None
        n_warn = n_held = 0
        for k, row in enumerate(ROWS):
            now = 100.0 + 0.1 * k                       # drawn at 10 Hz
            s = al.at(row["pair"], now)
            self.assertEqual(s["warning"], row["warning"], "pair %d" % row["pair"])
            self.assertEqual(s["log_pair"], row["pair"])
            if row["warning"]:
                last_warn_t = now
                n_warn += 1
                self.assertAlmostEqual(s["contact_s"], min(t["nominal_contact_s"] for t in row["tracks"] if t["warning"]))
            expect_hold = (not row["warning"]) and last_warn_t is not None and now - last_warn_t < 1.0
            self.assertEqual(s["held"], expect_hold, "pair %d" % row["pair"])
            self.assertEqual(s["on"], row["warning"] or expect_hold)
            n_held += expect_hold
        self.assertEqual(n_warn, 3)
        self.assertGreater(n_held, 0)

    def test_hold_is_one_second_of_display_time_and_no_longer(self):
        al = V.Alarm(V.OnlineLogTail(self.path), hold_s=1.0)
        al.at(26, 10.0)                                  # warning
        self.assertTrue(al.at(27, 10.2)["on"])           # row 27 does not warn: held
        self.assertTrue(al.at(27, 10.99)["held"])
        self.assertFalse(al.at(27, 11.01)["on"])         # more than 1.0 s after the last warning draw
        self.assertIsNone(al.at(27, 11.01)["contact_s"])

    def test_row_27_is_not_a_warning_and_row_28_is_the_second_start(self):
        self.assertFalse(ROWS[27]["warning"])
        self.assertTrue(ROWS[28]["warning"] and ROWS[28]["warning_started"])
        self.assertTrue(ROWS[26]["warning_started"])

    def test_before_the_first_warning_nothing(self):
        al = V.Alarm(V.OnlineLogTail(self.path))
        for k in range(26):
            s = al.at(k, 5.0 + k)
            self.assertFalse(s["on"])

    def test_log_lags_the_frame_by_one_pair(self):
        """The module processes pair k after the loop wrote it: when pair k is drawn the log reaches k-1. The state is the one of k-1."""
        al = V.Alarm(V.OnlineLogTail(self.path))
        tail = al.tail
        cut = [r for r in ROWS if r["pair"] <= 25]
        tail.entries.clear()
        tail.entries.extend(cut)
        tail.poll = lambda: 0                            # the file is not read in this test
        s = al.at(26, 1.0)                               # pair 26 is the first warning; the log ends at 25
        self.assertFalse(s["warning"])
        self.assertEqual(s["log_pair"], 25)
        self.assertEqual(s["lag_pairs"], 1)


class TestTail(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.path = os.path.join(self.d, "online_log.jsonl")

    def tearDown(self):
        shutil.rmtree(self.d)

    def test_missing_file_is_no_alarm(self):
        al = V.Alarm(V.OnlineLogTail(self.path))
        s = al.at(5, 1.0)
        self.assertFalse(s["on"])
        self.assertIsNone(s["log_pair"])

    def test_partial_line_waits_for_its_newline_and_skipped_lines_are_ignored(self):
        text = open(FIXTURE).read().splitlines()
        tail = V.OnlineLogTail(self.path)
        with open(self.path, "w") as f:
            f.write(text[0] + "\n" + text[1][:50])        # one complete line and half of the next
        tail.poll()
        self.assertEqual([e["pair"] for e in tail.entries], [ROWS[0]["pair"]])
        with open(self.path, "a") as f:
            f.write(text[1][50:] + "\n" + json.dumps({"pair": 99, "skipped": "non_increasing_capture_time"}) + "\n" + "not json\n" + text[2] + "\n")
        tail.poll()
        self.assertEqual([e["pair"] for e in tail.entries], [ROWS[0]["pair"], ROWS[1]["pair"], ROWS[2]["pair"]])

    def test_a_restarted_file_starts_over(self):
        text = open(FIXTURE).read().splitlines()
        tail = V.OnlineLogTail(self.path)
        with open(self.path, "w") as f:
            f.write("\n".join(text[:10]) + "\n")
        tail.poll()
        self.assertEqual(len(tail.entries), 10)
        with open(self.path, "w") as f:
            f.write(text[0] + "\n")
        tail.poll()
        self.assertEqual(len(tail.entries), 1)

    def test_lines_of_an_older_session_are_ignored(self):
        with open(self.path, "w") as f:
            f.write(open(FIXTURE).read())
        tail = V.OnlineLogTail(self.path, min_t_now=ROWS[-1]["t_now"] + 1)
        tail.poll()
        self.assertEqual(len(tail.entries), 0)


def stub_heal():
    """The two HEAL modules LiveViewer imports in its constructor."""
    q = types.ModuleType("qcar")
    p = types.ModuleType("qcar.patches")
    pr = types.ModuleType("qcar.patches.patch_real_extrinsic")
    pr.CAMERA_PARAMS = {"1": (np.eye(3), np.zeros(3)), "2": (np.eye(3), np.zeros(3))}
    v = types.ModuleType("qcar.visualize")
    v.draw_boxes = lambda ctx, b, color: False
    for name, m in (("qcar", q), ("qcar.patches", p), ("qcar.patches.patch_real_extrinsic", pr), ("qcar.visualize", v)):
        sys.modules[name] = m


class TestDrawing(unittest.TestCase):
    def setUp(self):
        stub_heal()
        self.d = tempfile.mkdtemp()
        self.log = os.path.join(self.d, "online_log.jsonl")
        shutil.copy(FIXTURE, self.log)
        self.snap = os.path.join(self.d, "snap")

    def tearDown(self):
        shutil.rmtree(self.d)

    def viewer(self, **kw):
        walls = [{"a": [-3.0, -2.5], "b": [3.0, -2.5]}]
        v = V.LiveViewer(["1", "2"], "1", {"1": "qcar-ego", "2": "qcar-peer"}, {"1": None, "2": None}, walls, [-3.5, 3.5, -3.0, 3.0], 10.0,
                             online_log=self.log, mode="coop", offscreen=True, composite=True, snapshot_dir=self.snap, snapshot_every=1, **kw)
        v.tail.min_t_now = None                           # the fixture is a recording of an old session
        return v

    def frames(self):
        pose = {"x": 0.0, "y": 0.0, "yaw": 0.0, "z": 0.0}
        return {a: {"front_undist": np.full((480, 640, 3), 90, np.uint8), "ego_pose": dict(pose, x=float(a))} for a in ("1", "2")}

    def test_banner_in_png_iff_alarm_and_trace_agrees_with_the_log(self):
        v = self.viewer()
        for row in ROWS:
            v.update(self.frames(), np.zeros((0, 8, 3)), np.zeros(0), {"pair": row["pair"], "mode": "coop", "latency_s": 0.1, "peer_status": "used"})
        trace = [json.loads(l) for l in open(os.path.join(self.snap, "alarm_trace.jsonl"))]
        self.assertEqual(len(trace), len(ROWS))
        on = 0
        for tr in trace:
            png = os.path.join(self.snap, tr["snapshot"])
            frac = banner_pixels(png)
            self.assertEqual(frac > 0.5, tr["banner_on"], tr)
            self.assertEqual(tr["warning"], ROWS[tr["pair"]]["warning"])
            on += tr["banner_on"]
        self.assertGreaterEqual(on, 3)
        # the three warning pairs have the banner; pair 10 does not
        self.assertTrue(next(t for t in trace if t["pair"] == 26)["banner_on"])
        self.assertFalse(next(t for t in trace if t["pair"] == 10)["banner_on"])

    def test_banner_text_uses_the_smallest_warning_contact_time(self):
        row = ROWS[26]
        c = min(t["nominal_contact_s"] for t in row["tracks"] if t["warning"])
        self.assertEqual(V.LiveViewer.banner_text({"contact_s": c}), "COLLISION WARNING - contact in %.2f s" % c)
        self.assertEqual(V.LiveViewer.banner_text({"contact_s": None}), "COLLISION WARNING - contact in --.-- s")

    def test_stale_draw_drops_the_banner_and_does_not_extend_the_hold(self):
        v = self.viewer()
        v.update(self.frames(), np.zeros((0, 8, 3)), np.zeros(0), {"pair": 26, "mode": "coop"})
        self.assertTrue(v.alarm.at(26, v.alarm.last_warn_wall + 0.1, update_hold=False)["on"])
        t = v.alarm.last_warn_wall
        v._draw(*v.last_args, stale=True)
        self.assertEqual(v.alarm.last_warn_wall, t)
        self.assertEqual(v.n_draws, 1)                     # a stale redraw is not a draw
        self.assertEqual(len([f for f in os.listdir(self.snap) if f.startswith("snap_")]), 1)
        self.assertTrue(os.path.exists(os.path.join(self.snap, "ready.png")))      # the READY screen is rendered by the same code

    def test_stale_composite_is_grey_not_red(self):
        v = self.viewer()
        v.update(self.frames(), np.zeros((0, 8, 3)), np.zeros(0), {"pair": 26, "mode": "coop"})
        al = v.alarm.at(26, 0.0, update_hold=False)
        povs = dict((a, np.full((480, 640, 3), 90, np.uint8)) for a in "12")
        img = v._compose(povs, np.full((600, 700, 3), 255, np.uint8), al, {"pair": 26}, True)
        self.assertEqual(banner_pixels(img), 0.0)
        self.assertTrue(np.all(img[5, 5] == np.array((60, 60, 60))))

    def test_three_window_panels_get_banner_and_border(self):
        v = self.viewer()
        v.composite = False
        img = np.full((480, 640, 3), 90, np.uint8)
        al = {"on": True, "contact_s": 0.59}
        out = v._panel_alarm(img.copy(), al, False)
        self.assertTrue(np.all(out[2, 2] == np.array(V.ALARM_COLOR)))          # border
        self.assertGreater(banner_pixels(out, 44), 0.5)
        self.assertTrue(np.all(v._panel_alarm(img.copy(), {"on": False, "contact_s": None}, False) == img))


class TestWalls(unittest.TestCase):
    def test_walls_file_has_the_two_frames(self):
        d = json.load(open(os.path.join(HERE, "..", "walls.json")))["walls"]
        self.assertEqual(len(d["new"]), 5)
        self.assertEqual(len(d["old"]), 5)
        self.assertNotEqual(d["new"][0]["a"], d["old"][0]["a"])
        self.assertAlmostEqual(d["new"][0]["a"][0], -2.081)


if __name__ == "__main__":
    unittest.main()
