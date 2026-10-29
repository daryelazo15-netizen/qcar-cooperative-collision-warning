"""Tests of the HEAL-box -> world-box conversion and the export converter.   python3 -m unittest discover -s perception/tests"""
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO)
from perception.heal_export.world_boxes import box_world  # noqa: E402


def corners(cx, cy, l, w):
    b = [[cx + dx, cy + dy, 0.0] for dx, dy in ((l / 2, w / 2), (l / 2, -w / 2), (-l / 2, -w / 2), (-l / 2, w / 2))]
    return b + [[p[0], p[1], 1.0] for p in b]


class T(unittest.TestCase):
    def test_identity_pose_scales_by_ten(self):
        b = box_world(corners(20.0, 5.0, 4.25, 1.92), [0, 0, 0, 0, 0, 0])
        self.assertAlmostEqual(b["x"], 2.0)
        self.assertAlmostEqual(b["y"], 0.5)
        self.assertAlmostEqual(b["length"], 0.425)
        self.assertAlmostEqual(b["width"], 0.192)
        self.assertAlmostEqual(math.cos(b["yaw"]) ** 2, 1.0)           # the yaw lies along the long edge (modulo pi)

    def test_pose_translation_and_rotation(self):
        b = box_world(corners(10.0, 0.0, 4.25, 1.92), [30.0, -10.0, 0, 0, 90.0, 0])      # ego at (3, -1) m, heading 90 degrees
        self.assertAlmostEqual(b["x"], 3.0)
        self.assertAlmostEqual(b["y"], -1.0 + 1.0)
        self.assertAlmostEqual(math.sin(b["yaw"]) ** 2, 1.0)

    def test_long_edge_sets_length_and_width_whatever_the_corner_order(self):
        wide = box_world(corners(0.0, 0.0, 1.92, 4.25), [0, 0, 0, 0, 0, 0])
        self.assertAlmostEqual(wide["length"], 0.425)
        self.assertAlmostEqual(wide["width"], 0.192)
        self.assertAlmostEqual(math.sin(wide["yaw"]) ** 2, 1.0)

    def test_cli_matches_export_frames_to_poses(self):
        rec = {"split": "s", "scenario": "qcar_20261013_Ep1", "index": 3, "lidar_pose": [0, 0, 0, 0, 0, 0],
               "coop": {"boxes": [corners(20.0, 0.0, 4.25, 1.92)], "scores": [0.97]}, "ego_only": None, "gt": [corners(0.0, 0.0, 4.25, 1.92)]}
        far = dict(rec, index=4, lidar_pose=[50.0, 0, 0, 0, 0, 0])          # 5 mm from the nearest pose: dropped
        poses = ["2026-10-13|73|Ep1|000007|0.0|0.0|0.1|1|100.5|9|9|9|1", "2026-10-13|73|Ep1|000008|0.5|0.0|0.1|1|100.6|9|9|9|1", "2026-10-13|89|Ep1|000007|5.0|5.0|0|1|100.5|9|9|9|1"]
        with tempfile.TemporaryDirectory() as d:
            src, pos, out = (os.path.join(d, n) for n in ("e.jsonl", "p.txt", "w.jsonl"))
            with open(src, "w") as f:
                f.write(json.dumps(rec) + "\n" + json.dumps(far) + "\n")
            with open(pos, "w") as f:
                f.write("\n".join(poses) + "\n")
            r = subprocess.run([sys.executable, "-B", "-m", "perception.heal_export.to_world", src, pos, out, "--ego-car", "73"], cwd=REPO, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("matched 1 unmatched 1", r.stdout)
            with open(out) as f:
                rows = [json.loads(l) for l in f]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual((row["episode"], row["session"], row["frame"], row["time"], row["heal_index"]), ("Ep1", "2026-10-13", "000007", 100.5, 3))
        self.assertEqual(row["ego"], {"x": 0.0, "y": 0.0, "yaw": 0.1})
        self.assertEqual((row["ego_only"], row["coop"][0]["confidence"]), (None, 0.97))
        self.assertAlmostEqual(row["coop"][0]["x"], 2.0)
        self.assertAlmostEqual(row["gt"][0]["length"], 0.425)


if __name__ == "__main__":
    unittest.main()
