"""HEAL detections (ego frame, x10 model units, corner boxes) -> world SI boxes, matched to source frames.

world = R(ego_yaw) * (p / 10) + ego_xy / 10, with (ego_x, ego_y, ego_yaw) from the agent-1 yaml lidar_pose (x10, yaw in degrees); see
world_boxes.box_world. Each exported frame is matched to its source frame by the ego car's motion-capture pose (exact up to rounding: the
nearest pose row of the episode within 2 mm of the lidar pose, otherwise the frame is dropped).

Poses file (optionally gzip), one row per car and frame, fields separated by '|':
    session | car | episode | frame | x | y | yaw | valid | time | ...(further columns ignored)
`--ego-car` is the value of the `car` column that belongs to the ego car. A scenario name `qcar_<date>_<episode>` of the export maps to the episode
name `<episode>` (an episode folder that uses spaces instead of underscores is accepted).

Output, one row per matched frame:
    {"episode", "session", "split", "heal_index", "frame", "time", "ego": {"x", "y", "yaw"}, "pose_match_m",
     "coop": [box...] | null, "ego_only": [box...] | null, "gt": [box...]}     box = {"x", "y", "yaw", "length", "width"[, "confidence"]} in world metres
`image_time_vicon_s` (the frame's image capture time on the pose clock) has to be added before the rows go to collision/frames.py.

    python3 -m perception.heal_export.to_world <export.jsonl[.gz]> <poses.txt.gz> <out.jsonl> --ego-car <id>
"""
import argparse
import collections
import gzip
import json
import math

from .world_boxes import box_world


def source_name(scenario):
    s = scenario.split("_", 2)[2] if scenario.startswith("qcar_2026") else scenario
    return s.replace("Node22_1To9", "Node22.1To9")


def _open(p):
    return gzip.open(p, "rt") if p.endswith(".gz") else open(p)


def load_poses(path):
    poses = collections.defaultdict(list)   # (episode, car) -> rows
    with _open(path) as f:
        for l in f:
            d, car, ep, fr, ox, oy, oyaw, ov, t, *_ = l.strip().split("|")
            poses[(ep, car)].append({"session": d, "frame": fr, "x": float(ox), "y": float(oy), "yaw": float(oyaw), "time": float(t)})
    return poses


def convert(r, poses, ego_car, max_match_m=0.002):
    """One export record -> world row, or None when no pose of the episode is within `max_match_m` of the record's lidar pose."""
    ep = source_name(r["scenario"])
    if (ep, ego_car) not in poses and (ep.replace("_", " "), ego_car) in poses:  # an episode folder may use spaces
        ep = ep.replace("_", " ")
    cand = poses.get((ep, ego_car), [])
    px, py = r["lidar_pose"][0] / 10, r["lidar_pose"][1] / 10
    best = min(cand, key=lambda q: math.hypot(q["x"] - px, q["y"] - py), default=None)
    err = math.hypot(best["x"] - px, best["y"] - py) if best else None
    if err is None or err > max_match_m:
        return None
    rec = {"episode": ep, "session": best["session"], "split": r["split"], "heal_index": r["index"], "frame": best["frame"], "time": best["time"],
           "ego": {"x": best["x"], "y": best["y"], "yaw": best["yaw"]}, "pose_match_m": round(err, 5)}
    for m in ("coop", "ego_only"):
        rec[m] = None if r[m] is None else [dict(box_world(b, r["lidar_pose"]), confidence=s) for b, s in zip(r[m]["boxes"], r[m]["scores"])]
    rec["gt"] = [box_world(b, r["lidar_pose"]) for b in (r["gt"] or [])]
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export")
    ap.add_argument("poses")
    ap.add_argument("out")
    ap.add_argument("--ego-car", required=True, help="value of the `car` column of the poses file that belongs to the ego car")
    a = ap.parse_args(argv)
    poses = load_poses(a.poses)
    n_ok = n_bad = 0
    with open(a.out, "w") as out, _open(a.export) as f:
        for l in f:
            rec = convert(json.loads(l), poses, a.ego_car)
            if rec is None:
                n_bad += 1
                continue
            n_ok += 1
            out.write(json.dumps(rec) + "\n")
    print("matched", n_ok, "unmatched", n_bad)


if __name__ == "__main__":
    main()
