"""HEAL corner boxes (ego frame, x10 model units) -> world SI boxes."""
import math


def box_world(corners, pose):
    """world = R(ego_yaw) * (p / 10) + ego_xy / 10, with pose = [x, y, z, roll, yaw_deg, pitch] (x10 model units, yaw in degrees), i.e. the
    agent-1 yaml lidar_pose. Box = centroid of the 4 bottom corners, yaw along the long edge, length = long edge, width = short edge (metres)."""
    ex, ey, eyaw = pose[0] / 10, pose[1] / 10, math.radians(pose[4])
    c, s = math.cos(eyaw), math.sin(eyaw)
    pts = [(c * x / 10 - s * y / 10 + ex, s * x / 10 + c * y / 10 + ey) for x, y, *_ in corners[:4]]
    cx, cy = sum(p[0] for p in pts) / 4, sum(p[1] for p in pts) / 4
    e1, e2 = math.dist(pts[0], pts[1]), math.dist(pts[1], pts[2])
    a, b = (pts[1], pts[2]) if e2 >= e1 else (pts[0], pts[1])
    return {"x": cx, "y": cy, "yaw": math.atan2(b[1] - a[1], b[0] - a[0]), "length": max(e1, e2), "width": min(e1, e2)}
