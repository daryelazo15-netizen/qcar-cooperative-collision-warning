"""Detection rows in world coordinates -> replay frames (see collision.replay for the frame format).

One row per detector frame (the output of perception/heal_export/to_world.py, with the frame time added as `image_time_vicon_s`):

    {"frame": str, "time": pose sample time, "image_time_vicon_s": image capture time on the pose clock,
     "ego": {"x", "y", "yaw"} (motion-capture marker pose of the ego car),
     "coop": [box, ...], "ego_only": [box, ...]}   box = {"x", "y", "yaw", "length", "width", "confidence", ...} in world metres

Frame time is `image_time_vicon_s`; a row without a finite value is refused. Ego velocity is the backward difference of consecutive ego marker
positions divided by the step of the pose-sample times `time` (a repeated sample gives 0); the first frame has `ego_motion_valid` False.
"""
import math

from .own_vehicles import remove_known_vehicles, VEHICLE_LENGTH_M, VEHICLE_WIDTH_M

# marker -> footprint-centre offsets (forward, left) in the car's body frame
OFFSETS = {"ego": (0.03474432120351184, -0.022417966916566143), "peer": (0.04509606009254556, -0.01912982458045971)}
EGO_FOOTPRINT = {"length": VEHICLE_LENGTH_M, "width": VEHICLE_WIDTH_M}
IMAGE_TIME_KEY = "image_time_vicon_s"
BOX_KEYS = ("x", "y", "yaw", "length", "width", "confidence")


def centre(x, y, yaw, off):
    c, s = math.cos(yaw), math.sin(yaw)
    return x + c * off[0] - s * off[1], y + s * off[0] + c * off[1]


def box_fields(b):
    return {k: b[k] for k in BOX_KEYS}


def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def build_frames(rows, peer_poses=None, iou_min=0.10, yaw_offsets=None, removed=None):
    """Frames of one episode. `peer_poses` = [(time, marker x, marker y, raw yaw)] of the cooperating car (optional; the sample nearest in
    `time` to each row is used to remove the cooperating car's own detections). `yaw_offsets` = {session: degrees} is added to the ego yaw of the replay frames
    of the rows whose `session` key matches (the known-vehicle footprints keep the raw yaw). `removed`: optional collections.Counter, increased by one
    for every (stream, vehicle name) pair of every removed box."""
    for r in rows:
        if not _is_number(r.get(IMAGE_TIME_KEY)):
            raise ValueError("world row (frame %s) lacks a finite %s" % (r.get("frame"), IMAGE_TIME_KEY))
    rows = sorted(rows, key=lambda r: r[IMAGE_TIME_KEY])
    out, prev = [], None
    for r in rows:
        t = r[IMAGE_TIME_KEY]
        if prev is not None and t <= prev[0]:
            continue
        e = r["ego"]
        ego_c = centre(e["x"], e["y"], e["yaw"], OFFSETS["ego"])
        ego_yaw = e["yaw"] + math.radians((yaw_offsets or {}).get(r.get("session"), 0.0))
        vehicles = {"ego": (ego_c[0], ego_c[1], e["yaw"])}          # known-vehicle footprints use the RAW yaw
        if peer_poses:
            q = min(peer_poses, key=lambda q: abs(q[0] - r["time"]))
            vehicles["coop"] = centre(q[1], q[2], q[3], OFFSETS["peer"]) + (q[3],)
        vx, vy = (0.0, 0.0)
        if prev:
            dtp = r["time"] - prev[3]
            if dtp > 0:
                vx, vy = (e["x"] - prev[1]) / dtp, (e["y"] - prev[2]) / dtp
            elif not (dtp == 0 and e["x"] == prev[1] and e["y"] == prev[2]):
                raise ValueError("ego pose time of frame %s does not advance (time %r after %r) although the pose changes or goes back" % (r["frame"], r["time"], prev[3]))
        frame = {"frame_id": r["frame"], "time_s": t, "ego_motion_valid": prev is not None,
                 "ego": {"x": ego_c[0], "y": ego_c[1], "yaw": ego_yaw, "vx": vx, "vy": vy}}
        for stream, key in (("cooperative", "coop"), ("ego_only", "ego_only")):
            boxes = [box_fields(b) for b in (r[key] or [])]
            kept, _gone, who = remove_known_vehicles(boxes, vehicles, iou_min)
            if removed is not None:
                for names in who:
                    for n in names:
                        removed[(stream, n)] += 1
            frame[stream] = kept
        out.append(frame)
        prev = (t, e["x"], e["y"], r["time"])
    return out


def build_episode(episode_id, rows, peer_poses=None, iou_min=0.10, yaw_offsets=None, removed=None):
    return {"episode_id": episode_id, "coordinate_frame": "world_si", "time_basis": "source_time",
            "ego_footprint": EGO_FOOTPRINT, "frames": build_frames(rows, peer_poses, iou_min, yaw_offsets, removed)}
