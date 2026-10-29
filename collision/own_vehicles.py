"""Removal of the two instrumented cars from the detections.

The detector also sees the ego car's own body and the cooperating car. Their footprints are known from the motion-capture poses, so a
detection box whose bird's-eye-view IoU with a known footprint reaches a threshold is removed before tracking.
"""
import math

from shapely.geometry import Polygon

VEHICLE_LENGTH_M, VEHICLE_WIDTH_M = 0.425, 0.192      # footprint of the ego and the cooperating car
IOU_MIN = 0.10                                         # default removal threshold
EPS = 1e-9


def box_polygon(x, y, yaw, length, width):
    c, s = math.cos(yaw), math.sin(yaw)
    hl, hw = length / 2, width / 2
    return Polygon([(x + c * dx - s * dy, y + s * dx + c * dy) for dx, dy in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))])


def bev_iou(a, b):
    """Bird's-eye-view IoU of two (x, y, yaw, length, width) boxes."""
    pa, pb = box_polygon(*a), box_polygon(*b)
    inter = pa.intersection(pb).area
    return inter / (pa.area + pb.area - inter) if inter > 0 else 0.0


def vehicle_box(centre, yaw):
    """Footprint of a known vehicle: footprint centre and the raw motion-capture yaw."""
    return (centre[0], centre[1], yaw, VEHICLE_LENGTH_M, VEHICLE_WIDTH_M)


def vehicle_ious(box, vehicles):
    """BEV IoU of one detection box (dict) with each known vehicle's footprint: {name: iou}. `vehicles` = {name: (cx, cy, yaw)}."""
    b = (box["x"], box["y"], box["yaw"], box["length"], box["width"])
    return {name: bev_iou(b, vehicle_box((cx, cy), yaw)) for name, (cx, cy, yaw) in vehicles.items()}


def remove_known_vehicles(boxes, vehicles, iou_min=IOU_MIN):
    """Split `boxes` (dicts with x, y, yaw, length, width) into kept and removed: removed iff BEV IoU >= iou_min with any known footprint.
    Returns kept, removed, and for each removed box the set of vehicle names that matched."""
    kept, removed, who = [], [], []
    for b in boxes:
        hit = {a for a, v in vehicle_ious(b, vehicles).items() if v >= iou_min - EPS}
        if hit:
            removed.append(b)
            who.append(hit)
        else:
            kept.append(b)
    return kept, removed, who
