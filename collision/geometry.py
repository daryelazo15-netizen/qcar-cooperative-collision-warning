"""Right-handed SI footprint geometry and constant-orientation swept SAT."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class State:
    x: float
    y: float
    yaw: float
    vx: float
    vy: float


@dataclass(frozen=True)
class Footprint:
    length: float
    width: float

    def __post_init__(self):
        if not all(math.isfinite(v) and v > 0 for v in (self.length, self.width)):
            raise ValueError("Footprint dimensions must be finite positive SI values")


def axes(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return ((c, s), (-s, c))


def radius(shape, yaw, axis):
    forward, left = axes(yaw)
    return (shape.length / 2 * abs(sum(a * b for a, b in zip(forward, axis)))
            + shape.width / 2 * abs(sum(a * b for a, b in zip(left, axis))))


def swept_contact(ego, target, ego_shape, target_shape, horizon):
    """Closed interval of overlap for fixed yaws/constant velocities, or None.

    Continuous SAT avoids missing brief contacts between discretized predictions.
    This is a geometric extrapolation, not a simulator physics contact event.
    """
    if not math.isfinite(horizon) or horizon < 0:
        raise ValueError("horizon must be finite and nonnegative")
    if not all(math.isfinite(v) for state in (ego, target) for v in vars(state).values()):
        raise ValueError("Nonfinite state")
    enter, leave = 0.0, horizon
    delta = (target.x - ego.x, target.y - ego.y)
    velocity = (target.vx - ego.vx, target.vy - ego.vy)
    for axis in axes(ego.yaw) + axes(target.yaw):
        separation = sum(a * b for a, b in zip(delta, axis))
        rate = sum(a * b for a, b in zip(velocity, axis))
        extent = radius(ego_shape, ego.yaw, axis) + radius(target_shape, target.yaw, axis)
        if abs(rate) < 1e-12:
            if abs(separation) > extent + 1e-12:
                return None
            continue
        lo, hi = sorted(((-extent - separation) / rate, (extent - separation) / rate))
        enter, leave = max(enter, lo), min(leave, hi)
        if enter > leave + 1e-12:
            return None
    return max(0.0, enter), max(0.0, leave)

