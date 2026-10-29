"""Causal constant-velocity collision risk with bounded stale-track fallback.

This module is intentionally independent of HEAL and dataset storage. Callers
must provide states, shapes and timestamps in one consistent coordinate/unit
system. The conservative overlap is an uncertainty envelope, not a calibrated
collision probability.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from .geometry import Footprint, State, axes, swept_contact


def _finite(values):
    return all(math.isfinite(value) for value in values)


@dataclass(frozen=True)
class TrackEstimate:
    track_id: str
    state: State
    source_time_s: float
    position_variance: tuple
    velocity_variance: tuple
    confidence: float
    provenance: str

    def __post_init__(self):
        if not self.track_id or not self.provenance:
            raise ValueError("Track identity and provenance are required")
        if not _finite((self.source_time_s, self.confidence, *self.position_variance,
                        *self.velocity_variance, *vars(self.state).values())):
            raise ValueError("Track fields must be finite")
        if len(self.position_variance) != 2 or len(self.velocity_variance) != 2:
            raise ValueError("Uncertainty must contain world x/y diagonal variances")
        if any(value < 0 for value in self.position_variance + self.velocity_variance):
            raise ValueError("Variances must be nonnegative")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be in [0, 1]")


@dataclass(frozen=True)
class FallbackConfig:
    live_update_timeout_s: float
    expire_source_age_s: float
    forecast_horizon_s: float
    warning_horizon_s: float
    acceleration_noise_std: float
    sigma_multiplier: float
    forecast_interval_s: float = 0.1

    def __post_init__(self):
        values = (self.live_update_timeout_s, self.expire_source_age_s,
                  self.forecast_horizon_s, self.warning_horizon_s,
                  self.acceleration_noise_std, self.sigma_multiplier,
                  self.forecast_interval_s)
        if not _finite(values) or any(value < 0 for value in values):
            raise ValueError("Fallback parameters must be finite and nonnegative")
        if self.live_update_timeout_s > self.expire_source_age_s:
            raise ValueError("Live timeout cannot exceed source-age expiry")
        if self.warning_horizon_s > self.forecast_horizon_s:
            raise ValueError("Warning horizon cannot exceed forecast horizon")
        if self.sigma_multiplier == 0:
            raise ValueError("sigma_multiplier must be positive")
        if self.forecast_interval_s <= 0:
            raise ValueError("forecast_interval_s must be positive")


@dataclass(frozen=True)
class RiskResult:
    mode: str
    warning: bool
    source_age_s: float | None
    delivery_gap_s: float | None
    nominal_contact_s: float | None
    conservative_contact_s: float | None
    urgency_score: float
    target_state: State | None
    target_position_std: tuple | None
    track_id: str | None
    provenance: str | None


def propagate_cv(track, now_s, acceleration_noise_std):
    """Propagate a track without future observations and grow diagonal variance."""
    if not math.isfinite(now_s) or now_s < track.source_time_s:
        raise ValueError("Evaluation time cannot precede the source observation")
    if not math.isfinite(acceleration_noise_std) or acceleration_noise_std < 0:
        raise ValueError("Acceleration noise must be finite and nonnegative")
    dt = now_s-track.source_time_s
    state = State(track.state.x+track.state.vx*dt,
                  track.state.y+track.state.vy*dt,
                  track.state.yaw, track.state.vx, track.state.vy)
    q = acceleration_noise_std**2
    position_variance = tuple(track.position_variance[i]
                              + dt*dt*track.velocity_variance[i]
                              + 0.25*dt**4*q for i in range(2))
    velocity_variance = tuple(value+dt*dt*q for value in track.velocity_variance)
    return TrackEstimate(track.track_id, state, now_s, position_variance,
                         velocity_variance, track.confidence, track.provenance)


def inflate_footprint(shape, yaw, world_variance, sigma_multiplier):
    """Project diagonal world-position variance onto target body axes."""
    forward, left = axes(yaw)
    def projected_std(axis):
        return math.sqrt(axis[0]**2*world_variance[0]
                         + axis[1]**2*world_variance[1])
    return Footprint(shape.length+2*sigma_multiplier*projected_std(forward),
                     shape.width+2*sigma_multiplier*projected_std(left))


def growing_envelope_contact(ego, track, now_s, ego_shape, target_shape, config):
    """Conservative earliest possible-contact bound for fixed-yaw CV motion.

    Within each horizon interval, inflate by uncertainty at its END and use
    continuous SAT over that whole interval. Monotone variance makes this an
    outer bound; it may warn early. This is neither an exact time of uncertain
    contact nor a probability. Smaller intervals tighten the bound.
    """
    start = 0.0
    while True:
        end = min(start+config.forecast_interval_s, config.forecast_horizon_s)
        at_start = propagate_cv(track, now_s+start, config.acceleration_noise_std)
        at_end = propagate_cv(track, now_s+end, config.acceleration_noise_std)
        inflated = inflate_footprint(target_shape, track.state.yaw,
                                     at_end.position_variance,
                                     config.sigma_multiplier)
        ego_start = State(ego.x+ego.vx*start, ego.y+ego.vy*start,
                          ego.yaw, ego.vx, ego.vy)
        interval = swept_contact(ego_start, at_start.state, ego_shape,
                                 inflated, end-start)
        if interval is not None:
            return start+interval[0]
        if end >= config.forecast_horizon_s:
            return None
        start = end


class FallbackCollisionModel:
    """Hold the last delivered cooperative track and evaluate it causally."""

    def __init__(self, config):
        self.config = config
        self._track = None
        self._received_time_s = None

    def update(self, track, received_time_s):
        if not math.isfinite(received_time_s) or received_time_s < track.source_time_s:
            raise ValueError("Receive time cannot precede the source observation")
        if self._received_time_s is not None and received_time_s < self._received_time_s:
            raise ValueError("Receive times must be nondecreasing")
        if self._track is not None and track.track_id == self._track.track_id:
            if track.source_time_s <= self._track.source_time_s:
                raise ValueError("Track updates must advance source time")
        self._track, self._received_time_s = track, received_time_s

    def clear(self):
        self._track = None
        self._received_time_s = None

    def evaluate(self, ego, now_s, ego_shape, target_shape, strategy="uncertainty"):
        if strategy not in ("uncertainty", "cv", "hold"):
            raise ValueError("Unknown propagation strategy")
        if not math.isfinite(now_s) or not _finite(vars(ego).values()):
            raise ValueError("Evaluation state/time must be finite")
        if self._track is None:
            return RiskResult("unavailable", False, None, None, None, None,
                              0.0, None, None, None, None)
        if now_s < self._received_time_s:
            raise ValueError("Evaluation time cannot precede track reception")
        source_age = now_s-self._track.source_time_s
        delivery_gap = now_s-self._received_time_s
        if source_age > self.config.expire_source_age_s:
            return RiskResult("expired", False, source_age, delivery_gap, None,
                              None, 0.0, None, None, self._track.track_id,
                              self._track.provenance)

        target = propagate_cv(self._track, now_s,
                              self.config.acceleration_noise_std)
        if strategy == "hold":
            target = TrackEstimate(target.track_id, self._track.state, now_s,
                                   target.position_variance, target.velocity_variance,
                                   target.confidence, target.provenance)
        nominal = swept_contact(ego, target.state, ego_shape, target_shape,
                                self.config.forecast_horizon_s)
        contact_s = (growing_envelope_contact(ego, self._track, now_s,
                                             ego_shape, target_shape, self.config)
                     if strategy == "uncertainty"
                     else (None if nominal is None else nominal[0]))
        warning = contact_s is not None and contact_s <= self.config.warning_horizon_s
        urgency = 0.0 if contact_s is None else max(
            0.0, 1-contact_s/max(self.config.forecast_horizon_s, 1e-12))
        mode = ("live" if delivery_gap <= self.config.live_update_timeout_s
                else "fallback")
        std = tuple(math.sqrt(value) for value in target.position_variance)
        return RiskResult(mode, warning, source_age, delivery_gap,
                          None if nominal is None else nominal[0], contact_s,
                          urgency, target.state, std, target.track_id,
                          target.provenance)
