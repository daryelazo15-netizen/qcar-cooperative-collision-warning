"""Constant-velocity Kalman tracker.

`KalmanTracker` subclasses `ReferenceTracker` and calls its `step()` unchanged, so association (gate, prediction from the two-point velocity,
the max-speed bound), confirmation (>= 2 observations), expiry and track ids are the reference ones by construction. Only the `TrackEstimate`
that `step()` returns is rebuilt:

  per-axis constant-velocity Kalman filter, state [p, v]. Measurement = the associated detection centre, R = sigma_p^2 (setting
  `position_noise_std_m`). White-noise-acceleration process noise q = sigma_a^2, Q = q [[dt^4/4, dt^3/2], [dt^3/2, dt^2]].
  Initialised at the 2nd observation from the two-point difference: p = z2, v = (z2-z1)/dt, P = [[s2, s2/dt], [s2/dt, 2 s2/dt^2]]
  (s2 = sigma_p^2), i.e. at that frame the filter emits exactly what the reference tracker emits. Emitted: filtered position (x, y), filtered
  velocity, position_variance = (P00_x, P00_y), velocity_variance = (P11_x, P11_y). The position-velocity covariance P01 is not passed
  (TrackEstimate has no field for it; the reference tracker does not model it either).
  sigma_a here is the filter's process noise only; `risk.acceleration_noise_std` (used by the collision module's propagation) is untouched.

The associated detections of a track are read from `self.records` after the reference step (rows with time == now), so the history contains
exactly the detections the reference association attached.
"""
import dataclasses
import functools
import math

from collision.geometry import State
from .reference import ReferenceTracker


def _kf_init(z1, z2, dt, s2):
    """Per-axis initial [p, v] and covariance from two observations (see module docstring)."""
    return [z2, (z2 - z1) / dt], [[s2, s2 / dt], [s2 / dt, 2 * s2 / (dt * dt)]]


def _kf_step(x, P, z, dt, s2, q):
    """One predict + update of the per-axis filter. Returns new (x, P)."""
    p = x[0] + dt * x[1]
    v = x[1]
    p00 = P[0][0] + 2 * dt * P[0][1] + dt * dt * P[1][1] + q * dt ** 4 / 4
    p01 = P[0][1] + dt * P[1][1] + q * dt ** 3 / 2
    p11 = P[1][1] + q * dt * dt
    s = p00 + s2
    k0, k1 = p00 / s, p01 / s
    y = z - p
    nx = [p + k0 * y, v + k1 * y]
    nP = [[(1 - k0) * p00, (1 - k0) * p01], [(1 - k0) * p01, p11 - k1 * p01]]
    return nx, nP


class KalmanTracker(ReferenceTracker):
    def __init__(self, settings, sigma_a=0.1):
        if not (math.isfinite(sigma_a) and sigma_a >= 0):
            raise ValueError("KalmanTracker needs a finite nonnegative sigma_a")
        super().__init__(settings)
        self.sigma_a = sigma_a
        self.hist = {}      # track key -> [(t, x, y)] associated detections
        self.kf = {}        # track key -> (x-axis [p, v], P, y-axis [p, v], P, time)

    def step(self, detections, now_s, provenance):
        updated = super().step(detections, now_s, provenance)
        for key, row in self.records.items():
            if row['time'] == now_s:
                d = row['detection']
                self.hist.setdefault(key, []).append((now_s, d.x, d.y))
        for key in [k for k in self.hist if k not in self.records]:
            del self.hist[key]
            self.kf.pop(key, None)
        for key, h in self.hist.items():
            if len(h) > 12:
                del h[:-12]
        out = []
        s2 = self.settings['position_noise_std_m'] ** 2
        for est, shape in updated:
            key = est.track_id
            h = self.hist[key]
            out.append((self._kalman(est, key, h, s2), shape))
        return out

    def _kalman(self, est, key, h, s2):
        q = self.sigma_a ** 2
        (t2, x2, y2) = h[-1]
        if key not in self.kf:
            (t1, x1, y1) = h[-2]
            dt = t2 - t1
            xs, Px = _kf_init(x1, x2, dt, s2)
            ys, Py = _kf_init(y1, y2, dt, s2)
        else:
            xs, Px, ys, Py, t_prev = self.kf[key]
            dt = t2 - t_prev
            xs, Px = _kf_step(xs, Px, x2, dt, s2, q)
            ys, Py = _kf_step(ys, Py, y2, dt, s2, q)
        self.kf[key] = (xs, Px, ys, Py, t2)
        state = State(xs[0], ys[0], est.state.yaw, xs[1], ys[1])
        return dataclasses.replace(est, state=state, position_variance=(Px[0][0], Py[0][0]),
                                   velocity_variance=(Px[1][1], Py[1][1]))


def kalman_factory(sigma_a=0.1):
    """Tracker factory for `collision.replay.replay(..., tracker_factory=...)`."""
    return functools.partial(KalmanTracker, sigma_a=sigma_a)
