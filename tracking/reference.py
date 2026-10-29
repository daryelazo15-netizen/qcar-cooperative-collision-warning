"""Reference tracker: gated nearest-neighbour association and a two-point (backward difference) velocity estimate."""
import math

from collision.geometry import Footprint, State
from collision.risk import TrackEstimate


class ReferenceTracker:
    """Minimal gated association and backward velocity estimates, no ground-truth IDs.

    Confirmations require >=2 observations. No forward/backward smoothing; shape/yaw come from the latest observation.
    Covariance below assumes independent isotropic position noise.

    `settings` keys: association_gate_m, max_speed_mps, position_noise_std_m, expire_source_age_s.
    `step(detections, now_s, provenance)` takes the frame's detections (objects with x, y, yaw, length, width, confidence) and returns
    `[(TrackEstimate, Footprint)]` for the tracks updated in this frame that have at least two observations.
    """
    def __init__(self, settings):
        self.settings = settings
        self.records = {}
        self.next_id = 0
        self.time = None

    def step(self, detections, now_s, provenance):
        if not math.isfinite(now_s) or (self.time is not None and now_s <= self.time):
            raise ValueError("Replay timestamps must be strictly increasing")
        self.time = now_s
        self.records = {key: row for key, row in self.records.items()
                        if now_s-row['time'] <= self.settings['expire_source_age_s']}
        candidates = []
        for key, row in self.records.items():
            dt = now_s-row['time']
            prediction = (row['detection'].x+row['velocity'][0]*dt,
                          row['detection'].y+row['velocity'][1]*dt)
            gate = self.settings['association_gate_m']
            if row['count'] == 1:
                gate += self.settings['max_speed_mps']*dt
            for index, detection in enumerate(detections):
                distance = math.hypot(detection.x-prediction[0], detection.y-prediction[1])
                observed_speed = math.hypot(detection.x-row['detection'].x,
                                            detection.y-row['detection'].y)/dt
                if distance <= gate and observed_speed <= self.settings['max_speed_mps']:
                    candidates.append((distance, key, index))
        matched, used = {}, set()
        for _, key, index in sorted(candidates):
            if key not in matched and index not in used:
                matched[key] = index
                used.add(index)
        updated = []
        for index, detection in enumerate(detections):
            key = next((key for key, value in matched.items() if value == index), None)
            variance = self.settings['position_noise_std_m']**2
            if key is None:
                key = 'track-%d' % self.next_id
                self.next_id += 1
                row = {'count': 1, 'velocity': (0.0, 0.0), 'velocity_variance': (0.0, 0.0)}
            else:
                previous = self.records[key]
                dt = now_s-previous['time']
                velocity = ((detection.x-previous['detection'].x)/dt,
                            (detection.y-previous['detection'].y)/dt)
                if math.hypot(*velocity) > self.settings['max_speed_mps']:
                    raise ValueError("Associated velocity exceeds configured physical bound")
                row = {'count': previous['count']+1, 'velocity': velocity,
                       'velocity_variance': (2*variance/(dt*dt),)*2}
            row.update(time=now_s, detection=detection)
            self.records[key] = row
            if row['count'] >= 2:
                estimate = TrackEstimate(key, State(detection.x, detection.y,
                                                    detection.yaw, *row['velocity']),
                                         now_s, (variance, variance),
                                         row['velocity_variance'], detection.confidence,
                                         provenance)
                updated.append((estimate, Footprint(detection.length, detection.width)))
        return updated
