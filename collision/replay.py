"""Causal frame-by-frame replay of detection streams with message loss.

An episode is one dictionary (one line of an episode file):

    {"episode_id": str, "coordinate_frame": "world_si", "time_basis": "source_time",
     "ego_footprint": {"length": m, "width": m},
     "frames": [{"frame_id": str, "time_s": float, "ego_motion_valid": bool,
                 "ego": {"x", "y", "yaw", "vx", "vy"},
                 "cooperative": [box, ...], "ego_only": [box, ...]}, ...]}

with box = {"x", "y", "yaw", "length", "width", "confidence"} in world metres. Extra keys (for example an "evaluation" block) are ignored.
`cooperative` holds the boxes of the fused two-car detector, `ego_only` the boxes of the single-car detector. A boolean mask says, per frame,
whether the cooperative stream was delivered. Nothing after the current frame is read.

Methods (METHODS): `ego_only` and `cooperative_ideal` ignore the mask (own stream / cooperative stream on every frame); `no_fallback`, `hold`, `cv`,
`uncertainty` and `uncertainty_nofb` use the cooperative stream on delivered frames and the ego-only stream on lost ones. They differ in how a
track that did not get an update in the frame is treated: `no_fallback` skips it, `hold` keeps its last state, `cv` propagates it at constant
velocity, `uncertainty` propagates it and inflates its footprint with the growing position uncertainty, `uncertainty_nofb` applies the
`uncertainty` envelope to updated tracks only.

    python -m collision.replay --episodes episodes.jsonl --out rows.jsonl [--config collision/configs/default.json]
                               [--tracker reference|kalman] [--sigma-a 0.1] [--methods uncertainty,cv]
                               [--loss ideal | bernoulli --drop-probability 0.25 --seed 3 | burst --burst START END ...]
"""
import argparse
import hashlib
import json
import math
import os
import random
from dataclasses import asdict

from .detection import Detection
from .geometry import Footprint, State
from .risk import FallbackCollisionModel, FallbackConfig
from tracking.reference import ReferenceTracker

METHODS = ("ego_only", "cooperative_ideal", "no_fallback", "hold", "cv", "uncertainty", "uncertainty_nofb")
STREAMS = ("cooperative", "ego_only")                 # replay stream names
STREAM_KEYS = {"cooperative": "coop", "ego_only": "ego_only"}   # names in the config
DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "default.json")


def per_stream(value, name):
    """A config value is a single number (applies to both streams) or {"coop": v, "ego_only": v}.
    Returns {replay stream name: value}."""
    if isinstance(value, dict):
        if set(value) != set(STREAM_KEYS.values()):
            raise ValueError("%s needs exactly the keys %s, got %s" % (name, sorted(STREAM_KEYS.values()), sorted(value)))
        return {s: value[STREAM_KEYS[s]] for s in STREAMS}
    return {s: value for s in STREAMS}


def stream_risk_configs(settings):
    """FallbackConfig of each stream: the stream's own warning_horizon_s, everything else shared."""
    horizons = per_stream(settings['risk']['warning_horizon_s'], 'risk.warning_horizon_s')
    return {s: FallbackConfig(**dict(settings['risk'], warning_horizon_s=horizons[s])) for s in STREAMS}


def validate_settings(settings):
    stream_risk_configs(settings)
    tracker = settings['tracker']
    for key in ('association_gate_m', 'max_speed_mps', 'position_noise_std_m'):
        if not math.isfinite(tracker[key]) or tracker[key] <= 0:
            raise ValueError("Tracker parameters must be finite positive")
    for value in per_stream(settings['min_detection_confidence'], 'min_detection_confidence').values():
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid detection threshold")


def delivery_mask(frames, pattern, seed=0, drop_probability=0.0, bursts=()):
    """Outcome-independent masks; intervals are half-open source-time outages."""
    if pattern not in ('ideal', 'bernoulli', 'burst'):
        raise ValueError("Unknown loss pattern")
    if not math.isfinite(drop_probability) or not 0 <= drop_probability <= 1:
        raise ValueError("Invalid drop probability")
    if any(not all(math.isfinite(v) for v in pair) or pair[0] >= pair[1] for pair in bursts):
        raise ValueError("Invalid outage interval")
    rng = random.Random(seed)
    return [True if pattern == 'ideal' else
            (rng.random() >= drop_probability if pattern == 'bernoulli' else
             not any(start <= frame['time_s'] < end for start, end in bursts))
            for frame in frames]


def replay(episode, settings, mask, method, tracker_factory=None):
    """Rows of one episode (one per frame) for one method. `tracker_factory(tracker_settings)` builds the tracker (default: ReferenceTracker);
    it is used by every method, including the two that ignore the mask."""
    validate_settings(settings)
    if method not in METHODS:
        raise ValueError("Unknown replay method")
    if episode['coordinate_frame'] != 'world_si' or episode['time_basis'] != 'source_time':
        raise ValueError("Replay requires world SI coordinates and source-time seconds")
    frames = episode['frames']
    if len(mask) != len(frames) or any(type(value) is not bool for value in mask):
        raise ValueError("One boolean delivery flag is required per source frame")
    if not frames:
        raise ValueError("Empty episode")
    if len({frame['frame_id'] for frame in frames}) != len(frames):
        raise ValueError("Frame IDs must be unique within an episode")
    shape = Footprint(**episode['ego_footprint'])
    stream_configs = stream_risk_configs(settings)
    thresholds = per_stream(settings['min_detection_confidence'], 'min_detection_confidence')
    # expiry and the other shared fields do not depend on the stream; the warning horizon is chosen per frame below
    risk_config = stream_configs['cooperative']
    tracker = (tracker_factory or ReferenceTracker)(dict(settings['tracker'],
                                                         expire_source_age_s=risk_config.expire_source_age_s))
    models, rows = {}, []
    for frame, delivered in zip(frames, mask):
        now = frame['time_s']
        use_cooperative = (method == 'cooperative_ideal' or
                           (method != 'ego_only' and delivered))
        stream = 'cooperative' if use_cooperative else 'ego_only'
        detections = [Detection(**item) for item in frame[stream]]
        detections = [item for item in detections
                      if item.confidence >= thresholds[stream]]
        updates = tracker.step(detections, now, stream)
        updated_ids = set()
        for estimate, target_shape in updates:
            if estimate.track_id not in models:
                models[estimate.track_id] = [FallbackCollisionModel(risk_config), target_shape]
            model = models[estimate.track_id][0]
            model.update(estimate, now)
            models[estimate.track_id][1] = target_shape
            updated_ids.add(estimate.track_id)
        ego = State(**frame['ego'])
        if not all(math.isfinite(value) for value in vars(ego).values()):
            raise ValueError("Ego state must be finite even when targets are unavailable")
        results = []
        if not frame.get('ego_motion_valid', True):
            rows.append({'frame_id': frame['frame_id'], 'time_s': now,
                         'partner_delivered': delivered, 'observation_stream': stream,
                         'method': method, 'warning': False, 'evidence_status': 'unknown',
                         'reason': 'ego_velocity_warmup', 'tracks': []})
            continue
        for key, (model, target_shape) in sorted(models.items()):
            if method in ('no_fallback', 'uncertainty_nofb') and key not in updated_ids:
                continue
            strategy = ('uncertainty' if method in ('uncertainty', 'uncertainty_nofb')
                        else ('hold' if method == 'hold' else 'cv'))
            # the frame's stream sets the warning horizon (cooperative on delivered frames, ego_only on lost ones)
            model.config = stream_configs[stream]
            result = model.evaluate(ego, now, shape, target_shape, strategy)
            results.append(asdict(result))
        active = [item for item in results if item['mode'] not in ('unavailable', 'expired')]
        # Unknown evidence is kept separate from a negative warning decision.
        rows.append({'frame_id': frame['frame_id'], 'time_s': now,
                     'partner_delivered': delivered, 'observation_stream': stream,
                     'method': method, 'warning': any(item['warning'] for item in active),
                     'evidence_status': 'available' if active else 'unknown',
                     'tracks': results})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", required=True, help="JSON lines, one episode per line (format above)")
    ap.add_argument("--out", required=True, help="JSON lines: one row per episode, method and frame")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--tracker", choices=("reference", "kalman"), default="reference")
    ap.add_argument("--sigma-a", type=float, default=0.1, help="process noise of the Kalman tracker, m/s^2")
    ap.add_argument("--methods", default=",".join(METHODS), help="comma list, default all")
    ap.add_argument("--loss", choices=("ideal", "bernoulli", "burst"), default="ideal")
    ap.add_argument("--drop-probability", type=float, default=0.0, help="bernoulli: probability that a frame's cooperative message is lost")
    ap.add_argument("--seed", default="0", help="bernoulli: base seed; the episode's mask seed is the first 8 hex digits of SHA-256('<seed>:<episode_id>')")
    ap.add_argument("--burst", nargs=2, type=float, action="append", default=[], metavar=("START", "END"),
                    help="burst: outage [START, END) in source-time seconds (repeatable)")
    a = ap.parse_args(argv)
    methods = a.methods.split(",")
    for m in methods:
        if m not in METHODS:
            ap.error("unknown method %r" % m)
    cfg = json.load(open(a.config))
    validate_settings(cfg)
    factory = None
    if a.tracker == "kalman":
        from tracking.kalman import kalman_factory
        factory = kalman_factory(a.sigma_a)
    n_rows = 0
    with open(a.episodes) as fin, open(a.out, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            episode = json.loads(line)
            frames = episode["frames"]
            if a.loss == "bernoulli":
                mask = delivery_mask(frames, "bernoulli", seed=episode_seed(a.seed, episode["episode_id"]), drop_probability=a.drop_probability)
            elif a.loss == "burst":
                mask = delivery_mask(frames, "burst", bursts=[tuple(b) for b in a.burst])
            else:
                mask = delivery_mask(frames, "ideal")
            for m in methods:
                for row in replay(episode, cfg, mask, m, factory):
                    fout.write(json.dumps(dict(row, episode_id=episode["episode_id"])) + "\n")
                    n_rows += 1
    print("rows", n_rows, "->", a.out)


def episode_seed(seed, episode_id):
    """Per-episode mask seed: first 8 hex digits of SHA-256("{seed}:{episode_id}"), so that all methods share one mask per episode."""
    return int(hashlib.sha256(("%s:%s" % (seed, episode_id)).encode("utf-8")).hexdigest()[:8], 16)


if __name__ == "__main__":
    main()
