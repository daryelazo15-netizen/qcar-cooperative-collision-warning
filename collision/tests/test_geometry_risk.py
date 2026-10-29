"""Tests of the swept-footprint geometry and the collision risk model.   python3 -m unittest discover -s collision/tests"""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from collision.geometry import Footprint, State, axes, radius, swept_contact  # noqa: E402
from collision.risk import (FallbackCollisionModel, FallbackConfig, TrackEstimate, growing_envelope_contact, inflate_footprint,  # noqa: E402
                            propagate_cv)

CAR = Footprint(0.4, 0.2)
CFG = FallbackConfig(live_update_timeout_s=0.1, expire_source_age_s=0.6, forecast_horizon_s=2.0, warning_horizon_s=0.8,
                     acceleration_noise_std=0.01, sigma_multiplier=0.25, forecast_interval_s=0.05)


def track(x, vx, t=0.0, var=1e-4, vvar=1e-3, yaw=0.0, y=0.0, vy=0.0):
    return TrackEstimate("t", State(x, y, yaw, vx, vy), t, (var, var), (vvar, vvar), 0.9, "cooperative")


class TestGeometry(unittest.TestCase):
    def test_footprint_validation(self):
        for bad in ((0, 1), (1, -1), (float("nan"), 1)):
            with self.assertRaises(ValueError):
                Footprint(*bad)

    def test_axes_are_orthonormal(self):
        (a, b), (c, d) = axes(0.7)
        self.assertAlmostEqual(a * c + b * d, 0.0)
        self.assertAlmostEqual(math.hypot(a, b), 1.0)

    def test_radius_of_axis_aligned_box(self):
        self.assertAlmostEqual(radius(CAR, 0.0, (1, 0)), 0.2)
        self.assertAlmostEqual(radius(CAR, 0.0, (0, 1)), 0.1)
        self.assertAlmostEqual(radius(CAR, math.pi / 2, (1, 0)), 0.1)

    def test_head_on_contact_time(self):
        ego = State(0, 0, 0, 0, 0)
        target = State(2.0, 0, 0, -1.0, 0)
        lo, hi = swept_contact(ego, target, CAR, CAR, 5.0)
        self.assertAlmostEqual(lo, 1.6)       # gap 2.0 - 0.4 closes at 1 m/s
        self.assertAlmostEqual(hi, 2.4)       # fully through after 2.0 + 0.4

    def test_no_contact_beyond_horizon_or_laterally(self):
        ego = State(0, 0, 0, 0, 0)
        self.assertIsNone(swept_contact(ego, State(2.0, 0, 0, -1.0, 0), CAR, CAR, 1.0))
        self.assertIsNone(swept_contact(ego, State(2.0, 0.5, 0, -1.0, 0), CAR, CAR, 5.0))     # passes 0.5 m to the side

    def test_already_overlapping_gives_zero(self):
        lo, _ = swept_contact(State(0, 0, 0, 0, 0), State(0.1, 0, 0, 0, 0), CAR, CAR, 1.0)
        self.assertEqual(lo, 0.0)

    def test_rotated_target(self):
        ego = State(0, 0, 0, 0, 0)
        lo, _ = swept_contact(ego, State(2.0, 0, math.pi / 2, -1.0, 0), CAR, CAR, 5.0)
        self.assertAlmostEqual(lo, 2.0 - 0.2 - 0.1)         # the target's half-width faces the ego

    def test_invalid_inputs(self):
        ego = State(0, 0, 0, 0, 0)
        with self.assertRaises(ValueError):
            swept_contact(ego, ego, CAR, CAR, -1.0)
        with self.assertRaises(ValueError):
            swept_contact(ego, State(float("nan"), 0, 0, 0, 0), CAR, CAR, 1.0)


class TestRisk(unittest.TestCase):
    def test_config_validation(self):
        base = dict(live_update_timeout_s=0.1, expire_source_age_s=0.6, forecast_horizon_s=2.0, warning_horizon_s=0.8,
                    acceleration_noise_std=0.01, sigma_multiplier=0.25)
        FallbackConfig(**base)
        for patch in ({"live_update_timeout_s": 0.7}, {"warning_horizon_s": 2.5}, {"sigma_multiplier": 0.0}, {"forecast_interval_s": 0.0},
                      {"expire_source_age_s": -1.0}):
            with self.assertRaises(ValueError):
                FallbackConfig(**dict(base, **patch))

    def test_track_validation(self):
        with self.assertRaises(ValueError):
            TrackEstimate("", State(0, 0, 0, 0, 0), 0.0, (0, 0), (0, 0), 0.5, "x")
        with self.assertRaises(ValueError):
            TrackEstimate("t", State(0, 0, 0, 0, 0), 0.0, (-1, 0), (0, 0), 0.5, "x")
        with self.assertRaises(ValueError):
            TrackEstimate("t", State(0, 0, 0, 0, 0), 0.0, (0, 0), (0, 0), 1.5, "x")

    def test_propagate_cv(self):
        p = propagate_cv(track(1.0, 0.5, var=1e-4, vvar=1e-3), 2.0, 0.1)
        self.assertAlmostEqual(p.state.x, 2.0)
        self.assertAlmostEqual(p.position_variance[0], 1e-4 + 4 * 1e-3 + 0.25 * 16 * 0.01)
        self.assertAlmostEqual(p.velocity_variance[0], 1e-3 + 4 * 0.01)
        with self.assertRaises(ValueError):
            propagate_cv(track(1.0, 0.5, t=1.0), 0.5, 0.1)

    def test_inflate_footprint_projects_variance_on_body_axes(self):
        s = inflate_footprint(CAR, 0.0, (0.04, 0.0), 0.5)         # std 0.2 along x only
        self.assertAlmostEqual(s.length, 0.4 + 2 * 0.5 * 0.2)
        self.assertAlmostEqual(s.width, 0.2)
        s = inflate_footprint(CAR, math.pi / 2, (0.04, 0.0), 0.5)  # body rotated: x variance now lies along the width
        self.assertAlmostEqual(s.length, 0.4)
        self.assertAlmostEqual(s.width, 0.2 + 0.2)

    def test_envelope_is_conservative_against_nominal(self):
        ego = State(0, 0, 0, 0, 0)
        t = track(2.0, -1.0, var=0.01, vvar=0.01)
        nominal = swept_contact(ego, t.state, CAR, CAR, CFG.forecast_horizon_s)[0]
        env = growing_envelope_contact(ego, t, 0.0, CAR, CAR, CFG)
        self.assertLessEqual(env, nominal + 1e-9)

    def evaluate(self, strategy, now, ego=State(0, 0, 0, 0, 0), update_at=0.0, x=2.0, vx=-1.0):
        m = FallbackCollisionModel(CFG)
        m.update(track(x, vx, t=update_at), update_at)
        return m.evaluate(ego, now, CAR, CAR, strategy)

    def test_unavailable_live_fallback_expired(self):
        self.assertEqual(FallbackCollisionModel(CFG).evaluate(State(0, 0, 0, 0, 0), 0.0, CAR, CAR).mode, "unavailable")
        self.assertEqual(self.evaluate("cv", 0.05).mode, "live")
        self.assertEqual(self.evaluate("cv", 0.3).mode, "fallback")
        r = self.evaluate("cv", 0.7)
        self.assertEqual((r.mode, r.warning), ("expired", False))

    def test_warning_horizon(self):
        # gap closes at 1 m/s from 1.6 m: contact at 1.6 s, warns once it is within 0.8 s
        self.assertFalse(self.evaluate("cv", 0.0).warning)
        self.assertTrue(self.evaluate("cv", 0.0, x=1.2).warning)

    def test_hold_does_not_move_the_stale_track_but_cv_does(self):
        hold = self.evaluate("hold", 0.5)
        cv = self.evaluate("cv", 0.5)
        self.assertAlmostEqual(hold.target_state.x, 2.0)
        self.assertAlmostEqual(cv.target_state.x, 1.5)
        self.assertGreater(cv.urgency_score, hold.urgency_score)

    def test_strategy_validation_and_ordering_rules(self):
        m = FallbackCollisionModel(CFG)
        m.update(track(2.0, -1.0, t=1.0), 1.0)
        with self.assertRaises(ValueError):
            m.evaluate(State(0, 0, 0, 0, 0), 1.0, CAR, CAR, "nope")
        with self.assertRaises(ValueError):
            m.update(track(2.0, -1.0, t=1.0), 1.1)           # same track, source time did not advance
        with self.assertRaises(ValueError):
            m.update(track(2.0, -1.0, t=0.5), 0.9)           # receive time went back

    def test_uncertainty_warns_no_later_than_cv(self):
        for now in (0.0, 0.2, 0.4):
            u = self.evaluate("uncertainty", now, x=1.7)
            c = self.evaluate("cv", now, x=1.7)
            if c.warning:
                self.assertTrue(u.warning)
            self.assertLessEqual(u.conservative_contact_s, c.conservative_contact_s + 1e-9)


if __name__ == "__main__":
    unittest.main()
