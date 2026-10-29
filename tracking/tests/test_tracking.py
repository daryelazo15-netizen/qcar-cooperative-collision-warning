"""Tests of the reference and Kalman trackers.   python3 -m unittest discover -s tracking/tests"""
import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from collision.detection import Detection  # noqa: E402
from tracking import kalman as K  # noqa: E402
from tracking.kalman import KalmanTracker, kalman_factory  # noqa: E402
from tracking.reference import ReferenceTracker  # noqa: E402

SET = {"association_gate_m": 0.15, "max_speed_mps": 2.0, "position_noise_std_m": 0.0207, "expire_source_age_s": 0.6}


def det(x, y):
    return Detection(x, y, 0.0, 0.4, 0.2, 0.9)


def run(factory, seq):
    """seq = [(t, [(x, y), ...])] -> [(t, {key: estimate})] of the updated tracks."""
    tr = factory(SET)
    out = []
    for t, pts in seq:
        out.append((t, {e.track_id: e for e, _ in tr.step([det(*p) for p in pts], t, "cooperative")}))
    return out, tr


def noisy_line(n=30, vx=0.35, vy=-0.1, sd=0.0207, seed=1, dt=0.1):
    rng = random.Random(seed)
    return [(round(i * dt, 6), [(1.0 + vx * i * dt + rng.gauss(0, sd), 2.0 + vy * i * dt + rng.gauss(0, sd))]) for i in range(n)]


class TestReference(unittest.TestCase):
    def test_two_point_velocity_and_confirmation(self):
        out, _ = run(ReferenceTracker, [(0.0, [(1.0, 1.0)]), (0.1, [(1.03, 1.0)]), (0.2, [(1.07, 1.01)])])
        self.assertEqual(out[0][1], {})                                   # one observation: not confirmed
        e = out[1][1]["track-0"]
        self.assertAlmostEqual(e.state.vx, 0.3)
        self.assertAlmostEqual(e.state.vy, 0.0)
        self.assertAlmostEqual(e.velocity_variance[0], 2 * SET["position_noise_std_m"] ** 2 / 0.01)
        self.assertEqual(e.position_variance, (SET["position_noise_std_m"] ** 2,) * 2)
        self.assertEqual(e.provenance, "cooperative")
        self.assertAlmostEqual(out[2][1]["track-0"].state.vx, 0.4)

    def test_gate_new_track_and_expiry(self):
        seq = [(0.0, [(1.0, 1.0)]), (0.1, [(1.0, 1.0)]), (0.2, [(3.0, 3.0)]), (1.0, [(3.0, 3.0)]), (1.1, [(3.0, 3.0)])]
        out, tr = run(ReferenceTracker, seq)
        self.assertEqual(sorted(out[1][1]), ["track-0"])
        self.assertEqual(out[2][1], {})                                    # outside the gate: a new track, one observation
        self.assertEqual(out[3][1], {})                                    # track-0 and track-1 expired / not reachable within the speed bound
        self.assertIn("track-2", tr.records)

    def test_timestamps_must_increase(self):
        tr = ReferenceTracker(SET)
        tr.step([det(1, 1)], 1.0, "cooperative")
        with self.assertRaises(ValueError):
            tr.step([det(1, 1)], 1.0, "cooperative")

    def test_speed_bound(self):
        tr = ReferenceTracker(dict(SET, association_gate_m=5.0))
        tr.step([det(0, 0)], 0.0, "cooperative")
        tr.step([det(0.1, 0)], 0.1, "cooperative")
        # 0.5 m in 0.1 s = 5 m/s > max_speed 2 m/s: never associated, becomes a new track
        self.assertEqual(tr.step([det(0.6, 0)], 0.2, "cooperative"), [])


class TestKalman(unittest.TestCase):
    def test_association_and_ids_identical_to_reference(self):
        seq = noisy_line(40)
        seq = [(t, p + [(5.0 + 0.3 * t, 1.0)] if t < 2.0 else p) for t, p in seq]      # a second target that disappears
        base, _ = run(ReferenceTracker, seq)
        v, _ = run(kalman_factory(0.3), seq)
        self.assertEqual([(t, sorted(d)) for t, d in base], [(t, sorted(d)) for t, d in v])
        for (t, d0), (_, d1) in zip(base, v):
            for k in d0:
                self.assertEqual(d0[k].state.yaw, d1[k].state.yaw)
                self.assertEqual(d0[k].source_time_s, d1[k].source_time_s)

    def test_equals_reference_at_second_observation(self):
        seq = noisy_line(5)
        base, _ = run(ReferenceTracker, seq)
        for sa in (0.01, 1.0):
            v, _ = run(kalman_factory(sa), seq)
            self.assertEqual(base[1][1], v[1][1])           # second observation: whole estimate identical (state and variances)
            self.assertNotEqual(base[2][1], v[2][1])         # third: differs

    def test_noise_free_constant_velocity(self):
        seq = noisy_line(20, sd=0.0)
        v, _ = run(kalman_factory(0.1), seq)
        e = list(v[-1][1].values())[0]
        self.assertAlmostEqual(e.state.vx, 0.35, places=9)
        self.assertAlmostEqual(e.state.x, 1.0 + 0.35 * 1.9, places=9)
        self.assertLess(e.position_variance[0], SET["position_noise_std_m"] ** 2)          # filtered variance below the raw detection variance
        self.assertLess(e.velocity_variance[0], 2 * SET["position_noise_std_m"] ** 2 / 0.01)

    def test_matches_matrix_form_recursion(self):
        # an independent 2x2 recursion for one axis, written in matrix form with the Joseph covariance update
        s2, q, dt = 0.0207 ** 2, 0.3 ** 2, 0.1
        zs = [1.0, 1.04, 1.07, 1.12, 1.15]
        x, P = K._kf_init(zs[0], zs[1], dt, s2)
        for z in zs[2:]:
            F = [[1, dt], [0, 1]]
            Q = [[q * dt ** 4 / 4, q * dt ** 3 / 2], [q * dt ** 3 / 2, q * dt ** 2]]
            xp = [F[0][0] * x[0] + F[0][1] * x[1], F[1][0] * x[0] + F[1][1] * x[1]]
            FP = [[sum(F[i][m] * P[m][j] for m in range(2)) for j in range(2)] for i in range(2)]
            Pp = [[sum(FP[i][m] * F[j][m] for m in range(2)) + Q[i][j] for j in range(2)] for i in range(2)]
            S_ = Pp[0][0] + s2
            Kg = [Pp[0][0] / S_, Pp[1][0] / S_]
            xn = [xp[0] + Kg[0] * (z - xp[0]), xp[1] + Kg[1] * (z - xp[0])]
            IKH = [[1 - Kg[0], 0], [-Kg[1], 1]]
            A = [[sum(IKH[i][m] * Pp[m][j] for m in range(2)) for j in range(2)] for i in range(2)]
            Pn = [[sum(A[i][m] * IKH[j][m] for m in range(2)) + Kg[i] * s2 * Kg[j] for j in range(2)] for i in range(2)]
            x, P = K._kf_step(x, P, z, dt, s2, q)
            for a, b in zip(x, xn):
                self.assertAlmostEqual(a, b, places=12)
            for i in range(2):
                for j in range(2):
                    self.assertAlmostEqual(P[i][j], Pn[i][j], places=12)
            x, P = xn, Pn

    def test_history_is_bounded_and_dropped_with_the_track(self):
        _, tr = run(kalman_factory(0.1), noisy_line(40))
        self.assertLessEqual(max(len(h) for h in tr.hist.values()), 12)
        tr.step([], 100.0, "cooperative")
        self.assertEqual((tr.hist, tr.kf), ({}, {}))

    def test_invalid_sigma_a(self):
        for bad in (-0.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                KalmanTracker(SET, sigma_a=bad)


if __name__ == "__main__":
    unittest.main()
