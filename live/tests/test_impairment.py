import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from impairment import Injector  # noqa: E402


def fr(t):
    return {"stamp": t}


class T(unittest.TestCase):
    def test_none(self):
        d = Injector().decide(5.0, fr(10.0), fr(10.01))
        self.assertEqual(d["status"], "used")
        self.assertAlmostEqual(d["delay_s"], -0.01)

    def test_window_half_open_after_ready(self):
        inj = Injector({"type": "window", "x": 20, "y": 10})
        inj.set_ready(1000.0)
        st = lambda now: inj.decide(now, fr(now), fr(now))["status"]
        self.assertEqual([st(1019.99), st(1020.0), st(1029.99), st(1030.0)], ["used", "withheld", "withheld", "used"])
        with self.assertRaises(RuntimeError):
            Injector({"type": "window", "x": 1, "y": 1}).decide(0, fr(0), fr(0))

    def test_trigger_window(self):
        with tempfile.TemporaryDirectory() as d:
            trig = os.path.join(d, "go")
            inj = Injector({"type": "window_trigger", "y": 3.0, "trigger_file": trig})
            inj.set_ready(0.0)
            self.assertEqual(inj.decide(10.0, fr(10), fr(10))["status"], "used")
            open(trig, "w").close()
            self.assertEqual(inj.decide(11.0, fr(11), fr(11))["status"], "withheld")
            self.assertEqual(inj.decide(13.9, fr(13.9), fr(13.9))["status"], "withheld")
            self.assertEqual(inj.decide(14.0, fr(14), fr(14))["status"], "used")
            self.assertEqual(inj.t_trigger, 11.0)

    def test_bernoulli_seeded_and_rate(self):
        run = lambda s: [Injector({"type": "bernoulli", "p": 0.25, "seed": s}).decide(0, fr(0), fr(0))["status"] for _ in range(1)]
        a, b = Injector({"type": "bernoulli", "p": 0.25, "seed": 7}), Injector({"type": "bernoulli", "p": 0.25, "seed": 7})
        sa = [a.decide(i, fr(i), fr(i))["status"] for i in range(2000)]
        sb = [b.decide(i, fr(i), fr(i))["status"] for i in range(2000)]
        self.assertEqual(sa, sb)
        self.assertAlmostEqual(sa.count("withheld") / 2000, 0.25, delta=0.03)
        self.assertEqual([Injector({"type": "bernoulli", "p": 0.0, "seed": 1}).decide(0, fr(0), fr(0))["status"]], ["used"])
        self.assertEqual([Injector({"type": "bernoulli", "p": 1.0, "seed": 1}).decide(0, fr(0), fr(0))["status"]], ["withheld"])

    def test_delay_picks_older_frame_or_withholds(self):
        inj = Injector({"type": "delay", "d": 0.5})
        hist = [fr(9.40), fr(9.50), fr(9.60), fr(9.70)]
        d = inj.decide(10.0, fr(10.0), fr(10.0), hist)
        self.assertEqual((d["status"], d["peer"]["stamp"]), ("delayed", 9.50))
        self.assertAlmostEqual(d["delay_s"], 0.5)
        d = inj.decide(10.0, fr(10.0), fr(10.0), [fr(9.0)])
        self.assertEqual((d["status"], d["reason"]), ("withheld", "no_delayed_frame"))
        self.assertTrue(Injector({"type": "delay", "d": 0.5}).needs_history())
        self.assertEqual(Injector({"type": "delay", "d": 0.0}).decide(1, fr(1.0), fr(1.01))["status"], "used")

    def test_bad_specs(self):
        for s in ({"type": "x"}, {"type": "bernoulli", "p": 2, "seed": 1}, {"type": "window", "x": -1, "y": 1}, {"type": "delay", "d": -1}):
            with self.assertRaises(ValueError):
                Injector(s)


if __name__ == "__main__":
    unittest.main()
