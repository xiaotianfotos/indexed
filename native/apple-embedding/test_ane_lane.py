import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from ane_lane import ANELane


class ANELaneTests(unittest.TestCase):
    def test_recurrence_precedes_queued_vision_and_exception_releases_lane(self):
        lane = ANELane()
        order = []
        def claim(label, priority):
            with lane.claim(priority=priority): order.append(label)
        with ThreadPoolExecutor(2) as pool:
            with lane.claim():
                low = pool.submit(claim, "vision", False)
                high = pool.submit(claim, "recurrence", True)
                deadline = time.monotonic() + 2
                while not lane._priority_waiters and time.monotonic() < deadline:
                    time.sleep(0.001)
                self.assertEqual(lane._priority_waiters, 1)
            high.result(timeout=2)
            low.result(timeout=2)
        self.assertEqual(order, ["recurrence", "vision"])
        with self.assertRaises(RuntimeError):
            with lane.claim(priority=True): raise RuntimeError("test")
        with lane.claim(): pass


if __name__ == '__main__': unittest.main()
