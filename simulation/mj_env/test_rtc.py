"""Safety and time-alignment checks for the asynchronous RTC controller."""

from __future__ import annotations

import threading
import unittest

import numpy as np

from mj_env.rtc import RTCController, RTCDeadlineMissed


class RTCControllerTest(unittest.TestCase):
    def test_overlap_is_aligned_to_request_tick(self):
        requests = []

        def infer(request):
            requests.append(request)
            value = len(requests)
            return {
                "actions": np.full((10, 6), value, dtype=np.float32),
            }

        controller = RTCController(infer, horizon=10, min_horizon=3, initial_delay=2)
        try:
            controller.start({"observation": 0})
            for _ in range(4):
                controller.get_action({"observation": 1})
            controller._pending.result(timeout=2)
            action = controller.get_action({"observation": 2})
            self.assertEqual(requests[1]["rtc_delay_steps"], 2)
            self.assertEqual(requests[1]["rtc_overlap_steps"], 7)
            np.testing.assert_array_equal(
                requests[1]["rtc_previous_actions"][:7],
                np.full((7, 6), 1, dtype=np.float32),
            )
            self.assertTrue(np.all(requests[1]["rtc_previous_actions"][7:] == 0))
            self.assertTrue(np.all(action == 2))
        finally:
            controller.close()

    def test_never_uses_an_action_beyond_chunk(self):
        gate = threading.Event()
        calls = 0

        def infer(_request):
            nonlocal calls
            calls += 1
            if calls > 1:
                gate.wait(timeout=2)
            return {
                "actions": np.zeros((4, 6), dtype=np.float32),
            }

        controller = RTCController(infer, horizon=4, min_horizon=1, initial_delay=1)
        try:
            controller.start({})
            for _ in range(4):
                controller.get_action({})
            with self.assertRaises(RTCDeadlineMissed):
                controller.get_action({})
        finally:
            gate.set()
            controller.close()

    def test_request_window_advances_for_measured_latency(self):
        controller = RTCController(
            lambda _request: {"actions": np.zeros((10, 6), dtype=np.float32)},
            horizon=10,
            min_horizon=5,
            initial_delay=7,
            latency_margin_steps=1,
        )
        try:
            self.assertEqual(controller.request_step(), 2)
            self.assertEqual(controller.forecast_delay_steps, 7)
        finally:
            controller.close()

    def test_rejects_latency_that_cannot_fit(self):
        controller = RTCController(
            lambda _request: {"actions": np.zeros((10, 6), dtype=np.float32)},
            horizon=10,
            min_horizon=3,
            initial_delay=9,
            latency_margin_steps=1,
        )
        try:
            with self.assertRaises(RTCDeadlineMissed):
                controller.request_step()
        finally:
            controller.close()


if __name__ == "__main__":
    unittest.main()
