"""Asynchronous action-chunk execution for the opt-in RTC policy server."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import math
import time
from typing import Callable

import numpy as np


class RTCDeadlineMissed(RuntimeError):
    """Inference did not finish before the current action chunk ran out."""


class RTCController:
    """Execute old actions while the next, overlap-guided chunk is sampled.

    ``infer`` takes a request dictionary and returns the RTC server response.
    Calls are serialized on one worker thread, including the initial chunk.
    ``get_action`` is called once per control tick, before ``env.step``.
    """

    def __init__(
        self,
        infer: Callable[[dict], dict],
        *,
        horizon: int,
        action_dim: int = 6,
        min_horizon: int = 3,
        initial_delay: int = 2,
        delay_history: int = 10,
    ) -> None:
        if not (0 < min_horizon < horizon):
            raise ValueError("min_horizon must be between 1 and horizon-1")
        if not (0 <= initial_delay <= horizon - min_horizon):
            raise ValueError("initial_delay exceeds the overlap budget")
        self._infer = infer
        self.horizon = horizon
        self.action_dim = action_dim
        self.min_horizon = min_horizon
        self._delays: deque[int] = deque([initial_delay], maxlen=delay_history)
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rtc-inference")
        self._current: dict | None = None
        self._chunk_start = 0
        self._tick = 0
        self._pending: Future | None = None
        self._request_tick = 0
        self._deadline_misses = 0
        self._infer_times_ms: list[float] = []

    @property
    def deadline_misses(self) -> int:
        return self._deadline_misses

    @property
    def infer_times_ms(self) -> tuple[float, ...]:
        return tuple(self._infer_times_ms)

    def start(self, observation: dict) -> None:
        if self._current is not None:
            raise RuntimeError("RTCController.start may only be called once")
        self._current = self._check(self._worker.submit(self._infer, dict(observation)).result())

    def _check(self, response: dict) -> dict:
        timing = response.get("policy_timing", {}).get("infer_ms")
        if timing is not None and math.isfinite(float(timing)):
            self._infer_times_ms.append(float(timing))
        actions = np.asarray(response["actions"], dtype=np.float32)
        if actions.shape != (self.horizon, self.action_dim):
            raise ValueError(f"invalid RTC actions shape {actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("RTC policy returned non-finite actions")
        return {"actions": actions}

    def get_action(self, observation: dict) -> np.ndarray:
        if self._current is None:
            raise RuntimeError("call start before get_action")
        if self._pending is not None and self._pending.done():
            next_chunk = self._check(self._pending.result())
            elapsed = self._tick - self._request_tick
            self._delays.append(elapsed)
            self._current = next_chunk
            self._chunk_start = self._request_tick
            self._pending = None

        index = self._tick - self._chunk_start
        if index >= self.horizon:
            self._deadline_misses += 1
            raise RTCDeadlineMissed(f"inference exceeded the {self.horizon}-step chunk")

        if self._pending is None and index >= self.min_horizon:
            overlap = self.horizon - index
            delay = max(self._delays)
            if delay > overlap:
                self._deadline_misses += 1
                raise RTCDeadlineMissed(f"forecast delay {delay} exceeds overlap {overlap}")
            previous = np.zeros_like(self._current["actions"])
            previous[:overlap] = self._current["actions"][index:]
            request = dict(observation)
            request.update(
                rtc_previous_actions=previous,
                rtc_delay_steps=delay,
                rtc_overlap_steps=overlap,
            )
            self._request_tick = self._tick
            self._pending = self._worker.submit(self._infer, request)

        action = np.array(self._current["actions"][index], copy=True)
        self._tick += 1
        return action

    def close(self) -> None:
        self._worker.shutdown(wait=True, cancel_futures=True)


class RealTimePacer:
    """Keep simulation control ticks at the same wall-clock rate as the robot."""

    def __init__(self, control_hz: float):
        if not math.isfinite(control_hz) or control_hz <= 0:
            raise ValueError("control_hz must be positive")
        self._period = 1.0 / control_hz
        self._next = time.monotonic()

    def wait(self) -> None:
        remaining = self._next - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        self._next = max(self._next + self._period, time.monotonic())
