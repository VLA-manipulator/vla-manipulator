"""RTC soft overlap weights for action chunks."""

from __future__ import annotations

import numpy as np


def prefix_weights(delay: int, overlap: int, horizon: int) -> np.ndarray:
    """Return the exponential soft mask from Black et al., Eq. 5."""
    if not (0 <= delay <= overlap <= horizon):
        raise ValueError("require 0 <= delay <= overlap <= horizon")
    index = np.arange(horizon, dtype=np.float32)
    c = np.clip((delay - 1 - index) / (overlap - delay + 1) + 1, 0, 1)
    weights = c * np.expm1(c) / np.expm1(1.0)
    weights[index < delay] = 1.0
    weights[index >= overlap] = 0.0
    return weights
