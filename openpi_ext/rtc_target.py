"""Convert an old physical action tail into the current observation's model space."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def prepare_guided_inputs(
    input_transform: Callable[[dict], dict],
    observation: dict,
    previous_actions: np.ndarray,
    *,
    horizon: int,
    action_dim: int,
) -> tuple[dict, np.ndarray]:
    """Re-anchor previous absolute targets using OpenPI's existing SO101 transforms.

    SO101's first five model actions are joint deltas relative to the state at
    inference time.  A previous normalized chunk therefore cannot be used as a
    guidance target at a later observation without recomputing those deltas.
    """
    state = np.asarray(observation["observation/state"])
    actions = np.asarray(previous_actions, dtype=np.float32)
    if state.ndim != 1 or actions.shape != (horizon, state.shape[0]):
        raise ValueError("RTC previous actions must match the horizon and physical state dimension")
    if not np.isfinite(state).all() or not np.isfinite(actions).all():
        raise ValueError("RTC previous actions and state must be finite")
    request = dict(observation)
    request["actions"] = actions.copy()
    inputs = input_transform(request)
    target = np.asarray(inputs.pop("actions"), dtype=np.float32)
    if target.shape != (horizon, action_dim) or not np.isfinite(target).all():
        raise ValueError("RTC action transform returned an invalid model-space target")
    return inputs, target
