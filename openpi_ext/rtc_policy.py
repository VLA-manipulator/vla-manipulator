"""Opt-in OpenPI policy exposing RTC actions in model-normalized space."""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.shared import nnx_utils

from openpi_ext.rtc_sampler import install_rtc_sampler


class RTCPolicy:
    """Wrap an existing JAX OpenPI policy without changing normal inference."""

    def __init__(self, base):
        if base._is_pytorch_model:
            raise TypeError("RTCPolicy currently supports only the JAX Pi0/Pi0.5 model")
        install_rtc_sampler()
        self._base = base
        self._sample_rtc = nnx_utils.module_jit(base._model.sample_actions_rtc)

    @property
    def metadata(self):
        return {**self._base.metadata, "rtc": True}

    def infer(self, obs: dict) -> dict:
        request = dict(obs)
        previous = request.pop("rtc_previous_model_actions", None)
        delay = request.pop("rtc_delay_steps", None)
        overlap = request.pop("rtc_overlap_steps", None)
        if previous is not None and (delay is None or overlap is None):
            raise ValueError("RTC request needs delay and overlap with previous actions")
        inputs = self._base._input_transform(request)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
        self._base._rng, rng = jax.random.split(self._base._rng)
        observation = _model.Observation.from_dict(inputs)
        start = time.monotonic()
        if previous is None:
            model_actions = self._base._sample_actions(rng, observation, **self._base._sample_kwargs)
        else:
            previous = np.asarray(previous, dtype=np.float32)
            horizon = self._base._model.action_horizon
            dimension = self._base._model.action_dim
            if previous.shape != (horizon, dimension) or not np.isfinite(previous).all():
                raise ValueError(f"RTC previous actions must be finite {(horizon, dimension)}")
            if not (0 <= int(delay) <= int(overlap) <= horizon):
                raise ValueError("RTC requires 0 <= delay <= overlap <= horizon")
            sample_kwargs = dict(self._base._sample_kwargs)
            sample_kwargs.pop("noise", None)
            model_actions = self._sample_rtc(
                rng,
                observation,
                previous_actions=jnp.asarray(previous)[None, ...],
                inference_delay=jnp.asarray(delay, dtype=jnp.int32),
                overlap=jnp.asarray(overlap, dtype=jnp.int32),
                **sample_kwargs,
            )
        model_actions = np.asarray(model_actions[0], dtype=np.float32).copy()
        outputs = {"state": np.asarray(inputs["state"][0]), "actions": model_actions}
        result = self._base._output_transform(outputs)
        if not np.isfinite(np.asarray(result["actions"])).all():
            raise ValueError("RTC model returned non-finite actions")
        result["rtc_model_actions"] = model_actions
        result["policy_timing"] = {"infer_ms": (time.monotonic() - start) * 1000}
        return result
