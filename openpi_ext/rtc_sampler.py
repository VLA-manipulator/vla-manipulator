"""Guided RTC sampler for the JAX OpenPI Pi0/Pi0.5 model.

Import this module only inside an OpenPI environment.  The model is unchanged
until ``install_rtc_sampler`` is called by the opt-in RTC server.
"""

from __future__ import annotations

import einops
import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models import pi0


def _weights(delay: jax.Array, overlap: jax.Array, horizon: int) -> jax.Array:
    index = jnp.arange(horizon, dtype=jnp.float32)
    delay = jnp.asarray(delay, dtype=jnp.float32)
    overlap = jnp.asarray(overlap, dtype=jnp.float32)
    c = jnp.clip((delay - 1 - index) / (overlap - delay + 1) + 1, 0, 1)
    soft = c * jnp.expm1(c) / jnp.expm1(1.0)
    return jnp.where(index < delay, 1.0, jnp.where(index >= overlap, 0.0, soft))


def sample_actions_rtc(
    self: pi0.Pi0,
    rng: jax.Array,
    observation: _model.Observation,
    *,
    previous_actions: jax.Array,
    inference_delay: jax.Array,
    overlap: jax.Array,
    num_steps: int = 5,
    max_guidance_weight: float = 5.0,
) -> jax.Array:
    """Sample a chunk with the paper's soft-mask pseudoinverse guidance.

    ``previous_actions`` is right-padded to the model horizon and must already
    be in OpenPI's normalized model action space.  OpenPI integrates flow from
    noise at t=1 to actions at t=0, the reverse of the paper's convention.
    """
    observation = _model.preprocess_observation(None, observation, train=False)
    batch_size = observation.state.shape[0]
    horizon = self.action_horizon
    previous_actions = jnp.asarray(previous_actions, dtype=jnp.float32)
    if previous_actions.shape != (batch_size, horizon, self.action_dim):
        raise ValueError("previous_actions must have shape (batch, horizon, action_dim)")

    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    prefix_attn_mask = pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
    kv_cache = jax.lax.stop_gradient(kv_cache)
    weights = _weights(inference_delay, overlap, horizon)[None, :, None]
    target = jax.lax.stop_gradient(previous_actions)

    def velocity(actions: jax.Array, time: jax.Array) -> jax.Array:
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, actions, jnp.broadcast_to(time, batch_size)
        )
        suffix_mask_local = pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_mask_local = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_mask = jnp.concatenate([prefix_mask_local, suffix_mask_local], axis=-1)
        suffix_positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_mask,
            positions=suffix_positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        return self.action_out_proj(suffix_out[:, -horizon:])

    def step(index: int, actions: jax.Array) -> jax.Array:
        time = 1.0 - index / num_steps

        def denoiser(x: jax.Array) -> tuple[jax.Array, jax.Array]:
            v = velocity(x, time)
            return x - time * v, v

        predicted, vjp, v = jax.vjp(denoiser, actions, has_aux=True)
        correction = vjp((target - predicted) * weights)[0]
        # Eq. 2 with tau=1-time.  Clip the singular endpoint as in the paper.
        coefficient = (time**2 + (1.0 - time) ** 2) / jnp.maximum(time * (1.0 - time), 1e-6)
        coefficient = jnp.minimum(coefficient, max_guidance_weight)
        return actions - (v - coefficient * correction) / num_steps

    noise = jax.random.normal(rng, (batch_size, horizon, self.action_dim))
    return jax.lax.fori_loop(0, num_steps, step, noise)


def install_rtc_sampler() -> None:
    """Attach the sampler to Pi0 without editing an upstream checkout."""
    pi0.Pi0.sample_actions_rtc = sample_actions_rtc
