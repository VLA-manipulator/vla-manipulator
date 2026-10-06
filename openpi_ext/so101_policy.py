"""SO101 transforms for OpenPI training and inference.

This follows OpenPI's official ``libero_policy.py`` adapter.  The only robot
specific choices are the two available camera streams and six action values.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from openpi import transforms
from openpi.models import model as _model


ACTION_DIM = 6


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * np.clip(image, 0.0, 1.0)).astype(np.uint8)
    if image.ndim != 3:
        raise ValueError(f"image must have 3 dimensions, got {image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] != 3:
        raise ValueError(f"image must have 3 channels, got {image.shape}")
    return np.asarray(image, dtype=np.uint8)


@dataclasses.dataclass(frozen=True)
class SO101Inputs(transforms.DataTransformFn):
    """Map SO101 observations into OpenPI's canonical model input schema."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["observation/state"], dtype=np.float32)
        if state.shape[-1] != ACTION_DIM:
            raise ValueError(f"SO101 state must end in 6 values, got {state.shape}")
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        if wrist_image.shape != base_image.shape:
            raise ValueError("global and wrist images must have the same shape")

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_
                if self.model_type == _model.ModelType.PI0_FAST
                else np.False_,
            },
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.shape[-1] != ACTION_DIM:
                raise ValueError(f"SO101 actions must end in 6 values, got {actions.shape}")
            inputs["actions"] = actions
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class SO101Outputs(transforms.DataTransformFn):
    """Remove OpenPI's action padding and return six SO101 motor targets."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"], dtype=np.float32)
        if actions.shape[-1] < ACTION_DIM:
            raise ValueError(f"model returned fewer than 6 action values: {actions.shape}")
        return {"actions": actions[..., :ACTION_DIM]}
