from __future__ import annotations

import enum
import importlib.util
from pathlib import Path
import sys
import types
import unittest

import numpy as np


class _DataTransformFn:
    pass


class _ModelType(enum.Enum):
    PI0 = "pi0"
    PI05 = "pi05"
    PI0_FAST = "pi0_fast"


def _load_policy_module():
    openpi = types.ModuleType("openpi")
    transforms = types.ModuleType("openpi.transforms")
    transforms.DataTransformFn = _DataTransformFn
    models = types.ModuleType("openpi.models")
    model = types.ModuleType("openpi.models.model")
    model.ModelType = _ModelType
    openpi.transforms = transforms
    models.model = model
    modules = {
        "openpi": openpi,
        "openpi.transforms": transforms,
        "openpi.models": models,
        "openpi.models.model": model,
    }
    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        path = Path(__file__).with_name("so101_policy.py")
        spec = importlib.util.spec_from_file_location("testable_so101_policy", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


class SO101PolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = _load_policy_module()

    def test_maps_two_cameras_six_values_and_prompt(self) -> None:
        transform = self.policy.SO101Inputs(model_type=_ModelType.PI05)
        result = transform(
            {
                "observation/image": np.ones((3, 8, 10), dtype=np.float32),
                "observation/wrist_image": np.zeros((3, 8, 10), dtype=np.float32),
                "observation/state": np.arange(6, dtype=np.float32),
                "actions": np.zeros((10, 6), dtype=np.float32),
                "prompt": "pick up the red cube",
            }
        )
        self.assertEqual(result["image"]["base_0_rgb"].shape, (8, 10, 3))
        self.assertEqual(result["image"]["base_0_rgb"].dtype, np.uint8)
        self.assertEqual(result["state"].shape, (6,))
        self.assertEqual(result["actions"].shape, (10, 6))
        self.assertFalse(bool(result["image_mask"]["right_wrist_0_rgb"]))
        self.assertEqual(result["prompt"], "pick up the red cube")

    def test_output_keeps_first_six_dimensions(self) -> None:
        output = self.policy.SO101Outputs()({"actions": np.arange(320).reshape(10, 32)})
        self.assertEqual(output["actions"].shape, (10, 6))
        np.testing.assert_array_equal(output["actions"][0], np.arange(6))

    def test_rejects_wrong_state_dimension(self) -> None:
        transform = self.policy.SO101Inputs(model_type=_ModelType.PI05)
        with self.assertRaisesRegex(ValueError, "state"):
            transform(
                {
                    "observation/image": np.zeros((8, 10, 3), dtype=np.uint8),
                    "observation/wrist_image": np.zeros((8, 10, 3), dtype=np.uint8),
                    "observation/state": np.zeros(7),
                }
            )


if __name__ == "__main__":
    unittest.main()
