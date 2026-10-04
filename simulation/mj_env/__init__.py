"""Gymnasium MuJoCo environment for SO101 tabletop grasping.

Built to evaluate a LeRobot PI05 policy against the SO101 arm in simulation.
See ``README.md`` for the frame convention, the object catalogue and the unit
conventions on ``observation.state`` / ``action``.
"""

from gymnasium.envs.registration import register, registry

from .adapters import LeRobotSO101Adapter
from .env import HOME_QPOS, JOINT_NAMES, LIFT_THRESHOLD, SO101TabletopEnv
from .gripper import JawCalibration, calibrate
from .objects import CATALOGUE, NAMES as OBJECT_NAMES
from .scene import (
    GRASP_ZONE_HALF_ANGLE,
    GRASP_ZONE_RADIUS,
    OVERHEAD_CAMERA,
    SCENE_CAMERA,
    TABLE_HEIGHT,
    WRIST_CAMERA,
    build_model,
)

ENV_ID = "SO101Tabletop-v0"

if ENV_ID not in registry:
    register(id=ENV_ID, entry_point="mj_env.env:SO101TabletopEnv")

__all__ = [
    "CATALOGUE",
    "ENV_ID",
    "GRASP_ZONE_HALF_ANGLE",
    "GRASP_ZONE_RADIUS",
    "HOME_QPOS",
    "JOINT_NAMES",
    "JawCalibration",
    "LIFT_THRESHOLD",
    "LeRobotSO101Adapter",
    "OBJECT_NAMES",
    "OVERHEAD_CAMERA",
    "SCENE_CAMERA",
    "SO101TabletopEnv",
    "TABLE_HEIGHT",
    "WRIST_CAMERA",
    "build_model",
    "calibrate",
]
