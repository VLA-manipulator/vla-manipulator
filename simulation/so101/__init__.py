"""SO-101 kinematics and Cartesian-control interfaces."""

from importlib import import_module
from .joints import ARM_JOINT_NAMES


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = ".robot" if name in {"SO101", "LeRobotCartesianCommand"} else ".kinematics"
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value

__all__ = [
    "ARM_JOINT_NAMES",
    "IKResult",
    "JOINT_LIMITS",
    "LeRobotCartesianCommand",
    "SO101",
    "SO101Arm",
]
