"""Lazy control exports: hardware imports do not load simulation or plotting."""
from importlib import import_module


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name in {"CartesianCommand", "CartesianPositionController"}:
        module = ".cartesian"
    elif name in {"RealCameraConfig", "RealCameraReader"}:
        module = ".real_camera"
    elif name in {"RealSenseConfig", "RealSenseRGBDReader"}:
        module = ".realsense_camera"
    else:
        module = ".motor_mapper"
    try:
        value = getattr(import_module(module, __name__), name)
    except ModuleNotFoundError as exc:
        if module != ".realsense_camera" or exc.name != "pyrealsense2":
            raise
        value = None
    globals()[name] = value
    return value


__all__ = [
    "CartesianCommand",
    "CartesianPositionController",
    "GripperEndpointCalibration",
    "JointEndpointCalibration",
    "JointMotorAnchors",
    "SO101MotorMapper",
    "RealCameraConfig",
    "RealCameraReader",
    "RealSenseConfig",
    "RealSenseRGBDReader",
]
