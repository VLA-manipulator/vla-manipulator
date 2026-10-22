"""Task registry for the SO101 scene system.

    from mj_env.specs import LEVELS, order, describe_all, get

    spec = get("p1_place_disc")
    print(spec.difficulty_axis, spec.criterion_text)

Importing this package does **not** import MuJoCo: the environment and the
criteria are pulled in lazily, so the registry can be inspected (and the
ladder table printed) on a machine without a simulator, and the criteria tests
stay lightweight.
"""

from __future__ import annotations

from .registry import (
    TABLE_HEIGHT,
    TaskSpec,
    describe_all,
    get,
    levels,
    order,
)

# Importing a family module is what registers its levels.  Importing the
# registry alone yields an empty table, which would be a silent trap.
from . import family_o, family_p  # noqa: F401  (registration side effects)

__version__ = "0.1.0"

#: Populated at import time by the family modules.
LEVELS = levels()
LEVEL_ORDER = order()

__all__ = [
    "LEVELS",
    "LEVEL_ORDER",
    "TABLE_HEIGHT",
    "TaskSpec",
    "describe_all",
    "get",
    "levels",
    "order",
]


def __getattr__(name: str):
    """``TaskEnv`` is imported lazily so this package never needs MuJoCo."""
    if name == "TaskEnv":
        from mj_env.task_env import TaskEnv

        return TaskEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")