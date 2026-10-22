"""Success criteria, as small composable predicates over a :class:`TaskEnv`.

Two rules, both paid for in the WYSD projects:

1. **Criteria live here and nowhere else.**  Collection and evaluation import
   the same functions, so the training label and the evaluation label cannot
   drift apart.  A constant re-typed in a second file is how that happens.
2. **Every criterion needs a negative control.**  "The expert did not do the
   step" must return ``False``.  Otherwise a criterion that is satisfied by the
   starting layout ("free success") inflates every success rate in the table.
   The controls live in ``mj_env/tests/test_criteria.py``.

Criteria only use the public :class:`TaskEnv` surface, never raw ``MjData``
indices, so a criterion keeps working when a level's geometry changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from mj_env.task_env import TaskEnv

__all__ = [
    "LIFT_HEIGHT",
    "RELEASE_MARGIN",
    "in_container",
    "lifted",
    "on_place_target",
    "released",
    "rested",
    "undisturbed",
]

#: How far above its resting height a body must be to count as lifted.
LIFT_HEIGHT = 0.06
#: How far past the object's own width the jaw must open, in metres.
#:
#: **Relative, not absolute.**  A 50 mm absolute threshold looked right for the
#: 35 mm cube (its release opening is 57 mm) and silently failed every thin
#: object: the 16 mm bar is released at 38 mm, so "has let go" never became true
#: and the whole level scored zero with a perfect placement (measured: 7 of 10
#: episodes landed within 3 mm and still failed).
RELEASE_MARGIN = 0.012


def lifted(env: "TaskEnv", name: str = "object", height: float = LIFT_HEIGHT) -> bool:
    """The body is this far above where it rests on the table."""
    return bool(env.height_above_rest(name) >= height)


def released(env: "TaskEnv", name: str = "object", margin: float = RELEASE_MARGIN) -> bool:
    """The jaw is open at least ``margin`` wider than the object's grip width.

    Checked on the *jaw angle*, not on the object: after a release the object is
    on the table and the pads are apart, and a criterion that looked at contact
    would have to reason about which side let go first.
    """
    return bool(env.grasp_opening() >= env.grip_width(name) + margin)


def rested(env: "TaskEnv", name: str = "object", max_speed: float = 0.02) -> bool:
    """The body has actually stopped, not merely passed through the goal."""
    return bool(env.body_speed(name) <= max_speed)


def on_place_target(
    env: "TaskEnv", name: str = "object", tol: float = 0.025, rest_tol: float = 0.006
) -> bool:
    """The body is sitting on the marked goal area.

    Three conditions, and all three matter:

    * horizontally inside ``tol`` of the goal centre;
    * vertically back down at its resting height -- without this, carrying the
      object *over* the goal reads as success;
    * has stopped -- without this, a body sliding across the goal at speed
      reads as success on the frame it crosses.
    """
    position = env.prop_pos(name)
    goal = env.goal_xy()
    if float(np.linalg.norm(position[:2] - goal)) > tol:
        return False
    rest_z = env.rest_height(name)
    if abs(float(position[2]) - rest_z) > rest_tol:
        return False
    return rested(env, name)


def undisturbed(env: "TaskEnv", name: str, tol: float = 0.010) -> bool:
    """A distractor or support body has not been shoved out of place."""
    return bool(env.moved(name) <= tol)


def in_container(
    env: "TaskEnv", name: str = "object", inset: float = 0.008
) -> bool:
    """The body is inside a container's inner cavity and below its rim.

    ``inset`` keeps the check off the wall itself: a body resting against the
    inside face is in, and a body balanced on the rim is not.
    """
    return bool(env.inside_container(name, inset=inset))