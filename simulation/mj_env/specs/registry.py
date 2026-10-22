"""Task registry: one declarative table describing every graded scene level.

Design follows ``WYSD/underwater_sampling/tasks.py`` and
``WYSD/obstacle_grasp/tasks.py`` -- a single source of truth where a level is
*data*: its scene file, its layout sampler, its success criterion, its
instruction pool, and the one difficulty axis it adds over the baseline.

Why data and not code: the scene system only stays comparable across levels if
the criterion and the layout rule travel with the level.  A level that needs
new `if` branches in `env.py` cannot be diffed against the level above it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from mj_env.task_env import TaskEnv

INSTRUCTIONS_PATH = Path(__file__).resolve().parent / "instructions.json"

#: Table top surface; the whole project's z datum.
TABLE_HEIGHT = 0.0

Layout = dict[str, Any]
LayoutSampler = Callable[[np.random.Generator], Layout]
Criterion = Callable[["TaskEnv"], bool]


@dataclass(frozen=True)
class TaskSpec:
    """One graded level of one task family.

    ``sample_layout`` is the *only* source of randomness for the scene: given a
    seeded generator it returns where every free prop goes, where every static
    prop goes, and where the goal is.  Anything that moves must be described
    there, or two episodes with the same seed will not be comparable -- which
    is exactly what the mandatory clean/underwater pairing needs.
    """

    task_id: str
    family: str
    level: int
    title: str
    #: Scene file name under ``mj_env/assets/scenes/``.
    scene: str
    #: Body the episode is about (must own a ``<name>_free`` joint).
    target: str
    #: Free bodies the layout moves (usually just ``(target,)``).
    prop_bodies: tuple[str, ...]
    #: Static bodies the layout moves (obstacles, fence walls, the goal disc).
    statics: tuple[str, ...]
    #: One sentence: what this level adds over its family's baseline.
    difficulty_axis: str
    #: Human-readable criterion; must match what ``success`` computes.
    criterion_text: str
    sample_layout: LayoutSampler
    success: Criterion
    #: Where the object has to end up, as a z in metres.  ``None`` means "on the
    #: tabletop at its resting height", which is every level except stacking.
    #: The expert uses this as its place target, so a stack level does not need
    #: a second expert.
    place_height: Callable[["TaskEnv", str], float] | None = None
    #: Whether this level needs more than one pick-and-place (O5: clear the
    #: cover first).  The harness picks the expert from this flag rather than
    #: from a task_id list, so adding a multi-stage level needs no harness edit.
    multi_stage: bool = False
    #: Statics positioned relative to the *level* rather than to the target, so
    #: the "keeps a constant offset in the target's frame" invariant does not
    #: apply to them (O5's parking spot has to dodge a random goal).  Declared
    #: explicitly so the self-test reports them as by-design rather than
    #: silently skipping them.
    independent_statics: tuple[str, ...] = ()
    #: The goal carrier, if the level has one.
    goal_body: str | None = None
    #: Criterion must hold this long, in seconds, before it counts.
    hold_s: float = 0.4
    #: Explicit jaw widths for bodies whose graspable width is not their
    #: narrow horizontal extent (a dumbbell's handle, say).
    grip_widths: dict[str, float] = field(default_factory=dict)
    #: Per-body jaw interference while holding, in metres per side.  A grasp
    #: parameter of the *object*, not of the level: a flat-faced cube wants a
    #: light squeeze (measured: 2 mm holds, 8 mm creeps out), while a 16 mm
    #: cylindrical handle is a line contact on a long body and needs a firmer
    #: one.  Defaults to ``PlaceTuning.carry_squeeze``.
    grip_squeezes: dict[str, float] = field(default_factory=dict)
    instructions: tuple[str, ...] = ()
    holdout: tuple[str, ...] = ()
    #: Instruction text -> body name, for levels where the language *chooses*
    #: which object the episode is about (P2).  Empty means the target is fixed
    #: by :attr:`target`, which is what every other level wants.
    instruction_targets: dict[str, str] = field(default_factory=dict)
    notes: str = ""

    def target_for_instruction(self, text: str) -> str:
        """Which body an instruction refers to; ``target`` when it is fixed."""
        if not self.instruction_targets:
            return self.target
        try:
            return self.instruction_targets[text]
        except KeyError as exc:
            raise KeyError(
                f"{self.task_id}: instruction {text!r} has no target mapping; every "
                "instruction must name exactly one object or the language branch "
                "cannot be evaluated"
            ) from exc

    def instruction_pool(self, kind: str = "train") -> tuple[str, ...]:
        """Instruction templates for one pool: ``train``, ``holdout`` or ``all``."""
        if kind == "train":
            return self.instructions
        if kind == "holdout":
            return self.holdout
        if kind == "all":
            return (*self.instructions, *self.holdout)
        raise ValueError(f"unknown instruction pool {kind!r}")

    def hash(self) -> str:
        """Stable fingerprint of everything that changes what an episode means."""
        payload = json.dumps(
            {
                "task_id": self.task_id,
                "scene": self.scene,
                "target": self.target,
                "prop_bodies": list(self.prop_bodies),
                "statics": list(self.statics),
                "hold_s": self.hold_s,
                "criterion": self.criterion_text,
                "instructions": list(self.instructions),
                "holdout": list(self.holdout),
                "instruction_targets": dict(sorted(self.instruction_targets.items())),
                "grip_widths": {k: round(v, 6) for k, v in self.grip_widths.items()},
                "grip_squeezes": {k: round(v, 6) for k, v in self.grip_squeezes.items()},
                "place_height": self.place_height is not None,
                "multi_stage": self.multi_stage,
                "difficulty_axis": self.difficulty_axis,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


_REGISTRY: dict[str, TaskSpec] = {}


def register(spec: TaskSpec) -> TaskSpec:
    if spec.task_id in _REGISTRY:
        raise ValueError(f"duplicate task_id {spec.task_id!r}")
    _REGISTRY[spec.task_id] = spec
    return spec


def load_instructions(task_id: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """``(train, holdout)`` instruction templates for a task, from JSON."""
    if not INSTRUCTIONS_PATH.is_file():
        return (), ()
    payload = json.loads(INSTRUCTIONS_PATH.read_text(encoding="utf-8"))
    entry = payload.get(task_id, {})
    return tuple(entry.get("train", ())), tuple(entry.get("holdout", ()))


def load_instruction_targets(task_id: str) -> dict[str, str]:
    """Instruction -> body name, for levels where the language picks the object.

    Lives in the same JSON as the templates so the text and its referent cannot
    drift apart; a level that names two objects but maps only one would make the
    language ablation silently meaningless.
    """
    if not INSTRUCTIONS_PATH.is_file():
        return {}
    payload = json.loads(INSTRUCTIONS_PATH.read_text(encoding="utf-8"))
    raw = payload.get(task_id, {}).get("targets", {})
    targets = {str(text): str(body) for text, body in raw.items()}
    if not targets:
        return {}
    train, holdout = load_instructions(task_id)
    missing = [text for text in (*train, *holdout) if text not in targets]
    if missing:
        raise ValueError(
            f"{task_id}: instruction(s) without a target mapping: {missing}"
        )
    return targets


def levels() -> dict[str, TaskSpec]:
    return dict(_REGISTRY)


def order() -> list[str]:
    return [
        key
        for key, _ in sorted(
            _REGISTRY.items(), key=lambda item: (item[1].family, item[1].level)
        )
    ]


def get(task_id: str) -> TaskSpec:
    if task_id not in _REGISTRY:
        raise KeyError(f"unknown task_id {task_id!r}; known: {', '.join(order())}")
    return _REGISTRY[task_id]


def describe_all() -> str:
    """The ladder table, ready to paste into a document."""
    lines = [
        f"{'等级':<20} {'族':<4} {'新增难度轴':<52} {'保持':>5}",
        "-" * 86,
    ]
    for task_id in order():
        spec = _REGISTRY[task_id]
        lines.append(
            f"{task_id:<20} {spec.family:<4} {spec.difficulty_axis[:50]:<52}"
            f" {spec.hold_s:>4.1f}s"
        )
    return "\n".join(lines)


__all__ = [
    "Criterion",
    "Layout",
    "LayoutSampler",
    "TABLE_HEIGHT",
    "TaskSpec",
    "describe_all",
    "get",
    "levels",
    "load_instructions",
    "order",
    "register",
]