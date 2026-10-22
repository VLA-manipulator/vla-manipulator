"""P family: pick-and-place, and the language that says *where* and *which*.

P1 is the foundation the whole scene system rests on.  Everything before it
only answers "can it lift the thing"; P1 is the first level that needs a
*place* step, so it is also the first level whose criterion any later level can
reuse unchanged.  The O family keeps exactly this criterion and adds one
geometric difficulty on top, which is what makes O1 a usable ruler.

Geometry is not invented here.  Every number below comes from
``docs/SO101_ENVELOPE.md``:

* grasp band ``r in [0.17, 0.30]`` and ``+-60 deg``: measured 100% success for
  the *complete* scripted grasp out to r = 0.30 (the shipped constant stops at
  0.28 and is conservative);
* the goal disc's 45 mm radius is the cube's 35 mm footprint plus a 10 mm ring;
* the 0.10 m minimum separation is the layout sampler's tested margin.
"""

from __future__ import annotations

import math

import numpy as np

from mj_env import layout
from mj_env.objects import Footprint

from .criteria import on_place_target
from .registry import (
    TABLE_HEIGHT,
    TaskSpec,
    load_instruction_targets,
    load_instructions,
    register,
)

__all__ = [
    "CUBE_FOOTPRINT",
    "CUBE_HALF",
    "DISC_FOOTPRINT",
    "MAX_SEPARATION",
    "MIN_SEPARATION",
    "P1_PLACE_DISC",
    "P2_LANGUAGE_REF",
    "layout_p1",
    "layout_p2",
    "sample_object_and_goal",
    "success_p1",
    "success_p2",
]

#: 35 mm cube.  ``measure_footprint`` gives half_length = 0 (a square's long and
#: short horizontal extents are equal), capsule radius = short_half * sqrt(2),
#: and the origin height above the tabletop.
CUBE_HALF = 0.0175
CUBE_FOOTPRINT = Footprint(half_length=0.0, radius=CUBE_HALF * math.sqrt(2.0), rest_clearance=CUBE_HALF)

#: The goal disc: 45 mm radius, 4 mm thick, zero collision (the cube must rest
#: on the *table*, or its seat height would depend on the disc).
DISC_RADIUS = 0.045
DISC_FOOTPRINT = Footprint(half_length=0.0, radius=DISC_RADIUS * math.sqrt(2.0), rest_clearance=0.002)

#: M0-measured: the complete scripted grasp is 100% out to r = 0.30.  Sampling
#: stops at 0.27 so the *whole* band stays inside the region where the layout
#: sampler itself never falls back, and so adding an obstacle later has slack.
BAND = (0.185, 0.270)
HALF_ANGLE = math.radians(55.0)
BASE_KEEPOUT = 0.12
TABLE_MARGIN = 0.012
MIN_SEPARATION = 0.10
#: Longest carry this level asks for.  Measured: the cube slowly walks out of
#: the pads during a carry, more so the longer it lasts, and every failed P1
#: episode was one of the long ones (0.29-0.35 m start-to-goal).  P1's
#: difficulty is *placing accurately*, not carrying the length of the table, so
#: the carry is capped and the failure mode is designed out.
MAX_SEPARATION = 0.20
#: Gap between the two P2 cubes.  They are 35 mm wide, so 60 mm is a clear gap
#: and keeps both pick corridors free; asking for more costs acceptance rate
#: without adding anything the level needs.
CUBE_MIN_SEPARATION = 0.060
TABLE_BOUNDS_X = (-0.02, 0.82)
TABLE_BOUNDS_Y = (-0.39, 0.39)


def sample_object_and_goal(
    rng: np.random.Generator, object_footprint: Footprint = CUBE_FOOTPRINT
) -> tuple[np.ndarray, np.ndarray]:
    """An object position and a goal position that satisfy every invariant.

    Reuses :func:`mj_env.layout.sample_layout` rather than hand-rolling
    rejection sampling: that function is the one the existing 320-episode
    dataset was built with, and it already enforces the base keep-out, the
    table bounds and pairwise clearance.

    The rejection loop adds the **carry window**, and every single-object level
    must come through here.  Without it a level can place the object and the
    goal at opposite ends of the band, and a carry longer than
    ``MAX_SEPARATION`` sheds the object whatever the grip -- measured on O2,
    which sampled its two positions directly and lost the bar in 2 of 3 seeds at
    *every* squeeze tried.
    """
    for _ in range(400):
        positions, yaws, fallback = layout.sample_layout(
            rng,
            (object_footprint, DISC_FOOTPRINT),
            radius_range=BAND,
            half_angle=HALF_ANGLE,
            base_keepout=BASE_KEEPOUT,
            margin=MIN_SEPARATION,
            bounds_x=TABLE_BOUNDS_X,
            bounds_y=TABLE_BOUNDS_Y,
        )
        if fallback:
            raise RuntimeError(
                "layout sampler fell back for P1; the band or the margin is too tight"
            )
        separation = float(np.linalg.norm(positions[0] - positions[1]))
        if MIN_SEPARATION <= separation <= MAX_SEPARATION:
            return positions[0], positions[1]
    raise RuntimeError(
        f"could not sample a P1 layout with separation in "
        f"[{MIN_SEPARATION}, {MAX_SEPARATION}] m in 400 tries"
    )


def layout_p1(rng: np.random.Generator) -> dict:
    object_xy, goal_xy = sample_object_and_goal(rng)
    return {
        # (x, y, z, yaw) -- z is the drop height, the env lets it settle.
        "props": {
            "object": (
                float(object_xy[0]),
                float(object_xy[1]),
                TABLE_HEIGHT + CUBE_HALF + 0.004,
                float(rng.uniform(-np.pi, np.pi)),
            )
        },
        # Statics are written straight into model.body_pos, so the disc sits on
        # the tabletop plane.
        "statics": {"place_target": (float(goal_xy[0]), float(goal_xy[1]), TABLE_HEIGHT)},
        "goal_xy": (float(goal_xy[0]), float(goal_xy[1])),
    }


def success_p1(env) -> bool:
    """Cube on the disc, jaw open, everything stopped."""
    from .criteria import released

    return on_place_target(env, "object") and released(env, "object")


# ---------------------------------------------------------------------------
# P2: language reference -- two same-sized cubes, the sentence picks one
# ---------------------------------------------------------------------------
def layout_p2(rng: np.random.Generator) -> dict:
    """Two cubes and a goal, sampled directly.

    Not via :func:`mj_env.layout.sample_layout`: that sampler enforces one
    ``margin`` between *every* pair, and 0.10 m between the two cubes as well as
    between each cube and the disc leaves almost no acceptance region in the
    grasp band -- measured at **1.1 s per layout**, with the two cubes always
    290 mm apart and the goal pinned at the far end of its window.  Sampling the
    goal first and then placing each cube against it is both fast and gives the
    separation the level actually needs: cubes 60 mm apart (they are 35 mm wide,
    so that is a clear gap), each 0.10-0.20 m from the goal.
    """
    for _ in range(2000):
        goal = sample_band_xy(rng)
        if float(np.linalg.norm(goal)) < BASE_KEEPOUT:
            continue
        cubes: list[np.ndarray] = []
        for _ in range(2):
            for _ in range(200):
                candidate = sample_band_xy(rng)
                if not inside_table(candidate, CUBE_FOOTPRINT):
                    continue
                if float(np.linalg.norm(candidate)) < BASE_KEEPOUT + CUBE_FOOTPRINT.max_radius:
                    continue
                if any(
                    float(np.linalg.norm(candidate - other)) < CUBE_MIN_SEPARATION
                    for other in cubes
                ):
                    continue
                distance = float(np.linalg.norm(candidate - goal))
                if not (MIN_SEPARATION <= distance <= MAX_SEPARATION):
                    continue
                cubes.append(candidate)
                break
            else:
                break
        else:
            return {
                "props": {
                    "object": (
                        float(cubes[0][0]),
                        float(cubes[0][1]),
                        TABLE_HEIGHT + CUBE_HALF + 0.004,
                        float(rng.uniform(-np.pi, np.pi)),
                    ),
                    "object2": (
                        float(cubes[1][0]),
                        float(cubes[1][1]),
                        TABLE_HEIGHT + CUBE_HALF + 0.004,
                        float(rng.uniform(-np.pi, np.pi)),
                    ),
                },
                "statics": {
                    "place_target": (float(goal[0]), float(goal[1]), TABLE_HEIGHT)
                },
                "goal_xy": (float(goal[0]), float(goal[1])),
            }
    raise RuntimeError("could not sample a P2 layout in 2000 tries")


def sample_band_xy(rng: np.random.Generator) -> np.ndarray:
    """Area-uniform polar sample inside the measured grasp band."""
    radius = float(np.sqrt(rng.uniform(BAND[0] ** 2, BAND[1] ** 2)))
    angle = float(rng.uniform(-HALF_ANGLE, HALF_ANGLE))
    return np.array([radius * np.cos(angle), radius * np.sin(angle)])


def inside_table(position: np.ndarray, footprint) -> bool:
    """Whether a footprint at ``position`` stays inside the tabletop bounds."""
    reach = footprint.max_radius
    return bool(
        TABLE_BOUNDS_X[0] + reach <= position[0] <= TABLE_BOUNDS_X[1] - reach
        and TABLE_BOUNDS_Y[0] + reach <= position[1] <= TABLE_BOUNDS_Y[1] - reach
    )


def success_p2(env) -> bool:
    """The *named* cube is on the disc, jaw open, and the other cube untouched.

    ``env.target_object`` is whichever body the instruction named, so this one
    criterion serves every sentence in the pool.  The "other cube untouched"
    clause is what makes a wrong-cube attempt fail: without it, a policy that
    ignores the language and always grabs the red cube would score 50% on a
    two-cube scene, which is exactly the number that would hide the problem.
    """
    from .criteria import released, undisturbed

    named = env.target_object
    others = [name for name in env.spec.prop_bodies if name != named]
    return (
        on_place_target(env, named)
        and released(env, named)
        and all(undisturbed(env, name, tol=0.010) for name in others)
    )


_p1_instructions, _p1_holdout = load_instructions("p1_place_disc")

P1_PLACE_DISC = register(
    TaskSpec(
        task_id="p1_place_disc",
        family="P",
        level=1,
        title="P1 放到圆盘",
        scene="p1_place_disc.xml",
        target="object",
        prop_bodies=("object",),
        statics=("place_target",),
        goal_body="place_target",
        difficulty_axis="多了「放到哪」：搬运到指定区并松手",
        criterion_text="方块水平落在圆盘上（<25mm）+ 已静置 + 夹爪张开，连续 0.4s",
        sample_layout=layout_p1,
        success=success_p1,
        hold_s=0.4,
        instructions=_p1_instructions,
        holdout=_p1_holdout,
        notes=(
            "O 族全部沿用这一条的判据，所以 O1 与 O2~O5 的成功率差可以直接归因。"
            "采样带取 r∈[0.185, 0.270]、方位 ±55°：M0 实测完整抓取在 r≤0.30 是 100%，"
            "这里收窄是为了给后续加障碍留余量。"
        ),
    )
)


_p2_instructions, _p2_holdout = load_instructions("p2_language_ref")
_p2_targets = load_instruction_targets("p2_language_ref")

P2_LANGUAGE_REF = register(
    TaskSpec(
        task_id="p2_language_ref",
        family="P",
        level=2,
        title="P2 语言指称",
        scene="p2_language_ref.xml",
        # The target is not fixed: the instruction picks it.  `target` is only
        # the default for bookkeeping.
        target="object",
        prop_bodies=("object", "object2"),
        statics=("place_target",),
        goal_body="place_target",
        difficulty_axis="语言必须被读：两个同尺寸方块，句子说是哪一个",
        criterion_text=(
            "**指定**的方块落在圆盘上（<25mm）+ 已静置 + 夹爪张开，"
            "且另一个方块位移 <10mm，连续 0.4s"
        ),
        sample_layout=layout_p2,
        success=success_p2,
        hold_s=0.4,
        instructions=_p2_instructions,
        holdout=_p2_holdout,
        instruction_targets=_p2_targets,
        notes=(
            "这一级是「语言分支到底有没有被用上」的最小实验：场景里两个方块同尺寸，"
            "只有颜色和句子能区分。判据里特意加了「另一个方块位移 <10mm」——否则一个"
            "永远抓红方块、完全不看指令的策略在两级场景上能拿 50%，正好把问题盖住。"
            "建模上两个方块同尺寸也是故意的：尺寸/形状差异会让策略用轮廓走捷径，"
            "绕开语言。"
        ),
    )
)


# ---------------------------------------------------------------------------
# P3: the object goes *inside* a container
# ---------------------------------------------------------------------------
#: Container measured from p3_in_container.xml: four 6 mm walls around a **120 mm**
#: cavity put the AABB at +/-66 mm, and the body sits on the tabletop so its
#: resting clearance is zero.  Every one of those numbers is checked against the
#: model's own geometry by the scene self-test.
#:
#: The cavity started at 110 mm -- 6 mm of nominal clearance on the jaw's wider
#: side, which looks generous -- and the self-test caught the fingers grazing a
#: wall by 0.54 mm during the descent.  The fixed finger hangs *below* the
#: aperture, so the clearance that matters is not the one an outline drawing
#: shows.
CONTAINER_HALF = 0.066
CONTAINER_FOOTPRINT = Footprint(
    half_length=0.0, radius=0.066 * math.sqrt(2.0), rest_clearance=0.0
)
#: Clear space between the cube and the container's outer wall, so the cube is
#: never dropped against the side of the container by a layout that looked legal.
MIN_CONTAINER_GAP = 0.020


def layout_p3(rng: np.random.Generator) -> dict:
    """Container and cube, with the cube outside the walls and inside the carry window.

    Sampled directly rather than through :func:`sample_object_and_goal` because
    the goal is a 122 mm-wide body, not a disc: the cube has to clear the
    container's *outer* wall, which the generic margin does not express.
    """
    for _ in range(2000):
        goal = sample_band_xy(rng)
        if float(np.linalg.norm(goal)) < BASE_KEEPOUT + CONTAINER_FOOTPRINT.max_radius:
            continue
        if not inside_table(goal, CONTAINER_FOOTPRINT):
            continue
        for _ in range(200):
            cube = sample_band_xy(rng)
            if not inside_table(cube, CUBE_FOOTPRINT):
                continue
            if float(np.linalg.norm(cube)) < BASE_KEEPOUT + CUBE_FOOTPRINT.max_radius:
                continue
            separation = float(np.linalg.norm(cube - goal))
            if separation < CONTAINER_HALF + CUBE_FOOTPRINT.max_radius + MIN_CONTAINER_GAP:
                continue
            if not (MIN_SEPARATION <= separation <= MAX_SEPARATION):
                continue
            return {
                "props": {
                    "object": (
                        float(cube[0]),
                        float(cube[1]),
                        TABLE_HEIGHT + CUBE_HALF + 0.004,
                        float(rng.uniform(-np.pi, np.pi)),
                    )
                },
                "statics": {"container": (float(goal[0]), float(goal[1]), TABLE_HEIGHT)},
                "goal_xy": (float(goal[0]), float(goal[1])),
            }
    raise RuntimeError("could not sample a P3 layout in 2000 tries")


def success_p3(env) -> bool:
    """The cube is inside the cavity, stopped, and the jaw has let go."""
    from .criteria import in_container, released, rested

    return (
        in_container(env, "object")
        and rested(env, "object")
        and released(env, "object")
    )


_p3_instructions, _p3_holdout = load_instructions("p3_in_container")

P3_IN_CONTAINER = register(
    TaskSpec(
        task_id="p3_in_container",
        family="P",
        level=3,
        title="P3 放进容器",
        scene="p3_in_container.xml",
        target="object",
        prop_bodies=("object",),
        statics=("container",),
        goal_body="container",
        difficulty_axis="目标是一个体积而不是一个点：手必须下进容器里",
        criterion_text=(
            "方块位于容器内腔（含 8mm 内缩余量）且低于上沿 + 已静置 + 夹爪张开，连续 0.4s"
        ),
        sample_layout=layout_p3,
        success=success_p3,
        hold_s=0.4,
        instructions=_p3_instructions,
        holdout=_p3_holdout,
        notes=(
            "内腔半宽 55mm（= 手部较宽一侧 48.9mm + 6mm 余量）、壁高 45mm（A=55mm 处全轨迹净空 154mm）。"
            "判据用 container_cavity 这个零碰撞 geom 显式声明的内腔，而不是从外包围盒猜壁厚——"
            "否则「放进去了」会变成「算出来了」。"
        ),
    )
)


# ---------------------------------------------------------------------------
# P4: stack the object on a support that is itself free to move
# ---------------------------------------------------------------------------
#: How far off the support's centre a stacked cube may land, and how far its
#: centre may be from the ideal stacked height.
#:
#: The support is 35 mm wide, so a 15 mm centre offset still leaves 20 mm of
#: overlap -- unambiguously "on top of".  A 12 mm limit had no physical
#: justification: it failed a seed that landed 13.1 mm out while every other
#: seed landed within 4 mm.  The criterion has to describe the task, not the
#: expert's precision on a good day.
STACK_HORIZONTAL_TOLERANCE = 0.015
STACK_HEIGHT_TOLERANCE = 0.004


def layout_p4(rng: np.random.Generator) -> dict:
    """Support cube (on the marking disc) and the cube that has to go on top."""
    for _ in range(2000):
        base_xy = sample_band_xy(rng)
        if float(np.linalg.norm(base_xy)) < BASE_KEEPOUT + CUBE_FOOTPRINT.max_radius:
            continue
        if not inside_table(base_xy, CUBE_FOOTPRINT):
            continue
        for _ in range(200):
            object_xy = sample_band_xy(rng)
            if not inside_table(object_xy, CUBE_FOOTPRINT):
                continue
            if float(np.linalg.norm(object_xy)) < BASE_KEEPOUT + CUBE_FOOTPRINT.max_radius:
                continue
            separation = float(np.linalg.norm(object_xy - base_xy))
            if not (MIN_SEPARATION <= separation <= MAX_SEPARATION):
                continue
            drop = TABLE_HEIGHT + CUBE_HALF + 0.004
            return {
                "props": {
                    "object": (
                        float(object_xy[0]),
                        float(object_xy[1]),
                        drop,
                        float(rng.uniform(-np.pi, np.pi)),
                    ),
                    "support": (
                        float(base_xy[0]),
                        float(base_xy[1]),
                        drop,
                        float(rng.uniform(-np.pi, np.pi)),
                    ),
                },
                "statics": {
                    "place_target": (float(base_xy[0]), float(base_xy[1]), TABLE_HEIGHT)
                },
                "goal_xy": (float(base_xy[0]), float(base_xy[1])),
            }
    raise RuntimeError("could not sample a P4 layout in 2000 tries")


def place_height_p4(env, name: str) -> float:
    """Top of the support plus the stacked body's half height, both measured.

    Measured, not computed from the nominal cube size: the support settles a
    fraction of a millimetre into the table, and the whole point of the level is
    that the placement height is a real surface.
    """
    return (
        float(env.prop_pos("support")[2])
        + env.rest_height("support")
        + env.rest_height(name)
    )


def success_p4(env) -> bool:
    """The target sits on the support, stopped, released, and the support stayed put."""
    from .criteria import released, rested, undisturbed

    base = env.prop_pos("support")
    target = env.prop_pos("object")
    ideal_z = float(base[2]) + env.rest_height("support") + env.rest_height("object")
    if abs(float(target[2]) - ideal_z) > STACK_HEIGHT_TOLERANCE:
        return False
    if float(np.linalg.norm(target[:2] - base[:2])) > STACK_HORIZONTAL_TOLERANCE:
        return False
    return (
        rested(env, "object")
        and released(env, "object")
        and undisturbed(env, "support", 0.010)
    )


_p4_instructions, _p4_holdout = load_instructions("p4_stack")

P4_STACK = register(
    TaskSpec(
        task_id="p4_stack",
        family="P",
        level=4,
        title="P4 堆叠",
        scene="p4_stack.xml",
        target="object",
        prop_bodies=("object", "support"),
        statics=("place_target",),
        goal_body="place_target",
        difficulty_axis="垂直精度 + 不许碰倒支撑：要放到另一个方块上面",
        criterion_text=(
            "方块中心落在支撑顶面 ±4mm、水平 ±15mm + 已静置 + 夹爪张开，"
            "且支撑位移 <10mm，连续 0.4s"
        ),
        sample_layout=layout_p4,
        success=success_p4,
        place_height=place_height_p4,
        hold_s=0.4,
        instructions=_p4_instructions,
        holdout=_p4_holdout,
        notes=(
            "支撑块是自由体而不是布景，所以「碰倒支撑」直接判失败——这才让这一级不只是"
            "「放高一点」。堆叠高度不写死在场景里，而是由 TaskSpec.place_height 量支撑块的"
            "实际位置得到，所以各级共用一个专家，支撑块在静置中微动也不会让高度失真。"
        ),
    )
)