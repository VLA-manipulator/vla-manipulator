"""O family: geometric obstacles, one added difficulty axis per level.

The design rule the whole family rests on: **O1 is the ruler**.  It keeps
exactly P1's scene and criterion, so the success-rate difference between O1 and
any later level is attributable to that level's single added obstacle.  A level
that also changes the criterion destroys that comparison.

Status (see ``docs/SO101_ENVELOPE.md`` for the measurements behind every size):

* **O1 open baseline** -- registered; physically identical to P1, so it is the
  ruler every other level is compared against.
* **O2 handle only** -- registered.  A dumbbell whose 44 mm end discs are wider
  than the jaw can open around, so only the 16 mm handle is graspable.  This
  replaced the original "side low bar" idea, which M0 disproved: on a
  vertical-approach arm anything that clears the swept volume passes *under* the
  wrist and blocks nothing.
* **O3 wrist-camera occlusion** -- pending.  The original "vertical board under
  the top-down camera" was measured infeasible (69 candidate boards, one without
  arm interference); occluding the wrist camera is the replacement, and it needs
  an occlusion measurement before it can be sized.
* **O4 narrow slot** -- registered and verified (12/12, zero wall contact).
  Slot 106 mm = 2 x (48.9 mm wider side + 4 mm), walls 45 mm tall, rotating with
  the object.
* **O5 covered stack** -- pending; needs a two-stage expert (clear the cover,
  then fetch the target).
"""

from __future__ import annotations

import math

import numpy as np

from mj_env import layout
from mj_env.objects import Footprint

from .criteria import on_place_target

from .family_p import (
    BAND,
    BASE_KEEPOUT,
    CUBE_FOOTPRINT,
    CUBE_HALF,
    DISC_FOOTPRINT,
    HALF_ANGLE,
    MAX_SEPARATION,
    MIN_SEPARATION,
    TABLE_BOUNDS_X,
    TABLE_BOUNDS_Y,
    inside_table,
    layout_p1,
    sample_band_xy,
    sample_object_and_goal,
    success_p1,
)
from .registry import TABLE_HEIGHT, TaskSpec, load_instructions, register

__all__ = [
    "O1_OPEN_BASELINE",
    "O2_THIN_BAR",
    "O4_NARROW_SLOT",
    "O5_COVERED_STACK",
]

#: O2: a rectangular bar.  ``grip_width`` is the *narrow* dimension (16 mm), not
#: the object's long extent -- the whole level is that the only graspable faces
#: are the two long ones.  Footprint from the geometry: half-extents
#: (0.035, 0.008, 0.011) give a 27 mm half-spine, an 11.3 mm capsule radius and
#: an 11 mm resting height.
BAR_GRIP_WIDTH = 0.016
BAR_FOOTPRINT = Footprint(
    half_length=0.035 - 0.008,
    radius=0.008 * math.sqrt(2.0),
    rest_clearance=0.011,
)


def layout_o2(rng: np.random.Generator) -> dict:
    """Bar and goal, through the shared sampler so the carry window applies."""
    object_xy, goal_xy = sample_object_and_goal(rng, BAR_FOOTPRINT)
    return {
        "props": {
            "object": (
                float(object_xy[0]),
                float(object_xy[1]),
                TABLE_HEIGHT + BAR_FOOTPRINT.rest_clearance + DROP_CLEARANCE,
                # A random heading is the difficulty: the jaw has to align its
                # hinge with the bar, so a policy that ignores the object's
                # orientation cannot close on it at all.
                float(rng.uniform(-math.pi, math.pi)),
            )
        },
        "statics": {"place_target": (float(goal_xy[0]), float(goal_xy[1]), TABLE_HEIGHT)},
        "goal_xy": (float(goal_xy[0]), float(goal_xy[1])),
    }


_o2_instructions, _o2_holdout = load_instructions("o2_thin_bar")

O2_THIN_BAR = register(
    TaskSpec(
        task_id="o2_thin_bar",
        family="O",
        level=2,
        title="O2 只能抓窄面",
        scene="o2_thin_bar.xml",
        target="object",
        prop_bodies=("object",),
        statics=("place_target",),
        goal_body="place_target",
        grip_widths={"object": BAR_GRIP_WIDTH},
        difficulty_axis="70×16mm 的条只有窄面能抓：颌必须对齐长轴，否则合不上",
        criterion_text="方块水平落在圆盘上（<25mm）+ 已静置 + 夹爪张开，连续 0.4s",
        sample_layout=layout_o2,
        success=success_p1,
        hold_s=0.4,
        instructions=_o2_instructions,
        holdout=_o2_holdout,
        notes=(
            "抓取面只有两条 70×22mm 的长面、间距 16mm；颌若转 90° 就会撞上 70mm 那对端面而合不上，"
            "所以「看不看物体的朝向」直接决定成败。判据与 O1/P1 完全一致，"
            "所以与 O1 的差就是「只有窄面能抓」的代价。"
        ),
    )
)

#: O4 geometry, all from the M0-measured hand outline.
#:
#: The jaw at the descent aperture spans 82.8 mm but **asymmetrically about the
#: aperture centre** -- 48.9 mm on the fixed-finger side, 33.9 mm on the other.
#: The first draft of this level said "90 mm slot, 3.6 mm per side"; that is
#: wrong and would have been a collision, because the wider side alone needs
#: 48.9 mm of half-width.  Half-width is therefore the wider side plus a margin.
WALL_OFFSET = 0.053
WALL_THICKNESS = 0.006
WALL_LENGTH = 0.100
WALL_HEIGHT = 0.045
DROP_CLEARANCE = 0.004


def layout_o4(rng: np.random.Generator) -> dict:
    """Cube between two guide walls that rotate with it.

    Rotating the walls with the object is what keeps the clearance meaningful:
    the jaw's closing direction follows the object's heading, so a fixed-orientation
    slot would be generous at some headings and impossible at others -- the
    difficulty would drift with the randomisation instead of being a property of
    the level.
    """
    object_xy, goal_xy = sample_object_and_goal(rng)
    yaw = float(rng.uniform(-math.pi, math.pi))
    centre_x, centre_y = float(object_xy[0]), float(object_xy[1])
    # The walls separate along the object's local y, which is the direction the
    # jaw closes across.
    offset_x, offset_y = -math.sin(yaw), math.cos(yaw)
    walls = {
        name: (
            centre_x + sign * WALL_OFFSET * offset_x,
            centre_y + sign * WALL_OFFSET * offset_y,
            WALL_HEIGHT / 2.0,
            yaw,
        )
        for name, sign in (("slot_wall_pos", 1.0), ("slot_wall_neg", -1.0))
    }
    return {
        "props": {
            "object": (
                centre_x,
                centre_y,
                TABLE_HEIGHT + CUBE_HALF + DROP_CLEARANCE,
                yaw,
            )
        },
        "statics": {**walls, "place_target": (float(goal_xy[0]), float(goal_xy[1]), TABLE_HEIGHT)},
        "goal_xy": (float(goal_xy[0]), float(goal_xy[1])),
    }


_o4_instructions, _o4_holdout = load_instructions("o4_narrow_slot")

O4_NARROW_SLOT = register(
    TaskSpec(
        task_id="o4_narrow_slot",
        family="O",
        level=4,
        title="O4 窄缝插取",
        scene="o4_narrow_slot.xml",
        target="object",
        prop_bodies=("object",),
        statics=("slot_wall_pos", "slot_wall_neg", "place_target"),
        goal_body="place_target",
        difficulty_axis="毫米级横向精度：两堵导墙只剩每侧 4mm 余量",
        criterion_text="方块水平落在圆盘上（<25mm）+ 已静置 + 夹爪张开，连续 0.4s",
        sample_layout=layout_o4,
        success=success_p1,
        hold_s=0.4,
        instructions=_o4_instructions,
        holdout=_o4_holdout,
        notes=(
            "缝宽 106mm = 2 x (48.9mm 手部外廓较宽一侧 + 4mm 余量)。"
            "导墙随目标一起移动并旋转，所以余量不随随机朝向漂移。"
            "判据与 O1/P1 完全一致，所以与 O1 的成功率差就是「窄缝的代价」。"
        ),
    )
)

_o1_instructions, _o1_holdout = load_instructions("o1_open_baseline")

O1_OPEN_BASELINE = register(
    TaskSpec(
        task_id="o1_open_baseline",
        family="O",
        level=1,
        title="O1 开阔基线",
        # Same scene, same layout rule and same criterion as P1 on purpose:
        # this level exists to be the ruler the obstacle levels are measured
        # against, so it must differ from them in *nothing* but the obstacle.
        scene="p1_place_disc.xml",
        target="object",
        prop_bodies=("object",),
        statics=("place_target",),
        goal_body="place_target",
        difficulty_axis="基线（对照组）：无额外障碍，视野与路径都通畅",
        criterion_text="方块水平落在圆盘上（<25mm）+ 已静置 + 夹爪张开，连续 0.4s",
        sample_layout=layout_p1,
        success=success_p1,
        hold_s=0.4,
        instructions=_o1_instructions,
        holdout=_o1_holdout,
        notes=(
            "这是整个 O 族的标尺：O2~O5 与它的成功率差就是「加一个障碍要付多少代价」。"
            "它故意与 P1 共用场景、布局规则与判据，只允许障碍物不同。"
        ),
    )
)


# ---------------------------------------------------------------------------
# O5: the target is covered -- clear it first, then fetch
# ---------------------------------------------------------------------------
#: How far the cover has to end up from the target, and how close to the table
#: it has to be, for "it has been cleared" to be true.
COVER_CLEAR_DISTANCE = 0.050
COVER_SETTLED_HEIGHT = 0.006


def layout_o5(rng: np.random.Generator) -> dict:
    """Target, cover resting on it, goal, and a parking spot for the cover.

    The parking spot has to be clear of both the target and the goal: park the
    cover on the carry path and the second stage collides with its own debris,
    which would look like a geometric failure in a level that is about order.
    """
    object_xy, goal_xy = sample_object_and_goal(rng)
    for _ in range(500):
        park = sample_band_xy(rng)
        if not inside_table(park, CUBE_FOOTPRINT):
            continue
        if float(np.linalg.norm(park)) < BASE_KEEPOUT + CUBE_FOOTPRINT.max_radius:
            continue
        if float(np.linalg.norm(park - object_xy)) < MIN_SEPARATION:
            continue
        if float(np.linalg.norm(park - goal_xy)) < MIN_SEPARATION:
            continue
        # The cover is carried from the target to the parking spot, so that hop
        # is subject to the same carry limit as every other carry.
        if float(np.linalg.norm(park - object_xy)) > MAX_SEPARATION:
            continue
        break
    else:
        raise RuntimeError("could not find a parking spot for O5 in 500 tries")

    drop = TABLE_HEIGHT + CUBE_HALF + 0.004
    stack = TABLE_HEIGHT + 3.0 * CUBE_HALF + 0.004
    yaw = float(rng.uniform(-np.pi, np.pi))
    return {
        "props": {
            "object": (float(object_xy[0]), float(object_xy[1]), drop, yaw),
            "cover": (float(object_xy[0]), float(object_xy[1]), stack, yaw),
        },
        "statics": {
            "place_target": (float(goal_xy[0]), float(goal_xy[1]), TABLE_HEIGHT),
            "park_spot": (float(park[0]), float(park[1]), TABLE_HEIGHT),
        },
        "goal_xy": (float(goal_xy[0]), float(goal_xy[1])),
    }


def success_o5(env) -> bool:
    """Target on the disc, released and stopped, **and** the cover cleared.

    The cover clause is what stops a policy from scoring by shoving the stack
    around: it has to be down on the tabletop (not balanced on the target) and
    50 mm clear of it.
    """
    from .criteria import released, rested

    cover = env.prop_pos("cover")
    target = env.prop_pos("object")
    cover_cleared = (
        float(np.linalg.norm(cover[:2] - target[:2])) > COVER_CLEAR_DISTANCE
        and env.height_above_rest("cover") < COVER_SETTLED_HEIGHT
    )
    return (
        on_place_target(env, "object")
        and rested(env, "object")
        and released(env, "object")
        and cover_cleared
    )


_o5_instructions, _o5_holdout = load_instructions("o5_covered_stack")

O5_COVERED_STACK = register(
    TaskSpec(
        task_id="o5_covered_stack",
        family="O",
        level=5,
        title="O5 覆盖堆叠",
        scene="o5_covered_stack.xml",
        target="object",
        prop_bodies=("object", "cover"),
        statics=("place_target", "park_spot"),
        independent_statics=("park_spot",),
        goal_body="place_target",
        multi_stage=True,
        difficulty_axis="顺序：目标被压住，不先搬开覆盖块就根本抓不到",
        criterion_text=(
            "方块落在圆盘上（<25mm）+ 已静置 + 夹爪张开，"
            "且覆盖块已落到台面、离目标 >50mm，连续 0.4s"
        ),
        sample_layout=layout_o5,
        success=success_o5,
        hold_s=0.4,
        instructions=_o5_instructions,
        holdout=_o5_holdout,
        notes=(
            "全阶梯唯一一级的难度是**顺序**而不是几何：障碍不是要挤过去，而是要先处理掉，"
            "整段长度约是其他级的两倍。停车位 park_spot 是一个**无几何、不可见**的标记体——"
            "给覆盖块一个可见目的地会把它变成第二个放置任务，改变这一级测的东西。"
        ),
    )
)


# ---------------------------------------------------------------------------
# O3: the wrist camera's view of the grasp point is partly blocked
# ---------------------------------------------------------------------------
#: Wall geometry, **measured** with scripts/occlusion_probe.py rather than
#: drawn: the probe renders the wrist camera at the grasp pose and reports the
#: fraction of the object's pixels the wall changes, the smallest distance to
#: any robot geom, and the smallest distance to the target itself.
#:
#: The measured optimum hides 13.5% of the target's pixels.  That is a *corner*,
#: not half the object, and the reason is a property of this arm: the pads are
#: arranged along the target's local +x, so anything long that sits within the
#: finger corridor collides, and the target's own corners sweep a 24.7 mm circle,
#: so the plate cannot come closer than about 76 mm.  From 76 mm away, at the
#: wrist camera's 60 mm standoff, a plate simply cannot cover much.  O3 is
#: therefore a *mild* perceptual handicap, and that is a measurement, not a
#: design preference.
#:
#: The azimuth is in the **object's own frame**, which is what makes this level
#: well defined: the wrist roll is a function of the object's heading, so the
#: camera's azimuth relative to the object is fixed, and a wall fixed in the
#: object's frame always lands in the same place in the camera's view.
OCCLUDER_AZIMUTH_DEG = 0.0
OCCLUDER_OFFSET = 0.076
OCCLUDER_HEIGHT = 0.046
OCCLUDER_LENGTH = 0.100
OCCLUDER_THICKNESS = 0.006
#: Fraction of the target's pixels the wall hides, from the probe.
OCCLUDER_OCCLUSION = 0.135


def layout_o3(rng: np.random.Generator) -> dict:
    """Cube, goal, and one wall rigid in the object's frame.

    The wall rotates with the object for the same reason O4's guide walls do:
    the invariant "the wall sits at this azimuth in the object's frame" is the
    level's definition, and a wall that stayed put while the object's heading
    randomised would occlude the camera at some headings and not others -- the
    difficulty would drift with the randomisation.
    """
    object_xy, goal_xy = sample_object_and_goal(rng)
    yaw = float(rng.uniform(-math.pi, math.pi))
    angle = yaw + math.radians(OCCLUDER_AZIMUTH_DEG)
    return {
        "props": {
            "object": (
                float(object_xy[0]),
                float(object_xy[1]),
                TABLE_HEIGHT + CUBE_HALF + DROP_CLEARANCE,
                yaw,
            )
        },
        "statics": {
            "place_target": (float(goal_xy[0]), float(goal_xy[1]), TABLE_HEIGHT),
            "occluder": (
                float(object_xy[0] + OCCLUDER_OFFSET * math.cos(angle)),
                float(object_xy[1] + OCCLUDER_OFFSET * math.sin(angle)),
                TABLE_HEIGHT,
                angle,
            ),
        },
        "goal_xy": (float(goal_xy[0]), float(goal_xy[1])),
    }


_o3_instructions, _o3_holdout = load_instructions("o3_wrist_occlusion")

O3_WRIST_OCCLUSION = register(
    TaskSpec(
        task_id="o3_wrist_occlusion",
        family="O",
        level=3,
        title="O3 腕部遮挡",
        scene="o3_wrist_occlusion.xml",
        target="object",
        prop_bodies=("object",),
        statics=("place_target", "occluder"),
        goal_body="place_target",
        difficulty_axis="看得见但看不清：矮墙挡住腕部相机对抓取点的部分视野",
        criterion_text="方块水平落在圆盘上（<25mm）+ 已静置 + 夹爪张开，连续 0.4s",
        sample_layout=layout_o3,
        # Deliberately the P1 criterion: this level's difficulty is *perceptual*,
        # so keeping the geometric criterion identical is what makes the success
        # rate difference attributable to the blocked view and nothing else.
        success=success_p1,
        hold_s=0.4,
        instructions=_o3_instructions,
        holdout=_o3_holdout,
        notes=(
            "原方案（俯视相机下挂立板）经 M0 判定不可行：俯视相机几乎垂直向下，"
            "立板要么近到进了钳口走廊、要么高到撞臂，两个约束永不相交。"
            "腕部相机离抓取点只有 60mm，目标旁的一面矮墙就能进画面。"
            "墙的偏置/方位/高度全部由 occlusion_probe 实测标定；方位定义在**目标自身坐标系**，"
            "因为腕部滚转角是目标朝向的函数，所以固定方位的墙总落在画面同一处。"
        ),
    )
)