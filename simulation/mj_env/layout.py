"""Reset-time object placement.

Objects are sampled into the polar band in front of the arm that was measured
to admit a zero-orientation-error top-down grasp.  Overlap is tested between
the 2-D capsule footprints from :mod:`mj_env.objects` rather than between
enclosing circles, because a circle around a 168 mm rod would claim most of
the table and make the layout unpackable.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .objects import Footprint


def spine(footprint: Footprint, position: Sequence[float], yaw: float) -> tuple[np.ndarray, np.ndarray]:
    """Endpoints of a footprint's capsule spine, in world xy."""
    offset = footprint.half_length * np.array([np.cos(yaw), np.sin(yaw)])
    centre = np.asarray(position, dtype=float)[:2]
    return centre - offset, centre + offset


def segment_distance_2d(
    p0: np.ndarray, p1: np.ndarray, q0: np.ndarray, q1: np.ndarray
) -> float:
    """Shortest distance between two 2-D segments, degenerate ends allowed."""
    u, v, w = p1 - p0, q1 - q0, p0 - q0
    denominator = u[0] * v[1] - u[1] * v[0]
    if abs(denominator) > 1e-12:
        # Non-parallel: they cross iff both intersection parameters are in [0, 1].
        s = (v[0] * w[1] - v[1] * w[0]) / denominator
        t = (u[0] * w[1] - u[1] * w[0]) / denominator
        if 0.0 <= s <= 1.0 and 0.0 <= t <= 1.0:
            return 0.0
    return min(
        _point_segment_distance(p0, q0, q1),
        _point_segment_distance(p1, q0, q1),
        _point_segment_distance(q0, p0, p1),
        _point_segment_distance(q1, p0, p1),
    )


def _point_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    length_squared = float(ab @ ab)
    if length_squared < 1e-18:
        return float(np.linalg.norm(point - a))
    t = float(np.clip((point - a) @ ab / length_squared, 0.0, 1.0))
    return float(np.linalg.norm(point - (a + t * ab)))


def clearance(
    footprint_a: Footprint,
    position_a: Sequence[float],
    yaw_a: float,
    footprint_b: Footprint,
    position_b: Sequence[float],
    yaw_b: float,
) -> float:
    """Gap between two placed footprints; negative means they overlap."""
    a0, a1 = spine(footprint_a, position_a, yaw_a)
    b0, b1 = spine(footprint_b, position_b, yaw_b)
    return segment_distance_2d(a0, a1, b0, b1) - footprint_a.radius - footprint_b.radius


def point_clearance(
    footprint: Footprint, position: Sequence[float], yaw: float, point: Sequence[float]
) -> float:
    """Gap between a placed footprint and a world-xy point."""
    a0, a1 = spine(footprint, position, yaw)
    target = np.asarray(point, dtype=float)[:2]
    return _point_segment_distance(target, a0, a1) - footprint.radius


def sample_layout(
    rng: np.random.Generator,
    footprints: Sequence[Footprint],
    *,
    radius_range: tuple[float, float],
    half_angle: float,
    base_keepout: float,
    margin: float,
    bounds_x: tuple[float, float],
    bounds_y: tuple[float, float],
    attempts_per_object: int = 400,
    layout_attempts: int = 40,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Sample non-overlapping xy positions and yaws for every footprint.

    Returns positions ``(n, 2)``, yaws ``(n,)`` and a flag that is ``True``
    when rejection sampling gave up and the deterministic arc fallback was
    used.  A reset must never fail, so the fallback is always returned rather
    than raising.
    """
    count = len(footprints)
    # Placing the bulkiest objects first roughly doubles the acceptance rate.
    order = sorted(range(count), key=lambda i: -footprints[i].max_radius)
    r_min, r_max = radius_range

    for _ in range(layout_attempts):
        positions = np.zeros((count, 2))
        yaws = np.zeros(count)
        placed: list[int] = []
        for index in order:
            footprint = footprints[index]
            for _ in range(attempts_per_object):
                # sqrt keeps samples area-uniform across the annulus.
                radius = float(np.sqrt(rng.uniform(r_min**2, r_max**2)))
                angle = float(rng.uniform(-half_angle, half_angle))
                yaw = float(rng.uniform(-np.pi, np.pi))
                position = np.array([radius * np.cos(angle), radius * np.sin(angle)])
                if _accepts(
                    footprint, position, yaw, footprints, positions, yaws, placed,
                    base_keepout=base_keepout, margin=margin,
                    bounds_x=bounds_x, bounds_y=bounds_y,
                ):
                    positions[index], yaws[index] = position, yaw
                    placed.append(index)
                    break
            else:
                break
        if len(placed) == count:
            return positions, yaws, False

    return (*_arc_fallback(footprints, radius_range, half_angle), True)


def _accepts(
    footprint: Footprint,
    position: np.ndarray,
    yaw: float,
    footprints: Sequence[Footprint],
    positions: np.ndarray,
    yaws: np.ndarray,
    placed: Sequence[int],
    *,
    base_keepout: float,
    margin: float,
    bounds_x: tuple[float, float],
    bounds_y: tuple[float, float],
) -> bool:
    if point_clearance(footprint, position, yaw, (0.0, 0.0)) < base_keepout:
        return False

    a0, a1 = spine(footprint, position, yaw)
    low = np.minimum(a0, a1) - footprint.radius
    high = np.maximum(a0, a1) + footprint.radius
    if low[0] < bounds_x[0] or high[0] > bounds_x[1]:
        return False
    if low[1] < bounds_y[0] or high[1] > bounds_y[1]:
        return False

    return all(
        clearance(footprint, position, yaw, footprints[i], positions[i], yaws[i]) >= margin
        for i in placed
    )


def _arc_fallback(
    footprints: Sequence[Footprint],
    radius_range: tuple[float, float],
    half_angle: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Evenly spaced arc, staggered in radius, used only if sampling fails."""
    count = len(footprints)
    r_min, r_max = radius_range
    angles = np.linspace(-half_angle, half_angle, count)
    radii = np.where(np.arange(count) % 2 == 0, r_min, r_max)
    positions = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
    # Tangential alignment keeps long objects from pointing at their neighbours.
    yaws = angles + np.pi / 2.0
    return positions, yaws
