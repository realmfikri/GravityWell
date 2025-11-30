"""Galaxy initialization utilities."""
from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np

from .engine import Body


def _tangential_vector(position: np.ndarray) -> np.ndarray:
    perp = np.array([-position[1], position[0]], dtype=float)
    norm = np.linalg.norm(perp)
    if norm == 0:
        return perp
    return perp / norm


def _logarithmic_spiral_radius(core_radius: float, arm_length: float, radial_factor: float) -> float:
    return core_radius + arm_length * radial_factor


def generate_spiral_galaxy(
    *,
    num_arms: int = 2,
    stars_per_arm: int = 250,
    arm_spread: float = 0.3,
    core_radius: float = 0.5,
    arm_length: float = 8.0,
    star_mass: float = 1.0,
    black_hole_mass: float = 1e5,
    gravitational_constant: float = 1.0,
    velocity_dispersion: float = 0.02,
    rng: np.random.Generator | None = None,
) -> List[Body]:
    """
    Build a spiral galaxy with a central black hole.

    The spiral uses a logarithmic pattern with gaussian angular noise per-arm. Stars
    receive tangential velocities scaled to the enclosed mass so the disk roughly
    traces a rotationally supported profile.
    """

    if num_arms <= 0:
        raise ValueError("num_arms must be positive")
    if stars_per_arm <= 0:
        raise ValueError("stars_per_arm must be positive")

    rng = rng or np.random.default_rng()
    total_stars = num_arms * stars_per_arm
    total_disk_mass = total_stars * star_mass
    galaxy_radius = core_radius + arm_length

    bodies: list[Body] = [
        Body(mass=black_hole_mass, position=np.zeros(2), velocity=np.zeros(2))
    ]

    for arm_idx in range(num_arms):
        base_angle = arm_idx * (2 * np.pi / num_arms)
        for i in range(stars_per_arm):
            radial_factor = rng.uniform(0.05, 1.0)
            radius = _logarithmic_spiral_radius(core_radius, arm_length, radial_factor)
            winding = radial_factor * 4 * np.pi
            angle = base_angle + winding + rng.normal(scale=arm_spread)
            radius += rng.normal(scale=core_radius * 0.05)

            position = radius * np.array([np.cos(angle), np.sin(angle)])
            enclosed_fraction = min(1.0, radius / galaxy_radius)
            enclosed_mass = black_hole_mass + total_disk_mass * enclosed_fraction

            tangential_direction = _tangential_vector(position)
            circular_speed = np.sqrt(
                gravitational_constant * enclosed_mass / (np.linalg.norm(position) + 1e-8)
            )
            velocity = tangential_direction * circular_speed
            velocity += rng.normal(scale=velocity_dispersion, size=2)

            bodies.append(Body(mass=star_mass, position=position, velocity=velocity))

    return bodies


def inject_black_hole(
    bodies: Iterable[Body],
    *,
    position: Sequence[float],
    mass: float,
    velocity: Sequence[float] | None = None,
) -> Body:
    """Add a user defined black hole to the running simulation."""

    velocity = np.zeros(2) if velocity is None else np.asarray(velocity, dtype=float)
    black_hole = Body(mass=float(mass), position=np.asarray(position, dtype=float), velocity=velocity)
    if isinstance(bodies, list):
        bodies.append(black_hole)
    return black_hole


def load_intersecting_spirals(
    *,
    separation: float = 30.0,
    approach_speed: float = 1.0,
    impact_parameter: float = 5.0,
    spiral_kwargs: dict | None = None,
    rng: np.random.Generator | None = None,
) -> List[Body]:
    """Create two spiral galaxies on intersecting trajectories."""

    rng = rng or np.random.default_rng()
    spiral_kwargs = dict(spiral_kwargs or {})

    galaxy_kwargs = {**spiral_kwargs}
    galaxy_kwargs.setdefault("rng", rng)

    galaxy_a = generate_spiral_galaxy(**galaxy_kwargs)
    galaxy_b = generate_spiral_galaxy(**galaxy_kwargs)

    def _apply_bulk(bodies: list[Body], offset: np.ndarray, velocity: np.ndarray) -> None:
        for body in bodies:
            body.position = body.position + offset
            body.velocity = body.velocity + velocity

    half_sep = np.array([separation / 2.0, 0.0])
    offset_a = -half_sep + np.array([0.0, -impact_parameter / 2.0])
    offset_b = half_sep + np.array([0.0, impact_parameter / 2.0])

    bulk_a = np.array([approach_speed / 2.0, approach_speed])
    bulk_b = np.array([-approach_speed / 2.0, -approach_speed])

    _apply_bulk(galaxy_a, offset_a, bulk_a)
    _apply_bulk(galaxy_b, offset_b, bulk_b)

    return galaxy_a + galaxy_b


__all__ = [
    "generate_spiral_galaxy",
    "inject_black_hole",
    "load_intersecting_spirals",
]
