import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from simulation.initializers import (
    generate_spiral_galaxy,
    inject_black_hole,
    load_intersecting_spirals,
)


def test_generate_spiral_galaxy_has_black_hole_and_velocities():
    rng = np.random.default_rng(123)
    bodies = generate_spiral_galaxy(
        num_arms=2,
        stars_per_arm=10,
        black_hole_mass=5e4,
        velocity_dispersion=0.0,
        rng=rng,
    )

    assert len(bodies) == 2 * 10 + 1
    assert bodies[0].mass == pytest.approx(5e4)
    np.testing.assert_array_equal(bodies[0].position, np.zeros(2))

    velocities = np.array([np.linalg.norm(body.velocity) for body in bodies[1:]])
    assert velocities.min() > 0.0


def test_load_intersecting_spirals_offsets_and_bulk_velocity():
    rng = np.random.default_rng(7)
    bodies = load_intersecting_spirals(
        separation=20.0,
        approach_speed=2.0,
        impact_parameter=4.0,
        spiral_kwargs={"num_arms": 1, "stars_per_arm": 5, "rng": rng},
        rng=rng,
    )

    assert len(bodies) == 2 * (1 * 5 + 1)

    positions = np.array([body.position for body in bodies])
    assert positions[:, 0].min() < 0
    assert positions[:, 0].max() > 0

    velocities = np.array([body.velocity for body in bodies])
    # Should have motion in both axes for intersecting trajectories
    assert np.ptp(velocities[:, 0]) > 0
    assert np.ptp(velocities[:, 1]) > 0


def test_inject_black_hole_adds_body():
    rng = np.random.default_rng(1)
    bodies = generate_spiral_galaxy(stars_per_arm=1, rng=rng)
    before = len(bodies)
    new_bh = inject_black_hole(bodies, position=[5.0, -2.0], mass=1e3, velocity=[-1.0, 0.0])

    assert len(bodies) == before + 1
    assert new_bh in bodies
    np.testing.assert_allclose(new_bh.velocity, [-1.0, 0.0])
    np.testing.assert_allclose(new_bh.position, [5.0, -2.0])
