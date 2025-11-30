import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from simulation.engine import Body, build_tree, compute_accelerations, leapfrog_step


def direct_sum_accelerations(bodies, gravitational_constant: float, softening: float):
    accelerations = []
    for i, target in enumerate(bodies):
        acc = np.zeros(2)
        for j, other in enumerate(bodies):
            if i == j:
                continue
            displacement = other.position - target.position
            distance_sq = float(np.dot(displacement, displacement) + softening**2)
            acc += gravitational_constant * other.mass * displacement / (distance_sq ** 1.5)
        accelerations.append(acc)
    return accelerations


def test_barnes_hut_matches_direct_sum():
    rng = np.random.default_rng(42)
    bodies = [
        Body(
            mass=float(rng.uniform(0.5, 2.0)),
            position=rng.uniform(-1, 1, size=2),
            velocity=np.zeros(2),
        )
        for _ in range(6)
    ]

    g_const = 1.3
    softening = 0.01
    theta = 0.3

    tree = build_tree(bodies)
    assert tree.mass == pytest.approx(sum(body.mass for body in bodies))

    bh_acc = compute_accelerations(bodies, theta=theta, gravitational_constant=g_const, softening=softening)
    direct_acc = direct_sum_accelerations(bodies, g_const, softening)

    for approx, exact in zip(bh_acc, direct_acc):
        np.testing.assert_allclose(approx, exact, rtol=1e-2, atol=1e-3)


def test_leapfrog_symmetry_for_equal_masses():
    bodies = [
        Body(mass=1.0, position=np.array([-1.0, 0.0]), velocity=np.zeros(2)),
        Body(mass=1.0, position=np.array([1.0, 0.0]), velocity=np.zeros(2)),
    ]

    leapfrog_step(bodies, dt=0.01, theta=0.2, gravitational_constant=1.0, softening=0.01)

    np.testing.assert_allclose(bodies[0].position, -bodies[1].position, atol=1e-8)
    np.testing.assert_allclose(bodies[0].velocity, -bodies[1].velocity, atol=1e-8)
