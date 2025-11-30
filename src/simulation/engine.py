from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np


@dataclass
class Body:
    mass: float
    position: np.ndarray
    velocity: np.ndarray

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float)
        self.velocity = np.asarray(self.velocity, dtype=float)


class QuadNode:
    def __init__(self, center: np.ndarray, half_size: float) -> None:
        self.center = np.asarray(center, dtype=float)
        self.half_size = float(half_size)
        self.mass = 0.0
        self.center_of_mass = np.zeros(2)
        self.body: Body | None = None
        self.children: List[QuadNode] | None = None

    def contains(self, position: np.ndarray) -> bool:
        offset = np.abs(position - self.center)
        return bool(np.all(offset <= self.half_size))

    def subdivide(self) -> None:
        quarter = self.half_size / 2.0
        offsets = np.array(
            [
                [-quarter, -quarter],
                [quarter, -quarter],
                [-quarter, quarter],
                [quarter, quarter],
            ]
        )
        self.children = [QuadNode(self.center + offset, quarter) for offset in offsets]

    def _insert_into_children(self, body: Body) -> None:
        assert self.children is not None
        for child in self.children:
            if child.contains(body.position):
                child.insert(body)
                return
        # Should not happen if root bounds were computed correctly.
        raise ValueError("Body position outside of quadtree bounds.")

    def insert(self, body: Body) -> None:
        # Update mass distribution for this node
        total_mass = self.mass + body.mass
        if total_mass > 0:
            self.center_of_mass = (self.center_of_mass * self.mass + body.mass * body.position) / total_mass
        self.mass = total_mass

        if self.children is None and self.body is None:
            self.body = body
            return

        if self.children is None:
            self.subdivide()
            assert self.body is not None
            self._insert_into_children(self.body)
            self.body = None

        self._insert_into_children(body)

    def compute_force(
        self,
        target: Body,
        theta: float,
        gravitational_constant: float,
        softening: float,
    ) -> np.ndarray:
        if self.mass == 0 or (self.body is target and self.children is None):
            return np.zeros(2)

        displacement = self.center_of_mass - target.position
        distance = np.linalg.norm(displacement) + 1e-16  # avoid division by zero
        width = self.half_size * 2

        if self.children is None or (width / distance) < theta:
            softened = distance**2 + softening**2
            factor = gravitational_constant * self.mass / (softened ** 1.5)
            return factor * displacement

        force = np.zeros(2)
        for child in self.children or []:
            force += child.compute_force(target, theta, gravitational_constant, softening)
        return force


def _compute_bounds(bodies: Iterable[Body]) -> tuple[np.ndarray, float]:
    positions = np.array([body.position for body in bodies])
    min_corner = positions.min(axis=0)
    max_corner = positions.max(axis=0)
    center = (min_corner + max_corner) / 2.0
    half_size = float(np.max(max_corner - min_corner) / 2.0 + 1e-9)
    return center, half_size


def build_tree(bodies: Iterable[Body]) -> QuadNode:
    bodies = list(bodies)
    if not bodies:
        raise ValueError("Cannot build quadtree with no bodies.")
    center, half_size = _compute_bounds(bodies)
    root = QuadNode(center, half_size)
    for body in bodies:
        root.insert(body)
    return root


def compute_accelerations(
    bodies: Iterable[Body],
    theta: float = 0.5,
    gravitational_constant: float = 1.0,
    softening: float = 0.0,
) -> List[np.ndarray]:
    bodies_list = list(bodies)
    tree = build_tree(bodies_list)
    accelerations = []
    for body in bodies_list:
        force = tree.compute_force(body, theta, gravitational_constant, softening)
        accelerations.append(force)
    return accelerations


def leapfrog_step(
    bodies: Iterable[Body],
    dt: float,
    theta: float = 0.5,
    gravitational_constant: float = 1.0,
    softening: float = 0.0,
) -> None:
    bodies_list = list(bodies)
    accelerations = compute_accelerations(bodies_list, theta, gravitational_constant, softening)
    half_step_velocities = []
    for body, acc in zip(bodies_list, accelerations):
        v_half = body.velocity + 0.5 * dt * acc
        half_step_velocities.append(v_half)
        body.position = body.position + dt * v_half

    new_accelerations = compute_accelerations(bodies_list, theta, gravitational_constant, softening)
    for body, v_half, acc_new in zip(bodies_list, half_step_velocities, new_accelerations):
        body.velocity = v_half + 0.5 * dt * acc_new


__all__ = [
    "Body",
    "QuadNode",
    "build_tree",
    "compute_accelerations",
    "leapfrog_step",
]
