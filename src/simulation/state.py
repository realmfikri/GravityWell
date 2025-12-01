from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .engine import Body, leapfrog_step
from .initializers import inject_black_hole


@dataclass
class SimulationState:
    """Mutable simulation state and runtime controls."""

    bodies: list[Body]
    dt: float = 0.01
    gravitational_constant: float = 1.0
    theta: float = 0.5
    softening: float = 0.0
    max_dt: float = 0.25
    min_dt: float = 1e-4
    max_gravitational_constant: float = 10.0
    min_gravitational_constant: float = 1e-3
    paused: bool = False
    black_hole_mass: float = 1e6

    def step(self) -> None:
        """Advance the simulation by one integrator step if unpaused."""

        if self.paused:
            return

        leapfrog_step(
            self.bodies,
            dt=self.dt,
            theta=self.theta,
            gravitational_constant=self.gravitational_constant,
            softening=self.softening,
        )

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def scale_dt(self, factor: float) -> None:
        """Scale dt with clamping to avoid numerical blow-ups."""

        scaled = float(self.dt * factor)
        self.dt = float(np.clip(scaled, self.min_dt, self.max_dt))

    def set_dt(self, value: float) -> None:
        self.dt = float(np.clip(value, self.min_dt, self.max_dt))

    def scale_gravitational_constant(self, factor: float) -> None:
        scaled = float(self.gravitational_constant * factor)
        self.gravitational_constant = float(
            np.clip(scaled, self.min_gravitational_constant, self.max_gravitational_constant)
        )

    def set_gravitational_constant(self, value: float) -> None:
        self.gravitational_constant = float(
            np.clip(value, self.min_gravitational_constant, self.max_gravitational_constant)
        )

    def drop_black_hole(self, position: Sequence[float]) -> Body:
        """Inject a stationary supermassive black hole at the requested position."""

        return inject_black_hole(self.bodies, position=np.asarray(position, dtype=float), mass=self.black_hole_mass)


__all__ = ["SimulationState"]
