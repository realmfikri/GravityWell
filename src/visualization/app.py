from __future__ import annotations

from typing import Sequence

import pyglet

from src.simulation.state import SimulationState
from src.visualization.controls import ControlButton, ControlPanel
from src.visualization.renderer import Renderer


class SimulationApp:
    """Coordinate the simulation state, renderer, and interactive controls."""

    def __init__(self, state: SimulationState, renderer: Renderer) -> None:
        self.state = state
        self.renderer = renderer
        self.control_panel = ControlPanel()

        self._configure_controls()
        self.renderer.add_overlay_drawer(self.control_panel.draw)
        self.renderer.window.push_handlers(self)

        # Initialize buffers and bounds so early clicks map correctly
        self.renderer.update_scene(self.state.bodies)

    def _configure_controls(self) -> None:
        self.control_panel.layout(
            [
                [
                    ControlButton("Pause/Resume", 0, 0, 110, 26, self.state.toggle_pause),
                    ControlButton("dt -", 0, 0, 70, 26, lambda: self.state.scale_dt(0.8)),
                    ControlButton("dt +", 0, 0, 70, 26, lambda: self.state.scale_dt(1.25)),
                ],
                [
                    ControlButton("G -", 0, 0, 70, 26, lambda: self.state.scale_gravitational_constant(0.9)),
                    ControlButton("G +", 0, 0, 70, 26, lambda: self.state.scale_gravitational_constant(1.1)),
                    ControlButton("Drop BH", 0, 0, 110, 26, self._drop_black_hole_center),
                ],
            ]
        )

    def _drop_black_hole_center(self) -> None:
        center = self.renderer.screen_to_world(self.renderer.window_size / 2, self.renderer.window_size / 2)
        self.state.drop_black_hole(center)

    def _overlay_lines(self) -> Sequence[str]:
        return [
            f"dt: {self.state.dt:.4f} (min {self.state.min_dt}, max {self.state.max_dt})",
            f"G: {self.state.gravitational_constant:.3f} (min {self.state.min_gravitational_constant}, max {self.state.max_gravitational_constant})",
            f"Black hole mass: {self.state.black_hole_mass:.2e}",
            "Space: pause/resume, Arrow Up/Down: dt +/-",
            "Left/Right: G +/-; Click canvas to drop black hole",
        ]

    def _tick(self) -> None:
        self.state.step()
        self.renderer.update_scene(self.state.bodies)
        self.renderer.set_overlay(self._overlay_lines())

    def start(self) -> None:
        self.renderer.run(self._tick)

    # Event handlers -------------------------------------------------
    def on_key_press(self, symbol: int, modifiers: int) -> None:
        if symbol == pyglet.window.key.SPACE:
            self.state.toggle_pause()
        elif symbol == pyglet.window.key.UP:
            self.state.scale_dt(1.1)
        elif symbol == pyglet.window.key.DOWN:
            self.state.scale_dt(0.9)
        elif symbol == pyglet.window.key.RIGHT:
            self.state.scale_gravitational_constant(1.05)
        elif symbol == pyglet.window.key.LEFT:
            self.state.scale_gravitational_constant(0.95)

    def on_mouse_press(self, x: float, y: float, button: int, modifiers: int) -> None:
        if self.control_panel.handle_mouse_press(x, y, button):
            return
        if button == pyglet.window.mouse.LEFT:
            world = self.renderer.screen_to_world(x, y)
            self.state.drop_black_hole(world)


__all__ = ["SimulationApp"]
