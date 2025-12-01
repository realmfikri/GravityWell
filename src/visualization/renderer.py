from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pyglet
from pyglet import gl

from src.simulation.engine import Body


@dataclass
class EnergyHistory:
    """Tracks recent kinetic and potential energy values for plotting."""

    max_samples: int = 300
    kinetic: deque[float] = field(default_factory=deque)
    potential: deque[float] = field(default_factory=deque)

    def append(self, kinetic: float, potential: float) -> None:
        self.kinetic.append(kinetic)
        self.potential.append(potential)
        while len(self.kinetic) > self.max_samples:
            self.kinetic.popleft()
        while len(self.potential) > self.max_samples:
            self.potential.popleft()


class Renderer:
    """High-performance renderer for particle simulations using pyglet.

    The renderer draws particles on a black background, maintains short
    motion trails, and overlays kinetic/potential energy history graphs.

    Parameters
    ----------
    window_size:
        Width and height of the square window in pixels.
    trail_length:
        Number of historical positions to store per particle. Lower values
        are faster while still producing visible motion streaks.
    energy_samples:
        Number of frames to keep in the energy plot.
    point_size:
        Size of each particle when rendered as a GL point.
    """

    def __init__(
        self,
        window_size: int = 900,
        trail_length: int = 12,
        energy_samples: int = 300,
        point_size: int = 2,
    ) -> None:
        self.window = pyglet.window.Window(window_size, window_size, "GravityWell")
        self.window.push_handlers(self)

        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glPointSize(point_size)

        self.batch = pyglet.graphics.Batch()
        self.particle_vertices: pyglet.graphics.vertexdomain.VertexList | None = None
        self.trail_vertices: pyglet.graphics.vertexdomain.VertexList | None = None

        self.trail_length = trail_length
        self.position_history: list[deque[tuple[float, float]]] = []

        self.energy_history = EnergyHistory(max_samples=energy_samples)

        self.window_size = window_size
        self.bounds = (-1.0, 1.0, -1.0, 1.0)
        self.view_scale = 0.9

        self.overlay_lines: list[str] = []
        self.overlay_drawers: list[callable] = []

    def on_draw(self) -> None:
        self.window.clear()
        if self.particle_vertices is not None or self.trail_vertices is not None:
            self.batch.draw()
        self._draw_energy_graphs()
        self._draw_overlay_text()
        for drawer in self.overlay_drawers:
            drawer()

    def update_scene(
        self,
        bodies: Sequence[Body],
        potential_energy: float | None = None,
    ) -> None:
        """Update particle and energy buffers before the next frame.

        Parameters
        ----------
        bodies:
            Current simulation bodies to render.
        potential_energy:
            Potential energy value for this frame. If omitted, no new
            potential energy sample is added. This allows re-using a
            pre-computed value from the simulation loop.
        """

        positions = np.array([body.position for body in bodies], dtype=float)
        masses = np.array([body.mass for body in bodies], dtype=float)
        velocities = np.array([body.velocity for body in bodies], dtype=float)

        self._update_bounds(positions)
        normalized = self._normalize_positions(positions)
        self._update_particles(normalized)
        self._update_trails(normalized)

        kinetic_energy = float(0.5 * np.sum(masses * np.sum(velocities**2, axis=1)))
        potential = potential_energy if potential_energy is not None else float("nan")
        if potential_energy is not None:
            self.energy_history.append(kinetic_energy, potential)
        else:
            # keep kinetic-only updates aligned with potential history length
            self.energy_history.append(kinetic_energy, self.energy_history.potential[-1] if self.energy_history.potential else 0.0)

    def set_overlay(self, lines: Sequence[str]) -> None:
        """Display simple text overlays in the upper-left corner."""

        self.overlay_lines = list(lines)

    def add_overlay_drawer(self, drawer: callable) -> None:
        """Register a callable invoked after the scene and graphs are drawn."""

        if drawer not in self.overlay_drawers:
            self.overlay_drawers.append(drawer)

    def run(self, update_callable) -> None:
        """Start the pyglet event loop.

        Parameters
        ----------
        update_callable:
            Callable invoked every frame with no arguments. It should step
            the simulation forward and call :meth:`update_scene` with the
            latest body state and energy values.
        """

        @self.window.event
        def on_draw() -> None:  # noqa: WPS430 - pyglet event handler
            self.on_draw()

        pyglet.clock.schedule_interval(lambda dt: update_callable(), 1 / 60.0)
        pyglet.app.run()

    # Internal helpers -------------------------------------------------

    def _update_bounds(self, positions: np.ndarray) -> None:
        min_x, min_y = positions.min(axis=0)
        max_x, max_y = positions.max(axis=0)
        padding_x = (max_x - min_x) * 0.1 + 1e-6
        padding_y = (max_y - min_y) * 0.1 + 1e-6
        self.bounds = (min_x - padding_x, max_x + padding_x, min_y - padding_y, max_y + padding_y)

    def _normalize_positions(self, positions: np.ndarray) -> np.ndarray:
        min_x, max_x, min_y, max_y = self.bounds
        span_x = max_x - min_x
        span_y = max_y - min_y
        norm_x = (positions[:, 0] - min_x) / span_x * 2.0 - 1.0
        norm_y = (positions[:, 1] - min_y) / span_y * 2.0 - 1.0
        return np.column_stack((norm_x * self.view_scale, norm_y * self.view_scale))

    def screen_to_world(self, x: float, y: float) -> np.ndarray:
        """Convert window pixel coordinates into simulation-space positions."""

        min_x, max_x, min_y, max_y = self.bounds
        span_x = max_x - min_x
        span_y = max_y - min_y
        norm_x = (((x / self.window_size) * 2.0) - 1.0) / self.view_scale
        norm_y = (((y / self.window_size) * 2.0) - 1.0) / self.view_scale
        norm_x = float(np.clip(norm_x, -1.0, 1.0))
        norm_y = float(np.clip(norm_y, -1.0, 1.0))
        world_x = min_x + (norm_x + 1.0) / 2.0 * span_x
        world_y = min_y + (norm_y + 1.0) / 2.0 * span_y
        return np.array([world_x, world_y], dtype=float)

    def _update_particles(self, normalized_positions: np.ndarray) -> None:
        flat_positions = normalized_positions.astype("f4", copy=False).ravel()
        count = len(normalized_positions)
        colors = (np.ones((count, 4), dtype="u1") * np.array([255, 255, 255, 255], dtype="u1")).ravel()

        if self.particle_vertices is None or self.particle_vertices.get_size() != count:
            if self.particle_vertices is not None:
                self.particle_vertices.delete()
            self.particle_vertices = self.batch.add(
                count,
                gl.GL_POINTS,
                None,
                ("v2f/static", flat_positions.tolist()),
                ("c4B/static", colors.tolist()),
            )
        else:
            self.particle_vertices.vertices = flat_positions.tolist()

    def _update_trails(self, normalized_positions: np.ndarray) -> None:
        required = len(normalized_positions)
        while len(self.position_history) < required:
            self.position_history.append(deque(maxlen=self.trail_length))

        for history, pos in zip(self.position_history, normalized_positions):
            history.append(tuple(pos))

        vertices: list[float] = []
        colors: list[int] = []
        for history in self.position_history[:required]:
            if len(history) < 2:
                continue
            faded = np.linspace(0.2, 1.0, num=len(history) - 1)
            history_points = list(history)
            for (x0, y0), (x1, y1), alpha in zip(history_points, history_points[1:], faded):
                vertices.extend([x0, y0, x1, y1])
                intensity = int(255 * alpha)
                colors.extend([255, 255, 255, intensity, 255, 255, 255, intensity])

        vertex_count = len(vertices) // 2
        if self.trail_vertices is not None:
            self.trail_vertices.delete()
        if vertex_count:
            self.trail_vertices = self.batch.add(
                vertex_count,
                gl.GL_LINES,
                None,
                ("v2f/static", vertices),
                ("c4B/static", colors),
            )
        else:
            self.trail_vertices = None

    def _draw_energy_graphs(self) -> None:
        if not self.energy_history.kinetic:
            return

        kinetic = np.array(self.energy_history.kinetic, dtype=float)
        potential = np.array(self.energy_history.potential, dtype=float)
        samples = len(kinetic)

        max_energy = np.nanmax(np.abs(np.concatenate([kinetic, potential])))
        if max_energy == 0 or np.isnan(max_energy):
            max_energy = 1.0

        width = self.window_size
        height = self.window_size
        margin = 60
        plot_height = height * 0.25
        plot_bottom = margin
        plot_top = plot_bottom + plot_height

        x_coords = np.linspace(margin, width - margin, samples)

        def to_screen(values: np.ndarray) -> np.ndarray:
            normalized = values / max_energy
            return plot_bottom + (normalized + 1) * (plot_height / 2)

        kinetic_y = to_screen(kinetic)
        potential_y = to_screen(potential)

        kinetic_vertices = np.empty(samples * 2, dtype="f4")
        potential_vertices = np.empty(samples * 2, dtype="f4")
        kinetic_vertices[0::2] = x_coords
        kinetic_vertices[1::2] = kinetic_y
        potential_vertices[0::2] = x_coords
        potential_vertices[1::2] = potential_y

        gl.glColor4f(0.2, 0.8, 1.0, 1.0)
        pyglet.graphics.draw(samples, gl.GL_LINE_STRIP, ("v2f", kinetic_vertices.tolist()))

        gl.glColor4f(1.0, 0.6, 0.2, 1.0)
        pyglet.graphics.draw(samples, gl.GL_LINE_STRIP, ("v2f", potential_vertices.tolist()))

        pyglet.text.Label(
            "Energy",
            font_size=12,
            x=margin,
            y=plot_top + 8,
            color=(200, 200, 200, 255),
        ).draw()
        pyglet.text.Label(
            "Kinetic",
            font_size=10,
            x=margin,
            y=plot_top - 12,
            color=(51, 204, 255, 255),
        ).draw()
        pyglet.text.Label(
            "Potential",
            font_size=10,
            x=margin,
            y=plot_top - 28,
            color=(255, 153, 51, 255),
        ).draw()

    def _draw_overlay_text(self) -> None:
        if not self.overlay_lines:
            return

        y = self.window_size - 18
        for line in self.overlay_lines:
            pyglet.text.Label(
                line,
                font_size=10,
                x=12,
                y=y,
                anchor_x="left",
                anchor_y="center",
                color=(230, 230, 230, 255),
            ).draw()
            y -= 16


__all__ = ["Renderer", "EnergyHistory"]
