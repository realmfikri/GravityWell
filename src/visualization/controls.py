from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import pyglet
from pyglet import shapes


@dataclass
class ControlButton:
    label: str
    x: float
    y: float
    width: float
    height: float
    on_click: Callable[[], None]

    def draw(self) -> None:
        background = shapes.Rectangle(self.x, self.y, self.width, self.height, color=(40, 40, 50))
        border = shapes.BorderedRectangle(
            self.x,
            self.y,
            self.width,
            self.height,
            border=2,
            color=(55, 55, 70),
            border_color=(110, 170, 255),
        )
        background.draw()
        border.draw()
        pyglet.text.Label(
            self.label,
            font_size=10,
            x=self.x + self.width / 2,
            y=self.y + self.height / 2,
            anchor_x="center",
            anchor_y="center",
            color=(220, 220, 230, 255),
        ).draw()

    def hit_test(self, x: float, y: float) -> bool:
        return self.x <= x <= self.x + self.width and self.y <= y <= self.y + self.height


class ControlPanel:
    """Lightweight clickable widgets to drive the simulation."""

    def __init__(self, *, margin: float = 12.0, spacing: float = 8.0) -> None:
        self.buttons: list[ControlButton] = []
        self.margin = margin
        self.spacing = spacing

    def layout(self, rows: Sequence[Sequence[ControlButton]]) -> None:
        """Position buttons in simple rows with consistent spacing."""

        self.buttons.clear()
        y = self.margin
        for row in rows:
            x = self.margin
            for button in row:
                button.x = x
                button.y = y
                self.buttons.append(button)
                x += button.width + self.spacing
            y += (row[0].height if row else 0) + self.spacing

    def draw(self) -> None:
        for button in self.buttons:
            button.draw()

    def handle_mouse_press(self, x: float, y: float, button: int) -> bool:
        if button != pyglet.window.mouse.LEFT:
            return False
        for control in self.buttons:
            if control.hit_test(x, y):
                control.on_click()
                return True
        return False


__all__ = ["ControlButton", "ControlPanel"]
