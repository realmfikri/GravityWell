# GravityWell

Interactive N-body playground with a Barnes–Hut integrator and pyglet renderer.

## Running and controls

* Create a galaxy using `generate_spiral_galaxy` or `load_intersecting_spirals` from `src.simulation`, wrap the bodies in `SimulationState`, and pass them to `SimulationApp(Renderer(...))`.
* Keyboard
  * **Space** – pause/resume.
  * **Arrow Up/Down** – increase/decrease the integration timestep (`dt`) within safety bounds.
  * **Arrow Left/Right** – decrease/increase the gravitational constant (`G`) within safety bounds.
* Mouse/UI
  * Click the on-screen buttons for the same actions or to drop a new supermassive black hole at the center.
  * Left-click anywhere on the canvas to insert a supermassive black hole at that position.

Timestep and gravitational constant adjustments are clamped to configured minimum/maximum values to reduce the risk of numerical instability.
