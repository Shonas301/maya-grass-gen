"""Grass-like flow field visualization with obstacle avoidance.

This sketch demonstrates flow fields that curve around obstacles, with
grass blade (point) clustering around obstacle edges. designed as a
preview/testing tool for maya grass animation development.

features:
- interactive obstacle placement (click to add)
- flow field visualization showing wind direction
- clustered points representing grass blade positions
- animated flow that curves naturally around obstacles
"""

import time
from pathlib import Path

import click
import numpy as np
from py5 import Sketch

from maya_grass_gen.flow_field import (
    ClusteringConfig,
    FlowField,
    FlowFieldConfig,
    Obstacle,
    PointClusterer,
)


class GrassFlowFieldSketch(Sketch):
    """Interactive grass flow field visualization.

    click to place obstacles, watch flow curve around them and grass
    points cluster at edges.
    """

    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        output_path: str | None = None,
        num_points: int = 2000,
        show_flow_lines: bool = True,
        show_points: bool = True,
        show_obstacles: bool = True,
        seed: int | None = None,
    ) -> None:
        """Initialize the sketch.

        Args:
            width: canvas width in pixels
            height: canvas height in pixels
            output_path: if provided, save output and exit
            num_points: number of grass points to generate
            show_flow_lines: whether to draw flow particle traces
            show_points: whether to draw grass points
            show_obstacles: whether to draw obstacle circles
            seed: random seed for reproducibility
        """
        super().__init__()
        self.canvas_width = width
        self.canvas_height = height
        self.output_path = output_path
        self.num_points = num_points
        self.show_flow_lines = show_flow_lines
        self.show_points = show_points
        self.show_obstacles = show_obstacles
        self.seed = seed

        # flow field configuration
        self.flow_config = FlowFieldConfig(
            noise_scale=0.004,
            flow_strength=2.5,
            octaves=4,
            persistence=0.5,
            time_scale=0.008,
        )
        self.flow_field = FlowField(config=self.flow_config)

        # clustering configuration
        self.clustering_config = ClusteringConfig(
            base_density=0.001,
            obstacle_density_multiplier=4.0,
            min_distance=8.0,
            cluster_falloff=0.6,
            edge_offset=15.0,
        )

        # visualization state
        self.time_offset = 0.0
        self.grass_points: list[tuple[float, float]] = []
        self.flow_particles: list[list[float]] = []
        self.needs_regeneration = True

    def settings(self) -> None:
        """Configure sketch size."""
        self.size(self.canvas_width, self.canvas_height)

    def setup(self) -> None:
        """Set up drawing environment."""
        if self.seed is not None:
            self.random_seed(self.seed)

        self.background(15, 25, 15)  # dark green-ish background

        # add some default obstacles for demonstration
        self._add_default_obstacles()

        # generate initial points and particles
        self._regenerate_points()

        # no_loop for static output, loop for interactive
        if self.output_path:
            self.no_loop()

    def _add_default_obstacles(self) -> None:
        """Add demonstration obstacles."""
        # central rock
        self.flow_field.add_obstacle(
            Obstacle(
                x=self.canvas_width * 0.5,
                y=self.canvas_height * 0.5,
                radius=80,
                influence_radius=200,
                strength=1.0,
            )
        )

        # smaller rocks scattered around
        self.flow_field.add_obstacle(
            Obstacle(
                x=self.canvas_width * 0.25,
                y=self.canvas_height * 0.35,
                radius=40,
                influence_radius=100,
                strength=0.8,
            )
        )

        self.flow_field.add_obstacle(
            Obstacle(
                x=self.canvas_width * 0.75,
                y=self.canvas_height * 0.65,
                radius=55,
                influence_radius=140,
                strength=0.9,
            )
        )

        self.flow_field.add_obstacle(
            Obstacle(
                x=self.canvas_width * 0.15,
                y=self.canvas_height * 0.75,
                radius=35,
                influence_radius=90,
                strength=0.7,
            )
        )

        self.flow_field.add_obstacle(
            Obstacle(
                x=self.canvas_width * 0.85,
                y=self.canvas_height * 0.25,
                radius=45,
                influence_radius=110,
                strength=0.85,
            )
        )

    def _regenerate_points(self) -> None:
        """Regenerate grass points and flow particles."""
        # create clusterer with current obstacles
        clusterer = PointClusterer(
            width=self.canvas_width,
            height=self.canvas_height,
            config=self.clustering_config,
            obstacles=self.flow_field.obstacles,
            seed=self.seed,
        )

        # generate grass points
        self.grass_points = clusterer.generate_points_grid_based(self.num_points)

        # generate flow particles (subset for visualization)
        num_flow_particles = min(500, self.num_points // 4)
        self.flow_particles = []

        # distribute flow particles across canvas
        for _ in range(num_flow_particles):
            x = self.random(self.canvas_width)
            y = self.random(self.canvas_height)

            # check not inside obstacle
            valid = True
            for obs in self.flow_field.obstacles:
                dx = x - obs.x
                dy = y - obs.y
                if np.sqrt(dx**2 + dy**2) < obs.radius:
                    valid = False
                    break

            if valid:
                self.flow_particles.append([x, y])

        self.needs_regeneration = False

    def draw(self) -> None:
        """Main drawing function."""
        # semi-transparent background for trail effect
        self.fill(15, 25, 15, 30)
        self.no_stroke()
        self.rect(0, 0, self.canvas_width, self.canvas_height)

        # draw layers
        if self.show_flow_lines:
            self._draw_flow_lines()

        if self.show_points:
            self._draw_grass_points()

        if self.show_obstacles:
            self._draw_obstacles()

        # advance time
        self.time_offset += 0.5

        # save if output path specified
        if self.output_path:
            self.save(self.output_path)
            print(f"    saved: {self.output_path}")
            self.exit_sketch()

    def _draw_flow_lines(self) -> None:
        """Draw flow particle traces."""
        self.stroke_weight(1.0)
        self.no_fill()

        for particle in self.flow_particles:
            x, y = particle

            # get flow at current position
            vx, vy = self.flow_field.get_flow(x, y, self.time_offset)

            # draw short line segment showing flow direction
            # color based on flow angle
            angle = np.arctan2(vy, vx)
            hue = (angle + np.pi) / (2 * np.pi)  # normalize to 0-1

            # green-cyan color scheme (grass/wind aesthetic)
            r = int(50 + 80 * hue)
            g = int(150 + 80 * (1 - hue))
            b = int(100 + 50 * hue)

            self.stroke(r, g, b, 150)

            # draw flow line
            steps = 30
            line_x, line_y = x, y
            for step in range(steps):
                vx, vy = self.flow_field.get_flow(line_x, line_y, self.time_offset)
                new_x = line_x + vx * 0.8
                new_y = line_y + vy * 0.8

                # fade alpha
                alpha = int(150 * (1 - step / steps))
                self.stroke(r, g, b, alpha)
                self.line(line_x, line_y, new_x, new_y)

                line_x, line_y = new_x, new_y

                # boundary check
                if (
                    line_x < 0
                    or line_x > self.canvas_width
                    or line_y < 0
                    or line_y > self.canvas_height
                ):
                    break

            # update particle position for next frame
            particle[0] += vx * 0.5
            particle[1] += vy * 0.5

            # wrap around edges
            if particle[0] < 0:
                particle[0] = self.canvas_width
            elif particle[0] > self.canvas_width:
                particle[0] = 0
            if particle[1] < 0:
                particle[1] = self.canvas_height
            elif particle[1] > self.canvas_height:
                particle[1] = 0

    def _draw_grass_points(self) -> None:
        """Draw grass blade positions as oriented marks."""
        self.stroke_weight(2.0)

        for x, y in self.grass_points:
            # get flow angle at this point
            angle = self.flow_field.get_flow_angle(x, y, self.time_offset)

            # grass blade represented as short angled line
            blade_length = 6 + 4 * np.sin(x * 0.01 + y * 0.01)  # slight variation

            # calculate blade endpoints
            # blade points upward but leans with wind
            base_angle = -np.pi / 2  # pointing up
            lean = angle * 0.3  # partial lean toward flow
            final_angle = base_angle + lean

            dx = np.cos(final_angle) * blade_length
            dy = np.sin(final_angle) * blade_length

            # color - varying greens
            hue_offset = np.sin(x * 0.02) * 30
            r = int(40 + hue_offset)
            g = int(140 + hue_offset + np.cos(y * 0.02) * 40)
            b = int(50 + hue_offset * 0.5)

            self.stroke(r, g, b, 200)
            self.line(x, y, x + dx, y + dy)

    def _draw_obstacles(self) -> None:
        """Draw obstacle circles."""
        self.stroke_weight(2.0)

        for obstacle in self.flow_field.obstacles:
            # influence_radius is guaranteed to be set by __post_init__
            assert obstacle.influence_radius is not None

            # outer influence radius (faint)
            self.stroke(100, 100, 100, 30)
            self.no_fill()
            self.ellipse(
                obstacle.x,
                obstacle.y,
                obstacle.influence_radius * 2,
                obstacle.influence_radius * 2,
            )

            # inner obstacle (solid)
            self.stroke(80, 70, 60, 200)
            self.fill(50, 45, 40, 180)
            self.ellipse(
                obstacle.x, obstacle.y, obstacle.radius * 2, obstacle.radius * 2
            )

    def mouse_pressed(self) -> None:
        """Handle mouse click to add obstacle."""
        # add new obstacle at click position
        new_obstacle = Obstacle(
            x=float(self.mouse_x),
            y=float(self.mouse_y),
            radius=30 + self.random(40),
            strength=0.7 + self.random(0.3),
        )
        self.flow_field.add_obstacle(new_obstacle)

        # regenerate points
        self._regenerate_points()

        print(f"added obstacle at ({self.mouse_x}, {self.mouse_y})")

    def key_pressed(self) -> None:
        """Handle key presses."""
        if self.key == "c":
            # clear all obstacles
            self.flow_field.clear_obstacles()
            self._regenerate_points()
            print("cleared all obstacles")
        elif self.key == "r":
            # regenerate points
            self._regenerate_points()
            print("regenerated points")
        elif self.key == "s":
            # save current frame
            timestamp = int(time.time())
            save_path = f"output/grass_flow_{timestamp}.png"
            Path("output").mkdir(exist_ok=True)
            self.save(save_path)
            print(f"saved: {save_path}")


def parse_resolution(resolution_str: str) -> tuple[int, int]:
    """Parse resolution string to (width, height) tuple.

    Args:
        resolution_str: resolution as shorthand (e.g., "1080p", "4k") or "WIDTHxHEIGHT"

    Returns:
        tuple of (width, height) in pixels

    Raises:
        ValueError: if resolution string is invalid
    """
    shorthand_map = {
        "4k": (3840, 2160),
        "1440p": (2560, 1440),
        "1080p": (1920, 1080),
        "720p": (1280, 720),
    }

    resolution_lower = resolution_str.lower()
    if resolution_lower in shorthand_map:
        return shorthand_map[resolution_lower]

    expected_parts = 2
    if "x" in resolution_str:
        try:
            parts = resolution_str.split("x")
            if len(parts) == expected_parts:
                width = int(parts[0])
                height = int(parts[1])
                if width > 0 and height > 0:
                    return (width, height)
        except ValueError:
            pass

    valid_formats = ", ".join(shorthand_map.keys())
    msg = (
        f"invalid resolution: '{resolution_str}'. "
        f"use shorthand ({valid_formats}) or WIDTHxHEIGHT format"
    )
    raise ValueError(msg)


@click.command()
@click.option(
    "--render",
    is_flag=True,
    help="render single frame instead of interactive preview",
)
@click.option(
    "--resolution",
    default="1080p",
    help="resolution (1080p, 4k, etc.) or WIDTHxHEIGHT",
)
@click.option(
    "--output",
    default="output/grass_flow_field.png",
    help="output file path (only used with --render)",
)
@click.option(
    "--points",
    default=2000,
    type=int,
    help="number of grass points",
)
@click.option(
    "--seed",
    default=None,
    type=int,
    help="random seed for reproducibility",
)
@click.option(
    "--no-flow-lines",
    is_flag=True,
    help="hide flow line visualization",
)
@click.option(
    "--no-points",
    is_flag=True,
    help="hide grass points",
)
@click.option(
    "--no-obstacles",
    is_flag=True,
    help="hide obstacle circles",
)
def main(
    render: bool,
    resolution: str,
    output: str,
    points: int,
    seed: int | None,
    no_flow_lines: bool,
    no_points: bool,
    no_obstacles: bool,
) -> None:
    """Run grass flow field visualization.

    interactive mode (default): click to add obstacles, 'c' to clear, 'r' to
    regenerate points, 's' to save screenshot.

    render mode (--render): save single frame and exit.
    """
    try:
        width, height = parse_resolution(resolution)
    except ValueError as e:
        click.echo(f"error: {e}", err=True)
        raise click.Abort() from None

    if render:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        click.echo("rendering grass flow field...")
        click.echo(f"   resolution: {resolution} ({width}x{height})")
        click.echo(f"   points: {points}")
        click.echo(f"   output: {output_path.absolute()}")

        sketch = GrassFlowFieldSketch(
            width=width,
            height=height,
            output_path=str(output_path),
            num_points=points,
            show_flow_lines=not no_flow_lines,
            show_points=not no_points,
            show_obstacles=not no_obstacles,
            seed=seed,
        )
        sketch.run_sketch(block=True)

        click.echo("render complete!")
    else:
        click.echo("interactive grass flow field")
        click.echo(f"   resolution: {resolution} ({width}x{height})")
        click.echo(f"   points: {points}")
        click.echo("")
        click.echo("controls:")
        click.echo("   click: add obstacle")
        click.echo("   c: clear obstacles")
        click.echo("   r: regenerate points")
        click.echo("   s: save screenshot")
        click.echo("")

        sketch = GrassFlowFieldSketch(
            width=width,
            height=height,
            num_points=points,
            show_flow_lines=not no_flow_lines,
            show_points=not no_points,
            show_obstacles=not no_obstacles,
            seed=seed,
        )
        sketch.run_sketch()


if __name__ == "__main__":
    main()
