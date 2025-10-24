from __future__ import annotations

# Manim scenes to visualize SDEs, Euler–Maruyama simulation, ensemble paths,
# distribution contours, and direct one-step sampling from p(x_t | x_0, ...).
#
# Language: English (narration text). Comments are also in English.

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
from manim import (
    BLUE, WHITE, BLACK,
    UP, DOWN, LEFT,
    Axes, Create, CurvedArrow, Dot, FadeIn, FadeOut,
    LaggedStart, Line, MathTex, MovingCameraScene, Scene, Succession,
    Tex, VGroup, Write, config,
)

config.background_color = WHITE

# ---------------------------
# Utilities and SDE helpers
# ---------------------------

@dataclass
class SDESpec:
    """Simple 1D SDE: dX_t = f(X_t, t) dt + g(X_t, t) dW_t"""
    drift: Callable[[np.ndarray, float], np.ndarray]
    diffusion: Callable[[np.ndarray, float], np.ndarray]


def euler_maruyama(
    sde: SDESpec,
    x0: float,
    t0: float,
    t1: float,
    num_steps: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate 1D SDE path using Euler–Maruyama.

    Returns times and states arrays of shape (num_steps+1,).
    """
    times = np.linspace(t0, t1, num_steps + 1)
    dt = times[1] - times[0]
    xs = np.zeros_like(times)
    xs[0] = x0
    for k in range(num_steps):
        t = times[k]
        x = xs[k]
        drift_val = sde.drift(np.array([x]), t)[0]
        diff_val = sde.diffusion(np.array([x]), t)[0]
        dW = rng.normal(loc=0.0, scale=np.sqrt(dt))
        xs[k + 1] = x + drift_val * dt + diff_val * dW
    return times, xs


def make_ou_sde(kappa: float = 1.0, theta: float = 0.0, sigma: float = 1.0) -> SDESpec:
    """Ornstein–Uhlenbeck SDE as a concrete example.
    dX_t = kappa (theta - X_t) dt + sigma dW_t
    """
    def drift(x: np.ndarray, t: float) -> np.ndarray:
        return kappa * (theta - x)

    def diffusion(x: np.ndarray, t: float) -> np.ndarray:
        return np.full_like(x, sigma)

    return SDESpec(drift=drift, diffusion=diffusion)


# ------------------------------------------------------
# Scene: Euler–Maruyama step-by-step then zoom out
# ------------------------------------------------------

class EulerMaruyamaZoom(MovingCameraScene):
    """Zoomed-in time grid demo, then zoom out to continuous-looking paths."""

    def construct(self):
        # Axes: horizontal time, vertical state
        axes = Axes(
            x_range=[0, 1.0, 0.1],
            y_range=[-3, 3, 1],
            x_length=10,
            y_length=5,
            tips=False,
        ).to_edge(DOWN).set_color(BLACK)
        axes.x_axis.set_opacity(0)
        axes.y_axis.set_opacity(0)

        origin_lower = axes.c2p(0, axes.y_range[0])
        time_axis_line = Line(
            origin_lower,
            axes.c2p(axes.x_range[1], axes.y_range[0]),
            color=BLACK,
        )
        state_axis_line = Line(
            origin_lower,
            axes.c2p(0, axes.y_range[1]),
            color=BLACK,
        )

        x_label = Tex(r"time $t$").scale(0.6).next_to(time_axis_line, DOWN, buff=0.2).set_color(BLACK)
        y_label = Tex(r"state $x$").scale(0.6).next_to(state_axis_line, LEFT, buff=0.2).set_color(BLACK)

        self.add(axes)
        self.play(Create(state_axis_line), Create(time_axis_line), FadeIn(x_label), FadeIn(y_label))

        # On-screen text per storyboard
        discretize_text = Tex(r"Discretize time: $t=0, \Delta t, 2\Delta t, \dots$").scale(0.7).set_color(BLACK)
        discretize_text.to_edge(UP)
        self.play(FadeIn(discretize_text))

        # Define SDE and EM parameters
        sde = make_ou_sde(kappa=1.2, theta=0.0, sigma=0.8)
        x0 = 0.0
        t0, t1 = 0.0, 1.0
        num_steps_dense = 120
        stretch_factor = 4.0  # Start 4x coarser (shows more of the trajectory)

        # Calculate coarse steps
        num_steps_coarse = int(num_steps_dense / stretch_factor)
        
        # Generate COARSE trajectory for the first simulation
        rng_coarse = np.random.default_rng(42)
        times_coarse, xs_coarse = euler_maruyama(sde, x0, t0, t1, num_steps_coarse, rng_coarse)
        
        # Generate DENSE trajectory for the second simulation  
        rng_dense = np.random.default_rng(42)
        times_dense, xs_dense = euler_maruyama(sde, x0, t0, t1, num_steps_dense, rng_dense)
        
        # Create grid lines at dense positions (121 lines)
        grid_marks = VGroup(
            *[
                Line(
                    axes.c2p(time_val, axes.y_range[0]),
                    axes.c2p(time_val, axes.y_range[1]),
                    color=BLACK,
                    stroke_opacity=0.18,
                )
                for time_val in times_dense
            ]
        )
        
        # Stretch the entire grid horizontally to spread it out
        grid_marks.stretch_about_point(stretch_factor, 0, origin_lower)
        
        # Add clipping: manually hide lines outside the x-axis range
        x_min = axes.c2p(0, 0)[0]
        x_max = axes.c2p(axes.x_range[1], 0)[0]
        
        # First, set all lines to be visible for FadeIn
        self.add(grid_marks)
        
        # During FadeIn, only show lines within range
        visible_lines = VGroup(*[line for line in grid_marks if x_min <= line.get_center()[0] <= x_max])
        hidden_lines = VGroup(*[line for line in grid_marks if line.get_center()[0] < x_min or line.get_center()[0] > x_max])
        
        # Calculate how many coarse grid lines are visible
        num_visible_grid_lines = len(visible_lines)
        
        # We need to generate a coarse trajectory whose vertices lie exactly on the
        # visible grid lines. Use the exact time span of the last visible dense grid line.
        num_steps_coarse_visible = num_visible_grid_lines - 1
        visible_time_end = times_dense[num_steps_coarse_visible]
        
        # Regenerate coarse trajectory over [t0, visible_time_end] so that, after stretching,
        # its vertices align with the visible grid lines.
        rng_coarse = np.random.default_rng(42)
        times_coarse_visible, xs_coarse_visible = euler_maruyama(
            sde, x0, t0, float(visible_time_end), num_steps_coarse_visible, rng_coarse
        )
        
        # Debug: print grid info
        print(f"Total grid lines: {len(grid_marks)}, Visible: {len(visible_lines)}, Hidden: {len(hidden_lines)}")
        print(f"x_min: {x_min}, x_max: {x_max}")
        print(f"Coarse trajectory time end (visible): {visible_time_end:.6f}")
        print(f"Coarse trajectory points (visible): {len(times_coarse_visible)}, segments: {num_steps_coarse_visible}")
        
        # Hide lines outside range before animation
        for line in hidden_lines:
            line.set_opacity(0)
        
        self.play(FadeIn(visible_lines))

        # Draw coarse trajectory using the visible coarse trajectory
        # Create points using times_coarse_visible positions and xs_coarse_visible values
        coarse_points = [axes.c2p(times_coarse_visible[j], xs_coarse_visible[j]) 
                        for j in range(len(times_coarse_visible))]
        coarse_curve = VGroup(
            *[
                Line(coarse_points[i - 1], coarse_points[i], color=BLUE, stroke_width=2)
                for i in range(1, len(coarse_points))
            ]
        )
        # Stretch the curve to match the stretched grid
        coarse_curve.stretch_about_point(stretch_factor, 0, origin_lower)
        
        # Clip curve manually - check if ANY part of the segment is visible
        visible_segments = VGroup()
        hidden_segments = VGroup()
        for line in coarse_curve:
            # Get start and end points of the line segment
            start_x = line.get_start()[0]
            end_x = line.get_end()[0]
            # Show segment if any part is within visible range
            if (start_x >= x_min and start_x <= x_max) or (end_x >= x_min and end_x <= x_max) or (start_x < x_min and end_x > x_max):
                visible_segments.add(line)
            else:
                line.set_opacity(0)
                hidden_segments.add(line)

        # Debug: print how many segments are visible
        print(f"Total coarse segments: {len(coarse_curve)}, Visible: {len(visible_segments)}, Hidden: {len(hidden_segments)}")
        print(f"Expected: Grid lines visible = {num_visible_grid_lines}, Trajectory points = {len(times_coarse_visible)}, Trajectory segments = {len(coarse_curve)}")
        
        # Step-by-step plotting: for each segment, draw the line then place a dot at its end
        dots = VGroup()
        step_animations = []
        for seg in visible_segments:
            end_dot = Dot(seg.get_end(), radius=0.04, color=BLUE)
            dots.add(end_dot)
            step_animations.extend([Create(seg), FadeIn(end_dot)])

        self.play(Succession(*step_animations), run_time=1.6)
        self.wait(0.6)
        self.play(FadeOut(visible_segments), FadeOut(dots))
        # No need to remove hidden segments since they were never added

        # Animate: compress grid to dense spacing with dynamic clipping
        def update_grid_clipping(mob):
            """Updater to hide lines outside x-axis range during animation"""
            for line in mob:
                line_x = line.get_center()[0]
                if x_min <= line_x <= x_max:
                    # Line is in visible range
                    if line.get_stroke_opacity() < 0.08:
                        line.set_stroke(opacity=0.08)
                else:
                    # Line is outside visible range
                    if line.get_stroke_opacity() > 0:
                        line.set_opacity(0)
        
        grid_marks.add_updater(update_grid_clipping)
        
        self.play(
            grid_marks.animate.stretch_about_point(
                1 / stretch_factor,
                0,
                origin_lower,
            ).set_stroke(opacity=0.08),
            run_time=2.0,
        )
        
        grid_marks.remove_updater(update_grid_clipping)
        # Final cleanup: ensure all visible lines have correct opacity
        for line in grid_marks:
            line_x = line.get_center()[0]
            if x_min <= line_x <= x_max:
                line.set_stroke(opacity=0.08)
            else:
                line.set_opacity(0)

        self.play(FadeOut(discretize_text))

        curve_points = [axes.c2p(times_dense[i], xs_dense[i]) for i in range(len(times_dense))]
        continuous_curve = VGroup(
            *[Line(curve_points[i - 1], curve_points[i], color=BLUE, stroke_width=2) for i in range(1, len(curve_points))]
        )

        self.play(LaggedStart(*[Create(seg) for seg in continuous_curve], lag_ratio=0.01, run_time=1.4))
        self.play(FadeOut(grid_marks))
        self.wait(0.8)


# ----------------------------------------------------------------------
# Scene: Many trajectories then switch to a distribution contour view
# ----------------------------------------------------------------------

class EnsembleToContour(Scene):
    """Show many trajectories from same x0 with different seeds, then a contour."""

    def construct(self):
        axes = Axes(
            x_range=[0, 1.0, 0.2],
            y_range=[-3, 3, 1],
            x_length=10,
            y_length=5,
            tips=False,
        ).to_edge(DOWN).set_color(BLACK)
        x_label = Tex("time t").scale(0.6).next_to(axes.x_axis, DOWN, buff=0.2).set_color(BLACK)
        y_label = Tex("state X").scale(0.6).next_to(axes.y_axis, LEFT, buff=0.2).set_color(BLACK)
        self.play(Create(axes), FadeIn(x_label), FadeIn(y_label))

        sde = make_ou_sde(kappa=1.2, theta=0.0, sigma=0.8)
        x0 = 0.0
        t0, t1 = 0.0, 1.0
        num_steps = 300
        num_paths = 50

        rng = np.random.default_rng(123)

        path_groups = []
        for j in range(num_paths):
            times, xs = euler_maruyama(sde, x0, t0, t1, num_steps, rng)
            pts = [axes.c2p(times[i], xs[i]) for i in range(len(times))]
            vg = VGroup(*[Line(pts[i - 1], pts[i], color=BLACK, stroke_opacity=0.25) for i in range(1, len(pts))])
            path_groups.append(vg)

        self.play(LaggedStart(*[Create(pg) for pg in path_groups], lag_ratio=0.02, run_time=1.4))
        self.wait(0.8)

        # Gather final-time samples and render a smooth density via KDE -> contour
        # Note: Implement a simple Gaussian KDE for 1D over y conditioned on fixed t.
        final_samples = []
        for j in range(num_paths):
            rng_local = np.random.default_rng(1000 + j)
            _, xs = euler_maruyama(sde, x0, t0, t1, num_steps, rng_local)
            final_samples.append(xs[-1])
        final_samples = np.asarray(final_samples)

        # Draw a vertical density at t=1 as a colored band + iso-contours
        t_final = 1.0
        t_pix = axes.c2p(t_final, 0.0)[0]
        y_vals = np.linspace(-3, 3, 200)
        # KDE bandwidth (Silverman's rule for normal-ish data)
        std = np.std(final_samples) + 1e-6
        h = 1.06 * std * (len(final_samples) ** (-1 / 5)) + 1e-3
        def kde(y):
            z = (y[:, None] - final_samples[None, :]) / h
            vals = np.exp(-0.5 * z**2).mean(axis=1) / (h * np.sqrt(2 * np.pi))
            return vals

        dens = kde(y_vals)
        dens /= dens.max() + 1e-8

        bars = VGroup()
        for i in range(len(y_vals) - 1):
            y0, y1 = y_vals[i], y_vals[i + 1]
            x0 = t_pix - 0.15 * dens[i]
            x1 = t_pix + 0.15 * dens[i]
            p00 = np.array([x0, axes.c2p(0, y0)[1], 0])
            p01 = np.array([x1, axes.c2p(0, y0)[1], 0])
            p10 = np.array([x0, axes.c2p(0, y1)[1], 0])
            p11 = np.array([x1, axes.c2p(0, y1)[1], 0])
            bar = VGroup(Line(p00, p01, color=BLUE, stroke_opacity=0.7), Line(p10, p11, color=BLUE, stroke_opacity=0.7), Line(p00, p10, color=BLUE, stroke_opacity=0.7), Line(p01, p11, color=BLUE, stroke_opacity=0.7))
            bars.add(bar)

        label = Tex("Distribution at t=1").scale(0.6).next_to(axes.c2p(1.0, 3.0), UP).set_color(BLACK)
        self.play(LaggedStart(*[FadeOut(pg) for pg in path_groups], lag_ratio=0.02, run_time=0.6))
        self.play(LaggedStart(*[Create(b) for b in bars], lag_ratio=0.01, run_time=1.0), FadeIn(label))
        self.wait(0.8)


# ----------------------------------------------------------------------
# Scene: Direct p(x_t|...) one-step sampling visualization
# ----------------------------------------------------------------------

class DirectSampling(Scene):
    """Illustrate direct sampling from p(x_t | x_0, ...) at arbitrary t in one step."""

    def construct(self):
        title = Tex("Direct one-shot sampling of $x_t$").scale(0.9).set_color(BLACK)
        self.play(Write(title))

        formula = MathTex(r"x_t \sim p_\theta(x_t \mid x_s;\, \Delta t)").next_to(title, DOWN).set_color(BLACK)
        note = Tex("Learn $p(x_t|x_s)$ with conditional normalizing flows.").scale(0.7).next_to(formula, DOWN).set_color(BLACK)
        self.play(FadeIn(formula), FadeIn(note))

        # Visual metaphor: arrows from a single x0 to multiple samples at different t values
        axis = Axes(x_range=[0, 1.0, 0.2], y_range=[-3, 3, 1], x_length=10, y_length=5, tips=False).to_edge(DOWN).set_color(BLACK)
        self.play(Create(axis))
        x0_val = 0.0
        x0_dot = Dot(axis.c2p(0.0, x0_val), color=BLACK)
        x0_label = Tex("$x_s$").scale(0.7).next_to(x0_dot, LEFT).set_color(BLACK)
        self.play(FadeIn(x0_dot), FadeIn(x0_label))

        times = [0.25, 0.5, 0.75, 1.0]
        colors = [BLUE, BLUE, BLUE, BLUE]
        rng = np.random.default_rng(7)

        # For demo, sample from a simple Gaussian with time-dependent variance
        for t, c in zip(times, colors):
            num_samples = 10
            std = 0.6 * np.sqrt(t + 1e-6)
            samples = x0_val + rng.normal(0, std, size=num_samples)
            dots = VGroup(*[Dot(axis.c2p(t, s), color=c, radius=0.03) for s in samples])
            arr = CurvedArrow(x0_dot.get_center(), axis.c2p(t, 0.0), angle=-0.8, color=c)
            t_lab = Tex(f"t={t:.2f}").scale(0.6).next_to(axis.c2p(t, 3.0), UP).set_color(BLACK)
            self.play(Create(arr), FadeIn(t_lab), LaggedStart(*[FadeIn(d) for d in dots], lag_ratio=0.05, run_time=0.6))
            self.wait(0.2)

        self.wait(1.0)


# --------------
# CLI reference
# --------------

"""
How to render (examples):

From the project root or the scripts directory:

    manim -pqh scripts/sde_animation.py EulerMaruyamaZoom
    manim -pqh scripts/sde_animation.py EnsembleToContour
    manim -pqh scripts/sde_animation.py DirectSampling

Flags:
    -p  : preview
    -q  : quality (l for low, h for high)
    -s  : save last frame as image
    --format=gif : save as gif
"""
