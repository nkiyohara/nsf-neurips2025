from __future__ import annotations

# Manim scenes to visualise SDEs, Euler–Maruyama simulation, ensemble paths,
# distribution contours, and direct one-step sampling from p(x_t | x_0, ...).
#
# Language: English (narration text). Comments are also in English.

import os
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
from manim import (
    BLUE,
    WHITE,
    BLACK,
    RED,
    UP,
    DOWN,
    LEFT,
    RIGHT,
    UL,
    UR,
    ORIGIN,
    Arc,
    ArcBetweenPoints,
    Axes,
    CapStyleType,
    Create,
    CurvedArrow,
    DecimalNumber,
    DoubleArrow,
    Dot,
    AnimationGroup,
    FadeIn,
    FadeOut,
    LaggedStart,
    Line,
    MathTex,
    MovingCameraScene,
    Scene,
    Succession,
    Tex,
    Text,
    VGroup,
    VMobject,
    ValueTracker,
    config,
)
from manim.utils.color import color_gradient

THEMES = {
    "light": {
        "background": WHITE,
        "axes": BLACK,
        "grid": "#4b5563",
        "text": BLACK,
        "primary": BLUE,
        "secondary": RED,
        "annotation": "#155e75",
    },
    "dark": {
        "background": BLACK,
        "axes": "#e2e8f0",
        "grid": "#94a3b8",
        "text": "#f8fafc",
        "primary": "#38bdf8",
        "secondary": "#f87171",
        "annotation": "#fbbf24",
    },
}

# Toggle via the SDE_ANIMATION_THEME environment variable (light or dark).
DEFAULT_THEME_NAME = "light"
THEME_NAME = os.environ.get("SDE_ANIMATION_THEME", DEFAULT_THEME_NAME).lower()
if THEME_NAME not in THEMES:
    THEME_NAME = DEFAULT_THEME_NAME

COLORS = THEMES[THEME_NAME]

config.background_color = COLORS["background"]
config.default_font = "Inter"  # Ensure Text elements use Inter for visual consistency

GRID_INITIAL_OPACITY = 0.3
GRID_FINAL_OPACITY = 0.18

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

class EulerMaruyama(MovingCameraScene):
    """Euler–Maruyama step-by-step simulation of an SDE."""

    def construct(self):
        # Axes: horizontal time, vertical state
        axes = Axes(
            x_range=[0, 1.0, 0.1],
            y_range=[-3, 3, 1],
            x_length=10,
            y_length=5,
            tips=False,
        ).move_to(ORIGIN).set_color(COLORS["axes"])
        axes.x_axis.set_opacity(0)
        axes.y_axis.set_opacity(0)

        origin_lower = axes.c2p(0, axes.y_range[0])
        time_axis_line = Line(
            origin_lower,
            axes.c2p(axes.x_range[1], axes.y_range[0]),
            color=COLORS["axes"],
        )
        time_axis_line.set_cap_style(CapStyleType.ROUND)
        state_axis_line = Line(
            origin_lower,
            axes.c2p(0, axes.y_range[1]),
            color=COLORS["axes"],
        )
        state_axis_line.set_cap_style(CapStyleType.ROUND)
        x_label = (
            Tex(r"Time $t$")
            .scale(0.75)
            .next_to(time_axis_line, DOWN, buff=0.2)
            .set_color(COLORS["text"])
        )
        y_label = (
            Tex(r"State $x$")
            .scale(0.75)
            .next_to(state_axis_line, LEFT, buff=0.2)
            .set_color(COLORS["text"])
        )

        self.add(axes, state_axis_line, time_axis_line, x_label, y_label)

        # On-screen text per storyboard
        discretise_text = (
            Tex(r"Discretise time: $t=0, \Delta t, 2\Delta t, \dots$")
            .scale(0.75)
            .set_color(COLORS["text"])
        )
        recursive_text = (
            Tex(r"Each $x_{k+1}$ needs $x_k$: iterate step by step.")
            .scale(0.75)
            .set_color(COLORS["annotation"])
        )
        discretise_text.to_edge(UP)
        recursive_text.next_to(discretise_text, DOWN, buff=0.2).align_to(discretise_text, ORIGIN)
        self.play(FadeIn(discretise_text), FadeIn(recursive_text))

        # Define SDE and EM parameters
        sde = make_ou_sde(kappa=1.2, theta=0.0, sigma=0.8)
        x0 = 0.0
        t0, t1 = 0.0, 1.0
        num_steps_dense = 120
        stretch_factor = 10.0  # Start 4x coarser (shows more of the trajectory)

        # Generate DENSE trajectory for the second simulation
        rng_dense = np.random.default_rng(42)
        times_dense, xs_dense = euler_maruyama(sde, x0, t0, t1, num_steps_dense, rng_dense)

        # Create grid lines at dense positions (121 lines)
        grid_marks = VGroup(
            *[
                Line(
                    axes.c2p(time_val, axes.y_range[0]),
                    axes.c2p(time_val, axes.y_range[1]),
                    color=COLORS["grid"],
                    stroke_opacity=GRID_INITIAL_OPACITY,
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
        visible_lines = VGroup(
            *[line for line in grid_marks if x_min <= line.get_center()[0] <= x_max]
        )
        hidden_lines = VGroup(
            *[
                line
                for line in grid_marks
                if line.get_center()[0] < x_min or line.get_center()[0] > x_max
            ]
        )

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

        # Hide lines outside range before animation
        for line in hidden_lines:
            line.set_opacity(0)

        # Create Δt visual indicator for the first grid interval

        # Use the actual grid line positions directly
        first_x = visible_lines[0].get_start()[0]
        second_x = visible_lines[1].get_start()[0]
        print(f"First x: {first_x}, Second x: {second_x}")
        
        indicator_y = axes.c2p(0, axes.y_range[1])[1] - 0.2
        
        # Create double arrow for the interval
        dt_arrow = DoubleArrow(
            start=[first_x, indicator_y, 0],
            end=[second_x, indicator_y, 0],
            color=COLORS["annotation"],
            stroke_width=4,
            tip_length=0.2,
            buff=0,  # zero buffer so arrowheads land exactly on the coarse grid lines
        )
        
        # Create Δt label
        dt_label = Tex(r"$\Delta t$", font_size=32).set_color(COLORS["annotation"])
        dt_label.next_to(dt_arrow, UP, buff=0.1)
        
        # Show grid and Δt indicator together
        self.play(FadeIn(visible_lines), FadeIn(dt_arrow), FadeIn(dt_label))

        # Draw coarse trajectory using the visible coarse trajectory
        # Create points using times_coarse_visible positions and xs_coarse_visible values
        coarse_points = [
            axes.c2p(times_coarse_visible[j], xs_coarse_visible[j])
            for j in range(len(times_coarse_visible))
        ]
        print(f"Coarse points: {coarse_points[0]}, {coarse_points[1]}, ...")
        coarse_curve = VGroup(
            *[
                Line(
                    coarse_points[i - 1],
                    coarse_points[i],
                    color=COLORS["primary"],
                    stroke_width=2,
                )
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
            if (
                (start_x >= x_min and start_x <= x_max)
                or (end_x >= x_min and end_x <= x_max)
                or (start_x < x_min and end_x > x_max)
            ):
                visible_segments.add(line)
            else:
                line.set_opacity(0)
                hidden_segments.add(line)

        # Step-by-step plotting: for each segment, draw the line then place a dot at its end
        dots = VGroup()
        start_dot = Dot(visible_segments[0].get_start(), radius=0.04, color=COLORS["primary"])
        dots.add(start_dot)
        self.play(FadeIn(start_dot))
        
        # Create all dots and segments but don't add them to scene yet
        for seg in visible_segments:
            end_dot = Dot(seg.get_end(), radius=0.04, color=COLORS["primary"])
            dots.add(end_dot)
        
        # Step-by-step animation: one arrow at a time
        arrow_intro_limit = min(5, len(visible_segments))
        for i, seg in enumerate(visible_segments):
            end_dot = dots[i + 1]  # +1 because start_dot is at index 0
            if i < arrow_intro_limit:
                arrow = CurvedArrow(
                    seg.get_start(),
                    seg.get_end(),
                    color=COLORS["annotation"],
                    stroke_width=2.2,
                    tip_length=0.15,
                )
                # Show arrow, then replace with segment and dot
                self.play(Create(arrow), run_time=0.3)
                self.play(
                    AnimationGroup(
                        FadeOut(arrow),
                        Create(seg),
                        FadeIn(end_dot),
                    ),
                    run_time=0.4,
                )
            else:
                # After the first few steps, skip the arrow cue for a quicker reveal
                self.play(
                    AnimationGroup(
                        Create(seg),
                        FadeIn(end_dot),
                    ),
                    run_time=0.1,
                )
        self.wait(0.6)
        # Fade out all visible elements
        fade_out_elements = [FadeOut(visible_segments), FadeOut(dots)]
        if dt_arrow is not None and dt_label is not None:
            fade_out_elements.extend([FadeOut(dt_arrow), FadeOut(dt_label)])
        self.play(*fade_out_elements)
        # No need to remove hidden segments since they were never added

        # Animate: compress grid to dense spacing with dynamic clipping
        def update_grid_clipping(mob):
            """Updater to hide lines outside x-axis range during animation"""
            for line in mob:
                line_x = line.get_center()[0]
                if x_min <= line_x <= x_max:
                    # Line is in visible range
                    if line.get_stroke_opacity() < GRID_FINAL_OPACITY:
                        line.set_stroke(opacity=GRID_FINAL_OPACITY)
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
            ).set_stroke(opacity=GRID_FINAL_OPACITY),
            run_time=2.0,
        )

        grid_marks.remove_updater(update_grid_clipping)
        # Final cleanup: ensure all visible lines have correct opacity
        for line in grid_marks:
            line_x = line.get_center()[0]
            if x_min <= line_x <= x_max:
                line.set_stroke(opacity=GRID_FINAL_OPACITY)
            else:
                line.set_opacity(0)

        self.play(FadeOut(discretise_text), FadeOut(recursive_text))

        num_dense_paths = 5
        dense_states = [xs_dense]
        for idx in range(1, num_dense_paths):
            rng_path = np.random.default_rng(42 + idx)
            _, xs_path = euler_maruyama(
                sde, x0, t0, t1, num_steps_dense, rng_path
            )
            dense_states.append(xs_path)

        dense_colors = color_gradient(
            [COLORS["primary"], COLORS["secondary"]], num_dense_paths
        )
        dense_curves = []
        for path_idx, xs_path in enumerate(dense_states):
            curve_points = [
                axes.c2p(times_dense[i], xs_path[i]) for i in range(len(times_dense))
            ]
            dense_curves.append(
                VGroup(
                    *[
                        Line(
                            curve_points[i - 1],
                            curve_points[i],
                            color=dense_colors[path_idx],
                            stroke_width=2,
                        )
                        for i in range(1, len(curve_points))
                    ]
                )
            )

        path_animations = [
            LaggedStart(
                *[Create(seg) for seg in curve],
                lag_ratio=0.01,
                run_time=1.4,
            )
            for curve in dense_curves
        ]

        self.play(AnimationGroup(*path_animations, lag_ratio=0.0))
        self.play(FadeOut(grid_marks))
        self.wait(0.8)


# ----------------------------------------------------------------------
# Scene: Many trajectories then switch to a distribution density view
# ----------------------------------------------------------------------

class EnsemblePathsToDensity(Scene):
    """Show many trajectories from same x0, then collapse them into a density plot."""

    def construct(self):
        axes = Axes(
            x_range=[0, 1.0, 0.1],
            y_range=[-3, 3, 1],
            x_length=10,
            y_length=5,
            tips=False,
        ).move_to(ORIGIN).set_color(COLORS["axes"])
        axes.x_axis.set_opacity(0)
        axes.y_axis.set_opacity(0)

        origin_lower = axes.c2p(0, axes.y_range[0])
        time_axis_line = Line(
            origin_lower,
            axes.c2p(axes.x_range[1], axes.y_range[0]),
            color=COLORS["axes"],
        )
        time_axis_line.set_cap_style(CapStyleType.ROUND)
        state_axis_line = Line(
            origin_lower,
            axes.c2p(0, axes.y_range[1]),
            color=COLORS["axes"],
        )
        state_axis_line.set_cap_style(CapStyleType.ROUND)

        x_label = (
            Tex(r"Time $t$")
            .scale(0.75)
            .next_to(time_axis_line, DOWN, buff=0.2)
            .set_color(COLORS["text"])
        )
        y_label = (
            Tex(r"State $x$")
            .scale(0.75)
            .next_to(state_axis_line, LEFT, buff=0.2)
            .set_color(COLORS["text"])
        )

        self.add(axes, state_axis_line, time_axis_line, x_label, y_label)

        sde = make_ou_sde(kappa=1.2, theta=0.0, sigma=0.8)
        x0 = 0.0
        t0, t1 = 0.0, 1.0
        num_steps = 300
        num_paths = 50

        rng = np.random.default_rng(123)

        # Precompute ensemble trajectories for dynamic density over time
        path_groups = []
        all_xs = []  # shape: (num_paths, num_steps+1)
        times_ref = None
        for j in range(num_paths):
            times, xs = euler_maruyama(sde, x0, t0, t1, num_steps, rng)
            if times_ref is None:
                times_ref = times
            pts = [axes.c2p(times[i], xs[i]) for i in range(len(times))]
            vg = VGroup(
                *[
                    Line(
                        pts[i - 1],
                        pts[i],
                        color=COLORS["primary"],
                        stroke_width=2,
                        stroke_opacity=0.25,
                    )
                    for i in range(1, len(pts))
                ]
            )
            path_groups.append(vg)
            all_xs.append(xs)
        xs_matrix = np.asarray(all_xs)  # (num_paths, num_steps+1)
        ensemble_group = VGroup(*path_groups)

        self.play(
            LaggedStart(
                *[Create(pg) for pg in path_groups], lag_ratio=0.02, run_time=1.4
            )
        )
        self.wait(0.8)
        grey_color = COLORS["grid"]
        self.play(
            LaggedStart(
                *[
                    pg.animate.set_stroke(color=grey_color, opacity=0.18)
                    for pg in path_groups
                ],
                lag_ratio=0.02,
                run_time=0.6,
            )
        )

        # Show initial density at t=1, then attach updaters to move it over time
        current_t = ValueTracker(1.0)
        y_vals = np.linspace(-3, 3, 200)
        width_scale = 0.15

        # Build initial bars from samples at t=1
        init_samples = xs_matrix[:, -1]
        std0 = np.std(init_samples) + 1e-6
        h0 = 1.06 * std0 * (len(init_samples) ** (-1 / 5)) + 1e-3
        z0 = (y_vals[:, None] - init_samples[None, :]) / h0
        dens0 = np.exp(-0.5 * z0**2).mean(axis=1) / (h0 * np.sqrt(2 * np.pi))
        dens0 /= dens0.max() + 1e-8
        t_pix0 = axes.c2p(1.0, 0.0)[0]

        bars = VGroup()
        for i in range(len(y_vals) - 1):
            y0v, y1v = y_vals[i], y_vals[i + 1]
            x0v = t_pix0 - width_scale * dens0[i]
            x1v = t_pix0 + width_scale * dens0[i]
            p00 = np.array([x0v, axes.c2p(0, y0v)[1], 0])
            p01 = np.array([x1v, axes.c2p(0, y0v)[1], 0])
            p10 = np.array([x0v, axes.c2p(0, y1v)[1], 0])
            p11 = np.array([x1v, axes.c2p(0, y1v)[1], 0])
            bar = VGroup(
                Line(p00, p01, color=COLORS["primary"], stroke_opacity=0.7),
                Line(p10, p11, color=COLORS["primary"], stroke_opacity=0.7),
                Line(p00, p10, color=COLORS["primary"], stroke_opacity=0.7),
                Line(p01, p11, color=COLORS["primary"], stroke_opacity=0.7),
            )
            bars.add(bar)

        # Dynamic label (two lines): title with t-value, and p(x_t | x_0)
        t_text = Tex(r"Distribution at $t=\,$").scale(0.75).set_color(COLORS["text"])
        t_value = DecimalNumber(1.00, num_decimal_places=2).scale(0.75).set_color(
            COLORS["text"]
        )
        title_row = VGroup(t_text, t_value).arrange(RIGHT, buff=0.06)
        subtitle = MathTex(r"p(x_t\mid x_0;\,\Delta t)").scale(0.75).set_color(
            COLORS["text"]
        )
        t_label_group = VGroup(title_row, subtitle).arrange(DOWN, buff=0.06)

        # Updater for bars to follow current_t and recompute KDE
        def update_bars(mob: VGroup):
            t = float(current_t.get_value())
            idx = int(np.clip(round((t - t0) / (t1 - t0) * num_steps), 0, num_steps))
            samples = xs_matrix[:, idx]
            std = np.std(samples) + 1e-6
            h = 1.06 * std * (len(samples) ** (-1 / 5)) + 1e-3
            z = (y_vals[:, None] - samples[None, :]) / h
            dens = np.exp(-0.5 * z**2).mean(axis=1) / (h * np.sqrt(2 * np.pi))
            dens /= dens.max() + 1e-8
            t_pix = axes.c2p(t, 0.0)[0]
            for i, bar in enumerate(mob):
                y0v, y1v = y_vals[i], y_vals[i + 1]
                x0v = t_pix - width_scale * dens[i]
                x1v = t_pix + width_scale * dens[i]
                y0_pix = axes.c2p(0, y0v)[1]
                y1_pix = axes.c2p(0, y1v)[1]
                p00 = np.array([x0v, y0_pix, 0])
                p01 = np.array([x1v, y0_pix, 0])
                p10 = np.array([x0v, y1_pix, 0])
                p11 = np.array([x1v, y1_pix, 0])
                # Replace each segment geometry to avoid cross-product issues during updates
                bar[0].become(Line(p00, p01, color=COLORS["primary"], stroke_opacity=0.7))
                bar[1].become(Line(p10, p11, color=COLORS["primary"], stroke_opacity=0.7))
                bar[2].become(Line(p00, p10, color=COLORS["primary"], stroke_opacity=0.7))
                bar[3].become(Line(p01, p11, color=COLORS["primary"], stroke_opacity=0.7))

        bars.add_updater(update_bars)

        # Updater for moving/refreshing the t label
        def update_label(group: VGroup):
            t = float(current_t.get_value())
            t_value.set_value(t)
            title_row.arrange(RIGHT, buff=0.06)
            group.arrange(DOWN, buff=0.06)
            anchor = axes.c2p(t, 3.0)
            group.move_to(anchor + np.array([0, 0.4, 0]))

        t_label_group.add_updater(update_label)

        # Ensure the parent group with updater is part of the scene, then fade it in
        self.add(bars)
        bars.set_z_index(2)
        ensemble_group.set_z_index(0)
        self.play(FadeIn(bars), FadeIn(t_label_group))

        # Sweep density left and right across time with real-time updates
        self.play(current_t.animate.set_value(0.02), run_time=1.6)
        self.play(current_t.animate.set_value(0.9), run_time=1.6)
        self.play(current_t.animate.set_value(0.5), run_time=1.2)
        self.wait(0.6)

        # Transition to direct sampling view
        bars.remove_updater(update_bars)
        t_label_group.remove_updater(update_label)
        self.play(FadeOut(bars), FadeOut(t_label_group), FadeOut(ensemble_group))

        t_title = (
            Text("One-step sampling at arbitrary time")
            .scale(0.75)
            .to_edge(UP)
            .set_color(COLORS["text"])
        )
        formula = (
            MathTex(r"x_t \sim p_\theta(x_t \mid x_s;\, \Delta t)")
            .scale(0.75)
            .next_to(t_title, DOWN, buff=0.12)
            .set_color(COLORS["text"])
        )
        self.play(FadeIn(t_title), FadeIn(formula))

        x0_val = 0.0
        start_dot = Dot(axes.c2p(0.0, x0_val), color=COLORS["annotation"], radius=0.06)
        self.play(FadeIn(start_dot))

        times_direct = [0.25, 0.5, 0.75, 1.0]
        time_colors = color_gradient(
            [COLORS["primary"], COLORS["secondary"]], len(times_direct)
        )
        rng_direct = np.random.default_rng(7)

        arrow_end_offset = 0.03  # shorten arrow so it doesn't overlap the sample column
        for t_val, color in zip(times_direct, time_colors):
            num_samples = 10
            std = 0.6 * np.sqrt(t_val + 1e-6)
            samples = x0_val + rng_direct.normal(0, std, size=num_samples)
            dots = VGroup(
                *[Dot(axes.c2p(t_val, s), color=color, radius=0.034) for s in samples]
            )
            t_end = max(t_val - arrow_end_offset, 0.02)
            arrow = CurvedArrow(
                axes.c2p(0.0, 0.0),
                axes.c2p(t_end, 0.0),
                angle=-0.85,
                color=color,
                stroke_width=5,
            )
            arrow.set(stroke_cap=CapStyleType.SQUARE)
            self.play(
                Create(arrow),
                LaggedStart(*[FadeIn(d) for d in dots], lag_ratio=0.05, run_time=0.6),
            )
            self.wait(0.2)

        self.wait(0.8)


# Chapman–Kolmogorov consistency helper functions
def a_of_dt(dt: float, lam: float, kappa: float = 1.2) -> float:
    # OU contraction raised to (1 + c*λ) to introduce a mild semigroup violation
    c = 0.35
    return float(np.exp(-kappa * dt) ** (1.0 + c * lam))

def v_of_dt(dt: float, lam: float, kappa: float = 1.2, sigma: float = 0.8) -> float:
    # Blend OU variance (semigroup) and Brownian variance (non-semigroup) by λ
    v_ou = (sigma**2 / (2.0*kappa)) * (1.0 - np.exp(-2.0*kappa*dt))
    v_bm = (sigma**2) * dt
    return float((1.0 - lam) * v_ou + lam * v_bm)

# ---------------------------
# Scene
# ---------------------------

class ChapmanKolmogorovConsistency(Scene):
    """Requested layout: 
    - TOP: one CurvedArrow s→u, one sample column at u (one-step)
    - BOTTOM: two CurvedArrows s→t→u, two sample columns (at t and at u) (two-step)
    - Finally: show distributions at u in both panels and compare at far right with CurvedArrows.
    """

    def construct(self):
        # ===== Layout =====
        def make_axes():
            ax = Axes(
                x_range=[0, 1.0, 0.1],
                y_range=[-3, 3, 1],
                x_length=9,
                y_length=2.6,
                tips=False,
            ).set_color(COLORS["axes"])
            ax.x_axis.set_opacity(0)
            ax.y_axis.set_opacity(0)
            return ax
        # Simpler positioning to avoid dependency on constants
        axes_top = make_axes()
        axes_top.move_to([-1.1, 1.9, 0])  # Moved left by 0.3 units from -0.8
        axes_bot = make_axes().move_to([-1.1, -2.0, 0])  # Moved left by 0.3 units from -0.8

        def add_axes_lines(ax: Axes):
            origin_lower = ax.c2p(0, ax.y_range[0])
            xline = Line(
                origin_lower,
                ax.c2p(ax.x_range[1], ax.y_range[0]),
                color=COLORS["axes"],
            )
            xline.set_cap_style(CapStyleType.ROUND)
            yline = Line(
                origin_lower,
                ax.c2p(0, ax.y_range[1]),
                color=COLORS["axes"],
            )
            yline.set_cap_style(CapStyleType.ROUND)
            return VGroup(xline, yline)
        lines_top = add_axes_lines(axes_top)
        lines_bot = add_axes_lines(axes_bot)

        # Add y-axis labels
        y_label_top = Tex(r"State $x$").scale(0.75).set_color(COLORS["text"])
        y_label_top.next_to(lines_top[1], LEFT, buff=0.15)
        y_label_bot = Tex(r"State $x$").scale(0.75).set_color(COLORS["text"])
        y_label_bot.next_to(lines_bot[1], LEFT, buff=0.15)
        
        # Add time axis labels
        x_label_top = Tex(r"Time").scale(0.75).set_color(COLORS["text"])
        x_label_top.next_to(lines_top[0], DOWN, buff=0.3)
        x_label_bot = Tex(r"Time").scale(0.75).set_color(COLORS["text"])
        x_label_bot.next_to(lines_bot[0], DOWN, buff=0.3)
        
        self.add(axes_top, axes_bot, lines_top[0], lines_top[1], lines_bot[0], lines_bot[1])
        self.add(y_label_top, y_label_bot, x_label_top, x_label_bot)

        # Titles
        sub_top = (
            Tex(r"One-step: $x_u \sim p_\theta(x_u\mid x_s;\,u-s)$")
            .scale(0.75)
            .set_color(COLORS["text"])
        )
        sub_bot = (
            Tex(
                r"Two-step: $x_u \sim \int p_\theta(x_u\mid x_t;\,u-t)\,p_\theta(x_t\mid x_s;\,t-s)\,\mathrm{d}x_t$"
            )
            .scale(0.75)
            .set_color(COLORS["text"])
        )
        sub_top.next_to(axes_top, UP, buff=0.18)
        sub_bot.next_to(axes_bot, UP, buff=0.18)
        self.play(FadeIn(sub_top), FadeIn(sub_bot))

        # Ticks and time variables
        # Display positions on graph
        s_display, t_display, u_display = 0.0, 0.5, 1.0
        # Actual time values for computation
        s_actual, t_actual, u_actual = 0.0, 0.4, 1.0
        
        s, t, u = s_display, t_display, u_display  # For graph positions
        
        def add_ticks(ax: Axes, show_u=False):
            tick_s = Line(
                ax.c2p(s, -3), ax.c2p(s, -2.85), color=COLORS["axes"], stroke_width=2
            )
            tick_t = Line(
                ax.c2p(t, -3), ax.c2p(t, -2.85), color=COLORS["axes"], stroke_width=2
            )
            # Center labels below ticks by using move_to with tick's x-coordinate
            lbl_s = Tex(r"$s$").scale(0.75).set_color(COLORS["text"])
            lbl_s.move_to([ax.c2p(s, -3)[0], tick_s.get_bottom()[1] - 0.15, 0])
            lbl_t = Tex(r"$t$").scale(0.75).set_color(COLORS["text"])
            lbl_t.move_to([ax.c2p(t, -3)[0], tick_t.get_bottom()[1] - 0.15, 0])
            if show_u:
                tick_u = Line(
                    ax.c2p(u, -3),
                    ax.c2p(u, -2.85),
                    color=COLORS["axes"],
                    stroke_width=2,
                )
                lbl_u = Tex(r"$u$").scale(0.75).set_color(COLORS["text"])
                lbl_u.move_to([ax.c2p(u, -3)[0], tick_u.get_bottom()[1] - 0.15, 0])
                return VGroup(tick_s, tick_t, tick_u, lbl_s, lbl_t, lbl_u)
            else:
                return VGroup(tick_s, tick_t, lbl_s, lbl_t)
        
        # Add ticks and labels with fade-in animation
        ticks_top = add_ticks(axes_top, True)
        ticks_bot = add_ticks(axes_bot, True)
        self.play(FadeIn(ticks_top), FadeIn(ticks_bot))

        # ===== Model =====
        kappa, sigma = 1.2, 0.8
        lam = ValueTracker(0.7)
        x_s = 0.0
        rng = np.random.default_rng(123)
        def a_of_dt(dt: float, lamv: float) -> float:
            c = 0.35
            return float(np.exp(-kappa * dt) ** (1.0 + c * lamv))
        def v_of_dt(dt: float, lamv: float) -> float:
            v_ou = (sigma**2 / (2.0*kappa)) * (1.0 - np.exp(-2.0*kappa*dt))
            v_bm = (sigma**2) * dt
            return float((1.0 - lamv) * v_ou + lamv * v_bm)
        def sample_one_step(dt: float, n: int, lamv: float):
            aT, vT = a_of_dt(dt, lamv), v_of_dt(dt, lamv)
            # Create bimodal distribution: mix two Gaussians with wider separation
            n1 = n // 2
            n2 = n - n1
            samples1 = aT * x_s + np.sqrt(max(vT, 1e-9)) * rng.normal(size=n1) - 1.0
            samples2 = aT * x_s + np.sqrt(max(vT, 1e-9)) * rng.normal(size=n2) + 1.2
            return np.concatenate([samples1, samples2])
        def sample_two_step(dt_total: float, n: int, lamv: float):
            # Actually compute both t and u as independent 1-step samples from s
            # (not a true 2-step, just showing two different time points)
            dt_to_t = t_actual - s_actual  # s to t
            dt_to_u = u_actual - s_actual  # s to u
            
            # Sample at t (one-step from s)
            at, vt = a_of_dt(dt_to_t, lamv), v_of_dt(dt_to_t, lamv)
            n1 = n // 2
            n2 = n - n1
            X_t1 = at * x_s + np.sqrt(max(vt, 1e-9)) * rng.normal(size=n1) - 0.9
            X_t2 = at * x_s + np.sqrt(max(vt, 1e-9)) * rng.normal(size=n2) + 1.1
            X_t = np.concatenate([X_t1, X_t2])
            
            # Sample at u (one-step from s)
            au, vu = a_of_dt(dt_to_u, lamv), v_of_dt(dt_to_u, lamv)
            X_u1 = au * x_s + np.sqrt(max(vu, 1e-9)) * rng.normal(size=n1) - 1.0
            X_u2 = au * x_s + np.sqrt(max(vu, 1e-9)) * rng.normal(size=n2) + 1.2
            X_u = np.concatenate([X_u1, X_u2])
            
            return X_t, X_u

        # ===== Stage 1: arrows + samples =====
        N_VIS = 14
        # Top: one arrow and one column at u (display position u, actual time u_actual)
        # Start with a larger black dot at s
        dot_top_s = Dot(axes_top.c2p(s, x_s), color=COLORS["annotation"], radius=0.06)
        
        arr_top = CurvedArrow(
            axes_top.c2p(s, 0.0),
            axes_top.c2p(u - 0.03, 0.0),
            angle=-0.7,
            color=COLORS["primary"],
            stroke_width=5,
        )
        arr_top.set(stroke_cap=CapStyleType.SQUARE)
        xs_top = sample_one_step(u_actual - s_actual, N_VIS, lam.get_value())
        dots_top_u = VGroup(
            *[
                Dot(axes_top.c2p(u, y), color=COLORS["primary"], radius=0.032)
                for y in xs_top
            ]
        )
        
        # Bottom: two arrows and two columns at t and u (display positions, actual times)
        # Start with a larger black dot at s
        dot_bot_s = Dot(axes_bot.c2p(s, x_s), color=COLORS["annotation"], radius=0.06)
        
        # Fade in the starting dots
        self.play(FadeIn(dot_top_s), FadeIn(dot_bot_s))
        
        arr_b1 = CurvedArrow(
            axes_bot.c2p(s, 0.0),
            axes_bot.c2p(t - 0.02, 0.0),
            angle=+0.8,
            color=COLORS["secondary"],
            stroke_width=5,
        )
        arr_b2 = CurvedArrow(
            axes_bot.c2p(t, 0.0),
            axes_bot.c2p(u - 0.03, 0.0),
            angle=+0.8,
            color=COLORS["secondary"],
            stroke_width=5,
        )
        for a in (arr_b1, arr_b2):
            a.set(stroke_cap=CapStyleType.SQUARE)
        xt, xu = sample_two_step(u_actual - s_actual, N_VIS, lam.get_value())
        dots_bot_t = VGroup(
            *[
                Dot(axes_bot.c2p(t, y), color=COLORS["secondary"], radius=0.03)
                for y in xt
            ]
        )
        dots_bot_u = VGroup(
            *[
                Dot(axes_bot.c2p(u, y), color=COLORS["secondary"], radius=0.032)
                for y in xu
            ]
        )
        
        # Fade in one-step and two-step elements immediately
        # Top panel animations
        self.play(Create(arr_top))
        self.play(LaggedStart(*[FadeIn(d) for d in dots_top_u], lag_ratio=0.05, run_time=0.5))
        
        # Bottom panel animations
        self.play(Create(arr_b1))
        self.play(LaggedStart(*[FadeIn(d) for d in dots_bot_t], lag_ratio=0.05, run_time=0.5))
        self.play(Create(arr_b2))
        self.play(LaggedStart(*[FadeIn(d) for d in dots_bot_u], lag_ratio=0.05, run_time=0.5))
        self.wait(0.4)
        
        # ===== Stage 2: distributions at u and t, comparator on the right with CurvedArrows =====
        y_min, y_max, N_BINS = -3.0, 3.0, 160
        y_grid = np.linspace(y_min, y_max, N_BINS + 1)
        ys_center = 0.5*(y_grid[:-1] + y_grid[1:])
        WIDTH, EPS = 0.04, 1e-6  # Narrower density lobes (quarter thickness)
        def kde_density(samples: np.ndarray, bw: float = 0.25):
            z = (ys_center[:, None] - samples[None, :]) / (bw + EPS)
            dens = np.exp(-0.5 * z**2).mean(axis=1) / (np.sqrt(2*np.pi)*(bw+EPS))
            dens = dens / (dens.sum()*(ys_center[1]-ys_center[0]) + EPS)
            return dens

        # Animate bars from lines to distributions
        N_BACK = 1500
        xs1 = sample_one_step(u_actual - s_actual, N_BACK, lam.get_value())
        xt2, xu2 = sample_two_step(u_actual - s_actual, N_BACK, lam.get_value())
        d1 = kde_density(xs1, bw=0.25)
        dt = kde_density(xt2, bw=0.15)  # Narrower bandwidth for intermediate distribution
        d2 = kde_density(xu2, bw=0.25)
        
        # Create alpha tracker for smooth transition of the filled patches
        alpha = ValueTracker(0.0)

        def make_density_patch(ax, time_pos, density, color):
            patch = VMobject()
            patch.set_style(
                stroke_color=color,
                stroke_width=2,
                stroke_opacity=0.0,
                fill_color=color,
                fill_opacity=0.0,
            )

            def updater(mob):
                a = alpha.get_value()
                norm = density.max() + EPS
                left_pts = []
                right_pts = []
                for y, dens in zip(ys_center, density):
                    half_width = a * WIDTH * (dens / norm)
                    left_pts.append(ax.c2p(time_pos - half_width, y))
                    right_pts.append(ax.c2p(time_pos + half_width, y))
                if not left_pts:
                    return mob
                outline = left_pts + right_pts[::-1] + [left_pts[0]]
                mob.set_points_as_corners(outline)
                mob.make_smooth()
                mob.set_style(
                    stroke_color=color,
                    stroke_width=2,
                    stroke_opacity=0.7 * a,
                    fill_color=color,
                    fill_opacity=0.3 * a,
                )
                return mob

            patch.add_updater(updater)
            return patch

        top_patch = make_density_patch(axes_top, u, d1, COLORS["primary"])
        bot_t_patch = make_density_patch(axes_bot, t, dt, COLORS["secondary"])
        bot_u_patch = make_density_patch(axes_bot, u, d2, COLORS["secondary"])
        self.add(top_patch, bot_t_patch, bot_u_patch)

        fade_top = LaggedStart(*[FadeOut(d) for d in dots_top_u], lag_ratio=0.05)
        fade_mid = LaggedStart(*[FadeOut(d) for d in dots_bot_t], lag_ratio=0.05)
        fade_bot = LaggedStart(*[FadeOut(d) for d in dots_bot_u], lag_ratio=0.05)
        self.play(
            AnimationGroup(fade_top, fade_mid, fade_bot, lag_ratio=0.15),
            alpha.animate.set_value(1.0),
            run_time=1.8,
        )

        for patch, color in (
            (top_patch, COLORS["primary"]),
            (bot_t_patch, COLORS["secondary"]),
            (bot_u_patch, COLORS["secondary"]),
        ):
            patch.clear_updaters()
            patch.set_style(
                stroke_color=color,
                stroke_width=2,
                stroke_opacity=0.7,
                fill_color=color,
                fill_opacity=0.3,
            )

        self.wait(0.3)
        
        # Now show the comparison arrow and text
        # Right comparator using a curved double-ended arrow
        right_x = axes_top.c2p(1.0, 0)[0] + 0.45  # Move arrow slightly left
        dist_top_y = axes_top.c2p(u, 0.0)[1]
        dist_bot_y = axes_bot.c2p(u, 0.0)[1]
        # Shorten the arrow vertically by moving endpoints closer to center
        center_y = (dist_top_y + dist_bot_y) / 2
        arrow_length_factor = 0.6  # Make arrow 60% of original vertical length
        p_top = np.array([right_x, center_y + (dist_top_y - center_y) * arrow_length_factor, 0])
        p_bot = np.array([right_x, center_y + (dist_bot_y - center_y) * arrow_length_factor, 0])
        
        # Create curved double-ended arrow using ArcBetweenPoints with tips on both ends
        comp_arrow = ArcBetweenPoints(
            p_top, p_bot, angle=-0.5, color=COLORS["annotation"], stroke_width=4
        )
        comp_arrow.add_tip(at_start=True)
        comp_arrow.add_tip(at_start=False)
        
        # Text with line break to prevent overflow
        match_txt = (
            Text("Match these\ndistributions", slant="ITALIC")
            .scale(0.65)
            .set_color(COLORS["text"])
        )
        # Position text to the right of the arrow
        match_txt.next_to(comp_arrow, RIGHT, buff=0.25)
        
        self.play(FadeIn(comp_arrow), FadeIn(match_txt))
        self.wait(1.0)



# --------------
# CLI reference
# --------------

"""
How to render (examples):

From the project root or the scripts directory:

    manim -pqh scripts/sde_animation.py EulerMaruyama
    manim -pqh scripts/sde_animation.py EnsemblePathsToDensity

Flags:
    -p  : preview
    -q  : quality (l for low, h for high)
    -s  : save last frame as image
    --format=gif : save as gif
"""
