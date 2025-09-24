from __future__ import annotations

# Manim scenes to visualize SDEs, Euler–Maruyama simulation, ensemble paths,
# distribution contours, and direct one-step sampling from p(x_t | x_0, ...).
#
# Language: English (narration text). Comments are also in English.

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
from manim import (
    BLUE, GREEN, ORANGE, RED, WHITE, YELLOW,
    UP, DOWN, LEFT,
    Axes, Create, CurvedArrow, Dot, FadeIn, FadeOut,
    LaggedStart, Line, MathTex, MovingCameraScene, Scene,
    Tex, VGroup, Write,
)


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


# ---------------------------
# Scene 1: SDE introduction
# ---------------------------

class SDEIntro(Scene):
    """Introduce continuous-time stochastic dynamics via SDE and how to simulate."""

    def construct(self):
        title = Tex("Continuous-time Stochastic Dynamics").scale(0.9)
        subtitle = Tex("Stochastic Differential Equation (SDE)").next_to(title, DOWN)

        sde_eq = MathTex(r"dX_t = f(X_t, t)\, dt + g(X_t, t)\, dW_t").scale(1.0)
        explain = Tex(
            "We simulate SDEs numerically using discretization methods,",
            " e.g., Euler--Maruyama."
        ).scale(0.7).next_to(sde_eq, DOWN)

        self.play(Write(title))
        self.play(FadeIn(subtitle, shift=0.2 * DOWN))
        self.wait(0.3)
        self.play(LaggedStart(Create(Line([-4, 2.2, 0], [4, 2.2, 0])), FadeIn(sde_eq), lag_ratio=0.2))
        self.play(FadeIn(explain))
        self.wait(1.2)

        steps = VGroup(
            Tex("1. Choose drift f and diffusion g.").scale(0.7),
            Tex("2. Pick time grid $t = 0, \\Delta t, 2\\Delta t, \\dots$").scale(0.7),
            Tex("3. Iterate with Gaussian noise increments $\\Delta W \\sim N(0, \\Delta t)$.").scale(0.7),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(explain, DOWN, buff=0.6)

        self.play(LaggedStart(*[FadeIn(item, shift=0.2 * DOWN) for item in steps], lag_ratio=0.15))
        self.wait(1.5)
        self.play(*[FadeOut(m) for m in [steps, explain, sde_eq, subtitle, title]])


# ------------------------------------------------------
# Scene 2: Euler–Maruyama step-by-step then zoom out
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
        ).to_edge(DOWN)

        x_label = Tex("time t").scale(0.6).next_to(axes.x_axis, DOWN, buff=0.2)
        y_label = Tex("state X").scale(0.6).next_to(axes.y_axis, LEFT, buff=0.2)

        self.play(Create(axes), FadeIn(x_label), FadeIn(y_label))

        # Define SDE and EM parameters
        sde = make_ou_sde(kappa=1.2, theta=0.0, sigma=0.8)
        x0 = 0.0
        t0, t1 = 0.0, 1.0
        num_steps_zoom = 20
        rng = np.random.default_rng(42)

        times_zoom, xs_zoom = euler_maruyama(sde, x0, t0, t1, num_steps_zoom, rng)

        # Plot piecewise linear path step-by-step
        path_lines = VGroup()
        dots = VGroup()
        for k in range(len(times_zoom)):
            point = axes.c2p(times_zoom[k], xs_zoom[k])
            dot = Dot(point, radius=0.04, color=YELLOW)
            dots.add(dot)
            if k > 0:
                prev_point = axes.c2p(times_zoom[k - 1], xs_zoom[k - 1])
                seg = Line(prev_point, point, color=YELLOW)
                path_lines.add(seg)

        # Draw time grid marks
        grid_marks = VGroup()
        for t in times_zoom:
            p0 = axes.c2p(t, -3)
            p1 = axes.c2p(t, 3)
            grid_marks.add(Line(p0, p1, color=WHITE, stroke_opacity=0.15))

        self.play(FadeIn(grid_marks))

        # Animate step-by-step EM
        self.play(FadeIn(dots[0]))
        for k in range(1, len(times_zoom)):
            self.play(Create(path_lines[k - 1]), FadeIn(dots[k]), run_time=0.2)
        self.wait(0.6)

        # Zoom out, hide grid, extend to a longer, denser path that looks continuous
        self.play(self.camera.frame.animate.scale(1.2).move_to(axes.get_center()))

        num_steps_long = 400
        times_long, xs_long = euler_maruyama(sde, x0, t0, t1, num_steps_long, rng)
        curve_points = [axes.c2p(times_long[i], xs_long[i]) for i in range(len(times_long))]
        continuous_curve = VGroup(
            *[Line(curve_points[i - 1], curve_points[i], color=BLUE, stroke_width=2) for i in range(1, len(curve_points))]
        )

        self.play(FadeOut(grid_marks), FadeOut(dots), FadeOut(path_lines))
        self.play(LaggedStart(*[Create(seg) for seg in continuous_curve], lag_ratio=0.01, run_time=1.2))
        self.wait(0.8)


# ----------------------------------------------------------------------
# Scene 3: Many trajectories then switch to a distribution contour view
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
        ).to_edge(DOWN)
        x_label = Tex("time t").scale(0.6).next_to(axes.x_axis, DOWN, buff=0.2)
        y_label = Tex("state X").scale(0.6).next_to(axes.y_axis, LEFT, buff=0.2)
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
            vg = VGroup(*[Line(pts[i - 1], pts[i], color=WHITE, stroke_opacity=0.25) for i in range(1, len(pts))])
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

        label = Tex("Distribution at t=1").scale(0.6).next_to(axes.c2p(1.0, 3.0), UP)
        self.play(LaggedStart(*[FadeOut(pg) for pg in path_groups], lag_ratio=0.02, run_time=0.6))
        self.play(LaggedStart(*[Create(b) for b in bars], lag_ratio=0.01, run_time=1.0), FadeIn(label))
        self.wait(0.8)


# ----------------------------------------------------------------------
# Scene 4: Direct p(x_t|...) one-step sampling visualization
# ----------------------------------------------------------------------

class DirectSampling(Scene):
    """Illustrate direct sampling from p(x_t | x_0, ...) at arbitrary t in one step."""

    def construct(self):
        title = Tex("Direct one-step sampling of $x_t$").scale(0.9)
        self.play(Write(title))

        formula = MathTex(r"x_t \sim p(x_t\,|\,x_0, t)").next_to(title, DOWN)
        note = Tex("Model learns the transition distribution to sample at any time.").scale(0.7).next_to(formula, DOWN)
        self.play(FadeIn(formula), FadeIn(note))

        # Visual metaphor: arrows from a single x0 to multiple samples at different t values
        axis = Axes(x_range=[0, 1.0, 0.2], y_range=[-3, 3, 1], x_length=10, y_length=5, tips=False).to_edge(DOWN)
        self.play(Create(axis))
        x0_val = 0.0
        x0_dot = Dot(axis.c2p(0.0, x0_val), color=YELLOW)
        x0_label = Tex("$x_0$").scale(0.7).next_to(x0_dot, LEFT)
        self.play(FadeIn(x0_dot), FadeIn(x0_label))

        times = [0.25, 0.5, 0.75, 1.0]
        colors = [ORANGE, GREEN, BLUE, RED]
        rng = np.random.default_rng(7)

        # For demo, sample from a simple Gaussian with time-dependent variance
        for t, c in zip(times, colors):
            num_samples = 10
            std = 0.6 * np.sqrt(t + 1e-6)
            samples = x0_val + rng.normal(0, std, size=num_samples)
            dots = VGroup(*[Dot(axis.c2p(t, s), color=c, radius=0.03) for s in samples])
            arr = CurvedArrow(x0_dot.get_center(), axis.c2p(t, 0.0), angle=-0.8, color=c)
            t_lab = Tex(f"t={t:.2f}").scale(0.6).next_to(axis.c2p(t, 3.0), UP)
            self.play(Create(arr), FadeIn(t_lab), LaggedStart(*[FadeIn(d) for d in dots], lag_ratio=0.05, run_time=0.6))
            self.wait(0.2)

        self.wait(1.0)


# --------------
# CLI reference
# --------------

"""
How to render (examples):

From the project root or the scripts directory:

    manim -pqh scripts/sde_animation.py SDEIntro
    manim -pqh scripts/sde_animation.py EulerMaruyamaZoom
    manim -pqh scripts/sde_animation.py EnsembleToContour
    manim -pqh scripts/sde_animation.py DirectSampling

Flags:
    -p  : preview
    -q  : quality (l for low, h for high)
    -s  : save last frame as image
    --format=gif : save as gif
"""


