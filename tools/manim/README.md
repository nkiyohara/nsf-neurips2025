# Manim project

This module contains the Manim implementation of the SDE animation scenes used
on the NSF NeurIPS 2025 project page.

- `nsf_sde_scenes/scenes.py` – collection of Manim scenes (light & dark themes)
- `render_scenes.sh` – helper that renders every scene for the requested theme(s)
  into `build/manim/<theme>`

Usage (from the repository root):

```bash
uv run --project tools/manim manim -a tools/manim/nsf_sde_scenes/scenes.py
./tools/manim/render_scenes.sh            # render both themes
./tools/manim/render_scenes.sh dark light # pass themes explicitly
```

Downstream conversion and staging steps live in `tools/ffmpeg` and `tools/cli`.

## System dependencies

Manim relies on Cairo and Pango libraries that must be available on the host.

```bash
sudo apt-get install ffmpeg pkg-config build-essential \
  libcairo2-dev libpango1.0-dev
```

Install the equivalents offered by your package manager when working on other
platforms. Scenes that render LaTeX objects also require a TeX distribution
such as TeX Live with the packages installed in the CI workflow. If `pkg-config`
cannot locate `pangocairo`, export `PKG_CONFIG_PATH=/usr/lib/x86_64-linux-gnu/pkgconfig:/usr/lib/pkgconfig:/usr/share/pkgconfig`
before running the render script.
