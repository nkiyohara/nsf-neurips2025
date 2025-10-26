# Neural Stochastic Flows – project page

This repository hosts the NeurIPS 2025 project page for *Neural Stochastic
Flows*. The codebase is organised to keep reproducible build steps separate
from the lightweight site that gets deployed.

## Layout

- `site/` – deployable artefacts (HTML, favicon, posters, optimised videos)
- `assets/` – version-controlled research assets (figures, data extracts)
- `tools/` – media pipeline
  - `manim/` – Manim project (`nsf_sde_scenes`) and helpers
  - `ffmpeg/` – encoding and poster extraction utilities
  - `cli/` – orchestration scripts (render → encode → stage)
- `build/` – throw-away outputs created during local or CI builds
- `docs/` – long-form documentation or supplementary notes

## Typical workflow

```bash
# Render scenes into build/manim/<theme>
./tools/manim/render_scenes.sh [dark|light]

# Convert to WebM + extract posters
./tools/ffmpeg/encode_webm.sh [dark|light]
./tools/ffmpeg/posters.sh [dark|light]

# Run the entire pipeline and stage files into site/assets
./tools/cli/build_media.sh [dark|light]
```

The scripts rely on `uv` and `manim` being available. The render step
automatically resolves dependencies from `tools/manim/pyproject.toml`.

## Continuous integration

GitHub workflow definitions live under `.github/workflows/` (see
`build-media.yml` for automated rendering and `deploy.yml` for publishing).

## Local development tips

- Generated files live under `build/`; wipe the directory to start fresh.
- Keep long-lived source artefacts (SVG, reference data) under `assets/`.
- Commit only the optimised media inside `site/assets/` to keep diffs and
  deploy size small.
