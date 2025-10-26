#!/usr/bin/env bash
#
# Render the Manim scenes for the NSF NeurIPS 2025 project page.
# Outputs are stored in build/manim/<theme> so that downstream tooling
# can convert, posterise, and stage the assets.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MANIM_PROJECT="${PROJECT_ROOT}/tools/manim"
BUILD_ROOT="${PROJECT_ROOT}/build/manim"

if [[ ! -f "${MANIM_PROJECT}/pyproject.toml" ]]; then
  echo "Unable to locate Manim project at ${MANIM_PROJECT}" >&2
  exit 1
fi

declare -a THEMES=()
if [[ "$#" -eq 0 ]]; then
  THEMES=("dark" "light")
else
  THEMES=("$@")
fi

for theme in "${THEMES[@]}"; do
  theme_lower="$(echo "${theme}" | tr '[:upper:]' '[:lower:]')"
  output_root="${BUILD_ROOT}/${theme_lower}"
  mkdir -p "${output_root}"

  echo "Rendering scenes for theme: ${theme_lower}"
  SDE_ANIMATION_THEME="${theme_lower}" \
    uv run --project "${MANIM_PROJECT}" manim \
    -a "${MANIM_PROJECT}/nsf_sde_scenes/scenes.py" \
    -r 1280,720 \
    -qh \
    --media_dir "${output_root}" \
    --disable_caching
done
