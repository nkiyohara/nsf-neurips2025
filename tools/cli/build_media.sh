#!/usr/bin/env bash
#
# Convenience wrapper to generate, convert, and deploy SDE animation media.
# Runs, in order:
#   1. manim/render_scenes.sh   (Manim renders)
#   2. ffmpeg/encode_webm.sh    (VP9 conversions)
#   3. ffmpeg/posters.sh        (JPEG posters)
#   4. stage_assets.sh          (sync outputs into site/assets)
#
# All arguments are forwarded to each step where applicable.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run_step() {
  local script="$1"
  shift
  echo "==> ${script} $*"
  "${script}" "$@"
}

run_step "${SCRIPT_DIR}/../manim/render_scenes.sh" "$@"
run_step "${SCRIPT_DIR}/../ffmpeg/encode_webm.sh" "$@"
run_step "${SCRIPT_DIR}/../ffmpeg/posters.sh" "$@"
run_step "${SCRIPT_DIR}/stage_assets.sh" "$@"

echo "Media build complete."
