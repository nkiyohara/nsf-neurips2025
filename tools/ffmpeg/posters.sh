#!/usr/bin/env bash
#
# Extract first-frame poster images from Manim renders.
# The script searches for MP4 files under build/manim/<theme>/videos/sde_animation
# and writes JPEG posters into build/manim/posters.
#
# Usage:
#   ./make_posters.sh          # process both dark and light themes
#   ./make_posters.sh dark     # process a single theme
#   ./make_posters.sh --force  # re-generate even if posters are newer than inputs

set -euo pipefail

QUALITY="2"      # ffmpeg JPEG quality (1=best, 31=worst)
OVERWRITE="false"

usage() {
  cat >&2 <<EOF
Usage: $0 [OPTIONS] [THEME...]
Themes default to: dark light

Options:
  --quality INT   JPEG quality for ffmpeg (default: 2)
  --force         Re-generate posters even if they appear up to date
  -h, --help      Show this help

Examples:
  $0
  $0 dark
  $0 --quality 4 light
EOF
}

declare -a ARGS_THEMES=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --quality)
      QUALITY="${2:-}"
      shift 2
      ;;
    --force)
      OVERWRITE="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
    *)
      ARGS_THEMES+=("$1")
      shift
      ;;
  esac
done

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Error: ffmpeg not found in PATH." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MEDIA_ROOT="${PROJECT_ROOT}/build/manim"
POSTER_ROOT="${PROJECT_ROOT}/build/manim/posters"

if [[ "${#ARGS_THEMES[@]}" -eq 0 ]]; then
  THEMES=("dark" "light")
else
  THEMES=("${ARGS_THEMES[@]}")
fi

mkdir -p "${POSTER_ROOT}"

find_theme_root() {
  local theme="$1"
  local -a candidates=(
    "${MEDIA_ROOT}/${theme}/videos/sde_animation"
    "${MEDIA_ROOT}/${theme}/videos/scenes"
    "${MEDIA_ROOT}/videos/${theme}/videos/sde_animation"
    "${MEDIA_ROOT}/videos/${theme}/videos/scenes"
  )

  for candidate in "${candidates[@]}"; do
    if [[ -d "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done

  return 1
}

generate_for_theme() {
  local theme="$1"
  local source_root
  if ! source_root="$(find_theme_root "${theme}")"; then
    echo "Skip ${theme}: render directory not found under ${MEDIA_ROOT}." >&2
    return
  fi

  local inputs_found=0
  while IFS= read -r input_path; do
    inputs_found=1
    local filename stem output_path tmp_output
    filename="$(basename "${input_path}")"
    stem="${filename%.mp4}"
    output_path="${POSTER_ROOT}/${stem}-${theme}.jpg"
    tmp_output="${output_path}.tmp"

    if [[ "${OVERWRITE}" != "true" && -f "${output_path}" && "${output_path}" -nt "${input_path}" ]]; then
      echo "[${theme}] ${stem}: up to date."
      continue
    fi

    echo "[${theme}] ${stem}: extracting poster..."
    ffmpeg -hide_banner -loglevel error -y \
      -i "${input_path}" \
      -frames:v 1 \
      -q:v "${QUALITY}" \
      -f image2 -update 1 "${tmp_output}" < /dev/null

    mv -f "${tmp_output}" "${output_path}"
  done < <(find "${source_root}" -type f -name '*.mp4' ! -path '*/partial_movie_files/*' -print | sort)

  if [[ "${inputs_found}" -eq 0 ]]; then
    echo "Skip ${theme}: no MP4 files found in ${source_root}." >&2
    return
  fi
}

for theme in "${THEMES[@]}"; do
  generate_for_theme "${theme}"
done

echo "Posters available in ${POSTER_ROOT}"
