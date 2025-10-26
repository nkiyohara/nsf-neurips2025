#!/usr/bin/env bash
#
# Convert 720p60 Manim MP4 renders into VP9 WebM files (no transparency).
# Usage examples:
#   ./prepare_webm.sh          # process both dark and light themes
#   ./prepare_webm.sh dark     # process only the dark theme
#
# Directory layout assumed:
#   <repo>/build/manim/<theme>/videos/sde_animation/720p60/*.mp4
#
# Requirements: ffmpeg (with libvpx-vp9)

set -euo pipefail

CRF="32"            # lower = higher quality (e.g., 28)
PIX_FMT="yuv420p"   # VP9 without alpha
OVERWRITE="false"   # if true, ignore timestamp checks and re-encode

usage() {
  cat >&2 <<EOF
Usage: $0 [OPTIONS] [THEME...]
Themes default to: dark light

Options:
  --crf INT         VP9 CRF (default: 32; try lower like 28 for higher quality)
  --pix-fmt FORMAT  Pixel format (default: yuv420p)
  --force           Re-encode even if output is newer than input
  -h, --help        Show this help

Examples:
  $0
  $0 dark
  $0 --crf 28 light
EOF
}

declare -a ARGS_THEMES=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --crf)
      CRF="${2:-}"; shift 2;;
    --pix-fmt)
      PIX_FMT="${2:-}"; shift 2;;
    --force)
      OVERWRITE="true"; shift;;
    -h|--help)
      usage; exit 0;;
    -*)
      echo "Unknown option: $1" >&2
      usage
      exit 1;;
    *)
      ARGS_THEMES+=("$1"); shift;;
  esac
done

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Error: ffmpeg not found in PATH." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MEDIA_ROOT="${PROJECT_ROOT}/build/manim"

if [[ "${#ARGS_THEMES[@]}" -eq 0 ]]; then
  THEMES=("dark" "light")
else
  THEMES=("${ARGS_THEMES[@]}")
fi

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

process_theme() {
  local theme="$1"
  local source_root
  if ! source_root="$(find_theme_root "${theme}")"; then
    echo "Skip ${theme}: render directory not found under ${MEDIA_ROOT}." >&2
    return
  fi

  local inputs_found=0
  while IFS= read -r input_path; do
    inputs_found=1
    local filename output_path
    filename="$(basename "${input_path}")"
    output_path="${input_path%.mp4}.webm"

    if [[ "${OVERWRITE}" != "true" && -f "${output_path}" && "${output_path}" -nt "${input_path}" ]]; then
      echo "[${theme}] ${filename}: WebM up to date."
      continue
    fi

    echo "[${theme}] ${filename} -> $(basename "${output_path}"): encoding WebM (crf=${CRF}, pix_fmt=${PIX_FMT})..."

    ffmpeg -y -i "${input_path}" \
      -c:v libvpx-vp9 -b:v 0 -crf "${CRF}" -pix_fmt "${PIX_FMT}" -auto-alt-ref 0 -an \
      "${output_path}" < /dev/null
  done < <(find "${source_root}" -type f -name '*.mp4' ! -path '*/partial_movie_files/*' -print | sort)

  if [[ "${inputs_found}" -eq 0 ]]; then
    echo "Skip ${theme}: no MP4 files found in ${source_root}." >&2
    return
  fi
}

for theme in "${THEMES[@]}"; do
  process_theme "${theme}"
done
