#!/usr/bin/env bash
#
# Move SDE animation assets from the build workspace into the site assets.
# - Videos (MP4/WebM) -> site/assets/videos/<theme>/sde_animation
# - Posters (JPEG)    -> site/assets/posters
#
# Usage:
#   ./stage_assets.sh          # move dark + light themes
#   ./stage_assets.sh light    # move a single theme
#   ./stage_assets.sh --copy   # copy instead of move

set -euo pipefail

MODE="move"   # move | copy

usage() {
  cat >&2 <<EOF
Usage: $0 [OPTIONS] [THEME...]
Themes default to: dark light

Options:
  --copy         Copy files instead of moving them
  -h, --help     Show this help

Examples:
  $0
  $0 --copy dark
EOF
}

declare -a ARGS_THEMES=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --copy)
      MODE="copy"
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SOURCE_VIDEO_ROOT="${PROJECT_ROOT}/build/manim"
DEST_VIDEO_ROOT="${PROJECT_ROOT}/site/assets/videos"
SOURCE_POSTER_ROOT="${PROJECT_ROOT}/build/manim/posters"
DEST_POSTER_ROOT="${PROJECT_ROOT}/site/assets/posters"
SOURCE_FIGURE_ROOT="${PROJECT_ROOT}/assets/figures"
DEST_FIGURE_ROOT="${PROJECT_ROOT}/site/assets/images"

if [[ "${#ARGS_THEMES[@]}" -eq 0 ]]; then
  THEMES=("dark" "light")
else
  THEMES=("${ARGS_THEMES[@]}")
fi

mkdir -p "${DEST_VIDEO_ROOT}"
mkdir -p "${DEST_POSTER_ROOT}"
mkdir -p "${DEST_FIGURE_ROOT}"

find_theme_video_root() {
  local theme="$1"
  local -a candidates=(
    "${SOURCE_VIDEO_ROOT}/${theme}/videos/sde_animation"
    "${SOURCE_VIDEO_ROOT}/${theme}/videos/scenes"
    "${SOURCE_VIDEO_ROOT}/videos/${theme}/videos/sde_animation"
    "${SOURCE_VIDEO_ROOT}/videos/${theme}/videos/scenes"
  )

  for candidate in "${candidates[@]}"; do
    if [[ -d "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done

  return 1
}

perform_transfer() {
  local src="$1"
  local dest="$2"

  mkdir -p "$(dirname "${dest}")"

  if [[ "${MODE}" == "copy" ]]; then
    cp -f "${src}" "${dest}"
  else
    mv -f "${src}" "${dest}"
  fi
}

process_videos() {
  local theme="$1"
  local theme_source

  if ! theme_source="$(find_theme_video_root "${theme}")"; then
    echo "Skip videos for ${theme}: render directory not found under ${SOURCE_VIDEO_ROOT}." >&2
    return
  fi

  local files_found=0
  while IFS= read -r src_path; do
    files_found=1
    local relative
    relative="${src_path#"${theme_source}/"}"
    local dest_path="${DEST_VIDEO_ROOT}/${theme}/sde_animation/${relative}"

    echo "[${theme}] video: ${relative}"
    perform_transfer "${src_path}" "${dest_path}"
  done < <(find "${theme_source}" -type f \( -name '*.mp4' -o -name '*.webm' \) ! -path '*/partial_movie_files/*' -print | sort)

  if [[ "${files_found}" -eq 0 ]]; then
    echo "Skip videos for ${theme}: no MP4/WebM files in ${theme_source}." >&2
    return
  fi
}

process_posters() {
  local theme="$1"
  if [[ ! -d "${SOURCE_POSTER_ROOT}" ]]; then
    echo "Skip posters for ${theme}: source directory ${SOURCE_POSTER_ROOT} not present." >&2
    return
  fi
  local pattern
  pattern="${SOURCE_POSTER_ROOT}"/*-"${theme}".jpg
  shopt -s nullglob
  local posters=(${pattern})
  shopt -u nullglob

  if [[ "${#posters[@]}" -eq 0 ]]; then
    echo "Skip posters for ${theme}: none found in ${SOURCE_POSTER_ROOT}." >&2
    return
  fi

  for src_path in "${posters[@]}"; do
    local filename
    filename="$(basename "${src_path}")"
    local dest_path="${DEST_POSTER_ROOT}/${filename}"

    echo "[${theme}] poster: ${filename}"
    perform_transfer "${src_path}" "${dest_path}"
  done
}

sync_figures() {
  if [[ ! -d "${SOURCE_FIGURE_ROOT}" ]]; then
    echo "Skip figures: source directory ${SOURCE_FIGURE_ROOT} not present." >&2
    return
  fi

  echo "Syncing figures into site/assets/images"
  mkdir -p "${DEST_FIGURE_ROOT}"
  cp -a "${SOURCE_FIGURE_ROOT}/." "${DEST_FIGURE_ROOT}/"
}

for theme in "${THEMES[@]}"; do
  process_videos "${theme}"
  process_posters "${theme}"
done

sync_figures

if [[ "${MODE}" != "copy" ]]; then
  # Tidy up empty directories under build/manim after moving.
  find "${SOURCE_VIDEO_ROOT}" -type d -empty -delete 2>/dev/null || true
  find "${SOURCE_POSTER_ROOT}" -type d -empty -delete 2>/dev/null || true
fi

echo "Assets deployed to ${DEST_VIDEO_ROOT} and ${DEST_POSTER_ROOT}"
