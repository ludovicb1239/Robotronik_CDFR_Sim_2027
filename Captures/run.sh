#!/usr/bin/env bash
#
# Run the capture-to-field overlay pipeline.
#
# Reads every capture_*.png in this folder, takes its pose from the filename
# (written by OnboardCamScript), rectifies the capture onto the ground plane
# using the fixed camera geometry, and writes an overlay of the capture placed
# on FieldBW.png for each one.
#
# Usage:
#   ./run.sh                          # all captures, headless
#   ./run.sh --steps                  # also write every intermediate image
#   ./run.sh capture_xxx.png          # one specific capture
#   ./run.sh --show                   # open a window with the result
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(dirname "$HERE")"
VENV_PY="$PROJECT/.venv/bin/python"
SCRIPT="$HERE/opencv.py"

if [[ ! -x "$VENV_PY" ]]; then
    echo "error: no Python environment at $VENV_PY" >&2
    echo "create one with:" >&2
    echo "  python3 -m venv \"$PROJECT/.venv\"" >&2
    echo "  \"$PROJECT/.venv/bin/pip\" install opencv-contrib-python numpy matplotlib" >&2
    exit 1
fi

if [[ ! -f "$SCRIPT" ]]; then
    echo "error: $SCRIPT not found" >&2
    exit 1
fi

# The script reads captures relative to its own folder, so run from there.
cd "$HERE"

# Default to headless: this script is normally called from a terminal or a task,
# where popping a GUI window would block. Pass --show to opt in.
args=("$@")
has_show=0
for a in "${args[@]:-}"; do
    [[ "$a" == "--show" ]] && has_show=1
done
if [[ $has_show -eq 0 ]]; then
    args+=(--no-show)
fi

exec "$VENV_PY" "$SCRIPT" "${args[@]}"
